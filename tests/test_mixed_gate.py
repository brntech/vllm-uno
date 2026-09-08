# SPDX-License-Identifier: Apache-2.0
import ast
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "gates" / "lossless_spec.py"
VERIFY = ROOT / "release" / "verify.sh"


class MockHandler(BaseHTTPRequestHandler):
    behavior = "success"
    background_started = threading.Event()
    sample_started = threading.Event()
    background_response_sent = threading.Event()

    def log_message(self, _format, *_args):
        pass

    def send_json(self, payload, status=200):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path != "/metrics":
            self.send_json({"error": "not found"}, 404)
            return
        self.send_json({})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        if self.path == "/tokenize":
            self.send_json({"tokens": [10, 11]})
            return
        if self.path != "/v1/completions":
            self.send_json({"error": "not found"}, 404)
            return
        if payload["temperature"] == 0:
            type(self).background_started.set()
            if self.behavior == "http_400":
                self.send_json({"error": "background rejected"}, 400)
            elif self.behavior == "disconnect":
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
            elif self.behavior == "missing_choices":
                self.send_json({"usage": {}})
            else:
                if not type(self).sample_started.wait(2):
                    self.send_json({"error": "sample never overlapped"}, 500)
                    return
                self.send_json({"choices": [{"text": "background"}]})
                type(self).background_response_sent.set()
            return
        type(self).sample_started.set()
        if self.behavior == "success":
            assert type(self).background_started.wait(2)
            assert type(self).background_response_sent.wait(2)
        count = payload.get("n", 1)
        self.send_json({"choices": [{"token_ids": [101]} for _ in range(count)]})


@contextmanager
def mock_server(behavior):
    for event in (MockHandler.background_started, MockHandler.sample_started,
                  MockHandler.background_response_sent):
        event.clear()
    MockHandler.behavior = behavior
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


def run_gate(tmp_path, behavior):
    prefixes = tmp_path / "prefixes.json"
    output = tmp_path / "out.json"
    prefixes.write_text(json.dumps([{"id": "p", "text": "prompt"}]))
    with mock_server(behavior) as url:
        result = subprocess.run(
            [sys.executable, str(GATE), "--url", url, "--prefixes", str(prefixes),
             "--out", str(output), "--n", "1", "--chunk", "1", "--max-tokens", "1",
             "--mixed-greedy"],
            text=True, capture_output=True, timeout=15,
        )
    return result, json.loads(output.read_text())


def test_mixed_gate_records_successful_overlap(tmp_path):
    result, output = run_gate(tmp_path, "success")
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = output["mixed_background"]
    assert evidence["successes"] >= 1
    assert evidence["successful_overlaps"] >= 1


@pytest.mark.parametrize("behavior", ["http_400", "disconnect", "missing_choices"])
def test_mixed_gate_fails_closed_on_background_error(tmp_path, behavior):
    result, output = run_gate(tmp_path, behavior)
    assert result.returncode != 0
    assert output["mixed_background"]["successful_overlaps"] == 0
    assert "mixed background did not complete" in result.stderr


def load_verify_sample():
    source = VERIFY.read_text()
    embedded = source.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    tree = ast.parse(embedded)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sample")
    namespace = {
        "a": SimpleNamespace(model="m", n=1, max_tokens=1),
        "json": json,
        "RuntimeError": RuntimeError,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(VERIFY), "exec"), namespace)
    return namespace


def test_verify_wrapper_rejects_missing_overlap_evidence(tmp_path):
    prefixes = tmp_path / "prefixes.json"
    prefixes.write_text(json.dumps([{"id": "p"}]))
    (tmp_path / "mixed.json").write_text(json.dumps({
        "prefixes": {"p": {"samples": [[1]]}},
        "mixed_background": {"successes": 1, "successful_overlaps": 0},
    }))
    namespace = load_verify_sample()
    namespace["run"] = lambda *_args: SimpleNamespace(returncode=0)
    with pytest.raises(RuntimeError, match="lacks a successful overlapping"):
        namespace["sample"](tmp_path, "mixed", "http://unused", prefixes, 1, mixed=True)
