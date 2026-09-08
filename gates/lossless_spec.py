#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Sampled-distribution checks after speculative verification.

Fixed token-id prefixes (tokenized once via /tokenize and sent as ids), temperature 1.0 / top_p 0.95 / top_k 50,
n samples per prefix, T generated tokens each, token ids recorded per sample. Run once per arm (plain, plain
again for the noise floor, two-pass Uno, Uno with UNO_DEBUG_DRAFT_JUNK, fused ...) and compare with
lossless_spec_compare.py, which tests every position separately (never pooled).

Prefix set (gates/prefixes_spec.json): a near-deterministic counting prefix (drafts accept almost always),
a math and a prose chat prefix (mixed), plus the same math prefix under a forced all-reject server
(UNO_DEBUG_DRAFT_JUNK) — the regime is reported from /metrics acceptance per prefix.

--mixed-greedy keeps a greedy 256-token stream on another prompt in flight for the whole run, so the sampled
requests share batches with a greedy request (heterogeneous sampling params in one batch).
Usage: lossless_spec.py --out runs/ls-<arm>.json [--n 256] [--max-tokens 16] [--chunk 64] [--mixed-greedy]"""
import argparse
from pathlib import Path
import json
import re
import threading
import time

import requests

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--prefixes", default=str(Path(__file__).with_name("prefixes_spec.json")))
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--model", default="uno-qwen3-8b")
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--chunk", type=int, default=64, help="samples per request (n per request)")
ap.add_argument("--max-tokens", type=int, default=16)
ap.add_argument("--only", default="", help="comma list of prefix ids")
ap.add_argument("--mixed-greedy", action="store_true")
a = ap.parse_args()


def metrics():
    response = requests.get(f"{a.url}/metrics", timeout=30)
    response.raise_for_status()
    t = response.text

    def g(name):
        return sum(float(m.group(1)) for m in re.finditer(rf"^{re.escape(name)}(?:\{{[^}}]*\}})? (\S+)$", t, re.M))

    return {"accepted": g("vllm:spec_decode_num_accepted_tokens_total"), "drafts": g("vllm:spec_decode_num_drafts_total"),
            "draft_tokens": g("vllm:spec_decode_num_draft_tokens_total")}


def tokenize(text):
    response = requests.post(f"{a.url}/tokenize", json={"model": a.model, "prompt": text}, timeout=60)
    response.raise_for_status()
    tokens = response.json()["tokens"]
    if not isinstance(tokens, list) or not tokens:
        raise RuntimeError("tokenizer returned no token ids")
    return tokens


stop = threading.Event()
background_lock = threading.Lock()
background = {
    "attempts": 0,
    "successes": 0,
    "failures": 0,
    "successful_overlaps": 0,
    "last_error": None,
    "request_in_flight": False,
    "request_overlapped_sample": False,
    "sample_requests_in_flight": 0,
}


def greedy_stream(prompt_ids):
    while not stop.is_set():
        with background_lock:
            background["attempts"] += 1
            background["request_in_flight"] = True
            background["request_overlapped_sample"] = background["sample_requests_in_flight"] > 0
        try:
            response = requests.post(
                f"{a.url}/v1/completions",
                json={"model": a.model, "prompt": prompt_ids, "max_tokens": 256, "temperature": 0},
                timeout=600,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload.get("choices"), list) or not payload["choices"]:
                raise RuntimeError("background completion returned no choices")
        except Exception as exc:  # noqa: BLE001 -- evidence is recorded and the run fails closed below
            with background_lock:
                background["failures"] += 1
                background["last_error"] = f"{type(exc).__name__}: {exc}"
                background["request_in_flight"] = False
            stop.wait(1)
        else:
            with background_lock:
                background["successes"] += 1
                if background["request_overlapped_sample"]:
                    background["successful_overlaps"] += 1
                background["request_in_flight"] = False


prefixes = json.load(open(a.prefixes))
if a.only:
    keep = set(a.only.split(","))
    prefixes = [p for p in prefixes if p["id"] in keep]
if not prefixes:
    raise SystemExit("no prefixes selected")
out = {"config": {"n": a.n, "max_tokens": a.max_tokens, "temperature": 1.0, "top_p": 0.95, "top_k": 50,
                  "mixed_greedy": a.mixed_greedy}, "prefixes": {}}
bg = None
if a.mixed_greedy:
    bg = threading.Thread(target=greedy_stream, args=(tokenize(prefixes[-1]["text"]),), daemon=True)
    bg.start()
t0 = time.time()
for p in prefixes:
    ids = p.get("ids") or tokenize(p["text"])
    m0 = metrics()
    samples = []
    t1 = time.time()
    while len(samples) < a.n:
        k = min(a.chunk, a.n - len(samples))
        with background_lock:
            background["sample_requests_in_flight"] += 1
            if background["request_in_flight"]:
                background["request_overlapped_sample"] = True
        try:
            response = requests.post(f"{a.url}/v1/completions", json={
                "model": a.model, "prompt": ids, "max_tokens": a.max_tokens, "n": k,
                "temperature": 1.0, "top_p": 0.95, "top_k": 50, "return_token_ids": True}, timeout=1800)
            response.raise_for_status()
            r = response.json()
        finally:
            with background_lock:
                background["sample_requests_in_flight"] -= 1
        if "choices" not in r:
            raise SystemExit(f"{p['id']}: {r}")
        samples += [c["token_ids"] for c in r["choices"]]
    m1 = metrics()
    d = {k: m1[k] - m0[k] for k in m1}
    acc = round(d["accepted"] / d["draft_tokens"], 4) if d["draft_tokens"] else None
    out["prefixes"][p["id"]] = {"ids": ids, "samples": samples, "seconds": round(time.time() - t1, 1),
                                "draft_acceptance_rate": acc, "accepted_per_step": (round((d["accepted"] + d["drafts"]) / d["drafts"], 3) if d["drafts"] else None)}
    print(f"{p['id']}: {len(samples)} samples of {a.max_tokens}, prefix {len(ids)} tok, acceptance {acc}, {time.time() - t1:.0f}s", flush=True)
stop.set()
if bg is not None:
    bg.join(timeout=2)
    with background_lock:
        evidence = {key: background[key] for key in (
            "attempts", "successes", "failures", "successful_overlaps", "last_error"
        )}
    evidence["thread_stopped"] = not bg.is_alive()
    out["mixed_background"] = evidence
else:
    out["mixed_background"] = None
with open(a.out, "w") as stream:
    json.dump(out, stream)
if a.mixed_greedy and (evidence["successes"] < 1 or evidence["successful_overlaps"] < 1):
    raise SystemExit(
        "mixed background did not complete a successful request overlapping sampled traffic: "
        + json.dumps(evidence, sort_keys=True)
    )
print(f"LS_DONE {a.out} in {time.time() - t0:.0f}s")
