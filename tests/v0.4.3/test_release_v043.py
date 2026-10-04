"""Release-only checks for v0.4.3 (CPU, no GPU): defaults off, only documented switches, exact lookup columns."""

import pathlib

import torch

import conftest
from vllm.v1.worker.gpu.spec_decode import uno_plookup as pl

VP = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm")


def test_defaults_are_off(monkeypatch):
    monkeypatch.delenv("UNO_PLOOKUP_L", raising=False)
    monkeypatch.delenv("UNO_PLOOKUP_MAX_CTX", raising=False)
    assert pl.plookup_len_from_env() == 0
    assert pl.plookup_max_ctx_from_env() is None
    assert pl.gated_num_spec(4, 0, None, 50_000) == 4


def test_gate_boundary():
    # K=4, L=2: contexts at or below the threshold verify K + L rows, above it K rows.
    assert pl.gated_num_spec(6, 2, 4096, 4096) == 6
    assert pl.gated_num_spec(6, 2, 4096, 4097) == 4
    assert pl.gated_num_spec(2, 2, 4096, 99_999) == 2  # Uno's length tail passes through


# Every file v0.4.3 patches, and the environment switches the release documents. A switch outside this set (a test-only
# or comparison-only knob) must not ship.
PATCHED = (
    "v1/worker/gpu/spec_decode/uno.py",
    "v1/worker/gpu/spec_decode/uno_plookup.py",
    "v1/worker/gpu/model_runner.py",
    "v1/attention/backends/triton_attn.py",
    "v1/attention/ops/triton_unified_attention.py",
    "v1/core/sched/scheduler.py",
    "v1/core/sched/async_scheduler.py",
    "v1/worker/gpu/cudagraph_utils.py",
)
DOCUMENTED_SWITCHES = {"UNO_PLOOKUP_L", "UNO_PLOOKUP_MAX_CTX", "UNO_DRAFT_VOCAB", "UNO_GEMMA_SPLITKV"}


def _env_reads(tree):
    """Names passed to os.environ.get / os.getenv / os.environ[...]; None for a name that is not a string literal."""
    import ast

    names = []
    for node in ast.walk(tree):
        arg = None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.args:
            f = node.func
            if f.attr == "getenv" or (f.attr == "get" and isinstance(f.value, ast.Attribute) and f.value.attr == "environ"):
                arg = node.args[0]
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "environ":
            arg = node.slice
        if arg is not None:
            names.append(arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else None)
    return names


def test_patched_files_read_only_documented_switches():
    import ast

    seen = set()
    for rel in PATCHED:
        names = _env_reads(ast.parse((VP / rel).read_text()))
        assert None not in names, f"{rel} reads an environment variable by a computed name"
        seen |= set(names)
    assert seen <= DOCUMENTED_SWITCHES, sorted(seen - DOCUMENTED_SWITCHES)
    assert {"UNO_PLOOKUP_L", "UNO_PLOOKUP_MAX_CTX"} <= seen


def test_release_lookup_columns_are_the_exact_point_mass():
    look = conftest.RELEASE_PROMPT_LOOKUP(4, 2, 3, torch.device("cpu"))
    logits = torch.zeros(3, 6, 50)
    look.init_draft_logits(logits)
    assert (look.hot, look.cold, look.stale_from) == (0.0, float("-inf"), 2)
    cols = logits[:, 4:]
    assert torch.all(cols[..., 0] == 0.0) and torch.all(torch.isneginf(cols[..., 1:]))
    assert torch.equal(logits[:, :4], torch.zeros(3, 4, 50))


def test_attention_verify_width_is_uno_only():
    attn = (VP / "v1/attention/backends/triton_attn.py").read_text()
    assert "use_uno=self._use_uno," in attn


def test_serve_script_bounds_lookup_len():
    sh = pathlib.Path("/opt/uno-kit/release/serve.sh").read_text()
    assert "UNO_PLOOKUP_L must be an integer in 0..8" in sh
    assert "num_speculative_tokens=k+int(pl)" in sh
    assert pathlib.Path("/opt/uno-kit/release/VERSION").read_text().strip() == "0.4.3"
