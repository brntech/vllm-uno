"""Release checks for v0.4.4 (CPU, no GPU): the launcher's one-switch recipe, the multi-request split-KV admission, and
that the release reads only documented switches.

v0.4.4 changes no proposal, rejection or sampling code: it changes which attention kernel runs for decode steps with
several requests in flight (UNO_GEMMA_SPLITKV_MULTI=1), the capture sizes that come with it, and the launcher. The
admission tests here call the release functions themselves with CPU tensors; the kernel equality is
test_splitkv_multi_cpu.py (Triton interpreter) and the GPU test of the release notes.
"""

import ast
import hashlib
import os
import pathlib
import re
import subprocess

import pytest
import torch

import conftest  # noqa: F401  (Triton interpreter + driver shim before vLLM imports)
from vllm.v1.attention.backends import triton_attn as ta
from vllm.v1.attention.ops import triton_unified_attention as tua

VP = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm")
SERVE = "/opt/uno-kit/release/serve.sh"
DEFAULT_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 28, 32, 40]
MULTI_SIZES = DEFAULT_SIZES + [48, 56, 64]
RECOMMENDED = dict(UNO_K="5", UNO_PLOOKUP_L="2", UNO_PLOOKUP_MAX_CTX="4096", UNO_DRAFT_MOE_TOPK="4",
                   UNO_GEMMA_SPLITKV="1", UNO_GEMMA_SPLITKV_MULTI="1")
SWITCHES = ("UNO_RECOMMENDED", "UNO_K", "UNO_PLOOKUP_L", "UNO_PLOOKUP_MAX_CTX", "UNO_DRAFT_MOE_TOPK",
            "UNO_GEMMA_SPLITKV", "UNO_GEMMA_SPLITKV_MULTI", "UNO_PROFILE")


def launch(**env):
    """Dry-run the shipped launcher (gemma4 unless UNO_PROFILE is given). Returns (rc, settings dict, stdout, stderr)."""
    e = {k: v for k, v in os.environ.items() if k not in SWITCHES}
    e.update({"UNO_PROFILE": "gemma4", "UNO_DRY_RUN": "1"})
    e.update(env)
    p = subprocess.run(["bash", SERVE, "m", "/tmp"], env=e, capture_output=True, text=True)
    m = re.search(r"^Uno settings: (.*) capture_sizes=\[([0-9,]*)\]$", p.stderr, re.M)
    s = None
    if m:
        s = dict(kv.split("=", 1) for kv in m.group(1).split())
        s["sizes"] = [int(x) for x in m.group(2).split(",")]
        spec = re.search(r"num_speculative_tokens\\?\"?:(\d+)", p.stdout)
        s["num_spec"] = int(spec.group(1)) if spec else None
    return p.returncode, s, p.stdout, p.stderr


def test_version_and_launcher_syntax():
    assert pathlib.Path("/opt/uno-kit/release/VERSION").read_text().strip() == "0.4.4"
    assert subprocess.run(["bash", "-n", SERVE]).returncode == 0


def test_unset_is_the_v043_profile():
    rc, s, out, _ = launch()
    assert rc == 0 and s["UNO_RECOMMENDED"] == "0" and s["UNO_K"] == "4" and s["num_spec"] == 4
    assert s["UNO_GEMMA_SPLITKV_MULTI"] == "" and s["UNO_DRAFT_MOE_TOPK"] == "" and s["UNO_PLOOKUP_L"] == ""
    assert s["sizes"] == DEFAULT_SIZES


def test_recommended_sets_the_measured_recipe():
    rc, s, out, err = launch(UNO_RECOMMENDED="1")
    assert rc == 0, err
    assert {k: s[k] for k in RECOMMENDED} == RECOMMENDED
    assert s["num_spec"] == 7  # K + L speculative tokens
    assert s["sizes"] == MULTI_SIZES


@pytest.mark.parametrize("var,value", [("UNO_K", "4"), ("UNO_PLOOKUP_L", "1"), ("UNO_PLOOKUP_MAX_CTX", "8192")])
def test_an_explicit_value_wins_over_the_recipe(var, value):
    rc, s, _, _ = launch(UNO_RECOMMENDED="1", **{var: value})
    assert rc == 0 and s[var] == value
    assert all(s[k] == v for k, v in RECOMMENDED.items() if k != var)


def test_empty_turns_one_setting_off():
    rc, s, _, _ = launch(UNO_RECOMMENDED="1", UNO_DRAFT_MOE_TOPK="")
    assert rc == 0 and s["UNO_DRAFT_MOE_TOPK"] == "" and s["UNO_K"] == "5" and s["sizes"] == MULTI_SIZES
    rc, s, _, _ = launch(UNO_RECOMMENDED="1", UNO_PLOOKUP_L="")
    assert rc == 0 and s["UNO_PLOOKUP_L"] == "" and s["num_spec"] == 5
    rc, s, _, _ = launch(UNO_RECOMMENDED="1", UNO_PLOOKUP_L="", UNO_PLOOKUP_MAX_CTX="")  # lookup off: no gate needed
    assert rc == 0 and s["UNO_PLOOKUP_L"] == "" and s["UNO_PLOOKUP_MAX_CTX"] == "" and s["num_spec"] == 5
    for off in ("", "0"):
        rc, s, _, _ = launch(UNO_RECOMMENDED="1", UNO_GEMMA_SPLITKV_MULTI=off)
        assert rc == 0 and s["UNO_GEMMA_SPLITKV_MULTI"] == off and s["sizes"] == DEFAULT_SIZES


@pytest.mark.parametrize("env,num_spec", [
    (dict(UNO_RECOMMENDED="1", UNO_K="6"), 8),                     # 6 + 2 + 1 = 9 verify rows
    (dict(UNO_RECOMMENDED="1", UNO_K="8", UNO_PLOOKUP_L=""), 8),   # 8 + 0 + 1 = 9
    (dict(UNO_GEMMA_SPLITKV_MULTI="1", UNO_K="8"), 8),
])
def test_multi_request_widths_up_to_9_start(env, num_spec):
    rc, s, _, err = launch(**env)
    assert rc == 0, err
    assert s["num_spec"] == num_spec and s["sizes"] == MULTI_SIZES


@pytest.mark.parametrize("var,value,parsed", [("UNO_K", " 5", "5"), ("UNO_K", "5 ", "5"), ("UNO_K", "05", "05"),
                                              ("UNO_PLOOKUP_MAX_CTX", " 4096 ", "4096")])
def test_numeric_overrides_are_parsed_once(var, value, parsed):
    # Surrounding spaces are stripped by the one parser; the settings line, the width check and the speculative config
    # all see the parsed value.
    rc, s, _, err = launch(UNO_RECOMMENDED="1", **{var: value})
    assert rc == 0, err
    assert s[var] == parsed and s["num_spec"] == 7 and s["sizes"] == MULTI_SIZES


@pytest.mark.parametrize("off", ["0", "", "off", "false", "no"])
def test_capture_sizes_follow_the_multi_switch(off):
    # The larger capture sizes come only with multi-request split-KV (alone they cost KV cache).
    rc, s, _, _ = launch(UNO_RECOMMENDED="1", UNO_GEMMA_SPLITKV_MULTI=off)
    assert rc == 0 and s["sizes"] == DEFAULT_SIZES
    rc, s, _, _ = launch(UNO_GEMMA_SPLITKV_MULTI="on")
    assert rc == 0 and s["sizes"] == MULTI_SIZES and s["UNO_RECOMMENDED"] == "0"


@pytest.mark.parametrize("env,msg", [
    (dict(UNO_RECOMMENDED="1", UNO_GEMMA_SPLITKV="0"), "UNO_GEMMA_SPLITKV_MULTI=1 needs UNO_GEMMA_SPLITKV=1"),
    (dict(UNO_GEMMA_SPLITKV_MULTI="1", UNO_GEMMA_SPLITKV=""), "UNO_GEMMA_SPLITKV_MULTI=1 needs UNO_GEMMA_SPLITKV=1"),
    (dict(UNO_GEMMA_SPLITKV_MULTI="2"), "UNO_GEMMA_SPLITKV_MULTI must be 1 or 0"),
    (dict(UNO_RECOMMENDED="yes"), "UNO_RECOMMENDED must be 0 or 1"),
    (dict(UNO_RECOMMENDED="1", UNO_PROFILE="qwen3"), "UNO_RECOMMENDED=1 applies to UNO_PROFILE=gemma4 only"),
    # An empty or wide override must mean what the documentation says, or be refused (fix round 1):
    (dict(UNO_RECOMMENDED="1", UNO_K=""), "UNO_K= (empty) has no meaning with UNO_RECOMMENDED=1"),
    (dict(UNO_RECOMMENDED="1", UNO_PLOOKUP_MAX_CTX=""), "UNO_PLOOKUP_MAX_CTX= (empty) with prompt lookup on"),
    (dict(UNO_RECOMMENDED="1", UNO_PLOOKUP_MAX_CTX=" ", UNO_PLOOKUP_L="1"),
     "UNO_PLOOKUP_MAX_CTX must be a whole number (got ' ')"),
    # One parser for every numeric override (fix round 2): no spelling can skip the width check.
    (dict(UNO_RECOMMENDED="1", UNO_K=" 8"), "needs UNO_K + UNO_PLOOKUP_L + 1 <= 9 (got 8 + 2 + 1)"),
    (dict(UNO_RECOMMENDED="1", UNO_K="8 "), "needs UNO_K + UNO_PLOOKUP_L + 1 <= 9 (got 8 + 2 + 1)"),
    (dict(UNO_RECOMMENDED="1", UNO_K="+8"), "UNO_K must be a whole number (got '+8')"),
    (dict(UNO_GEMMA_SPLITKV_MULTI="1", UNO_K=" 8 ", UNO_PLOOKUP_L=" 2"), "(got 8 + 2 + 1)"),
    (dict(UNO_K="+8"), "UNO_K must be a whole number"),
    (dict(UNO_PLOOKUP_L="x"), "UNO_PLOOKUP_L must be a whole number"),
    (dict(UNO_MAX_MODEL_LEN="8192 --enforce-eager"), "UNO_MAX_MODEL_LEN must be a whole number"),
    (dict(UNO_NOISE_SEED="-1"), "UNO_NOISE_SEED must be a whole number"),
    (dict(UNO_PROFILE="qwen3", UNO_K="+8"), "UNO_K must be a whole number"),
    (dict(UNO_RECOMMENDED="1", UNO_K="7"), "needs UNO_K + UNO_PLOOKUP_L + 1 <= 9 (got 7 + 2 + 1)"),
    (dict(UNO_RECOMMENDED="1", UNO_K="9", UNO_PLOOKUP_L=""), "needs UNO_K + UNO_PLOOKUP_L + 1 <= 9 (got 9 + 0 + 1)"),
    (dict(UNO_GEMMA_SPLITKV_MULTI="on", UNO_K="6", UNO_PLOOKUP_L="3"), "needs UNO_K + UNO_PLOOKUP_L + 1 <= 9"),
    (dict(UNO_GEMMA_SPLITKV_MULTI="1", UNO_PROFILE="qwen3"), "UNO_GEMMA_SPLITKV_MULTI=1 applies to UNO_PROFILE=gemma4 only"),
    (dict(UNO_GEMMA_SPLITKV_MULTI="2", UNO_PROFILE="qwen3"), "UNO_GEMMA_SPLITKV_MULTI must be 1 or 0"),
])
def test_launcher_refuses(env, msg):
    rc, s, _, err = launch(**env)
    assert rc == 2 and s is None and msg in err


# --- UNO_RECOMMENDED=1 alone starts exactly what the GPU session measured ----------------------------------------
# Recorded from the measured launcher (in-image sha256 b3e31af3...) in the published v0.4.3 image, with the same
# stand-in python as here: the engine's argv (printf %q) and every UNO_* / VLLM_* variable of the engine's environment.
# The launcher's fixes after the measurement must leave this launch byte-identical, so the measurements stay valid.
MEASURED_ENGINE_ARGV = r'''-m vllm.entrypoints.cli.main serve cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit --served-model-name uno-gemma4-26b-a4b --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 --api-server-count 1 --async-scheduling --enable-log-requests --jit-monitor-verbose --generation-config vllm --dtype bfloat16 --attention-backend TRITON_ATTN --language-model-only --enable-prefix-caching --seed 29 --max-model-len 32768 --max-num-seqs 8 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.90 --enable-lora --lora-dtype bfloat16 --max-lora-rank 16 --max-loras 2 --max-cpu-loras 2 --lora-target-modules qkv_proj o_proj gate_up_proj down_proj --compilation-config \{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\"\,\"cudagraph_capture_sizes\":\[1\,2\,3\,4\,5\,6\,7\,8\,9\,10\,12\,14\,16\,20\,24\,28\,32\,40\,48\,56\,64\]\} --speculative-config \{\"method\":\"uno\"\,\"uno_lora_path\":\"/tmp/ad\"\,\"uno_mask_token_id\":262144\,\"uno_noise_seed\":29\,\"num_speculative_tokens\":7\,\"uno_noise_low\":0\} --revision 0ef577a5710035bd2d3a3f27e4f5cb2e86a9a9ba --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4'''
MEASURED_ENGINE_ENV = {
    "UNO_DRAFT_MOE_TOPK": "4", "UNO_DRAFT_VOCAB": "/opt/uno-kit/release/gemma4-draft-vocab-65536.json",
    "UNO_GEMMA_SPLITKV": "1", "UNO_GEMMA_SPLITKV_MULTI": "1", "UNO_K": "5", "UNO_PLOOKUP_L": "2",
    "UNO_PLOOKUP_MAX_CTX": "4096", "UNO_PROFILE": "gemma4", "UNO_RECOMMENDED": "1", "VLLM_LORA_ENABLE_DUAL_STREAM": "1",
    "VLLM_TUNED_CONFIG_FOLDER": "/opt/uno-kit/release/lora-configs", "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
}
MEASURED_DRY_RUN_SHA256 = "3eaf58c655fc6b41c726369360ea1e832db7a148ec0e9eafd78bf63d0c8ea75d"  # UNO_DRY_RUN=1, args m /tmp
FAKE_PY = '''#!/bin/bash
if [[ ${1:-} == -m && ${2:-} == vllm.entrypoints.cli.main ]]; then
  echo "ARGV:$(printf ' %q' "$@")"; env -u PWD -u SHLVL -u _ -u OLDPWD | LC_ALL=C sort; exit 0
fi
exec python3 "$@"
'''
CLEAN_ENV = {"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "HOME": "/root", "LANG": "C.UTF-8"}


def test_recommended_launch_is_byte_identical_to_the_measured_one(tmp_path):
    fake = tmp_path / "fakepy"
    fake.write_text(FAKE_PY)
    fake.chmod(0o755)
    ad = pathlib.Path("/tmp/ad")
    ad.mkdir(exist_ok=True)
    (ad / "adapter_config.json").write_text("{}")
    args = ["cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit", str(ad), "--", "--enable-auto-tool-choice", "--tool-call-parser",
            "gemma4", "--reasoning-parser", "gemma4"]
    p = subprocess.run(["bash", SERVE, *args], env={**CLEAN_ENV, "UNO_PROFILE": "gemma4", "UNO_RECOMMENDED": "1",
                                                   "PYTHON": str(fake)}, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    lines = p.stdout.splitlines()
    assert lines[0] == "ARGV: " + MEASURED_ENGINE_ARGV
    env = dict(x.split("=", 1) for x in lines[1:])
    assert {k: v for k, v in env.items() if k.startswith(("UNO_", "VLLM_"))} == MEASURED_ENGINE_ENV
    dry = subprocess.run(["bash", SERVE, "m", "/tmp"], env={**CLEAN_ENV, "UNO_PROFILE": "gemma4", "UNO_RECOMMENDED": "1",
                                                            "UNO_DRY_RUN": "1"}, capture_output=True)
    assert dry.returncode == 0 and hashlib.sha256(dry.stdout).hexdigest() == MEASURED_DRY_RUN_SHA256


# Every file v0.4.4 patches over v0.4.1 (v0.4.3's set; v0.4.4 changes two of them), and the documented switches.
PATCHED = (
    "v1/worker/gpu/spec_decode/uno.py", "v1/worker/gpu/spec_decode/uno_plookup.py", "v1/worker/gpu/model_runner.py",
    "v1/attention/backends/triton_attn.py", "v1/attention/ops/triton_unified_attention.py",
    "v1/core/sched/scheduler.py", "v1/core/sched/async_scheduler.py", "v1/worker/gpu/cudagraph_utils.py",
)
DOCUMENTED_SWITCHES = {"UNO_PLOOKUP_L", "UNO_PLOOKUP_MAX_CTX", "UNO_DRAFT_VOCAB", "UNO_GEMMA_SPLITKV",
                       "UNO_GEMMA_SPLITKV_MULTI"}


def _env_reads(tree):
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
    seen = set()
    for rel in PATCHED:
        names = _env_reads(ast.parse((VP / rel).read_text()))
        assert None not in names, f"{rel} reads an environment variable by a computed name"
        seen |= set(names)
    assert seen <= DOCUMENTED_SWITCHES, sorted(seen - DOCUMENTED_SWITCHES)
    assert "UNO_GEMMA_SPLITKV_MULTI" in seen


# --- admission of a decode step to split-KV (triton_attn._uno_static_query_width) ----------------------------------
def _width(num_reqs, width, qsl=None, use_uno=True, **kw):
    qsl = torch.arange(0, (num_reqs + 1) * width, width, dtype=torch.int32) if qsl is None else qsl
    args = dict(use_uno=use_uno, num_reqs=num_reqs, num_actual_tokens=num_reqs * width, max_query_len=width,
                query_start_loc_cpu=qsl, causal=True, uno_custom_mask=None, mm_req_doc_ranges=None,
                rswa_prefix_lens=None)
    args.update(kw)
    return ta._uno_static_query_width(**args)


@pytest.fixture
def multi(monkeypatch):
    def set_(on):
        monkeypatch.setenv("UNO_GEMMA_SPLITKV_MULTI", "1" if on else "")
    return set_


@pytest.mark.parametrize("width", [2, 5, 6, 8, 9])
def test_one_request_is_admitted_with_or_without_the_switch(multi, width):
    for on in (False, True):
        multi(on)
        assert _width(1, width) == width


@pytest.mark.parametrize("num_reqs", [2, 4, 7, 8])
@pytest.mark.parametrize("width", [5, 6, 8])
def test_uniform_multi_request_step_needs_the_switch(multi, num_reqs, width):
    multi(False)
    assert _width(num_reqs, width) is None  # v0.4.3 behaviour: 2D kernel
    multi(True)
    assert _width(num_reqs, width) == width


def test_multi_request_admission_refuses_what_is_not_uniform(multi):
    multi(True)
    ragged = torch.tensor([0, 8, 14, 22], dtype=torch.int32)  # widths 8, 6, 8
    assert _width(3, 8, qsl=ragged) is None
    assert _width(3, 8, num_actual_tokens=22) is None
    assert _width(4, 8) == 8  # a uniform CPU query_start_loc is admitted
    assert _width(4, 8, qsl=torch.arange(0, 40, 8, dtype=torch.int32, device="meta")) is None  # never a device copy
    assert _width(2, 1) is None and _width(2, 10) is None  # widths outside 2..9
    assert _width(2, 8, use_uno=False) is None
    assert _width(2, 8, causal=False) is None
    assert _width(0, 8, qsl=torch.zeros(1, dtype=torch.int32), num_actual_tokens=0) is None


def _admissible(num_seqs, width, rows, multi_on, head=256, kvh=8, window=(1023, 0), q_rows=None, monkeypatch=None):
    q = torch.zeros(num_seqs * width if q_rows is None else q_rows, 16, head, dtype=torch.bfloat16)
    k = torch.zeros(4, 16, kvh, head, dtype=torch.bfloat16)
    so = torch.zeros(rows, 16, 16, head, dtype=torch.float32)
    sm = torch.zeros(rows, 16, 16, dtype=torch.float32)
    return tua._uno_gemma_splitkv_admissible(
        q=q, k=k, v=k.clone(), out=torch.zeros_like(q), max_seqlen_q=width, num_seqs=num_seqs, causal=True,
        window_size=window, uno_static_query_width=width, seq_threshold_3D=16, num_par_softmax_segments=16,
        softmax_segm_output=so, softmax_segm_max=sm, softmax_segm_expsum=sm.clone(), mm_prefix_range=None,
        rswa_prefix_lens=None, rswa_window=None, alibi_slopes=None, qq_bias=None, sinks=None, output_scale=None,
        q_descale=None, k_descale=None, v_descale=None, kv_quant_mode=tua.KVQuantMode.NONE, chunk_lookback=-1)


@pytest.mark.parametrize("head,kvh,window", [(256, 8, (1023, 0)), (512, 2, (-1, -1))])
def test_kernel_admission_follows_the_switch_and_the_scratch(multi, head, kvh, window):
    multi(False)
    assert _admissible(1, 8, 16, False, head, kvh, window)
    assert not _admissible(4, 8, 72, False, head, kvh, window)
    multi(True)
    assert _admissible(4, 8, 72, True, head, kvh, window)
    assert _admissible(8, 8, 72, True, head, kvh, window)
    assert not _admissible(8, 8, 63, True, head, kvh, window)  # scratch must hold every query row
    assert not _admissible(4, 8, 72, True, head, kvh, window, q_rows=30)  # rows must be num_seqs x width

