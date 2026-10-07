"""Startup refusals for UNO_DRAFT_MOE_TOPK=4 at the v0.4.4 capture sizes (CPU, no GPU).

v0.4.3 refuses UNO_DRAFT_MOE_TOPK=4 at startup when draft graphs will not be captured, or when a reachable draft batch
(up to max_num_seqs x UNO_K rows) has no captured graph. v0.4.4 does not change that code. It changes the capture sizes
the launcher passes: with UNO_GEMMA_SPLITKV_MULTI=1 (part of UNO_RECOMMENDED=1) they run to 64 instead of 40. These
tests re-derive, through the same real constructor, init_cudagraph_manager and capture() as the v0.4.3 tests, what
starts and what refuses at both lists:
- recommended list (to 64), at the profile's max_num_seqs 8: this engine check lets UNO_K up to 8 start (8 x 8 = 64
  rows); UNO_K=9 refuses (72 rows). Above 8 sequences UNO_K=5 still refuses at 9 and 16, as in v0.4.3. (The launcher
  refuses earlier, with UNO_GEMMA_SPLITKV_MULTI=1, any K + UNO_PLOOKUP_L + 1 above 9: test_release_v044.py. Only K=5
  with L=2 was measured on a GPU.)
- default list (to 40, no multi-request split-KV): unchanged from v0.4.3 (UNO_K >= 6 at 8 sequences refuses).
- the launches that never capture draft graphs refuse at init with either list, as in v0.4.3.
The profiles (max_num_seqs, cudagraph mode, capture sizes, K, L) are read from a dry run of the shipped launcher, so the
tests follow serve.sh. Test-side stand-ins are the v0.4.3 tests' (CUDA allocations on CPU, recorded capture, the
4-expert validator's capture state, the attention support set_attn would read).
"""

import contextlib
import logging
import os
import re
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.overrides import TorchFunctionMode

import conftest  # noqa: F401  (Triton interpreter + driver shim before vLLM imports)
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.worker.gpu.spec_decode import uno as uno_mod
from vllm.v1.worker.gpu.spec_decode import uno_draft_moe
from vllm.v1.worker.gpu.spec_decode.uno import UnoSpeculator
from vllm.v1.worker.gpu.spec_decode.uno_draft_moe import DraftMoEConfigurationError

UNO_LOG = "vllm.v1.worker.gpu.spec_decode.uno"
SWITCHES = ("UNO_RECOMMENDED", "UNO_K", "UNO_PLOOKUP_L", "UNO_PLOOKUP_MAX_CTX", "UNO_DRAFT_MOE_TOPK",
            "UNO_GEMMA_SPLITKV", "UNO_GEMMA_SPLITKV_MULTI")


def _profile(**env):
    """The gemma4 profile the shipped launcher would start, from its dry run."""
    e = {k: v for k, v in os.environ.items() if k not in SWITCHES}
    e.update(UNO_PROFILE="gemma4", UNO_DRY_RUN="1", **env)
    p = subprocess.run(["bash", "/opt/uno-kit/release/serve.sh", "m", "/tmp"], env=e, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    s = re.search(r"^Uno settings: UNO_RECOMMENDED=\S* UNO_K=(\d+) UNO_PLOOKUP_L=(\d*) .* capture_sizes=\[([0-9,]+)\]$",
                  p.stderr, re.M)
    seqs = re.search(r"--max-num-seqs (\d+)", p.stdout)
    mode = re.search(r"cudagraph_mode\W+(\w+)", p.stdout)
    assert s and seqs and mode, (p.stdout, p.stderr)
    return SimpleNamespace(k=int(s.group(1)), lookup=int(s.group(2) or 0), seqs=int(seqs.group(1)), mode=mode.group(1),
                           sizes=[int(x) for x in s.group(3).split(",")])


DEFAULT = _profile()
RECOMMENDED = _profile(UNO_RECOMMENDED="1")


class _CudaTensorsOnCpu(TorchFunctionMode):
    """Test-side: tensors the constructor allocates on the CUDA device are made on CPU instead."""

    @staticmethod
    def _cpu(x):
        return torch.device("cpu") if isinstance(x, torch.device) and x.type == "cuda" else x

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {k: self._cpu(v) for k, v in (kwargs or {}).items()}
        return func(*[self._cpu(a) for a in args], **kwargs)


@pytest.fixture(autouse=True)
def _cpu_stand_ins(monkeypatch):
    from vllm.v1.worker.gpu import cudagraph_utils as cg
    from vllm.v1.worker.gpu.spec_decode import speculator

    monkeypatch.setattr(cg, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True))
    monkeypatch.setattr(cg.current_platform, "get_global_graph_pool", lambda: object())
    monkeypatch.setattr(cg, "get_offloader", lambda: Mock())
    monkeypatch.setattr(uno_mod.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(speculator, "_target_feeds_hc_residual", lambda cfg: False)

    def record_planned_graphs(self, create_forward_fn, progress_bar_desc="Capturing CUDA graphs"):
        for desc in self._capture_descs.get(CUDAGraphMode.FULL, []):
            self.graphs[desc] = Mock()
        self._graphs_captured = True

    monkeypatch.setattr(cg.CudaGraphManager, "capture", record_planned_graphs)

    @contextlib.contextmanager
    def draft_moe_scope(proposer):
        yield SimpleNamespace(top_k=4, log_capture_receipt=lambda log: None) if uno_draft_moe.uno_draft_moe_enabled() else None

    monkeypatch.setattr(uno_mod, "draft_moe_capture_scope", draft_moe_scope)
    monkeypatch.delenv("UNO_DRAFT_MOE_TOPK", raising=False)
    uno_draft_moe._reset_process_variant_for_tests()
    yield
    uno_draft_moe._reset_process_variant_for_tests()


def _config(num_spec, max_num_seqs, sizes, mode, enforce_eager):
    from vllm.config import CompilationConfig, ParallelConfig, SchedulerConfig

    if enforce_eager:  # what VllmConfig does under --enforce-eager
        mode, sizes = "NONE", []
    cc = CompilationConfig(cudagraph_mode=mode, cudagraph_capture_sizes=sorted(sizes))
    cc.max_cudagraph_capture_size = max(sizes, default=0)
    if sizes:
        cc.post_init_cudagraph_sizes()
    spec = SimpleNamespace(
        method="uno", num_speculative_tokens=num_spec, uno_lora_path="/adapter", draft_sample_method="probabilistic",
        use_local_argmax_reduction=False, use_uno=lambda: True, uses_dynamic_speculative_decoding=lambda: False,
        draft_model_config=SimpleNamespace(get_hidden_size=lambda: 8, get_vocab_size=lambda: 64, hf_config=SimpleNamespace()),
    )
    model = SimpleNamespace(max_model_len=32768, dtype=torch.bfloat16, head_dtype=torch.float32, use_fp64_gumbel=False,
                            enforce_eager=enforce_eager)
    return SimpleNamespace(
        compilation_config=cc, scheduler_config=SchedulerConfig.default_factory(max_num_seqs=max_num_seqs,
                                                                                 max_num_batched_tokens=2048),
        parallel_config=ParallelConfig(), speculative_config=spec, model_config=model, watermark_config=None,
        cache_config=SimpleNamespace(use_kda_recoverssm=False), num_speculative_tokens=num_spec,
    )


def _start(monkeypatch, uno_k, lookup_len, max_num_seqs, sizes, *, mode="FULL_AND_PIECEWISE", enforce_eager=False,
           uniform_batch=True, draft_top_k=True):
    """Construct, init the draft graph manager and capture, as engine startup does. Returns the speculator."""
    monkeypatch.setenv("UNO_PLOOKUP_L", str(lookup_len))
    if lookup_len:
        monkeypatch.setenv("UNO_PLOOKUP_MAX_CTX", "4096")
    else:
        monkeypatch.delenv("UNO_PLOOKUP_MAX_CTX", raising=False)
    if draft_top_k:
        monkeypatch.setenv("UNO_DRAFT_MOE_TOPK", "4")
    cfg = _config(uno_k + lookup_len, max_num_seqs, sizes, mode, enforce_eager)  # serve.sh: K + L speculative tokens
    with _CudaTensorsOnCpu():
        p = UnoSpeculator(cfg, torch.device("cuda"))
        assert p.k == uno_k and p.num_speculative_steps == uno_k + lookup_len
        p.attn_cg_support = SimpleNamespace(
            min_cg_support=AttentionCGSupport.UNIFORM_BATCH if uniform_batch else AttentionCGSupport.NEVER)
        resolved = CUDAGraphMode.NONE if enforce_eager else CUDAGraphMode[mode]
        p.init_cudagraph_manager(resolved)
        if not enforce_eager:  # the worker never calls capture under --enforce-eager
            p.capture()
    return p


def test_profiles_are_the_measured_ones():
    assert (DEFAULT.seqs, max(DEFAULT.sizes), DEFAULT.mode, DEFAULT.k, DEFAULT.lookup) == (8, 40, "FULL_AND_PIECEWISE", 4, 0)
    assert (RECOMMENDED.seqs, max(RECOMMENDED.sizes), RECOMMENDED.mode) == (8, 64, "FULL_AND_PIECEWISE")
    assert (RECOMMENDED.k, RECOMMENDED.lookup) == (5, 2)
    assert RECOMMENDED.sizes == DEFAULT.sizes + [48, 56, 64]


def test_the_recommended_launch_starts(monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger=UNO_LOG):
        p = _start(monkeypatch, RECOMMENDED.k, RECOMMENDED.lookup, RECOMMENDED.seqs, RECOMMENDED.sizes)
    assert p.draft_moe_state is not None and p.draft_moe_state.top_k == 4
    assert "cover every request count" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# --- recommended list (to 64): what now starts ------------------------------------------------------------------
@pytest.mark.parametrize("uno_k", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("lookup_len", [0, 2])
def test_k_up_to_8_starts_at_the_recommended_sizes(monkeypatch, caplog, uno_k, lookup_len):
    with caplog.at_level(logging.INFO, logger=UNO_LOG):
        p = _start(monkeypatch, uno_k, lookup_len, RECOMMENDED.seqs, RECOMMENDED.sizes)
    assert p.draft_moe_state is not None
    assert "cover every request count" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.parametrize("max_num_seqs", [1, 4, 7])
def test_k5_below_eight_sequences_starts_at_the_recommended_sizes(monkeypatch, max_num_seqs):
    assert _start(monkeypatch, 5, 2, max_num_seqs, RECOMMENDED.sizes).draft_moe_state is not None


# --- recommended list: what still refuses -----------------------------------------------------------------------
@pytest.mark.parametrize("lookup_len", [0, 2])
def test_k9_refuses_at_the_recommended_sizes(monkeypatch, lookup_len):
    with pytest.raises(DraftMoEConfigurationError) as exc:
        _start(monkeypatch, 9, lookup_len, RECOMMENDED.seqs, RECOMMENDED.sizes)
    msg = str(exc.value)
    assert "= 72 rows" in msg and "num_tokens=72 " in msg
    assert f"largest captured draft batch {64 // 9 * 9})" in msg


@pytest.mark.parametrize("profile", ["default", "recommended"])
@pytest.mark.parametrize("max_num_seqs", [9, 16])
def test_k5_above_eight_sequences_still_refuses(monkeypatch, max_num_seqs, profile):
    # Above the profile's 8 sequences the draft batches of 9 and 16 requests have no graph at either list (as in v0.4.3).
    prof = RECOMMENDED if profile == "recommended" else DEFAULT
    with pytest.raises(DraftMoEConfigurationError, match=f"= {max_num_seqs * 5} rows"):
        _start(monkeypatch, 5, 2, max_num_seqs, prof.sizes)


# --- default list (to 40): unchanged from v0.4.3 ----------------------------------------------------------------
def test_k5_starts_at_the_default_sizes(monkeypatch):
    assert _start(monkeypatch, 5, 2, DEFAULT.seqs, DEFAULT.sizes).draft_moe_state is not None


@pytest.mark.parametrize("uno_k", [6, 7, 8])
def test_k6_and_above_still_refuses_at_the_default_sizes(monkeypatch, uno_k):
    with pytest.raises(DraftMoEConfigurationError) as exc:
        _start(monkeypatch, uno_k, 2, DEFAULT.seqs, DEFAULT.sizes)
    msg = str(exc.value)
    assert f"= {DEFAULT.seqs * uno_k} rows" in msg and f"largest captured draft batch {40 // uno_k * uno_k})" in msg


# --- launches that never capture draft graphs refuse at init, at either list -------------------------------------
NO_CAPTURE = {
    "enforce-eager": (dict(enforce_eager=True), "--enforce-eager skips CUDA graph capture"),
    "cudagraph-none": (dict(mode="NONE"), "cudagraph_mode NONE captures no FULL decode graph"),
    "piecewise-only": (dict(mode="PIECEWISE"), "cudagraph_mode PIECEWISE captures no FULL decode graph"),
    "no-uniform-batch-attention": (dict(uniform_batch=False), "no uniform-batch CUDA graph support"),
}


@pytest.mark.parametrize("profile", ["default", "recommended"])
@pytest.mark.parametrize("case", sorted(NO_CAPTURE))
def test_launch_without_draft_capture_refuses_at_init(monkeypatch, case, profile):
    kw, why = NO_CAPTURE[case]
    prof = RECOMMENDED if profile == "recommended" else DEFAULT
    captured = []
    monkeypatch.setattr(UnoSpeculator, "capture", lambda self: captured.append(1))
    with pytest.raises(DraftMoEConfigurationError) as exc:
        _start(monkeypatch, 5, 2, prof.seqs, prof.sizes, **kw)
    assert why in str(exc.value) and not captured
