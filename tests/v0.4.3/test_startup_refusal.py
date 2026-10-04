"""Startup refusals for UNO_DRAFT_MOE_TOPK=4 (CPU, no GPU).

UNO_DRAFT_MOE_TOPK=4 exists only inside captured draft graphs, and a real proposal with no captured graph is refused.
Before v0.4.3 that refusal fired mid-serve. v0.4.3 refuses at startup, in two places:
- `UnoSpeculator.init_cudagraph_manager` (engine init, before KV-cache warmup): a launch that will never capture draft
  graphs. That is --enforce-eager (the worker skips capture), cudagraph mode NONE or PIECEWISE (no FULL decode graph
  for the draft), attention without uniform-batch graphs, no capture size that holds one request's draft rows, or a
  FULL-only mode in which the target captures nothing (the runner calls the draft capture only after a target
  capture).
- `capture()` -> `_log_draft_graph_coverage`: a reachable draft batch (up to max_num_seqs x UNO_K rows) with no graph,
  e.g. UNO_K=6 and max_num_seqs=8 need 48 rows against the profile's largest capture size of 40.

Every speculator here goes through the real `UnoSpeculator.__init__` (which derives K from num_speculative_tokens and
UNO_PLOOKUP_L), the real `init_cudagraph_manager` with a real CudaGraphManager, and the real `capture()`, on CPU.
Test-side stand-ins, nothing else: CUDA tensor allocations land on CPU; the CUDA-platform check and the target-model
class lookup are answered; the graph manager's capture records every graph it planned without running a forward; the
4-expert validator (it needs the loaded Gemma 4 model on an sm_86 GPU) hands back a capture state; and the attention
support `set_attn` would read from the KV-cache groups is given. The Gemma profile's max_num_seqs, capture sizes and
cudagraph mode are read from the shipped serve.sh, so the tests follow the profile.
"""

import contextlib
import logging
import pathlib
import re
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

SERVE = pathlib.Path("/opt/uno-kit/release/serve.sh").read_text()
_GEMMA = re.search(r"--max-num-seqs (\d+) (?:(?!--max-num-seqs).)*?\"cudagraph_mode\":\"(\w+)\",\"cudagraph_capture_sizes\":\[([0-9,]+)\]", SERVE, re.S)
assert _GEMMA, "Gemma profile flags not found in serve.sh"
GEMMA_SEQS = int(_GEMMA.group(1))
GEMMA_MODE = _GEMMA.group(2)
GEMMA_SIZES = [int(x) for x in _GEMMA.group(3).split(",")]
UNO_LOG = "vllm.v1.worker.gpu.spec_decode.uno"


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


def _start(monkeypatch, uno_k, lookup_len, max_num_seqs=GEMMA_SEQS, sizes=GEMMA_SIZES, *, mode=GEMMA_MODE,
           enforce_eager=False, uniform_batch=True, draft_top_k=True):
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


def test_profile_is_the_measured_one():
    assert GEMMA_SEQS == 8 and max(GEMMA_SIZES) == 40 and GEMMA_MODE == "FULL_AND_PIECEWISE"


# --- covered launches START -------------------------------------------------------------------------------------
@pytest.mark.parametrize("uno_k,lookup_len,max_num_seqs", [
    (2, 2, 8), (3, 2, 8), (4, 0, 8), (4, 2, 8),  # K <= 4 at the profile
    (5, 0, 8), (5, 2, 8),                        # the recipe: 8 x 5 = 40 rows, the largest capture size
    (5, 2, 1), (5, 2, 4), (5, 2, 7),             # K5 below 8 sequences
])
def test_covered_launch_starts(monkeypatch, caplog, uno_k, lookup_len, max_num_seqs):
    with caplog.at_level(logging.INFO, logger=UNO_LOG):
        p = _start(monkeypatch, uno_k, lookup_len, max_num_seqs)
    assert p.draft_moe_state is not None and p.draft_moe_state.top_k == 4
    assert "cover every request count" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_full_decode_only_with_a_target_graph_starts(monkeypatch):
    # FULL-only, two sequences: the target captures its 8-row decode graph at size 8, the draft 10 rows at 10.
    p = _start(monkeypatch, 5, 2, 2, [8, 10], mode="FULL_DECODE_ONLY")
    assert p.draft_moe_state is not None


# --- uncovered draft batches REFUSE after capture ---------------------------------------------------------------
@pytest.mark.parametrize("uno_k", [6, 7, 8])
@pytest.mark.parametrize("lookup_len", [0, 2])
def test_k6_and_above_refuses_at_startup(monkeypatch, uno_k, lookup_len):
    with pytest.raises(DraftMoEConfigurationError) as exc:
        _start(monkeypatch, uno_k, lookup_len)
    msg = str(exc.value)
    assert "UNO_DRAFT_MOE_TOPK=4" in msg and f"= {GEMMA_SEQS * uno_k} rows" in msg
    assert f"num_tokens={GEMMA_SEQS * uno_k} " in msg  # names the top uncovered dispatch key
    largest = max(GEMMA_SIZES) // uno_k * uno_k  # the largest capture size that is a whole number of requests
    assert f"largest captured draft batch {largest})" in msg and "Lower UNO_K or max_num_seqs" in msg


@pytest.mark.parametrize("max_num_seqs", [9, 16])
def test_k5_above_eight_sequences_refuses_at_startup(monkeypatch, max_num_seqs):
    with pytest.raises(DraftMoEConfigurationError, match=f"= {max_num_seqs * 5} rows"):
        _start(monkeypatch, 5, 2, max_num_seqs)


# --- launches that never capture draft graphs REFUSE at init ----------------------------------------------------
NO_CAPTURE = {
    "enforce-eager": (dict(enforce_eager=True), "--enforce-eager skips CUDA graph capture"),
    "cudagraph-none": (dict(mode="NONE"), "cudagraph_mode NONE captures no FULL decode graph"),
    "piecewise-only": (dict(mode="PIECEWISE"), "cudagraph_mode PIECEWISE captures no FULL decode graph"),
    "no-uniform-batch-attention": (dict(uniform_batch=False), "no uniform-batch CUDA graph support"),
    "sizes-below-one-request": (dict(sizes=[1, 2, 4]), "holds the 5 draft rows"),
    "target-captures-nothing": (dict(max_num_seqs=2, sizes=[10], mode="FULL_DECODE_ONLY"),
                                "the target model captures no graph"),
}


@pytest.mark.parametrize("case", sorted(NO_CAPTURE))
def test_launch_without_draft_capture_refuses_at_init(monkeypatch, case):
    kw, why = NO_CAPTURE[case]
    kw = dict(kw)
    seqs = kw.pop("max_num_seqs", GEMMA_SEQS)
    captured = []
    monkeypatch.setattr(UnoSpeculator, "capture", lambda self: captured.append(1))
    with pytest.raises(DraftMoEConfigurationError) as exc:
        _start(monkeypatch, 5, 2, seqs, **kw)
    msg = str(exc.value)
    assert why in msg and "UNO_DRAFT_MOE_TOPK needs captured draft graphs" in msg and "unset UNO_DRAFT_MOE_TOPK" in msg
    assert not captured  # refused in init_cudagraph_manager, before any capture


@pytest.mark.parametrize("case", sorted(NO_CAPTURE))
def test_same_launches_without_draft_top_k_start(monkeypatch, case):
    # Default Uno (8 experts) drafts eagerly where it has no graph, as in v0.4.2: none of these refuse.
    kw = dict(NO_CAPTURE[case][0])
    seqs = kw.pop("max_num_seqs", GEMMA_SEQS)
    p = _start(monkeypatch, 5, 2, seqs, draft_top_k=False, **kw)
    assert p.draft_moe_state is None


def test_without_draft_top_k_uncovered_stays_a_warning(monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger=UNO_LOG):
        _start(monkeypatch, 6, 2, draft_top_k=False)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings and "will draft eagerly" in warnings[0].getMessage()
