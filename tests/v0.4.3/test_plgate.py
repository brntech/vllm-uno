"""CPU tests for prompt lookup's per-request length gate (v0.4.3). Run inside the release image, no GPU, as
test_plookup.py.

The gate lives in vLLM's scheduler: a request whose context exceeds UNO_PLOOKUP_MAX_CTX is scheduled with only the K
Uno draft rows of K + L. These tests check (1) the switch, (2) the real Scheduler methods choose K or K + L per request,
mixed batches included, and (3) under vLLM's own rejection kernels, a batch mixing gated (K rows) and full (K + L rows)
requests is exact for both, and a gated request never reads its lookup columns, even when they hold a deliberately
broken (stale) row. Tokens are integers only.
"""

import types

import numpy as np
import pytest
import torch

import test_plookup as tp  # sets TRITON_INTERPRET and the driver shim before vLLM imports

from vllm.v1.core.sched.async_scheduler import AsyncScheduler  # noqa: E402
from vllm.v1.core.sched.scheduler import Scheduler  # noqa: E402
from vllm.v1.worker.gpu.spec_decode import uno_plookup as pl  # noqa: E402


# ------------------------------------------------------------------ switch + width function

@pytest.mark.parametrize("raw,want", [(None, None), ("", None), ("1", 1), ("4096", 4096), (" 8192 ", 8192)])
def test_max_ctx_env_values(monkeypatch, raw, want):
    if raw is None:
        monkeypatch.delenv("UNO_PLOOKUP_MAX_CTX", raising=False)
    else:
        monkeypatch.setenv("UNO_PLOOKUP_MAX_CTX", raw)
    assert pl.plookup_max_ctx_from_env() == want


@pytest.mark.parametrize("raw", ["0", "-1", "4k", "1e4", "40.5", "x"])
def test_max_ctx_env_refuses_junk(monkeypatch, raw):
    monkeypatch.setenv("UNO_PLOOKUP_MAX_CTX", raw)
    with pytest.raises(ValueError):
        pl.plookup_max_ctx_from_env()


def test_gated_num_spec():
    g = pl.gated_num_spec
    assert g(6, 2, 4096, 4096) == 6  # at the threshold: lookup kept
    assert g(6, 2, 4096, 4097) == 4  # above: K only
    assert g(6, 2, None, 10**6) == 6  # no gate
    assert g(6, 0, 4096, 10**6) == 6  # lookup off
    assert g(0, 2, 4096, 10**6) == 0  # Uno length tail (no drafts) passes through
    assert g(2, 2, 4096, 10**6) == 2  # a width without lookup rows passes through
    assert g(10, 2, 1, 2) == 8


def test_gate_decode_query_lens(monkeypatch):
    spec = types.SimpleNamespace(method="uno")
    monkeypatch.setenv("UNO_PLOOKUP_L", "2")
    monkeypatch.setenv("UNO_PLOOKUP_MAX_CTX", "4096")
    assert pl.plookup_gate_decode_query_lens([7], spec) == [5, 7]
    assert pl.plookup_gate_decode_query_lens([7], types.SimpleNamespace(method="dflash")) == [7]
    assert pl.plookup_gate_decode_query_lens([7], None) == [7]
    monkeypatch.delenv("UNO_PLOOKUP_MAX_CTX")
    assert pl.plookup_gate_decode_query_lens([7], spec) == [7]  # no gate: unchanged capture set
    monkeypatch.setenv("UNO_PLOOKUP_MAX_CTX", "4096")
    monkeypatch.setenv("UNO_PLOOKUP_L", "0")
    assert pl.plookup_gate_decode_query_lens([7], spec) == [7]


def test_cudagraph_manager_routes_through_the_gate_widths():
    # Pin the call site: the non-dynamic branch of the real _init_candidates must pass its widths through the helper.
    import inspect

    from vllm.v1.worker.gpu import cudagraph_utils as cg

    src = inspect.getsource(cg)
    assert "decode_query_lens = plookup_gate_decode_query_lens(" in src
    # An earlier build applied it to every CudaGraphManager, including Uno's draft manager (decode_query_len = K), and crashed at
    # startup; only the target model's manager may gain the gated width.
    i = src.index("decode_query_lens = plookup_gate_decode_query_lens(")
    assert "if isinstance(self, ModelCudaGraphManager):" in src[max(0, i - 600):i]


# ------------------------------------------------------------------ the real Scheduler methods

def _sched(monkeypatch, cls, L, max_ctx, k=4, use_uno=True):
    for name, val in (("UNO_PLOOKUP_L", L), ("UNO_PLOOKUP_MAX_CTX", max_ctx)):
        if val is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, str(val))
    s = object.__new__(cls)  # only the gate's own state; the methods under test read nothing else
    s.use_uno = use_uno
    s.num_spec_tokens = k + (L or 0)
    s._init_uno_plookup_gate()
    return s


def _req(prompt, output, placeholders):
    return types.SimpleNamespace(num_tokens=prompt + output, num_output_placeholders=placeholders)


@pytest.mark.parametrize("cls", [Scheduler, AsyncScheduler])
def test_scheduler_gate_mixed_batch(monkeypatch, cls):
    s = _sched(monkeypatch, cls, L=2, max_ctx=4096)
    ph = [-1] * 6
    short, edge, over, long_ = _req(300, 400, 7), _req(4000, 89, 7), _req(4000, 90, 7), _req(27656, 384, 7)
    assert s._uno_plookup_gate(short, ph) is ph  # untouched list object
    assert s._uno_plookup_gate(edge, ph) == [-1] * 6  # context 4096
    assert s._uno_plookup_gate(over, ph) == [-1] * 4  # context 4097
    assert s._uno_plookup_gate(long_, ph) == [-1] * 4
    assert ph == [-1] * 6, "the shared placeholder list must never be mutated"
    drafts = [11, 12, 13, 14, 15, 16]  # sync path: real draft ids keep their order, lookup columns dropped
    assert s._uno_plookup_gate(long_, drafts) == [11, 12, 13, 14]
    assert s._uno_plookup_gate(long_, []) == []
    assert s._uno_plookup_gated_steps == 3


@pytest.mark.parametrize("L,max_ctx,use_uno", [(2, None, True), (None, 4096, True), (0, 4096, True), (2, 4096, False)])
def test_scheduler_gate_off(monkeypatch, L, max_ctx, use_uno):
    s = _sched(monkeypatch, AsyncScheduler, L=L, max_ctx=max_ctx, use_uno=use_uno)
    ph = [-1] * s.num_spec_tokens
    assert s._uno_plookup_gate(_req(30000, 0, 0), ph) is ph
    assert s._uno_plookup_max_ctx is None


def test_scheduler_init_calls_gate_after_its_inputs():
    # The tests above build the scheduler with object.__new__ and set use_uno themselves, which hid a real bug: the first
    # build called _init_uno_plookup_gate() in __init__ BEFORE self.use_uno was assigned, so the served gate was
    # silently off. Pin the order in the real constructor.
    import inspect

    src = inspect.getsource(Scheduler.__init__)
    call = src.index("self._init_uno_plookup_gate()")
    assert src.index("self.use_uno = (") < call and src.index("self.num_spec_tokens = ") < call
    assert src.count("self._init_uno_plookup_gate()") == 1


def test_scheduler_gate_refuses_no_uno_rows(monkeypatch):
    with pytest.raises(ValueError):
        _sched(monkeypatch, AsyncScheduler, L=2, max_ctx=4096, k=0)


# ------------------------------------------------------------------ exactness of a mixed-width batch

def _run_mixed(cycles, seed0, posctl_eps=None, k=4, L=2, vocab=20000, reqs=256, moderate_row=None):
    """Like test_plookup._run_cycles (T=1), but odd requests are gated: they verify only the K Uno rows (K + 1 logits)
    in the same rejection_sample call as the even (K + L) requests, while the lookup columns of EVERY request hold the
    real lookup proposal (the stale control when posctl_eps='stale')."""
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

    tp._interpreter_libdevice_shim()
    rng = np.random.default_rng(seed0)
    rows = k + L + 1
    p_logits, q_logits, lt = tp._targets(rng, k, L, vocab, True, None, None, moderate_row)
    p = torch.softmax(p_logits.double(), -1).numpy()
    q = torch.softmax(q_logits.double(), -1).numpy()
    q = q / q.sum(-1, keepdims=True)
    gated = np.arange(reqs) % 2 == 1
    n_rows = np.where(gated, k + 1, rows)
    counts = {False: np.zeros((rows, vocab)), True: np.zeros((rows, vocab))}
    hist = torch.zeros(reqs, 8, dtype=torch.int32)
    total = torch.zeros(reqs, dtype=torch.int32)
    look = pl.PromptLookup(k, L, reqs, tp.DEV, posctl_eps=posctl_eps)
    draft_logits = torch.zeros(reqs, k + L, vocab, dtype=torch.float32)
    look.init_draft_logits(draft_logits)
    look.bind_history(hist, total)
    max_emitted = {False: 0, True: 0}
    for _ in range(cycles):
        hist[:, :6] = lt
        hist[:, 0] = torch.tensor(rng.integers(0, vocab, size=reqs), dtype=torch.int32)
        total.fill_(6)
        drafts = torch.zeros(reqs, k + L, dtype=torch.int64)
        for j in range(k):
            drafts[:, j] = torch.tensor(rng.choice(vocab, size=reqs, p=q[j]))
        draft_logits[:, :k] = q_logits[None]
        idx = torch.arange(reqs, dtype=torch.int64)
        look.extend(reqs, idx, drafts, draft_logits)  # lookup columns written for every request, gated or not
        flat, tgt, pos, ex_idx, loc = [], [], [], [], []
        for r in range(reqs):
            n = int(n_rows[r])
            flat += [0] + drafts[r, : n - 1].tolist()
            tgt.append(p_logits[:n])
            pos += list(range(100, 100 + n))
            ex_idx += [r] * n
            loc += list(range(n))
        cu = torch.zeros(reqs + 1, dtype=torch.int32)
        cu[1:] = torch.tensor(np.cumsum(n_rows), dtype=torch.int32)
        sampled, num_sampled = rejection_sample(
            torch.cat(tgt).contiguous(), draft_logits, torch.tensor(flat, dtype=torch.int64), cu,
            torch.tensor(pos, dtype=torch.int64), idx, torch.tensor(ex_idx, dtype=torch.int64),
            torch.tensor(loc, dtype=torch.int32), torch.full((reqs,), 1.0, dtype=torch.float32),
            torch.tensor(rng.integers(1, 2**31, size=reqs), dtype=torch.int64), k + L)
        for r in range(reqs):
            g = bool(gated[r])
            n_r = int(num_sampled[r])
            max_emitted[g] = max(max_emitted[g], n_r)
            for j in range(n_r):
                counts[g][j, int(sampled[r, j])] += 1
    return counts, p, max_emitted


def test_mixed_batch_gated_and_full_requests_are_exact():
    counts, p, max_emitted = _run_mixed(cycles=6, seed0=21)
    print(f"mixed: max tokens per cycle full {max_emitted[False]}, gated {max_emitted[True]}")
    assert max_emitted[True] <= 5 and max_emitted[False] == 7
    tp._assert_exact(counts[False], p, "mixed full (K+L)", must_reach=range(7))
    tp._assert_exact(counts[True][:5], p[:5], "mixed gated (K)", must_reach=range(5))


def test_gated_request_never_reads_its_lookup_columns():
    # The lookup columns hold the stale control (q(x) = 0: a full request accepts every reached lookup token). A gated
    # request must be unaffected: exact at positions 0..K and never more than K + 1 tokens. The full requests in the
    # same batch show the stale row is live (every request that reaches lookup row 1 also reaches row 2 and the bonus).
    counts, p, max_emitted = _run_mixed(cycles=4, seed0=22, posctl_eps="stale", moderate_row=4)
    n4, n5, n6 = (counts[False][j].sum() for j in (4, 5, 6))
    print(f"stale columns: full reached pos 4 {int(n4)}, 5 {int(n5)}, 6 {int(n6)}; gated max tokens {max_emitted[True]}")
    assert n4 >= 100 and n5 == n4 and n6 == n5, "stale row not live in the full requests: test lacks power"
    assert max_emitted[True] <= 5
    tp._assert_exact(counts[True][:5], p[:5], "gated under stale columns", must_reach=range(5))
