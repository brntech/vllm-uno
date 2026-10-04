"""CPU tests for Uno's prompt lookup (v0.4.3). Run inside the release image, no GPU:

  docker run --rm --network none --cpus 4 -e TRITON_INTERPRET=1 -v <tests>:/t:ro --entrypoint python3 \
      vllm-uno:0.4.3 -m pytest -q /t/test_plookup.py

TRITON_INTERPRET=1 runs the real Triton kernels (the lookup matcher and vLLM's own rejection kernels) on CPU tensors.
Tokens are integers only.
"""

import os

os.environ.setdefault("TRITON_INTERPRET", "1")

# vllm.triton_utils swaps in placeholder decorators when no GPU driver is active; the interpreter needs none, so report
# the CUDA driver active (CPU-only test containers only) to keep the real triton.jit.
import triton.backends as _tb  # noqa: E402

if "nvidia" in _tb.backends:
    _tb.backends["nvidia"].driver.is_active = staticmethod(lambda: True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from vllm.v1.worker.gpu.spec_decode import uno_plookup as pl  # noqa: E402

DEV = torch.device("cpu")


# ------------------------------------------------------------------ env switch

@pytest.mark.parametrize("raw,want", [(None, 0), ("", 0), ("0", 0), ("2", 2), ("8", 8)])
def test_env_switch_values(monkeypatch, raw, want):
    if raw is None:
        monkeypatch.delenv("UNO_PLOOKUP_L", raising=False)
    else:
        monkeypatch.setenv("UNO_PLOOKUP_L", raw)
    assert pl.plookup_len_from_env() == want


@pytest.mark.parametrize("raw", ["9", "-1", "x", "2.0", "1e1"])
def test_env_switch_refuses_junk(monkeypatch, raw):
    monkeypatch.setenv("UNO_PLOOKUP_L", raw)
    with pytest.raises(ValueError):
        pl.plookup_len_from_env()


def test_lookup_column_logits():
    assert pl.lookup_column_logits(100, None) == (0.0, float("-inf"))
    assert pl.lookup_column_logits(100, 1.0) == (0.0, 0.0)
    assert pl.lookup_column_logits(100, "stale") == (float("-inf"), 0.0)
    hot, cold = pl.lookup_column_logits(100, 0.5)
    q = np.exp([hot] + [cold] * 99)
    assert abs(q.sum() - 1.0) < 1e-9 and abs(q[0] - (0.5 + 0.5 / 100)) < 1e-9


# ------------------------------------------------------------------ reference matcher (hand cases)

def test_reference_hand_cases():
    # longest suffix wins over a more recent shorter match
    assert pl.reference_lookup([1, 2, 3, 9, 7, 3, 5], [1, 2, 3], 2) == [9, 7]
    # most recent occurrence among equal lengths
    assert pl.reference_lookup([4, 5, 6, 4, 5, 8], [4, 5], 1) == [8]
    # the proposal may run into the drafts and repeats the matched period at the end
    assert pl.reference_lookup([7], [7, 7, 7], 4) == [7, 7, 7, 7]
    assert pl.reference_lookup([1, 2], [1, 2], 5) == [1, 2, 1, 2, 1]
    # no match at all: repeat the last context token
    assert pl.reference_lookup([1, 2, 3], [4, 5], 3) == [5, 5, 5]
    # n caps at NMAX: a 6-token repeat is matched on its last 4 tokens
    h = [10, 11, 12, 13, 14, 15, 99, 0, 12, 13, 14, 15, 42]
    assert pl.reference_lookup(h, [12, 13, 14, 15], 1) == [42]


# ------------------------------------------------------------------ kernels vs reference

def _make(num_slots, width, k, lookup_len, vocab):
    hist = torch.zeros(num_slots, width, dtype=torch.int32)
    total = torch.zeros(num_slots, dtype=torch.int32)
    look = pl.PromptLookup(k, lookup_len, num_slots, DEV)
    logits = torch.zeros(num_slots, k + lookup_len, vocab, dtype=torch.bfloat16)
    look.init_draft_logits(logits)
    look.bind_history(hist, total)
    return hist, total, look, logits


def _check_point_masses(look, logits, k):
    lk = logits[:, k:].float()
    finite = torch.isfinite(lk)
    assert int(finite.sum()) == lk.shape[0] * lk.shape[1], "exactly one finite entry per lookup column"
    assert torch.all(lk[finite] == 0)
    hot = lk.argmax(dim=-1)
    assert torch.equal(hot, look.prev), "one-hot sits at the tracked token"


@pytest.mark.parametrize("lookup_len", [1, 2, 4, 8])
def test_kernels_match_reference_with_slot_reuse(lookup_len):
    rng = np.random.default_rng(1000 + lookup_len)
    k, vocab, slots, width = 4, 64, 6, 300
    hist, total, look, logits = _make(slots, width, k, lookup_len, vocab)
    uno_cols_before = logits[:, :k].clone()
    for step in range(30):
        n = int(rng.integers(1, slots + 1))
        idx = torch.tensor(rng.permutation(slots)[:n], dtype=torch.int64)  # non-identity slot mapping
        # fresh histories: small alphabets and planted repeats so every branch (long/short/no match) occurs
        for s in idx.tolist():
            t = int(rng.integers(0, 120))
            alpha = int(rng.choice([3, 8, 60]))
            seq = rng.integers(0, alpha, size=t)
            if t > 20 and rng.random() < 0.5:
                a = int(rng.integers(0, t - 10))
                seq[-6:] = seq[a:a + 6]
            hist[s, :t] = torch.tensor(seq, dtype=torch.int32)
            hist[s, t:] = -7  # junk past total_len must never be read
            total[s] = t
        drafts = torch.full((slots, k + lookup_len), -5, dtype=torch.int64)
        drafts[:n, :k] = torch.tensor(rng.integers(0, 8, size=(n, k)))
        look.extend(n, idx, drafts, logits)
        for r, s in enumerate(idx.tolist()):
            t = int(total[s])
            want = pl.reference_lookup(hist[s, :t].tolist(), drafts[r, :k].tolist(), lookup_len)
            assert drafts[r, k:].tolist() == want, (step, r, s)
            assert look.prev[s].tolist() == want
        assert torch.all(drafts[n:] == -5), "rows past num_reqs untouched"
        _check_point_masses(look, logits, k)
    assert torch.equal(logits[:, :k], uno_cols_before), "Uno's own draft columns are never written"


@pytest.mark.parametrize("eps", [0.5, 1.0, "stale", "stale2"])
def test_posctl_columns_hold_hot_and_cold(eps):
    rng = np.random.default_rng(7)
    k, L, vocab, slots = 4, 2, 64, 4
    hist = torch.zeros(slots, 40, dtype=torch.int32)
    total = torch.zeros(slots, dtype=torch.int32)
    look = pl.PromptLookup(k, L, slots, DEV, posctl_eps=eps)
    logits = torch.zeros(slots, k + L, vocab, dtype=torch.float32)
    look.init_draft_logits(logits)
    look.bind_history(hist, total)
    hot, cold = pl.lookup_column_logits(vocab, eps)
    for _ in range(5):
        hist[:, :30] = torch.tensor(rng.integers(0, 5, size=(slots, 30)), dtype=torch.int32)
        total.fill_(30)
        drafts = torch.tensor(rng.integers(0, 5, size=(slots, k + L)), dtype=torch.int64)
        look.extend(slots, torch.arange(slots), drafts, logits)
        lk = logits[:, k:]
        for s in range(slots):
            for i in range(L):
                row = lk[s, i]
                t = int(look.prev[s, i])
                assert t == int(drafts[s, k + i])
                if eps == "stale2":  # column 1 exact, column 2 stale
                    hot, cold = (0.0, float("-inf")) if i == 0 else (float("-inf"), 0.0)
                assert float(row[t]) == hot or abs(float(row[t]) - hot) < 1e-6
                others = torch.cat([row[:t], row[t + 1:]])
                assert torch.all((others == cold) | ((others - cold).abs() < 1e-6))


def test_negative_slot_is_skipped():
    k, L = 4, 2
    hist, total, look, logits = _make(3, 32, k, L, 16)
    hist[1, :5] = torch.tensor([1, 2, 3, 1, 2], dtype=torch.int32)
    total[1] = 5
    drafts = torch.zeros(2, k + L, dtype=torch.int64)
    drafts[:, :k] = torch.tensor([[3, 1, 2, 3], [9, 9, 9, 9]])
    look.extend(2, torch.tensor([1, -1]), drafts, logits)
    assert drafts[0, k:].tolist() == pl.reference_lookup([1, 2, 3, 1, 2], [3, 1, 2, 3], L)
    assert drafts[1, k:].tolist() == [0, 0]
    _check_point_masses(look, logits, k)


def test_refuses_unbound_history_and_wrong_width():
    look = pl.PromptLookup(4, 2, 2, DEV)
    with pytest.raises(RuntimeError):
        look.extend(1, torch.tensor([0]), torch.zeros(2, 6, dtype=torch.int64), torch.zeros(2, 6, 8))
    look.bind_history(torch.zeros(2, 8, dtype=torch.int32), torch.zeros(2, dtype=torch.int32))
    with pytest.raises(RuntimeError):
        look.extend(1, torch.tensor([0]), torch.zeros(2, 5, dtype=torch.int64), torch.zeros(2, 5, 8))
    with pytest.raises(ValueError):
        pl.PromptLookup(4, 0, 2, DEV)


# ------------------------------------------------------------------ exactness through vLLM's own rejection kernels

def _chi2_pvalue(counts, probs):
    """Pearson chi-square p-value (bins with expected < 5 pooled)."""
    n = counts.sum()
    exp_c = probs * n
    order = np.argsort(exp_c)
    obs_b, exp_b, acc_o, acc_e = [], [], 0.0, 0.0
    for i in order:
        acc_o += counts[i]
        acc_e += exp_c[i]
        if acc_e >= 5:
            obs_b.append(acc_o)
            exp_b.append(acc_e)
            acc_o = acc_e = 0.0
    if acc_e > 0 and exp_b:
        obs_b[-1] += acc_o
        exp_b[-1] += acc_e
    obs_b, exp_b = np.array(obs_b), np.array(exp_b)
    stat = float(((obs_b - exp_b) ** 2 / exp_b).sum())
    dof = len(exp_b) - 1
    if dof < 1:
        return 1.0
    return float(torch.special.gammaincc(torch.tensor(dof / 2.0, dtype=torch.float64),
                                         torch.tensor(stat / 2.0, dtype=torch.float64)))


import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def _log1p_shim(x):
    return tl.log(1.0 + x)


class _LibdeviceShim:
    log1p = _log1p_shim


def _interpreter_libdevice_shim():
    """Triton's interpreter has no libdevice log1p; give the two sampler modules an equivalent (test-only)."""
    from vllm.v1.worker.gpu.sample import gumbel
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils as rsu

    rsu.tldevice = _LibdeviceShim
    gumbel.tldevice = _LibdeviceShim


WIDE_SUPPORT = (3, 4100, 8191, 8192, 9000, 12000, 16383, 16384, 17777)
WIDE_LOOKUP_TOKEN = 16385


def _targets(rng, k, L, vocab, wide, top_p_out_row, greedy_reject_row, moderate_row=None):
    """Synthetic context-free target logits p (rows 0..K+L) and Uno draft logits q (rows 0..K-1); lookup token id."""
    rows = k + L + 1
    if not wide:
        # Original design (vocab 16, one block): deep positions reached often; token 5 is the lookup token.
        p_logits = torch.tensor(rng.normal(0, 1.5, size=(rows, vocab)), dtype=torch.float32)
        p_logits[k - 1:k + L, 5] += 3.0
        q_logits = p_logits[:k] + torch.tensor(rng.normal(0, 0.3, size=(k, vocab)), dtype=torch.float32)
        return p_logits, q_logits, 5
    # Vocabulary above 8192 (several 8192-wide blocks of vLLM's kernels): the mass sits on tokens spread over every
    # block, and the lookup token lives in the last block, so maxima, sums and the one-hot cross block boundaries.
    lt = WIDE_LOOKUP_TOKEN
    sup = list(WIDE_SUPPORT) + [vocab - 1]
    p = rng.normal(-30.0, 0.5, size=(rows, vocab))
    p[:, sup] = rng.normal(0, 1.5, size=(rows, len(sup)))
    p[:, lt] = -30.0
    # The lookup token dominates every row (~0.85), so Uno's drafts are often all lookup tokens, the matcher then
    # proposes it too, and the lookup rows are reached and accepted often enough to test.
    p[:k + L, lt] = p[:k + L][:, sup].max(axis=1) + 3.0
    if moderate_row is not None:
        p[moderate_row, lt] = p[moderate_row][sup].max()  # lookup token only moderately likely (~0.3) in this row
    if top_p_out_row is not None:
        p[top_p_out_row, lt] = p[top_p_out_row][sup].min() - 6.0  # outside the nucleus: always rejected, resampled
    if greedy_reject_row is not None:
        p[greedy_reject_row, sup[0]] = p[greedy_reject_row, lt] + 1.0  # target argmax is not the lookup token
    q = p[:k].copy()
    cols = sup + [lt]
    q[:, cols] += rng.normal(0, 0.3, size=(k, len(cols)))
    return torch.tensor(p, dtype=torch.float32), torch.tensor(q, dtype=torch.float32), lt


def _run_cycles(onehot_lookup_columns: bool = True, cycles: int = 12, reqs: int = 256, seed0: int = 0, k: int = 2,
                L: int = 2, vocab: int = 16, temp: float = 1.0, top_p: float | None = None,
                posctl_eps: float | None = None, top_p_out_row: int | None = None,
                greedy_reject_row: int | None = None, moderate_row: int | None = None):
    """K Uno columns with probabilistic drafts sampled from q, then L lookup columns from the lookup matcher.

    The synthetic target is context-free (row j has distribution p_j whatever the prefix), so an exact verifier emits,
    at every position it reaches, a token distributed as p_j (temperature and top-p applied to the target as the
    served sampler does before ``rejection_sample``; greedy = temperature 0). Returns per-position token counts, the
    expected per-position distribution, and for greedy the list of (drafts, emitted) per request."""
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

    _interpreter_libdevice_shim()
    rng = np.random.default_rng(seed0)
    wide = vocab > 8192
    rows = k + L + 1
    # Deep positions must be reached often enough to test; any configuration must be exact, this one only gives the
    # test power at positions K..K+L.
    p_logits, q_logits, lt = _targets(rng, k, L, vocab, wide, top_p_out_row, greedy_reject_row, moderate_row)
    greedy = temp == 0.0
    if greedy:
        target_rows = p_logits.clone()
        p = np.eye(vocab)[p_logits.argmax(-1).numpy()]
        q = torch.softmax(q_logits.double(), -1).numpy()  # drafts still sampled from q
    else:
        target_rows = p_logits / temp  # the served sampler scales the target before top-p and rejection
        if top_p is not None:
            target_rows = apply_top_k_top_p_pytorch(target_rows.clone(), None, torch.full((rows,), top_p))
        p = torch.softmax(target_rows.double(), -1).numpy()
        q = torch.softmax(q_logits.double() / temp, -1).numpy()  # draft logits are stored pre-temperature
    q = q / q.sum(-1, keepdims=True)
    counts = np.zeros((rows, vocab))
    greedy_records = []
    hist = torch.zeros(reqs, 8, dtype=torch.int32)
    total = torch.zeros(reqs, dtype=torch.int32)
    look = pl.PromptLookup(k, L, reqs, DEV, posctl_eps=posctl_eps)
    draft_logits = torch.zeros(reqs, k + L, vocab, dtype=torch.float32)
    if onehot_lookup_columns:
        look.init_draft_logits(draft_logits)
    look.bind_history(hist, total)
    for cyc in range(cycles):
        # history of lookup tokens with a random token in front; the proposal depends on the drafts
        hist[:, :6] = lt
        hist[:, 0] = torch.tensor(rng.integers(0, vocab, size=reqs), dtype=torch.int32)
        total.fill_(6)
        drafts = torch.zeros(reqs, k + L, dtype=torch.int64)
        for j in range(k):
            drafts[:, j] = torch.tensor(rng.choice(vocab, size=reqs, p=q[j]))
        draft_logits[:, :k] = q_logits[None]
        idx = torch.arange(reqs, dtype=torch.int64)
        if onehot_lookup_columns:
            look.extend(reqs, idx, drafts, draft_logits)
        else:
            # negative control: same lookup tokens, but the cache keeps a non-point-mass (uniform) row
            tmp_logits = torch.zeros_like(draft_logits)
            look.extend(reqs, idx, drafts, tmp_logits)
            draft_logits[:, k:] = 0.0
        # flatten one cycle: row 0 = last verified token (unused by the kernel), rows 1.. = drafts
        draft_sampled = torch.zeros(reqs, rows, dtype=torch.int64)
        draft_sampled[:, 1:] = drafts
        target = target_rows.repeat(reqs, 1).contiguous()
        cu = torch.arange(0, reqs * rows + 1, rows, dtype=torch.int32)
        pos = (torch.arange(rows, dtype=torch.int64) + 100).repeat(reqs)
        expanded_idx = idx.repeat_interleave(rows)
        local_pos = torch.arange(rows, dtype=torch.int32).repeat(reqs)
        temperature = torch.full((reqs,), temp, dtype=torch.float32)
        seeds = torch.tensor(rng.integers(1, 2**31, size=reqs), dtype=torch.int64)
        sampled, num_sampled = rejection_sample(
            target, draft_logits, draft_sampled.flatten(), cu, pos, idx, expanded_idx, local_pos,
            temperature, seeds, k + L)
        for r in range(reqs):
            n_r = int(num_sampled[r])
            for j in range(n_r):
                counts[j, int(sampled[r, j])] += 1
            if greedy:
                greedy_records.append((drafts[r].tolist(), sampled[r, :n_r].tolist()))
    return counts, p, greedy_records


def _assert_exact(counts, p, label, must_reach):
    """Chi-square per reached position; positions in ``must_reach`` must have >= 200 samples (the test's power)."""
    for j in range(p.shape[0]):
        n = counts[j].sum()
        assert n >= 200 or j not in must_reach, f"{label}: position {j} reached only {n} times"
        if n < 200:
            print(f"{label} pos {j}: n={int(n)} too few, not tested")
            continue
        pv = _chi2_pvalue(counts[j], p[j])
        tv = 0.5 * np.abs(counts[j] / n - p[j]).sum()
        print(f"{label} pos {j}: n={int(n)} TV={tv:.4f} p={pv:.3g}")
        assert pv > 1e-4, (label, j, pv, tv)


def test_point_mass_lookup_is_exact_under_vllm_rejection_kernels():
    counts, p, _ = _run_cycles(onehot_lookup_columns=True)
    _assert_exact(counts, p, "exact arm", must_reach=range(p.shape[0] - 1))


def test_negative_control_without_point_mass_is_detected():
    counts, p, _ = _run_cycles(onehot_lookup_columns=False, cycles=3)
    k = 2
    pv = _chi2_pvalue(counts[k], p[k])
    n = counts[k].sum()
    tv = 0.5 * np.abs(counts[k] / n - p[k]).sum()
    print(f"control pos {k}: n={int(n)} TV={tv:.4f} p={pv:.3g}")
    assert pv < 1e-6, "a uniform cache row under a deterministic token must bias the first lookup position"


# ------------------------------------------------------------------ served shape: K=4, L=2, vocab 20,000 (3 blocks)
# Each is its own pytest item so a run stays under 10 minutes.

WIDE = dict(k=4, L=2, vocab=20000, reqs=256)
TOPP_SEED = 13  # first seed from 12 whose rows 4 and 5 both have a nucleus of >= 3 tokens (seed 12: row 5 had 1)


def test_wide_temperature1_k4_l2_exact():
    counts, p, _ = _run_cycles(cycles=6, seed0=11, temp=1.0, **WIDE)
    _assert_exact(counts, p, "wide T=1", must_reach=range(6))


def test_wide_temperature07_top_p_k4_l2_exact():
    # Row K+1 (second lookup) puts the lookup token outside the top-p nucleus: p(x) = 0 after processing, so that
    # proposal must always be rejected and the position resampled from the nucleus (the masked-target NaN path).
    # Row K (first lookup) keeps the lookup token only moderately likely, so both lookup rows have a nucleus of several
    # tokens (a first run boosted the lookup token in every row: at T=0.7 it alone filled the 0.9 nucleus, every row was
    # a point mass and TV was 0.0000 everywhere -- passing but uninformative). Rows 0..K-1 stay near point masses so the
    # drafts are lookup tokens and the lookup rows are reached.
    counts, p, _ = _run_cycles(cycles=6, seed0=TOPP_SEED, temp=0.7, top_p=0.9, top_p_out_row=5, moderate_row=4, **WIDE)
    nucleus = (p > 0).sum(-1)
    print(f"wide T=0.7 top-p 0.9 nucleus sizes per row {nucleus.tolist()}, p(lookup) row 4 {p[4, WIDE_LOOKUP_TOKEN]:.3f}")
    assert nucleus[4] >= 3 and nucleus[5] >= 3 and 0.1 < p[4, WIDE_LOOKUP_TOKEN] < 0.9, "degenerate top-p design"
    assert p[5, WIDE_LOOKUP_TOKEN] == 0.0
    assert counts[5, WIDE_LOOKUP_TOKEN] == 0, "a token outside the nucleus was emitted"
    _assert_exact(counts, p, "wide T=0.7 top-p 0.9", must_reach=range(6))


def test_wide_greedy_k4_l2_exact():
    # Greedy: every emitted token is the target argmax, accepted drafts are exactly the longest prefix that equals the
    # argmax rows, and one target token follows. Row K+1's argmax is NOT the lookup token (a greedy lookup rejection).
    counts, p, recs = _run_cycles(cycles=4, seed0=13, temp=0.0, greedy_reject_row=5, **WIDE)
    argmax = p.argmax(-1)
    reached_l1 = reached_l2 = 0
    for drafts, emitted in recs:
        a = 0
        while a < len(drafts) and drafts[a] == argmax[a]:
            a += 1
        assert emitted == drafts[:a] + [int(argmax[a])], (drafts, emitted)
        reached_l1 += a >= 4
        reached_l2 += a >= 5
    print(f"wide greedy: {len(recs)} cycles, lookup row 1 reached {reached_l1}, lookup row 2 reached {reached_l2}")
    assert reached_l1 >= 100 and reached_l2 >= 100, "greedy test lacks lookup coverage"
    assert argmax[5] != WIDE_LOOKUP_TOKEN and argmax[4] == WIDE_LOOKUP_TOKEN


def test_wide_posctl_stale_accepts_every_reached_lookup():
    # Stale control: q(x) = 0 at the proposed token, so no reached lookup row is ever rejected: every request that
    # reaches lookup row 1 (position 4) also reaches positions 5 and 6. (A first version asserted the lookup-token share at
    # position 4, but the proposal is not always that token; the invariant is acceptance, not the token.)
    counts, p, _ = _run_cycles(cycles=1, seed0=15, temp=1.0, posctl_eps="stale", moderate_row=4, **WIDE)
    n4, n5, n6 = counts[4].sum(), counts[5].sum(), counts[6].sum()
    print(f"wide posctl stale: reached pos 4 {int(n4)}, pos 5 {int(n5)}, pos 6 {int(n6)}")
    assert n4 >= 100 and n5 == n4 and n6 == n5


def test_wide_posctl_stale2_breaks_only_the_second_lookup_row():
    # Intermediate control: lookup row 1 exact (some reached row-1 proposals rejected), row 2 never rejected.
    # (cycles=2: one cycle reached row 2 only 44 times, below this test's own floor of 50)
    counts, p, _ = _run_cycles(cycles=2, seed0=16, temp=1.0, posctl_eps="stale2", moderate_row=4, **WIDE)
    n4, n5, n6 = counts[4].sum(), counts[5].sum(), counts[6].sum()
    print(f"wide posctl stale2: reached pos 4 {int(n4)}, pos 5 {int(n5)}, pos 6 {int(n6)}")
    assert n4 >= 100 and n5 < n4 and n5 >= 50 and n6 == n5


def test_wide_posctl_eps_is_detected():
    # Positive control: the lookup columns state q = (1-e) delta + e uniform while the
    # token stays deterministic; the first lookup position must be biased toward the lookup token. That row's lookup
    # token is only moderately likely (a first run at p = 0.92 left the bias no room: TV 0.058, chi2 p 2.8e-4 at n 459).
    counts, p, _ = _run_cycles(cycles=2, seed0=14, temp=1.0, posctl_eps=0.5, moderate_row=4, **WIDE)
    j = 4
    n = counts[j].sum()
    pv = _chi2_pvalue(counts[j], p[j])
    tv = 0.5 * np.abs(counts[j] / n - p[j]).sum()
    print(f"wide posctl eps=0.5 pos {j}: n={int(n)} TV={tv:.4f} p={pv:.3g} p(lookup)={p[j, WIDE_LOOKUP_TOKEN]:.3f} "
          f"emitted share={counts[j, WIDE_LOOKUP_TOKEN] / n:.3f}")
    assert pv < 1e-6
