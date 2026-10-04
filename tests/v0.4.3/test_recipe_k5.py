"""CPU exactness at the shape the v0.4.3 recipe serves: K=5 Uno drafts + L=2 lookup rows (verify width 8), gated
requests at K=5 (verify width 6), vocab 20,000 (three 8192-wide blocks), through vLLM's own rejection kernels in the
Triton interpreter.

The first three tests are the K=5 + L=2 versions of test_plookup.py's wide tests (own seeds and thresholds); the last
is test_plgate.py's mixed gated/full batch at K=5. Draft-only expert routing (UNO_DRAFT_MOE_TOPK=4) changes
which q the drafts are sampled from, never the verification; these tests sample drafts from an arbitrary q, so they
cover it. Run with the other release suites (workdir /t): pytest test_recipe_k5.py.
"""

import test_plookup as tp  # sets TRITON_INTERPRET and the driver shim before vLLM imports
import test_plgate as tg

WIDE5 = dict(k=5, L=2, vocab=20000, reqs=256)


def test_wide_temperature1_k5_l2_exact():
    counts, p, _ = tp._run_cycles(cycles=6, seed0=11, temp=1.0, **WIDE5)
    tp._assert_exact(counts, p, "K5 wide T=1", must_reach=range(7))


def test_wide_temperature07_top_p_k5_l2_exact():
    for seed in range(13, 40):  # first seed whose two lookup rows both keep a nucleus of >= 3 tokens (as TOPP_SEED)
        counts, p, _ = tp._run_cycles(cycles=1, seed0=seed, temp=0.7, top_p=0.9, top_p_out_row=6, moderate_row=5,
                                      reqs=8, k=5, L=2, vocab=20000)
        nucleus = (p > 0).sum(-1)
        if nucleus[5] >= 3 and nucleus[6] >= 3 and 0.1 < p[5, tp.WIDE_LOOKUP_TOKEN] < 0.9:
            break
    print(f"K5 top-p design seed {seed}")
    counts, p, _ = tp._run_cycles(cycles=6, seed0=seed, temp=0.7, top_p=0.9, top_p_out_row=6, moderate_row=5, **WIDE5)
    nucleus = (p > 0).sum(-1)
    print(f"K5 T=0.7 top-p 0.9 nucleus sizes per row {nucleus.tolist()}, p(lookup) row 5 {p[5, tp.WIDE_LOOKUP_TOKEN]:.3f}")
    assert nucleus[5] >= 3 and nucleus[6] >= 3 and 0.1 < p[5, tp.WIDE_LOOKUP_TOKEN] < 0.9, "degenerate top-p design"
    assert p[6, tp.WIDE_LOOKUP_TOKEN] == 0.0
    assert counts[6, tp.WIDE_LOOKUP_TOKEN] == 0, "a token outside the nucleus was emitted"
    tp._assert_exact(counts, p, "K5 wide T=0.7 top-p 0.9", must_reach=range(7))


def test_wide_greedy_k5_l2_exact():
    counts, p, recs = tp._run_cycles(cycles=4, seed0=13, temp=0.0, greedy_reject_row=6, **WIDE5)
    argmax = p.argmax(-1)
    reached_l1 = reached_l2 = 0
    for drafts, emitted in recs:
        a = 0
        while a < len(drafts) and drafts[a] == argmax[a]:
            a += 1
        assert emitted == drafts[:a] + [int(argmax[a])], (drafts, emitted)
        reached_l1 += a >= 5
        reached_l2 += a >= 6
    print(f"K5 greedy: {len(recs)} cycles, lookup row 1 reached {reached_l1}, lookup row 2 reached {reached_l2}")
    assert reached_l1 >= 100 and reached_l2 >= 100, "greedy test lacks lookup coverage"
    assert argmax[6] != tp.WIDE_LOOKUP_TOKEN and argmax[5] == tp.WIDE_LOOKUP_TOKEN


def test_mixed_batch_gated_and_full_requests_are_exact_k5():
    # Odd requests gated (K=5 rows, 6 logits), even requests full (K + L = 7 rows, 8 logits), one rejection_sample call.
    counts, p, max_emitted = tg._run_mixed(cycles=6, seed0=21, k=5, L=2)
    print(f"K5 mixed: max tokens per cycle full {max_emitted[False]}, gated {max_emitted[True]}")
    assert max_emitted[True] <= 6 and max_emitted[False] == 8
    tp._assert_exact(counts[False], p, "K5 mixed full (K+L)", must_reach=range(8))
    tp._assert_exact(counts[True][:6], p[:6], "K5 mixed gated (K)", must_reach=range(6))
