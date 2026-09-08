#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""G2 v2: sampled distribution discrepancy, per prefix and generated position.

Fatal by default: exchangeable-label permutation tests of unmerged empirical TV,
with p = (1 + # null TV >= observed TV) / (B + 1), rejected at p <= alpha / m.
m counts actual unique, nondegenerate, non-smoke tests: positions 2..T,
requested short joints within T, and the full-sequence joint. Position 1 is
smoke only; END includes early termination. No comparisons is FAIL.
PASS means no discrepancy detected at this sample size, not proven equivalence.
Samples must be exchangeable under the null; paired/reused seeds or correlated
requests can violate that assumption. Bonferroni covers this invocation only.

Default B = ceil(10*m/alpha)-1 (ten p-value grid steps at alpha/m).
Explicit --perm must allow rejection: 1/(B+1) <= alpha/m, otherwise refuse,
including --perm 0 (no asymptotic fallback). This validates resolution, not
power against every alternative. Larger budgets improve Monte Carlo precision.

TV vs --factor (1.5) times the maximum floor TV of the same kind is ADVISORY,
as is --tv-tol. --strict-tv restores fatal TV exceedances in addition to p.
With --floor, report that legacy threshold alongside a calibrated per-test
null TV quantile at 1-alpha/m from the CURRENT pooled-label permutations.
One old floor summary cannot reconstruct its sampling null. The quantile uses
the same finite-B correction as p, not factor*floor. New summaries include it
even without --floor. --min-count affects only reported merged support size;
sparse categories remain in the permutation test.

Inputs require equal sampling configs (except mixed_greedy), prefix token IDs,
and nonempty equal sample counts. Candidate subsets of reference prefixes are
allowed for legacy --only callers; m uses that subset. No candidate extras.
CPU only: NumPy multivariate-hypergeometric counts have exactly the same label
randomization law as shuffling pooled observations, without Python token loops.

Usage: lossless_spec_compare.py ref.json cand.json [--floor floor.json]
       [--summary out.json] [--perm B] [--strict-tv]
"""
import argparse
import collections
import json
import math

import numpy as np

END = "<END>"
DEFAULT_TAIL_STEPS = 10


class PermutationResolutionError(ValueError):
    """A valid comparison needs a finer permutation p-value grid."""

    def __init__(self, alpha, tests, requested, minimum):
        super().__init__(
            f"underpowered --perm {requested}: {tests} tests at alpha={alpha:g} "
            f"require at least {minimum} permutations so min_p <= alpha/m "
            f"({alpha / tests:.6g}); omit --perm for the derived default")
        self.summary = {
            "verdict": "ERROR", "error": "insufficient_permutation_resolution",
            "tests": tests, "alpha": alpha, "alpha_bonferroni": alpha / tests,
            "permutations": requested, "minimum_permutations": minimum,
        }


def permutation_budget(alpha, m, requested=None):
    """Choose B, or refuse an explicit budget whose p grid cannot reject."""
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and between 0 and 1")
    if m < 1:
        raise ValueError("zero comparisons were made")
    cutoff = alpha / m
    minimum = math.ceil(1 / cutoff) - 1
    while 1 / (minimum + 1) > cutoff:
        minimum += 1
    if requested is None:
        return max(minimum, math.ceil(DEFAULT_TAIL_STEPS / cutoff) - 1)
    if not isinstance(requested, int) or requested < minimum:
        raise PermutationResolutionError(alpha, m, requested, minimum)
    return requested


def _counts(ca, cb):
    if not ca or not cb:
        raise ValueError("samples must be nonempty")
    ha, hb = collections.Counter(ca), collections.Counter(cb)
    keys = list(dict.fromkeys([*ha, *hb]))
    return (np.array([ha[k] for k in keys], dtype=np.int64),
            np.array([hb[k] for k in keys], dtype=np.int64))


def tv_stat(ca, cb):
    ha, hb = _counts(ca, cb)
    return float(np.abs(ha / len(ca) - hb / len(cb)).sum() / 2)


def permutation_test(ca, cb, n_perm, alpha, rng):
    """Corrected MC p and null quantile conditional on CURRENT pooled counts.

    Integer TV numerators avoid floating-point tie-breaking. Let
    k=floor(alpha*(B+1)); the critical order statistic is sorted(null)[B-k].
    Exceeding it agrees with the conservative >=-ties p test. It is not an
    estimated population equivalence bound or the historical floor's null.
    """
    permutation_budget(alpha, 1, n_perm)
    ha, hb = _counts(ca, cb)
    na, nb = len(ca), len(cb)
    pooled = ha + hb
    denominator = 2 * na * nb
    observed = int(np.abs(ha * nb - hb * na).sum())
    if len(pooled) == 1:
        return {"p": None, "tv": 0.0, "null_tv_quantile": 0.0}
    null = np.empty(n_perm, dtype=np.int64)
    batch_size = max(1, min(1024, 1_000_000 // len(pooled)))
    for start in range(0, n_perm, batch_size):
        stop = min(start + batch_size, n_perm)
        draws = rng.multivariate_hypergeometric(pooled, na, size=stop - start)
        null[start:stop] = np.abs(draws * nb - (pooled - draws) * na).sum(axis=1)
    p = (1 + int(np.count_nonzero(null >= observed))) / (n_perm + 1)
    k = math.floor(alpha * (n_perm + 1))
    # Use the same floating-point cutoff comparison as the p verdict.
    while (k + 1) / (n_perm + 1) <= alpha:
        k += 1
    while k / (n_perm + 1) > alpha:
        k -= 1
    index = n_perm - k
    quantile = float(np.partition(null, index)[index] / denominator)
    return {"p": p, "tv": observed / denominator, "null_tv_quantile": quantile}


def cat_at(samples, t):
    return [(s[t - 1] if len(s) >= t else END) for s in samples]


def joint_at(samples, t):
    return [tuple(s[:t]) + ((END,) if len(s) < t else ()) for s in samples]


def _comparisons(A, B, joint, min_count):
    cfg_a = {k: v for k, v in A["config"].items() if k != "mixed_greedy"}
    cfg_b = {k: v for k, v in B["config"].items() if k != "mixed_greedy"}
    if cfg_a != cfg_b:
        raise ValueError(f"sampling configs differ: {cfg_a} vs {cfg_b}")
    T = cfg_a["max_tokens"]
    if not isinstance(T, int) or T < 1:
        raise ValueError("max_tokens must be a positive integer")
    if min_count < 1:
        raise ValueError("min-count must be positive")
    if any(not isinstance(t, int) or t < 1 for t in joint):
        raise ValueError("joint lengths must be positive integers")
    ids = [pid for pid in A["prefixes"] if pid in B["prefixes"]]
    if not ids:
        raise ValueError("no common prefixes")
    if not set(B["prefixes"]) <= set(A["prefixes"]):
        raise ValueError("candidate has prefixes the reference lacks")
    # Joint1 is identical to smoke pos1; out-of-range joints add no information.
    joints = sorted({t for t in (*joint, T) if 2 <= t <= T})
    comparisons = []
    for pid in ids:
        pa, pb = A["prefixes"][pid], B["prefixes"][pid]
        if pa["ids"] != pb["ids"]:
            raise ValueError(f"prefix token ids differ for {pid}")
        sa, sb = pa["samples"], pb["samples"]
        if not sa or not sb:
            raise ValueError(f"empty samples for {pid}")
        if len(sa) != len(sb):
            raise ValueError(f"sample counts differ for {pid}")
        if any(not isinstance(s, list) or len(s) > T for s in (*sa, *sb)):
            raise ValueError(f"invalid sample length for {pid}")
        if any(not isinstance(token, int) for s in (*sa, *sb) for token in s):
            raise ValueError(f"sample token ids must be integers for {pid}")
        specs = [(f"pos{t}", t == 1, cat_at, t) for t in range(1, T + 1)]
        specs += [(f"joint1..{t}", False, joint_at, t) for t in joints]
        for name, smoke, extract, t in specs:
            ca, cb = extract(sa, t), extract(sb, t)
            ha, hb = _counts(ca, cb)
            pooled = ha + hb
            merged = int(np.count_nonzero(pooled >= min_count)) + int(np.any(pooled < min_count))
            row = {"prefix": pid, "test": name, "support": len(pooled),
                   "merged": merged, "smoke": smoke}
            comparisons.append((row, ca, cb))
    return T, comparisons


def compare_runs(A, B, *, alpha=0.01, permutations=None, floor=None,
                 strict_tv=False, tv_tol=None, factor=1.5, min_count=5,
                 joint=(2, 3), seed=0):
    """Return JSON-serializable verdict; invalid inputs/budgets raise ValueError."""
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("factor must be finite and positive")
    if tv_tol is not None and (not math.isfinite(tv_tol) or tv_tol < 0):
        raise ValueError("tv-tol must be finite and nonnegative")
    T, comparisons = _comparisons(A, B, joint, min_count)
    m = sum(not r["smoke"] and r["support"] > 1 for r, _, _ in comparisons)
    budget = permutation_budget(alpha, m, permutations)
    alpha_b = alpha / m

    def kind(name):
        return "pos" if name.startswith("pos") else (
            "joint_full" if name == f"joint1..{T}" else "joint_short")

    floor_kind = {}
    if floor is not None:
        # Legacy floors have no sample/config metadata. Preserve their per-kind
        # diagnostic threshold without presenting it as a calibrated null.
        for r in floor["rows"]:
            if r["smoke"]:
                continue
            value = r["tv"]
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("floor TV must be finite and between 0 and 1")
            kd = kind(r["test"])
            floor_kind[kd] = max(floor_kind.get(kd, 0.0), value)
        needed = {kind(r["test"]) for r, _, _ in comparisons if not r["smoke"]}
        if not needed <= floor_kind.keys():
            raise ValueError(f"floor lacks test kinds: {sorted(needed - floor_kind.keys())}")

    rng = np.random.default_rng(seed)
    rows = []
    for row, ca, cb in comparisons:
        row.update(permutation_test(ca, cb, budget, alpha_b, rng))
        row["null_quantile_level"] = 1 - alpha_b
        row["null_quantile_source"] = "current pooled-label permutations"
        row["floor_tv"] = floor_kind.get(kind(row["test"])) if floor is not None else None
        row["tol"] = None if row["smoke"] else (
            factor * row["floor_tv"] if floor is not None else tv_tol)
        rows.append(row)

    tests = [r for r in rows if not r["smoke"] and r["p"] is not None]
    fails_p = [r for r in tests if r["p"] <= alpha_b]
    tv_exceedances = [r for r in rows if r["tol"] is not None and r["tv"] > r["tol"]]
    fails_tv = tv_exceedances if strict_tv else []
    return {"verdict": "FAIL" if fails_p or fails_tv else "PASS", "tests": m,
            "alpha": alpha, "alpha_bonferroni": alpha_b, "permutations": budget,
            "min_attainable_p": 1 / (budget + 1),
            "tail_grid_steps": alpha_b * (budget + 1), "seed": seed,
            "min_p": min(r["p"] for r in tests),
            "max_tv": max(r["tv"] for r in rows if not r["smoke"]),
            "tv_tol": tv_tol, "factor": factor, "strict_tv": strict_tv,
            "tv_mode": "strict" if strict_tv else "advisory",
            "fails_p": fails_p, "fails_tv": fails_tv,
            "tv_exceedances": tv_exceedances, "rows": rows}


def _parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("cand")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--tv-tol", type=float, default=None, help="advisory global TV threshold without --floor")
    ap.add_argument("--floor", default="", help="plain-vs-plain summary; legacy per-kind TV threshold is advisory")
    ap.add_argument("--factor", type=float, default=1.5)
    ap.add_argument("--strict-tv", action="store_true", help="make legacy TV exceedances fatal as well as permutation p")
    ap.add_argument("--min-count", type=int, default=5, help="merged support reporting only")
    ap.add_argument("--joint", default="2,3")
    ap.add_argument("--summary", default="")
    ap.add_argument("--perm", type=int, default=None, help="permutations; default ceil(10*m/alpha)-1; underpowered explicit budgets refused")
    return ap


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(argv=None):
    a = _parser().parse_args(argv)
    try:
        A, B = _load(a.ref), _load(a.cand)
        result = compare_runs(
            A, B, alpha=a.alpha, permutations=a.perm,
            floor=_load(a.floor) if a.floor else None, strict_tv=a.strict_tv,
            tv_tol=a.tv_tol, factor=a.factor, min_count=a.min_count,
            joint=tuple(int(x) for x in a.joint.split(",") if x))
    except PermutationResolutionError as exc:
        if a.summary:
            with open(a.summary, "w", encoding="utf-8") as f:
                json.dump(exc.summary, f, indent=1, allow_nan=False)
        print(f"G2v2 FAIL: {exc}")
        return 2
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"G2v2 FAIL: {exc}")
        return 2
    result.update(ref=a.ref, cand=a.cand, floor=a.floor)
    cur = None
    for r in result["rows"]:
        if r["prefix"] != cur:
            cur = r["prefix"]
            pa, pb = A["prefixes"][cur], B["prefixes"][cur]
            print(f"== {cur}: n={len(pa['samples'])}/{len(pb['samples'])} "
                  f"acceptance ref={pa.get('draft_acceptance_rate')} cand={pb.get('draft_acceptance_rate')}")
        ps = "deg" if r["p"] is None else f"{r['p']:.6g}"
        flag = " FAIL-p" if r in result["fails_p"] else ""
        if r in result["tv_exceedances"]:
            flag += " FAIL-tv" if a.strict_tv else " ADVISORY-tv"
        tol_s = "" if r["tol"] is None else f" legacy_tol={r['tol']:.3f}"
        q_s = f" null_TV_q={r['null_tv_quantile']:.3f}" if a.floor else ""
        print(f"  {r['test']:<10} p={ps:<10} TV={r['tv']:.3f}{tol_s}{q_s} "
              f"support={r['support']:<4} merged={r['merged']:<3}"
              f"{' (smoke)' if r['smoke'] else ''}{flag}")
    print(f"tests={result['tests']} bonferroni_alpha={result['alpha_bonferroni']:.6g} "
          f"perm={result['permutations']} min_attainable_p={result['min_attainable_p']:.6g} "
          f"tail_grid_steps={result['tail_grid_steps']:.3g} min_p={result['min_p']:.6g} "
          f"max_TV={result['max_tv']:.3f} fails_p={len(result['fails_p'])} "
          f"fails_tv={len(result['fails_tv'])} tv_exceedances={len(result['tv_exceedances'])}")
    if a.floor:
        print(f"null_TV_q: per-test level={1-result['alpha_bonferroni']:.9g}, "
              "current pooled-label permutations; floor max-by-kind is uncalibrated")
    print(f"G2v2 {result['verdict']} (permutation p-values, {result['tests']} tests, "
          f"TV {result['tv_mode']}; PASS means no discrepancy detected)")
    if a.summary:
        with open(a.summary, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1, allow_nan=False)
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
