# v0.2.0 release validation record

## Status

**READY.** The AMD64 image is eligible for the maintainer's tag, registry push,
and release attachment steps. The corresponding machine-readable receipt is
[`evidence/release-0.2.0/validation.json`](../../evidence/release-0.2.0/validation.json).

| Check | Result | Public receipt field |
|---|---|---|
| Patch manifest and package check | PASS | `local_package_gates` |
| Reverted manifest base | expected-red, exit 1 | `local_package_gates.test_release_inverted_base` |
| Plain reference | PASS, n=256 | `reference` |
| Functional greedy | PASS, 4 × 256-token requests | `candidate_comparison.greedy_functional` |
| Strict greedy | `NOT_A_RELEASE_GATE` | `strict_greedy_protocol` |
| Sampled G2v2 | PASS, n=256, 32 tests, 5,000 permutations | `candidate_comparison.sampled_chunk1` |
| Mixed G2v2 | PASS, n=256, 32 tests, 5,000 permutations | `candidate_comparison.mixed_chunk8` |
| Health, streaming, prefix, C=8, C=32 | PASS | `live_http` |
| In-serving JIT | PASS, 0 warnings | `capture_and_jit` |
| Wrong adapter revision | expected-red, inner exit 1 | `known_red` |

## Identity and profile

The Model Runner V2 source head is
`5da193919b44335ddf14eac193dfc9e8d5e59df5` over base
`b87339888d29329c42c42573e34cc2beebdcc48b`. The reconstructed patch tree is
`6ceef9dfa043d9a2d3f930522ecc7480105aa5a7`; LF-normalized patch SHA-256 is
`447acf006163385ec2b1d902f9cff1e04c60b7e8dac001941f1130164262e382`.

The validated local image is
`sha256:98034bbbf7042147838932d64e0ff7a8668bc91d1d203d4a76708d14231c1037`,
9,956,886,869 bytes, built from the digest-pinned AMD64 CI image. Its saved
relaunch archive is 9,956,934,144 bytes, SHA-256
`3fca059e167f3b8ce08b74a1d4baac7785c9b0ecfe31a512f04a0293272e74bc`.

The production shape is Qwen3-8B BF16, K=8, async scheduling, prefix caching,
FlashAttention 2, one API process, `max_num_seqs=16`,
`max_num_batched_tokens=2048`, explicit 2 GiB KV cache, and capture sizes
`[1,2,4,8,16,32,64,128,144]`. C=32 queues at the 16-request admission limit.

## Engagement, correctness, and scope

Candidate startup records the Uno speculative configuration, all 16 requested
warm-up shapes, startup self-check with no compilation, and graph replay. The
post-startup serving slice contains zero `JIT compilation during inference`
warnings across all validation traffic. Gate counters recorded 2,738 drafts,
21,904 draft tokens, and 8,092 accepted tokens; live traffic added 45 / 360 /
70 respectively.

The sampled and mixed G2v2 gates are the release lossless evidence. Strict
greedy is not a gate on this RTX 3090 CUDA-graph instrument: the documented
source control in `docs/lanes/bl-mrv2-final-gates.md` attributes graph-mode
plain self-flips and later completes the separate-engine seven-row control.
No plain-versus-plain double run was performed here.

Eight real served residency updates all observed 21,504 MiB / 24,576 MiB by
host `nvidia-smi`. This is device residency only, not allocated/reserved memory
or a leak conclusion. This lane reports no performance cells, ratios, repeat
medians, ranges, or overlaps.

Fixture evidence remains **CPU_OBSERVED/CUDA_UNVERIFIED**. The BF16 source
contract retains the 2,048 full-chunk two-iteration bound and K=8 production
extent; the RTX 3090 receipt proves the actual 16×8 draft-row variant. The
caller sweep covers `stack.py` apply, overlay, check, tests, and the Dockerfile.
The known-red changes only the pinned adapter revision in the same candidate
runner and exits 1, proving the stated behavior rather than a missing symbol.
