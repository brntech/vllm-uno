# v0.3.0 release validation record

## Status

**NOT CERTIFIED on the Gemma 4 profile.** The Qwen3-8B profile re-passes its
v0.2.0 gate set on the v0.3.0 image; the Gemma 4 26B A4B profile fails the
sampled-distribution gate against a matched plain reference, so the AMD64 image
is eligible for a Qwen3-8B release only. The machine-readable receipt is
[`evidence/release-0.3.0/validation.json`](../../evidence/release-0.3.0/validation.json).

| Check | Qwen3-8B | Gemma 4 26B A4B | Public receipt field |
|---|---|---|---|
| Patch manifest, package check, manifest inversion | PASS / expected-red exit 1 | same kit | `local_package_gates` |
| Plain reference, matched flags | PASS, n=256 | PASS, n=256 | `profiles.*.reference` |
| Functional greedy | PASS, 4 × 256-token requests | PASS | `profiles.*.greedy_functional` |
| Sampled gate, chunk 1 | PASS, 32 tests, 0 red | **FAIL**, 35 of 44 red | `profiles.gemma4.sampled_chunk1` |
| Mixed gate, chunk 8 | PASS, 32 tests, 0 red | **FAIL**, 34 of 36 red | `profiles.gemma4.mixed_chunk8` |
| Plain-versus-plain control | not run (identical flags both arms are the v0.2.0 convention) | PASS, 0 of 36 red | `profiles.gemma4.controls` |
| Health, streaming, prefix, C=8, C=32 | PASS | PASS | `profiles.*.live_http` |
| In-serving JIT | PASS, 0 warnings | 2 warnings, present in both arms | `profiles.*.serving_compilations` |
| Vision refusal | n/a | expected-red, exit 1, named | `profiles.gemma4.vision_refusal` |
| Draft MoE top-k variant | n/a | boots, serves, refuses by name | `profiles.gemma4.draft_moe_topk` |
| Wrong adapter revision | expected-red, exit 1 | same runner | `known_red` |

## Identity and profile

The release head is `cf87916880b051e8782521dfe2afa12e0627e172` over the v0.2.0
content base `3ad49350281a6b73de58449aadb293a8b398fb5d`. The ordered two-layer
series (`0001-uno-mrv2-base.patch`, 35 files, then `0002-uno-gemma4.patch`,
17 files) reconstructs tree `0149f03eb8287bdfdcc916752b3851405695d350`; the
image's own build provenance records the same tree, and the manifest hash is
`75d6abeefc7ff2094dce35f49198421c2ff973553e8c185a0d82b1148b0e2198` over the
LF-normalized `release/series.json`.

The tested local image is
`sha256:06e2266b10c2eaf1783f3c8a671ec4a43118ccb0540d6a4ed2e2ae46076a4ca9`,
9,992,635,623 bytes, built from the AMD64 postmerge CI image of the upstream
base (digest `sha256:d55cb6858435cda5ab080987213b4a6b6bfce14ca9e0ffa2ecfab2b222818497`).
Its saved archive is 9,992,681,472 bytes, SHA-256
`356edc6dbea34053a03492fb1a960be73fa59b31c25a6e9b1e0ef4fa93cb06cc`.

The Qwen3-8B profile is unchanged from v0.2.0: BF16, K=8, FlashAttention 2,
`max_num_seqs=16`, `max_num_batched_tokens=2048`, explicit 2 GiB KV cache,
capture sizes `[1,2,4,8,16,32,64,128,144]`. The Gemma profile is language-only
Gemma 4 26B A4B AWQ at `TRITON_ATTN`, K=4, `max_num_seqs=4`,
`max_model_len=8192`, `gpu_memory_utilization=0.85`, two LoRA slots, split-KV
engaged, and sixteen capture cells covering the 16-draft-row bound.

## Why the Gemma profile is not certified

Both Gemma arms ran the same flags and differ only by `--speculative-config`.
The Uno arm's sampled token distribution differs from the matched plain arm's at
every position and joint the gate tests, at chunk 1 (36 of 36 on the
three-prefix set), chunk 8 (34 of 36) and chunk 32 (23 of 36), with p at the
minimum the budget can express. On the prose prefix at chunk 1 the Uno arm
returns one token for all 256 draws where plain spreads over four, and position
one is the first draft-and-verify cycle, so the deviation is present before any
later cycle can compound it.

Three same-session controls bound the reading. Same-configuration
plain-versus-plain is clean (0 of 36, tightest p `0.058694`), so the instrument
is not inventing failures at this sample size. Plain with one LoRA slot against
plain with two is red 32 of 36 with no semantic difference between the arms, so
a kernel-selection difference between the plain arm and the Uno verify pass
cannot be excluded as the cause. And plain chunk 1 against plain chunk 32 is red
35 of 36, so the sampler's chunk size is a measured convention rather than a
detail; every comparison here holds it equal on both arms.

Either remaining cause — a verification path that differs semantically from
plain, or a plain arm that does not run the kernel the Uno verify pass runs —
blocks a losslessness claim on the profile as served. The separating follow-up
is a forced all-reject arm against the same reference, plus a dispatch-key line
proving both arms run the same attention kernel.

## What the receipts do certify

The Qwen3-8B regression is green end to end on the v0.3.0 image, including the
local package gates, the manifest inversion, both distributional gates, the live
API and capacity checks, zero in-serving compilations, and the wrong-adapter
expected-red. On the Gemma profile, the functional, capacity, refusal and
variant receipts all pass: greedy decoding, the live API checks, the config-time
vision refusal naming `Uno requires a language-only model`, and the draft MoE
top-k variant, which boots, serves a captured request and refuses the uncaptured
16-draft-row shape by dispatch key, environment variable and variant name.

Fixture evidence remains **CPU_OBSERVED/CUDA_UNVERIFIED**. The Gemma profile's
draft-row bound is 16, covered by sixteen capture cells; the failure recorded
here is a served-distribution result, not a fixture result. No performance cell
is measured by this release run: the 1.16x Gemma figure in the release notes is
the port's own evaluation record, not a number this gate run produced, and it is
not losslessness evidence.
