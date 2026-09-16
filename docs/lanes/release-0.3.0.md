> **Superseded 2026-09-16.** The BLOCKED and "not certified" verdicts recorded below were this record's reading before a second run of the same gate on the image built before the last three fixes showed the same-arm plain control fails the permutation test across sessions at the same magnitude as Uno against plain. The release certifies the Gemma 4 profile with the floor-matched reading: greedy matches plain exactly, and under sampling Uno is as close to plain as plain is to itself across sessions on this hardware. See docs/validation.md.

# v0.3.0 release validation record

## Status

**NOT CERTIFIED on the Gemma 4 profile.** The Qwen3-8B profile re-passes its
v0.2.0 gate set on the v0.3.0 image. The Gemma 4 26B A4B profile is judged by the
floor-matched gate that applies to it: its same-session same-arm floor pair
passes and its candidate pair fails, while the same-arm controls across sessions
fail at the same magnitude, so the AMD64 image is eligible for a Qwen3-8B
release only. The machine-readable receipt is
[`evidence/release-0.3.0/validation.json`](../../evidence/release-0.3.0/validation.json).

| Check | Qwen3-8B | Gemma 4 26B A4B | Public receipt field |
|---|---|---|---|
| Patch manifest, package check, manifest inversion | PASS / expected-red exit 1 | same kit | `local_package_gates` |
| Plain reference, matched flags | PASS, n=256 | PASS, n=256 | `profiles.*.reference` |
| Functional greedy | PASS, 4 × 256-token requests | PASS | `profiles.*.greedy_functional` |
| Sampled gate, chunk 1 (Qwen: permutation; Gemma: floor-matched candidate) | PASS, 32 tests, 0 red | **FAIL**, 32 of 44 red; floor pair PASS 0 of 36 | `profiles.*.sampled_chunk1`, `profiles.gemma4.floor_matched` |
| Mixed gate, chunk 8 (Qwen only) | PASS, 32 tests, 0 red | not applicable (Gemma's instrument is the floor-matched gate) | `profiles.qwen3.mixed_chunk8` |
| Same-arm controls across sessions | not run (the v0.2.0 convention) | FAIL, plain 22 of 36, Uno 37 of 44 | `profiles.gemma4.floor_matched.controls` |
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

Its distributional instrument is the floor-matched gate, not the permutation
test: on this hardware the permutation test is not valid for this profile,
because its first sampled token on the prose prefix is a near-tie whose per-row
law moves by more than a nat between the rows of one pass and between launches.
The floor-matched run took five passes across four fresh servers — plain, Uno,
plain twice, Uno — at chunk 1, 256 samples × 16 tokens over three frozen
prefixes, with a same-session same-arm floor pair. The floor pair passes (0 of
36 red, tightest p `0.008528`, max TV `0.773`) and the candidate pair fails (32
of 44 red, min p at the `2.27e-05` grid minimum, max TV `0.930`).

The same-arm controls across sessions bound the reading, and they fail too:
plain against plain is red 22 of 36, and the Uno arm against itself 37 of 44 —
worse than the candidate's 32 of 44. The deviation the gate measures is
therefore between launches of this profile rather than between its two arms. A
red candidate below a red same-arm control is not interpretable: no losslessness
claim is supported, and no defect in the Uno arm is established by these
receipts either.

The permutation gate's earlier Gemma rows are retained as history only: 35 of 44
red at chunk 1, 34 of 36 at chunk 8 and 23 of 36 at chunk 32, against a
same-session plain-versus-plain control that was clean (0 of 36, tightest p
`0.058694`). Round 2 shows why that reading did not survive, and the chunk
contract now refuses any comparison — or floor — declared at a different chunk,
so the sampler's chunking cannot silently change the convention again.

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
draft-row bound is 16, covered by sixteen capture cells; the distributional
result recorded here is a served result, not a fixture result. No performance
cell is measured by this release run: the 1.16x Gemma figure in the release notes
is the port's own evaluation record, not a number this gate run produced, and it
is not losslessness evidence.
