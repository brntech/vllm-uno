# v0.2.0 release validation record

## Status

**BLOCKED.** Do not tag, push, publish, or attach the AMD64 image to a
release. The exact candidate image and its source archive are preserved for a
retest, but the strict greedy gate failed and the candidate logged three JIT
compilations during serving traffic. Every number in this record is named in
the committed [validation receipt](../../evidence/release-0.2.0/validation.json).

| Check | Result | Receipt field |
|---|---|---|
| Package suite | PASS, 13 tests | `local_package_gates.test_release_green` |
| Inverted manifest base | expected-red, exit 1 | `local_package_gates.test_release_inverted_base` |
| Plain reference | PASS, n=256 | `reference` |
| Strict greedy equality | FAIL, 0 of 4 prompts at 256 tokens | `candidate_comparison.greedy` |
| Sampled chunk-1 gate | PASS, n=256, 32 tests, 5,000 permutations | `candidate_comparison.sampled_chunk1` |
| Mixed chunk-8 gate | PASS, n=256, 32 tests, 5,000 permutations | `candidate_comparison.mixed_chunk8` |
| Live HTTP and C=32 capacity | PASS, 32 of 32 completions | `live_http` |
| Monitored serving JIT | FAIL, 3 compilations | `capture_and_jit` |
| Wrong-adapter mutation | expected-red, inner exit 1 | `known_red` |

The sampled gates are real distributional checks, but they do not waive the
hard greedy comparator. The MRV2 source discusses graph-mode plain-engine
variation on an RTX 3090 in
`tests/v1/e2e/spec_decode/test_uno.py`; this run did not add the required
second plain-engine control, so it does not attribute the greedy failure to
that behavior. The distributed kit's `compare.py` remains authoritative here.

## Profile and engagement

The blocked artifact is the Model Runner V2 source head
`689b11a8cac0e6865786c41cc0d77afa6afaf885` over base
`b87339888d29329c42c42573e34cc2beebdcc48b`, on the digest-pinned AMD64 base
listed in the receipt. It serves Qwen3-8B BF16 with K=8, asynchronous
scheduling, prefix caching, FlashAttention 2, and the pinned Uno adapter.

The C=32 arithmetic is 32 requests × K=8 = 256 draft rows. The final profile
captures through 256, the startup log reports coverage for all 32 request
counts, and the final self-check reports graph replay. The live C=8 and C=32
requests completed in full. The candidate's speculative-config line, graph
replay line, and positive draft/draft-token/accepted-token deltas establish
that Uno engaged; the matched plain reference deliberately had no speculative
configuration.

The source fixtures are **CPU_OBSERVED/CUDA_UNVERIFIED** until a device run
proves them. The full-chunk fixture names the 2,048-token bound, while the
production K=8 fixture observes the two-output-token path at 11 tokens for
each of four sampling modes. The static draft dispatcher enumerates the full
request range; the device record then confirms the adopted 256-row C=32
shape. These fixtures are source-contract evidence, not a claim that this
lane ran CUDA unit fixtures.

## Capture correction and JIT failure

Failure class: the first release profile transposed the C=4 capture list to
C=32 without extending the draft-row bound. Corrective action: capture size
256 was added, and `tests/test_release.py::test_c32_profile_captures_full_draft_rows`
rejects a profile whose largest capture is below 32 × 8. Retrospective: profile
flags must be validated against derived request-row bounds rather than copied
from a smaller workload.

The final service did capture and replay its 256-row graph. An early eager
message belongs to a provisional startup pass before the final capture; it is
not used as a serving-capacity claim. The later final replay receipt is the
capacity evidence. That does not cure the independent cold-serving failure:
after the clean startup self-check, the sampled candidate traffic compiled
`_compute_local_logits_stats_kernel`, `_rejection_kernel`, and
`_resample_kernel`. The serving compilation count is therefore 3, not 0.

## Conventions and limits

The shared-prefix requests are `cold_validation_only`: health, greedy,
sampled, and streaming requests preceded them, and they are not a comparison
cell. The C=8 and C=32 requests are functional capacity checks, not latency or
throughput measurements.

No performance cells were measured. Consequently this record reports no
runner-to-runner ratios, no three-repeat performance cells, and no
median/range/overlap claim. The distributional gates state their own sample
count, permutation budget, and statistical test count in the receipt instead
of synthesizing a performance conclusion.

Eight successive real server updates produced eight `nvidia-smi` device-memory
samples at 20,698 MiB on a 24,576 MiB device. There is no training loop.
`torch.cuda.memory_allocated` and `torch.cuda.memory_reserved` were not
available to this external API harness, so this is a device-residency record,
not a leak or retained-tensor conclusion.

## Caller sweep and retest scope

The callers of `stack.py` `BASE` were swept: `apply`, overlay,
`release/check.py`, `tests/test_release.py`, and the Dockerfile production
caller. The fresh-base application and the green package suite exercise the
production path; reverting the manifest base produces the recorded exit 1.
The variant-wide sweep covered the package checker, image dry run, plain
reference, strict greedy, sampled chunk-1, mixed chunk-8, live HTTP modes,
C=8, C=32, residency, and the wrong-adapter mutation.

Retest only after the MRV2 source resolves the strict-parity and cold-serving
JIT failures, then rebuild from the corrected source head and rerun every
receipt above. Do not bypass the strict comparator or pre-warm away the
serving JIT warnings.
