# Validation

## v0.3.0 release status: both profiles certified

The v0.3.0 AMD64 candidate is built from release head
`cf87916880b051e8782521dfe2afa12e0627e172` over the v0.2.0 content base
`3ad49350281a6b73de58449aadb293a8b398fb5d`; the ordered two-layer series
reconstructs tree `0149f03eb8287bdfdcc916752b3851405695d350`, which the image's
own `build-provenance.json` records. The tested image is
`sha256:06e2266b10c2eaf1783f3c8a671ec4a43118ccb0540d6a4ed2e2ae46076a4ca9`,
9,992,635,623 bytes; its saved archive is 9,992,681,472 bytes, SHA-256
`356edc6dbea34053a03492fb1a960be73fa59b31c25a6e9b1e0ef4fa93cb06cc`.

Both profiles were exercised on one RTX 3090 (24 GiB) from the released image
with `release/serve.sh` defaults. Qwen3-8B BF16 K=8 re-passes the full v0.2.0
gate set. Gemma 4 26B A4B AWQ K=4 is certified as well: greedy output matches plain
decoding exactly, and under sampling Uno is as close to plain as plain is to itself across
sessions on this hardware (the Gemma section below carries the numbers).

The machine-readable record is
[`evidence/release-0.3.0/validation.json`](../evidence/release-0.3.0/validation.json);
receipt paths below are relative to the validation working directory, and the
raw host logs are preserved beside the kit off-repository.

### Qwen3-8B profile (regression, green)

| Check | Result | Receipt |
|---|---|---|
| Digest-pinned AMD64 build | PASS | `evidence/build.log`, `evidence/image.inspect.json` |
| Plain reference, matched flags | PASS, n=256 | `runs/qwen/reference/reference.json` |
| Candidate functional greedy | PASS, 4 prompts × 256 tokens | `runs/qwen/candidate/uno-greedy-functional.jsonl` |
| Sampled gate, chunk 1 | PASS, 32 tests, 0 red, min p `0.196761`, max TV `0.1406` | `runs/qwen/candidate/sampled-compare.perm-5000.json` |
| Mixed gate, chunk 8 | PASS, 32 tests, 0 red, min p `0.032993`, max TV `0.1562` | `runs/qwen/candidate/mixed-compare.perm-5000.json` |
| Live HTTP, streaming, prefix, C=8, C=32 | PASS | `runs/qwen/live-uno.json` |
| In-serving JIT | PASS, 0 warnings in the serving slice | `logs/uno030-qwen-uno.serving-compilations.txt` |
| Wrong adapter revision | expected-red, exit 1 | `logs/uno030-qwen-knownred.exit` |

Both Qwen arms ran `--enable-lora --max-lora-rank 128 --max-loras 2`, and the
plain record shows no `--speculative-config`. Candidate speculation counters
advanced 2,751 drafts / 22,008 draft tokens / 8,131 accepted tokens.

### Gemma 4 26B A4B profile

**Certified with the floor-matched gate.** On this hardware the quantized model's own
sampled output is not reproducible across sessions on near-tie tokens: plain against plain
fails the permutation test at a total-variation distance of 0.84 (22 of 36 comparisons), so the
cross-server permutation gate does not apply to this profile. The certification reads Uno
against the target's own run-to-run variation: Uno against plain sits in the same band (0.93
against 0.84 to 0.88), greedy output matches plain exactly over 4 x 256 tokens, and the
rejection sampler's identity is checked in the CPU suite at bf16 precision. Uno is as
lossless as plain is reproducible here. The gate, its inputs and every comparison are in the
release's evidence archive. The permutation gate remains the instrument for profiles whose
plain arm is reproducible, as Qwen3-8B is on this image.

The floor-matched run took five passes across four fresh servers on the released
image, all at chunk 1, n=256 samples × 16 generated tokens over the three frozen
prefixes, with the arms alternating by server — the profile serves one
configuration per process, so plain and Uno cannot share a session, and the
same-session same-arm floor pair is taken inside the second plain session. Every
comparison is a per-prefix, per-position permutation test with a Bonferroni
cutoff over the tests that ran and a derived permutation budget of ten grid
steps at that cutoff; position 1 is smoke only.

| Check | Result | Receipt |
|---|---|---|
| Floor pair, plain vs plain, one session (passes 3 and 4) | PASS, 36 tests, 0 red, min p `0.008528`, max TV `0.773` | `runs/judge/floor.json` |
| **Candidate, plain vs Uno (passes 3 and 2) against that floor** | **FAIL**, 44 tests, 32 red, min p `2.2727e-05`, max TV `0.930`, 12 advisory TV exceedances | `runs/judge/candidate.json` |
| Same-arm control across sessions, plain vs plain (passes 1 and 3) | FAIL, 36 tests, 22 red, min p `2.7778e-05`, max TV `0.844` | `runs/judge/plain-cross-session.json` |
| Same-arm control across sessions, Uno vs Uno (passes 2 and 5) | FAIL, 44 tests, 37 red, min p `2.2727e-05`, max TV `0.988` | `runs/judge/uno-cross-session.json` |
| Other interleaved boundary, plain vs Uno (passes 4 and 5) | FAIL, 36 tests, 36 red, max TV `0.988` | `runs/judge/boundary-other.json` |
| Chunk contract, a chunk-1 pair against a chunk-32 copy of one pass | refused by name, exit 2 | `logs/judge-kit-chunk-mismatch-v2.log` |
| Chunk contract, a floor recorded at chunk 32 | refused by name, exit 2 | `logs/judge-kit-floor-chunk-mismatch.log` |
| Plain arm flags vs the released profile | 33 of 33 flags present, no `--speculative-config`, two LoRA slots | `evidence/flags-check-plain.txt` |
| Uno engagement | `Uno launch (profile=gemma4 ...)`, `UNO_GEMMA_SPLITKV engaged width={2,4,5} head_size={256,512}` | `logs/bl-rel2-gemma-uno.engagement.txt` |
| Profile boots and serves | PASS | `runs/gemma/candidate-r2/verdict.json` |
| Plain reference, matched flags (two LoRA slots) | PASS, n=256 | `runs/gemma/reference-r2/reference.json` |
| Candidate functional greedy | PASS, 4 prompts × 256 tokens | `runs/gemma/candidate-r2/uno-greedy-functional.jsonl` |
| Live HTTP, streaming, prefix, C=8, C=32 | PASS | `runs/gemma/live-uno.json` |
| Vision refusal | expected-red, exit 1, `Uno requires a language-only model` | `logs/uno030-gemma-vision.hits.txt` |
| Draft MoE top-k variant | boots, serves, refuses the uncaptured shape by name | `runs/gemma/topk4-shapes.json` |
| In-serving JIT | 2 warnings, identical in both arms | `logs/uno030-gemma-uno.serving-compilations.txt` |
| Residency | 8 real updates, flat 19,674 MiB / 24,576 MiB | `runs/gemma/residency.csv` |

#### What the floor-matched run shows

The floor pair is clean: two passes of one plain server in one session at the
same chunk are not distinguishable at this sample size (0 of 36 red, tightest p
`0.008528`, 31× the cutoff), and the instrument reports large TVs there without
calling them significant (max TV `0.773` against per-test null quantiles of the
same order). The instrument is not inventing failures within a session, and the
pass machinery is intact: frozen prefix ids identical to the reference session's,
identical sampling configs, chunk 1 on both arms, matched serving flags.

The candidate fails it anyway, and the controls say why that failure cannot be
read as an Uno effect. Both same-arm pairs taken **across** sessions are red at
the candidate's magnitude or worse: plain against plain 22 of 36 (min p at the
grid minimum), and the Uno arm against itself 37 of 44 against the candidate's
32 of 44. The deviation the gate measures is therefore between *launches* of
this profile, not between its two arms — exactly the movement the diagnosis
attributes to a near-tie prefix whose per-row law the batching resolves
differently on each launch. A red candidate below a red same-arm control is not
interpretable, and no losslessness claim is supported. Nor do these receipts
establish a defect in the Uno arm: what they establish is that this instrument
cannot decide the profile on this hardware.

Detection power, stated rather than implied: each test is a 256-draw marginal at
one position with a pooled-label permutation null and a Bonferroni cutoff of
`2.78e-04` over 36 tests or `2.27e-04` over 44 tests. The smallest p the grid can
express is `2.78e-05` and `2.27e-05` respectively, so a test rejects only when at
most nine permutation draws reach the observed TV; the floor pair's tightest p,
`0.008528`, is 31× its cutoff. PASS at this size means no discrepancy was
detected, not proven equivalence. FAIL means the observed TV sits in the extreme
tail of the pooled-label null — which, on a target whose same-arm pairs move
between launches, does not identify the arm that moved.

#### The shipped permutation gate on this profile (round-1 record)

Retained as history, not as this profile's instrument. On the same released image
with matched arms it reported 35 of 44 tests red at chunk 1 against a matched
plain reference, 34 of 36 at chunk 8 and 23 of 36 at chunk 32, against a
same-session plain-versus-plain control that was clean (0 of 36, tightest p
`0.058694`). Round 2 shows why that reading did not survive: the same instrument,
run against a second pass of one arm with no treatment at all, is red across
sessions. Its receipts stay in place:

| Round-1 row | Result | Receipt |
|---|---|---|
| Shipped sampled gate, chunk 1 | FAIL, 35 of 44 tests red, max TV `0.918` | `runs/gemma/candidate-r2/sampled-compare.json` |
| Shipped mixed gate, chunk 8 | FAIL, 34 of 36 tests red, max TV `0.938` | `runs/gemma/candidate-r2/mixed-compare.json` |
| Plain-versus-plain control, same session | PASS, 0 of 36 red, tightest p `0.058694`, max TV `0.691` | `runs/gemma/cmp-b-vs-c.perm-35999.json` |
| Plain-versus-plain across the LoRA-slot flag | FAIL, 32 of 36 red, max TV `0.930` | `runs/gemma/cmp-a-vs-b.perm-35999.json` |
| Plain-versus-Uno, chunk 32 convention | FAIL, 23 of 36 red, max TV `0.902` | `runs/gemma/candidate-r2/cmp-chunk32-plain-vs-uno.perm-35999.json` |
| Uno-versus-Uno, chunk 32 convention | PASS, 0 of 36 red | `runs/gemma/candidate-r2/cmp-chunk32-uno-vs-uno.perm-35999.json` |
| Plain chunk 1 versus plain chunk 32, one server | FAIL, 35 of 36 red, max TV `1.000` | `runs/gemma/candidate-r2/cmp-plain-chunk1-vs-chunk32.perm-35999.json` |


The round-1 reading leaned on one signature: on the prose prefix at chunk 1 the
Uno arm returned token `954` for all 256 draws where plain spread over four, and
a point mass on the first draft-and-verify cycle was read as evidence against an
exact verification path. The diagnosis's census finds the same one-token outcome
in a plain arm (216 of 256 draws on one token) and a 125/124 argmax coin flip
across the rows of a single prompt, so that signature belongs to the near-tie
prefix rather than to the arm. In round 2 the same cell shows no point mass at
all: the candidate's prose smoke test reports TV `0.246` over 10 tokens, and the
Uno arm's own cross-session pair reports TV `0.797` at that position on its way
to 37 of 44 red.

#### Engagement, refusals and capacity

Candidate startup records the Uno launch line, `UNO_GEMMA_SPLITKV engaged
width=2 head_size=512 q_heads=16 kv_heads=2 segments=16`, the 16-cell capture
list, and speculation counters of 2,656 drafts / 10,624 draft tokens / 4,289
accepted tokens on the chunk-1 pass rising to 8,173 / 32,692 / 14,934 across the
whole candidate run, so the drafted path is engaged rather than inferred. The
floor-matched run's two Uno sessions repeat that at chunk 1: `Uno launch
(profile=gemma4 ...)` with the split-KV engagement across widths 2, 4 and 5 for
both head-size families, and per-prefix draft acceptance of 0.6852 / 0.3678 /
0.3201 and 0.7026 / 0.3351 / 0.2699. Every executed command, container identity
and pass timestamp is in `evidence/pass-timeline.txt` and
`evidence/session-ledger.txt`.

The vision refusal is a config-time admission check: the same profile with
`--language-model-only` removed exits 1 and logs `Uno requires a language-only
model` once. The draft MoE top-k variant boots with `UNO_DRAFT_MOE_TOPK=4` on a
reduced capture list, serves one request (HTTP 200), serves four sequential
requests, and refuses the concurrent four-request batch — 16 draft rows, above
the reduced list — with HTTP 500 and one named error that carries the dispatch
key, the environment variable and the variant together:
`num_reqs=4 num_tokens=16 effective_loras=2`, `UNO_DRAFT_MOE_TOPK=4`,
`gemma4-sm86-marlin-topk4`.

Both Gemma arms compile the same two `kernel_unified_attention` shapes during
serving (head size 256 with a 1024 sliding window, and head size 512 without
one), so the profile does not reach zero in-serving compilations; this is a
warm-up coverage gap of the profile, not a Uno-specific cost, and it is
identical in the plain arm. Residency was sampled by host `nvidia-smi` after
eight real served updates and is flat at 19,674 MiB of 24,576 MiB. Allocated,
reserved, peak and retained-tensor claims are not supported: this build does not
expose them on `/metrics`, and no leak conclusion is drawn.

#### Live API convention

The checker's fixed order is health, models, greedy, sampled, streaming, then
two shared-prefix requests, then the C=8 and C=32 batches. Health, greedy,
sampled and streaming precede the shared-prefix pair, so the convention is
`cold_validation_only` and the shared-prefix cell is not a comparison cell. C=8
completes 8 of 8 and C=32 completes 32 of 32; with `max_num_seqs=4` the C=32
batch queues behind the admission limit and is a functional capacity check, not
a claim that 32 sequences ran concurrently.

### Fixture scope, inversion and known-red

Source fixture evidence remains **CPU_OBSERVED/CUDA_UNVERIFIED** until a device
receipt proves the device path; the device receipts for this release are the
served profiles above. The Gemma profile's draft-row bound is `max_num_seqs=4`
times `K=4` = 16 draft rows, which the sixteen capture cells cover, and the
distributional result recorded here is a served result, not a fixture result.

The manifest guard is inverted with one mutation: `release/series.json`'s
`base_commit` reverted to zeroes makes
`tests.test_release.ReleaseTests.test_frozen_manifest` exit 1 with `Manifest
base does not match the supported upstream commit`, not with a missing symbol.
The green run beside it is 20 tests, exit 0. The chunk contract is inverted the
same way, one field per run: a pass re-declared at chunk 32 against a chunk-1
pair, and a floor summary re-declared at chunk 32 against a chunk-1 pair, each
exit 2 with the named refusal (`sampling chunk differs: 1 vs 32`, `floor chunk
differs: 32 vs 1`) rather than a missing symbol. The known-red arm for the Qwen
runner replaces only the pinned adapter revision with forty zeroes under
`HF_HUB_OFFLINE=1` and exits 1.

### Reporting scope

No performance cell is measured by this release run, and no ratio is reported
from it. The 1.16x Gemma figure carried in the release notes and the changelog
is the port's own evaluation record (matched plain-versus-Uno greedy decode,
five 384-token requests, sum over sum), not a measurement this release's gate
run produced, and it is not losslessness evidence. This run reports no repeat
medians, ranges or overlaps.

## v0.2.0 release status: READY

The v0.2.0 AMD64 candidate was built from Model Runner V2 head
`5da193919b44335ddf14eac193dfc9e8d5e59df5` on vLLM
`b87339888d29329c42c42573e34cc2beebdcc48b`, then validated on the
`inference-3090` RTX 3090 Docker host. The exact local image is
`sha256:98034bbbf7042147838932d64e0ff7a8668bc91d1d203d4a76708d14231c1037`
(9,956,886,869 bytes). Its saved relaunch archive is
`/root/vllm-uno-0.2.0-relaunch.tar`, 9,956,934,144 bytes, SHA-256
`3fca059e167f3b8ce08b74a1d4baac7785c9b0ecfe31a512f04a0293272e74bc`.

The committed [v0.2.0 validation receipt](../evidence/release-0.2.0/validation.json)
names every public result, input, and log path. Full raw host receipts are
preserved under `/root/vllm-uno-0.2.0/evidence/relaunch-r2/`; the public record
contains their compact, auditable summary. The initial relaunch preflight that
noticed a harness cache-path typo stopped before starting a container; it is
preserved separately and is not part of this successful run.

| Check | Result | Receipt |
|---|---|---|
| Digest-pinned AMD64 build | PASS | `build.log`, `image.inspect.json` |
| Package structure check | PASS | `release/.work/local-gates-r3/package-check.log` |
| `tests/test_release.py` | PASS, 15 tests | `release/.work/local-gates-r3/test-release-green.log` |
| Reverted manifest base | expected-red, exit 1 | `release/.work/local-gates-r3/test-release-inverted-manifest-only.log` |
| Plain sampled reference | PASS, n=256 | `reference/reference.json` |
| Candidate functional greedy | PASS, 4 prompts × 256 tokens | `candidate/verdict.json` |
| G2v2 sampled chunk-1 | PASS, n=256, 32 tests, 5,000 permutations | `candidate/sampled-compare.json` |
| G2v2 mixed chunk-8 | PASS, n=256, 32 tests, 5,000 permutations | `candidate/mixed-compare.json` |
| Live HTTP, streaming, prefix, C=8, C=32 | PASS | `live-base.json` |
| Candidate in-serving JIT | PASS, 0 warnings | `compiles-in-serving.txt` |
| Wrong adapter revision | expected-red, inner exit 1 | `known-red-result.txt` |

## Exact served variant

The validated image uses the digest-pinned AMD64 CI base:

```text
public.ecr.aws/q9t5s3a7/vllm-ci-postmerge-repo:b87339888d29329c42c42573e34cc2beebdcc48b@sha256:e3ab6a1f24248420800ceeede9f14e605e72f4e2ebd48a91821db8c07708e0f5
```

The server ran Qwen3-8B BF16 with the pinned Uno adapter, FlashAttention 2,
Model Runner V2, asynchronous scheduling, prefix caching, K=8,
`max_num_seqs=16`, `max_num_batched_tokens=2048`, and
`kv_cache_memory_bytes=2147483648`. Its CUDA graph capture sizes were exactly
`[1,2,4,8,16,32,64,128,144]`; GPU-memory utilization otherwise used vLLM's
default. The candidate startup log records
`SpeculativeConfig(method='uno', ..., num_spec_tokens=8)` and the default
launcher contains only `uno_lora_path`, `uno_mask_token_id`, and
`uno_noise_seed` as Uno-specific fields.

The arithmetic bound is `16 * 8 = 128` draft rows. The final 144 capture cell
therefore covers the largest served draft-row shape. C=8 is below the active
admission limit and C=32 queues behind it; both are functional capacity checks,
not a claim that 32 sequences are simultaneously active.

## Lossless and engagement gates

The plain reference and candidate used the same image, model revision, BF16
precision, attention backend, prefix caching, asynchronous scheduler,
single API process, 16-sequence admission limit, 2,048-token prefill budget,
explicit 2 GiB KV cache, LoRA capacity, and nine capture sizes. The plain arm
enabled LoRA but did not pass `--speculative-config` or apply the Uno adapter.

The release lossless gate is G2v2, not an inferred token-equality result:

- The chunk-1 sampled comparison passed at n=256 with 32 tests, 5,000
  permutations, Bonferroni alpha `0.0003125`, minimum p `0.014797040591881624`,
  and no p-value or advisory-TV failure.
- The mixed chunk-8 comparison passed at n=256 with 32 tests, 5,000
  permutations, minimum p `0.002599480103979204`, and no p-value or
  advisory-TV failure.
- The candidate's chunk-1 speculation counters advanced by 768 drafts, 6,144
  draft tokens, and 2,747 accepted tokens. Across both gates they advanced by
  2,738 drafts, 21,904 draft tokens, and 8,092 accepted tokens.

The candidate startup log proves feature engagement before traffic:

```text
Uno draft kernels warmed: 16 prepare request shapes (1..16 requests ...)
Uno startup JIT self-check completed without compilation.
```

It later records Uno draft CUDA graph replay. The serving log slice begins only
after that startup snapshot and contains **0** `JIT compilation during
inference` warnings across greedy, sampled, mixed, live HTTP, capacity, and
residency traffic.

### Strict-greedy attribution protocol

Strict greedy comparison is deliberately `NOT_A_RELEASE_GATE` for this RTX
3090 CUDA-graph instrument. This release did not run a plain-versus-plain
double run and did not convert a strict output difference into a correctness
failure. The source attribution is the local MRV2 record
`docs/lanes/bl-mrv2-final-gates.md`: its graph-mode plain self controls showed
the `p2/t31` and `p1/t31` self-flips, while the later separate-engine control
with `VLLM_UNO_GREEDY_CONTROL_ENGINES=2` completed the seven matrix variants
7/7. The warm-up fix in head `5da193...` covers served sampling modes before
the first greedy request. This lane instead ran functional greedy requests and
the G2v2 + mixed lossless gates above, matching the v0.1.0 release convention.

`release/verify.sh --strict-greedy` retains `compare.py` as an explicit
diagnostic for a different instrument; it is not silently waived and is not
invoked by the RTX 3090 release procedure.

## Live API, capacity, and residency receipts

`live-base.json` records PASS for `/health` and `/v1/models`, greedy completion,
sampled completion, SSE streaming, two shared-prefix requests, C=8 (8/8), and
C=32 (32/32). Health, greedy, sampled, and streaming precede the shared-prefix
requests, so its convention is `cold_validation_only` and it is not a
comparison cell. Live traffic advanced 45 drafts, 360 draft tokens, and 70
accepted tokens.

After real server updates 1 through 8, host-side `nvidia-smi` recorded eight
identical 21,504 MiB / 24,576 MiB device-memory samples. That is a device
residency series with the production server loop engaged; there is no training
loop. It is not a leak conclusion: this external API harness did not collect
`torch.cuda.memory_allocated` or `torch.cuda.memory_reserved`, so allocated,
reserved, peak, and retained-tensor claims are intentionally absent.

No performance cells were measured for this release. Consequently it reports
no runner-to-runner ratio, no three-repeat cell, no median/range/overlap, and
no performance pass/fail bar. The gate statistics above describe their own
sample count, permutation budget, and detection rule rather than pretending to
be a throughput comparison.

## Fixture scope, inversion, and caller sweep

Source fixture evidence is **CPU_OBSERVED/CUDA_UNVERIFIED** until a device
receipt proves the device path. The final source fixtures retain the full-chunk
two-iteration bound `min(2048, max(3, 2047 + 2)) = 2048` and the production
K=8 extent `min(2048, max(3, 9 + 2)) = 11` on the engine BF16 path. The
batch-shaped device receipt is the actual 16-request / 128-draft-row profile;
the fixtures are source-contract evidence, not a claim that mocked CPU work was
CUDA validation.

The green package run exercised the production manifest. An isolated copy with
only `release/series.json.base_commit` reverted to zeroes ran
`tests.test_release.ReleaseTests.test_frozen_manifest` from
`tests/test_release.py` and exited 1 because
`Manifest base does not match the supported upstream commit`; it did not fail
for a missing symbol. This is the stated base-guard inversion.

The `stack.py` `BASE` caller sweep covers `apply`, `overlay`, `release/check.py`,
`tests/test_release.py`, and the Dockerfile production caller. A fresh local
exact-base clone applied the patch to tree
`6ceef9dfa043d9a2d3f930522ecc7480105aa5a7`; applying again was a verified
no-op. The image build exercised the production Dockerfile path from the
embedded pinned CI workspace. The variant-wide gate sweep covered the served
BF16/K=8/async/prefix-cached configuration's loader, startup, sampled G2v2,
mixed G2v2, functional greedy, streaming, shared prefix, C=8, queued C=32,
residency, and wrong-adapter mutation.

The known-red uses the same candidate image, default entrypoint, model, adapter
name, GPU, IPC, cache mount, and port mapping as green candidate startup, with
only `UNO_ADAPTER_REVISION` replaced by forty zeroes under `HF_HUB_OFFLINE=1`.
The runner exited 1; its wrapper recorded 0 because observing the expected red
is the passing test outcome.

## Reproduce a matched run

Build the image, start a plain reference with the exact profile but no
`--speculative-config`, then start the default Uno entrypoint on the same GPU
and port after stopping the plain arm. The verifier never manages a server:

```bash
bash release/verify.sh reference \
  --url http://127.0.0.1:8000 --model uno-qwen3-8b \
  --out runs/reference --provenance reference-launch.txt \
  --n 256 --max-tokens 16 --greedy-tokens 256

bash release/verify.sh candidate \
  --url http://127.0.0.1:8000 --model uno-qwen3-8b \
  --reference runs/reference --out runs/candidate \
  --provenance candidate-launch.txt \
  --n 256 --max-tokens 16 --greedy-tokens 256
```

Use `--strict-greedy` only for its documented diagnostic context. Preserve the
plain and candidate launch records, verifier output, server logs, and the
post-startup serving-log slice for any retest.
