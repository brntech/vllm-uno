# Validation

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
