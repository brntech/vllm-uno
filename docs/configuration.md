# Configuration

Uno for vLLM v0.4.1 supports two reproducible single-GPU serving profiles on
Linux AMD64: Qwen3-8B BF16 at `K=8` and Gemma 4 26B A4B AWQ at `K=4`. Both
package the Model Runner V2 implementation: the Uno draft shares the target
model and KV cache, while the adapter is active only on noisy draft rows.

## Pinned inputs

| Input | Release value |
|---|---|
| Upstream vLLM base commit | `00972dfd72988942138a7a6089eaee08580210b8` |
| Code base (v0.2.0 content) | `3ad49350281a6b73de58449aadb293a8b398fb5d` |
| Release head (Gemma 4 layer) | `cf87916880b051e8782521dfe2afa12e0627e172` |
| Reconstructed release tree | `4aa655488f2c8d86fcc3692b037e03a991dcc9ba` (v0.3.0's was `0149f03eb8287bdfdcc916752b3851405695d350`) |
| Base image | `public.ecr.aws/q9t5s3a7/vllm-ci-postmerge-repo:00972dfd72988942138a7a6089eaee08580210b8@sha256:d55cb6858435cda5ab080987213b4a6b6bfce14ca9e0ffa2ecfab2b222818497` |
| Base vLLM version | `0.29.1rc1.dev99+g00972dfd7` |
| Patch series | `0001-uno-mrv2-base.patch`, `0002-uno-gemma4.patch`, then `0003-uno-hybrid-kv-draft-vocab.patch` |
| Distribution platform | `linux/amd64` only |

The per-commit CI image is AMD64-only. An ARM64 image follows when vLLM tags a
release image containing the pinned base.

`release/stack.py audit` validates the patch series. `release/apply.sh` applies
it to a clean exact-base checkout; a second invocation verifies the
already-applied tree without creating a source commit. The first patch
reconstructs the v0.2.0 release tree (`6ceef9dfa043d9a2d3f930522ecc7480105aa5a7`)
on the newer upstream commit; the second carries the Gemma 4 port, and every
file in it is Python or Markdown. The third (v0.4.0) changes three Python files:
`vllm/v1/worker/gpu/spec_decode/uno.py` (hybrid-KV drafting and the draft
vocabulary), `vllm/lora/layers/base_linear.py` (the inactive-adapter bypass) and
`vllm/v1/core/sched/scheduler.py` (no placeholder-draft padding for Uno), plus
their tests.

## Profiles

`UNO_PROFILE` selects the profile; the default is `qwen3`.

### `qwen3` - Qwen3-8B, the PR's measured nine-cell shape

| Setting | Default |
|---|---|
| Speculative method | `uno` |
| Candidate width | `num_speculative_tokens=8` |
| Uno adapter path | resolved pinned adapter `adapter/` directory |
| Uno mask-token bound | `uno_mask_token_id=151669` |
| Uno noise seed | `uno_noise_seed=0` (the `uno_noise_low` default of 1 applies) |
| Model and KV precision | BF16 |
| Attention backend | `FLASH_ATTN` with FlashAttention 2 override |
| Model runner | Model Runner V2, forced by `VLLM_USE_V2_MODEL_RUNNER=1` |
| Scheduling | asynchronous, required by Uno |
| Prefix caching | enabled |
| API processes | 1 |
| Maximum model length | 4,096 tokens |
| Maximum active sequences | 16 |
| Maximum batched tokens | 2,048 |
| KV cache | explicitly 2 GiB (`2147483648` bytes) |
| GPU memory utilization | vLLM default; no explicit override |
| CUDA graph capture sizes | `[1,2,4,8,16,32,64,128,144]` |
| LoRA capacity | rank 128, 2 slots |
| Native LoRA overlap | `VLLM_LORA_ENABLE_DUAL_STREAM=1` |
| Generation defaults | `--generation-config vllm` |

The speculative configuration contains exactly the MRV2 Uno fields
`uno_lora_path`, `uno_mask_token_id` and `uno_noise_seed`, alongside
`method=uno` and `num_speculative_tokens=8`. The Model Runner V1 graph,
replay, overlap, and fold fields are absent.

The capacity arithmetic is bounded independently on every relevant axis:
`max_num_seqs=16`, `K=8`, and `max_model_len=4096`. Thus the largest served
draft row shape is `16 * 8 = 128`, covered by the 128 capture cell; 144 remains
the final measured nine-cell capture. A C=32 functional request test is still
valid: 32 client requests queue behind the 16-request active admission limit.
It is not a 32-active-sequence profile or a performance cell.

### `gemma4` - Gemma 4 26B A4B AWQ

| Setting | Default |
|---|---|
| Speculative method | `uno` |
| Candidate width | `num_speculative_tokens=4` (`UNO_K`; K + `UNO_PLOOKUP_L` with prompt lookup) |
| Uno adapter path | the local adapter directory passed on the command line |
| Uno noise range | `uno_noise_low=0`, `uno_mask_token_id=262144` |
| Uno noise seed | `uno_noise_seed=29` |
| Model precision | BF16 activations over AWQ 4-bit weights |
| Attention backend | `TRITON_ATTN` (Gemma 4's head sizes force it for the Uno path) |
| Multimodal scope | `--language-model-only`; a vision request is refused by name |
| KV cache manager | vLLM's hybrid manager (five sliding-window groups and one full-attention group); the drafter writes its rows' slots into every group |
| Scheduling | asynchronous; prefix caching and vLLM V1 chunked prefill |
| Server seed | 29 |
| Maximum model length | 32,768 tokens (`UNO_MAX_MODEL_LEN` overrides) |
| Maximum active sequences | 8 |
| Maximum batched tokens | 2,048 |
| GPU memory utilization | `0.90` (79,022 tokens of KV at 32k on a 24 GB RTX 3090; 74,161 with the v0.4.3 recommended settings) |
| CUDA graph capture sizes | `[1,2,3,4,5,6,7,8,9,10,12,14,16,20,24,28,32,40]` |
| LoRA capacity | rank 16, 2 slots (the Uno path reserves one for its shared adapter), target modules `qkv_proj o_proj gate_up_proj down_proj` |
| Split-KV draft attention | on by default since v0.4.0 (`UNO_GEMMA_SPLITKV=0` turns it off); every release measurement used it, and without it draft attention over a long cache dominates (28k-token prompts on early v0.4.0 kit images, one server each: 13.6 vs 7.5 ms per token; `evidence/release-0.4.0/kit-*.jsonl`) |
| Tuned LoRA kernel configs | on by default since v0.4.1: `VLLM_TUNED_CONFIG_FOLDER=/opt/uno-kit/release/lora-configs` holds Triton configs for the adapter's shrink and expand kernels tuned on the RTX 3090 for the draft pass's shapes (about 1 % less time per token there; the output is the model's own); vLLM matches files by GPU name, so other GPUs keep the defaults; an empty value turns them off |
| Draft vocabulary | on by default since v0.4.0: `UNO_DRAFT_VOCAB=/opt/uno-kit/release/gemma4-draft-vocab-65536.json` (65,536 Gemma 4 token ids ranked on open data); an empty value restores the full-vocabulary draft head; verification always scores the full vocabulary |
| Prompt lookup | off by default; since v0.4.3 `UNO_PLOOKUP_L=L` (1 to 8) verifies L tokens copied from the request after the K Uno drafts, and `UNO_PLOOKUP_MAX_CTX=N` limits it to requests whose context is at most N tokens |
| Draft MoE top-k | off by default; `UNO_DRAFT_MOE_TOPK=4` opts in (SM86 only); since v0.4.3 a launch that would leave a reachable draft batch without a captured graph is refused at startup |
| Generation defaults | `--generation-config vllm` |

The Gemma capacity bound is `max_num_seqs=8` times `K=4`, so the largest served
draft row shape is 32 rows; every multiple of four up to 32 is a capture cell,
and 40 covers the largest verify batch (eight requests times `K+1`).

These engine-side switches are read from the process environment and are not
CLI flags:

| Variable | Effect |
|---|---|
| `UNO_GEMMA_SPLITKV` | `1` (gemma4 default) segments the draft attention over the KV axis and logs the engaged `width`, `head_size`, `q_heads`, `kv_heads` and `segments` per head family; `0` turns it off |
| `VLLM_TUNED_CONFIG_FOLDER` | vLLM's folder of tuned LoRA kernel configs (`<GPU name>_SHRINK.json`, `<GPU name>_EXPAND_FALSE.json`); gemma4 default `/opt/uno-kit/release/lora-configs` (RTX 3090 files); empty turns them off |
| `UNO_DRAFT_VOCAB` | Path to a JSON file `{"token_ids": [...]}`: the draft head scores only those ids (gemma4 default: the shipped 64k list); empty or unset is the full-vocabulary draft head |
| `UNO_DRAFT_MOE_TOPK=4` | Captures the draft MoE routers at top-4 while the verifier keeps the configured top-8; the 4-expert draft runs only inside captured draft graphs |
| `UNO_PLOOKUP_L` | Since v0.4.3: lookup tokens verified after the K Uno drafts, 1 to 8; unset or `0` is off (the v0.4.2 path). `release/serve.sh` reads the same variable and sets `num_speculative_tokens = K + L`; a value outside 0..8 stops the launcher |
| `UNO_PLOOKUP_MAX_CTX` | Since v0.4.3: a request whose context (prompt plus output) is above this many tokens verifies only the K Uno drafts; unset is no gate; must be a positive integer. The target also captures full CUDA graphs at the gated width |

`UNO_DRAFT_MOE_TOPK=4` requires an SM86 device and tensor parallelism 1. Since
v0.4.3 the server checks it at startup. It refuses to start when it will capture
no draft graph (`--enforce-eager`, cudagraph mode `NONE` or `PIECEWISE`,
attention without uniform-batch graphs, capture sizes too small for one
request's drafts, or a full-only mode in which the target captures nothing), and
when a draft batch it can be asked for (1 to `max_num_seqs` requests x `UNO_K`
rows) has no captured graph, naming the missing batch sizes. Under the profile
defaults (8 requests, capture sizes up to 40) `UNO_K` of 5 or less starts;
`UNO_K=6` or more, or `--max-num-seqs` above 8 at `UNO_K=5`, is refused. Before
v0.4.3 such a server started and stopped at its first uncovered batch.

### Recommended Gemma 4 settings (v0.4.3)

```bash
-e UNO_PROFILE=gemma4 -e UNO_K=5 -e UNO_PLOOKUP_L=2 -e UNO_PLOOKUP_MAX_CTX=4096 -e UNO_DRAFT_MOE_TOPK=4
```

Five Uno drafts, two lookup tokens after them for requests up to 4,096 tokens of
context, and the draft pass routed through 4 of the 8 experts (verification
keeps all 8). These are the settings measured for v0.4.3
([RELEASE-NOTES-0.4.3.md](../RELEASE-NOTES-0.4.3.md), `evidence/release-0.4.3/`).
On a GPU other than compute capability 8.6, leave out `UNO_DRAFT_MOE_TOPK=4`.
All four are off by default; without them the server runs the v0.4.2 path.

## Build settings

Build AMD64 only:

```bash
PLATFORM=linux/amd64 bash release/build.sh vllm-uno:0.4.1
```

That builds the v0.4.1 base. The v0.4.3 image is the published v0.4.1 image
(pinned by digest) plus the overlay in `patch/0004-v0.4.3/`, built from the
repository root:

```bash
docker build -f patch/0004-v0.4.3/Dockerfile -t vllm-uno:0.4.3 .
```

`BASE_IMAGE` must retain the shown immutable digest and commit-matched
dependency stack. The release Dockerfile clones the embedded CI workspace and
checks out the exact base locally; it preserves compiled vLLM/CUDA libraries
while overlaying verified Python source.

## Runtime settings

`release/serve.sh` accepts optional model and adapter arguments followed by
`--` and additional vLLM arguments:

```bash
UNO_PROFILE=gemma4 bash release/serve.sh [MODEL [LOCAL_ADAPTER]] [-- VLLM_ARGUMENTS...]
```

| Variable | Default | Purpose |
|---|---|---|
| `UNO_PROFILE` | `qwen3` | Profile selector: `qwen3` or `gemma4` |
| `UNO_K` | `8` (`qwen3`), `4` (`gemma4`) | Uno draft count; must be positive (gemma4 recommended: `5`) |
| `UNO_PLOOKUP_L` | unset (off) | Prompt-lookup tokens after the Uno drafts, 0 to 8 (gemma4 recommended: `2`) |
| `UNO_PLOOKUP_MAX_CTX` | unset (no gate) | Context length above which a request verifies no lookup tokens (gemma4 recommended: `4096`) |
| `UNO_MASK_TOKEN_ID` | `151669` (`qwen3`), `262144` (`gemma4`) | Exclusive Uno noise-range upper bound; must be greater than 1 |
| `UNO_NOISE_LOW` | unset (`qwen3`, field default 1), `0` (`gemma4`) | Inclusive Uno noise-range lower bound |
| `UNO_NOISE_SEED` | `0` (`qwen3`), `29` (`gemma4`) | Deterministic MRV2 noise-generator seed |
| `MODEL_REVISION` | pinned revision per profile (Qwen3-8B; Gemma 4 AWQ `0ef577a5710035bd2d3a3f27e4f5cb2e86a9a9ba`) | Model revision passed to vLLM |
| `UNO_MAX_MODEL_LEN` | `32768` (`gemma4`) | Gemma 4 context length |
| `UNO_ADAPTER_REVISION` | pinned adapter revision | Adapter snapshot revision |
| `SERVED_MODEL_NAME` | `uno-qwen3-8b` (`qwen3`), `uno-gemma4-26b-a4b` (`gemma4`) | OpenAI API model name |
| `HOST` | `0.0.0.0` | Address inside the container or direct process |
| `PORT` | `8000` | API port inside the container or direct process |
| `PYTHON` | `python3` | Python executable used by helpers |
| `UNO_DRY_RUN` | unset | Set to `1` to print the final command without loading vLLM |

The launcher always sets `VLLM_USE_V2_MODEL_RUNNER=1`,
`VLLM_WORKER_MULTIPROC_METHOD=spawn`, and `VLLM_LORA_ENABLE_DUAL_STREAM=1`.
Both profiles pass async scheduling, request logging, and the JIT monitor.
Changing the model, adapter, K, noise fields, precision, backend, capacity,
scheduler, or compilation profile changes the served variant; rerun all
validation gates for that variant.

## Cache and exposure

Use a named cache to retain the pinned model and adapter:

```bash
-v vllm-uno-hf-cache:/root/.cache/huggingface
```

The gemma4 profile additionally mounts the adapter directory read-only and
passes it as the second argument.

Publish loopback-only unless an authenticated network boundary is intentional:

```bash
-p 127.0.0.1:8000:8000
```

For an offline run, populate the cache first, mount it read-only if desired,
and set `HF_HUB_OFFLINE=1`. Keep snapshot-relative links intact when moving a
cache.
