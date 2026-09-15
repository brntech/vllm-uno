# Configuration

Uno for vLLM v0.3.0 supports two reproducible single-GPU serving profiles on
Linux AMD64: Qwen3-8B BF16 at `K=8` and Gemma 4 26B A4B AWQ at `K=4`. Both
package the Model Runner V2 implementation: the Uno draft shares the target
model and KV cache, while the adapter is active only on noisy draft rows.

## Pinned inputs

| Input | Release value |
|---|---|
| Upstream vLLM base commit | `00972dfd72988942138a7a6089eaee08580210b8` |
| Code base (v0.2.0 content) | `3ad49350281a6b73de58449aadb293a8b398fb5d` |
| Release head (Gemma 4 layer) | `cf87916880b051e8782521dfe2afa12e0627e172` |
| Reconstructed release tree | `7e90f900b039c96565100524700b2da8ef6761bd` |
| Base image | `public.ecr.aws/q9t5s3a7/vllm-ci-postmerge-repo:00972dfd72988942138a7a6089eaee08580210b8@sha256:d55cb6858435cda5ab080987213b4a6b6bfce14ca9e0ffa2ecfab2b222818497` |
| Base vLLM version | `0.29.1rc1.dev99+g00972dfd7` |
| Patch series | `0001-uno-mrv2-base.patch`, then `0002-uno-gemma4.patch` |
| Distribution platform | `linux/amd64` only |

The per-commit CI image is AMD64-only. An ARM64 image follows when vLLM tags a
release image containing the pinned base.

`release/stack.py audit` validates the patch series. `release/apply.sh` applies
it to a clean exact-base checkout; a second invocation verifies the
already-applied tree without creating a source commit. The first patch
reconstructs the v0.2.0 release tree (`6ceef9dfa043d9a2d3f930522ecc7480105aa5a7`)
on the newer upstream commit; the second carries the Gemma 4 port, and every
file in it is Python or Markdown.

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
| Candidate width | `num_speculative_tokens=4` |
| Uno adapter path | the local adapter directory passed on the command line |
| Uno noise range | `uno_noise_low=0`, `uno_mask_token_id=262144` |
| Uno noise seed | `uno_noise_seed=29` |
| Model precision | BF16 activations over AWQ 4-bit weights |
| Attention backend | `TRITON_ATTN` (Gemma 4's head sizes force it for the Uno path) |
| Multimodal scope | `--language-model-only`; a vision request is refused by name |
| KV cache manager | `--disable-hybrid-kv-cache-manager` (one KV group per draft layer) |
| Scheduling | asynchronous; prefix caching and vLLM V1 chunked prefill |
| Server seed | 29 |
| Maximum model length | 8,192 tokens |
| Maximum active sequences | 4 |
| Maximum batched tokens | 2,048 |
| GPU memory utilization | `0.85` |
| CUDA graph capture sizes | `[1,2,3,4,5,6,7,8,13,14,15,16]` |
| LoRA capacity | rank 16, 1 slot, target modules `qkv_proj o_proj gate_up_proj down_proj` |
| Generation defaults | `--generation-config vllm` |

The Gemma capacity bound is `max_num_seqs=4` times `K=4`, so the largest served
draft row shape is 16 rows, covered by the 16 capture cell. The intermediate
cells keep the 4-, 8- and 12-row shapes on captured graphs: a 12-row dispatch
pads up to the captured 13-row descriptor, which is why the capture list names
13 through 16 individually. Eight and 32 concurrent client requests are
functional capacity checks that queue behind the four-sequence admission limit.

Two engine-side switches are read from the process environment and are not
CLI flags:

| Variable | Effect |
|---|---|
| `UNO_GEMMA_SPLITKV=1` | Segments the draft attention over the KV axis and logs the engaged `width`, `head_size`, `q_heads`, `kv_heads` and `segments` per head family |
| `UNO_DRAFT_MOE_TOPK=4` | Captures the draft MoE routers at top-4 while the verifier keeps the configured top-8; an uncaptured serving shape is refused by dispatch key, environment variable and variant name |

`UNO_DRAFT_MOE_TOPK=4` requires an SM86 device, tensor parallelism 1, and a
capture list whose captured draft-row counts cover every serving shape the
operator intends to run. It is validated on the capture list
`[1,2,3,4,5,6,7,8]`, which serves one- and two-sequence requests and refuses a
four-sequence (16 draft row) request.

## Build settings

Build AMD64 only:

```bash
PLATFORM=linux/amd64 bash release/build.sh vllm-uno:0.3.0
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
| `UNO_K` | `8` (`qwen3`), `4` (`gemma4`) | Speculative candidate width; must be positive |
| `UNO_MASK_TOKEN_ID` | `151669` (`qwen3`), `262144` (`gemma4`) | Exclusive Uno noise-range upper bound; must be greater than 1 |
| `UNO_NOISE_LOW` | unset (`qwen3`, field default 1), `0` (`gemma4`) | Inclusive Uno noise-range lower bound |
| `UNO_NOISE_SEED` | `0` (`qwen3`), `29` (`gemma4`) | Deterministic MRV2 noise-generator seed |
| `MODEL_REVISION` | pinned Qwen revision | Model revision passed to vLLM |
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
