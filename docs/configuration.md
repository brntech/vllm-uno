# Configuration

Uno for vLLM v0.2.0 supports one reproducible serving profile for a single
NVIDIA GPU on Linux AMD64. It packages the Model Runner V2 implementation at
`5da193919b44335ddf14eac193dfc9e8d5e59df5`: the Uno draft shares the target
model and KV cache, while the adapter is active only on noisy draft rows.

## Pinned inputs

| Input | Release value |
|---|---|
| Upstream vLLM | `b87339888d29329c42c42573e34cc2beebdcc48b` |
| Base image | `public.ecr.aws/q9t5s3a7/vllm-ci-postmerge-repo:b87339888d29329c42c42573e34cc2beebdcc48b@sha256:e3ab6a1f24248420800ceeede9f14e605e72f4e2ebd48a91821db8c07708e0f5` |
| Base vLLM version | `0.1.1.dev22+gb87339888` |
| Target model | `Qwen/Qwen3-8B` |
| Model revision | `b968826d9c46dd6066d109eabc6255188de91218` |
| Uno adapter | `s-sahoo/uno-qwen3-8B` |
| Adapter revision | `8819e09ac901e7290d8d89d62c98b9f756c602fe` |
| Adapter subdirectory | `adapter/` |
| Distribution platform | `linux/amd64` only |

The per-commit CI image is AMD64-only. An ARM64 image follows when vLLM tags a
release image containing the pinned base; `v0.29.1rc0` is 53 commits past it
and has no image. Do not attempt an ARM64 build for this release.

`release/stack.py audit` validates the consolidated patch. `release/apply.sh`
applies it to a clean exact-base checkout; a second invocation verifies the
already-applied tree without creating a source commit.

## Default serving profile

This is the production shape measured by
`docs/lanes/bl-mrv2-final-ninecell.md`, not the older v0.1.0 profile.

| Setting | Default |
|---|---|
| Speculative method | `uno` |
| Candidate width | `num_speculative_tokens=8` |
| Uno adapter path | resolved pinned adapter `adapter/` directory |
| Uno mask-token bound | `uno_mask_token_id=151669` |
| Uno noise seed | `uno_noise_seed=0` |
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
`uno_lora_path`, `uno_mask_token_id`, and `uno_noise_seed`, alongside
`method=uno` and `num_speculative_tokens=8`. The Model Runner V1 graph,
replay, overlap, and fold fields are absent.

The capacity arithmetic is bounded independently on every relevant axis:
`max_num_seqs=16`, `K=8`, and `max_model_len=4096`. Thus the largest served
draft row shape is `16 * 8 = 128`, covered by the 128 capture cell; 144 remains
the final measured nine-cell capture. A C=32 functional request test is still
valid: 32 client requests queue behind the 16-request active admission limit.
It is not a 32-active-sequence profile or a performance cell.

## Build settings

Build AMD64 only:

```bash
PLATFORM=linux/amd64 bash release/build.sh vllm-uno:0.2.0
```

`BASE_IMAGE` must retain the shown immutable digest and commit-matched
dependency stack. The release Dockerfile clones the embedded CI workspace and
checks out the exact base locally; it preserves compiled vLLM/CUDA libraries
while overlaying verified Python source.

## Runtime settings

`release/serve.sh` accepts optional model and adapter arguments followed by
`--` and additional vLLM arguments:

```bash
bash release/serve.sh [MODEL [ADAPTER]] [-- VLLM_ARGUMENTS...]
```

| Variable | Default | Purpose |
|---|---|---|
| `UNO_K` | `8` | Speculative candidate width; must be positive |
| `UNO_MASK_TOKEN_ID` | `151669` | Exclusive Uno noise-range upper bound; must be greater than 1 |
| `UNO_NOISE_SEED` | `0` | Deterministic MRV2 noise-generator seed |
| `MODEL_REVISION` | pinned Qwen revision | Model revision passed to vLLM |
| `UNO_ADAPTER_REVISION` | pinned adapter revision | Adapter snapshot revision |
| `SERVED_MODEL_NAME` | `uno-qwen3-8b` | OpenAI API model name |
| `HOST` | `0.0.0.0` | Address inside the container or direct process |
| `PORT` | `8000` | API port inside the container or direct process |
| `PYTHON` | `python3` | Python executable used by helpers |
| `UNO_DRY_RUN` | unset | Set to `1` to print the final command without loading vLLM |

The launcher always sets `VLLM_USE_V2_MODEL_RUNNER=1`,
`VLLM_WORKER_MULTIPROC_METHOD=spawn`, and `VLLM_LORA_ENABLE_DUAL_STREAM=1`.
It passes async scheduling, request logging, the JIT monitor, and the nine
capture sizes above. Changing the model, adapter, K, noise fields, precision,
backend, capacity, scheduler, or compilation profile changes the served
variant; rerun all validation gates for that variant.

## Cache and exposure

Use a named cache to retain the pinned model and adapter:

```bash
-v vllm-uno-hf-cache:/root/.cache/huggingface
```

Publish loopback-only unless an authenticated network boundary is intentional:

```bash
-p 127.0.0.1:8000:8000
```

For an offline run, populate the cache first, mount it read-only if desired,
and set `HF_HUB_OFFLINE=1`. Keep snapshot-relative links intact when moving a
cache.
