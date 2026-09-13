# Configuration

Uno for vLLM v0.2.0 supports one reproducible serving profile for a single
NVIDIA GPU on Linux AMD64. It packages the Model Runner V2 implementation:
the Uno draft shares the target model and KV cache, while the adapter is active
only on noisy draft rows.

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

The commit-matched CI image is AMD64-only. An ARM64 image follows when vLLM
tags a release image containing the pinned base; `v0.29.1rc0` is 53 commits
past it and has no image. Do not attempt an ARM64 build for this release.

`release/stack.py audit` validates the consolidated patch against the pinned
upstream commit. `release/apply.sh` can apply it to a clean exact-base
checkout, and a second invocation verifies the already-applied tree without
creating a source commit.

```bash
bash release/apply.sh release/.work/vllm
bash release/apply.sh release/.work/vllm
```

## Default serving profile

| Setting | Default |
|---|---|
| Speculative method | `uno` |
| Candidate width | `num_speculative_tokens=8` |
| Uno adapter path | resolved pinned adapter `adapter/` directory |
| Uno noise range | `uno_mask_token_id=151669` |
| Uno deterministic noise seed | `uno_noise_seed=0` |
| Model and KV precision | BF16 |
| Attention backend | `FLASH_ATTN` with FlashAttention 2 override |
| Model runner | Model Runner V2, forced by `VLLM_USE_V2_MODEL_RUNNER=1` |
| Scheduling | asynchronous, required by Uno |
| Prefix caching | enabled |
| API processes | 2 |
| Maximum model length | 4,096 tokens |
| Maximum active sequences | 32 |
| Maximum batched tokens | 8,192 |
| GPU memory utilization | 0.90 |
| LoRA capacity | rank 128, 2 slots |
| Native LoRA overlap | enabled by `VLLM_LORA_ENABLE_DUAL_STREAM=1` |
| Generation defaults | vLLM defaults via `--generation-config vllm` |

The speculative configuration contains exactly the MRV2 Uno fields
`uno_lora_path`, `uno_mask_token_id`, and `uno_noise_seed`, alongside
`method=uno` and `num_speculative_tokens=8`. The old Model Runner V1
graph, replay, overlap, fold, and seed-row configuration fields are not
accepted by this release.

## Build settings

Build for AMD64 only:

```bash
PLATFORM=linux/amd64 bash release/build.sh vllm-uno:0.2.0
```

`BASE_IMAGE` may select a different immutable image only when it contains
the pinned upstream commit and a compatible dependency stack. It must use an
`@sha256:` digest, and `PLATFORM` must remain `linux/amd64`. The release
build overlays Python source and preserves the commit-matched compiled
vLLM/CUDA libraries in that image.

```bash
BASE_IMAGE='public.ecr.aws/q9t5s3a7/vllm-ci-postmerge-repo:b87339888d29329c42c42573e34cc2beebdcc48b@sha256:e3ab6a1f24248420800ceeede9f14e605e72f4e2ebd48a91821db8c07708e0f5' \
  PLATFORM=linux/amd64 \
  bash release/build.sh vllm-uno:0.2.0
```

## Runtime settings

`release/serve.sh` accepts optional model and adapter arguments, followed by
`--` and additional vLLM arguments:

```bash
bash release/serve.sh [MODEL [ADAPTER]] [-- VLLM_ARGUMENTS...]
```

| Variable | Default | Purpose |
|---|---|---|
| `UNO_K` | `8` | Speculative candidate width; must be positive |
| `UNO_MASK_TOKEN_ID` | `151669` | Exclusive upper bound of the Uno noise range; must be greater than 1 |
| `UNO_NOISE_SEED` | `0` | Deterministic MRV2 noise-generator seed |
| `MODEL_REVISION` | pinned Qwen revision | Model revision passed to vLLM |
| `UNO_ADAPTER_REVISION` | pinned adapter revision | Adapter snapshot revision |
| `SERVED_MODEL_NAME` | `uno-qwen3-8b` | OpenAI API model name |
| `HOST` | `0.0.0.0` | Address inside the container or direct process |
| `PORT` | `8000` | API port inside the container or direct process |
| `PYTHON` | `python3` | Python executable used by the helpers |
| `UNO_DRY_RUN` | unset | Set to `1` to print the final launch command without importing vLLM or downloading weights |

The launcher always invokes the engine with
`VLLM_USE_V2_MODEL_RUNNER=1`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`, and
`VLLM_LORA_ENABLE_DUAL_STREAM=1`, plus `--async-scheduling`, request logging,
the JIT monitor, and CUDA-graph capture sizes `[1,2,4,8,16,32,64,128,144]`.
Changing `UNO_K`, the noise fields, model, adapter, precision, backend,
capacity, scheduler, or compilation profile changes the serving profile. Run
the validation gates again and preserve the exact launch record.

## NVIDIA architecture notes

The default backend is vLLM `FLASH_ATTN`, with an explicit FlashAttention 2
override for the validated RTX 3090 profile. This is an AMD64 release; do not
substitute an ARM64 base image or use an ARM64 host for the published image.

## Model compatibility

The supported profile is Qwen3-8B with its matching Uno adapter. A different
model requires a compatible trained adapter and the correct noise-vocabulary
bound. `max_loras` must remain at least 2, and
`max_num_batched_tokens` must be at least `max_num_seqs * UNO_K`.

MRV2 Uno supports its target's tensor-parallel size upstream, but this
release's profile is single-GPU. Request-specific LoRA adapters,
pipeline/data/context parallelism, sliding or hybrid attention, KV
transfer/offloading, KV-sharing fast prefill, and dual batch overlap are
outside the supported configuration.

## Cache and network exposure

Use a named volume to avoid downloading the pinned model and adapter for every
container:

```bash
-v vllm-uno-hf-cache:/root/.cache/huggingface
```

Publish the service on loopback unless you have added authentication and an
intentional network boundary:

```bash
-p 127.0.0.1:8000:8000
```

For an offline run, populate a Hugging Face cache first, mount it into the
container, and add `-e HF_HUB_OFFLINE=1` to `docker run`. Pass the cached
adapter snapshot directory as the launcher's `ADAPTER` argument: use the
directory containing `adapter_config.json` inside the container. This avoids
a repository-tree lookup when resolving an adapter repository ID. Keep
snapshot-relative links intact when moving a cache.
