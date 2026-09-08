# Configuration

`0.1.0` provides one reproducible serving profile for single-GPU Linux AMD64 and ARM64. The defaults target Qwen3-8B in BF16 with IFM's original Uno adapter and keep the base-model verifier free of the draft adapter.

## Pinned inputs

| Input | Release value |
|---|---|
| Upstream vLLM | `e962733e08d10f7ca65dac4df99e116460b8b174` |
| Target model | `Qwen/Qwen3-8B` |
| Model revision | `b968826d9c46dd6066d109eabc6255188de91218` |
| Uno adapter | `s-sahoo/uno-qwen3-8B` |
| Adapter revision | `8819e09ac901e7290d8d89d62c98b9f756c602fe` |
| Adapter subdirectory | `adapter/` |
| Distribution platforms | `linux/amd64`, `linux/arm64` |

`release/stack.py audit` validates the consolidated patch against the pinned upstream commit. `release/apply.sh` can apply it to a clean exact-base checkout, and a second invocation verifies the already-applied tree without creating a source commit.

```bash
bash release/apply.sh release/.work/vllm
bash release/apply.sh release/.work/vllm
```

## Default serving profile

| Setting | Default |
|---|---|
| Speculative method | `uno` |
| Candidate width | `num_speculative_tokens=8` |
| Model and KV precision | BF16 |
| Attention backend | `FLASH_ATTN` |
| Parallelism | tensor parallel size 1; other parallel dimensions at single-device defaults |
| Prefix caching | enabled |
| Scheduling | asynchronous |
| API processes | 2 |
| Maximum model length | 4,096 tokens |
| Maximum active sequences | 32 |
| Maximum batched tokens | 8,192 |
| GPU memory utilization | 0.90 |
| LoRA capacity | rank 128, 2 slots |
| Qwen noise-vocabulary upper bound | `151669` (exclusive) |
| First-draft replay | enabled |
| Private draft graph | enabled |
| LoRA overlap | enabled |
| Fused draft preparation | enabled |
| Seed-row verification reuse | enabled |
| LoRA-B folding | disabled |
| Generation defaults | vLLM defaults via `--generation-config vllm` |

The adapter is active only for Uno's draft noise rows. Passing it as a conventional always-on LoRA does not reproduce this routing.

## Build settings

Build natively on either supported CPU architecture, or set `PLATFORM` explicitly:

```bash
PLATFORM=linux/amd64 bash release/build.sh vllm-uno:0.1.0
```

`BASE_IMAGE` may select a different immutable image only when it contains the pinned upstream commit and a compatible dependency stack. It must use an `@sha256:` digest. The release build overlays Python source and preserves the commit-matched compiled vLLM/CUDA libraries in that image.

```bash
BASE_IMAGE='vllm/vllm-openai@sha256:89dd8f442a3f4c08c6b3cd634c4f735cd709160651c296596673cf974ea6ee39' \
  PLATFORM=linux/amd64 \
  bash release/build.sh vllm-uno:0.1.0
```

## Runtime settings

`release/serve.sh` accepts optional model and adapter arguments, followed by `--` and additional vLLM arguments:

```bash
bash release/serve.sh [MODEL [ADAPTER]] [-- VLLM_ARGUMENTS...]
```

The following environment variables adjust the supported launcher:

| Variable | Default | Purpose |
|---|---|---|
| `UNO_K` | `8` | Speculative candidate width; must be positive |
| `UNO_MASK_TOKEN_ID` | `151669` | Exclusive noise-token upper bound; `auto` omits the Qwen override |
| `MODEL_REVISION` | pinned Qwen revision | Model revision passed to vLLM |
| `UNO_ADAPTER_REVISION` | pinned adapter revision | Adapter snapshot revision |
| `SERVED_MODEL_NAME` | `uno-qwen3-8b` | OpenAI API model name |
| `HOST` | `0.0.0.0` | Address inside the container or direct process |
| `PORT` | `8000` | API port inside the container or direct process |
| `PYTHON` | `python3` | Python executable used by the helpers |
| `UNO_DRY_RUN` | unset | Set to `1` to print the launch command without importing vLLM or downloading weights |

Changing `UNO_K`, the model, adapter, precision, backend, capacity, or scheduler creates a new experimental configuration. Run the validation gates again and preserve the exact launch record.

## NVIDIA architecture notes

The default backend is vLLM `FLASH_ATTN`. On Ampere, explicitly request FlashAttention 2:

```bash
bash release/serve.sh Qwen/Qwen3-8B s-sahoo/uno-qwen3-8B -- \
  --attention-config '{"flash_attn_version":2}'
```

Apply the same override to both the plain reference and Uno candidate when validating Ampere. The RTX 3090 integration run used this setting.

Hopper was used for the research measurements described in the paper. For Blackwell GB10, use the ARM64 image and the same explicit FlashAttention 2 setting above. Each image uses the matching architecture from the pinned upstream base.

## Model compatibility boundary

The supplied profile is for Qwen3-8B and its matching Uno adapter. A different model requires a compatible trained adapter and the correct noise-vocabulary bound. `UNO_MASK_TOKEN_ID=auto` is provided for models whose target vocabulary convention is already implemented, but it is not a compatibility guarantee.

Seed-row reuse in this release is scoped to one-dimensional positions, ordinary text-only causal attention, supported Punica execution, and TP/PP/DP/context parallel dimensions of one. Stateful or hybrid attention, offloading, multimodal models, and multi-GPU execution need separate engineering and validation.

## Cache and network exposure

Use a named volume to avoid downloading the pinned model and adapter for every container:

```bash
-v vllm-uno-hf-cache:/root/.cache/huggingface
```

Publish the service on loopback unless you have added authentication and an intentional network boundary:

```bash
-p 127.0.0.1:8000:8000
```

For an offline run, populate a Hugging Face cache first, mount it into the container, and set `HF_HUB_OFFLINE=1`. Keep snapshot-relative links intact when moving a cache.
