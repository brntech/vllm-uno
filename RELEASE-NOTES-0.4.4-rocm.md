# Uno for vLLM v0.4.4 on AMD Radeon

Uno for vLLM v0.4.4 now runs on **AMD Radeon AI PRO R9700** (RDNA4, gfx1201) with ROCm, in a container built from the
stock vLLM v0.30.0 ROCm image: `ghcr.io/brntech/vllm-uno:0.4.4-rocm`. On BroadNet's production traffic on one R9700,
Gemma 4 26B A4B with the recommended settings (`UNO_RECOMMENDED=1`) and the v0.4.4 adapter serves 310 tokens per
second with eight requests in flight (stock DFlash K8 in vLLM 0.30.0 ROCm: 298) and 241 with four (stock
DFlash K8 in vLLM 0.30.0 ROCm: 197). One request at a time it takes 25.0 % less time per token than
stock DFlash K8 in vLLM 0.30.0 ROCm, and on 14k and 28k-token documents 60.0 % and
68.4 % less than stock DFlash K8 in vLLM 0.30.0 ROCm. The output is the model's own.

The adapter is the v0.4.4 one, on [Hugging Face](https://huggingface.co/Broadnet/gemma-4-26B-A4B-uno-adapter) (main).
The code that proposes, accepts and samples tokens is the v0.4.4 release's; what the ROCm image adds is below.

## What is in the image

- **Uno v0.4.4 on vLLM v0.30.0.** The v0.4.4 patches (0001 to 0005 in this repository) are ported onto vLLM v0.30.0,
  the version vLLM publishes a ROCm image for, as one diff (`patch/rocm-0.4.4/vllm-v0.30.0-rocm.diff`); Python only, no
  compiled library changes. The build refuses any other base: every file the diff changes must have v0.30.0's bytes.
- **gfx1201 admitted.** Uno, its 4-expert drafts (`UNO_DRAFT_MOE_TOPK=4`) and the AWQ MoE path check the GPU before
  they start. Each check keeps its CUDA condition first; on ROCm, only gfx1201 (Radeon AI PRO R9700) is admitted, and
  every other ROCm GPU keeps the refusal. On ROCm the AWQ experts run vLLM's Triton kernel, so the 4-expert drafts use
  it there.
- **Tuned for the R9700, on by default.** A fused-MoE config for Gemma 4 26B A4B's AWQ experts (the launcher points
  `VLLM_TUNED_CONFIG_FOLDER` at it on gfx12), a dispatch table for the dense 4-bit layers (which kernel and tile serve
  each row count), the adapter's LoRA kernel configs (read under the GPU name ROCm reports), and larger attention tiles
  for prefill-shaped launches (patched into the container's vLLM at start; decode and verify launches keep vLLM's
  tiles). On gfx12 the adapter also runs on the main stream: inside HIP graphs the dual-stream LoRA path leaves the GPU
  idle at every cross-stream fork and join.
- **The same switch.** `UNO_RECOMMENDED=1` sets `UNO_K=5`, `UNO_PLOOKUP_L=2`, `UNO_PLOOKUP_MAX_CTX=4096`,
  `UNO_DRAFT_MOE_TOPK=4` and `UNO_GEMMA_SPLITKV_MULTI=1`, as in the CUDA release. `R9700_PREFILL_TILES=0` skips the tile
  patch; `VLLM_LORA_ENABLE_DUAL_STREAM=1` restores dual stream; `VLLM_TUNED_CONFIG_FOLDER=` (empty) drops the MoE config.

## Measured

One Radeon AI PRO R9700 (32 GB), ROCm 7.2, Gemma 4 26B A4B AWQ 4-bit, the v0.4.4 adapter. v0.4.4 = the
`0.4.4-rocm` container with `UNO_RECOMMENDED=1`; DFlash K8 and plain decoding on the stock `vllm/vllm-openai-rocm:v0.30.0`
image as published, with nothing added; the same serve flags for all. One session, three rounds per arm with the arm
order rotated, every server started fresh for its workload and sent two warmup requests (the first two production
requests) before the timed ones. Production traffic = 72 real requests, completion tokens only. Documents = open-data
texts, 4 sets x 2 tasks, up to 384 output tokens, one at a time, on the one-at-a-time servers after the production
requests. Production figures are the median of the three rounds, with the range of the three rounds in brackets for
v0.4.4's and DFlash K8's tokens per second; document figures are the median of all 24 requests of the three rounds
pooled; speed-ups over plain are ratios of these medians. Per-round numbers and the tables derived from them are in the
release's evidence files; v0.4.4's figures here are rounded so that none is better than measured.

| production traffic | Uno v0.4.4 | DFlash K8, vLLM 0.30.0 ROCm | plain |
| --- | ---: | ---: | ---: |
| eight at a time: tokens per second, whole batch | **310** (307-310) | 298 (297-299) | 202 |
| eight at a time: speed-up over plain | **1.53x** | 1.48x | 1.00x |
| four at a time: tokens per second, whole batch | **241** (237-241) | 197 (197-197) | 167 |
| four at a time: speed-up over plain | **1.44x** | — | 1.00x |
| one at a time: ms per token, rounds 1 / 2 / 3 | **8.920 / 8.995 / 8.951** | 11.920 / 11.948 / 11.966 | — |

| documents, one at a time: ms per token | Uno v0.4.4 | DFlash K8, vLLM 0.30.0 ROCm |
| --- | ---: | ---: |
| 2k-token prompts | **10.11** | 16.12 |
| 14k-token prompts | **13.52** | 33.84 |
| 28k-token prompts | **17.56** | 55.56 |

- Eight at a time, v0.4.4 took 3.1 %, 4.2 % and 3.7 % less time per token than
  stock DFlash K8 in vLLM 0.30.0 ROCm (rounds 1, 2 and 3); four at a time, 17.0 %, 18.3 % and
  18.4 % less; one at a time, 25.1 %, 24.7 % and 25.1 % less. On 2k-token
  documents 36.0 %, 37.3 % and 37.2 % less, on 14k 59.5 %,
  59.5 % and 60.8 %, on 28k 68.2 %, 68.9 % and
  68.9 %.
- A comparison is called when every round of both arms agrees on the direction by more than the larger round spread.
- KV cache at 32k context: 215,042 tokens on every v0.4.4 server with the recommended settings.

## Speed-up over plain decoding

v0.4.4 decodes 1.40x faster one request at a time than plain decoding in vLLM 0.30.0 ROCm, and 1.25x, 1.23x and
1.18x faster on 2k, 14k and 28k-token documents.

| workload | Uno v0.4.4 | plain decoding, vLLM 0.30.0 ROCm | speed-up over plain decoding in vLLM 0.30.0 ROCm |
| --- | ---: | ---: | ---: |
| one at a time: ms per token | **8.874** | 12.446 | **1.40x** |
| 2k-token documents: ms per token | **10.23** | 12.85 | **1.25x** |
| 14k-token documents: ms per token | **13.62** | 16.87 | **1.23x** |
| 28k-token documents: ms per token | **17.42** | 20.56 | **1.18x** |

These four come from a second session on the same card, image, model, settings and workloads: four rounds per arm with
the arm order balanced, every server's decode speed checked before its timed requests. v0.4.4's own times in it are
within 1.2 % of the session above on every workload. One-at-a-time figures are the median of the four rounds,
document figures the median of all 32 requests pooled; each speed-up is the ratio of the two figures beside it. All four
are called: in every round v0.4.4 takes less time per token than plain decoding in vLLM 0.30.0 ROCm by more than the
larger round spread. Per-round numbers are in `evidence/release-0.4.4-rocm/plain-rerun/`.

## Lossless

The ROCm image runs the v0.4.4 release's proposal, acceptance and sampling code: every candidate still goes through
vLLM's own rejection sampler against the draft distribution it was drawn from, so the output distribution is the
model's. What differs on ROCm is numerics (vLLM's ROCm kernels, the tuned configs, the prefill tiles), which can move
the last bits of a probability the same way a different GPU does, never which distribution is sampled. The v0.4.4 CPU
tests, with the v0.4.3 exactness suites (vLLM's rejection kernels at the served shapes, position by position against
the model's distribution), run on the ROCm image with three test-side lines changed (the image's version string, and a
field vLLM v0.30.0 reads on a test double): 174 CPU tests pass, rc 0 in every group, the same 174 as on the CUDA
release image (`evidence/release-0.4.4-rocm/cpu/`).

## Run

```bash
docker pull ghcr.io/brntech/vllm-uno:0.4.4-rocm
hf download Broadnet/gemma-4-26B-A4B-uno-adapter --local-dir /path/to/adapter
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --security-opt seccomp=unconfined --ipc=host --shm-size 8g -p 127.0.0.1:8000:8000 \
  -v /path/to/hf-cache:/root/.cache/huggingface -v /path/to/adapter:/adapter:ro \
  -e UNO_PROFILE=gemma4 -e UNO_RECOMMENDED=1 \
  ghcr.io/brntech/vllm-uno:0.4.4-rocm cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit /adapter \
  -- --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4
```

The flags after `--` turn on Gemma 4's tool-call and reasoning parsers, as on the measured servers. Uno starts only on
gfx1201 in this image; the CUDA image `ghcr.io/brntech/vllm-uno:0.4.4` is unchanged. Build it yourself from the
repository root with `docker build -f patch/rocm-0.4.4/Dockerfile .`.
