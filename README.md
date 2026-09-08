# Uno for vLLM

**Research pre-release — `0.1.0rc1`**

Uno for vLLM runs [IFM's Uno](https://github.com/ifm-ai/uno) diffusion adapter through vLLM's OpenAI-compatible server. The integration gives Uno a native two-pass speculative path with draft-only LoRA routing, asynchronous scheduling, private draft graph replay, prefix caching, and seed-row reuse.

This repository is an independent community implementation by the BroadNet Research Team. The Uno method and trained adapters are the work of IFM and the [Uno authors](https://arxiv.org/abs/2609.04010).

## Release scope

The `0.1.0rc1` profile is intentionally focused:

- Linux AMD64 on one NVIDIA GPU
- vLLM pinned to upstream commit `e962733e08d10f7ca65dac4df99e116460b8b174`
- `Qwen/Qwen3-8B` in BF16 with the original [`s-sahoo/uno-qwen3-8B`](https://huggingface.co/s-sahoo/uno-qwen3-8B) adapter
- eight speculative tokens per step (`K=8`)
- vLLM `FLASH_ATTN`; Ampere GPUs use an explicit FlashAttention 2 override
- tensor parallelism, pipeline parallelism, stateful or hybrid attention, and non-NVIDIA targets are outside this release profile

Hopper results in the accompanying paper are historical research measurements from a separate evaluation environment. Blackwell/GB10 uses an ARM64 software path and is not bundled in this AMD64 release.

## Build locally

The release is distributed as source. There is no registry image for this pre-release, so build it locally from the repository root.

Prerequisites are Linux AMD64, Python 3, Docker with NVIDIA GPU support, and enough disk space for the image and model cache.

```bash
python3 release/stack.py audit
bash release/build.sh vllm-uno:0.1.0rc1
```

The audit checks the pinned upstream base and the consolidated patch before the build overlays the Python source onto its commit-matched vLLM runtime. The build does not rebuild or replace the base image's compiled CUDA libraries.

## Start the server

The default command downloads the pinned Qwen model and Uno adapter on first use and retains them in a named Hugging Face cache. The API is published only on loopback.

```bash
docker run --rm --name vllm-uno --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v vllm-uno-hf-cache:/root/.cache/huggingface \
  vllm-uno:0.1.0rc1
```

On Ampere, including the RTX 3090, select FlashAttention 2 explicitly:

```bash
docker run --rm --name vllm-uno --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v vllm-uno-hf-cache:/root/.cache/huggingface \
  vllm-uno:0.1.0rc1 \
  Qwen/Qwen3-8B s-sahoo/uno-qwen3-8B -- \
  --attention-config '{"flash_attn_version":2}'
```

Model loading and the first graph capture can take several minutes. When `/health` is ready, send an OpenAI-compatible request:

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"uno-qwen3-8b","messages":[{"role":"user","content":"Explain in one sentence why 17 is prime."}],"temperature":0,"max_tokens":256}'
```

Configuration overrides and a complete profile table are in [docs/configuration.md](docs/configuration.md).

## Validation status

The release line has a concrete hardware record: 368 source-level CPU tests passed; a Linux AMD64 image build preserved all 17 checked compiled libraries; and the corresponding runtime implementation completed a 256-token request on an RTX 3090 while compiled CUDA loading, Uno drafting, private graph replay, and seed-row reuse were observed.

That bounded run establishes hardware integration. It is not recorded as a full sampled-distribution or greedy-equivalence pass. The included gates deliberately keep those stronger claims separate and require a matched plain-vLLM reference. See [docs/validation.md](docs/validation.md) for the evidence boundary and reproducible reference/candidate workflow.

## Repository map

- [`patch/`](patch/) — consolidated patch against the pinned vLLM commit
- [`release/stack.py`](release/stack.py) — audit and source assembly
- [`release/apply.sh`](release/apply.sh) — apply or verify the patch in an exact-base checkout
- [`release/build.sh`](release/build.sh) — build the local AMD64 image
- [`release/serve.sh`](release/serve.sh) — launch the supported profile
- [`release/verify.sh`](release/verify.sh) — capture and compare plain-reference and Uno-candidate evidence
- [`release/check.py`](release/check.py) — standard-library package checks
- [`release/bundle.py`](release/bundle.py) — create a reproducible source bundle
- [`gates/`](gates/) — selected correctness gates and fixed prompt data

## Research and attribution

- Paper: [Uno in vLLM: An Independent Implementation and Empirical Serving Study](https://doi.org/10.5281/zenodo.22652610)
- Uno method: [Unlocking Lossless Speedups in LLMs via Discrete Diffusion](https://arxiv.org/abs/2609.04010)
- IFM implementation: [ifm-ai/uno](https://github.com/ifm-ai/uno)
- Original Qwen3-8B adapter: [s-sahoo/uno-qwen3-8B](https://huggingface.co/s-sahoo/uno-qwen3-8B)

BroadNet-authored code and release tools are licensed under Apache-2.0. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for terms and upstream attribution. Citation metadata is available in [CITATION.cff](CITATION.cff).
