# Uno for vLLM

**v0.1.0**

Uno for vLLM runs [IFM's Uno](https://github.com/ifm-ai/uno) diffusion adapter through vLLM's OpenAI-compatible server. The integration gives Uno a native two-pass speculative path with draft-only LoRA routing, asynchronous scheduling, private draft graph replay, prefix caching, and seed-row reuse.

This repository is an independent community implementation by the BroadNet Research Team. The Uno method and trained adapters are the work of IFM and the [Uno authors](https://arxiv.org/abs/2609.04010).

## Supported profile

The included serving profile uses:

- Linux AMD64 or ARM64 on one NVIDIA GPU
- vLLM pinned to upstream commit `e962733e08d10f7ca65dac4df99e116460b8b174`
- `Qwen/Qwen3-8B` in BF16 with the original [`s-sahoo/uno-qwen3-8B`](https://huggingface.co/s-sahoo/uno-qwen3-8B) adapter
- eight speculative tokens per step (`K=8`)
- vLLM `FLASH_ATTN`; Ampere GPUs use an explicit FlashAttention 2 override
- single-device text generation with the original Qwen3-8B Uno adapter

The images cover x86 NVIDIA systems and ARM64 Blackwell systems such as GB10.
Model configuration and the measured Ampere, Hopper and Blackwell results are
documented in the accompanying paper and validation guide.

## Use the prebuilt container

Pull from GitHub Container Registry. Docker selects the image for your CPU architecture:

```bash
docker pull ghcr.io/brntech/vllm-uno:0.1.0
```

Use a Linux AMD64 or ARM64 host with a compatible NVIDIA driver and Docker configured with
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The Qwen3-8B profile runs on a 24 GB RTX 3090. Allow disk space for the CUDA image and
the downloaded model/adapter cache. First startup needs internet access.

The container is built from the exact `v0.1.0` source tag and pinned vLLM base.
The [release page](https://github.com/brntech/vllm-uno/releases/tag/v0.1.0)
records the published image digest and validation scope. No repository checkout
or local image build is needed to use it.

## Start the server

This command downloads the pinned Qwen model and Uno adapter on first use, retains them in a named Hugging Face cache, and uses the FlashAttention 2 profile validated on RTX 3090 and GB10. The API is published only on loopback.

```bash
docker run --rm --name vllm-uno --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v vllm-uno-hf-cache:/root/.cache/huggingface \
  ghcr.io/brntech/vllm-uno:0.1.0 \
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

## Build from source

A local build remains available for development. Check out the release tag and
run from the repository root with Python 3 and Docker installed:

```bash
python3 release/stack.py audit
bash release/build.sh vllm-uno:0.1.0
```

The build overlays the pinned Uno Python source onto its commit-matched vLLM
runtime and preserves the base image's compiled CUDA libraries. Use your local
image tag in the run commands above when testing a local build.

## Validation

The release includes source and packaging tests, GPU integration checks, and
reference/candidate tools for validating your own configuration. Versioned
[release records](https://github.com/brntech/vllm-uno/releases/tag/v0.1.0)
identify each image, its hardware checks, and the source used to build it.
See [docs/validation.md](docs/validation.md) for the test procedures and results.

## Repository map

- [`patch/`](patch/) — consolidated patch against the pinned vLLM commit
- [`release/stack.py`](release/stack.py) — audit and source assembly
- [`release/apply.sh`](release/apply.sh) — apply or verify the patch in an exact-base checkout
- [`release/build.sh`](release/build.sh) — build a local AMD64 or ARM64 image
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
