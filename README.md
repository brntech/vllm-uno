# Uno for vLLM

**v0.2.0** is the validated Model Runner V2 release kit for Uno on vLLM
`b87339888d29329c42c42573e34cc2beebdcc48b`. It is ready for the maintainer's
tag, registry push, and release publication steps recorded in this repository.

Uno for vLLM runs [IFM's Uno](https://github.com/ifm-ai/uno) diffusion adapter
through vLLM's OpenAI-compatible server. The integration provides a native
two-pass speculative path with draft-only LoRA routing, asynchronous scheduling,
private draft CUDA graph replay, prefix caching, and seed-row reuse.

This repository is an independent community implementation by the BroadNet
Research Team. The Uno method and trained adapters are the work of IFM and the
[Uno authors](https://arxiv.org/abs/2609.04010).

## Supported profile

The v0.2.0 production shape is the PR's measured nine-cell shape:

- Linux AMD64 on one NVIDIA GPU; this release does not ship ARM64.
- Model Runner V2 source head `5da193919b44335ddf14eac193dfc9e8d5e59df5`
  on the pinned vLLM base `b87339888d29329c42c42573e34cc2beebdcc48b`.
- `Qwen/Qwen3-8B` in BF16 and the pinned
  [`s-sahoo/uno-qwen3-8B`](https://huggingface.co/s-sahoo/uno-qwen3-8B)
  adapter, with `K=8` speculative tokens.
- FlashAttention 2, prefix caching, Model Runner V2, and asynchronous
  scheduling.
- `--max-num-seqs 16`, `--max-num-batched-tokens 2048`, and an explicit
  2 GiB KV cache; the remaining GPU-memory-utilization setting is vLLM's
  default.
- CUDA graph capture sizes `[1,2,4,8,16,32,64,128,144]`.

The digest-pinned per-commit base image is AMD64-only. An ARM64 image follows
when vLLM publishes a release image containing this base; `v0.29.1rc0` is 53
commits past it and has no image.

## Container

After the maintainer publishes the release, pull the AMD64 image:

```bash
docker pull ghcr.io/brntech/vllm-uno:0.2.0
```

Use a Linux AMD64 host with a compatible NVIDIA driver and Docker configured
with the NVIDIA Container Toolkit. The validated Qwen3-8B profile ran on a
24 GiB RTX 3090. Allow space for the CUDA image and model cache.

Start the server with a named Hugging Face cache and a loopback-only API:

```bash
docker run --rm --name vllm-uno --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v vllm-uno-hf-cache:/root/.cache/huggingface \
  ghcr.io/brntech/vllm-uno:0.2.0 \
  Qwen/Qwen3-8B s-sahoo/uno-qwen3-8B
```

When `/health` is ready, send an OpenAI-compatible request:

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"uno-qwen3-8b","messages":[{"role":"user","content":"Explain in one sentence why 17 is prime."}],"temperature":0,"max_tokens":256}'
```

`UNO_DRY_RUN=1 bash release/serve.sh` prints the exact default command without
loading vLLM or downloading weights. Configuration overrides change the
validated profile and require a new gate run.

## Build from source

From the checked-out release tag, audit and build the supported AMD64 image:

```bash
python3 release/stack.py audit
PLATFORM=linux/amd64 bash release/build.sh vllm-uno:0.2.0
```

The build starts from the digest-pinned CI image containing the exact base
repository, checks out the pinned commit locally inside that image, and overlays
only the verified Python source. It preserves the base image's compiled CUDA
libraries and does not clone vLLM from GitHub during the image build.

## Upstream contribution

BroadNet is contributing native Uno support to vLLM through
[PR #55947](https://github.com/vllm-project/vllm/pull/55947), as part of
[RFC #55267](https://github.com/vllm-project/vllm/issues/55267). The PR now
carries the **Model Runner V2** implementation agreed in the RFC, rebased onto
vLLM main at `b87339888d`: a native MRV2 speculator that shares the target
model, attention layers and KV cache with the drafter, applies the adapter only
to noise rows, fused draft-input preparation, native Gumbel sampling and
verification, async scheduling, draft CUDA graphs, tensor parallelism, and
fail-closed validation of unsupported configurations. It replaces the earlier
Model Runner V1 revision of the PR.

The PR revision has been run end-to-end on NVIDIA Ampere (RTX 3090), Hopper
(H100) and Blackwell (GB10 and RTX 5090, including tensor-parallel 2), with a
continuous-batching preemption test that forces a real KV-pool crossing and
checks the resumed request's tokens against its solo run. Against plain vLLM on
the same card and serving shape it measures 2.6x single-stream on an H100, 2.1 to
2.4x on a GB10 and 1.7 to 1.9x on an RTX 3090, and stays ahead under load on the
H100 and GB10; output is lossless and first-token latency is level with plain.
The tables and receipts are in the PR description. The PR is open and not yet
merged.

## Validation

The release record identifies the exact image, source patch, RTX 3090 checks,
and test receipts. See [docs/validation.md](docs/validation.md) and the concise
[v0.2.0 lane record](docs/lanes/release-0.2.0.md). The default verifier uses
sampled-distribution and mixed-chunk gates.

## Repository map

- [`patch/`](patch/) - consolidated patch against the pinned vLLM commit
- [`release/stack.py`](release/stack.py) - audit and source assembly
- [`release/apply.sh`](release/apply.sh) - apply or verify the patch in an
  exact-base checkout
- [`release/build.sh`](release/build.sh) - build the supported AMD64 image
- [`release/serve.sh`](release/serve.sh) - launch the supported profile
- [`release/verify.sh`](release/verify.sh) - capture plain-reference and Uno
  candidate evidence
- [`release/check.py`](release/check.py) - standard-library package checks
- [`release/bundle.py`](release/bundle.py) - create a reproducible source bundle
- [`gates/`](gates/) - fixed correctness gates and prompt data

## Research and attribution

- Paper: [Uno in vLLM: An Independent Implementation and Empirical Serving Study](https://doi.org/10.5281/zenodo.22652610)
- Uno method: [Unlocking Lossless Speedups in LLMs via Discrete Diffusion](https://arxiv.org/abs/2609.04010)
- IFM implementation: [ifm-ai/uno](https://github.com/ifm-ai/uno)
- Original Qwen3-8B adapter: [s-sahoo/uno-qwen3-8B](https://huggingface.co/s-sahoo/uno-qwen3-8B)

BroadNet-authored code and release tools are licensed under Apache-2.0. See
[LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for
terms and upstream attribution. Citation metadata is in [CITATION.cff](CITATION.cff).
