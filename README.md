# Uno for vLLM

**v0.3.0** is the Model Runner V2 release kit for Uno on vLLM
`00972dfd72988942138a7a6089eaee08580210b8`. It ships two serving
profiles: Qwen3-8B BF16 at `K=8`, which passes its full v0.2.0 gate set on this
image, and Gemma 4 26B A4B AWQ at `K=4`, which passes every functional,
capacity and refusal gate and is certified as well: greedy output matches plain
decoding exactly, and under sampling Uno is as close to plain as plain is to itself across
sessions on this hardware (see docs/validation.md).

Uno for vLLM runs [IFM's Uno](https://github.com/ifm-ai/uno) diffusion adapter
through vLLM's OpenAI-compatible server. The integration provides a native
two-pass speculative path with draft-only LoRA routing, asynchronous scheduling,
private draft CUDA graph replay, prefix caching, and seed-row reuse.

This repository is an independent community implementation by the BroadNet
Research Team. The Uno method and trained adapters are the work of IFM and the
[Uno authors](https://arxiv.org/abs/2609.04010).

## Supported profiles

Both profiles run on one NVIDIA GPU under Linux AMD64 and share the same
digest-pinned base image; this release does not ship ARM64.

**Qwen3-8B (default, `UNO_PROFILE=qwen3`)** is the PR's measured nine-cell
shape:

- Model Runner V2 source base `3ad49350281a6b73de58449aadb293a8b398fb5d`
  (the v0.2.0 release content) with the Gemma 4 layer
  `cf87916880b051e8782521dfe2afa12e0627e172` on top.
- `Qwen/Qwen3-8B` in BF16 and the pinned
  [`s-sahoo/uno-qwen3-8B`](https://huggingface.co/s-sahoo/uno-qwen3-8B)
  adapter, with `K=8` speculative tokens.
- FlashAttention 2, prefix caching, Model Runner V2, and asynchronous
  scheduling.
- `--max-num-seqs 16`, `--max-num-batched-tokens 2048`, and an explicit
  2 GiB KV cache; the remaining GPU-memory-utilization setting is vLLM's
  default.
- CUDA graph capture sizes `[1,2,4,8,16,32,64,128,144]`.

**Gemma 4 26B A4B (`UNO_PROFILE=gemma4`)** serves a language-only,
sliding-window MoE model with `K=4` speculative tokens:

- `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` (AWQ 4-bit) with a trained Uno
  adapter, `--language-model-only`, `TRITON_ATTN`, and the base image's
  Marlin MoE kernels.
- `--max-num-seqs 4`, `--max-num-batched-tokens 2048`,
  `--max-model-len 8192`, `--gpu-memory-utilization 0.85`, and capture sizes
  `[1,2,3,4,5,6,7,8,13,14,15,16]` covering the 16-draft-row bound
  (`4 sequences * K=4`).
- `UNO_GEMMA_SPLITKV=1` opts into split-KV draft attention;
  `UNO_DRAFT_MOE_TOPK=4` opts into top-4 draft MoE routing under captured
  graphs and refuses any serving shape it did not capture.

**Status:** certified. Greedy decoding, the live API and capacity checks, the vision
refusal and the draft MoE top-k variant all pass on the released image; under sampling Uno is
as close to plain as plain is to itself across sessions on this hardware, read with the
floor-matched gate (docs/validation.md).

The digest-pinned per-commit base image is AMD64-only. An ARM64 image follows
when vLLM publishes a release image containing this base.

## Container

After the maintainer publishes the release, pull the AMD64 image:

```bash
docker pull ghcr.io/brntech/vllm-uno:0.3.0
```

Use a Linux AMD64 host with a compatible NVIDIA driver and Docker configured
with the NVIDIA Container Toolkit. The Qwen3-8B profile ran on a 24 GiB RTX
3090, and so did the Gemma 4 26B A4B profile, which is served and certified.
Allow space for the CUDA image and model cache.

Start the default Qwen3-8B server with a named Hugging Face cache and a
loopback-only API:

```bash
docker run --rm --name vllm-uno --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v vllm-uno-hf-cache:/root/.cache/huggingface \
  ghcr.io/brntech/vllm-uno:0.3.0 \
  Qwen/Qwen3-8B s-sahoo/uno-qwen3-8B
```

For Gemma 4, mount the model cache and the adapter directory and select the
gemma4 profile:

```bash
docker run --rm --name vllm-uno-gemma --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v /path/to/hf-cache:/root/.cache/huggingface \
  -v /path/to/export-step1900:/adapter:ro \
  -e UNO_PROFILE=gemma4 -e UNO_GEMMA_SPLITKV=1 \
  ghcr.io/brntech/vllm-uno:0.3.0 \
  cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit /adapter
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
PLATFORM=linux/amd64 bash release/build.sh vllm-uno:0.3.0
```

The build starts from the digest-pinned CI image containing the exact base
repository, checks out the pinned commit locally inside that image, applies the
ordered two-layer patch series, and overlays only the verified Python source. It
preserves the base image's compiled CUDA libraries and does not clone vLLM from
GitHub during the image build.

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

The Gemma 4 26B A4B port carried by this release is developed in a fork of that
work and is published here; proposing it upstream is a separate step and is not
claimed by this release. Its draft-scope limits are stated in the release notes
and in [docs/validation.md](docs/validation.md).

## Validation

The release record identifies the exact image, source patch series, and the
RTX 3090 checks for both profiles. See [docs/validation.md](docs/validation.md)
and the concise [v0.3.0 lane record](docs/lanes/release-0.3.0.md). The default
verifier uses sampled-distribution and mixed-chunk gates; the Qwen3-8B profile
passes them on this image. The Gemma 4 profile's instrument is the floor-matched
gate ([`gates/lossless_floor.py`](gates/lossless_floor.py)), whose run and
numbers are recorded in [docs/validation.md](docs/validation.md).

## Repository map

- [`patch/`](patch/) - ordered patch series against the pinned vLLM commit
- [`release/stack.py`](release/stack.py) - audit and source assembly
- [`release/apply.sh`](release/apply.sh) - apply or verify the patch in an
  exact-base checkout
- [`release/build.sh`](release/build.sh) - build the supported AMD64 image
- [`release/serve.sh`](release/serve.sh) - launch a validated serving profile
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
