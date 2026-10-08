# Uno for vLLM

**v0.4.4** makes Uno fast with several requests in flight, turns the recommended Gemma 4 settings on with one switch,
`UNO_RECOMMENDED=1`, and ships a new Gemma 4 adapter whose last training stage drafts as many tokens as those settings
serve. On one RTX 3090, Gemma 4 26B A4B with the recommended settings and the new adapter serves BroadNet's production
traffic at **721 tokens per second with eight requests in flight** (stock DFlash K8 in vLLM 0.30.0: 679)
and 523 with four (stock DFlash K8 in vLLM 0.30.0: 497); one request at a time it decodes **1.69x
faster than plain** and takes 7.6 % less time per token than stock DFlash K8 in vLLM 0.30.0. The output
is the model's own (see [RELEASE-NOTES-0.4.4.md](RELEASE-NOTES-0.4.4.md)). With neither `UNO_RECOMMENDED` nor
`UNO_GEMMA_SPLITKV_MULTI` set, a v0.4.4 server runs the v0.4.3 path; image `ghcr.io/brntech/vllm-uno:0.4.4`. The new
adapter is on the Hub's main branch; the v0.4.3 adapter stays at Hub revision `s5`.

**v0.4.4 on AMD Radeon**: the same release runs on the Radeon AI PRO R9700 (RDNA4, gfx1201) with ROCm, image
`ghcr.io/brntech/vllm-uno:0.4.4-rocm`, built from the stock vLLM v0.30.0 ROCm image with R9700-tuned kernel configs on
by default. On BroadNet's production traffic on one R9700 it serves **310 tokens per second with eight requests
in flight** (stock DFlash K8 in vLLM 0.30.0 ROCm: 298); one request at a time it takes 25.0 % less time
per token than stock DFlash K8 in vLLM 0.30.0 ROCm, and 68.4 % less on 28k-token documents. The
output is the model's own (see
[RELEASE-NOTES-0.4.4-rocm.md](RELEASE-NOTES-0.4.4-rocm.md) and [AMD Radeon (ROCm)](#amd-radeon-rocm) below).

**v0.4.3** adds prompt lookup to the Gemma 4 profile: after Uno's drafts, two more candidates are copied from the
request itself, and the model checks them all in the same pass. With the recommended settings (five Uno drafts, two
lookup tokens, draft passes through 4 of the 8 experts) and a new adapter trained on 38,560 open prompts, on one RTX
3090 Gemma 4 26B A4B decodes BroadNet's production traffic, one request at a time, **1.68x faster than plain** (DFlash
K8 in vLLM 0.30.0: 1.57x); 14k and 28k-token documents decode **1.41x and 1.28x faster than plain**. The output is
the model's own (see [RELEASE-NOTES-0.4.3.md](RELEASE-NOTES-0.4.3.md)). The new settings are off by default, so an
unconfigured v0.4.3 server runs the v0.4.2 path; image `ghcr.io/brntech/vllm-uno:0.4.3`. The v0.4.2 adapter stays at
Hub revision `s3`, the 10k-prompt adapter at `p10k`.

**v0.4.2** ships a new Gemma 4 26B A4B adapter trained on 28,560 open prompts: on one RTX 3090 the Gemma 4 profile
decodes BroadNet's production traffic **1.57x faster than plain**, matching the DFlash drafter (1.56x in the same
session); prompts from 2k to 28k tokens decode 1.31x to 1.49x faster than plain (1.29x to 1.98x faster than DFlash);
the output is the model's own (see [RELEASE-NOTES-0.4.2.md](RELEASE-NOTES-0.4.2.md)). The image is unchanged:
v0.4.2 runs on `ghcr.io/brntech/vllm-uno:0.4.1`. Its adapter is at Hub revision `s3`.

**v0.4.1** ships tuned kernel configs for the adapter's LoRA layers on the RTX 3090: the Gemma 4 profile decodes about
1 % faster (1.53x plain on BroadNet's production traffic in the release session), and the output is the model's own
(see [RELEASE-NOTES-0.4.1.md](RELEASE-NOTES-0.4.1.md)); the rest is unchanged from v0.4.0.

**v0.4.0** makes Uno a long-context speculator on Gemma 4 26B A4B, the model BroadNet serves in production in English
and Arabic. On one RTX 3090 the Gemma 4 profile decodes BroadNet's production traffic **1.52x faster than plain** and
prompts from 2k to 28k tokens **1.26x to 1.47x faster** (1.3x to 1.8x faster than the DFlash drafter there), with the
output distribution of the model on its own; it starts at 32k context with 79k tokens of KV. The drafter now runs
on Gemma 4's hybrid (sliding-window + full-attention) KV layout, scores a 64k-token draft vocabulary while verification
keeps the full one, and the verify pass skips the adapter branch. The kit is v0.3.0 plus one patch on vLLM
`00972dfd72988942138a7a6089eaee08580210b8`; the Qwen3-8B BF16 profile at `K=8` is unchanged (see
[RELEASE-NOTES-0.4.0.md](RELEASE-NOTES-0.4.0.md) and [docs/validation.md](docs/validation.md)).

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
- vLLM's hybrid KV cache manager on (the drafter writes its rows into every
  sliding-window and full-attention group), `--max-model-len 32768`
  (`UNO_MAX_MODEL_LEN` overrides), `--max-num-seqs 8`,
  `--max-num-batched-tokens 2048`, `--gpu-memory-utilization 0.90`, and capture
  sizes covering every draft shape up to 32 rows (`8 sequences * K=4`) and the
  40-row verify batch.
- On by default: split-KV draft attention (`UNO_GEMMA_SPLITKV=0` turns it off)
  and the shipped 64k Gemma 4 draft vocabulary (an empty `UNO_DRAFT_VOCAB=`
  turns it off).
- Off by default, recommended since v0.4.3: prompt lookup after Uno's drafts
  (`UNO_PLOOKUP_L`) with a per-request length gate (`UNO_PLOOKUP_MAX_CTX`),
  `UNO_K=5`, and top-4 draft MoE routing (`UNO_DRAFT_MOE_TOPK=4`, RTX 30-series
  class GPUs only), which runs only inside captured draft graphs and refuses at
  startup any launch that would leave a reachable draft batch uncaptured
  ([docs/configuration.md](docs/configuration.md)).
- Off by default, recommended since v0.4.4: split-KV attention for steps with
  several requests in flight (`UNO_GEMMA_SPLITKV_MULTI=1`, which also adds
  capture sizes 48, 56 and 64). `UNO_RECOMMENDED=1` turns on all the recommended
  settings in one switch; a variable you set yourself wins.

**Status.** Uno is lossless by design: its verifier accepts drafts by rejection sampling against the full model, so the output
distribution is the model's own. In the Uno authors' words, it "accelerates generation without sacrificing the quality
of the underlying AR model" ([Sahoo et al.](https://arxiv.org/abs/2609.04010)). On this card Uno-versus-plain
distances are the same size as plain-versus-plain ones ([docs/validation.md](docs/validation.md)). The v0.3.0 certification and its floor-matched gate
([`gates/lossless_floor.py`](gates/lossless_floor.py)) are historical records for v0.3.0's image.

The digest-pinned per-commit base image is AMD64-only. An ARM64 image follows
when vLLM publishes a release image containing this base.

## Container

After the maintainer publishes the release, pull the AMD64 image:

```bash
docker pull ghcr.io/brntech/vllm-uno:0.4.4
```

Use a Linux AMD64 host with a compatible NVIDIA driver and Docker configured
with the NVIDIA Container Toolkit. Both profiles ran on a 24 GiB RTX 3090.
Allow space for the CUDA image and model cache.

Start the default Qwen3-8B server with a named Hugging Face cache and a
loopback-only API:

```bash
docker run --rm --name vllm-uno --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v vllm-uno-hf-cache:/root/.cache/huggingface \
  ghcr.io/brntech/vllm-uno:0.4.4 \
  Qwen/Qwen3-8B s-sahoo/uno-qwen3-8B
```

For Gemma 4, download the adapter (main is the v0.4.4 adapter, trained on 38,560
open prompts with a last stage at five-row draft blocks; the v0.4.3 adapter is at
revision `s5`, the v0.4.2 adapter at `s3`, the 10k-prompt adapter at `p10k`):

```bash
hf download Broadnet/gemma-4-26B-A4B-uno-adapter --local-dir /path/to/uno-adapter
```

then mount the model cache and the adapter directory and select the gemma4
profile with the recommended settings:

```bash
docker run --rm --name vllm-uno-gemma --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v /path/to/hf-cache:/root/.cache/huggingface \
  -v /path/to/uno-adapter:/adapter:ro \
  -e UNO_PROFILE=gemma4 -e UNO_RECOMMENDED=1 \
  ghcr.io/brntech/vllm-uno:0.4.4 \
  cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit /adapter \
  -- --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4
```

The flags after `--` turn on Gemma 4's tool-call and reasoning parsers, as on the
measured servers. `UNO_RECOMMENDED=1` sets `UNO_K=5`, `UNO_PLOOKUP_L=2`,
`UNO_PLOOKUP_MAX_CTX=4096`, `UNO_DRAFT_MOE_TOPK=4` and
`UNO_GEMMA_SPLITKV_MULTI=1`, and the launcher prints the settings it used.
`UNO_DRAFT_MOE_TOPK=4` is for compute-capability-8.6 GPUs (RTX 30-series class)
only and refuses to start elsewhere; there, add `-e UNO_DRAFT_MOE_TOPK=`.
Without `UNO_RECOMMENDED=1` or the settings it sets, the server runs the v0.4.2
path.

When `/health` is ready, send an OpenAI-compatible request using `uno-qwen3-8b`
for Qwen or `uno-gemma4-26b-a4b` for Gemma:

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"uno-qwen3-8b","messages":[{"role":"user","content":"Explain in one sentence why 17 is prime."}],"temperature":0,"max_tokens":256}'

curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"uno-gemma4-26b-a4b","messages":[{"role":"user","content":"Explain in one sentence why 17 is prime."}],"temperature":0,"max_tokens":256}'
```

`UNO_DRY_RUN=1 bash release/serve.sh` prints the exact default command without
loading vLLM or downloading weights. Configuration overrides change the
validated profile and require a new gate run.

### AMD Radeon (ROCm)

The ROCm image serves the Gemma 4 profile on a Radeon AI PRO R9700 (gfx1201,
32 GB); Uno starts only on gfx1201 in this image. Use a Linux AMD64 host with
the ROCm kernel driver (the image carries ROCm 7.2) and pass the GPU devices:

```bash
docker pull ghcr.io/brntech/vllm-uno:0.4.4-rocm
docker run --rm --name vllm-uno-gemma --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render --security-opt seccomp=unconfined \
  --ipc=host --shm-size 8g -p 127.0.0.1:8000:8000 \
  -v /path/to/hf-cache:/root/.cache/huggingface \
  -v /path/to/uno-adapter:/adapter:ro \
  -e UNO_PROFILE=gemma4 -e UNO_RECOMMENDED=1 \
  ghcr.io/brntech/vllm-uno:0.4.4-rocm \
  cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit /adapter \
  -- --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4
```

On gfx12 the launcher also points `VLLM_TUNED_CONFIG_FOLDER` at the image's
R9700 fused-MoE config, patches larger attention tiles for prefill-shaped
launches into the container's vLLM before the server starts
(`R9700_PREFILL_TILES=0` skips it), and runs the adapter on the main stream
(`VLLM_LORA_ENABLE_DUAL_STREAM=1` restores dual stream). The adapter's LoRA
kernel configs and the dispatch table of the dense 4-bit layers for the R9700
ship inside the image's vLLM and are read automatically. Measured numbers:
[RELEASE-NOTES-0.4.4-rocm.md](RELEASE-NOTES-0.4.4-rocm.md),
[`evidence/release-0.4.4-rocm/`](evidence/release-0.4.4-rocm/).

## Build from source

From the checked-out release tag, audit and build the supported AMD64 image:

```bash
python3 release/stack.py audit
PLATFORM=linux/amd64 bash release/build.sh vllm-uno:0.4.1
```

The build starts from the digest-pinned CI image containing the exact base
repository, checks out the pinned commit locally inside that image, applies the
ordered three-layer patch series, and overlays only the verified Python source. It
preserves the base image's compiled CUDA libraries and does not clone vLLM from
GitHub during the image build. That is the v0.4.1 image, the base of every later
release.

The v0.4.3 image is the published v0.4.1 image (pinned by digest) plus the
prompt-lookup overlay in [`patch/0004-v0.4.3/`](patch/0004-v0.4.3/); build it
from the repository root so the image holds this release's documents:

```bash
docker build -f patch/0004-v0.4.3/Dockerfile -t vllm-uno:0.4.3 .
```

[`evidence/release-0.4.3/image-files.sha256`](evidence/release-0.4.3/image-files.sha256)
lists the hashes of the nine patched runtime files (`/opt/uno-plookup.sha256` in
the image).

The v0.4.4 image is the published v0.4.3 image (pinned by digest) plus the
overlay in [`patch/0005-v0.4.4/`](patch/0005-v0.4.4/) (two attention files and
the launcher); build it from the repository root in the same way:

```bash
docker build -f patch/0005-v0.4.4/Dockerfile -t vllm-uno:0.4.4 .
```

[`evidence/release-0.4.4/image-files.sha256`](evidence/release-0.4.4/image-files.sha256)
lists the hashes of its nine runtime files (`/opt/uno-runtime.sha256` in the
image).

The ROCm image starts from the stock `vllm/vllm-openai-rocm:v0.30.0` image
(pinned by digest) and applies [`patch/rocm-0.4.4/`](patch/rocm-0.4.4/): one
diff that ports the v0.4.4 patches onto vLLM v0.30.0 with the gfx1201
admissions, the R9700 configs as data files, and the launcher changes. The
build refuses any other base and checks every engine file against
`patch/rocm-0.4.4/engine-files.sha256`:

```bash
docker build -f patch/rocm-0.4.4/Dockerfile -t vllm-uno:0.4.4-rocm .
```

[`evidence/release-0.4.4-rocm/image-files.sha256`](evidence/release-0.4.4-rocm/image-files.sha256)
lists the hashes of its runtime files (`/opt/uno-runtime.sha256` in the image).

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

The v0.4.4 checks are in [RELEASE-NOTES-0.4.4.md](RELEASE-NOTES-0.4.4.md) and
[`evidence/release-0.4.4/`](evidence/release-0.4.4/): 174 CPU tests on the
release image (the [`tests/v0.4.4/`](tests/v0.4.4/) suites and the v0.4.3 suites,
run unchanged), and a GPU test that the multi-request attention path equals,
bit for bit, running each request alone. The ROCm image's checks are in
[RELEASE-NOTES-0.4.4-rocm.md](RELEASE-NOTES-0.4.4-rocm.md) and
[`evidence/release-0.4.4-rocm/`](evidence/release-0.4.4-rocm/).
[docs/validation.md](docs/validation.md) records the v0.4.3 checks (96 CPU tests
in [`tests/v0.4.3/`](tests/v0.4.3/) on the release image, with vLLM's real
rejection kernels; the measurement session in
[`evidence/release-0.4.3/`](evidence/release-0.4.3/)), the v0.4.2, v0.4.1 and
v0.4.0 checks, and, below them, the v0.3.0 release record. The Qwen3-8B profile's
launcher settings are unchanged since v0.3.0, but patch 0003 changes shared
code (the LoRA linear layer and the scheduler); on the v0.4.0 image it has passed
the patched tree's Uno unit tests and the eight greedy end-to-end `test_uno.py`
cases, not a rerun of v0.3.0's sampled and mixed-chunk gates. Those gates, and
the Gemma 4 floor-matched gate ([`gates/lossless_floor.py`](gates/lossless_floor.py)),
are v0.3.0 records; v0.3.0's certification did not exercise the prefix-cache
path fixed in v0.4.0.

## Repository map

- [`patch/`](patch/) - ordered patch series against the pinned vLLM commit, the
  v0.4.3 overlay on the v0.4.1 image ([`patch/0004-v0.4.3/`](patch/0004-v0.4.3/))
  the v0.4.4 overlay on the v0.4.3 image ([`patch/0005-v0.4.4/`](patch/0005-v0.4.4/))
  and the ROCm build on the stock vLLM v0.30.0 ROCm image ([`patch/rocm-0.4.4/`](patch/rocm-0.4.4/))
- [`tests/`](tests/) - release tests; [`tests/v0.4.3/`](tests/v0.4.3/) and
  [`tests/v0.4.4/`](tests/v0.4.4/) run on the release image, and on the ROCm
  image through [`tests/rocm-0.4.4/`](tests/rocm-0.4.4/)
- [`evidence/`](evidence/) - per-release measurement and validation evidence
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
