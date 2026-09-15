# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.3.0] - 2026-09-15

### Added

- Gemma 4 26B A4B support on the Model Runner V2 Uno path: MoE drafting, sliding-window attention, and language-only multimodal admission, released as a second validated profile beside Qwen3-8B (`UNO_PROFILE=gemma4`: AWQ 4-bit weights, `TRITON_ATTN`, K=4, `max_num_seqs=4`, `max_model_len=8192`, `gpu_memory_utilization=0.85`, and sixteen base capture cells covering the 16-draft-row bound).
- Split-KV attention opt-in for the Gemma draft path (`UNO_GEMMA_SPLITKV=1`), which segments the draft attention over the KV axis and logs the engaged `width`, `head_size`, `q_heads`, `kv_heads` and `segments` per head family.
- Draft MoE top-k opt-in (`UNO_DRAFT_MOE_TOPK=4`): the drafter's MoE routers are captured and replayed at top-4 while the verifier keeps the configured top-8, with a partial-capture path that restores every router and clears the draft LoRA hook.
- Fail-closed admission for the top-k variant: an uncaptured serving shape is refused by dispatch key, environment variable and variant name instead of silently falling back to eager routing, and the startup receipt names the captured draft-row counts.
- A draft-graph coverage receipt and a dispatch-key diagnostic, so a captured-versus-dispatched `(num_tokens, effective_loras)` disagreement is visible in the log rather than inferred.
- `UNO_PROFILE` in `release/serve.sh`: the Gemma 4 profile is launched from the same pinned tooling as the Qwen3-8B profile, with `UNO_NOISE_LOW`, `UNO_GEMMA_SPLITKV` and `UNO_DRAFT_MOE_TOPK` documented and passed through.

### Changed

- The patch manifest is an ordered two-layer series against the digest-pinned upstream commit `00972dfd72988942138a7a6089eaee08580210b8`: `0001-uno-mrv2-base.patch` (the Model Runner V2 Uno base, tree `6ceef9dfa043d9a2d3f930522ecc7480105aa5a7`, the same tree v0.2.0 shipped) and `0002-uno-gemma4.patch` (the Gemma 4 port, head `cf87916880b051e8782521dfe2afa12e0627e172`). The reconstructed release tree is `7e90f900b039c96565100524700b2da8ef6761bd`.
- The base image is the postmerge CI image for that upstream commit, pinned by amd64 manifest digest `sha256:d55cb6858435cda5ab080987213b4a6b6bfce14ca9e0ffa2ecfab2b222818497`; the release remains Linux AMD64 only and preserves the base image's compiled CUDA libraries.

### Verified

- Gemma 4 26B A4B AWQ with the step-1900 pilot adapter, language-only, K=4, on an RTX 3090 (24 GiB): matched plain-versus-Uno greedy decode over five 384-token requests reports 166.713 versus 143.694 tokens/s (sum over sum), 1.16x for the Uno arm, with the same card, workload, sampling and repeat convention for both arms.
- The sampled-distribution lossless gate passes on the Gemma profile (three pinned prefixes, 36 tests, 35,999 permutations, Bonferroni alpha `2.778e-4`, tightest p `0.0003333`, TV advisory) beside a plain-versus-plain floor measured in the same run.
- The Qwen3-8B BF16 K=8 profile re-passes its v0.2.0 gate set on the v0.3.0 image.

### Scope

- G2 preparation/KV, G3 attention, and broader G4/G7 qualification remain CUDA_UNVERIFIED; the available receipts support only the bounded scenarios exercised.

## [0.2.0] - 2026-09-14

### Changed

- The package is now the Model Runner V2 Uno implementation on vLLM `b87339888d29329c42c42573e34cc2beebdcc48b`, distributed for Linux AMD64 only; an ARM64 image follows when vLLM tags a release image containing this base.
- Upstream PR #55947 updated to the Model Runner V2 implementation agreed in RFC #55267, rebased onto vLLM main `b87339888d` (tensor parallelism supported; pipeline, data and context parallelism, KV transfer and KV-sharing fast prefill refuse at startup).
- Documented the PR revision's compatibility runs: Ampere (RTX 3090), Hopper (H100, FlashAttention 3 and 2), Blackwell (GB10, RTX 5090 at tensor-parallel 2).
- The PR revision's throughput over plain vLLM, same card and serving shape: 2.6x single-stream on an H100 (2.2x at eight streams, 1.7x at 32), 2.1 to 2.4x on a GB10 (1.1 to 1.6x at four and sixteen streams), 1.7 to 1.9x on an RTX 3090; faster than the Model Runner V1 build in every measured cell; SGLang's independent Uno integration within 3% at every concurrency.
- The PR revision's first-token latency is level with plain vLLM (same card and client, single-stream 28–31 ms against plain's 27–29 ms; zero kernel compilations in serving): the draft-input kernel no longer specialises on prompt length, every served draft launch is warmed at startup under a JIT-monitor self-check, and a request stops drafting once its remaining budget cannot consume another draft. Throughput on the same card 1.7–1.85x plain single-stream and 1.15–1.5x at four streams.
- The PR's continuous-batching test now forces a real preemption under a pinned KV pool and checks the resumed request's tokens against its solo run; the greedy exact-token comparison carries a plain-vs-plain control with a tie-aware verdict, since a CUDA-graph engine on Ampere does not reproduce itself bit-exactly across processes.

## [0.1.0] - 2026-09-08

- Prebuilt NVIDIA containers for Linux AMD64 and ARM64, using the same pinned Uno implementation.
- Native build-platform selection and an explicit `PLATFORM` override.
- Direct pull-and-run instructions, with source builds and validation details in their own sections.
- Versioned per-platform image identities and hardware integration records.
- GPU serving checks passed on RTX 3090 (Ampere) and GB10 (Blackwell), including streaming, shared-prefix requests and batches of 8 and 32 concurrent requests.

## [0.1.0rc1] - 2026-09-08

First public tagged version of Uno for vLLM.

### Added

- Consolidated, sanitized Uno integration patch pinned to vLLM commit `e962733e08d10f7ca65dac4df99e116460b8b174`.
- Native two-pass Uno serving with draft-only LoRA routing, asynchronous scheduling, private draft graphs, first-draft replay, overlapping LoRA work, prefix caching, and seed-row reuse.
- Reproducible source audit, patch application, local image build, serving, verification, and bundle helpers under `release/`.
- Selected sampled, mixed-batch, greedy, and execution-path gates under `gates/`.
- Pinned Qwen3-8B BF16, `K=8`, single-GPU Linux AMD64 serving profile, including an explicit FlashAttention 2 setting for Ampere.
- Reader-facing configuration and validation guides, citation metadata, and upstream attribution.

### Validated

- 368 source-level CPU tests passed with no skips on the public source under Python 3.12.
- 16 package and HTTP-gate regression tests passed, including mixed-load failure detection.
- A Linux AMD64 image build preserved all 17 checked compiled libraries.
- A bounded RTX 3090 integration run loaded compiled CUDA, the pinned Qwen model, and the Uno adapter; served a 256-token completion; and observed active drafting, private graph replay, and seed-row reuse.

### Scope

- Published source and a prebuilt Linux AMD64 container at `ghcr.io/brntech/vllm-uno:0.1.0rc1`; local image builds remain available.
- Included matched sampled-distribution and strict greedy reference/candidate verification tools.
- Historical H100 research results are documented in the companion paper. ARM64/Blackwell packaging was added in `0.1.0`; multi-GPU execution requires separate engineering and validation.

[0.1.0rc1]: https://github.com/brntech/vllm-uno/releases/tag/v0.1.0rc1

[0.1.0]: https://github.com/brntech/vllm-uno/releases/tag/v0.1.0

[0.2.0]: https://github.com/brntech/vllm-uno/releases/tag/v0.2.0

[0.3.0]: https://github.com/brntech/vllm-uno/releases/tag/v0.3.0
