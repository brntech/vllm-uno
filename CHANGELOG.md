# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- Upstream PR #55947 updated to the Model Runner V2 implementation agreed in RFC #55267, rebased onto vLLM main `b87339888d` (tensor parallelism supported; pipeline, data and context parallelism, KV transfer and KV-sharing fast prefill refuse at startup). The Model Runner V1 revision is preserved on `uno-core-upstream`; this package's `0.1.0` container is unchanged.
- Documented the PR revision's compatibility runs: Ampere (RTX 3090), Hopper (H100, FlashAttention 2 ⟦PENDING: and 3⟧), Blackwell (GB10, RTX 5090 at tensor-parallel 2).
- The PR revision warms every draft-input kernel shape at startup and reports draft CUDA-graph coverage per request count, removing the first-request latency the earlier revision paid to Triton compilation at each new batch size (about 120 ms per size). ⟦PENDING: first-token numbers from the dedicated RTX 3090 measurement.⟧
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
