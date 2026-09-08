# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses a compact release-candidate suffix for pre-release versions.

## [0.1.0rc1] - 2026-09-08

First public research pre-release of Uno for vLLM.

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

### Release scope

- Research pre-release with source and a prebuilt Linux AMD64 container at `ghcr.io/brntech/vllm-uno:0.1.0rc1`; local image builds remain available.
- Full matched sampled-distribution and strict greedy reference/candidate passes are not claimed for this version. The included verifier keeps these gates explicit and reproducible.
- Historical H100 research results are documented in the companion paper. ARM64/Blackwell packaging and multi-GPU execution are outside this release.

[0.1.0rc1]: https://github.com/brntech/vllm-uno/releases/tag/v0.1.0rc1
