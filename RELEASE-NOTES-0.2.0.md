# Uno for vLLM v0.2.0

Uno for vLLM v0.2.0 packages the Model Runner V2 implementation from
`689b11a8cac0e6865786c41cc0d77afa6afaf885` on vLLM
`b87339888d29329c42c42573e34cc2beebdcc48b`. It uses a digest-pinned,
commit-matched vLLM CI base image and overlays Python source only, preserving
the base image's compiled CUDA libraries.

## What changed

- Replaced the Model Runner V1 release implementation with native MRV2 Uno.
- Uses the MRV2 speculative configuration: adapter path, mask-token bound, and
  deterministic noise seed, with eight speculative tokens per step.
- Forces Model Runner V2 and asynchronous scheduling; the supported profile
  keeps Qwen3-8B BF16, prefix caching, FlashAttention 2 on Ampere, and the
  pinned Qwen model and Uno adapter revisions.
- Includes reproducible patch, source-overlay, image-build, and validation
  tooling for the pinned upstream commit.

## What this release measures

The release validation exercises live serving on an RTX 3090: health, greedy
and sampled generation, streaming, shared-prefix requests, and concurrent
batches of 8 and 32 requests. It records Uno engagement, startup warm-up and
self-check evidence, in-serving compilation observations, and the specified
lossless gates. This release reports functional and correctness validation, not
new performance cells.

This draft is not releasable yet. The v0.2.0 candidate's strict greedy gate
and cold-serving JIT gate failed; see [docs/validation.md](docs/validation.md)
before tagging or publishing it.

## Platform

v0.2.0 is Linux AMD64 only. An ARM64 image follows when vLLM tags a release
image containing the pinned base.

See [README.md](README.md), [docs/configuration.md](docs/configuration.md),
and [docs/validation.md](docs/validation.md) for the exact profile, source
identity, and validation record.
