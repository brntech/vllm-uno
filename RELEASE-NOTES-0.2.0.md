# Uno for vLLM v0.2.0

Uno for vLLM v0.2.0 packages the Model Runner V2 implementation at
`5da193919b44335ddf14eac193dfc9e8d5e59df5` on vLLM
`b87339888d29329c42c42573e34cc2beebdcc48b`. It builds from vLLM's
digest-pinned per-commit CI image, overlays only verified Python source, and
preserves the base image's compiled CUDA libraries.

## What changed

- Replaces the Model Runner V1 release implementation with native MRV2 Uno.
- Uses the MRV2 speculative configuration: adapter path, mask-token bound, and
  deterministic noise seed, with eight speculative tokens per step.
- Uses the PR's measured production shape: BF16 Qwen3-8B, async scheduling,
  prefix caching, FlashAttention 2, `max_num_seqs=16`,
  `max_num_batched_tokens=2048`, an explicit 2 GiB KV cache, and nine CUDA
  graph capture sizes through 144.
- Includes a reproducible patch, manifest, source-overlay build, offline
  serving validation, and deterministic source bundle.

## What this release validates

The released-image run on an RTX 3090 passed health, functional greedy and
sampled generation, SSE streaming, cold shared-prefix requests, C=8, and a
queued C=32 capacity check. The G2v2 sampled and mixed chunk-8 gates passed at
n=256 with 32 tests and 5,000 permutations each. Uno engagement, full startup
warm-up, graph replay, eight real-request `nvidia-smi` residency samples, and
zero in-serving JIT-compilation warnings were recorded. A deliberately wrong
adapter revision failed through the same candidate launcher as expected.

The implementation is the one carried by upstream
[PR #55947](https://github.com/vllm-project/vllm/pull/55947), as part of
[RFC #55267](https://github.com/vllm-project/vllm/issues/55267); the measured
tables and receipts are in the PR description.

## Platform

v0.2.0 is Linux AMD64 only. An ARM64 image follows when vLLM tags a release
image containing the pinned base.

See [README.md](README.md), [docs/configuration.md](docs/configuration.md),
and [docs/validation.md](docs/validation.md) for the exact profile, source
identity, validation record, and artifact checksums.
