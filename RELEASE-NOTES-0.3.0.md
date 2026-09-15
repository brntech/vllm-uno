# Uno for vLLM v0.3.0

Uno for vLLM v0.3.0 adds Gemma 4 26B A4B to the Model Runner V2 implementation
at `cf87916880b051e8782521dfe2afa12e0627e172` on vLLM
`00972dfd72988942138a7a6089eaee08580210b8`. The package builds from vLLM's
digest-pinned per-commit CI image, overlays only verified Python source, and
preserves the base image's compiled CUDA libraries.

**Release status.** The Qwen3-8B profile passes its full gate set on this image.
The Gemma 4 profile passes every functional, capacity and refusal gate but
**does not pass the sampled-distribution gate** against a matched plain
reference, so this release does not certify it as a validated serving profile.
Read [docs/validation.md](docs/validation.md) before using the Gemma profile.

## What changed

- Gemma 4 26B A4B support on the Model Runner V2 Uno path: MoE drafting,
  sliding-window attention, and language-only admission for a multimodal
  checkpoint, served as a second profile (`UNO_PROFILE=gemma4`) beside the
  unchanged Qwen3-8B default.
- Fail-closed draft-scope admission: language-only targets, one KV cache group
  covering every draft layer, no KV transfer or encoder-cache consumer, and
  SM86 for the draft MoE top-k variant, each refused by name at startup.
- An opt-in split-KV draft attention path (`UNO_GEMMA_SPLITKV=1`) covering the
  checkpoint's two head-size families, logging the engaged width, head size,
  query and KV head counts and segment count per family.
- An opt-in draft MoE top-k variant (`UNO_DRAFT_MOE_TOPK=4`) that captures and
  replays the drafters' routers at top-4 while the verifier keeps its own
  routing, restores every router on a partial capture, and refuses an
  uncaptured serving shape by dispatch key, environment variable and variant
  name instead of falling back to eager routing.
- A draft-graph coverage receipt and a dispatch-key diagnostic, so a
  captured-versus-dispatched disagreement is visible in the log.
- An ordered two-layer patch series and a corrected release tree: the
  reconstruction now records the squashed head's own tree,
  `0149f03eb8287bdfdcc916752b3851405695d350`.

## What this release validates

The released-image run on an RTX 3090 (24 GiB) re-passes the v0.2.0 gate set
for Qwen3-8B BF16 at K=8: health, functional greedy and sampled generation, SSE
streaming, cold shared-prefix requests, C=8, a queued C=32 capacity check, and
both distributional gates at n=256 with 32 tests, 5,000 permutations, no red
test, and zero in-serving JIT-compilation warnings. A deliberately wrong adapter
revision failed through the same candidate launcher.

For Gemma 4 26B A4B AWQ at K=4 the same runner passes greedy decoding, the live
API and capacity checks, the config-time vision refusal that names
`Uno requires a language-only model`, and the draft MoE top-k variant boot. Its
serving evaluation records 166.713 against 143.694 tokens/s for matched
plain-versus-Uno greedy decode over five 384-token requests on one card
(sum over sum, 1.16x for the Uno arm); that is a throughput result from the
port's evaluation record, not a losslessness result.

The distributional gate does not pass for Gemma 4 with the arms on identical
flags. The Uno arm differs from the matched plain arm at every position and
joint tested at the shipped chunk-1 convention, and at the chunk-8 and chunk-32
conventions as well, while a plain-versus-plain control on the same image and
flags is clean. On the first sampled token of the prose prefix the Uno arm
returns one token for all 256 draws where plain spreads over four. Two
explanations survive the controls — a verification path that differs
semantically from plain, or a plain control that does not run the attention
kernel the Uno verify pass runs — and both must be excluded before the profile
can be claimed lossless. The separating run is described in
[docs/validation.md](docs/validation.md).

G2 preparation/KV, G3 attention, and broader G4/G7 qualification remain
CUDA_UNVERIFIED; the available receipts support only the bounded scenarios
exercised.

## Platform

v0.3.0 is Linux AMD64 only; the pinned base image is published for linux/amd64
alone. An ARM64 image follows when vLLM tags a release image containing the
pinned base.

See [README.md](README.md), [docs/configuration.md](docs/configuration.md), and
[docs/validation.md](docs/validation.md) for the exact profiles, source
identity, validation record, and artifact checksums.
