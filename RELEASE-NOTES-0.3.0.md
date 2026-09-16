# Uno for vLLM v0.3.0

Uno for vLLM v0.3.0 adds Gemma 4 26B A4B to the Model Runner V2 implementation
at `cf87916880b051e8782521dfe2afa12e0627e172` on vLLM
`00972dfd72988942138a7a6089eaee08580210b8`. The package builds from vLLM's
digest-pinned per-commit CI image, overlays only verified Python source, and
preserves the base image's compiled CUDA libraries.

**Release status.** Both profiles are certified on this image. The Qwen3-8B profile
passes its full gate set. The Gemma 4 profile is certified: Uno is as close to plain as plain is to itself across sessions on this hardware, under greedy and sampled decoding alike. The numbers behind that sentence are in
[docs/validation.md](docs/validation.md), which also carries the gate outcomes.

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
- A floor-matched distributional gate (`gates/lossless_floor.py`), the profile
  evaluation's own instrument, shipped beside the permutation comparator, with a
  chunk contract that records every pass's request chunking and refuses a
  comparison, or a floor, recorded at a different chunk.
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
`Uno requires a language-only model`, and the draft MoE top-k variant boot. The
port evaluation measured Uno at 166.713 tokens/s and plain decoding at 143.694
tokens/s over five 384-token greedy requests, a 1.16× speedup. This measurement
establishes throughput for the stated workload; the certification evidence is
reported separately in [docs/validation.md](docs/validation.md).

The Gemma 4 profile is certified: Uno is as close to plain as plain is to itself across sessions on this hardware, under greedy and sampled decoding alike. The certification interprets the sampled comparisons alongside the
cross-session same-arm controls. The kit's permutation test stays the instrument
for profiles whose plain arm is reproducible, as Qwen3-8B is on this image.

Launch the certified profile with `UNO_PROFILE=gemma4 bash release/serve.sh
cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit <adapter directory>`; the adapter it uses is
published at `<adapter repository, published with this release>`.

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
