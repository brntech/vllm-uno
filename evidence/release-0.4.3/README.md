# Uno for vLLM v0.4.3: measurement evidence

One session on 2026-10-04, one RTX 3090 (24 GB), Gemma 4 26B A4B AWQ 4-bit
(`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` revision `0ef577a5710035bd2d3a3f27e4f5cb2e86a9a9ba`), 32k context, two rounds
per workload with the arm order reversed in round 2. Three arms:

- `uno-v0.4.3-recipe`: the v0.4.3 container with `UNO_PROFILE=gemma4 UNO_K=5 UNO_PLOOKUP_L=2
  UNO_PLOOKUP_MAX_CTX=4096 UNO_DRAFT_MOE_TOPK=4` and the 38.6k-prompt adapter (`adapter_model.safetensors` sha256
  `e784e5df1c2235f354de2161894f9b244b0cd72930888a0600cdaccf7d842c60`).
- `dflash-k8-vllm-0.30.0`: the stock `vllm/vllm-openai:v0.30.0` image with the z-lab DFlash drafter, K=8.
- `plain`: the v0.4.3 container without speculative decoding.

All arms used the same serve flags (`common_serve_settings` in the request-level JSON files). The measured container is the v0.4.3
release candidate (`sha256:d4fa1d020ef4…`); the published image differs from it only by startup checks that refuse
`UNO_DRAFT_MOE_TOPK=4` when no draft CUDA graph will be captured or a reachable draft batch has none. They run once at
startup and do not change drafting, verification or sampling.

| file | what it holds |
| --- | --- |
| `production72-one-at-a-time.json` | 72 production requests replayed one at a time: per request, prompt and completion tokens and the full HTTP request time; per arm and round, the speculative-decoding counters (drafts, accepted tokens). Prompts and answers are private and not included. |
| `production72-four-at-a-time.json` | the same 72 requests with four in flight: the same fields plus the batch wall time. |
| `long-documents.json` | open-data documents at about 2k, 14k and 28k prompt tokens, 4 offsets x 2 tasks per level, up to 384 output tokens: per request, decode ms per output token and the request's drafts and accepted tokens. |
| `kv-cache.json` | the KV cache size each server reported at startup. |
| `lookup-ablation.json` | an earlier session (2026-10-03) on a pre-release build with the same prompt-lookup and length-gate code, the previous training stage and K=4: lookup with the gate at 4,096, lookup without the gate, and lookup off, on the same production requests (one at a time) and the same long documents. Uno arms only. |
| `TABLES.md` | every table in the release notes and the model card, computed only from the JSON files above. |

Call rule used throughout: a difference is called only when every compared round agrees in sign and each round's
difference is larger than the larger of the two arms' round-to-round spreads. Every comparison uses the same rounds for
both arms: in `lookup-ablation.json` the arms ran different numbers of rounds (lookup without the gate ran rounds 1 and 2
only), so each lookup comparison pools only the rounds both arms ran and names them.
