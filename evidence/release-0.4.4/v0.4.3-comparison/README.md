# Uno for vLLM v0.4.4 against v0.4.3: measurement evidence (session of 2026-10-05)

This folder holds the session that compares the v0.4.4 container with the v0.4.3 container, both serving the v0.4.3
adapter (Hub revision `s5`). It is an earlier session, superseded for the release card: the release's own figures, with the adapter it ships, come from a later session and are in the folder above.

One session on 2026-10-05, one RTX 3090 (24 GB), Gemma 4 26B A4B AWQ 4-bit
(`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` revision `0ef577a5710035bd2d3a3f27e4f5cb2e86a9a9ba`), 32k context, two rounds
per workload with the arm order reversed in round 2, every server started fresh for its workload. Before its timed
requests, every production run sent two warmup requests, one at a time, identical to the first two of the 72 requests
(each JSON file states it: `warmup`, `warmup_requests`); with prefix caching on, those two prompts were in the cache
when timed. Four arms:

- `uno-v0.4.4-recommended`: the v0.4.4 container with `UNO_PROFILE=gemma4 UNO_RECOMMENDED=1` (the launcher sets
  `UNO_K=5 UNO_PLOOKUP_L=2 UNO_PLOOKUP_MAX_CTX=4096 UNO_DRAFT_MOE_TOPK=4 UNO_GEMMA_SPLITKV_MULTI=1` and capture sizes up
  to 64) and the v0.4.3 adapter (`adapter_model.safetensors` sha256
  `e784e5df1c2235f354de2161894f9b244b0cd72930888a0600cdaccf7d842c60`).
- `uno-v0.4.3-recommended`: the published v0.4.3 container with its four recommended settings and the same adapter.
- `dflash-k8-vllm-0.30.0`: the stock `vllm/vllm-openai:v0.30.0` image with the z-lab DFlash drafter, K=8.
- `plain`: the v0.4.4 container without speculative decoding.

All arms used the same serve flags (`common_serve_settings` in the request-level JSON files). The measured v0.4.4
container was built from the release overlay by the release build script; its runtime files are the ones listed in
`image-files.sha256`.

| file | what it holds |
| --- | --- |
| `production72-one-at-a-time.json` | 72 production requests replayed one at a time: per request, prompt and completion tokens and the full HTTP request time; per arm and round, the speculative-decoding counters (drafts, accepted tokens). Prompts and answers are private and not included. |
| `production72-four-at-a-time.json` | the same 72 requests with four in flight, on a fresh server: the same fields plus the batch wall time. |
| `production72-eight-at-a-time.json` | the same with eight in flight. |
| `long-documents.json` | open-data documents at about 2k, 14k and 28k prompt tokens, 4 offsets x 2 tasks per level, up to 384 output tokens, served after the one-at-a-time production requests on the same server: per request, decode ms per output token and the request's drafts and accepted tokens. |
| `kv-cache.json` | the KV cache size each server reported at startup, per workload and round. |
| `TABLES.md` | this session's tables, computed only from the JSON files above: per-round values, two-round means, and per-round differences. The release notes quote it only for the v0.4.3 comparison at eight requests in flight; every other release figure comes from `../TABLES.md`. |

The CPU and GPU test logs of the release are in `../cpu/` and `../gpu/`.
