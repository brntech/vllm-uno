# Uno for vLLM v0.4.4: measurement evidence

One session on 2026-10-07, one RTX 3090 (24 GB), Gemma 4 26B A4B AWQ 4-bit
(`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` revision `0ef577a5710035bd2d3a3f27e4f5cb2e86a9a9ba`), 32k context, three rounds
per workload with the arm order rotated (each arm ran first, second and third once), every server started fresh for its
workload. Before its timed requests, every production run sent two warmup requests, one at a time, identical to the first
two of the 72 requests (each JSON file states it: `warmup`, `warmup_requests`); with prefix caching on, those two
prompts were in the cache when timed. On the one-at-a-time servers the long documents followed the production requests
after a prefix-cache reset. Three arms:

- `uno-v0.4.4-recommended`: the v0.4.4 container with `UNO_PROFILE=gemma4 UNO_RECOMMENDED=1` (the launcher sets
  `UNO_K=5 UNO_PLOOKUP_L=2 UNO_PLOOKUP_MAX_CTX=4096 UNO_DRAFT_MOE_TOPK=4 UNO_GEMMA_SPLITKV_MULTI=1` and capture sizes up
  to 64) and the adapter this release ships (`adapter_model.safetensors` sha256
  `5eda8f7879489737246b86c56f2836e2d4cd359e0bc994ea193ac9f9f658e534`).
- `dflash-k8-vllm-0.30.0`: the stock `vllm/vllm-openai:v0.30.0` image with the z-lab DFlash drafter, K=8.
- `plain`: the v0.4.4 container without speculative decoding.

All arms used the same serve flags (`common_serve_settings` in the request-level JSON files). The measured v0.4.4
container is the release build whose runtime files are listed in `image-files.sha256`.

| file | what it holds |
| --- | --- |
| `production72-one-at-a-time.json` | 72 production requests replayed one at a time: per request, prompt and completion tokens and the full HTTP request time; per arm and round, the speculative-decoding counters (drafts, accepted tokens). Prompts and answers are private and not included. |
| `production72-four-at-a-time.json` | the same 72 requests with four in flight, on a fresh server: the same fields plus the batch wall time. |
| `production72-eight-at-a-time.json` | the same with eight in flight. |
| `long-documents.json` | open-data documents at about 2k, 14k and 28k prompt tokens, 4 offsets x 2 tasks per level, up to 384 output tokens, served after the one-at-a-time production requests on the same server: per request, decode ms per output token and the request's drafts and accepted tokens. |
| `kv-cache.json` | the KV cache size each server reported at startup, per workload and round. |
| `TABLES.md` | every table in the release notes and the model card, computed only from the JSON files above: per-round values, medians, ranges, and per-round differences. |
| `v0.4.3-comparison/` | an earlier session (2026-10-05) of the v0.4.4 container against the v0.4.3 container, both with the v0.4.3 adapter: the evidence for what v0.4.4's code changes at four and eight requests in flight. |

The CPU and GPU test logs of the release are in the `cpu/` and `gpu/` folders next to this file.
