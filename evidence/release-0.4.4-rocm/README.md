# Uno for vLLM v0.4.4 on AMD Radeon: measurement evidence

One session on 2026-10-08, one AMD Radeon AI PRO R9700 (gfx1201, 32 GB), ROCm 7.2, Gemma 4 26B A4B AWQ 4-bit
(`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` revision `0ef577a5710035bd2d3a3f27e4f5cb2e86a9a9ba`), 32k context, three rounds
per workload with the arm order rotated (each arm ran first, second and third once), every server started fresh for its
workload. Before its timed requests, every production run sent two warmup requests, one at a time, identical to the first
two of the 72 requests (each JSON file states it: `warmup`, `warmup_requests`); with prefix caching on, those two prompts
were in the cache when timed. On the one-at-a-time servers the long documents followed the production requests after a
prefix-cache reset. The workloads, clients and sampling are those of the v0.4.4 release session on an RTX 3090
(`evidence/release-0.4.4/`). Three arms:

- `uno-v0.4.4-rocm-recommended`: the `0.4.4-rocm` container (built from `patch/rocm-0.4.4/`) with `UNO_PROFILE=gemma4
  UNO_RECOMMENDED=1` (the launcher sets `UNO_K=5 UNO_PLOOKUP_L=2 UNO_PLOOKUP_MAX_CTX=4096 UNO_DRAFT_MOE_TOPK=4
  UNO_GEMMA_SPLITKV_MULTI=1` and capture sizes up to 64, and on gfx12 its tuned MoE config folder, the prefill tiles and
  the adapter on the main stream) and the v0.4.4 adapter (`adapter_model.safetensors` sha256
  `5eda8f7879489737246b86c56f2836e2d4cd359e0bc994ea193ac9f9f658e534`).
- `dflash-k8-vllm-0.30.0-rocm`: the stock `vllm/vllm-openai-rocm:v0.30.0` image as published, with the z-lab DFlash
  drafter, K=8, and nothing else added (no tuned config, no tile patch, no environment setting of this repository).
- `plain-vllm-0.30.0-rocm`: the same stock image without speculative decoding.

Every server ran the serve flags in `common_serve_settings` (in the request-level JSON files; Model Runner V2 and
`--generation-config vllm` included). What differs between the arms is listed in `arm_flags`, exactly, with who passed
each flag: for Uno the flags its launcher sets (the LoRA settings, `--jit-monitor-verbose`, its capture sizes to 64 and
its speculative config); for Uno and DFlash K8 `--per-request-spec-decode-metrics summary`, passed by the measurement
harness; for DFlash K8 and plain their capture sizes (to 72 and to 40) and DFlash K8's speculative config, passed by the
measurement harness.
The measured `0.4.4-rocm` container's runtime files are listed in `image-files.sha256`.

Plain decoding ran at two speeds on servers of one configuration: one request at a time 12.4 ms per token on one server
and 20.3 on the other two, with identical startup logs and the same number of output tokens; at four and eight in flight
one server of three was the slow one. Every figure is in the JSON files as measured.

| file | what it holds |
| --- | --- |
| `production72-one-at-a-time.json` | 72 production requests replayed one at a time: per request, prompt and completion tokens and the full HTTP request time; per arm and round, the speculative-decoding counters (drafts, accepted tokens). Prompts and answers are private and not included. |
| `production72-four-at-a-time.json` | the same 72 requests with four in flight, on a fresh server: the same fields plus the batch wall time. |
| `production72-eight-at-a-time.json` | the same with eight in flight. |
| `long-documents.json` | open-data documents at about 2k, 14k and 28k prompt tokens, 4 offsets x 2 tasks per level, up to 384 output tokens, served after the one-at-a-time production requests on the same server: per request, decode ms per output token and the request's drafts and accepted tokens. |
| `kv-cache.json` | the KV cache size each server reported at startup, per workload and round. |
| `TABLES.md` | every table in the release notes, computed only from the JSON files above: per-round values, medians, ranges, and per-round differences. |
| `image-files.sha256` | sha256 of the release image's runtime files (`/opt/uno-runtime.sha256` in the image): the engine files under vLLM, the launcher, the prefill-tile patch and the R9700 data files. |
| `cpu/SUMMARY.md` | the CPU test groups on the release image, with the exact selectors, counts and exit codes. |
