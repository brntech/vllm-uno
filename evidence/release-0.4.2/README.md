# v0.4.2 release evidence

One RTX 3090, the unchanged release image `ghcr.io/brntech/vllm-uno:0.4.1`
(`sha256:f679879cada7b90d658246305801f2e9316e6f21f75cb7c3d7cd8fa37642ca34`), `UNO_PROFILE=gemma4`, K=4, the shipped 64k
draft vocabulary. In the arm names, `s3` / `add-s3-s17120` is the v0.4.2 adapter (28,560 open prompts, last stage step
17,120), `p10k` is the previous 10k-prompt adapter (Hub revision `p10k`), and `add-s2-s10000` is the unpublished
20,000-prompt stage between them.

| file | what it is |
| --- | --- |
| `v042-speed.jsonl` | production traffic (72 requests, one at a time) in one session, two rounds of plain / DFlash K=8 / P10K / 20,000-prompt stage / v0.4.2 adapter = two servers per arm; `ms_per_token_http` over the measured requests, `tau` and per-position acceptance from the spec-decode counters |
| `v042-dist.jsonl` | short-prompt distribution check: mean total-variation distance of per-position token distributions (24 production prompts x 96 samples x first 8 tokens = 192 cells per pair) between three plain servers and two Uno servers with the v0.4.2 adapter, in the order plain A / Uno / plain B / Uno rep / plain C, plus split halves within each arm. Raw samples stay private (model outputs on production prompts) |
| `ctx3-L3-s3.json`, `ctx3-L3-s3-rep.json`, `ctx3-L3-dflash.json`, `ctx3-L3-plain.json` | long prompts: open documents at six lengths (2k to 28k tokens), summarizing and story tasks, 8 requests per length per server, one session in the order Uno / DFlash K=8 / plain / Uno rep; per request `decode_ms_per_token` (time from the first to the last token, per output token after the first), `e2e_ms_per_token`, prompt and completion tokens, drafts and accepted tokens |
| `table.md` | the long-prompt summary: median `decode_ms_per_token` per length (Uno over both Uno servers, 16 requests; DFlash and plain 8 each) and the ratios |
| `SHA256SUMS` | SHA-256 of every file above |
