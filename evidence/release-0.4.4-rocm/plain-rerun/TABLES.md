# Uno vs plain decoding on one Radeon AI PRO R9700 (2026-10-08)

Part of the v0.4.4-rocm release evidence. Every figure below is built from the session's numbers through one
rounding rule (Uno's figures never better than measured; plain's to nearest).

- Uno arm: Uno for vLLM v0.4.4 on AMD (ghcr.io/brntech/vllm-uno:0.4.4-rocm, UNO_RECOMMENDED=1, the v0.4.4 adapter).
- Plain arm: plain decoding, stock vLLM 0.30.0 ROCm (vllm/vllm-openai-rocm:v0.30.0, no speculation).
- Same serve flags, container flags, model (Gemma 4 26B A4B AWQ, cyankiwi 0ef577a5), runner, sampling (each
  production request's own recorded settings; documents server-default), workload and repeat convention on both
  arms; 4 rounds per cell, one fresh server per arm, cell and round, within-round order balanced.
- Before any timed request, every server of both arms ran a short speed check (2 untimed warmups, then 3
  requests of 256 tokens of a synthetic prompt, one at a time); a server below the check's speed threshold
  was replaced by a fresh one. The check also warms Uno's just-in-time compiled kernels.
- Call rule: every round's difference has one sign and exceeds the larger round spread, AND every plain server
  lies within 5 % of the plain median with its end speed check under the gate.
- Derived figures are computed from the printed figures: a production median is the median of the printed
  round values; a range is their lowest and highest; a spread is 100 x (highest - lowest) / mean of them; a
  per-round difference is 100 x (Uno / plain - 1) of the printed round values of that round; Uno x plain (plain median / Uno median) is
  computed from the two medians as printed.

| row | workload | clock | plain median (range) | Uno median (range) | Uno x plain (plain median / Uno median, as printed) | Uno vs plain per round (printed round values) | verdict |
|---|---|---|---|---|---:|---|---|
| 1 | production traffic, one request at a time | ms/token, HTTP clock, median of the 4 printed round values | 12.446 (12.428-12.459) | 8.874 (8.848-8.941) | 1.40x | -28.6 %, -28.8 %, -28.0 %, -28.8 % | CALLED |
| 2 | production traffic, 4 requests in flight | ms/token, batch clock, median of the 4 printed round values | 5.930 (5.898-5.950) | 4.109 (4.104-4.115) | 1.44x | -30.7 %, -30.7 %, -30.2 %, -31.0 % | CALLED |
| 3 | production traffic, 8 requests in flight | ms/token, batch clock, median of the 4 printed round values | 4.934 (4.923-4.997) | 3.257 (3.241-3.268) | 1.51x | -35.0 %, -33.6 %, -33.6 %, -34.4 % | CALLED |
| 4 | long documents, 2k-token prompts | decode ms/token, pooled median of 32 requests (range: the printed per-round medians) | 12.847 (12.831-12.862) | 10.226 (9.653-10.359) | 1.25x | -19.3 %, -24.9 %, -21.1 %, -21.2 % | CALLED |
| 5 | long documents, 14k-token prompts | decode ms/token, pooled median of 32 requests (range: the printed per-round medians) | 16.865 (16.847-16.874) | 13.612 (12.919-13.757) | 1.23x | -18.8 %, -19.3 %, -23.3 %, -18.4 % | CALLED |
| 6 | long documents, 28k-token prompts | decode ms/token, pooled median of 32 requests (range: the printed per-round medians) | 20.559 (20.538-20.570) | 17.415 (16.994-17.647) | 1.18x | -15.1 %, -14.2 %, -17.2 %, -15.5 % | CALLED |

## Per round (time per token)

| row | arm | r1 | r2 | r3 | r4 | spread % (100 x (highest - lowest) / mean) |
|---|---|---:|---:|---:|---:|---:|
| 1 | plain | 12.459 | 12.449 | 12.428 | 12.442 | 0.25 |
| 1 | uno | 8.888 | 8.860 | 8.941 | 8.848 | 1.05 |
| 2 | plain | 5.933 | 5.927 | 5.898 | 5.950 | 0.88 |
| 2 | uno | 4.111 | 4.107 | 4.115 | 4.104 | 0.27 |
| 3 | plain | 4.997 | 4.923 | 4.925 | 4.943 | 1.50 |
| 3 | uno | 3.247 | 3.268 | 3.267 | 3.241 | 0.83 |
| 4 | plain | 12.842 | 12.862 | 12.831 | 12.851 | 0.24 |
| 4 | uno | 10.359 | 9.653 | 10.120 | 10.120 | 7.02 |
| 5 | plain | 16.847 | 16.874 | 16.862 | 16.874 | 0.16 |
| 5 | uno | 13.665 | 13.612 | 12.919 | 13.757 | 6.21 |
| 6 | plain | 20.558 | 20.570 | 20.538 | 20.556 | 0.16 |
| 6 | uno | 17.448 | 17.647 | 16.994 | 17.360 | 3.76 |
