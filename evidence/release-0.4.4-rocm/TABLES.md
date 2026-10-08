# Uno for vLLM v0.4.4 on AMD Radeon: tables derived from this folder's JSON files

Every figure is recomputed from the JSON files in this folder. Three rounds per workload.
Production tables: r1, r2 and r3 are the rounds; 'median' (and the tokens per second at the median) is the
median of the three rounds and 'range' their range; 'x plain' is a ratio of medians, plain's median time per
token divided by the arm's (not a median of per-round ratios); tokens per cycle are per round.
Long-document table: r1, r2 and r3 are each round's median over its requests; 'pooled' is the median of all the
requests of the three rounds pooled; 'x plain' is the ratio of the pooled medians; tokens per cycle are pooled
over the three rounds. Differences are per round (round 1 / round 2 / round 3) and of the medians (production)
or of the pooled medians (documents); time per token, so negative = less time per token than the comparison arm.
Rounding: uno-v0.4.4-rocm-recommended's figures are rounded so that none is better than measured (speeds, speed-ups and tokens per
cycle down; times and time differences up); the other arms to nearest.
Calls: a comparison is CALLED when all three per-round differences share a sign and each exceeds the larger of
the two arms' round spreads (100 x (max - min) / mean); an observed ratio or difference the rule does not call
is printed with 'not called' beside it and is not a result. The full call table is the last section.

## production72, one request at a time (HTTP clock)

| arm | ms/token r1 | r2 | r3 | median | range | tok/s per request at the median | x plain | tokens per cycle r1 / r2 / r3 |
|---|---:|---:|---:|---:|---|---:|---:|---|
| uno-v0.4.4-rocm-recommended | 8.920 | 8.995 | 8.951 | 8.951 | 8.920-8.995 | 111.7 | 2.263 (not called) | 4.12 / 4.10 / 4.12 |
| dflash-k8-vllm-0.30.0-rocm | 11.920 | 11.948 | 11.966 | 11.948 | 11.920-11.966 | 83.7 | 1.696 (not called) | 3.22 / 3.22 / 3.22 |
| plain-vllm-0.30.0-rocm | 12.425 | 20.260 | 20.283 | 20.260 | 12.425-20.283 | 49.4 | 1.000 | - |

- uno-v0.4.4-rocm-recommended vs dflash-k8-vllm-0.30.0-rocm: per round -25.16 % / -24.71 % / -25.19 %; medians -25.08 %. Called.
- uno-v0.4.4-rocm-recommended vs plain-vllm-0.30.0-rocm: per round -28.21 % / -55.60 % / -55.87 %; medians -55.81 %. Not called.

## production72, four requests in flight (batch clock, fresh server)

| arm | tok/s r1 | r2 | r3 | tok/s at the median | range (tok/s) | x plain | tokens per cycle r1 / r2 / r3 |
|---|---:|---:|---:|---:|---|---:|---|
| uno-v0.4.4-rocm-recommended | 237.4 | 241.0 | 241.5 | 241.0 | 237.4-241.5 | 1.443 | 4.14 / 4.11 / 4.10 |
| dflash-k8-vllm-0.30.0-rocm | 197.1 | 196.9 | 196.9 | 196.9 | 196.9-197.1 | 1.179 (not called) | 3.26 / 3.26 / 3.27 |
| plain-vllm-0.30.0-rocm | 126.5 | 167.0 | 168.7 | 167.0 | 126.5-168.7 | 1.000 | - |

- uno-v0.4.4-rocm-recommended vs dflash-k8-vllm-0.30.0-rocm (ms/token): per round -17.01 % / -18.34 % / -18.47 %; medians -18.32 %. Called.
- uno-v0.4.4-rocm-recommended vs plain-vllm-0.30.0-rocm (ms/token): per round -46.73 % / -30.72 % / -30.14 %; medians -30.72 %. Called.

## production72, eight requests in flight (batch clock, fresh server)

| arm | tok/s r1 | r2 | r3 | tok/s at the median | range (tok/s) | x plain | tokens per cycle r1 / r2 / r3 |
|---|---:|---:|---:|---:|---|---:|---|
| uno-v0.4.4-rocm-recommended | 307.4 | 310.1 | 310.7 | 310.1 | 307.4-310.7 | 1.537 | 4.14 / 4.13 / 4.17 |
| dflash-k8-vllm-0.30.0-rocm | 297.6 | 296.9 | 299.0 | 297.6 | 296.9-299.0 | 1.475 | 3.28 / 3.23 / 3.26 |
| plain-vllm-0.30.0-rocm | 165.0 | 201.8 | 204.9 | 201.8 | 165.0-204.9 | 1.000 | - |

- uno-v0.4.4-rocm-recommended vs dflash-k8-vllm-0.30.0-rocm (ms/token): per round -3.19 % / -4.26 % / -3.79 %; medians -4.04 %. Called.
- uno-v0.4.4-rocm-recommended vs plain-vllm-0.30.0-rocm (ms/token): per round -46.32 % / -34.95 % / -34.05 %; medians -34.95 %. Called.

## Long documents, one request at a time (median decode ms per output token)

| arm | level | r1 | r2 | r3 | pooled | x plain | tokens per cycle | mean prompt tokens | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uno-v0.4.4-rocm-recommended | 2000 | 10.297 | 10.106 | 10.130 | 10.106 | 2.038 (not called) | 3.80 | 2051 | 24 |
| uno-v0.4.4-rocm-recommended | 14000 | 13.673 | 13.681 | 13.251 | 13.518 | 1.810 (not called) | 3.19 | 13913 | 24 |
| uno-v0.4.4-rocm-recommended | 28000 | 17.638 | 17.264 | 17.267 | 17.551 | 1.619 (not called) | 3.01 | 27656 | 24 |
| dflash-k8-vllm-0.30.0-rocm | 2000 | 16.107 | 16.138 | 16.139 | 16.124 | 1.278 (not called) | 2.56 | 2051 | 24 |
| dflash-k8-vllm-0.30.0-rocm | 14000 | 33.814 | 33.851 | 33.860 | 33.840 | 0.723 | 2.26 | 13913 | 24 |
| dflash-k8-vllm-0.30.0-rocm | 28000 | 55.569 | 55.554 | 55.560 | 55.560 | 0.511 | 1.92 | 27656 | 24 |
| plain-vllm-0.30.0-rocm | 2000 | 12.836 | 20.623 | 20.644 | 20.600 | 1.000 | - | 2051 | 24 |
| plain-vllm-0.30.0-rocm | 14000 | 16.862 | 24.614 | 24.627 | 24.477 | 1.000 | - | 13913 | 24 |
| plain-vllm-0.30.0-rocm | 28000 | 20.525 | 28.437 | 28.473 | 28.416 | 1.000 | - | 27656 | 24 |

- uno-v0.4.4-rocm-recommended vs dflash-k8-vllm-0.30.0-rocm at 2000: per round -36.06 % / -37.38 % / -37.23 %; pooled -37.32 %. Called.
- uno-v0.4.4-rocm-recommended vs dflash-k8-vllm-0.30.0-rocm at 14000: per round -59.56 % / -59.58 % / -60.86 %; pooled -60.05 %. Called.
- uno-v0.4.4-rocm-recommended vs dflash-k8-vllm-0.30.0-rocm at 28000: per round -68.25 % / -68.92 % / -68.92 %; pooled -68.41 %. Called.
- uno-v0.4.4-rocm-recommended vs plain-vllm-0.30.0-rocm at 2000: per round -19.77 % / -50.99 % / -50.93 %; pooled -50.94 %. Not called.
- uno-v0.4.4-rocm-recommended vs plain-vllm-0.30.0-rocm at 14000: per round -18.91 % / -44.41 % / -46.19 %; pooled -44.77 %. Not called.
- uno-v0.4.4-rocm-recommended vs plain-vllm-0.30.0-rocm at 28000: per round -14.06 % / -39.28 % / -39.35 %; pooled -38.23 %. Not called.

## KV cache at the 32k profile (server-reported tokens)

- uno-v0.4.4-rocm-recommended: 215,042 (9 servers)
- dflash-k8-vllm-0.30.0-rocm: 192,059 (9 servers)
- plain-vllm-0.30.0-rocm: 234,212 (9 servers)

## Call table (time per token; d per round = 100 x (A / B - 1), negative = A faster)

| cell | comparison | d round 1 / 2 / 3 (%) | larger round spread (%) | called |
|---|---|---|---:|---|
| production72 one at a time | uno-v0.4.4-rocm-recommended vs dflash-k8-vllm-0.30.0-rocm | -25.16 / -24.71 / -25.19 | 0.83 | yes |
| production72 one at a time | uno-v0.4.4-rocm-recommended vs plain-vllm-0.30.0-rocm | -28.21 / -55.60 / -55.87 | 44.50 | not called |
| production72 one at a time | dflash-k8-vllm-0.30.0-rocm vs plain-vllm-0.30.0-rocm | -4.07 / -41.03 / -41.01 | 44.50 | not called |
| production72 four in flight | uno-v0.4.4-rocm-recommended vs dflash-k8-vllm-0.30.0-rocm | -17.01 / -18.34 / -18.47 | 1.70 | yes |
| production72 four in flight | uno-v0.4.4-rocm-recommended vs plain-vllm-0.30.0-rocm | -46.73 / -30.72 / -30.14 | 29.94 | yes |
| production72 four in flight | dflash-k8-vllm-0.30.0-rocm vs plain-vllm-0.30.0-rocm | -35.81 / -15.16 / -14.32 | 29.94 | not called |
| production72 eight in flight | uno-v0.4.4-rocm-recommended vs dflash-k8-vllm-0.30.0-rocm | -3.19 / -4.26 / -3.79 | 1.07 | yes |
| production72 eight in flight | uno-v0.4.4-rocm-recommended vs plain-vllm-0.30.0-rocm | -46.32 / -34.95 / -34.05 | 22.27 | yes |
| production72 eight in flight | dflash-k8-vllm-0.30.0-rocm vs plain-vllm-0.30.0-rocm | -44.55 / -32.06 / -31.45 | 22.27 | yes |
| documents 2000 | uno-v0.4.4-rocm-recommended vs dflash-k8-vllm-0.30.0-rocm | -36.06 / -37.38 / -37.23 | 1.88 | yes |
| documents 2000 | uno-v0.4.4-rocm-recommended vs plain-vllm-0.30.0-rocm | -19.77 / -50.99 / -50.93 | 43.30 | not called |
| documents 2000 | dflash-k8-vllm-0.30.0-rocm vs plain-vllm-0.30.0-rocm | +25.48 / -21.75 / -21.82 | 43.30 | not called |
| documents 14000 | uno-v0.4.4-rocm-recommended vs dflash-k8-vllm-0.30.0-rocm | -59.56 / -59.58 / -60.86 | 3.18 | yes |
| documents 14000 | uno-v0.4.4-rocm-recommended vs plain-vllm-0.30.0-rocm | -18.91 / -44.41 / -46.19 | 35.24 | not called |
| documents 14000 | dflash-k8-vllm-0.30.0-rocm vs plain-vllm-0.30.0-rocm | +100.53 / +37.53 / +37.49 | 35.24 | yes |
| documents 28000 | uno-v0.4.4-rocm-recommended vs dflash-k8-vllm-0.30.0-rocm | -68.25 / -68.92 / -68.92 | 2.15 | yes |
| documents 28000 | uno-v0.4.4-rocm-recommended vs plain-vllm-0.30.0-rocm | -14.06 / -39.28 / -39.35 | 30.79 | not called |
| documents 28000 | dflash-k8-vllm-0.30.0-rocm vs plain-vllm-0.30.0-rocm | +170.74 / +95.36 / +95.13 | 30.79 | yes |
