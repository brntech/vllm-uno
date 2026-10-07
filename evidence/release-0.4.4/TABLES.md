# Uno for vLLM v0.4.4: tables derived from this folder's JSON files

Every figure is recomputed from the JSON files in this folder. Three rounds per workload.
Production tables: r1, r2 and r3 are the rounds; 'median' (and the tokens per second at the median) is the
median of the three rounds and 'range' their range; 'x plain' is a ratio of medians, plain's median time per
token divided by the arm's (not a median of per-round ratios); tokens per cycle are per round.
Long-document table: r1, r2 and r3 are each round's median over its requests; 'pooled' is the median of all the
requests of the three rounds pooled; 'x plain' is the ratio of the pooled medians; tokens per cycle are pooled
over the three rounds. Differences are per round (round 1 / round 2 / round 3) and of the medians (production)
or of the pooled medians (documents); time per token, so negative = less time per token than the comparison arm.
Rounding: uno-v0.4.4-recommended's figures are rounded so that none is better than measured (speeds, speed-ups and tokens per
cycle down; times and time differences up); the other arms to nearest.

## production72, one request at a time (HTTP clock)

| arm | ms/token r1 | r2 | r3 | median | range | tok/s per request at the median | x plain | tokens per cycle r1 / r2 / r3 |
|---|---:|---:|---:|---:|---|---:|---:|---|
| uno-v0.4.4-recommended | 4.463 | 4.411 | 4.420 | 4.420 | 4.411-4.463 | 226.2 | 1.690 | 4.09 / 4.11 / 4.11 |
| dflash-k8-vllm-0.30.0 | 4.650 | 4.786 | 4.820 | 4.786 | 4.650-4.820 | 208.9 | 1.561 | 3.32 / 3.22 / 3.24 |
| plain | 7.506 | 7.471 | 7.469 | 7.471 | 7.469-7.506 | 133.9 | 1.000 | - |

- uno-v0.4.4-recommended vs dflash-k8-vllm-0.30.0: per round -4.03 % / -7.85 % / -8.31 %; medians -7.66 %.
- uno-v0.4.4-recommended vs plain: per round -40.54 % / -40.96 % / -40.82 %; medians -40.84 %.

## production72, four requests in flight (batch clock, fresh server)

| arm | tok/s r1 | r2 | r3 | tok/s at the median | range (tok/s) | x plain | tokens per cycle r1 / r2 / r3 |
|---|---:|---:|---:|---:|---|---:|---|
| uno-v0.4.4-recommended | 521.9 | 528.8 | 523.7 | 523.7 | 521.9-528.8 | 1.480 | 4.09 / 4.14 / 4.12 |
| dflash-k8-vllm-0.30.0 | 486.9 | 497.0 | 503.9 | 497.0 | 486.9-503.9 | 1.405 | 3.21 / 3.20 / 3.23 |
| plain | 353.7 | 356.4 | 352.7 | 353.7 | 352.7-356.4 | 1.000 | - |

- uno-v0.4.4-recommended vs dflash-k8-vllm-0.30.0 (ms/token): per round -6.70 % / -6.02 % / -3.79 %; medians -5.10 %.
- uno-v0.4.4-recommended vs plain (ms/token): per round -32.23 % / -32.60 % / -32.66 %; medians -32.47 %.

## production72, eight requests in flight (batch clock, fresh server)

| arm | tok/s r1 | r2 | r3 | tok/s at the median | range (tok/s) | x plain | tokens per cycle r1 / r2 / r3 |
|---|---:|---:|---:|---:|---|---:|---|
| uno-v0.4.4-recommended | 721.5 | 721.6 | 723.9 | 721.6 | 721.5-723.9 | 1.387 | 4.08 / 4.15 / 4.09 |
| dflash-k8-vllm-0.30.0 | 679.3 | 679.1 | 673.0 | 679.1 | 673.0-679.3 | 1.306 | 3.23 / 3.24 / 3.24 |
| plain | 517.2 | 524.3 | 520.1 | 520.1 | 517.2-524.3 | 1.000 | - |

- uno-v0.4.4-recommended vs dflash-k8-vllm-0.30.0 (ms/token): per round -5.85 % / -5.89 % / -7.03 %; medians -5.89 %.
- uno-v0.4.4-recommended vs plain (ms/token): per round -28.31 % / -27.34 % / -28.15 %; medians -27.92 %.

## Long documents, one request at a time (median decode ms per output token)

| arm | level | r1 | r2 | r3 | pooled | x plain | tokens per cycle | mean prompt tokens | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uno-v0.4.4-recommended | 2000 | 4.916 | 4.710 | 4.870 | 4.837 | 1.536 | 3.69 | 2051 | 24 |
| uno-v0.4.4-recommended | 14000 | 6.278 | 6.258 | 6.386 | 6.273 | 1.371 | 3.17 | 13913 | 24 |
| uno-v0.4.4-recommended | 28000 | 7.298 | 7.735 | 7.486 | 7.298 | 1.319 | 3.02 | 27656 | 24 |
| dflash-k8-vllm-0.30.0 | 2000 | 6.385 | 5.994 | 6.603 | 6.334 | 1.174 | 2.55 | 2051 | 24 |
| dflash-k8-vllm-0.30.0 | 14000 | 9.638 | 11.152 | 10.304 | 10.244 | 0.840 | 2.05 | 13913 | 24 |
| dflash-k8-vllm-0.30.0 | 28000 | 14.071 | 14.196 | 14.274 | 14.071 | 0.685 | 1.88 | 27656 | 24 |
| plain | 2000 | 7.487 | 7.424 | 7.421 | 7.433 | 1.000 | - | 2051 | 24 |
| plain | 14000 | 8.660 | 8.586 | 8.582 | 8.605 | 1.000 | - | 13913 | 24 |
| plain | 28000 | 9.699 | 9.609 | 9.627 | 9.632 | 1.000 | - | 27656 | 24 |

- uno-v0.4.4-recommended vs dflash-k8-vllm-0.30.0 at 2000: per round -23.00 % / -21.42 % / -26.24 %; pooled -23.62 %.
- uno-v0.4.4-recommended vs dflash-k8-vllm-0.30.0 at 14000: per round -34.85 % / -43.88 % / -38.02 %; pooled -38.76 %.
- uno-v0.4.4-recommended vs dflash-k8-vllm-0.30.0 at 28000: per round -48.13 % / -45.51 % / -47.55 %; pooled -48.13 %.
- uno-v0.4.4-recommended vs plain at 2000: per round -34.33 % / -36.55 % / -34.37 %; pooled -34.92 %.
- uno-v0.4.4-recommended vs plain at 14000: per round -27.50 % / -27.11 % / -25.59 %; pooled -27.09 %.
- uno-v0.4.4-recommended vs plain at 28000: per round -24.75 % / -19.50 % / -22.24 %; pooled -24.23 %.

## KV cache at the 32k profile (server-reported tokens)

- uno-v0.4.4-recommended: 75,731 to 75,768 (9 servers)
- dflash-k8-vllm-0.30.0: 68,463 (9 servers)
- plain: 101,370 (9 servers)
