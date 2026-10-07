# Uno for vLLM v0.4.4 against v0.4.3, session of 2026-10-05 (v0.4.3 adapter, Hub revision s5): tables derived from this folder's JSON files

An earlier session, superseded for the release card: the release's own tables, with the adapter it ships, are
in ../TABLES.md.

Every figure is recomputed from the JSON files in this folder. Differences are per round (round 1 / round 2)
and of the two-round means; time per token, so negative = less time per token than the comparison arm.
Rounding: uno-v0.4.4-recommended's figures are rounded so that none is better than measured (speeds, speed-ups
and tokens per cycle down; times and time differences up); the other arms to nearest.

## production72, one request at a time (HTTP clock)

The uno-v0.4.3-recommended arm's one-at-a-time requests are in production72-one-at-a-time.json; they are not tabled: one request at a time, v0.4.4 runs v0.4.3's attention path.

| arm | ms/token r1 | ms/token r2 | mean | tok/s per request r1 / r2 | x plain | tokens per cycle r1 / r2 |
|---|---:|---:|---:|---:|---:|---|
| uno-v0.4.4-recommended | 4.444 | 4.525 | 4.485 | 225.0 / 221.0 | 1.672 | 4.07 / 4.04 |
| dflash-k8-vllm-0.30.0 | 4.775 | 4.812 | 4.794 | 209.4 / 207.8 | 1.565 | 3.24 / 3.21 |
| plain | 7.476 | 7.528 | 7.502 | 133.8 / 132.8 | 1.000 | - |

- uno-v0.4.4-recommended vs dflash-k8-vllm-0.30.0: per round -6.94 % / -5.97 %; means -6.45 %.

## production72, four requests in flight (batch clock, fresh server)

| arm | tok/s r1 | tok/s r2 | mean ms/token | tok/s (from mean) | x plain | tokens per cycle r1 / r2 |
|---|---:|---:|---:|---:|---:|---|
| uno-v0.4.4-recommended | 517.5 | 515.6 | 1.936 | 516.5 | 1.486 | 4.06 / 4.05 |
| uno-v0.4.3-recommended | 493.5 | 468.5 | 2.080 | 480.7 | 1.384 | 4.09 / 4.08 |
| dflash-k8-vllm-0.30.0 | 496.3 | 502.0 | 2.003 | 499.1 | 1.437 | 3.22 / 3.28 |
| plain | 360.2 | 335.5 | 2.878 | 347.4 | 1.000 | - |

- uno-v0.4.4-recommended vs dflash-k8-vllm-0.30.0: per round -4.10 % / -2.63 %; means -3.37 %.
- uno-v0.4.3-recommended vs dflash-k8-vllm-0.30.0: per round +0.57 % / +7.17 %; means +3.85 %.
- uno-v0.4.4-recommended vs uno-v0.4.3-recommended: per round -4.64 % / -9.14 %; means -6.95 %.

## production72, eight requests in flight (batch clock, fresh server)

| arm | tok/s r1 | tok/s r2 | mean ms/token | tok/s (from mean) | x plain | tokens per cycle r1 / r2 |
|---|---:|---:|---:|---:|---:|---|
| uno-v0.4.4-recommended | 716.2 | 712.6 | 1.400 | 714.4 | 1.391 | 4.06 / 4.09 |
| uno-v0.4.3-recommended | 363.6 | 365.3 | 2.744 | 364.5 | 0.710 | 4.07 / 4.11 |
| dflash-k8-vllm-0.30.0 | 680.7 | 676.2 | 1.474 | 678.4 | 1.322 | 3.26 / 3.23 |
| plain | 514.8 | 511.8 | 1.948 | 513.3 | 1.000 | - |

- uno-v0.4.4-recommended vs dflash-k8-vllm-0.30.0: per round -4.97 % / -5.10 %; means -5.04 %.
- uno-v0.4.3-recommended vs dflash-k8-vllm-0.30.0: per round +87.18 % / +85.11 %; means +86.14 %.
- uno-v0.4.4-recommended vs uno-v0.4.3-recommended: per round -49.23 % / -48.73 %; means -48.98 %.

## Long documents, one request at a time (median decode ms per output token)

The uno-v0.4.3-recommended arm's document requests are in long-documents.json; they are not tabled: one request at a time, v0.4.4 runs v0.4.3's attention path.

| arm | level | r1 | r2 | pooled | x plain | tokens per cycle | mean prompt tokens | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uno-v0.4.4-recommended | 2000 | 5.231 | 4.885 | 5.037 | 1.477 | 3.67 | 2051 | 16 |
| uno-v0.4.4-recommended | 14000 | 6.345 | 6.309 | 6.332 | 1.355 | 3.17 | 13913 | 16 |
| uno-v0.4.4-recommended | 28000 | 7.600 | 7.683 | 7.683 | 1.254 | 3.01 | 27656 | 16 |
| dflash-k8-vllm-0.30.0 | 2000 | 6.128 | 6.190 | 6.190 | 1.202 | 2.59 | 2051 | 16 |
| dflash-k8-vllm-0.30.0 | 14000 | 10.491 | 10.047 | 10.143 | 0.846 | 2.06 | 13913 | 16 |
| dflash-k8-vllm-0.30.0 | 28000 | 14.796 | 14.715 | 14.796 | 0.652 | 1.88 | 27656 | 16 |
| plain | 2000 | 7.438 | 7.457 | 7.443 | 1.000 | - | 2051 | 16 |
| plain | 14000 | 8.579 | 8.586 | 8.582 | 1.000 | - | 13913 | 16 |
| plain | 28000 | 9.648 | 9.635 | 9.642 | 1.000 | - | 27656 | 16 |

- uno-v0.4.4-recommended vs dflash-k8-vllm-0.30.0 at 2000: per round -14.63 % / -21.09 %; pooled -18.62 %.
- uno-v0.4.4-recommended vs dflash-k8-vllm-0.30.0 at 14000: per round -39.51 % / -37.20 %; pooled -37.57 %.
- uno-v0.4.4-recommended vs dflash-k8-vllm-0.30.0 at 28000: per round -48.63 % / -47.78 %; pooled -48.07 %.
- uno-v0.4.4-recommended vs plain at 2000: per round -29.67 % / -34.49 %; pooled -32.32 %.
- uno-v0.4.4-recommended vs plain at 14000: per round -26.04 % / -26.52 %; pooled -26.21 %.
- uno-v0.4.4-recommended vs plain at 28000: per round -21.22 % / -20.26 %; pooled -20.31 %.

## KV cache at the 32k profile (server-reported tokens)

- uno-v0.4.4-recommended: 75,731 to 75,768 (6 servers)
- uno-v0.4.3-recommended: 74,161 (6 servers)
- dflash-k8-vllm-0.30.0: 68,463 (6 servers)
- plain: 101,370 (6 servers)
