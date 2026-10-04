# Uno for vLLM v0.4.3: tables derived from this folder's JSON files

Call rule: a difference is called only when both rounds agree in sign and each round's |difference| exceeds the larger of the two arms' round spreads (|round 1 - round 2| / mean of the rounds).

## production72, one request at a time

| arm | ms/token r1 | ms/token r2 | mean | tok/s r1 / r2 | x plain | round spread % | tokens per cycle r1 / r2 |
|---|---:|---:|---:|---:|---:|---:|---|
| uno-v0.4.3-recipe | 4.401 | 4.473 | 4.437 | 227.2 / 223.5 | 1.681 | 1.63 | 4.11 / 4.05 |
| dflash-k8-vllm-0.30.0 | 4.763 | 4.712 | 4.738 | 209.9 / 212.2 | 1.574 | 1.08 | 3.22 / 3.26 |
| plain | 7.450 | 7.466 | 7.458 | 134.2 / 133.9 | 1.000 | 0.20 | - |

Uno vs DFlash, ms/token per round: -7.61 % / -5.06 %; mean -6.34 %; spread bar 1.63 % -> called.
Pooled single-run SD 0.80 % (df 3); smallest resolvable gap between two-round means (95 %, t 3.18) 2.56 %.

## production72, four requests in flight (batch clock)

| arm | tok/s r1 | tok/s r2 | mean ms/token | x plain | round spread % |
|---|---:|---:|---:|---:|---:|
| uno-v0.4.3-recipe | 500.8 | 490.4 | 2.018 | 1.381 | 2.10 |
| dflash-k8-vllm-0.30.0 | 504.5 | 505.5 | 1.980 | 1.407 | 0.19 |
| plain | 357.5 | 360.3 | 2.786 | 1.000 | 0.79 |

Uno vs DFlash, ms/token per round: +0.74 % / +3.08 %; spread bar 2.10 % -> not called.

## Long documents, one request at a time (median decode ms per output token)

| arm | level | r1 | r2 | pooled | x plain | round spread % | tokens per cycle | mean prompt tokens | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uno-v0.4.3-recipe | 2000 | 5.157 | 4.497 | 5.070 | 1.462 | 13.68 | 3.58 | 2050 | 16 |
| uno-v0.4.3-recipe | 14000 | 6.039 | 6.329 | 6.079 | 1.407 | 4.69 | 3.15 | 13913 | 16 |
| uno-v0.4.3-recipe | 28000 | 7.599 | 7.482 | 7.513 | 1.276 | 1.56 | 3.00 | 27656 | 16 |
| dflash-k8-vllm-0.30.0 | 2000 | 5.828 | 6.232 | 6.103 | 1.215 | 6.69 | 2.61 | 2050 | 16 |
| dflash-k8-vllm-0.30.0 | 14000 | 10.224 | 9.515 | 9.921 | 0.862 | 7.19 | 2.08 | 13913 | 16 |
| dflash-k8-vllm-0.30.0 | 28000 | 14.493 | 14.354 | 14.493 | 0.661 | 0.97 | 1.89 | 27656 | 16 |
| plain | 2000 | 7.412 | 7.418 | 7.415 | 1.000 | 0.07 | - | 2050 | 16 |
| plain | 14000 | 8.558 | 8.553 | 8.553 | 1.000 | 0.06 | - | 13913 | 16 |
| plain | 28000 | 9.581 | 9.584 | 9.583 | 1.000 | 0.03 | - | 27656 | 16 |

- Uno vs dflash-k8-vllm-0.30.0 at 2000: per round -11.51 % / -27.84 %; pooled -16.92 %; spread bar 13.68 % -> not called.
- Uno vs dflash-k8-vllm-0.30.0 at 14000: per round -40.94 % / -33.48 %; pooled -38.73 %; spread bar 7.19 % -> called.
- Uno vs dflash-k8-vllm-0.30.0 at 28000: per round -47.57 % / -47.88 %; pooled -48.16 %; spread bar 1.56 % -> called.
- Uno vs plain at 2000: per round -30.42 % / -39.38 %; pooled -31.62 %; spread bar 13.68 % -> called.
- Uno vs plain at 14000: per round -29.43 % / -26.00 %; pooled -28.92 %; spread bar 4.69 % -> called.
- Uno vs plain at 28000: per round -20.69 % / -21.94 %; pooled -21.61 %; spread bar 1.56 % -> called.

## KV cache at the 32k profile (server-reported tokens)

- uno-v0.4.3-recipe: 74,161
- uno-v0.4.2 (vllm-uno:0.4.1 image + the 28.6k-prompt adapter, profile defaults): 79,022
- plain: 101,370
- dflash-k8-vllm-0.30.0: 68,463

## Prompt lookup on, off and gated (earlier session, K=4, previous training stage; lookup-ablation.json)

Every comparison below pools only the rounds both arms ran, named in the line (the ungated arm ran rounds 1 and 2 only). The per-arm rows describe each arm over all of its own rounds.

| arm | production72 ms/token per round | mean | round spread % | tokens per cycle per round |
|---|---|---:|---:|---|
| uno-lookup-gated-4096 | 4.659 / 4.641 / 4.644 | 4.648 | 0.39 | 4.05 / 4.06 / 4.07 |
| uno-lookup-ungated | 4.622 / 4.661 | 4.642 | 0.84 | 4.08 / 4.05 |
| uno-lookup-off | 4.795 / 4.787 / 4.787 | 4.790 | 0.18 | 3.80 / 3.80 / 3.79 |

- uno-lookup-gated-4096 vs uno-lookup-off, production72 ms/token, rounds 1, 2, 3 of both arms: per round -2.84 % / -3.06 % / -2.98 %; means over those rounds -2.96 %.
- uno-lookup-ungated vs uno-lookup-off, production72 ms/token, rounds 1, 2 of both arms: per round -3.61 % / -2.63 %; means over those rounds -3.12 %.

| arm | level | per-round medians | pooled median, all its rounds | n |
|---|---:|---|---:|---:|
| uno-lookup-gated-4096 | 2000 | 4.92 / 5.15 / 5.20 | 5.04 | 24 |
| uno-lookup-gated-4096 | 14000 | 6.71 / 6.75 / 6.19 / 6.23 | 6.37 | 32 |
| uno-lookup-gated-4096 | 28000 | 7.45 / 7.37 / 7.52 / 7.28 | 7.47 | 32 |
| uno-lookup-ungated | 2000 | 5.34 / 5.25 | 5.27 | 16 |
| uno-lookup-ungated | 14000 | 6.74 / 6.90 | 6.74 | 16 |
| uno-lookup-ungated | 28000 | 7.64 / 7.83 | 7.71 | 16 |
| uno-lookup-off | 2000 | 4.94 / 5.14 / 4.99 | 5.02 | 24 |
| uno-lookup-off | 14000 | 6.54 / 6.44 / 6.50 / 6.33 | 6.40 | 32 |
| uno-lookup-off | 28000 | 7.37 / 7.47 / 6.93 / 7.43 | 7.26 | 32 |

- uno-lookup-gated-4096 vs uno-lookup-off at 2000, rounds 1, 2, 3 of both arms (n 24 per arm): per round -0.39 % / +0.24 % / +4.25 %; pooled over those rounds +0.42 % (5.04 vs 5.02); spread bar 5.55 % -> not called.
- uno-lookup-gated-4096 vs uno-lookup-off at 14000, rounds 1, 2, 3, 4 of both arms (n 32 per arm): per round +2.64 % / +4.87 % / -4.69 % / -1.51 %; pooled over those rounds -0.54 % (6.37 vs 6.40); spread bar 8.65 % -> not called.
- uno-lookup-gated-4096 vs uno-lookup-off at 28000, rounds 1, 2, 3, 4 of both arms (n 32 per arm): per round +1.18 % / -1.30 % / +8.40 % / -2.02 %; pooled over those rounds +2.93 % (7.47 vs 7.26); spread bar 7.27 % -> not called.
- uno-lookup-ungated vs uno-lookup-off at 2000, rounds 1, 2 of both arms (n 16 per arm): per round +8.10 % / +2.13 %; pooled over those rounds +4.90 % (5.27 vs 5.03); spread bar 3.88 % -> not called.
- uno-lookup-ungated vs uno-lookup-off at 14000, rounds 1, 2 of both arms (n 16 per arm): per round +3.08 % / +7.19 %; pooled over those rounds +4.66 % (6.74 vs 6.44); spread bar 2.39 % -> called.
- uno-lookup-ungated vs uno-lookup-off at 28000, rounds 1, 2 of both arms (n 16 per arm): per round +3.73 % / +4.88 %; pooled over those rounds +4.25 % (7.71 vs 7.39); spread bar 2.43 % -> called.

KV cache at the 32k profile (server-reported tokens): uno-lookup-gated-4096 74,036; uno-lookup-ungated 77,227; uno-lookup-off 79,022.

At 2k the gated arm is not gated (2k prompt + 384 output tokens < 4,096), so there it runs the ungated configuration. Round spreads here are max minus min over the compared rounds, divided by their mean.
