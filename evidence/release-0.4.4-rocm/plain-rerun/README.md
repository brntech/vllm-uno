# Uno vs plain decoding on the R9700 (2026-10-08)

Part of the v0.4.4-rocm release evidence: Uno for vLLM v0.4.4 on AMD against plain decoding in stock vLLM 0.30.0
ROCm on one Radeon AI PRO R9700, on the release's workloads (production traffic one at a time and with 4 and 8
requests in flight; long documents at 2k, 14k and 28k tokens), four balanced rounds, one fresh server per arm,
workload and round.

- `TABLES.md`: every cell, both arms, per round, medians, ranges, spreads and the call verdict.
- `plain-rerun.json`: the same numbers, raw and as printed.

- Before any timed request, every server of both arms ran a short speed check (2 untimed warmups, then 3
  requests of 256 tokens of a synthetic prompt, one at a time); a server below the check's speed threshold
  was replaced by a fresh one. The check also warms Uno's just-in-time compiled kernels.
- One rounding rule: Uno's figures are never better than measured; plain's are rounded to nearest. Every derived
  figure in `TABLES.md` is computed from the figures printed beside it.
