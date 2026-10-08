# CPU tests on the v0.4.4-rocm image

`tests/rocm-0.4.4/run_rocm_cpu.sh` on the release image: the public `tests/v0.4.4/` and `tests/v0.4.3/` suites with
`tests/rocm-0.4.4/tests-rocm.diff` applied (three test-side lines: the image's version string `0.4.4-rocm`, and
`enable_adaptive_verification=False` on the startup test double in both startup suites, a field vLLM v0.30.0's
speculative config reads; False is its default). Selectors = `tests/v0.4.4/release_cpu_groups.sh`. Every group is one
CPU-only container (no GPU devices, no network, Triton's interpreter); each `.rc` is pytest's own exit code, captured
inside the container that ran the tests.

| group | suite | tests | passed | rc |
| --- | --- | --- | ---: | ---: |
| fast | v0.4.4: launcher recipe, multi-request admission, documented switches, startup refusals at both capture lists | `test_release_v044.py`, `test_startup_refusal_v044.py` | 100 | 0 |
| fast043 | v0.4.3: the lookup and gate unit tests | `test_plookup.py`, `test_plgate.py` (not the slow cases) | 44 | 0 |
| kernel | v0.4.4: multi-request split-KV == one-request split-KV, Triton interpreter | `test_splitkv_multi_cpu.py` | 16 | 0 |
| t1 | v0.4.3 exactness, K=4 L=2, temperature 1 | `test_plookup.py -k wide_temperature1_k4` | 1 | 0 |
| t07 | v0.4.3 exactness, K=4 L=2, temperature 0.7 top-p | `test_plookup.py -k wide_temperature07_top_p_k4` | 1 | 0 |
| exact | v0.4.3 exactness: point-mass lookup, negative control, greedy, positive control | `test_plookup.py` | 6 | 0 |
| gate | v0.4.3: mixed batches with the length gate | `test_plgate.py` | 2 | 0 |
| k5-t1 | v0.4.3 exactness at the recommended K=5 L=2, temperature 1 | `test_recipe_k5.py` | 1 | 0 |
| k5-t07 | the same, temperature 0.7 top-p | `test_recipe_k5.py` | 1 | 0 |
| k5-greedy | the same, greedy | `test_recipe_k5.py` | 1 | 0 |
| k5-mixed | the same, mixed gated and full requests | `test_recipe_k5.py` | 1 | 0 |

**174 passed, rc 0 in every group** (the same 174 tests as on the CUDA release image, `evidence/release-0.4.4/cpu/`).
The runner reports a failing group: with the version line of `tests-rocm.diff` removed, `test_version_and_launcher_syntax`
fails (`'0.4.4-rocm' == '0.4.4'`), group `fast` exits 1 and the runner exits 1.
