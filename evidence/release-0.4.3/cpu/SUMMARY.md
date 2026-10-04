# Uno for vLLM v0.4.3: CPU tests (2026-10-04)

Image `vllm-uno:0.4.3` built from `patch/0004-v0.4.3` on the v0.4.1 release image; CPU only, `--network none`, no
GPU. The Triton interpreter (`TRITON_INTERPRET=1`) runs vLLM's own rejection and resample kernels and the prompt-lookup
kernels on CPU tensors. Suites in `tests/v0.4.3/`: `test_plookup.py` (lookup matcher, kernels, exactness at K=4 L=2),
`test_plgate.py` (per-request length gate), `test_recipe_k5.py` (exactness at the recommended K=5 L=2 shape),
`test_release_v043.py` (release checks), `test_startup_refusal.py` (startup check for `UNO_DRAFT_MOE_TOPK=4`),
`conftest.py` (test-side broken lookup rows for the power checks). Each group runs in its own detached container
(`run_release_cpu.sh`, real pytest exit code in `<group>.rc`); the selectors are in `release_cpu_groups.sh`. These logs
are the run on the final image, on a 16-core Linux host, two groups at a time with `--cpus 4` each (the Triton
interpreter is single-threaded, hence the longer walls); every exactness statistic is identical to the earlier run
of the same tests.

| group | --cpus | what | result | pytest rc | wall |
| --- | ---: | --- | --- | ---: | ---: |
| fast | 4 | every test except the ten slow exactness tests | 82 passed | 0 | 17 s |
| t1 | 4 | T=1, vocab 20,000, K=4 L=2 | 1 passed: 7 positions incl. both lookup rows and the bonus token, chi2 p 0.18-0.47, n 673-1,536 | 0 | 17 min |
| t07 | 4 | T=0.7 + top-p 0.9, vocab 20,000, K=4 L=2 | 1 passed: lookup rows p 0.32 (n 1,427) / 0.60 (n 608); the out-of-nucleus lookup token is never emitted | 0 | 19 min |
| exact | 4 | T=1 vocab-16 K=2 test and its negative control, greedy vocab 20,000 K=4 L=2, broken-row power checks | 6 passed: exact p 0.13-0.47; negative control p 6e-104; greedy 1,024 cycles, both lookup rows reached 723 times; a stale lookup row is never rejected (225/225/225); a stale second row: row 1 exact (445 -> 97), row 2 never rejected (97 -> 97); a lookup column mixed with uniform mass (eps 0.5) is detected (TV 0.213, p 6e-16) | 0 | 25 min |
| gate | 4 | mixed gated + ungated batch (odd requests gated), and gated requests under broken lookup columns | 2 passed: full requests exact at 7 positions (p 0.076-0.81, n 344-768), gated exact at 0-4 (p 0.052-0.96), gated never emit more than 5 tokens; under broken columns full requests accept every reached lookup token (412/412/412) and gated stay exact (p 0.067-0.90) | 0 | 28 min |
| k5-t1 | 4 | T=1, vocab 20,000, K=5 L=2 | 1 passed: all 8 positions incl. both lookup rows and the bonus token, chi2 p 0.108-0.875, n 653-1,536 | 0 | 19 min |
| k5-t07 | 4 | T=0.7 + top-p 0.9, K=5 L=2 | 1 passed: lookup rows p 0.684 (n 1,304) / 0.917 (n 321); the out-of-nucleus lookup token is never emitted | 0 | 19 min |
| k5-greedy | 4 | greedy, vocab 20,000, K=5 L=2 | 1 passed: 1,024 cycles exact; both lookup rows reached 502 times; includes a greedy lookup rejection | 0 | 3 min |
| k5-mixed | 4 | odd requests gated (5 rows), even full (7 rows), one rejection call | 1 passed: full exact at 8 positions (p 0.125-0.475, n 299-768), gated exact at 6 (p 0.238-0.997, n 614-768); gated never emit more than 6 tokens | 0 | 17 min |

**Total: 96 passed, 0 failed, every rc 0.**

The fast group includes `test_startup_refusal.py` (32 tests). Each builds the speculator through the real
`UnoSpeculator` constructor, the real draft CUDA graph manager setup and the real `capture()` on CPU, at the profile's
shape read from the shipped `serve.sh` (up to 8 requests, `FULL_AND_PIECEWISE`, capture sizes up to 40). With
`UNO_DRAFT_MOE_TOPK=4`: covered launches start (`UNO_K` 2, 3 and 4, `UNO_K=5` with and without lookup at 8 requests,
`UNO_K=5` at 1, 4 and 7 requests, and a FULL-decode-only mode with a target graph); `UNO_K` 6, 7 and 8 (with and without
lookup) and `UNO_K=5` at 9 and 16 requests refuse after capture with the missing batch sizes named; and six launches
that would never capture a draft graph refuse before capture: `--enforce-eager`, cudagraph mode `NONE`, `PIECEWISE`
only, attention without uniform-batch graphs, capture sizes below one request's drafts, and a FULL-only mode in which
the target captures nothing. With the setting off, the same six launches start and an uncovered batch only warns, as
in v0.4.2. On the previous build (without the pre-capture check) exactly the six pre-capture refusal tests fail.

`test_release_v043.py` checks that the defaults are off, that the eight patched vLLM files read no environment variable
outside the documented switches (`UNO_PLOOKUP_L`, `UNO_PLOOKUP_MAX_CTX`, `UNO_DRAFT_VOCAB`, `UNO_GEMMA_SPLITKV`), and
that the release lookup columns hold the exact point mass.
