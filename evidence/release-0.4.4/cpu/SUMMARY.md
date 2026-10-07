# Uno for vLLM v0.4.4: CPU tests (2026-10-05)

Image `vllm-uno:0.4.4` built from `patch/0005-v0.4.4` on the v0.4.3 release image; CPU only, `--network none`, no GPU.
Two builds of that overlay ran these tests. The launcher was corrected after the GPU measurements (it now refuses empty
or too-wide overrides that had no measured meaning, and parses every numeric setting once: surrounding spaces stripped,
anything but digits refused); the fast, fast043 and kernel groups below ran on the corrected image, built with
`docker build --no-cache` so the Dockerfile's own checks ran in that build. The v0.4.3 exactness groups (t1 to k5-mixed) ran on the measured image: both images carry byte-identical engine
files (the in-image hashes of every runtime file except the launcher are equal), and the launcher's own tests check
that `UNO_RECOMMENDED=1` starts exactly the measured engine command and environment.
The Triton interpreter (`TRITON_INTERPRET=1`) runs vLLM's own rejection and resample kernels, the prompt-lookup kernels
and the unified attention kernel on CPU tensors. Each group runs in its own detached container (`run_release_cpu.sh`,
real pytest exit code in `<group>.rc`); the selectors are in `release_cpu_groups.sh` (the kernel group ran as two
containers, `kernel` and `kernel-padded`). Linux hosts, at most 6 CPUs per run (the interpreter is single-threaded).

v0.4.4 changes no proposal, rejection or sampling code (the in-image hashes of those files equal v0.4.3's), so the
v0.4.3 exactness suites (`tests/v0.4.3`) run unchanged on the v0.4.4 image; every statistic they print is identical to
the v0.4.3 release run (same seeds).

| group | suite | --cpus | what | result | pytest rc | wall |
| --- | --- | ---: | --- | --- | ---: | ---: |
| fast | v0.4.4 | 3 | launcher (`UNO_RECOMMENDED` launches the measured engine command and environment byte for byte; explicit and empty values; widths up to 9 start; refusals of empty `UNO_K`, gate-less lookup, more than 9 verify rows and multi-request on Qwen3; numeric settings parsed once (`' 8'`, `'8 '` and `'+8'` cannot skip the width check); capture sizes following `UNO_GEMMA_SPLITKV_MULTI`), multi-request admission at the backend and the kernel, documented switches, startup checks at both capture lists | 100 passed | 0 | 3 s |
| fast043 | v0.4.3 | 3 | lookup and gate unit tests | 44 passed | 0 | 5 s |
| kernel | v0.4.4 | 3 | multi-request split-KV == one-request split-KV (`torch.equal`), switch off == the 2D kernel, both Gemma 4 layouts, widths 5/6/8, 2 and 4 requests | 12 passed | 0 | 1 min |
| kernel-padded | v0.4.4 | 3 | the same equality on the real rows of a step with one padded request (zero query rows, seq_len 0, as a larger captured graph gives it) and requests whose rows are a prompt chunk (no cached context, 3 tokens of context), both layouts, widths 6/8 | 4 passed | 0 | 18 s |
| t1 | v0.4.3 | 1 | T=1, vocab 20,000, K=4 L=2 | 1 passed | 0 | 18 min |
| t07 | v0.4.3 | 1 | T=0.7 + top-p 0.9, K=4 L=2 | 1 passed | 0 | 19 min |
| exact | v0.4.3 | 1 | exactness, negative control, greedy, broken-row power checks | 6 passed | 0 | 25 min |
| gate | v0.4.3 | 1 | mixed gated + ungated batch, gated requests under broken lookup columns | 2 passed | 0 | 30 min |
| k5-t1 | v0.4.3 | 1 | T=1, K=5 L=2 (the recommended shape) | 1 passed | 0 | 19 min |
| k5-t07 | v0.4.3 | 1 | T=0.7 + top-p 0.9, K=5 L=2 | 1 passed | 0 | 20 min |
| k5-greedy | v0.4.3 | 1 | greedy, K=5 L=2 | 1 passed | 0 | 3 min |
| k5-mixed | v0.4.3 | 1 | gated (5 rows) and full (7 rows) requests in one rejection call | 1 passed | 0 | 19 min |

**Total: 174 passed, 0 failed, every rc 0.**

Startup checks (`test_startup_refusal_v044.py`, in fast): the profiles are read from dry runs of the shipped launcher.
With the recommended settings (capture sizes up to 64, up to 8 requests) and `UNO_DRAFT_MOE_TOPK=4`, the engine check
lets `UNO_K` 2 to 8, with and without lookup, start with every draft batch covered (the launcher refuses earlier any
`UNO_K` + `UNO_PLOOKUP_L` + 1 above 9 with the multi-request path on); `UNO_K=9` (72 rows) refuses at startup naming the missing
batch; `UNO_K=5` above 8 requests (9, 16) still refuses, as in v0.4.3. With the default capture sizes (up to 40) the
v0.4.3 behaviour is unchanged (`UNO_K` 6 to 8 refuse). Launches that never capture draft graphs (`--enforce-eager`,
cudagraph mode NONE or PIECEWISE, attention without uniform-batch graphs) refuse before capture at either list.

Inverted run (`../startup/`): the v0.4.4 suites on the v0.4.3 image. `test_release_v044.py`: 58 failed, 5 passed (the
five that pass are one-request admission, which v0.4.4 leaves unchanged), rc 1. `test_startup_refusal_v044.py` stops at
collection (the v0.4.3 launcher prints no settings line), rc 2.

The kernel groups use short contexts (0 to 300 tokens) to keep the interpreter fast; the GPU test of the release
(`../gpu/`) runs the same equality at 300 to 6,000 tokens and 2, 4 and 8 requests, plus the padded-request and
prompt-chunk steps with 2, 4 and 7 requests (36 of 36 equal).
