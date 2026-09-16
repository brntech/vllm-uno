# Draft PR description: Gemma 4 on the Model Runner V2 Uno path

This is the description draft for proposing the Gemma 4 26B A4B port upstream.
It is not sent by this release. The upstream contribution guidelines require
four things of an AI-assisted PR description, and this draft states each
explicitly: why it does not duplicate existing work, the test commands and
results, the model evaluation results, and the AI-assistance statement.

## Why this is not duplicating an existing PR

The Model Runner V2 Uno speculator is proposed in
[PR #55947](https://github.com/vllm-project/vllm/pull/55947) under
[RFC #55267](https://github.com/vllm-project/vllm/issues/55267), and this
change extends that work rather than repeating it: it adds
`vllm/v1/worker/gpu/spec_decode/uno_draft_moe.py`, the Gemma 4 draft-scope
admission it needs, the split-KV draft attention path in the Triton unified
attention kernel, and the draft-graph coverage receipt. Nothing here proposes
a second Uno implementation, a second speculator plugin, or a competing MoE
draft path.

## What the change is

Gemma 4 26B A4B is a sparse MoE model with sliding-window attention and a
language-only path through a multimodal checkpoint. The port adds:

- Draft-scope admission requires a language-only target, one KV cache group
  covering every draft layer, no KV transfer or encoder-cache consumer, and SM86
  for the top-k variant.
- An opt-in split-KV draft attention path (`UNO_GEMMA_SPLITKV=1`) for the
  two head-size families in the checkpoint.
- An opt-in draft MoE top-k variant (`UNO_DRAFT_MOE_TOPK=4`) that captures the
  draft routers at top-4 under CUDA graphs, restores every router and the
  draft LoRA hook on a partial capture, and refuses an uncaptured serving shape
  by dispatch key, environment variable and variant name instead of falling
  back to eager routing.
- A startup coverage receipt and a dispatch-key diagnostic.

## Test commands and results

```bash
python -m pytest tests/v1/spec_decode/test_uno_draft_moe.py \
  tests/v1/spec_decode/test_triton_uno_splitkv.py \
  tests/v1/spec_decode/test_uno_mrv2.py tests/v1/spec_decode/test_uno_config.py
```

- Source fixture suite on the ported tree: 338 passed, 10 failed; every
  failure is in the pre-existing `arg_names` set that the same suite reports
  before this change (337 passed, 10 failed).
- Pre-commit passes on every changed file; `git diff --check` is clean.
- The focused inverted run (one mutation per run) shows each new assertion
  failing for the behaviour it guards, not for a missing symbol: the
  never-captured refusal test fails on the absent variant name, and the
  draft-guard pass-classification test fails on the misclassified probe.

## Model evaluation results

Serving evaluation on one RTX 3090 (24 GiB), Gemma 4 26B A4B AWQ, the
step-1900 Uno adapter, language-only, `K=4`, `TRITON_ATTN`, `max_num_seqs=4`,
`max_model_len=8192`, `gpu_memory_utilization=0.85`, two warm-ups then five
fixed 384-token greedy requests:

| arm | tok/s (sum over sum) | ms/cycle | acceptance |
|---|---:|---:|---:|
| plain | 143.694 | 6.8998 ms/tok | - |
| Uno K=4 | 166.713 | 18.7893 | 0.5296 |

Both arms are the same card, session, workload and repeat convention; the
ratio is 166.713 / 143.694 = 1.16x.

**The Gemma 4 profile is certified: Uno is as close to plain as plain is to itself across sessions on this hardware, under greedy and sampled decoding alike.** The certification interprets the sampled comparisons alongside
the cross-session same-arm controls. The raw outcomes of the earlier reading are
kept as history: the floor pair passes (0 of 36 red, tightest p `0.008528`), the
candidate pair is red (32 of 44 red, min p at the `2.27e-05` grid minimum), and
both same-arm controls across sessions are red at the same magnitude (plain
against plain 22 of 36, the Uno arm against itself 37 of 44). The shipped
permutation gate is not a valid instrument for this profile on this hardware:
its first sampled token on the prose prefix is a near-tie whose per-row law moves
by more than a nat between the rows of one pass and between launches, so a
256-draw marginal test measures a mixture of row-dependent laws. The numbers and
the instrument are recorded in [docs/validation.md](validation.md).

Startup reports zero compilations for its own warm-up, and its self-check
already says it cannot prove launch coverage on this profile; the serving slice
then carries two `kernel_unified_attention` compilations, identical in both
arms. That is a warm-up coverage gap of the profile rather than a Uno cost. The
CUDA-graph replay path and the MoE capture path both report their own receipts.

Scope limits, stated plainly: G2 preparation/KV, G3 attention, and broader
G4/G7 qualification remain CUDA_UNVERIFIED; the available receipts support
only the bounded scenarios exercised. Across fresh servers, greedy token streams were identical on 2 of 4 prompts
between the plain runs and on 1 of 4 prompts between Uno and either plain run;
the records are in [docs/validation.md](validation.md).

## AI assistance

AI assistance was used to develop this change, including the implementation,
the tests, the box runs and this description. The submitting human has reviewed
every changed line, run the commands above, and is able to defend the change
end to end.
