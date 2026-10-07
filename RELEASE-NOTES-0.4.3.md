# Uno for vLLM v0.4.3

Uno for vLLM v0.4.3 adds **prompt lookup** to the Gemma 4 profile: after Uno's drafted tokens, two more candidates
are copied from the request itself, and the model checks them all in the same pass. With the recommended settings (five
Uno drafts, two lookup tokens, draft passes through 4 of the 8 experts), on BroadNet's production traffic, one request
at a time on an RTX 3090, Gemma 4 26B A4B decodes **1.68x faster than plain decoding and takes 6.3 % less time per
token than DFlash in vLLM 0.30.0**; on 14k and 28k-token documents it is 1.41x and 1.28x plain, where DFlash is
slower than plain. The output is the model's own.

The new settings are **off by default**: an unconfigured v0.4.3 server runs exactly the v0.4.2 path. Four environment
variables turn the recommended settings on (below). The release also points the Gemma 4 instructions at a new adapter, trained on 38,560
open prompts ([Hugging Face](https://huggingface.co/Broadnet/gemma-4-26B-A4B-uno-adapter/tree/s5), revision `s5`; the v0.4.2 adapter stays at
revision `s3`).

## What changed

- **Prompt lookup after Uno's drafts** (`UNO_PLOOKUP_L=L`, 1 to 8). Each proposal is K Uno drafts followed by L tokens
  copied from the request: the tokens that followed the most recent earlier occurrence of the longest matching suffix
  (up to 4 tokens) of the prompt, the answer so far and this step's drafts. The serve script sets
  `num_speculative_tokens = K + L`, so verification covers all K + L candidates in one target pass.
- **A per-request length gate** (`UNO_PLOOKUP_MAX_CTX=N`). A request whose context (prompt plus output) is above N
  tokens verifies only Uno's K drafts. The choice depends on the request's length alone, never on token values, and is
  made before the step runs. Lookup paid on short production requests and cost time on long prompts; the gate keeps the
  first and drops the second. The target model also captures full CUDA graphs at the gated width, so a gated step does
  not fall back to piecewise execution.
- Uno's static-width verify attention path now covers up to 9 rows (K + L + 1), so the wider verify keeps the
  attention kernel Uno's verify already used.
- **Recommended settings documented**: `UNO_K=5` (one more Uno draft than the profile default of 4) and
  `UNO_DRAFT_MOE_TOPK=4` (the draft pass routes each token through 4 of Gemma 4's 8 experts; verification keeps all 8).
  Both switches already exist in v0.4.1; v0.4.3 measures and recommends them with lookup. `UNO_DRAFT_MOE_TOPK=4` is for
  compute-capability-8.6 GPUs (RTX 30-series class) only and refuses to start elsewhere.
- **Startup checks for `UNO_DRAFT_MOE_TOPK=4`.** The 4-expert draft runs only inside captured CUDA graphs. The server
  now refuses to start when it will capture no draft graph (`--enforce-eager`, a CUDA graph mode without full decode
  graphs, or capture sizes too small for one request's drafts), and checks at startup that every draft batch it can be
  asked for (up to `max_num_seqs` x `UNO_K` rows) has a captured graph, refusing with a message naming the missing
  batch sizes if one does not. Before, such a server started and stopped at its first request, or once that many
  requests were in flight. Under the profile defaults (up to 8 requests, capture sizes up to 40) `UNO_K` of 5 or less
  starts; `UNO_K=6` or more, or `--max-num-seqs` above 8 at `UNO_K=5`, is refused. The checks run once at startup and
  change nothing in drafting, verification or sampling.
- Nothing else moves: same base image and vLLM build as v0.4.1 (the image is the v0.4.1 release image plus these
  files), same profile defaults, same draft vocabulary, same LoRA kernel configs.

## Measured

One RTX 3090, the v0.4.3 container, Gemma 4 26B A4B AWQ 4-bit, the new adapter, the recommended settings (`UNO_K=5`,
`UNO_PLOOKUP_L=2`, `UNO_PLOOKUP_MAX_CTX=4096`, `UNO_DRAFT_MOE_TOPK=4`). DFlash K8 ran on the stock
`vllm/vllm-openai:v0.30.0` image with the same serve flags. One session, two rounds per arm in balanced order.
Production traffic = 72 real requests, completion tokens only. Documents = open-data texts, 4 sets x 2 tasks, up to
384 output tokens, one at a time; median decode time per token, both rounds pooled. Per-arm numbers and the tables
derived from them are in the release's evidence files.

| workload | Uno v0.4.3 | DFlash K8, vLLM 0.30.0 | plain |
| --- | ---: | ---: | ---: |
| production traffic, one at a time: ms per token (two rounds) | 4.401 / 4.473 | 4.763 / 4.712 | 7.450 / 7.466 |
| tokens per second per request | 227.2 / 223.5 | 209.9 / 212.2 | 134.2 / 133.9 |
| speed-up over plain | **1.68x** | 1.57x | 1.00x |
| production traffic, four at a time: tokens per second, whole batch | 501 / 490 | 505 / 506 | 358 / 360 |
| speed-up over plain | 1.38x | 1.41x | 1.00x |
| documents, 2k-token prompts: ms per token (speed-up) | 5.07 (1.46x) | 6.10 (1.22x) | 7.41 |
| documents, 14k-token prompts | 6.08 (**1.41x**) | 9.92 (0.86x) | 8.55 |
| documents, 28k-token prompts | 7.51 (**1.28x**) | 14.49 (0.66x) | 9.58 |

- One at a time, Uno took 7.6 % and 5.1 % less time per token than DFlash (round 1, round 2). On 14k documents, 41 %
  and 33 % less; on 28k documents, 48 % less in both rounds.
- Lookup on vs off (a separate session, the previous adapter, K=4; `lookup-ablation.json`): with the gate at 4,096,
  3.0 % less time per token on production traffic than lookup off. Without the gate, lookup was 4.7 % and 4.3 % slower
  than lookup off on 14k and 28k-token documents; with the gate, 0.5 % faster at 14k and 2.9 % slower at 28k.
- KV cache at 32k context on a 24 GB RTX 3090: 74,161 tokens with the recommended settings; 79,022 for v0.4.2 as
  published (the 0.4.1 container with the 28.6k-prompt adapter, default settings), measured in the same session. Most
  of the difference comes with prompt lookup and its gate, which capture CUDA graphs at more decode widths: in the
  lookup session, 74,036 tokens with lookup and the gate, 77,227 with lookup alone, 79,022 with lookup off.

## Lossless

Every candidate, drafted or copied, goes through vLLM's own rejection sampler, checked against the draft distribution
it was drawn from (so 4-expert drafts change only what is proposed, never what is accepted); a copied token is a fixed (point-mass)
proposal, which the sampler accepts with exactly the model's probability for it, so the output distribution is the
model's. The release's CPU tests run vLLM's real rejection kernels in Triton's interpreter on the served shapes (K=4
and K=5 drafts, L=2 lookup tokens, vocabularies above 8,192 tokens) and check the emitted tokens position by position against
the model's distribution: greedy, temperature 1, temperature 0.7 with top-p 0.9, and batches that mix gated and
ungated requests. Deliberately broken lookup rows are caught by the same tests. 96 CPU tests pass (rc 0 in
every group), including the startup check's own tests; the per-test numbers are in the validation evidence.

## Upgrade and rollback

```bash
docker pull ghcr.io/brntech/vllm-uno:0.4.3
docker run ... -e UNO_PROFILE=gemma4 -e UNO_K=5 -e UNO_PLOOKUP_L=2 -e UNO_PLOOKUP_MAX_CTX=4096 -e UNO_DRAFT_MOE_TOPK=4 \
  ghcr.io/brntech/vllm-uno:0.4.3 cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit /adapter \
  -- --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4
```

The flags after `--` turn on Gemma 4's tool-call and reasoning parsers, as on the measured servers.

Rollback: unset the four settings (the v0.4.2 path in the same image), or run `ghcr.io/brntech/vllm-uno:0.4.1` with the
`s3` adapter revision. Prompt lookup was built and measured on the Gemma 4 profile only; the Qwen3-8B
profile's defaults are unchanged.
