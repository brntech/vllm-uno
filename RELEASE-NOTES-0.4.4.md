# Uno for vLLM v0.4.4

Uno for vLLM v0.4.4 makes Uno fast with **several requests in flight**, turns the recommended Gemma 4 settings on
with **one switch**, `UNO_RECOMMENDED=1`, and comes with a new Gemma 4 adapter whose last training stage drafts as many
tokens as those settings serve. On BroadNet's production traffic on one RTX 3090, Gemma 4 26B A4B with the recommended
settings and the new adapter serves 721 tokens per second with eight requests in flight (stock DFlash K8 in vLLM
0.30.0: 679) and 523 with four (stock DFlash K8 in vLLM 0.30.0: 497). One request at a time it
decodes 1.69x faster than plain decoding and takes 7.6 % less time per token than stock DFlash K8 in vLLM
0.30.0. The output is the model's own.

The adapter is on [Hugging Face](https://huggingface.co/Broadnet/gemma-4-26B-A4B-uno-adapter) (main); the v0.4.3
adapter stays there at revision `s5`. The code that proposes, accepts and samples tokens is unchanged; what changes is
which attention kernel computes a decode step with several requests in flight (below).

## What changed

- **Split-KV attention for steps with several requests** (`UNO_GEMMA_SPLITKV_MULTI=1`). Uno's split-KV attention
  kernel, used for its draft and verify passes, ran only when one request was in the step; with two or more, every
  attention call fell back to vLLM's general kernel. v0.4.4 admits a step to the split-KV kernel when every request in
  it has the same number of query rows, and keeps one launch for the whole step. The kernel itself is unchanged: each
  request's attention is computed exactly as when it runs alone (a GPU test checks the outputs are bit for bit equal
  for steps of 2, 4 and 8 requests, and for steps of 2, 4 and 7 requests plus one padded request from a CUDA graph
  with prompt-chunk rows among them, both Gemma 4 attention layouts, widths 5, 6 and 8). Against the general kernel
  v0.4.3 used for these steps, the split-KV kernel adds up each request's attention in a different order, so in steps
  with several requests the attention output can differ from v0.4.3's in the last bits (bfloat16), and with it the
  draft and target probabilities of that step. Steps whose requests have different widths keep the general kernel, as
  before; with the recommended settings that includes steps that mix requests above and below the 4,096-token length
  gate (6 and 8 rows).
- **CUDA graphs for 6 to 8 requests in flight.** With the recommended settings each request verifies 8 rows per step,
  so 6, 7 and 8 requests need graph sizes 48, 56 and 64; the profile's capture sizes ended at 40, and those steps ran
  without a CUDA graph. With `UNO_GEMMA_SPLITKV_MULTI=1` the launcher adds capture sizes 48, 56 and 64. The two come
  together: with the general attention kernel the larger sizes would also take memory from the KV cache at startup;
  with the multi-request kernel the KV cache holds 75,731 to 75,768 tokens at 32k context on a 24 GB RTX 3090.
- **One switch for the recommended settings** (`UNO_RECOMMENDED=1`, Gemma 4 profile). It sets `UNO_K=5`,
  `UNO_PLOOKUP_L=2`, `UNO_PLOOKUP_MAX_CTX=4096`, `UNO_DRAFT_MOE_TOPK=4` and `UNO_GEMMA_SPLITKV_MULTI=1`. A variable you
  set yourself wins. Set to empty, `UNO_PLOOKUP_L=` turns prompt lookup off, `UNO_DRAFT_MOE_TOPK=` drafts with all 8
  experts and `UNO_GEMMA_SPLITKV_MULTI=` turns the multi-request path off; an empty `UNO_K=`, or an empty
  `UNO_PLOOKUP_MAX_CTX=` while lookup is on, is refused at startup. The launcher prints the settings it used at startup
  (`Uno settings: ...`). It refuses `UNO_GEMMA_SPLITKV_MULTI=1` without split-KV attention, on the Qwen3 profile, or
  with more than 9 verify rows per request (`UNO_K` + `UNO_PLOOKUP_L` + 1), which the split-KV kernel does not take;
  and it refuses `UNO_RECOMMENDED=1` with the Qwen3 profile.
- **Startup checks at the new capture sizes.** The v0.4.3 check for `UNO_DRAFT_MOE_TOPK=4` (every draft batch the
  server can be asked for must have a captured graph) now finds graphs for up to 8 requests with the recommended
  settings; `--max-num-seqs` above 8 still stops at startup. Only `UNO_K=5` with `UNO_PLOOKUP_L=2` was measured.
  Without `UNO_GEMMA_SPLITKV_MULTI` the capture sizes and the check are those of v0.4.3.
- **A new adapter.** Every earlier Gemma 4 adapter was trained with four-row draft blocks; since v0.4.3 the recommended
  settings draft five tokens per block (`UNO_K=5`). The new adapter continues the v0.4.3 adapter for 2,000 updates with
  five-row draft blocks and with the draft passes routed as they are served (4 of the 8 experts, the 64k draft
  vocabulary). It has the same rank, the same configuration file and the same file layout, so it is a drop-in
  replacement. The v0.4.3 adapter stays on Hugging Face at revision `s5`.
- Nothing else moves: the image is the v0.4.3 release image plus two attention files and the launcher; same vLLM
  build, profile defaults, draft vocabulary and LoRA kernel configs. With neither new variable set, a v0.4.4 server
  runs the v0.4.3 path.

## Measured

One RTX 3090, Gemma 4 26B A4B AWQ 4-bit, the new adapter. v0.4.4 = the v0.4.4 container with `UNO_RECOMMENDED=1`;
DFlash K8 on the stock `vllm/vllm-openai:v0.30.0` image; plain decoding on the v0.4.4 container; the same serve flags
for all. One session, three rounds per arm with the arm order rotated, every server started fresh for its workload and
sent two warmup requests (the first two production requests) before the timed ones. Production traffic = 72 real
requests, completion tokens only. Documents = open-data texts, 4 sets x 2 tasks, up to 384 output tokens, one at a time,
on the one-at-a-time servers after the production requests. Production figures are the median of the three rounds, with
the range of the three rounds in brackets for tokens per second; document figures are the median of all 24 requests of
the three rounds pooled; speed-ups over plain are ratios of these medians. Per-round numbers and the tables derived
from them are in the release's evidence files; v0.4.4's figures here are rounded so that none is better than measured.

| production traffic | Uno v0.4.4 | DFlash K8, vLLM 0.30.0 | plain |
| --- | ---: | ---: | ---: |
| eight at a time: tokens per second, whole batch | **721** (721-723) | 679 (673-679) | 520 (517-524) |
| eight at a time: speed-up over plain | **1.38x** | 1.31x | 1.00x |
| four at a time: tokens per second, whole batch | **523** (521-528) | 497 (487-504) | 354 (353-356) |
| four at a time: speed-up over plain | **1.48x** | 1.41x | 1.00x |
| one at a time: ms per token | **4.420** | 4.786 | 7.471 |
| one at a time: speed-up over plain | **1.69x** | 1.56x | 1.00x |

| documents, one at a time: ms per token (speed-up over plain) | Uno v0.4.4 | DFlash K8, vLLM 0.30.0 | plain |
| --- | ---: | ---: | ---: |
| 2k-token prompts | 4.84 (1.53x) | 6.33 (1.17x) | 7.43 |
| 14k-token prompts | 6.28 (1.37x) | 10.24 (0.84x) | 8.60 |
| 28k-token prompts | 7.30 (1.31x) | 14.07 (0.68x) | 9.63 |

- Eight at a time, v0.4.4 took 5.8 %, 5.8 % and 7.0 % less time per token than
  stock DFlash K8 in vLLM 0.30.0 (rounds 1, 2 and 3); four at a time, 6.7 %, 6.0 % and 3.7 % less; one
  at a time, 4.0 %, 7.8 % and 8.3 % less. On 14k-token documents,
  34.8 %, 43.8 % and 38.0 % less; on 28k, 48.1 %,
  45.5 % and 47.5 % less.
- v0.4.3, whose capture sizes end at 40, runs steps with 6 to 8 requests without a CUDA graph. In an earlier session
  with the v0.4.3 adapter (`v0.4.3-comparison/` in the evidence), v0.4.3 with its recommended settings served
  364 tokens per second with eight requests in flight, less than plain decoding (513), and
  v0.4.4 served 714, 1.96x as much.
- KV cache at 32k context: 75,731 to 75,768 tokens for v0.4.4 with the recommended settings.

## Lossless

v0.4.4 changes which attention kernel computes the target's and the drafter's attention when several requests share a
step, and the adapter, which only proposes tokens; the code that proposes, accepts and samples candidates is unchanged.
Every candidate still goes through vLLM's own rejection sampler against the draft distribution it was drawn from, so the
output distribution is the model's. The new path computes each request's attention with the same kernel and the same
per-request split as the one-request path: a GPU test on the release image checks that the outputs are identical, bit
for bit, to running each request alone, for both Gemma 4 attention layouts, query widths 5, 6 and 8, and 2, 4 and 8
requests at contexts from 300 to 6,000 tokens, plus steps of 2, 4 and 7 requests with one padded request (as a larger
captured CUDA graph runs them) and requests whose rows are a prompt chunk (36 of 36 cases equal); the release's CPU tests
check the same equality in Triton's interpreter and check which steps the new path admits. The v0.4.3 CPU exactness
tests (vLLM's rejection kernels at the served shapes, position by position against the model's distribution) run
unchanged on the v0.4.4 image, and every statistic they print is identical to the v0.4.3 release run. 174 CPU tests pass
(rc 0 in every group), including the launcher's and the startup checks' own tests; the per-test numbers are in the
validation evidence.

## Upgrade and rollback

```bash
docker pull ghcr.io/brntech/vllm-uno:0.4.4
hf download Broadnet/gemma-4-26B-A4B-uno-adapter --local-dir /path/to/adapter
docker run ... -v /path/to/adapter:/adapter:ro -e UNO_PROFILE=gemma4 -e UNO_RECOMMENDED=1 \
  ghcr.io/brntech/vllm-uno:0.4.4 cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit /adapter \
  -- --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4
```

The flags after `--` turn on Gemma 4's tool-call and reasoning parsers, as on the measured servers.
`UNO_DRAFT_MOE_TOPK=4` (part of the recommended settings) runs on compute-capability-8.6 GPUs (RTX 30-series class)
only and refuses to start elsewhere; there, add `-e UNO_DRAFT_MOE_TOPK=`.

Rollback: the v0.4.3 adapter with `hf download Broadnet/gemma-4-26B-A4B-uno-adapter --revision s5`;
`UNO_GEMMA_SPLITKV_MULTI=0` (the v0.4.3 recommended path in the same image); or run `ghcr.io/brntech/vllm-uno:0.4.3`
with its four settings. The Qwen3-8B profile serves as in v0.4.3.
