# Uno for vLLM v0.4.0

Uno for vLLM v0.4.0 makes Gemma 4 26B A4B, the model BroadNet runs in production for its clients in English and
Arabic, the fastest Gemma 4 26B on long prompts we have measured: **1.3x to 1.8x faster than DFlash**, the fastest
Gemma 4 drafter we found, at every length from 2k to 28k tokens, and **1.52x faster than plain decoding** on our
production traffic, on a single 24 GB RTX 3090 with the new
[P10K adapter](https://huggingface.co/Broadnet/gemma-4-26B-A4B-uno-adapter). The output is the model's own. The
drafter now runs on Gemma 4's own hybrid KV layout, scores a 64k-token draft vocabulary while verification keeps the
full one, and the verify pass skips the adapter branch.

**Why Gemma 4 26B A4B.** It is the model behind BroadNet's production agent workloads: email triage and drafting,
service-health monitoring and incident triage, multi-agent coordination, code review, in English and Arabic. The
Gemma 4 family leads the open models in our published Arabic benchmark, on the strength of dialect-authentic
generation, and this model is the base of our Arabic-tuned models. Every speed number here is measured on 72 real
requests from that traffic, replayed on our own hardware, not on a synthetic benchmark; the adapter itself is trained
on open data only.

The package is v0.3.0 plus one patch (`0003-uno-hybrid-kv-draft-vocab.patch`: three vLLM files and their tests) on the same digest-pinned
vLLM `00972dfd72988942138a7a6089eaee08580210b8` CI image; the reconstructed release tree is
`4aa655488f2c8d86fcc3692b037e03a991dcc9ba`.

## What changed

- **Hybrid-KV drafting.** v0.3.0 refused any KV layout other than one full-attention group, so Gemma 4 ran with
  `--disable-hybrid-kv-cache-manager`, stored every layer's cache for the whole context and could not start at 32k on a
  24 GB card. The drafter now accepts several KV groups when every layer spec is full attention or a plain sliding
  window, and one Triton launch writes the draft rows' slot mappings for the extra groups. Chunked, circular and
  compressed layouts, and specs whose `tokens_per_state` is not 1, are still refused by name at startup.
- **Draft vocabulary (`UNO_DRAFT_VOCAB`).** Given a JSON list of token ids, the draft pass scores only those rows of the
  tied LM head and samples from a full-vocabulary buffer that is minus infinity elsewhere; that buffer is also the
  proposal distribution handed to verification, which scores the full vocabulary, so rejection sampling stays exact.
  The release ships a 65,536-id Gemma 4 list ranked on open data and turns it on in the Gemma 4 profile.
- **Inactive-adapter bypass.** When no token in a batch carries an adapter (every verify pass), the dual-stream LoRA
  path runs the base layer alone instead of launching and zero-filling the auxiliary branch.
- **Gemma 4 profile:** split-KV draft attention on by default (`UNO_GEMMA_SPLITKV=0` turns it off), 32k context
  (`UNO_MAX_MODEL_LEN` overrides), hybrid KV cache manager on, `max_num_seqs=8`,
  `gpu_memory_utilization=0.90`, CUDA graphs covering 32 draft rows and 40-row verify batches. At 32k the Gemma 4 profile holds 79k tokens of KV
  (plain 101k, the DFlash drafter 68k with the same serving flags).

## Fixed, and a known issue in v0.3.0

With prefix caching on (the default in both profiles since v0.3.0), a new request whose cached prefix left exactly one
prompt token to compute could come back as `<pad>` tokens: an exact repeat of a prompt whose length is one more than a
multiple of the KV block size (a retry, `n > 1`, a benchmark loop). vLLM's scheduler padded that one-token prefill like
a resumed decode, with placeholder draft tokens, and Uno's probabilistic verification accepted them. Ordinary chat
traffic rarely meets the condition. v0.4.0 fixes it in the scheduler, with a regression test; on v0.3.0, upgrade or add
`--no-enable-prefix-caching`.

## What this release validates

`docs/validation.md` carries the runs; the short version, one RTX 3090, P10K adapter, K=4. "Final" rows ran on the
release image; "first" rows on the first release image, which lacks only the two fixes above:

| check | result |
| --- | --- |
| Production traffic (72 real requests, one at a time), final | 4.934 / 4.905 ms per token vs plain 7.489: **1.52x** |
| Long prompts, 2k / 6k / 10k / 14k / 20k / 28k tokens, first (median decode time; Uno 16 requests per length over two servers, plain and DFlash 8) | **1.47x / 1.35x / 1.36x / 1.33x / 1.28x / 1.26x** plain (DFlash 1.14x / 0.94x / 0.86x / 0.75x / 0.74x / 0.69x) |
| Greedy replays vs plain servers (72 requests), first | 61 and 57 diverge; plain vs plain 60 |
| Sampled distributions, short prompts, prefix caching on (mean TV), final Uno vs first-image plain servers | 0.056 to 0.067; plain vs plain 0.048 to 0.062, two halves of one server 0.060 to 0.073 |
| Sampled distributions, 3k-14k documents (mean TV), first | 0.222 to 0.246; plain vs plain 0.233 to 0.245 |

## Scope

One GPU type (RTX 3090, AMD64). The production number is one request at a time; with several users sending long
prompts at once, prompt processing dominated and plain decoding kept pace with Uno and DFlash in our concurrent
long-prompt test. The Qwen3-8B
profile is unchanged from v0.3.0.

## Credits

Uno is the work of IFM and the [Uno authors](https://arxiv.org/abs/2609.04010). Gemma 4 is by Google; the AWQ checkpoint
is by cyankiwi; the DFlash comparator is by z-lab. This implementation and its measurements are by the BroadNet Research
Team.
