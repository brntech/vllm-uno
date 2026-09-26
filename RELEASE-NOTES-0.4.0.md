# Uno for vLLM v0.4.0

Uno for vLLM v0.4.0 makes Uno a long-context speculator on Gemma 4 26B A4B. The drafter now runs on Gemma 4's own
hybrid KV layout (sliding-window and full-attention groups) instead of forcing every layer into full attention, the
draft head scores a 64k-token Gemma 4 vocabulary while verification keeps the full one, and the verify pass skips the
adapter branch entirely. On one RTX 3090 with the new
[P10K adapter](https://huggingface.co/Broadnet/gemma-4-26B-A4B-uno-adapter) (repo id pending), Gemma 4 26B decodes
**1.53x faster than plain** on production traffic and **1.31x to 1.44x faster on prompts from 2k to 28k tokens**, where
the DFlash drafter measured on the same card falls to 0.66x of plain. The output distribution is the model's own.

The package is v0.3.0 plus one patch (`0003-uno-hybrid-kv-draft-vocab.patch`, two files) on the same digest-pinned
vLLM `00972dfd72988942138a7a6089eaee08580210b8` CI image; the reconstructed release tree is
`e5c6278877bab3a6fdba5a26babdf321c3ce2bdb`.

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
  `gpu_memory_utilization=0.90`, CUDA graphs covering 32 draft rows and 40-row verify batches. At 32k the Gemma 4 profile holds 79k tokens of KV,
  more than the 46k to 69k the DFlash drafter gets on the same card.

## What this release validates

`docs/validation.md` carries the runs; the short version, one RTX 3090, P10K adapter, K=4:

| check | result |
| --- | --- |
| Production traffic (72 real requests, one at a time), release image | 4.926 / 4.910 ms per token vs plain 7.542 / 7.551: **1.53x** |
| Long prompts, 2k / 6k / 10k / 14k / 20k / 28k tokens, release image | **1.44x / 1.40x / 1.37x / 1.37x / 1.31x / 1.31x** plain (DFlash 1.24x to 0.66x) |
| Greedy replays vs plain servers (72 requests) | 56 and 57 diverge; plain vs plain 58 |
| Sampled distributions, short prompts (mean TV) | 0.034 to 0.047; plain vs plain 0.038 to 0.045, two halves of one plain server 0.046 to 0.059 |
| Sampled distributions, 3k-14k documents (mean TV) | 0.197 to 0.224; plain vs plain 0.203 to 0.227 |

## Scope

One GPU type (RTX 3090, AMD64). The production number is one request at a time; with several users sending long
prompts at once, prompt processing dominates and plain decoding keeps pace with any speculative method. The Qwen3-8B
profile is unchanged from v0.3.0.

## Credits

Uno is the work of IFM and the [Uno authors](https://arxiv.org/abs/2609.04010). Gemma 4 is by Google; the AWQ checkpoint
is by cyankiwi; the DFlash comparator is by z-lab. This implementation and its measurements are by the BroadNet Research
Team.
