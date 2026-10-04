# Uno for vLLM v0.4.2

Uno for vLLM v0.4.2 ships a new Gemma 4 26B A4B adapter, trained on 28,560 open prompts, nearly three times the
10,000 of the previous one. With it the Gemma 4 profile runs BroadNet's production traffic **1.57x faster than plain
decoding**, matching the DFlash drafter (K=8, 1.56x in the same session), and decodes long prompts from 2k to 28k
tokens **1.31x to 1.49x faster than plain and 1.29x to 1.98x faster than DFlash**. The output is the model's own.

On a single RTX 3090 at 32k context that is **216 tokens per second per request on production traffic** (DFlash 215,
plain 137) and **137 tokens per second on a 28k-token prompt** (DFlash 69, plain 105).

The image does not change: v0.4.2 runs on `ghcr.io/brntech/vllm-uno:0.4.1`
(`sha256:f679879cada7b90d658246305801f2e9316e6f21f75cb7c3d7cd8fa37642ca34`), with the same patch series, profile and
64k draft vocabulary.

## What changed

- **New adapter** at [Broadnet/gemma-4-26B-A4B-uno-adapter](https://huggingface.co/Broadnet/gemma-4-26B-A4B-uno-adapter/tree/s3)
  (Hub commit `90e7186d225d7d0c5746e6f84ba5280e6d3077a0`). The previous 10k-prompt adapter, then three more stages of
  new open rows (5,000, 5,000 and 8,560), each continuing from the one before; the published checkpoint is the last
  stage's step 17,120. Same rank-16 shape and the same `adapter_config.json`, so it drops in where the old one was.
  `adapter_model.safetensors` sha256 `abfdde5cfefad7bbbd8525ca38da4da2464b07aa0724beb65dca1fa7f592f2ee` (75,627,016
  bytes).
- Tokens per cycle keep rising with more open prompts: 3.71 with 10,000, 3.76 with 20,000 and 3.78 with 28,560, same
  card, same recipe.

## Measured

One RTX 3090, the v0.4.1 image, K=4; plain decoding, the previous adapter and DFlash K=8 in the same session on the same
image, two servers each. Production traffic: 72 real requests, one at a time, milliseconds per output token on the HTTP
clock.

| | v0.4.2 adapter | 10k-prompt adapter | DFlash K=8 | plain |
| --- | --- | --- | --- | --- |
| ms per output token (two servers) | 4.657 / 4.619 | 4.719 / 4.705 | 4.663 / 4.649 | 7.276 / 7.281 |
| speed-up over plain | **1.569x** | 1.545x | 1.563x | 1.00x |
| tokens per cycle (τ) | 3.76 / 3.80 | 3.70 / 3.72 | 3.25 / 3.25 | 1 |

Long prompts (open documents, median decode time per output token; Uno 16 requests per length over two servers, plain
and DFlash K=8 8 each, same session and card):

| prompt tokens | 2k | 6k | 10k | 14k | 20k | 28k |
| --- | --- | --- | --- | --- | --- | --- |
| Uno over plain | 1.49x | 1.42x | 1.41x | 1.36x | 1.32x | 1.31x |
| DFlash over plain | 1.15x | 0.96x | 0.94x | 0.75x | 0.72x | 0.66x |
| Uno over DFlash | 1.29x | 1.47x | 1.50x | 1.81x | 1.83x | **1.98x** |

Lossless: Uno-versus-plain distances are the same size as plain-versus-plain ones (24 short production prompts, 96
samples each, two Uno and three plain servers: mean total variation 0.057 over the six Uno-versus-plain pairs against
0.056 over the three plain-versus-plain pairs; ranges 0.048 to 0.068 and 0.053 to 0.059). Details and raw files in
[docs/validation.md](docs/validation.md) and `evidence/release-0.4.2/`.

## Upgrade

Keep `ghcr.io/brntech/vllm-uno:0.4.1` and download the adapter again; the run command is unchanged. Since v0.4.3 the
Hub's main branch holds a newer adapter; this release's adapter is at revision `s3`:

```bash
hf download Broadnet/gemma-4-26B-A4B-uno-adapter --revision s3 --local-dir /path/to/uno-adapter
```

The previous 10k-prompt adapter stays available at revision `p10k`:

```bash
hf download Broadnet/gemma-4-26B-A4B-uno-adapter --revision p10k --local-dir /path/to/p10k-adapter
```

then mount the downloaded directory as `/adapter`.
