# Uno for vLLM v0.4.1

Uno for vLLM v0.4.1 makes Gemma 4 26B A4B decode about 1 % faster on the RTX 3090, with no change to its output: the
Gemma 4 profile now ships tuned kernel configs for the adapter's LoRA layers. In the release session the profile runs
BroadNet's production traffic **1.53x faster than plain decoding** (1.51x with the tuned configs turned off, same image,
same session).

Everything else is v0.4.0: the same patch series, the same reconstructed release tree
(`4aa655488f2c8d86fcc3692b037e03a991dcc9ba`), the same digest-pinned base image and the same
[P10K adapter](https://huggingface.co/Broadnet/gemma-4-26B-A4B-uno-adapter). The vLLM package inside the image is
byte-identical to v0.4.0's.

## What changed

- **Tuned LoRA kernel configs for the RTX 3090.** Uno's draft pass runs the adapter through vLLM's LoRA shrink and
  expand Triton kernels, which used vLLM's generic default configs. v0.4.1 ships configs tuned for every shape the
  Gemma 4 draft pass launches with the rank-16 adapter (shrink over 2,112 to 8,192 inputs, expand to 2,112 to 8,192
  outputs, 4 to 40 rows), chosen by timing each candidate under CUDA graphs, as served, and checking every winner
  against a reference product. At one request (four draft rows) the adapter's kernel time per layer falls from about
  35 to about 24.5 microseconds.
- The profile sets `VLLM_TUNED_CONFIG_FOLDER=/opt/uno-kit/release/lora-configs`. vLLM matches the files by GPU name, so
  any other GPU keeps its default configs. `VLLM_TUNED_CONFIG_FOLDER=` (empty) turns the tuned configs off.

## Measured

Production traffic (72 requests, one at a time, HTTP clock), one RTX 3090, P10K adapter, two servers per arm in
balanced order:

| session | tuned configs | configs off | change | τ (tuned / off) |
| --- | --- | --- | --- | --- |
| release image (v0.4.1 candidate) | 4.970 / 4.921 ms per token | 4.970 / 5.002 | -0.8 % | 3.70 / 3.71 |
| v0.4.0 image + the same config files | 4.882 / 4.875 | 4.942 / 4.923 | -1.1 % | 3.70 / 3.70 |

Tokens per cycle do not move: only the draft pass runs these kernels, and verification skips the adapter branch, so
the emitted distribution is the model's own. The release session's short-prompt distribution check matches plain
decoding as closely as v0.4.0's did (details and receipts in [docs/validation.md](docs/validation.md)).

## Upgrade

`docker pull ghcr.io/brntech/vllm-uno:0.4.1` and run it exactly as v0.4.0. On a GPU other than the RTX 3090 the image
behaves as v0.4.0. The tuning script used for these files is in the release evidence (`tune_lora.py`), for anyone who
wants to generate configs for another card.
