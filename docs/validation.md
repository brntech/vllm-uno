# Validation

The v0.1.0 record includes source and package tests and serving checks on Ampere and ARM64 Blackwell. This page also provides matched plain/Uno comparison tools for evaluating another configuration.

## Public package checks

The consolidated public source passed **368 CPU tests across the four focused
modules listed below**, with no skips under Python 3.12 in an isolated Linux container. The **16 packaging and HTTP-gate
regressions** cover altered patch checksums, missing or unsafe assets, deterministic
archives, and failed or malformed mixed-load background requests.

A fresh exact-base checkout reconstructed the tree recorded in the
[patch manifest](../release/series.json); applying it again verified the
whole staged tree as a no-op. These checks require no GPU.

## Published containers

Use `ghcr.io/brntech/vllm-uno:0.1.0`; Docker selects AMD64 or ARM64.
The [release](https://github.com/brntech/vllm-uno/releases/tag/v0.1.0) records
per-platform digests, the source commit, hardware configuration and actual test
outcomes. Build and hardware records describe their exact tested artifacts.

In the record, `final_image_id` is the value reported by each build host's Docker
engine. Use the explicit `manifest_digest` and `config_digest` fields when
verifying registry content.

## Recorded evidence

| Check | Recorded result | What it establishes |
|---|---|---|
| Source-level CPU suite | **PASS:** 368 tests, no skips | Core Uno state, draft-graph, configuration, and LoRA-overlap behavior under CPU models and stand-ins |
| Linux AMD64 v0.1.0 image build | **PASS:** all 17 checked compiled libraries preserved | The Python overlay retained the commit-matched binary runtime |
| RTX 3090 v0.1.0 image integration | **PASS:** compiled CUDA, model and adapter load, HTTP health, greedy and sampled completion, streaming, shared-prefix reuse, and concurrent batches of 8 and 32 requests | The supported Qwen/Uno profile executed on Ampere across the tested request modes |
| Uno engagement on RTX 3090 | **PASS:** 517 drafts, 4,136 draft tokens and 1,361 accepted tokens; graph replay and seed-row reuse observed | Uno generated and accepted candidate tokens during the final-image run |
| GB10 ARM64 v0.1.0 image integration | **PASS:** compiled CUDA, model and adapter load, HTTP health, greedy and sampled completion, SSE streaming, shared-prefix reuse, and concurrent batches of 8 and 32 requests | The supported Qwen/Uno profile executed on ARM64 Blackwell across the tested request modes |
| Uno engagement on GB10 | **PASS:** 102 drafts, 816 draft tokens, 190 accepted tokens; graph replay and seed-row reuse observed at batch 32 | Uno generated and accepted candidate tokens during the ARM64 run |

The GPU runs used runtime built at commit `c4175a5397578f3761152c0463be62286a47f089`. The released images retain that runtime and include updated documentation. The RTX 3090 used the supported profile with FlashAttention 2. The GB10 used the same model profile on ARM64 with `--gpu-memory-utilization 0.30` so the server could coexist with other loaded models.

The H100 numbers in the [paper](https://doi.org/10.5281/zenodo.22652610) are historical measurements from a separate research configuration. Versioned release records identify validation performed on each newly built image, including the separate AMD64/RTX 3090 and ARM64/GB10 hardware records.

The recorded hardware checks exercise live serving. Matched statistical and
strict greedy reference comparisons were not rerun for v0.1.0; the procedures
below are available for those comparisons.

## Run the standard-library package checks

From the repository root:

```bash
python3 release/check.py
```

This checks release structure and metadata without third-party dependencies or a GPU. The focused CPU suite and GPU integration checks are separate procedures below.

To reproduce the recorded focused CPU suite from a reconstructed vLLM source tree, use its test environment and run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
python -m pytest tests/v1/spec_decode/test_uno_seedrow.py \
  tests/v1/spec_decode/test_uno_draftgraph.py \
  tests/v1/spec_decode/test_uno_draftgraph_config.py \
  tests/v1/spec_decode/test_uno_lora_overlap_config.py \
  --noconftest -p no:cacheprovider -q --tb=short
```

## Install gate dependencies

Use Python 3.12 in a clean environment for the HTTP and statistical gates:

```bash
python3 -m venv .venv-gates
source .venv-gates/bin/activate
python3 -m pip install -r release/requirements-gates.txt
```

The verifier starts or stops no server. It records results from already-running endpoints into a new output directory and refuses to overwrite an existing run.

## Capture a matched plain reference

Use the same image, model revision, BF16 precision, attention backend, prefix caching, asynchronous scheduling, API-process count, context, admission limit, and prefill budget as the Uno candidate. Enable LoRA support with the same rank and slot capacity, but do not pass `--speculative-config` and do not apply the Uno adapter to requests.

For Ampere, append `--attention-config '{"flash_attn_version":2}'` to both launches. A plain launch from the published image can override its Uno entrypoint:

```bash
docker run --rm --name vllm-uno-reference --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v vllm-uno-hf-cache:/root/.cache/huggingface \
  --entrypoint python3 ghcr.io/brntech/vllm-uno:0.1.0 \
  -m vllm.entrypoints.cli.main serve Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --served-model-name uno-qwen3-8b \
  --host 0.0.0.0 --port 8000 \
  --attention-backend FLASH_ATTN --enable-prefix-caching \
  --tensor-parallel-size 1 --api-server-count 2 \
  --max-model-len 4096 --max-num-seqs 32 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 --trust-remote-code --dtype bfloat16 \
  --async-scheduling --enable-lora --max-lora-rank 128 --max-loras 2 \
  --generation-config vllm
```

Record the exact command, image digest, GPU, model revision, and relevant environment in `reference-launch.txt`. Once the endpoint is healthy, capture the reference:

```bash
bash release/verify.sh reference \
  --url http://127.0.0.1:8000 \
  --model uno-qwen3-8b \
  --out runs/reference \
  --provenance reference-launch.txt
```

Reference mode freezes prompt token IDs, captures strict greedy outputs, collects two independent plain samples, and establishes a plain-versus-plain statistical floor. Speculation counters must remain unchanged.

## Run the Uno candidate

Stop the reference server, then start the candidate on the same GPU and port. For Ampere:

```bash
docker run --rm --name vllm-uno-candidate --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v vllm-uno-hf-cache:/root/.cache/huggingface \
  ghcr.io/brntech/vllm-uno:0.1.0 \
  Qwen/Qwen3-8B s-sahoo/uno-qwen3-8B -- \
  --attention-config '{"flash_attn_version":2}'
```

Record the exact candidate command, image digest, GPU, model and adapter revisions, and environment in `candidate-launch.txt`. Then run:

```bash
bash release/verify.sh candidate \
  --url http://127.0.0.1:8000 \
  --model uno-qwen3-8b \
  --reference runs/reference \
  --out runs/candidate \
  --provenance candidate-launch.txt
```

Candidate mode verifies the tokenizer against the frozen reference IDs, compares 256-token greedy outputs, runs sampled chunk-1 and mixed chunk-8 comparisons, and confirms that speculation counters advanced. Statistical comparison uses the recorded plain floor, family-wise adjustment, and sufficient permutation resolution.

## Interpret the result

The candidate passes only when all of the following pass together:

- strict token-ID equality for the greedy prompt set
- sampled comparison against the plain reference
- mixed chunk-8 comparison against the plain reference
- evidence that Uno drafting actually ran

If the requested permutation count cannot resolve the adjusted threshold, the verifier increases the permutation budget while keeping the acceptance rule fixed. A statistical failure remains a failure.

A greedy difference remains a failure even when the sampled gates pass. Preserve the reference and candidate outputs, commands, image digests, and logs for a position-level investigation. Report a sampled pass with the tested prompts, positions, sample budget, and configuration.

For release reporting, retain the complete `runs/reference` and `runs/candidate` directories together with server logs showing the configured Uno layout, private-graph replay, seed-row reuse, and any fallback reason.
