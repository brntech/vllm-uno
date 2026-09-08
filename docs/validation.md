# Validation

This release separates implementation evidence from stronger statistical and exact-output claims. The existing record establishes that the Uno source assembles, its focused CPU checks pass, a Linux AMD64 image preserves the pinned runtime, and the selected path executes on an RTX 3090. The repository also includes matched reference/candidate gates for users who need to validate a particular image, GPU, or configuration.

## Public package checks

The consolidated public source passed all **368 focused CPU tests** with no skips
under Python 3.12 in an isolated Linux container. The **16 packaging and HTTP-gate
regressions** cover altered patch checksums, missing or unsafe assets, deterministic
archives, and failed or malformed mixed-load background requests.

A fresh exact-base checkout reconstructed the tree recorded in the
[patch manifest](../release/series.json); applying it again verified the
whole staged tree as a no-op. These checks require no GPU.

## Published container

The prebuilt Linux AMD64 image is `ghcr.io/brntech/vllm-uno:0.1.0rc1`.
Its digest and container verification record are attached to the
[release](https://github.com/brntech/vllm-uno/releases/tag/v0.1.0rc1).
The image contains the exact tagged source kit; the current repository README
also documents the subsequent addition of registry distribution.

Package-content, compiled-library, import and offline-launch checks ran without
GPU access. The published image's model generation was not rerun on a GPU;
the prior integration evidence below remains accurately scoped to its runtime.

## Recorded evidence

| Check | Recorded result | What it establishes |
|---|---|---|
| Source-level CPU suite | **PASS:** 368 tests, no skips | Core Uno state, draft-graph, configuration, and LoRA-overlap behavior under CPU models and stand-ins |
| Linux AMD64 development image build | **PASS:** all 17 checked compiled libraries preserved | The corresponding Python overlay retained the commit-matched binary runtime |
| RTX 3090 bounded integration | **PASS:** compiled CUDA import, CUDA tensor, model and adapter load, HTTP health, and a 256-token completion | The corresponding runtime implementation executed the supported Qwen/Uno path on Ampere |
| Uno engagement on RTX 3090 | **PASS:** drafting counters advanced; private draft graph replay and seed-row reuse were observed | The response did not silently fall back to plain autoregressive generation |
| Final sampled-distribution comparison | Not recorded as a pass for `0.1.0rc1` | Run the reference/candidate procedure below for the image and configuration being evaluated |
| Final greedy reference comparison | Not recorded as a pass for `0.1.0rc1` | Greedy output differences remain failures and require retained artifacts plus investigation |

The RTX 3090 run used frozen source tree `73b0a21b198e2b6af29240f4dcef7cd3140c3996`. The public release consolidates and sanitizes that work; its current reconstructed tree identity is authoritative in [`release/series.json`](../release/series.json). After documentation, comment, and test-only removals, executable ASTs for all 38 runtime Python files were verified equivalent between the two trees.

The run used Qwen3-8B BF16, `K=8`, BF16 KV cache, prefix caching, two API processes, and FlashAttention 2. It returned HTTP 200, reached the requested 256-token cap, recorded 41 drafts and 328 draft tokens, and observed both graph replay and one-request seed-row reuse. This is an integration result, not a throughput benchmark or a general answer-quality result.

The H100 numbers in the [paper](https://doi.org/10.5281/zenodo.22652610) are historical measurements from a separate research configuration. They are not substituted for validation of a newly built image. ARM64/Blackwell is likewise a separate distribution target and is not covered by this AMD64 release.

## Run the standard-library package checks

From the repository root:

```bash
python3 release/check.py
```

This checks release structure and metadata without third-party dependencies. It does not run the source-level pytest suite, require a GPU, or establish CUDA execution.

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
  --entrypoint python3 ghcr.io/brntech/vllm-uno:0.1.0rc1 \
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
  ghcr.io/brntech/vllm-uno:0.1.0rc1 \
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

An ordinary statistical failure is not retried into a pass. If the requested permutation count cannot resolve the adjusted threshold, the verifier increases the permutation budget; this changes resolution, not the acceptance rule.

A greedy difference remains a failure even when the sampled gates pass. Preserve the reference and candidate outputs, commands, image digests, and logs for a position-level investigation. A sampled pass describes the tested prompts, positions, sample budget, and configuration; it does not prove equality for every possible input or runtime setting.

For release reporting, retain the complete `runs/reference` and `runs/candidate` directories together with server logs showing the configured Uno layout, private-graph replay, seed-row reuse, and any fallback reason.
