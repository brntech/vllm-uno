#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Linux/Bash + Python 3. Profiles: qwen3 (Qwen3-8B BF16) and gemma4 (Gemma 4 26B A4B AWQ).
set -euo pipefail
if [[ ${1:-} == --help ]]; then
  cat <<'HELP'
Usage: bash release/serve.sh [MODEL [LOCAL_ADAPTER_OR_HF_REPO]] [-- VLLM_ARGS...]
Profiles, chosen by UNO_PROFILE:
  qwen3 (default) Qwen/Qwen3-8B BF16, K=8, s-sahoo/uno-qwen3-8B (adapter/ subdirectory),
        FlashAttention 2, max_num_seqs=16, explicit 2 GiB KV cache,
        capture sizes [1,2,4,8,16,32,64,128,144].
  gemma4 Gemma 4 26B A4B AWQ, K=4, language-only, TRITON_ATTN, max_num_seqs=4,
        max_model_len=8192, gpu_memory_utilization 0.85, capture sizes
        [1,2,3,4,5,6,7,8,13,14,15,16]. Pass the model and the local adapter directory.
Environment: UNO_PROFILE, UNO_K, UNO_MASK_TOKEN_ID, UNO_NOISE_SEED, UNO_NOISE_LOW,
  MODEL_REVISION / UNO_ADAPTER_REVISION (recorded Qwen pins by default),
  SERVED_MODEL_NAME=uno-qwen3-8b, HOST=0.0.0.0, PORT=8000, PYTHON=python3.
  Engine-side switches read by the runner: UNO_GEMMA_SPLITKV=1 (split-KV
  opt-in) and UNO_DRAFT_MOE_TOPK=4 (draft MoE top-k opt-in; gemma4 only).
UNO_DRY_RUN=1 prints the command without importing vLLM/downloading weights.
Extra flags after -- override defaults; re-gate any changed configuration.
HELP
  exit 0
fi
profile=${UNO_PROFILE:-qwen3}
case $profile in
  qwen3)
    model=Qwen/Qwen3-8B
    adapter=s-sahoo/uno-qwen3-8B
    served=uno-qwen3-8b
    k=${UNO_K:-8}
    mask=${UNO_MASK_TOKEN_ID:-151669}
    seed=${UNO_NOISE_SEED:-0}
    noise_low=${UNO_NOISE_LOW:-}
    flags=(--attention-backend FLASH_ATTN --attention-config '{"flash_attn_version":2}' --enable-prefix-caching
      --max-model-len 4096 --max-num-seqs 16 --max-num-batched-tokens 2048
      --kv-cache-memory-bytes 2147483648 --dtype bfloat16
      --enable-lora --max-lora-rank 128 --max-loras 2
      --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64,128,144]}')
    ;;
  gemma4)
    model=cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit
    adapter=
    served=uno-gemma4-26b-a4b
    k=${UNO_K:-4}
    mask=${UNO_MASK_TOKEN_ID:-262144}
    seed=${UNO_NOISE_SEED:-29}
    noise_low=${UNO_NOISE_LOW:-0}
    flags=(--dtype bfloat16 --attention-backend TRITON_ATTN --language-model-only
      --disable-hybrid-kv-cache-manager --enable-prefix-caching --seed 29
      --max-model-len 8192 --max-num-seqs 4 --max-num-batched-tokens 2048
      --gpu-memory-utilization 0.85
      --enable-lora --lora-dtype bfloat16 --max-lora-rank 16 --max-loras 1 --max-cpu-loras 1
      --lora-target-modules qkv_proj o_proj gate_up_proj down_proj
      --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,13,14,15,16]}')
    ;;
  *)
    echo 'UNO_PROFILE must be qwen3 or gemma4.' >&2
    exit 2
    ;;
esac
if (($#)) && [[ $1 != --* ]]; then model=$1; shift; fi
if (($#)) && [[ $1 != --* ]]; then adapter=$1; shift; fi
if [[ ${1:-} == -- ]]; then shift; fi
py=${PYTHON:-python3}
model_rev=${MODEL_REVISION:-}
adapter_rev=${UNO_ADAPTER_REVISION:-}
if [[ $model == Qwen/Qwen3-8B && -z $model_rev ]]; then
  model_rev=b968826d9c46dd6066d109eabc6255188de91218
fi
if [[ $adapter == s-sahoo/uno-qwen3-8B && -z $adapter_rev ]]; then
  adapter_rev=8819e09ac901e7290d8d89d62c98b9f756c602fe
fi
if [[ $profile == gemma4 && -z $adapter ]]; then
  echo 'The gemma4 profile needs an adapter: pass the local adapter directory as the second argument.' >&2
  exit 2
fi
if [[ ${UNO_DRY_RUN:-0} != 1 ]]; then
  adapter=$("$py" - "$adapter" "$adapter_rev" <<'PY'
import pathlib, sys
root = pathlib.Path(sys.argv[1]).expanduser()
if not root.is_dir():
    from huggingface_hub import snapshot_download
    root = pathlib.Path(snapshot_download(sys.argv[1], revision=sys.argv[2] or None,
                        allow_patterns=['adapter/*', 'adapter_config.json',
                                        'adapter_model.safetensors', 'adapter_model.bin']))
if not (root / 'adapter_config.json').is_file() and (root / 'adapter/adapter_config.json').is_file():
    root /= 'adapter'
if not (root / 'adapter_config.json').is_file():
    raise SystemExit(f'No adapter_config.json found at {root} or its adapter/ directory')
print(root.resolve())
PY
)
elif [[ $adapter == s-sahoo/uno-qwen3-8B ]]; then
  adapter="<HF_CACHE>/models--s-sahoo--uno-qwen3-8B/snapshots/$adapter_rev/adapter"
fi
spec=$("$py" - "$adapter" "$k" "$mask" "$seed" "$noise_low" <<'PY'
import json,sys
k=int(sys.argv[2])
if k < 1: raise SystemExit('UNO_K must be positive')
mask_token_id=int(sys.argv[3])
if mask_token_id <= 1: raise SystemExit('UNO_MASK_TOKEN_ID must be greater than 1')
noise_seed=int(sys.argv[4])
c=dict(method='uno', uno_lora_path=sys.argv[1],
       uno_mask_token_id=mask_token_id, uno_noise_seed=noise_seed,
       num_speculative_tokens=k)
if sys.argv[5]:
    noise_low=int(sys.argv[5])
    if not 0 <= noise_low < mask_token_id: raise SystemExit('UNO_NOISE_LOW must satisfy 0 <= UNO_NOISE_LOW < UNO_MASK_TOKEN_ID')
    c['uno_noise_low']=noise_low
print(json.dumps(c,separators=(',',':')))
PY
)
cmd=(env VLLM_USE_V2_MODEL_RUNNER=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_LORA_ENABLE_DUAL_STREAM=1
  "$py" -m vllm.entrypoints.cli.main serve "$model"
  --served-model-name "${SERVED_MODEL_NAME:-$served}"
  --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
  --tensor-parallel-size 1 --api-server-count 1
  --async-scheduling --enable-log-requests --jit-monitor-verbose
  --generation-config vllm "${flags[@]}"
  --speculative-config "$spec")
[[ -z $model_rev ]] || cmd+=(--revision "$model_rev")
cmd+=("$@")
printf 'Uno launch (profile=%s adapter_revision=%s): ' "$profile" "$adapter_rev" >&2
printf '%q ' "${cmd[@]}" >&2
printf '\n' >&2
if [[ ${UNO_DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${cmd[@]}"; printf '\n'
  exit 0
fi
exec "${cmd[@]}"
