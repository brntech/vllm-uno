#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Linux/Bash + Python 3. The default profile uses Qwen3-8B BF16.
set -euo pipefail
if [[ ${1:-} == --help ]]; then
  cat <<'HELP'
Usage: bash release/serve.sh [MODEL [LOCAL_ADAPTER_OR_HF_REPO]] [-- VLLM_ARGS...]
Defaults: Qwen/Qwen3-8B; s-sahoo/uno-qwen3-8B (adapter/ subdirectory).
Environment: UNO_K=8, UNO_MASK_TOKEN_ID=151669, UNO_NOISE_SEED=0,
  MODEL_REVISION / UNO_ADAPTER_REVISION (recorded Qwen pins by default),
  SERVED_MODEL_NAME=uno-qwen3-8b, HOST=0.0.0.0, PORT=8000, PYTHON=python3.
UNO_DRY_RUN=1 prints the command without importing vLLM/downloading weights.
Extra flags after -- override defaults; re-gate any changed configuration.
HELP
  exit 0
fi
model=Qwen/Qwen3-8B
adapter=s-sahoo/uno-qwen3-8B
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
spec=$("$py" - "$adapter" "${UNO_K:-8}" "${UNO_MASK_TOKEN_ID:-151669}" "${UNO_NOISE_SEED:-0}" <<'PY'
import json,sys
k=int(sys.argv[2])
if k < 1: raise SystemExit('UNO_K must be positive')
mask_token_id=int(sys.argv[3])
if mask_token_id <= 1: raise SystemExit('UNO_MASK_TOKEN_ID must be greater than 1')
noise_seed=int(sys.argv[4])
c=dict(method='uno', uno_lora_path=sys.argv[1],
       uno_mask_token_id=mask_token_id, uno_noise_seed=noise_seed,
       num_speculative_tokens=k)
print(json.dumps(c,separators=(',',':')))
PY
)
cmd=(env VLLM_USE_V2_MODEL_RUNNER=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_LORA_ENABLE_DUAL_STREAM=1
  "$py" -m vllm.entrypoints.cli.main serve "$model"
  --served-model-name "${SERVED_MODEL_NAME:-uno-qwen3-8b}"
  --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
  --attention-backend FLASH_ATTN --attention-config '{"flash_attn_version":2}' --enable-prefix-caching
  --tensor-parallel-size 1 --api-server-count 2
  --max-model-len 4096 --max-num-seqs 32 --max-num-batched-tokens 8192
  --gpu-memory-utilization 0.90 --trust-remote-code --dtype bfloat16
  --async-scheduling --enable-lora --max-lora-rank 128 --max-loras 2
  --generation-config vllm --speculative-config "$spec")
[[ -z $model_rev ]] || cmd+=(--revision "$model_rev")
cmd+=("$@")
printf 'Uno launch (adapter_revision=%s): ' "$adapter_rev" >&2
printf '%q ' "${cmd[@]}" >&2
printf '\n' >&2
if [[ ${UNO_DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${cmd[@]}"; printf '\n'
  exit 0
fi
exec "${cmd[@]}"
