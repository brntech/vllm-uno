#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
KIT=$(cd -- "$HERE/.." && pwd)
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  echo 'Usage: bash release/build.sh [IMAGE_TAG=vllm-uno:0.1.0rc1]'
  echo 'This recipe targets linux/amd64. BASE_IMAGE must be commit-matched and digest-pinned.'
  echo 'BUILD_DRY_RUN=1 audits and prints the build without invoking Docker.'
  exit 0
fi
if (($# > 1)) || [[ ${1:-} == -* ]]; then
  echo 'Expected one image tag; use --help for usage.' >&2
  exit 2
fi
image=${1:-vllm-uno:0.1.0rc1}
base=${BASE_IMAGE:-vllm/vllm-openai@sha256:89dd8f442a3f4c08c6b3cd634c4f735cd709160651c296596673cf974ea6ee39}
[[ $base =~ @sha256:[0-9a-f]{64}$ ]] || { echo 'BASE_IMAGE must pin a complete sha256 digest.' >&2; exit 2; }
[[ ${PLATFORM:-linux/amd64} == linux/amd64 ]] || { echo 'This release recipe supports linux/amd64 only.' >&2; exit 2; }
py=${PYTHON:-python3}
"$py" "$HERE/check.py"
cmd=(docker build --platform linux/amd64 --build-arg "BASE_IMAGE=$base" --tag "$image" --file release/Dockerfile -)
printf '%q ' "${cmd[@]}"; printf '\n'
[[ ${BUILD_DRY_RUN:-0} != 1 ]] || exit 0
"$py" "$HERE/bundle.py" --stdout | "${cmd[@]}"
