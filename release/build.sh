#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
KIT=$(cd -- "$HERE/.." && pwd)
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  echo 'Usage: bash release/build.sh [IMAGE_TAG=vllm-uno:0.2.0]'
  echo 'Builds linux/amd64 only.'
  echo 'BASE_IMAGE must be commit-matched and digest-pinned.'
  echo 'BUILD_DRY_RUN=1 audits and prints the build without invoking Docker.'
  exit 0
fi
if (($# > 1)) || [[ ${1:-} == -* ]]; then
  echo 'Expected one image tag; use --help for usage.' >&2
  exit 2
fi
image=${1:-vllm-uno:0.2.0}
base=${BASE_IMAGE:-public.ecr.aws/q9t5s3a7/vllm-ci-postmerge-repo:b87339888d29329c42c42573e34cc2beebdcc48b@sha256:e3ab6a1f24248420800ceeede9f14e605e72f4e2ebd48a91821db8c07708e0f5}
[[ $base =~ @sha256:[0-9a-f]{64}$ ]] || { echo 'BASE_IMAGE must pin a complete sha256 digest.' >&2; exit 2; }
platform=${PLATFORM:-linux/amd64}
if [[ $platform != linux/amd64 ]]; then
  echo 'v0.2.0 is AMD64 only: PLATFORM must be linux/amd64.' >&2
  exit 2
fi
py=${PYTHON:-python3}
"$py" "$HERE/check.py"
cmd=(docker build --platform "$platform" --build-arg "BASE_IMAGE=$base" --tag "$image" --file release/Dockerfile -)
printf '%q ' "${cmd[@]}"; printf '\n'
[[ ${BUILD_DRY_RUN:-0} != 1 ]] || exit 0
"$py" "$HERE/bundle.py" --stdout | "${cmd[@]}"
