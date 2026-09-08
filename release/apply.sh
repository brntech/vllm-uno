#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Applies to the index/worktree; creates no commits.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  echo 'Usage: bash release/apply.sh [NEW_OR_CLEAN_CHECKOUT]'
  echo 'Default: release/.work/vllm. Requires Git and Python 3.'
  exit 0
fi
if (($# > 1)) || [[ ${1:-} == -* ]]; then
  echo 'Expected one checkout path; use --help for usage.' >&2
  exit 2
fi
exec "${PYTHON:-python3}" "$HERE/stack.py" apply "${1:-$HERE/.work/vllm}"
