#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# run_release_cpu.sh NAME SUITE SELECTOR [FILES...] -- start one CPU-only pytest container on the release image,
# DETACHED (named uno044-cpu-NAME, --cpus ${CPUS:-2}, --network none, no GPU). SUITE is v0.4.4 (this directory) or
# v0.4.3 (the v0.4.3 tests, whose exactness suites run unchanged on the v0.4.4 image: v0.4.4 changes no proposal,
# rejection or sampling code). Poll with wait_release_cpu.sh; the log and the real pytest exit code land in
# ../results/cpu/NAME.log and NAME.rc. The exit code is never piped.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
NAME=$1
SUITE=$2
SEL=$3
shift 3
FILES=("$@")
case $SUITE in
  v0.4.4) DIR=$HERE ;;
  v0.4.3) if [[ -d $HERE/../v0.4.3 ]]; then DIR=$(cd "$HERE/../v0.4.3" && pwd)  # repository layout: tests/v0.4.3
          else DIR=$(cd "$HERE/../../../v0.4.3/validate/tests" && pwd); fi ;;          # release-kit layout
  *) echo "SUITE must be v0.4.4 or v0.4.3" >&2; exit 2 ;;
esac
OUT=$HERE/../results/cpu
mkdir -p "$OUT"
rm -f "$OUT/$NAME.rc"
export MSYS_NO_PATHCONV=1
TARGETS=""
for f in "${FILES[@]}"; do TARGETS="$TARGETS /t/$f"; done
docker run -d --rm --name "uno044-cpu-$NAME" --network none --cpus "${CPUS:-2}" --memory 8g \
  -e TRITON_INTERPRET=1 -e SEL="$SEL" -e TARGETS="$TARGETS" -e NAME="$NAME" \
  -v "$DIR":/t:ro -v "$OUT":/out --workdir /t --entrypoint bash "${IMG:-vllm-uno:0.4.4}" -c \
  'python3 -m pytest -q -s -p no:cacheprovider $TARGETS -k "$SEL" > /out/$NAME.log 2>&1; rc=$?; echo "rc=$rc" >> /out/$NAME.log; echo $rc > /out/$NAME.rc' \
  > /dev/null
echo "started uno044-cpu-$NAME ($SUITE)"
