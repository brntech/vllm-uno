#!/usr/bin/env bash
# run_release_cpu.sh NAME SELECTOR [FILES...] -- start one CPU-only pytest container on the release image, DETACHED
# (named uno043-cpu-NAME, --cpus ${CPUS:-2}, --network none, no GPU). Poll it with wait_release_cpu.sh; the log and
# the real pytest exit code land in ../results/cpu/NAME.log and NAME.rc. The exit code is never piped.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
NAME=$1
SEL=$2
shift 2
FILES=("$@")
[[ ${#FILES[@]} -gt 0 ]] || FILES=(test_plookup.py)
OUT=$HERE/../results/cpu
mkdir -p "$OUT"
rm -f "$OUT/$NAME.rc"
export MSYS_NO_PATHCONV=1
TARGETS=""
for f in "${FILES[@]}"; do TARGETS="$TARGETS /t/$f"; done
docker run -d --name "uno043-cpu-$NAME" --network none --cpus "${CPUS:-2}" --memory 8g \
  -e TRITON_INTERPRET=1 -e SEL="$SEL" -e TARGETS="$TARGETS" -e NAME="$NAME" \
  -v "$HERE":/t:ro -v "$OUT":/out --workdir /t --entrypoint bash "${IMG:-vllm-uno:0.4.3}" -c \
  'python3 -m pytest -q -s -p no:cacheprovider $TARGETS -k "$SEL" > /out/$NAME.log 2>&1; rc=$?; echo "rc=$rc" >> /out/$NAME.log; echo $rc > /out/$NAME.rc' \
  > /dev/null
echo "started uno043-cpu-$NAME"
