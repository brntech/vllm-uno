#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# run_rocm_cpu.sh IMAGE REPO DIFF OUT -- the v0.4.4 release CPU suites on the ROCm image, on any Linux host with Docker
# (no GPU used). REPO = a git checkout of this repository: its tracked tests/v0.4.3 and tests/v0.4.4 (git archive HEAD),
# with DIFF (tests/rocm-0.4.4/tests-rocm.diff) applied: three test-side lines, the image's VERSION string and the field
# vLLM v0.30.0's speculative config reads on the startup test double. Selectors = tests/v0.4.4/release_cpu_groups.sh,
# group by group; every group is one CPU-only container (no GPU devices, --network none, TRITON_INTERPRET=1,
# VLLM_LORA_ENABLE_DUAL_STREAM=0 as the launcher sets on gfx12), two at a time. Per group OUT/cpu/NAME.{cmd,log,rc}: the
# rc is pytest's own, captured inside the container that ran it. OUT (must not exist) also gets tests/ (the tree that
# ran), PROGRESS and SUMMARY.txt. Exit 0 only when every group's rc is 0.
#   bash tests/rocm-0.4.4/run_rocm_cpu.sh ghcr.io/brntech/vllm-uno:0.4.4-rocm . tests/rocm-0.4.4/tests-rocm.diff cpu-out
set -uo pipefail
IMG=${1:?usage: run_rocm_cpu.sh IMAGE REPO DIFF OUT}; LIVE=${2:?usage}; DIFF=${3:?usage}; OUT=${4:?usage}
for p in "$LIVE" "$DIFF" "$OUT"; do case "$p" in *[!A-Za-z0-9._/@+-]*|'') echo "REFUSE: non-plain path '$p'"; exit 2 ;; esac; done
[[ ! -e $OUT ]] || { echo "REFUSE: $OUT exists"; exit 2; }
[[ -s $DIFF ]] || { echo "REFUSE: $DIFF missing or empty"; exit 2; }
docker image inspect "$IMG" > /dev/null 2>&1 || { echo "REFUSE: image $IMG not on disk"; exit 2; }
DIFF=$(cd "$(dirname "$DIFF")" && pwd)/$(basename "$DIFF")
mkdir "$OUT" && OUT=$(cd "$OUT" && pwd) || exit 2
LIVE=$(cd "$LIVE" && pwd)
exec < /dev/null
mkdir "$OUT/cpu" "$OUT/src"
git -C "$LIVE" archive HEAD tests/v0.4.3 tests/v0.4.4 | tar -x -C "$OUT/src" || { echo "tests archive failed"; exit 1; }
sha256sum "$DIFF" > "$OUT/tests-diff.sha256"
patch -p1 --forward --batch -d "$OUT/src/tests" < "$DIFF" > "$OUT/tests-patch.log" 2>&1 \
  || { echo "tests diff did not apply"; exit 1; }
mv "$OUT/src/tests" "$OUT/tests" && rmdir "$OUT/src"
[ -z "$(find "$OUT/tests" -name '*.orig' -o -name '*.rej')" ] || { echo "patch left .orig/.rej"; exit 1; }
SLOW='point_mass_lookup_is_exact or negative_control or wide_ or mixed_batch_gated or gated_request_never_reads'
SUITE_GROUPS=(  # not GROUPS: bash's own GROUPS array ignores assignment
  "fast|4|v0.4.4||test_release_v044.py test_startup_refusal_v044.py"
  "fast043|4|v0.4.3|not ($SLOW)|test_plookup.py test_plgate.py"
  "kernel|2|v0.4.4||test_splitkv_multi_cpu.py"
  "t1|2|v0.4.3|wide_temperature1_k4|test_plookup.py"
  "t07|2|v0.4.3|wide_temperature07_top_p_k4|test_plookup.py"
  "exact|2|v0.4.3|point_mass_lookup_is_exact or negative_control or wide_greedy_k4 or wide_posctl|test_plookup.py"
  "gate|2|v0.4.3|mixed_batch_gated or gated_request_never_reads|test_plgate.py"
  "k5-t1|2|v0.4.3|wide_temperature1_k5|test_recipe_k5.py"
  "k5-t07|2|v0.4.3|wide_temperature07_top_p_k5|test_recipe_k5.py"
  "k5-greedy|2|v0.4.3|wide_greedy_k5|test_recipe_k5.py"
  "k5-mixed|2|v0.4.3|mixed_batch_gated_and_full_requests_are_exact_k5|test_recipe_k5.py"
)
IMGID=$(docker image inspect -f '{{.Id}}' "$IMG")
RUN=$(basename "$OUT")  # in every container name: two runs at once never share a name
if docker ps -a --format '{{.Names}}' | grep -q "^rel-cpu-$RUN-"; then echo "REFUSE: containers rel-cpu-$RUN-* exist"; exit 2; fi
run_group() {  # run_group SPEC: blocks until the group's container exits
  local name cpus suite sel files t f
  IFS='|' read -r name cpus suite sel files <<< "$1"
  t=""; for f in $files; do t="$t /t/$f"; done
  printf 'image=%s id=%s suite=%s sel=%s files=%s cpus=%s\n' "$IMG" "$IMGID" "$suite" "$sel" "$files" "$cpus" > "$OUT/cpu/$name.cmd"
  echo "$(date -u +%FT%TZ) start $name" >> "$OUT/PROGRESS"
  docker run --rm --name "rel-cpu-$RUN-$name" --network none --cpus "$cpus" --memory 8g \
    -e TRITON_INTERPRET=1 -e SEL="$sel" -e TARGETS="$t" -e NAME="$name" -e VLLM_LORA_ENABLE_DUAL_STREAM=0 \
    -v "$OUT/tests/$suite":/t:ro -v "$OUT/cpu":/out --workdir /t --entrypoint bash "$IMG" -c \
    'python3 -m pytest -q -s -p no:cacheprovider $TARGETS -k "$SEL" > /out/$NAME.log 2>&1; rc=$?; echo "rc=$rc" >> /out/$NAME.log; echo $rc > /out/$NAME.rc' \
    > /dev/null 2>&1
  local drc=$?  # captured first: any later command substitution resets $?
  echo "$(date -u +%FT%TZ) end $name docker-rc=$drc pytest-rc=$(cat "$OUT/cpu/$name.rc" 2>/dev/null || echo none)" >> "$OUT/PROGRESS"
}
i=0
for g in "${SUITE_GROUPS[@]}"; do
  run_group "$g" &
  i=$((i + 1))
  if (( i % 2 == 0 )); then wait; fi
done
wait
st=0
{ for g in "${SUITE_GROUPS[@]}"; do
    n=${g%%|*}
    if [[ -f $OUT/cpu/$n.rc ]]; then rc=$(cat "$OUT/cpu/$n.rc"); echo "$n rc=$rc $(grep -E '[0-9]+ (passed|failed)' "$OUT/cpu/$n.log" | tail -1)"; [[ $rc == 0 ]] || st=1
    else echo "$n rc=missing"; st=1; fi
  done
  echo "total passed: $(cat "$OUT"/cpu/*.log | grep -oE '[0-9]+ passed' | awk '{s+=$1} END {print s+0}')"
  echo "verdict rc=$st"; } > "$OUT/SUMMARY.txt"
cat "$OUT/SUMMARY.txt"
exit $st
