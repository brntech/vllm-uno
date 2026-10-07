#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# wait_release_cpu.sh NAME... -- one status line per group: running, or its real pytest exit code from NAME.rc.
# Exit 0 only when every named group has finished with rc 0; 1 when any finished non-zero; 3 while any still runs.
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=$HERE/../results/cpu
st=0
for n in "$@"; do
  if [[ -f $OUT/$n.rc ]]; then
    rc=$(cat "$OUT/$n.rc"); echo "$n rc=$rc $(grep -E '[0-9]+ (passed|failed)' "$OUT/$n.log" | tail -1)"
    [[ $rc == 0 ]] || st=1
  elif docker ps -q -f "name=^uno044-cpu-$n$" | grep -q .; then
    echo "$n running"; [[ $st == 1 ]] || st=3
  else
    echo "$n missing (no rc file and no container)"; st=1
  fi
done
exit $st
