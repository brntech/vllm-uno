#!/usr/bin/env bash
# wait_release_cpu.sh SECONDS NAME... -- bounded foreground poll of detached CPU test containers. Returns when every
# named container has exited (or is gone), or after SECONDS. Prints each container's state, the pytest rc file and the
# summary line; removes a container only once it has exited and its rc file exists.
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=$HERE/../results/cpu
limit=$1
shift
t=0
while :; do
  running=0
  for n in "$@"; do
    s=$(docker inspect -f '{{.State.Status}}' "uno043-cpu-$n" 2>/dev/null || echo gone)
    [[ $s == running ]] && running=$((running + 1))
  done
  ((running == 0 || t >= limit)) && break
  sleep 20
  t=$((t + 20))
done
for n in "$@"; do
  s=$(docker inspect -f '{{.State.Status}} exit={{.State.ExitCode}}' "uno043-cpu-$n" 2>/dev/null || echo gone)
  rc=$(cat "$OUT/$n.rc" 2>/dev/null || echo pending)
  echo "$n: container $s pytest_rc=$rc | $(grep -E '[0-9]+ (passed|failed|error)' "$OUT/$n.log" 2>/dev/null | tail -1)"
  if [[ $rc != pending && $s == exited* ]]; then docker rm "uno043-cpu-$n" > /dev/null; fi
done
