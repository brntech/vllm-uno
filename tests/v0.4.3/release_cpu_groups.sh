#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# release_cpu_groups.sh WAVE -- start one wave of the v0.4.3 CPU suites on ${IMG:-vllm-uno:0.4.3} through
# run_release_cpu.sh (detached, real rc to ../results/cpu/NAME.rc); poll with wait_release_cpu.sh. The selectors are
# written here so every result in ../results/cpu/SUMMARY.md names the exact tests it ran.
#   collect  -- pytest --collect-only counts per group (no tests run)
#   fast     -- unit tests (all files), CPUS=4
#   wave1    -- t1, t07, exact, gate (K=4 L=2 exactness), CPUS=2 each
#   wave2    -- k5-t1, k5-t07, k5-greedy, k5-mixed (recipe shape), CPUS=2 each
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SLOW='point_mass_lookup_is_exact or negative_control or wide_ or mixed_batch_gated or gated_request_never_reads'
FAST_FILES=(test_plookup.py test_plgate.py test_release_v043.py test_startup_refusal.py)
case ${1:?wave} in
  collect)
    export MSYS_NO_PATHCONV=1
    docker run --rm --network none --cpus 2 --memory 8g -e TRITON_INTERPRET=1 -v "$HERE":/t:ro --workdir /t \
      --entrypoint bash "${IMG:-vllm-uno:0.4.3}" -c "
        for s in 'not ($SLOW)' 'wide_temperature1_k4' 'wide_temperature07_top_p_k4' \
                 'point_mass_lookup_is_exact or negative_control or wide_greedy_k4 or wide_posctl' \
                 'mixed_batch_gated or gated_request_never_reads'; do
          echo \"[\$s] \$(python3 -m pytest -q -p no:cacheprovider --collect-only ${FAST_FILES[*]} -k \"\$s\" 2>/dev/null | tail -1)\"
        done
        echo \"[recipe] \$(python3 -m pytest -q -p no:cacheprovider --collect-only test_recipe_k5.py 2>/dev/null | tail -1)\""
    ;;
  fast) CPUS=4 bash "$HERE/run_release_cpu.sh" fast "not ($SLOW)" "${FAST_FILES[@]}" ;;
  wave1)
    bash "$HERE/run_release_cpu.sh" t1 "wide_temperature1_k4" test_plookup.py
    bash "$HERE/run_release_cpu.sh" t07 "wide_temperature07_top_p_k4" test_plookup.py
    bash "$HERE/run_release_cpu.sh" exact "point_mass_lookup_is_exact or negative_control or wide_greedy_k4 or wide_posctl" test_plookup.py
    bash "$HERE/run_release_cpu.sh" gate "mixed_batch_gated or gated_request_never_reads" test_plgate.py
    ;;
  wave2)
    bash "$HERE/run_release_cpu.sh" k5-t1 "wide_temperature1_k5" test_recipe_k5.py
    bash "$HERE/run_release_cpu.sh" k5-t07 "wide_temperature07_top_p_k5" test_recipe_k5.py
    bash "$HERE/run_release_cpu.sh" k5-greedy "wide_greedy_k5" test_recipe_k5.py
    bash "$HERE/run_release_cpu.sh" k5-mixed "mixed_batch_gated_and_full_requests_are_exact_k5" test_recipe_k5.py
    ;;
  *) echo "unknown wave $1" >&2; exit 2 ;;
esac
