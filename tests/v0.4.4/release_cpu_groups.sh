#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# release_cpu_groups.sh WAVE -- start one wave of the v0.4.4 CPU suites on ${IMG:-vllm-uno:0.4.4} through
# run_release_cpu.sh (detached, real rc to ../results/cpu/NAME.rc); poll with wait_release_cpu.sh. The selectors are
# written here so every result in ../results/cpu/SUMMARY.md names the exact tests it ran.
#   fast    -- v0.4.4: launcher recipe, multi-request admission, documented switches, startup refusals at both capture
#              lists (test_release_v044.py, test_startup_refusal_v044.py); v0.4.3: the lookup and gate unit tests
#   kernel  -- v0.4.4: multi-request split-KV == one-request split-KV, Triton interpreter (test_splitkv_multi_cpu.py),
#              including a step with a padded request and prompt-chunk rows
#   wave1   -- v0.4.3 exactness at K=4 L=2 (t1, t07, exact, gate), unchanged, on this image
#   wave2   -- v0.4.3 exactness at the recommended K=5 L=2 shape (k5-t1, k5-t07, k5-greedy, k5-mixed), unchanged
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SLOW='point_mass_lookup_is_exact or negative_control or wide_ or mixed_batch_gated or gated_request_never_reads'
R=$HERE/run_release_cpu.sh
case ${1:?wave} in
  fast)
    CPUS=4 bash "$R" fast v0.4.4 "" test_release_v044.py test_startup_refusal_v044.py
    CPUS=4 bash "$R" fast043 v0.4.3 "not ($SLOW)" test_plookup.py test_plgate.py ;;
  kernel) CPUS=2 bash "$R" kernel v0.4.4 "" test_splitkv_multi_cpu.py ;;
  wave1)
    bash "$R" t1 v0.4.3 "wide_temperature1_k4" test_plookup.py
    bash "$R" t07 v0.4.3 "wide_temperature07_top_p_k4" test_plookup.py
    bash "$R" exact v0.4.3 "point_mass_lookup_is_exact or negative_control or wide_greedy_k4 or wide_posctl" test_plookup.py
    bash "$R" gate v0.4.3 "mixed_batch_gated or gated_request_never_reads" test_plgate.py ;;
  wave2)
    bash "$R" k5-t1 v0.4.3 "wide_temperature1_k5" test_recipe_k5.py
    bash "$R" k5-t07 v0.4.3 "wide_temperature07_top_p_k5" test_recipe_k5.py
    bash "$R" k5-greedy v0.4.3 "wide_greedy_k5" test_recipe_k5.py
    bash "$R" k5-mixed v0.4.3 "mixed_batch_gated_and_full_requests_are_exact_k5" test_recipe_k5.py ;;
  *) echo "unknown wave $1" >&2; exit 2 ;;
esac
