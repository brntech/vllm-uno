# v0.4.1 release evidence

| file | what it is |
| --- | --- |
| `v041-speed.jsonl` | production traffic (72 requests, one at a time) on the release-candidate image, balanced order tuned / off / DFlash K=8 / off / tuned / plain; `ms_per_token_http` over the measured requests, `tau` and per-position acceptance from the spec-decode counters |
| `v041-receipts.txt` | per Uno server, counts of vLLM's log lines: `tuned` = "Using tuned LoRA kernel configs", `default` = "Using default LoRA kernel configs", `nofile` = "No LoRA kernel configs found" (servers started with the variable empty, before the launcher unset it) |
| `v041-dist.jsonl` | short-prompt distribution check: mean total-variation distance of per-position token distributions (24 production prompts x 96 samples x first 8 tokens = 192 cells per pair) between three plain servers and two Uno servers (tuned configs on), plus split halves within each arm. Raw samples stay private (model outputs on production prompts); archive SHA-256 `dbfab6c9cd028e44cefa1a53e20dfa63c11e9447589b00f64cebc45ac2e7d1f5` |
| `overlay-session-2.jsonl`, `-receipts.txt` | the same config files on the v0.4.0 image (`relT` arms = exactly v0.4.1's runtime) against v0.4.0 (`rel`), balanced; `z2T` arms ran an unshipped diagnostic image (`uno-zerofill:z2`, an expand kernel that writes zeros for rows without an adapter) with the configs |
| `tune-report.json` | per shape and row count: default vs best kernel microseconds and the chosen config; timed on the diagnostic image above (its shrink kernel is stock; its expand kernel differs only on rows without an adapter) |
| `tune_lora.py` | the tuner, adapted for the stock kernels of the release image (run with `VLLM_TUNED_CONFIG_FOLDER` unset) |
| `check_configs.py`, `check-configs.txt` | the shipped files through vLLM's own loader on the release image's kernels: every shape and row count picks a shipped entry and matches a float32 reference |
| `image-identity.txt` | image ids and the vLLM package hash, identical in v0.4.0 and v0.4.1 |
