#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reference / candidate outputs via /v1/completions (no chat template, raw model).
Usage: golden.py --prompts prompts.json --out runs/x.jsonl [--temperature 0] [--max-tokens 512] [--url http://localhost:8000]"""
import argparse, json, time, requests
ap = argparse.ArgumentParser()
ap.add_argument("--prompts", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--url", default="http://localhost:8000"); ap.add_argument("--temperature", type=float, default=0.0)
ap.add_argument("--max-tokens", type=int, default=512); ap.add_argument("--model", default="uno-qwen3-8b")
a = ap.parse_args()
prompts = json.load(open(a.prompts))
t0 = time.time(); tot = 0
with open(a.out, "w") as f:
    for p in prompts:
        t1 = time.time()
        r = requests.post(f"{a.url}/v1/completions", json={
            "model": a.model, "prompt": p["prompt"], "max_tokens": a.max_tokens,
            "temperature": a.temperature, "seed": 0, "return_token_ids": True}, timeout=1200).json()
        if "choices" not in r: raise SystemExit(f"{p['id']}: {r}")
        c = r["choices"][0]; n = r["usage"]["completion_tokens"]; dt = time.time() - t1; tot += n
        f.write(json.dumps({"id": p["id"], "token_ids": c.get("token_ids"), "text": c["text"],
                            "finish": c["finish_reason"], "usage": r["usage"], "seconds": round(dt, 2)}) + "\n")
        print(p["id"], c["finish_reason"], n, f"{n/dt:.1f} tok/s", flush=True)
print(f"TOTAL {tot} tok in {time.time()-t0:.1f}s = {tot/(time.time()-t0):.1f} tok/s (serial)")
