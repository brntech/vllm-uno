#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""G1: token-identical check. Usage: compare.py golden.jsonl candidate.jsonl
Exit 0 iff every prompt id in golden has identical token_ids in candidate."""
import json
import sys


def load(p):
    return {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8") if l.strip()}


g, c = load(sys.argv[1]), load(sys.argv[2])
bad = 0
for k in g:
    if k not in c:
        print(f"{k}: MISSING in candidate"); bad += 1; continue
    a, b = g[k]["token_ids"], c[k]["token_ids"]
    if a is None or b is None:
        same = g[k]["text"] == c[k]["text"]
        print(f"{k}: {'identical (text)' if same else 'DIVERGE (text)'}"); bad += 0 if same else 1; continue
    if a == b or (len(b) < len(a) and a[:len(b)] == b):
        tag = "identical" if a == b else f"identical-prefix ({len(b)}/{len(a)})"
        print(f"{k}: {tag} ({len(b)} tok, {c[k].get('seconds', '?')} s)"); continue
    i = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
    print(f"{k}: DIVERGE at {i}/{len(a)} (golden {a[i:i+3]} vs {b[i:i+3]}); lens {len(a)} vs {len(b)}")
    bad += 1
print(f"{'G1 PASS' if not bad else 'G1 FAIL'}: {len(g) - bad}/{len(g)} identical")
sys.exit(1 if bad else 0)
