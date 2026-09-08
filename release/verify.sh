#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fast finite correctness checks; never starts, stops, or modifies a server.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec "${PYTHON:-python3}" - "$ROOT" "$@" <<'PY'
import argparse
import hashlib
import json
import math
import pathlib
import re
import shutil
import subprocess
import sys
import time

root = pathlib.Path(sys.argv[1])
p = argparse.ArgumentParser(description="Capture plain references, then gate an already-running Uno server.")
p.add_argument("mode", choices=("reference", "candidate", "all"))
p.add_argument("--url", default="http://127.0.0.1:8000", help="server root URL, without /v1")
p.add_argument("--model", default="uno-qwen3-8b", help="served model alias; use the same alias for both arms")
p.add_argument("--out", type=pathlib.Path, required=True, help="new output directory (never overwritten)")
p.add_argument("--reference", type=pathlib.Path, help="directory produced by reference mode")
p.add_argument("--provenance", type=pathlib.Path, required=True,
               help="text file: exact server launch command, model/adapter revisions, image digest and GPU")
p.add_argument("--plain-url", help="plain server URL, required by all mode")
p.add_argument("--plain-provenance", type=pathlib.Path, help="plain launch record, required by all mode")
p.add_argument("--prefixes", type=pathlib.Path, default=root / "gates/prefixes_spec.json")
p.add_argument("--prompts", type=pathlib.Path, default=root / "gates/prompts_dbg_chat.json")
p.add_argument("--n", type=int, default=256)
p.add_argument("--max-tokens", type=int, default=16, help="sampled sequence length")
p.add_argument("--greedy-tokens", type=int, default=256)
a = p.parse_args(sys.argv[2:])
if a.n < 256 or a.max_tokens < 16 or a.greedy_tokens < 256:
    p.error("release checks require n >= 256, sampled tokens >= 16, greedy tokens >= 256")
if a.mode == "candidate" and not a.reference:
    p.error("candidate requires --reference")
if a.mode == "all" and (not a.plain_url or not a.plain_provenance):
    p.error("all requires --plain-url and --plain-provenance")
if a.mode == "all" and a.plain_url.rstrip("/") == a.url.rstrip("/"):
    p.error("all requires different plain and Uno endpoints; use reference/candidate for one GPU")
for launch in [a.provenance] + ([a.plain_provenance] if a.mode == "all" else []):
    if not launch.is_file() or not launch.read_text(encoding="utf-8").strip():
        p.error(f"missing or empty provenance record: {launch}")
for url in [a.url] + ([a.plain_url] if a.mode == "all" else []):
    if not url.startswith(("http://", "https://")) or url.rstrip("/").endswith("/v1"):
        p.error("URLs must be server roots, e.g. http://127.0.0.1:8000")

# No automatic dependency installation or hidden server management.
import requests

gates = root / "gates"
for tool in ("lossless_spec.py", "lossless_spec_compare.py", "golden.py", "compare.py"):
    if not (gates / tool).is_file():
        p.error(f"missing gate tool {gates / tool}; distribute release/ together with gates/")


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def api(url, path, payload=None):
    r = (requests.get(url.rstrip("/") + path, timeout=60) if payload is None else
         requests.post(url.rstrip("/") + path, json=payload, timeout=60))
    r.raise_for_status()
    return r


def health(url):
    api(url, "/health")
    models = api(url, "/v1/models").json()
    if a.model not in {m["id"] for m in models["data"]}:
        raise RuntimeError(f"{a.model!r} is not served at {url}")
    return models


def metrics(url):
    raw = api(url, "/metrics").text
    result = {}
    for name in ("drafts", "draft_tokens", "accepted_tokens"):
        metric = "vllm:spec_decode_num_" + name + "_total"
        result[name] = sum(float(m.group(1)) for m in re.finditer(
            rf"^{re.escape(metric)}(?:\{{[^}}]*\}})? (\S+)$", raw, re.M))
    return result


def run(out, tag, tool, *args):
    command = [sys.executable, str(gates / tool), *map(str, args)]
    print("GATE_COMMAND " + json.dumps(command), flush=True)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    (out / f"{tag}.log").write_text(result.stdout, encoding="utf-8")
    print(result.stdout, end="", flush=True)
    with (out / "commands.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps({"command": command, "exit": result.returncode}) + "\n")
    return result


def sample(out, name, url, prefixes, chunk, mixed=False):
    args = ["--url", url, "--model", a.model, "--prefixes", prefixes,
            "--out", out / f"{name}.json", "--n", a.n, "--max-tokens", a.max_tokens,
            "--chunk", chunk]
    if mixed:
        args += ["--mixed-greedy"]
    result = run(out, name, "lossless_spec.py", *args)
    if result.returncode:
        raise RuntimeError(f"sampling failed: {out / (name + '.log')}")
    data = json.loads((out / f"{name}.json").read_text(encoding="utf-8"))
    expected = {x["id"] for x in json.loads(prefixes.read_text(encoding="utf-8"))}
    if set(data["prefixes"]) != expected or any(
        len(x["samples"]) != a.n or any(not isinstance(t, list) or not t for t in x["samples"])
        for x in data["prefixes"].values()
    ):
        raise RuntimeError("incomplete samples, missing prefixes, or empty token sequences")
    if mixed:
        evidence = data.get("mixed_background")
        if (not isinstance(evidence, dict)
                or evidence.get("successes", 0) < 1
                or evidence.get("successful_overlaps", 0) < 1):
            raise RuntimeError("mixed sample lacks a successful overlapping background completion")
    return data


def compare_sample(out, name, reference, candidate, floor=None):
    permutations = 5000
    # Always perform the requested --perm 5000 first. A larger prefix/token set
    # can make 1/5001 exceed Bonferroni alpha: raise resolution, never waive FAIL.
    for attempt in range(2):
        summary = out / f"{name}.perm-{permutations}.json"
        args = [reference, candidate, "--perm", permutations, "--summary", summary]
        if floor:
            args += ["--floor", floor]
        result = run(out, f"{name}.perm-{permutations}", "lossless_spec_compare.py", *args)
        if not summary.is_file():
            raise RuntimeError(f"comparison did not produce a verdict: {name}")
        verdict = json.loads(summary.read_text(encoding="utf-8"))
        resolution_error = (result.returncode == 2 and verdict.get("verdict") == "ERROR"
                            and verdict.get("error") == "insufficient_permutation_resolution")
        if not resolution_error and (result.returncode not in (0, 1) or
                                     verdict.get("verdict") not in ("PASS", "FAIL")):
            raise RuntimeError(f"comparison failed: {name}")
        if verdict.get("tests", 0) <= 0:
            raise RuntimeError(f"zero statistical comparisons: {name}")
        alpha = verdict["alpha_bonferroni"]
        if not 0 < alpha < 1:
            raise RuntimeError(f"invalid Bonferroni alpha: {name}")
        if resolution_error:
            minimum = verdict.get("minimum_permutations")
            if (verdict.get("permutations") != permutations or type(minimum) is not int
                    or minimum <= permutations or 1 / (permutations + 1) <= alpha
                    or 1 / (minimum + 1) > alpha):
                raise RuntimeError(f"invalid permutation resolution response: {name}")
            permutations = max(minimum, math.ceil(1 / alpha))
            continue
        if 1 / (permutations + 1) > alpha:
            raise RuntimeError(f"under-resolved statistical verdict: {name}")
        break
    else:
        raise RuntimeError("insufficient permutation resolution")
    verdict["permutations"] = permutations
    dump(out / f"{name}.json", verdict)
    # Both process exit and verdict bind; the comparator declares its TV policy.
    return (result.returncode == 0 and verdict["verdict"] == "PASS" and
            not verdict["fails_p"] and not verdict["fails_tv"])


def golden(out, name, url, prompts):
    result = run(out, name, "golden.py", "--url", url, "--model", a.model,
                 "--prompts", prompts, "--temperature", 0, "--max-tokens", a.greedy_tokens,
                 "--out", out / f"{name}.jsonl")
    if result.returncode:
        raise RuntimeError(f"greedy generation failed: {name}")


def create_output(out, launch, url):
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(launch, out / "server-launch.txt")
    dump(out / "models.json", health(url))
    return {"schema": 1, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": a.model, "url": url, "n": a.n, "max_tokens": a.max_tokens,
            "greedy_tokens": a.greedy_tokens,
            "server_launch_sha256": sha(out / "server-launch.txt"),
            "gate_tools_sha256": {name: sha(gates / name) for name in
                                  ("lossless_spec.py", "lossless_spec_compare.py", "golden.py", "compare.py")}}


def reference(out, url, launch):
    record = create_output(out, launch, url)
    launch_text = launch.read_text(encoding="utf-8")
    if "--enable-lora" not in launch_text or "--speculative-config" in launch_text:
        raise RuntimeError("plain launch record must show --enable-lora and no --speculative-config; "
                           "match LoRA rank/slots and precision to Uno")
    print("REFERENCE: LoRA-enabled plain; launch record is operator supplied, not server introspection.", flush=True)
    prefixes = json.loads(a.prefixes.read_text(encoding="utf-8"))
    prompts = json.loads(a.prompts.read_text(encoding="utf-8"))
    if not prefixes or not prompts:
        raise RuntimeError("empty prompt/prefix set")
    for rows in (prefixes, prompts):
        if len({row["id"] for row in rows}) != len(rows):
            raise RuntimeError("duplicate prompt/prefix ids")
    for prefix in prefixes:
        # Freeze ids from this tokenizer once; every subsequent arm uses them.
        prefix["ids"] = api(url, "/tokenize", {"model": a.model, "prompt": prefix["text"]}).json()["tokens"]
        if not prefix["ids"]:
            raise RuntimeError("empty tokenized prefix")
    dump(out / "prefixes.json", prefixes)
    dump(out / "prompts.json", prompts)
    before = metrics(url)
    golden(out, "plain-greedy", url, out / "prompts.json")
    # Independent requests in one continuously running server, no reset or fixed
    # sample seed; copying sample-a to sample-b is never a valid noise floor.
    sample(out, "plain-sample-a", url, out / "prefixes.json", 1)
    sample(out, "plain-sample-b", url, out / "prefixes.json", 8)
    after = metrics(url)
    if any(after[k] != before[k] for k in before):
        raise RuntimeError("plain reference used speculation (or shared-server traffic); use a dedicated plain server")
    good = compare_sample(out, "plain-floor", out / "plain-sample-a.json", out / "plain-sample-b.json")
    record["verdict"] = "PASS" if good else "FAIL"
    record["files_sha256"] = {name: sha(out / name) for name in
                             ("prefixes.json", "prompts.json", "plain-greedy.jsonl",
                              "plain-sample-a.json", "plain-sample-b.json", "plain-floor.json")}
    dump(out / "reference.json", record)
    if not good:
        raise RuntimeError("plain-vs-plain floor failed; retain artifacts and investigate before gating Uno")
    print(f"REFERENCE_READY {out}", flush=True)


def candidate(out, url, launch, ref):
    baseline = json.loads((ref / "reference.json").read_text(encoding="utf-8"))
    if baseline["verdict"] != "PASS":
        raise RuntimeError("reference floor did not pass")
    for name, digest in baseline["files_sha256"].items():
        if sha(ref / name) != digest:
            raise RuntimeError(f"reference artifact changed after capture: {name}")
    for key in ("model", "n", "max_tokens", "greedy_tokens"):
        if baseline[key] != getattr(a, key):
            raise RuntimeError(f"candidate {key} does not match reference ({baseline[key]!r})")
    record = create_output(out, launch, url)
    if record["gate_tools_sha256"] != baseline["gate_tools_sha256"]:
        raise RuntimeError("gate source changed after reference capture")
    record["reference"] = str(ref.resolve())
    prefix_file = ref / "prefixes.json"
    # Verify the candidate tokenizer maps the text to the frozen reference ids.
    for prefix in json.loads(prefix_file.read_text(encoding="utf-8")):
        tokens = api(url, "/tokenize", {"model": a.model, "prompt": prefix["text"]}).json()["tokens"]
        if tokens != prefix["ids"]:
            raise RuntimeError(f"tokenizer mismatch: {prefix['id']}")
    golden(out, "uno-greedy", url, ref / "prompts.json")
    g1 = run(out, "greedy-compare", "compare.py", ref / "plain-greedy.jsonl", out / "uno-greedy.jsonl")
    # compare.py accepts shortened identical-prefix outputs and text fallback.
    # The release check additionally requires complete, nonempty token-id lists.
    def rows(path):
        data = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not data or len({x["id"] for x in data}) != len(data):
            raise RuntimeError(f"empty/duplicate greedy records: {path}")
        return {x["id"]: x for x in data}
    ga, gb = rows(ref / "plain-greedy.jsonl"), rows(out / "uno-greedy.jsonl")
    greedy_ok = (g1.returncode == 0 and "G1 PASS:" in g1.stdout and set(ga) == set(gb)
                 and all(isinstance(x.get("token_ids"), list) and x["token_ids"]
                         and x["token_ids"] == gb[k].get("token_ids") for k, x in ga.items()))
    before = metrics(url)
    single = sample(out, "uno-sample", url, prefix_file, 1)
    single_after = metrics(url)
    sampled_ok = compare_sample(out, "sampled-compare", ref / "plain-sample-a.json",
                                out / "uno-sample.json", ref / "plain-floor.json")
    sample(out, "uno-sample-mixed", url, prefix_file, 8, mixed=True)
    mixed_ok = compare_sample(out, "mixed-compare", ref / "plain-sample-a.json",
                              out / "uno-sample-mixed.json", ref / "plain-floor.json")
    after = metrics(url)
    delta = {k: after[k] - before[k] for k in before}
    # Check chunk-1 independently: the mixed run's greedy background must not
    # make an entirely non-speculative sampled path appear exercised.
    single_delta = {k: single_after[k] - before[k] for k in before}
    used = (single_delta["drafts"] > 0 and single_delta["draft_tokens"] > 0
            and any(x.get("accepted_per_step") is not None for x in single["prefixes"].values())
            and delta["drafts"] > 0 and delta["draft_tokens"] > 0)
    dump(out / "speculation-metrics.json", {"chunk1": single_delta, "total": delta})
    record.update(greedy="PASS" if greedy_ok else "FAIL", sampled="PASS" if sampled_ok else "FAIL",
                  mixed_chunk8="PASS" if mixed_ok else "FAIL", speculation_used=used,
                  verdict="PASS" if greedy_ok and sampled_ok and mixed_ok and used else "FAIL")
    dump(out / "verdict.json", record)
    print("VERIFY_RESULT " + json.dumps({k: record[k] for k in
          ("greedy", "sampled", "mixed_chunk8", "speculation_used", "verdict")}), flush=True)
    if not greedy_ok:
        print("Greedy differs: preserve outputs; Qwen bf16 near-tie attribution requires the "
              "teacher-forced and plain-self comparison described in docs/validation.md. "
              "This script does not waive differences or call them a PASS.", file=sys.stderr)
    if record["verdict"] != "PASS":
        raise RuntimeError(f"one or more gates failed; inspect {out / 'verdict.json'}")


try:
    if a.mode == "reference":
        reference(a.out, a.url, a.provenance)
    elif a.mode == "candidate":
        candidate(a.out, a.url, a.provenance, a.reference)
    else:
        a.out.mkdir(parents=True, exist_ok=False)
        reference(a.out / "reference", a.plain_url, a.plain_provenance)
        candidate(a.out / "candidate", a.url, a.provenance, a.out / "reference")
except (OSError, ValueError, KeyError, RuntimeError, requests.RequestException) as exc:
    print(f"VERIFY_FAILED: {exc}", file=sys.stderr)
    sys.exit(1)
PY
