#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Release patch application and binary-preserving Python overlay. Stdlib only."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid

HERE = Path(__file__).resolve().parent
BASE = "00972dfd72988942138a7a6089eaee08580210b8"


@contextmanager
def scratch(root):
    # Default directory ACLs work in Windows sandboxed workspaces too; Python
    # 3.14's private TemporaryDirectory ACL can make its child processes fail.
    parent = Path(git(root, "rev-parse", "--absolute-git-dir")).resolve()
    path = parent / ("uno-apply-" + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        if path.resolve().parent != parent or not path.name.startswith("uno-apply-"):
            raise RuntimeError("Unsafe temporary cleanup path")
        shutil.rmtree(path)


def git(root, *args):
    return subprocess.check_output(
        ["git", "-C", str(root), "-c", "core.autocrlf=false", *args],
        text=True, encoding="utf-8",
    ).strip()


def manifest():
    data = json.loads((HERE / "series.json").read_text(encoding="utf-8"))
    if data["base_commit"] != BASE:
        raise RuntimeError("Manifest base does not match the supported upstream commit")
    entries = data["patches"]
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("Release must contain a nonempty ordered patch manifest")
    names = [entry["file"] for entry in entries]
    if len(names) != len(set(names)) or any(
        not re.fullmatch(r"[A-Za-z0-9_.-]+\.patch", name) for name in names
    ):
        raise RuntimeError("Patch manifest contains duplicate or unsafe filenames")
    for entry in data["patches"]:
        path = HERE.parent / "patch" / entry["file"]
        raw = path.read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(raw).hexdigest() != entry["sha256_lf"]:
            raise RuntimeError(f"Patch checksum mismatch: {path}")
        paths = re.findall(rb"^diff --git a/(\S+) b/\S+$", raw, re.M)
        if [p.decode() for p in paths] != entry["files"]:
            raise RuntimeError(f"Patch path inventory mismatch: {path}")
        allowed_metadata = {b".buildkite/test_areas/spec_decode.yaml"}
        if any(
            not p.endswith((b".py", b".md")) and p not in allowed_metadata
            for p in paths
        ):
            raise RuntimeError(f"Overlay requires a new compiled-code audit: {path}")
    return data


def manifest_sha256():
    raw = (HERE / "series.json").read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def apply(checkout):
    data = manifest()
    root = Path(checkout).resolve()
    if not root.exists() or (root.is_dir() and not any(root.iterdir())):
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none",
                        "https://github.com/vllm-project/vllm.git", str(root)], check=True)
        git(root, "fetch", "origin", BASE)
        git(root, "checkout", "--detach", BASE)
    if git(root, "rev-parse", "--show-toplevel").replace("\\", "/").lower() != root.as_posix().lower():
        raise RuntimeError("Provide the checkout root, not a nested directory")
    head = git(root, "rev-parse", "HEAD")
    if head != BASE:
        raise RuntimeError(f"Expected clean base HEAD {BASE}; found {head}. Use a fresh checkout.")
    if git(root, "diff", "--name-only"):
        raise RuntimeError("Unstaged tracked changes present; refusing to overwrite")
    if git(root, "ls-files", "--others", "--exclude-standard"):
        raise RuntimeError("Untracked files present; use a clean checkout")
    tree = git(root, "write-tree")
    if tree == data["final_tree"]:
        print("Already applied: verified the entire staged tree.")
    elif tree == git(root, "rev-parse", "HEAD^{tree}"):
        # First preflight every patch in a separate index. On failure, the user's
        # index and worktree remain untouched. Git objects may be added harmlessly.
        with scratch(root) as tmp:
            env = os.environ.copy()
            env["GIT_INDEX_FILE"] = str(Path(tmp) / "index")
            normalized = []
            subprocess.run(["git", "-C", str(root), "read-tree", BASE], env=env, check=True)
            for entry in data["patches"]:
                patch = Path(tmp) / entry["file"]
                patch.write_bytes((HERE.parent / "patch" / entry["file"]).read_bytes().replace(b"\r\n", b"\n"))
                normalized.append(patch)
                subprocess.run(["git", "-C", str(root), "apply", "--cached", "--3way",
                                "--whitespace=nowarn", str(patch)], env=env, check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            staged = subprocess.check_output(["git", "-C", str(root), "write-tree"], env=env, text=True).strip()
            if staged != data["final_tree"]:
                raise RuntimeError(f"Reconstructed tree {staged} differs from release {data['final_tree']}")
            for patch in normalized:
                print(f"Applying {patch.name}", flush=True)
                subprocess.run(["git", "-C", str(root), "-c", "core.autocrlf=false",
                                "apply", "--3way", "--whitespace=nowarn", str(patch)], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if git(root, "write-tree") != data["final_tree"] or git(root, "diff", "--name-only"):
            raise RuntimeError("Final index/worktree verification failed")
    else:
        raise RuntimeError("Index is neither the clean base nor the exact release tree; refusing")
    print(f"HEAD={head} (upstream base; no commit created)")
    print(f"UNO_PATCHED_TREE={data['final_tree']}")
    print(f"UNO_SERIES_SHA256={manifest_sha256()}")


def overlay(source):
    """Run in image BEFORE installing any Python overlay, then copy only .py."""
    data = manifest()
    root = Path(source).resolve()
    if git(root, "write-tree") != data["final_tree"] or git(root, "diff", "--name-only"):
        raise RuntimeError("Overlay source is not the verified staged release")
    dist = importlib.metadata.distribution("vllm")
    version = dist.version
    package = Path(dist.locate_file("vllm")).resolve()
    # 00972dfd7 is the build's SCM version, not a tag guessed from its date.
    import vllm
    commit = getattr(vllm, "__commit__", None)
    if not (re.search(r"(?:\+|\.)g" + BASE[:8] + r"[0-9a-f]*(?:\.|$)", version)
            or (isinstance(commit, str) and len(commit) >= 8 and BASE.startswith(commit))):
        raise RuntimeError(f"Binary provenance mismatch: version={version}, commit={commit}")
    if Path(vllm.__file__).resolve().parent != package:
        raise RuntimeError("PYTHONPATH shadows installed vLLM; clear it before overlay")
    binaries = {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in package.rglob("*.so")}
    if not binaries:
        raise RuntimeError("No compiled vLLM libraries found in base image")
    tracked = git(root, "ls-files", "vllm").splitlines()
    for name in tracked:
        if name.endswith(".py"):
            rel = Path(name).relative_to("vllm")
            dest = package / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, dest)
            for pyc in (dest.parent / "__pycache__").glob(dest.stem + ".*.pyc"):
                pyc.unlink()
    # Preserve installed _version.py, kernels, flash-attn package, and data files.
    for rel, digest in binaries.items():
        if hashlib.sha256((package / rel).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Compiled library changed: {rel}")
    info = {"upstream_commit": BASE, "patched_tree": data["final_tree"],
            "base_vllm_version": version, "base_vllm_commit_attribute": commit,
            "compiled_libraries_sha256": binaries,
            "series_sha256": manifest_sha256()}
    (HERE / "build-provenance.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items() if k != "compiled_libraries_sha256"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["audit", "apply", "overlay"])
    parser.add_argument("checkout", nargs="?")
    args = parser.parse_args()
    if args.action == "audit":
        data = manifest()
        print(f"AUDIT_PASS: {len(data['patches'])} patch files; Python, Markdown, and approved Buildkite metadata only")
    elif not args.checkout:
        parser.error("checkout is required")
    elif args.action == "apply":
        apply(args.checkout)
    else:
        overlay(args.checkout)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"RELEASE_FAILED: {exc}", file=sys.stderr)
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            print(exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr, file=sys.stderr)
        sys.exit(1)
