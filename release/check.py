#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline package checks. No Docker, network, GPU, downloads or file writes."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from bundle import ROOT, assets
from stack import manifest


def main():
    series = manifest()
    names = assets()
    python_count = 0
    shell_count = 0
    shell = shutil.which("bash")
    if os.name == "nt":
        shell = shutil.which("bash.exe", path=str(Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin"))
    for name in names:
        path = ROOT / name
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".py":
            ast.parse(text, filename=name)
            python_count += 1
        if path.suffix == ".json":
            json.loads(text)
        if path.suffix == ".sh":
            for block in re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", text, re.S):
                ast.parse(block, filename=name + ":embedded-python")
            if shell:
                subprocess.run([shell, "-n", str(path)], check=True)
                shell_count += 1
    for name in ("gates/prefixes_spec.json", "gates/prompts_dbg_chat.json"):
        rows = json.loads((ROOT / name).read_text())
        assert rows and len({row["id"] for row in rows}) == len(rows), name
        assert "<|ifm|" not in (ROOT / name).read_text(), name
    manifest_hash = hashlib.sha256((ROOT / "release/series.json").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    print(json.dumps({"package_check": "PASS", "assets": len(names), "python_files": python_count, "shell_files_checked": shell_count,
                      "patched_tree": series["final_tree"], "series_sha256": manifest_hash}))


if __name__ == "__main__":
    main()
