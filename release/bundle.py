#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create a deterministic archive from the explicit public asset list."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

from stack import manifest

ROOT = Path(__file__).resolve().parent.parent


def assets():
    names = json.loads((ROOT / "release/assets.json").read_text(encoding="utf-8"))
    if not names or len(names) != len(set(names)):
        raise ValueError("Asset list must be nonempty and unique")
    for name in names:
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or "\\" in name:
            raise ValueError(f"Unsafe asset path: {name}")
        path = ROOT / rel
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f"Missing or unsafe asset: {name}")
    expected = {"patch/" + entry["file"] for entry in manifest()["patches"]}
    if {name for name in names if name.startswith("patch/")} != expected:
        raise ValueError("Asset list and patch manifest disagree")
    return sorted(names)


def write_tar(stream):
    with tarfile.open(fileobj=stream, mode="w|") as archive:
        for name in assets():
            raw = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            info.mode = 0o755 if name.endswith(".sh") else 0o644
            info.mtime = 0
            archive.addfile(info, io.BytesIO(raw))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output", type=Path, nargs="?")
    ap.add_argument("--stdout", action="store_true", help="write an uncompressed Docker build context")
    args = ap.parse_args()
    if args.stdout == (args.output is not None):
        ap.error("provide either OUTPUT.tar.gz or --stdout")
    assets()
    if args.stdout:
        write_tar(sys.stdout.buffer)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            write_tar(compressed)
    print(json.dumps({"files": len(assets()), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
