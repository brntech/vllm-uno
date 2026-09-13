# SPDX-License-Identifier: Apache-2.0
"""Offline packaging regressions: unsafe or changed inputs must be refused."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "release"))
import bundle
import stack


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "release").mkdir()
        (self.root / "patch").mkdir()
        self.data = json.loads((ROOT / "release/series.json").read_text())
        for entry in self.data["patches"]:
            shutil.copyfile(ROOT / "patch" / entry["file"], self.root / "patch" / entry["file"])
        self.write_manifest()
        self.override = mock.patch.object(stack, "HERE", self.root / "release")
        self.override.start()
        self.addCleanup(self.override.stop)

    def write_manifest(self):
        (self.root / "release/series.json").write_text(json.dumps(self.data))

    def test_frozen_manifest(self):
        self.assertEqual(stack.manifest()["base_commit"], stack.BASE)

    def test_wrong_base_refused(self):
        self.data["base_commit"] = "0" * 40
        self.write_manifest()
        with self.assertRaises(RuntimeError):
            stack.manifest()

    def test_empty_series_refused(self):
        self.data["patches"] = []
        self.write_manifest()
        with self.assertRaises(RuntimeError):
            stack.manifest()

    def test_duplicate_patch_refused(self):
        self.data["patches"].append(copy.deepcopy(self.data["patches"][0]))
        self.write_manifest()
        with self.assertRaises(RuntimeError):
            stack.manifest()

    def test_unsafe_patch_path_refused(self):
        self.data["patches"][0]["file"] = "../escape.patch"
        self.write_manifest()
        with self.assertRaises(RuntimeError):
            stack.manifest()

    def test_changed_patch_refused(self):
        path = self.root / "patch" / self.data["patches"][0]["file"]
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaises(RuntimeError):
            stack.manifest()

    def test_wrong_file_inventory_refused(self):
        self.data["patches"][0]["files"] = []
        self.write_manifest()
        with self.assertRaises(RuntimeError):
            stack.manifest()

    def test_unapproved_patch_metadata_refused(self):
        path = self.root / "patch" / self.data["patches"][0]["file"]
        raw = path.read_bytes().replace(
            b".buildkite/test_areas/spec_decode.yaml",
            b".buildkite/test_areas/spec_decode.txt",
        )
        path.write_bytes(raw)
        self.data["patches"][0]["files"][0] = ".buildkite/test_areas/spec_decode.txt"
        self.data["patches"][0]["sha256_lf"] = hashlib.sha256(
            raw.replace(b"\r\n", b"\n")
        ).hexdigest()
        self.write_manifest()
        with self.assertRaises(RuntimeError):
            stack.manifest()

    def test_missing_asset_refused(self):
        (self.root / "release/assets.json").write_text(json.dumps(["missing.md"]))
        with mock.patch.object(bundle, "ROOT", self.root):
            with self.assertRaises(ValueError):
                bundle.assets()

    def test_parent_asset_refused(self):
        (self.root / "release/assets.json").write_text(json.dumps(["../escape.md"]))
        with mock.patch.object(bundle, "ROOT", self.root):
            with self.assertRaises(ValueError):
                bundle.assets()

    def test_duplicate_asset_refused(self):
        (self.root / "release/assets.json").write_text(json.dumps(["a", "a"]))
        with mock.patch.object(bundle, "ROOT", self.root):
            with self.assertRaises(ValueError):
                bundle.assets()

    def test_archive_is_deterministic_and_complete(self):
        self.override.stop()
        for name in ("first.tar.gz", "second.tar.gz"):
            subprocess.run([sys.executable, str(ROOT / "release/bundle.py"), str(self.root / name)],
                           check=True, capture_output=True)
        self.assertEqual((self.root / "first.tar.gz").read_bytes(), (self.root / "second.tar.gz").read_bytes())
        import tarfile
        with tarfile.open(self.root / "first.tar.gz") as archive:
            self.assertEqual(sorted(archive.getnames()), bundle.assets())
            self.assertTrue(all(item.mtime == 0 and item.isfile() for item in archive.getmembers()))


if __name__ == "__main__":
    unittest.main()
