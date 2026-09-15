# SPDX-License-Identifier: Apache-2.0
"""Offline packaging regressions: unsafe or changed inputs must be refused."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
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

    def profile_flags(self, profile):
        launcher = (ROOT / "release/serve.sh").read_text(encoding="utf-8")
        match = re.search(rf"^  {profile}\)\n(.*?)^    ;;", launcher, re.S | re.M)
        self.assertIsNotNone(match, f"serve.sh has no {profile} profile block")
        block = match.group(1)
        flags = re.search(r"flags=\((.*?)\)\n", block, re.S)
        self.assertIsNotNone(flags, f"serve.sh {profile} block has no flags array")
        return block, flags.group(1)

    def test_qwen3_profile_captures_full_draft_rows(self):
        block, flags = self.profile_flags("qwen3")
        self.assertIn("--max-num-seqs 16", flags)
        self.assertIn("--max-num-batched-tokens 2048", flags)
        self.assertIn("--kv-cache-memory-bytes 2147483648", flags)
        self.assertNotIn("--gpu-memory-utilization", flags)
        self.assertIn("${UNO_K:-8}", block)
        match = re.search(
            r"--compilation-config '(\{[^']+\})'", flags
        )
        self.assertIsNotNone(match)
        config = json.loads(match.group(1))
        self.assertEqual(config["cudagraph_capture_sizes"], [1, 2, 4, 8, 16, 32, 64, 128, 144])
        self.assertGreaterEqual(max(config["cudagraph_capture_sizes"]), 16 * 8)

    def test_gemma4_profile_covers_its_served_draft_rows(self):
        block, flags = self.profile_flags("gemma4")
        self.assertIn("--max-num-seqs 4", flags)
        self.assertIn("--max-model-len 8192", flags)
        self.assertIn("--max-num-batched-tokens 2048", flags)
        self.assertIn("--gpu-memory-utilization 0.85", flags)
        self.assertIn("--attention-backend TRITON_ATTN", flags)
        self.assertIn("--language-model-only", flags)
        self.assertIn("--disable-hybrid-kv-cache-manager", flags)
        self.assertIn("--max-lora-rank 16", flags)
        self.assertIn("${UNO_K:-4}", block)
        self.assertIn("${UNO_MASK_TOKEN_ID:-262144}", block)
        match = re.search(
            r"--compilation-config '(\{[^']+\})'", flags
        )
        self.assertIsNotNone(match)
        config = json.loads(match.group(1))
        self.assertGreaterEqual(max(config["cudagraph_capture_sizes"]), 4 * 4)
        self.assertIn(4, config["cudagraph_capture_sizes"])
        self.assertIn(8, config["cudagraph_capture_sizes"])

    def test_mrv2_speculative_config_has_only_the_mrv2_uno_fields(self):
        launcher = (ROOT / "release/serve.sh").read_text(encoding="utf-8")
        for field in ("uno_lora_path", "uno_mask_token_id", "uno_noise_seed", "uno_noise_low"):
            self.assertIn(field, launcher)
        for retired in ("uno_graph", "uno_replay", "uno_overlap", "uno_fold",
                        "uno_no_host_sync", "uno_draft_full_graph"):
            self.assertNotIn(retired, launcher)
        qwen3_block, _ = self.profile_flags("qwen3")
        gemma4_block, _ = self.profile_flags("gemma4")
        self.assertIn("noise_low=${UNO_NOISE_LOW:-}", qwen3_block)
        self.assertIn("noise_low=${UNO_NOISE_LOW:-0}", gemma4_block)

    def test_dockerfile_reconstructs_from_the_pinned_ci_workspace(self):
        dockerfile = (ROOT / "release/Dockerfile").read_text(encoding="utf-8")
        self.assertIn("git clone --shared --no-checkout /vllm-workspace", dockerfile)
        self.assertIn("checkout --detach 00972dfd72988942138a7a6089eaee08580210b8", dockerfile)
        self.assertIn("vllm.v1.worker.gpu.spec_decode.uno_draft_moe", dockerfile)
        self.assertNotIn("https://github.com/vllm-project/vllm.git", dockerfile)

    def test_release_identity_is_consistent(self):
        version = (ROOT / "release/VERSION").read_text(encoding="utf-8").strip()
        self.assertEqual(version, "0.3.0")
        dockerfile = (ROOT / "release/Dockerfile").read_text(encoding="utf-8")
        self.assertIn(f'org.opencontainers.image.version="{version}"', dockerfile)
        self.assertIn("ai.uno.upstream.commit=\"00972dfd72988942138a7a6089eaee08580210b8\"", dockerfile)
        self.assertIn("ai.uno.head.commit=\"cf87916880b051e8782521dfe2afa12e0627e172\"", dockerfile)
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        self.assertIn(f"version: {version}\n", citation)
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertIn(f"## [{version}]", changelog)
        self.assertIn(f"[{version}]: https://github.com/brntech/vllm-uno/releases/tag/v{version}", changelog)

    def test_patch_series_is_the_two_ordered_layers(self):
        data = json.loads((ROOT / "release/series.json").read_text())
        self.assertEqual(data["code_base"], "3ad49350281a6b73de58449aadb293a8b398fb5d")
        self.assertEqual([entry["file"] for entry in data["patches"]],
                         ["0001-uno-mrv2-base.patch", "0002-uno-gemma4.patch"])
        gemma = data["patches"][1]["files"]
        for module in ("vllm/v1/worker/gpu/spec_decode/uno_draft_moe.py",
                       "vllm/v1/attention/ops/triton_unified_attention.py",
                       "tests/v1/spec_decode/UNO_DRAFT_GUARD_CALL_SITES.md"):
            self.assertIn(module, gemma)

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
