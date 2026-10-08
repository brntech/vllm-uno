"""RDNA4 (gfx12) prefill launch tiles for vLLM's Triton unified attention, applied at container start.

The ROCm launcher (release/serve.sh) runs this before the API server starts when every GPU is gfx12:
    python3 /opt/uno-kit/release/gfx12_prefill_tiles.py
It rewrites the image's own triton_unified_attention.py in the (ephemeral) container filesystem; nothing outside
the container changes.

WHY: on ROCm vLLM launches the prefill pass of the Triton kernel with decode-shaped tiles (BLOCK_M 16, TILE 32,
default warps): 8 query tokens per program on Gemma 4's sliding layers (head 256) and 2 on its 512-dim global
layers. Upstream's tuned large-head branch is gated to B200 (capability family 100) and head 256.

Per head size: BLOCK_M, TILE, warps, stages. Engages only on prefill-shaped launches (max_seqlen_q >= MIN_Q and a
mean of >= MIN_MEAN_Q query rows per sequence), so
plain decode (q=1), speculative verify passes (q=K+1) and batches that are mostly decode/verify
rows keep the stock launch.
Env:
    R9700_PREFILL_TILES=0           do nothing (the stock launch)
    R9700_PREFILL_CFG_256=128,16,8,1  R9700_PREFILL_CFG_512=64,16,8,1   (defaults below)
    R9700_PREFILL_MIN_Q=32
    R9700_PREFILL_MIN_MEAN_Q=16     also require q rows / sequences >= this; 0 = gate on the longest query only
Fails loudly (non-zero exit, so the container does not start half-configured) if the anchor it patches is
missing or ambiguous, or if the GPU is not gfx12. vLLM 0.30.0 ships the file it patches with sha256
65d0fbcf33d0...5829.
"""
import importlib.util
import os
import subprocess
import sys

DEFAULTS = {256: "128,16,8,1", 512: "64,16,8,1"}
ANCHOR = "    if tuned_large_head:\n        TILE_SIZE_PREFILL = 128\n"
MARK = "_R9700_PREFILL_SEEN"


def gpu_arch() -> str:
    out = ""
    for exe in ("rocminfo", "/opt/rocm/bin/rocminfo"):
        try:
            out = subprocess.run([exe], capture_output=True, text=True, timeout=60).stdout
            break
        except (OSError, subprocess.TimeoutExpired):
            continue
    names = [ln.split()[-1] for ln in out.splitlines() if ln.strip().startswith("Name:") and "gfx" in ln]
    return names[0] if names else ""


def main() -> int:
    if os.environ.get("R9700_PREFILL_TILES", "1") == "0":
        print("[gfx12-prefill-tiles] disabled by R9700_PREFILL_TILES=0")
        return 0
    arch = gpu_arch()
    if not arch.startswith("gfx12"):
        print(f"[gfx12-prefill-tiles] REFUSE: GPU arch {arch or 'unknown'} is not gfx12", file=sys.stderr)
        return 3
    cfgs = {hd: os.environ.get(f"R9700_PREFILL_CFG_{hd}", d) for hd, d in DEFAULTS.items()}
    for hd, c in cfgs.items():
        parts = c.split(",")
        if len(parts) != 4 or not all(p.strip().isdigit() for p in parts):
            print(f"[gfx12-prefill-tiles] REFUSE: bad R9700_PREFILL_CFG_{hd}={c!r}", file=sys.stderr)
            return 3
    min_q = int(os.environ.get("R9700_PREFILL_MIN_Q", "32"))
    # Mean query rows per sequence in the batch (q.shape[0] / num_seqs, host ints, no sync). 0 = gate on max only.
    min_mean_q = int(os.environ.get("R9700_PREFILL_MIN_MEAN_Q", "16"))

    spec = importlib.util.find_spec("vllm")
    path = os.path.join(spec.submodule_search_locations[0], "v1", "attention", "ops", "triton_unified_attention.py")
    src = open(path).read()
    if MARK in src:
        print(f"[gfx12-prefill-tiles] already applied: {path}")
        return 0
    if src.count(ANCHOR) != 1 or src.count("import torch\n") < 1:
        print(f"[gfx12-prefill-tiles] REFUSE: anchor count {src.count(ANCHOR)} in {path}", file=sys.stderr)
        return 4

    table = "{" + ", ".join(f"{hd}: ({c})" for hd, c in cfgs.items()) + "}"
    inject = ANCHOR + (
        f"    _r9700_cfg = {table}.get(head_size)\n"
        f"    if _r9700_cfg is not None and max_seqlen_q >= {min_q} and q.shape[0] >= {min_mean_q} * num_seqs:\n"
        "        BLOCK_M, TILE_SIZE_PREFILL, launch_num_warps, launch_num_stages = _r9700_cfg\n"
        "        BLOCK_Q = BLOCK_M // num_queries_per_kv\n"
        "        total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs\n"
        f"        if head_size not in {MARK}:\n"
        f"            {MARK}.add(head_size)\n"
        "            logger.info('gfx12 prefill tiles engaged: head_size=%d BLOCK_M=%d BLOCK_Q=%d TILE=%d warps=%d "
        "stages=%d', head_size, BLOCK_M, BLOCK_Q, TILE_SIZE_PREFILL, launch_num_warps, launch_num_stages)\n")
    new = src.replace(ANCHOR, inject).replace("import torch\n", f"import torch\n{MARK} = set()\n", 1)
    with open(path, "w") as f:
        f.write(new)
    print(f"[gfx12-prefill-tiles] applied to {path} (arch {arch}): {cfgs}, min_q {min_q}, min_mean_q {min_mean_q}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
