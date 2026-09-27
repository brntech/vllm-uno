"""Check the shipped LoRA kernel configs on the release image's own kernels: for every draft-pass shape and row count,
vLLM's loader (VLLM_TUNED_CONFIG_FOLDER) must return the tuned config, not the default, and shrink -> expand into a
zeroed output (as vLLM's dual-stream path allocates it) must match a float32 reference, with rows that carry no
adapter left at zero. Run in the release image on the target GPU:
  docker run --rm --gpus device=0 --network none -v $PWD:/t:ro --entrypoint python3 \
    -e VLLM_TUNED_CONFIG_FOLDER=/opt/uno-kit/release/lora-configs ghcr.io/brntech/vllm-uno:0.4.1 /t/check_configs.py"""
import json, sys
import torch
from vllm.lora.ops.triton_ops import utils as U
from vllm.lora.ops.triton_ops.lora_expand_op import _lora_expand
from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta
from vllm.lora.ops.triton_ops.lora_shrink_op import _lora_shrink

dev = torch.device("cuda"); dt = torch.bfloat16; R = 16; MAXL = 2
torch.manual_seed(1)
LAYERS = {"qkv_local": (2816, (4096, 2048, 2048)), "qkv_global": (2816, (8192, 1024, 1024)),
          "o_local": (4096, (2816,)), "o_global": (8192, (2816,)), "gate_up": (2816, (2112, 2112)), "down": (2112, (2816,))}
import os
folder = os.environ["VLLM_TUNED_CONFIG_FOLDER"]
gpu = torch.cuda.get_device_name().replace(" ", "_").replace("-", "_")
SHR = json.load(open(f"{folder}/{gpu}_SHRINK.json"))
EXP = json.load(open(f"{folder}/{gpu}_EXPAND_FALSE.json"))
fails = checked = 0
for name, (K, outs) in LAYERS.items():
    ns = len(outs); N = max(outs)
    A = [torch.randn(MAXL, R, K, device=dev, dtype=dt) * 0.05 for _ in outs]
    B = [torch.randn(MAXL, o, R, device=dev, dtype=dt) * 0.05 for o in outs]
    for m in [4, 8, 12, 16, 20, 24, 28, 32, 40, 6, 36]:
        mp = torch.zeros(m, dtype=torch.int32); mp[::4] = -1
        meta = LoRAKernelMeta.make(MAXL, 64, dev); meta.prepare_tensors(mp.to(dev)); args = meta.meta_args(m, False)
        U.get_lora_op_configs.cache_clear()
        shr = U.get_lora_op_configs("shrink", MAXL + 1, m, K, R, ns)
        exp = U.get_lora_op_configs("expand", MAXL + 1, m, N, R, ns, add_inputs=False)
        # The loader must hand back one of the shipped entries for this slice count and shape (m may be interpolated).
        shr_file = [by_k[str(K)][str(R)] for by_k in SHR[str(MAXL + 1)][str(ns)].values()]
        exp_file = [by_k[str(R)][str(N)] for by_k in EXP[str(MAXL + 1)][str(ns)].values()]
        tuned = shr in shr_file and exp in exp_file
        x = torch.randn(m, K, device=dev, dtype=dt)
        buf = torch.zeros(ns, m, R, device=dev, dtype=torch.float32)
        out = torch.zeros(m, sum(outs), device=dev, dtype=dt)
        _lora_shrink(x, A, buf, *args, 1.0)
        _lora_expand(buf, B, out, *args, offset_start=0, add_inputs=False)
        torch.cuda.synchronize()
        rows = (mp != -1).to(dev)
        ref = torch.cat([(x.float() @ A[s][0].float().T) @ B[s][0].float().T for s in range(ns)], dim=1)
        ok = bool(torch.isfinite(out).all()) and bool((out[~rows] == 0).all()) and \
            bool(torch.allclose(out[rows].float(), ref[rows], atol=3e-2, rtol=3e-2))
        checked += 1
        if not (ok and tuned):
            fails += 1
            print(json.dumps({"layer": name, "m": m, "ok": ok, "tuned": tuned, "shrink": shr, "expand": exp}))
print(json.dumps({"checked": checked, "fails": fails, "gpu": torch.cuda.get_device_name()}))
sys.exit(1 if fails else 0)
