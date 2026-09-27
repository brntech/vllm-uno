"""Tune vLLM LoRA shrink/expand Triton configs for Uno's draft-pass shapes (Gemma 4 26B A4B, rank 16) on this GPU.
Times each candidate under CUDA graphs (as served), checks the winner against a torch reference, and writes
VLLM_TUNED_CONFIG_FOLDER files: <gpu>_SHRINK.json and <gpu>_EXPAND_FALSE.json, keyed [max_loras][slices][m][k][n].
Runs on the release image (stock LoRA kernels). Usage: python3 tune_lora.py OUT_DIR   (one JSON line per shape and m on stdout: default vs best microseconds; run with VLLM_TUNED_CONFIG_FOLDER unset)."""
import itertools, json, sys, time
import torch, triton
import vllm.lora.ops.triton_ops.lora_shrink_op as S
import vllm.lora.ops.triton_ops.lora_expand_op as E
from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta
from vllm.lora.ops.triton_ops.utils import get_lora_op_configs

out_dir = sys.argv[1]
import os
if os.environ.get("VLLM_TUNED_CONFIG_FOLDER"):
    sys.exit("unset VLLM_TUNED_CONFIG_FOLDER: the baseline must be vLLM's default configs")
os.makedirs(out_dir, exist_ok=True)
dev = torch.device("cuda"); dt = torch.bfloat16; R = 16; MAXL = 2
torch.manual_seed(0)
MS = [4, 8, 12, 16, 20, 24, 28, 32, 40]
SHRINK = {"qkv": (2816, 3), "gate_up": (2816, 2), "o_local": (4096, 1), "o_global": (8192, 1), "down": (2112, 1)}
EXPAND = {"qkv_local": (4096, 2048, 2048), "qkv_global": (8192, 1024, 1024), "o_down": (2816,), "gate_up": (2112, 2112)}
gpu = torch.cuda.get_device_name().replace(" ", "_").replace("-", "_")


def meta(m):
    mp = torch.zeros(m, dtype=torch.int32)
    mp[::4] = -1  # the seed row of each request carries no adapter
    mt = LoRAKernelMeta.make(MAXL, 64, dev)
    mt.prepare_tensors(mp.to(dev))
    return mt.meta_args(m, False), mp


def bench(fn):
    try:
        fn(); torch.cuda.synchronize()
        return triton.testing.do_bench_cudagraph(fn, rep=40)
    except Exception:
        return float("inf")


def fixed(cfg):
    return lambda *a, **k: cfg


shr_space = [dict(block_m=bm, block_n=16, block_k=bk, split_k=sk, num_warps=w, num_ctas=1, group_size_m=8,
                  num_stages=st, max_nreg=None)
             for bm, bk, sk, w, st in itertools.product([16, 32], [64, 128, 256, 512], [8, 16, 32, 64, 128], [2, 4, 8], [2, 3])]
exp_space = [dict(block_m=bm, block_n=bn, block_k=16, num_warps=w, num_ctas=1, num_stages=st, max_nreg=None)
             for bm, bn, w, st in itertools.product([16, 32, 64], [32, 64, 128, 256], [2, 4, 8], [1, 2, 3])]
shr_json, exp_json, report = {}, {}, []
t0 = time.time()

for name, (K, ns) in SHRINK.items():
    A = [torch.randn(MAXL, R, K, device=dev, dtype=dt) * 0.05 for _ in range(ns)]
    for m in MS:
        args, mp = meta(m)
        x = torch.randn(m, K, device=dev, dtype=dt)
        buf = torch.empty(ns, m, R, device=dev, dtype=torch.float32)
        default = get_lora_op_configs("shrink", MAXL + 1, m, K, R, ns)
        res = []
        for cfg in [default] + shr_space:
            S.get_lora_op_configs = fixed(cfg)
            res.append((bench(lambda: S._lora_shrink(x, A, buf, *args, 1.0)), cfg))
        d_t = res[0][0]
        best_t, best = min(res, key=lambda r: r[0])
        S.get_lora_op_configs = fixed(best)
        S._lora_shrink(x, A, buf, *args, 1.0); torch.cuda.synchronize()
        ref = torch.stack([x.float() @ A[s][0].float().T for s in range(ns)])
        rows = (mp != -1).to(dev)
        ok = bool(torch.allclose(buf[:, rows], ref[:, rows], atol=2e-2, rtol=2e-2))
        if not ok:
            best, best_t = default, d_t
        shr_json.setdefault(str(MAXL + 1), {}).setdefault(str(ns), {}).setdefault(str(m), {}).setdefault(str(K), {})[str(R)] = best
        report.append(dict(elapsed_s=round(time.time() - t0), op="shrink", shape=name, m=m, default_us=round(d_t * 1e3, 2), best_us=round(best_t * 1e3, 2), ok=ok, best=best))
        print(json.dumps(report[-1]), flush=True)

for name, outs in EXPAND.items():
    ns = len(outs); N = max(outs)
    B = [torch.randn(MAXL, o, R, device=dev, dtype=dt) * 0.05 for o in outs]
    for m in MS:
        args, mp = meta(m)
        inp = torch.randn(ns, m, R, device=dev, dtype=torch.float32)
        out = torch.zeros(m, sum(outs), device=dev, dtype=dt)  # vLLM's dual-stream path allocates the LoRA output with zeros
        default = get_lora_op_configs("expand", MAXL + 1, m, N, R, ns, add_inputs=False)
        res = []
        for cfg in [default] + exp_space:
            E.get_lora_op_configs = fixed(cfg)
            res.append((bench(lambda: E._lora_expand(inp, B, out, *args, offset_start=0, add_inputs=False)), cfg))
        d_t = res[0][0]
        best_t, best = min(res, key=lambda r: r[0])
        E.get_lora_op_configs = fixed(best)
        out.zero_()
        E._lora_expand(inp, B, out, *args, offset_start=0, add_inputs=False); torch.cuda.synchronize()
        ref = torch.cat([inp[s].to(dt).float() @ B[s][0].float().T for s in range(ns)], dim=1)
        rows = (mp != -1).to(dev)
        ok = bool(torch.isfinite(out).all()) and \
            bool(torch.allclose(out[rows].float(), ref[rows], atol=3e-2, rtol=3e-2))
        if not ok:
            best, best_t = default, d_t
        exp_json.setdefault(str(MAXL + 1), {}).setdefault(str(ns), {}).setdefault(str(m), {}).setdefault(str(R), {})[str(N)] = best
        report.append(dict(elapsed_s=round(time.time() - t0), op="expand", shape=name, m=m, default_us=round(d_t * 1e3, 2), best_us=round(best_t * 1e3, 2), ok=ok, best=best))
        print(json.dumps(report[-1]), flush=True)

json.dump(shr_json, open(f"{out_dir}/{gpu}_SHRINK.json", "w"), indent=1)
json.dump(exp_json, open(f"{out_dir}/{gpu}_EXPAND_FALSE.json", "w"), indent=1)
json.dump(report, open(f"{out_dir}/tune-report.json", "w"), indent=1)
print("DONE", gpu, f"{time.time() - t0:.0f}s", file=sys.stderr, flush=True)
