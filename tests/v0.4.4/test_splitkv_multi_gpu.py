"""GPU check of multi-request split-KV (UNO_GEMMA_SPLITKV_MULTI=1) in the v0.4.4 image. Run on one CUDA GPU:
  docker run --rm --gpus all --network none -v "$PWD":/t:ro --entrypoint python3 vllm-uno:0.4.4 /t/test_splitkv_multi_gpu.py

For each Gemma 4 attention layout (head 256 / 8 KV heads, sliding window 1023; head 512 / 2 KV heads, full) and widths
5, 6, 8, a uniform batch of B = 2, 4, 8 requests with different context lengths (300 to 6,000 tokens, paged KV,
shuffled blocks) runs:
  MULTI : one unified_attention call, switch on (split-KV for all B requests);
  SINGLE: B calls, one request each, switch off = the v0.4.3 one-request split-KV path;
  TWO_D : one call without the split-KV width = the 2D kernel v0.4.3 runs for B > 1.
MULTI must equal SINGLE bit for bit (the same per-sequence segmentation); MULTI vs TWO_D is reported (a different
reduction order, so a small bf16 difference is expected). Also checks that the switch-off multi-request call is the 2D
kernel (equal to TWO_D) and that MULTI used the split-KV scratch.
Then the step shapes a served uniform graph also replays: B real requests plus one padded request (a full CUDA graph
captured for more requests than the step has: zero query rows, query_start_loc repeating its last value, seq_len 0),
with real requests whose rows are a prompt chunk (no cached context, or less context than the width) next to verify
rows; MULTI's real rows must equal SINGLE bit for bit. Inputs are random tensors only. Exit 0 = all PASS."""
import os
import sys

import torch

from vllm.v1.attention.ops import triton_unified_attention as tua

torch.manual_seed(0)
dev = "cuda"
BLOCK = 16
fails = 0


def run(q, kc, vc, qsl, seqlens, bt, width, window, static, multi_env, segm_rows):
    os.environ["UNO_GEMMA_SPLITKV"] = "1"
    os.environ["UNO_GEMMA_SPLITKV_MULTI"] = "1" if multi_env else ""
    out = torch.empty_like(q)
    H, D = q.shape[1], q.shape[2]
    so = torch.full((segm_rows, H, 16, D), float("nan"), device=dev, dtype=torch.float32)
    sm = torch.full((segm_rows, H, 16), float("nan"), device=dev, dtype=torch.float32)
    se = torch.full((segm_rows, H, 16), float("nan"), device=dev, dtype=torch.float32)
    tua.unified_attention(
        q=q, k=kc, v=vc, out=out, cu_seqlens_q=qsl, max_seqlen_q=width, seqused_k=seqlens,
        max_seqlen_k=int(seqlens.max()), softmax_scale=1.0, causal=True, window_size=window, block_table=bt,
        softcap=0, q_descale=None, k_descale=None, v_descale=None, seq_threshold_3D=16, num_par_softmax_segments=16,
        softmax_segm_output=so, softmax_segm_max=sm, softmax_segm_expsum=se,
        uno_static_query_width=width if static else None)
    torch.cuda.synchronize()
    touched = bool(torch.isfinite(sm).any())
    return out, touched


for D, KVH, window in ((256, 8, (1023, 0)), (512, 2, (-1, -1))):
    for width in (5, 6, 8):
        for B in (2, 4, 8):
            ctx = torch.randint(300, 6000, (B,)).tolist()
            seqlens = torch.tensor([c + width for c in ctx], dtype=torch.int32, device=dev)
            nblk = [(s + BLOCK - 1) // BLOCK for s in seqlens.tolist()]
            total = sum(nblk) + 8
            perm = torch.randperm(total, device=dev).to(torch.int32)
            bt = torch.zeros((B, max(nblk)), dtype=torch.int32, device=dev)
            o = 0
            for i, n in enumerate(nblk):
                bt[i, :n] = perm[o:o + n]
                o += n
            kc = torch.randn((total, BLOCK, KVH, D), device=dev, dtype=torch.bfloat16)
            vc = torch.randn((total, BLOCK, KVH, D), device=dev, dtype=torch.bfloat16)
            q = torch.randn((B * width, 16, D), device=dev, dtype=torch.bfloat16) * 0.1
            qsl = torch.arange(0, (B + 1) * width, width, dtype=torch.int32, device=dev)
            multi, touched = run(q, kc, vc, qsl, seqlens, bt, width, window, True, True, B * 9)
            two_d, t2 = run(q, kc, vc, qsl, seqlens, bt, width, window, False, True, B * 9)
            off, toff = run(q, kc, vc, qsl, seqlens, bt, width, window, True, False, B * 9)
            single = torch.empty_like(q)
            for i in range(B):
                rows = slice(i * width, (i + 1) * width)
                s_out, _ = run(q[rows].contiguous(), kc, vc, qsl[:2] - 0, seqlens[i:i + 1], bt[i:i + 1], width, window,
                               True, False, 16)
                single[rows] = s_out
            exact = torch.equal(multi, single)
            off_is_2d = torch.equal(off, two_d) and not toff
            d2 = (multi.float() - two_d.float()).abs().max().item()
            ok = exact and touched and off_is_2d and not t2
            fails += 0 if ok else 1
            print(f"head={D} kv={KVH} window={window} width={width} B={B} ctx={ctx}: multi==single {exact}, "
                  f"engaged {touched}, off==2D {off_is_2d}, max|multi-2D| {d2:.3e} -> {'PASS' if ok else 'FAIL'}",
                  flush=True)
# Padded request + prompt-chunk rows.
for D, KVH, window in ((256, 8, (1023, 0)), (512, 2, (-1, -1))):
    for width in (5, 6, 8):
        for B in (2, 4, 7):
            ctx = ([0, 3] + torch.randint(300, 6000, (B - 2,)).tolist())[:B]
            seqlens = torch.tensor([c + width for c in ctx] + [0], dtype=torch.int32, device=dev)
            nblk = [(s + BLOCK - 1) // BLOCK for s in seqlens.tolist()[:B]]
            total = sum(nblk) + 8
            perm = torch.randperm(total, device=dev).to(torch.int32)
            bt = torch.zeros((B + 1, max(nblk)), dtype=torch.int32, device=dev)
            o = 0
            for i, n in enumerate(nblk):
                bt[i, :n] = perm[o:o + n]
                o += n
            kc = torch.randn((total, BLOCK, KVH, D), device=dev, dtype=torch.bfloat16)
            vc = torch.randn((total, BLOCK, KVH, D), device=dev, dtype=torch.bfloat16)
            q = torch.randn(((B + 1) * width, 16, D), device=dev, dtype=torch.bfloat16) * 0.1
            qsl = torch.tensor([i * width for i in range(B + 1)] + [B * width], dtype=torch.int32, device=dev)
            multi, touched = run(q, kc, vc, qsl, seqlens, bt, width, window, True, True, (B + 1) * 9)
            single = torch.empty_like(q[: B * width])
            for i in range(B):
                rows = slice(i * width, (i + 1) * width)
                s_out, _ = run(q[rows].contiguous(), kc, vc, qsl[:2] - 0, seqlens[i:i + 1], bt[i:i + 1], width, window,
                               True, False, 16)
                single[rows] = s_out
            exact = torch.equal(multi[: B * width], single)
            ok = exact and touched
            fails += 0 if ok else 1
            print(f"padded: head={D} kv={KVH} window={window} width={width} B={B}+1 padded ctx={ctx}: "
                  f"real rows multi==single {exact}, engaged {touched} -> {'PASS' if ok else 'FAIL'}", flush=True)
print(f"RESULT {'PASS' if fails == 0 else 'FAIL'} fails={fails}")
sys.exit(1 if fails else 0)
