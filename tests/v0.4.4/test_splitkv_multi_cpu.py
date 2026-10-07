"""Multi-request split-KV equals one-request split-KV, bit for bit (CPU, Triton interpreter; no GPU).

With UNO_GEMMA_SPLITKV_MULTI=1, a decode step whose B requests all have the same query width runs ONE split-KV launch
for all of them; v0.4.3 ran split-KV only for one request per step and the 2D kernel otherwise. For both Gemma 4
attention layouts (head 256 / 8 KV heads with the 1023-token sliding window, head 512 / 2 KV heads, full) and the
widths the recommended settings serve (5 draft rows, 6 gated verify rows, 8 verify rows), a uniform batch of B
requests with different context lengths on shuffled paged blocks runs:
  MULTI : one unified_attention call, switch on (split-KV for all B requests);
  SINGLE: B calls, one request each, switch off = the v0.4.3 one-request split-KV path;
  TWO_D : one call without the split-KV width = the 2D kernel v0.4.3 runs for B > 1.
MULTI must equal SINGLE exactly (torch.equal); with the switch off the multi-request call must equal TWO_D (the v0.4.3
path); MULTI must have used the split-KV scratch. The same checks run on the GPU in the release validation. Small
contexts keep the interpreter fast; the kernel's segmentation is the same code at any length.

Two step shapes a served uniform graph also replays are checked the same way: a padded request (a full CUDA graph
captured for more requests than the step has: vLLM gives the extra request zero query rows, query_start_loc repeating
its last value, and seq_len 0), and requests whose rows are a prompt chunk rather than verify rows (no cached context,
or less context than the width). Only the real requests' rows are compared.
"""

import os

import pytest
import torch

import conftest  # noqa: F401  (Triton interpreter + driver shim before vLLM imports)
from vllm.v1.attention.ops import triton_unified_attention as tua

BLOCK = 16


def _run(q, kc, vc, qsl, seqlens, bt, width, window, static, multi_env, segm_rows):
    os.environ["UNO_GEMMA_SPLITKV"] = "1"
    os.environ["UNO_GEMMA_SPLITKV_MULTI"] = "1" if multi_env else ""
    out = torch.empty_like(q)
    H, D = q.shape[1], q.shape[2]
    so = torch.full((segm_rows, H, 16, D), float("nan"), dtype=torch.float32)
    sm = torch.full((segm_rows, H, 16), float("nan"), dtype=torch.float32)
    se = torch.full((segm_rows, H, 16), float("nan"), dtype=torch.float32)
    tua.unified_attention(
        q=q, k=kc, v=vc, out=out, cu_seqlens_q=qsl, max_seqlen_q=width, seqused_k=seqlens,
        max_seqlen_k=int(seqlens.max()), softmax_scale=1.0, causal=True, window_size=window, block_table=bt,
        softcap=0, q_descale=None, k_descale=None, v_descale=None, seq_threshold_3D=16, num_par_softmax_segments=16,
        softmax_segm_output=so, softmax_segm_max=sm, softmax_segm_expsum=se,
        uno_static_query_width=width if static else None)
    return out, bool(torch.isfinite(sm).any())


@pytest.fixture(autouse=True)
def _restore_env():
    saved = {k: os.environ.get(k) for k in ("UNO_GEMMA_SPLITKV", "UNO_GEMMA_SPLITKV_MULTI")}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.mark.parametrize("head,kv_heads,window", [(256, 8, (1023, 0)), (512, 2, (-1, -1))])
@pytest.mark.parametrize("width", [5, 6, 8])
@pytest.mark.parametrize("batch", [2, 4])
def test_multi_request_split_kv_equals_one_request_split_kv(head, kv_heads, window, width, batch):
    g = torch.Generator().manual_seed(1000 * head + 10 * width + batch)
    ctx = torch.randint(20, 300, (batch,), generator=g).tolist()
    seqlens = torch.tensor([c + width for c in ctx], dtype=torch.int32)
    nblk = [(s + BLOCK - 1) // BLOCK for s in seqlens.tolist()]
    total = sum(nblk) + 4
    perm = torch.randperm(total, generator=g).to(torch.int32)
    bt = torch.zeros((batch, max(nblk)), dtype=torch.int32)
    o = 0
    for i, n in enumerate(nblk):
        bt[i, :n] = perm[o:o + n]
        o += n
    kc = torch.randn((total, BLOCK, kv_heads, head), generator=g).to(torch.bfloat16)
    vc = torch.randn((total, BLOCK, kv_heads, head), generator=g).to(torch.bfloat16)
    q = (torch.randn((batch * width, 16, head), generator=g) * 0.1).to(torch.bfloat16)
    qsl = torch.arange(0, (batch + 1) * width, width, dtype=torch.int32)
    rows = batch * 9
    multi, touched = _run(q, kc, vc, qsl, seqlens, bt, width, window, True, True, rows)
    two_d, touched_2d = _run(q, kc, vc, qsl, seqlens, bt, width, window, False, True, rows)
    off, touched_off = _run(q, kc, vc, qsl, seqlens, bt, width, window, True, False, rows)
    single = torch.empty_like(q)
    for i in range(batch):
        r = slice(i * width, (i + 1) * width)
        single[r], _ = _run(q[r].contiguous(), kc, vc, qsl[:2], seqlens[i:i + 1], bt[i:i + 1], width, window,
                            True, False, 16)
    print(f"head {head} width {width} B {batch} ctx {ctx}: max|multi - 2D| "
          f"{(multi.float() - two_d.float()).abs().max().item():.3e}")
    assert touched and not touched_2d and not touched_off
    assert torch.equal(multi, single), "multi-request split-KV differs from one-request split-KV"
    assert torch.equal(off, two_d), "switch off must be the v0.4.3 2D path"


@pytest.mark.parametrize("head,kv_heads,window", [(256, 8, (1023, 0)), (512, 2, (-1, -1))])
@pytest.mark.parametrize("width", [6, 8])
def test_padded_request_and_prompt_chunk_rows_equal_one_request_split_kv(head, kv_heads, window, width):
    g = torch.Generator().manual_seed(7000 + head + width)
    ctx = [0, 3, int(torch.randint(20, 300, (1,), generator=g))]  # prompt chunk, short context, verify rows
    real = len(ctx)
    seqlens = torch.tensor([c + width for c in ctx] + [0], dtype=torch.int32)  # + one padded request: seq_len 0
    nblk = [(s + BLOCK - 1) // BLOCK for s in seqlens.tolist()[:real]]
    total = sum(nblk) + 4
    perm = torch.randperm(total, generator=g).to(torch.int32)
    bt = torch.zeros((real + 1, max(nblk)), dtype=torch.int32)
    o = 0
    for i, n in enumerate(nblk):
        bt[i, :n] = perm[o:o + n]
        o += n
    kc = torch.randn((total, BLOCK, kv_heads, head), generator=g).to(torch.bfloat16)
    vc = torch.randn((total, BLOCK, kv_heads, head), generator=g).to(torch.bfloat16)
    q = (torch.randn(((real + 1) * width, 16, head), generator=g) * 0.1).to(torch.bfloat16)  # padded rows included
    qsl = torch.tensor([i * width for i in range(real + 1)] + [real * width], dtype=torch.int32)
    multi, touched = _run(q, kc, vc, qsl, seqlens, bt, width, window, True, True, (real + 1) * 9)
    single = torch.empty_like(q[: real * width])
    for i in range(real):
        r = slice(i * width, (i + 1) * width)
        single[r], _ = _run(q[r].contiguous(), kc, vc, qsl[:2], seqlens[i:i + 1], bt[i:i + 1], width, window,
                            True, False, 16)
    print(f"head {head} width {width} ctx {ctx} + 1 padded request: multi == single on the real rows")
    assert touched
    assert torch.equal(multi[: real * width], single), "a padded request or prompt-chunk rows changed the real rows"
