# SPDX-License-Identifier: Apache-2.0
"""Prompt-lookup continuation after Uno's K drafts, with a per-request length gate.

``UNO_PLOOKUP_L=L`` (1..8) makes every Uno proposal K Uno drafts followed by L
prompt-lookup tokens, verified in the same target pass (``num_speculative_tokens``
is K + L; the release serve script sets it). Unset or ``0`` is the v0.4.2 path.

The lookup token at column K+i is the token that followed the most recent
earlier occurrence of the longest suffix n-gram (n = NMAX..1) of the request's
context, where the context is the verified history (prompt + output) followed
by this step's K Uno drafts. Proposals past the end of the context repeat the
matched period; with no match at all the last context token is repeated.

Exactness: a lookup token is a deterministic function of the prefix it is
verified against, so its draft distribution is the point mass at that token.
Column K+i of the draft-logits cache is kept one-hot (0 at the proposed token,
-inf elsewhere); vLLM's rejection kernel then accepts with probability p(x)
and resamples from p without x, which is standard speculative sampling with a
point-mass proposal. Tokens are integers only; nothing is decoded or executed.

LENGTH GATE: ``UNO_PLOOKUP_MAX_CTX=N`` makes the scheduler verify only the K Uno draft
rows of a request whose context (prompt + output, in-flight tokens included) exceeds N tokens; shorter requests
verify K + L. Unset = no gate. The lookup columns are still written on the worker for every row,
but a gated request never schedules them, so they are neither verified nor paid for in the target pass.
"""

import os

import torch

from vllm.triton_utils import tl, triton

NMAX = 4
MAX_L = 8
_BLOCK = 1024
_SCORE_SHIFT = 32  # score = match_len << 32 | end_index


def plookup_len_from_env() -> int:
    """Lookup tokens per proposal from UNO_PLOOKUP_L; 0 (off) when unset."""
    raw = os.environ.get("UNO_PLOOKUP_L", "").strip()
    if raw == "":
        return 0
    if not raw.isdigit():
        raise ValueError(f"UNO_PLOOKUP_L must be an integer in 0..{MAX_L}")
    value = int(raw)
    if value > MAX_L:
        raise ValueError(f"UNO_PLOOKUP_L must be an integer in 0..{MAX_L}")
    return value


def plookup_max_ctx_from_env() -> int | None:
    """Context length above which a request verifies no lookup rows, from UNO_PLOOKUP_MAX_CTX; None (no gate) when unset."""
    raw = os.environ.get("UNO_PLOOKUP_MAX_CTX", "").strip()
    if raw == "":
        return None
    if not raw.isdigit() or int(raw) < 1:
        raise ValueError("UNO_PLOOKUP_MAX_CTX must be a positive integer (tokens)")
    return int(raw)


def gated_num_spec(num_spec: int, lookup_len: int, max_ctx: int | None, context_len: int) -> int:
    """Draft rows to schedule for one request: all ``num_spec`` (K + L), or the K Uno rows when its context exceeds
    ``max_ctx``. Depends on the request's length only, never on token values. Widths that do not include the lookup
    rows (0 or <= L, e.g. Uno's length tail) pass through unchanged."""
    if max_ctx is None or lookup_len == 0 or num_spec <= lookup_len:
        return num_spec
    return num_spec - lookup_len if context_len > max_ctx else num_spec


def plookup_gate_decode_query_lens(decode_query_lens: list[int], speculative_config) -> list[int]:
    """Uniform decode widths to capture as FULL CUDA graphs: with the length gate on, add each width minus L (a gated
    request verifies K + 1 rows instead of K + L + 1). Without it a gated step misses the FULL graph and runs PIECEWISE
    (measured without it: ms/token at 14k / 28k prompts about 1.3-1.5x the lookup-off speed)."""
    if speculative_config is None or getattr(speculative_config, "method", None) != "uno":
        return decode_query_lens
    lookup_len = plookup_len_from_env()
    if lookup_len == 0 or plookup_max_ctx_from_env() is None:
        return decode_query_lens
    extra = {q - lookup_len for q in decode_query_lens if q - lookup_len >= 2}
    out = sorted(set(decode_query_lens) | extra)
    from vllm.logger import init_logger

    init_logger(__name__).info(
        "Uno prompt lookup gate: FULL CUDA graphs captured at uniform decode widths %s", out
    )
    return out


@triton.jit
def _ctx_tok(hist_ptr, hist_row, draft_ptr, draft_row, T, c, idx):
    """Context token at index ``idx`` (vector or scalar); -1 outside [0, c)."""
    in_hist = (idx >= 0) & (idx < T)
    in_draft = (idx >= T) & (idx < c)
    h = tl.load(hist_ptr + hist_row + idx, mask=in_hist, other=-1).to(tl.int64)
    d = tl.load(draft_ptr + draft_row + (idx - T), mask=in_draft, other=-1).to(tl.int64)
    return tl.where(in_hist, h, tl.where(in_draft, d, -1))


@triton.jit
def _plookup_score_kernel(
    idx_mapping_ptr,  # [num_reqs] batch row -> request-state slot
    hist_ptr,  # [max_num_reqs, max_model_len] int32 (UVA), prompt + output
    hist_stride,
    total_len_ptr,  # [max_num_reqs] int32, verified tokens in hist
    draft_ptr,  # [num_reqs, K + L] int64, Uno drafts in columns 0..K-1
    draft_stride,
    best_ptr,  # [num_reqs] int64, zeroed by the caller
    K,
    NMAX_C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    b = tl.program_id(1)
    s = tl.load(idx_mapping_ptr + r).to(tl.int64)
    if s < 0:
        return
    T = tl.load(total_len_ptr + s).to(tl.int64)
    c = T + K
    hist_row = s * hist_stride
    draft_row = r.to(tl.int64) * draft_stride
    # Candidate end index j: an occurrence ctx[j-m:j] of the m-token suffix whose
    # continuation ctx[j] exists, so 1 <= j <= c - 1 (j = c is the suffix itself).
    j = b * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    valid = (j >= 1) & (j <= c - 1)
    run = valid
    m = tl.zeros([BLOCK], dtype=tl.int64)
    for t in tl.static_range(1, NMAX_C + 1):
        z = _ctx_tok(hist_ptr, hist_row, draft_ptr, draft_row, T, c, c - t)
        x = _ctx_tok(hist_ptr, hist_row, draft_ptr, draft_row, T, c, j - t)
        run = run & (j - t >= 0) & (c - t >= 0) & (x == z)
        m += run.to(tl.int64)
    score = tl.where(m > 0, (m << 32) | j, 0)
    top = tl.max(score, axis=0)
    if top > 0:
        tl.atomic_max(best_ptr + r, top)


@triton.jit
def _plookup_emit_kernel(
    idx_mapping_ptr,
    hist_ptr,
    hist_stride,
    total_len_ptr,
    draft_ptr,  # [num_reqs, K + L] int64; columns K.. receive the lookup tokens
    draft_stride,
    best_ptr,
    prev_ptr,  # [max_num_reqs, L] int64, token holding the one-hot in each column
    logits_ptr,  # [max_num_reqs, K + L, V] draft-logits cache
    logits_stride0,
    logits_stride1,
    K,
    L,
    HOT,  # logit of the proposed token (0.0 exact)
    COLD,  # logit of every other token (-inf exact)
    STALE_FROM,  # lookup columns >= this hold the stale row (-inf at the token, 0 elsewhere); L = none
    LPAD: tl.constexpr,
):
    r = tl.program_id(0)
    s = tl.load(idx_mapping_ptr + r).to(tl.int64)
    if s < 0:
        return
    T = tl.load(total_len_ptr + s).to(tl.int64)
    c = T + K
    hist_row = s * hist_stride
    draft_row = r.to(tl.int64) * draft_stride
    best = tl.load(best_ptr + r)
    m = best >> 32
    j = tl.where(m > 0, best - (m << 32), c - 1)
    period = tl.maximum(c - j, 1)
    i = tl.arange(0, LPAD).to(tl.int64)
    col = i < L
    tok = _ctx_tok(hist_ptr, hist_row, draft_ptr, draft_row, T, c, j + i % period)
    tok = tl.where(tok < 0, 0, tok)
    tl.store(draft_ptr + draft_row + K + i, tok, mask=col)
    old = tl.load(prev_ptr + s * L + i, mask=col, other=0)
    row = logits_ptr + s * logits_stride0 + (K + i) * logits_stride1
    stale = i >= STALE_FROM
    f0 = tl.zeros([LPAD], dtype=tl.float32)
    finf = tl.full([LPAD], float("-inf"), tl.float32)
    neg = tl.where(stale, f0, f0 + COLD).to(logits_ptr.dtype.element_ty)
    zero = tl.where(stale, finf, f0 + HOT).to(logits_ptr.dtype.element_ty)
    tl.store(row + old, neg, mask=col & (old != tok))
    tl.store(row + tok, zero, mask=col)
    tl.store(prev_ptr + s * L + i, tok, mask=col)


class PromptLookup:
    """Device state for the lookup columns of one Uno speculator."""

    def __init__(
        self,
        k: int,
        lookup_len: int,
        max_num_reqs: int,
        device: torch.device,
    ):
        if lookup_len < 1 or lookup_len > MAX_L:
            raise ValueError(f"prompt lookup length must be 1..{MAX_L}")
        # Every lookup column is the exact point mass: hot 0, cold -inf, no column past stale_from.
        self.stale_from = lookup_len
        self.hot, self.cold = 0.0, float("-inf")
        self.k = k
        self.lookup_len = lookup_len
        self.best = torch.zeros(max_num_reqs, dtype=torch.int64, device=device)
        # The one-hot of every lookup column starts at token 0 (see init_draft_logits).
        self.prev = torch.zeros(max_num_reqs, lookup_len, dtype=torch.int64, device=device)
        self.hist: torch.Tensor | None = None
        self.total_len: torch.Tensor | None = None
        self.num_calls = 0
        self.num_served = 0

    def init_draft_logits(self, draft_logits: torch.Tensor) -> None:
        """Make every lookup column the point mass at token 0 (matches ``prev``)."""
        cols = draft_logits[:, self.k : self.k + self.lookup_len]
        cols.fill_(self.cold)
        cols[..., 0] = self.hot
        self.prev.zero_()

    def bind_history(self, all_token_ids: torch.Tensor, total_len: torch.Tensor) -> None:
        if all_token_ids.dtype != torch.int32 or all_token_ids.stride(1) != 1:
            raise ValueError("prompt lookup needs row-contiguous int32 token history")
        self.hist = all_token_ids
        self.total_len = total_len

    def extend(
        self,
        num_reqs: int,
        idx_mapping: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_logits: torch.Tensor,
    ) -> None:
        """Write the L lookup tokens and their point masses for ``num_reqs`` rows."""
        if self.hist is None or self.total_len is None:
            raise RuntimeError("Uno prompt lookup: token history was never bound")
        if num_reqs == 0:
            return
        if draft_tokens.shape[1] != self.k + self.lookup_len:
            raise RuntimeError("Uno prompt lookup: draft width is not K + L")
        if draft_tokens.stride(1) != 1 or draft_logits.stride(2) != 1:
            raise RuntimeError("Uno prompt lookup: draft buffers must be contiguous in the last dim")
        best = self.best[:num_reqs]
        best.zero_()
        max_ctx = self.hist.shape[1] + self.k
        grid = (num_reqs, triton.cdiv(max_ctx, _BLOCK))
        _plookup_score_kernel[grid](
            idx_mapping,
            self.hist,
            self.hist.stride(0),
            self.total_len,
            draft_tokens,
            draft_tokens.stride(0),
            best,
            self.k,
            NMAX_C=NMAX,
            BLOCK=_BLOCK,
        )
        _plookup_emit_kernel[(num_reqs,)](
            idx_mapping,
            self.hist,
            self.hist.stride(0),
            self.total_len,
            draft_tokens,
            draft_tokens.stride(0),
            best,
            self.prev,
            draft_logits,
            draft_logits.stride(0),
            draft_logits.stride(1),
            self.k,
            self.lookup_len,
            self.hot,
            self.cold,
            self.stale_from,
            LPAD=triton.next_power_of_2(MAX_L),
        )
        self.num_calls += 1


def reference_lookup(history: list[int], drafts: list[int], lookup_len: int, nmax: int = NMAX) -> list[int]:
    """Pure-Python statement of the matcher (the CPU tests compare the kernels with it)."""
    ctx = list(history) + list(drafts)
    c = len(ctx)
    best_m, best_j = 0, -1
    for j in range(1, c):
        m = 0
        while m < nmax and j - m - 1 >= 0 and c - m - 1 >= 0 and ctx[j - m - 1] == ctx[c - m - 1]:
            m += 1
        if m > 0 and (m, j) > (best_m, best_j):
            best_m, best_j = m, j
    j = best_j if best_m > 0 else c - 1
    period = max(c - j, 1)
    return [ctx[j + i % period] for i in range(lookup_len)]
