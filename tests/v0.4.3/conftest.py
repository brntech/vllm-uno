"""CPU test setup for the v0.4.3 release image (no GPU, no network; Triton interpreter).

test_plookup.py and test_plgate.py show their own power with deliberately broken lookup rows. The release module has
no switch for that, so the broken rows are built HERE, test-side: PromptLookup(..., posctl_eps=None) returns the release
class itself, untouched; any other posctl_eps returns a test-only subclass that overwrites the lookup columns and the
hot / cold / stale_from values the release kernels read. Nothing here changes the exact path.
"""

import math
import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import triton.backends as _tb  # noqa: E402

if "nvidia" in _tb.backends:
    _tb.backends["nvidia"].driver.is_active = staticmethod(lambda: True)

from vllm.v1.worker.gpu.spec_decode import uno_plookup as pl  # noqa: E402

assert not hasattr(pl, "lookup_column_logits"), "this conftest targets the release module (no control knob)"

STALE, STALE2 = "stale", "stale2"
RELEASE_PROMPT_LOOKUP = pl.PromptLookup


def lookup_column_logits(vocab, eps):
    """(hot, cold) logits of a broken lookup column (test-only)."""
    if eps is None:
        return 0.0, float("-inf")
    if eps == STALE:
        return float("-inf"), 0.0
    if eps == STALE2:
        return 0.0, float("-inf")
    if eps >= 1.0:
        return 0.0, 0.0
    return math.log((1.0 - eps) + eps / vocab), math.log(eps / vocab)


class BrokenLookup(RELEASE_PROMPT_LOOKUP):
    """TEST-ONLY positive control: lookup columns that misstate the proposal (never part of the release)."""

    def __init__(self, k, lookup_len, max_num_reqs, device, posctl_eps):
        super().__init__(k, lookup_len, max_num_reqs, device)
        self.posctl_eps = posctl_eps
        self.stale_from = 1 if posctl_eps == STALE2 else lookup_len

    def init_draft_logits(self, draft_logits):
        self.hot, self.cold = lookup_column_logits(draft_logits.shape[-1], self.posctl_eps)
        cols = draft_logits[:, self.k : self.k + self.lookup_len]
        cols.fill_(self.cold)
        cols[..., 0] = self.hot
        if self.stale_from < self.lookup_len:
            stale = cols[:, self.stale_from :]
            stale.fill_(0.0)
            stale[..., 0] = float("-inf")
        self.prev.zero_()


def prompt_lookup(k, lookup_len, max_num_reqs, device, posctl_eps=None):
    if posctl_eps is None:
        return RELEASE_PROMPT_LOOKUP(k, lookup_len, max_num_reqs, device)
    if posctl_eps not in (STALE, STALE2) and not (0.0 < posctl_eps <= 1.0):
        raise ValueError("positive-control eps must be 'stale' or in (0, 1]")
    return BrokenLookup(k, lookup_len, max_num_reqs, device, posctl_eps)


pl.PromptLookup = prompt_lookup
pl.lookup_column_logits = lookup_column_logits
pl.STALE, pl.STALE2 = STALE, STALE2
