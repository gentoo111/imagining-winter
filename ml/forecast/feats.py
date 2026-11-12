"""History features / targets for learned forecasters (GRU, diffusion)."""
import numpy as np
from common import *

HIST = 168        # hours of precursor history
NFEAT = 8 + 8     # see hist_feats
NCTX = 6


def hist_feats(d, h):
    """(HIST, NFEAT) precursor history ending at hour h (inclusive), zero-padded before hour 1."""
    lo = h - HIST
    f = d[max(lo, 0):h]
    out = np.zeros((HIST, NFEAT), np.float32)
    k = len(f)
    if k:
        o = out[HIST - k:]
        o[:, 0] = (f[:, FI["pressure"]] - 10130) / 100.0
        o[:, 1] = f[:, FI["trend"]] / 100.0
        o[:, 2] = f[:, FI["wind_ms"]] / 10.0
        o[:, 3] = f[:, FI["animals"]] / 2.0
        o[:, 4] = f[:, FI["sky"]] / 3.0
        o[:, 5] = f[:, FI["in_storm"]]
        o[:, 6] = f[:, FI["landed"]]
        o[:, 7] = 1.0  # valid
        o[np.arange(k), 8 + f[:, FI["wind_dir"]]] = 1.0
    return out


def ctx_feats(d, h, profile):
    idx, after_end, ins = ctx_at(d, h)
    stormed = after_end > -(1 << 39)
    since = (h * 60 - after_end) / 60.0 if stormed else -1
    scen = 1.0 if idx < scripted_storms(profile) else 0.0
    # hours left in the current storm (observer knows the duration)
    left = 0.0
    if ins:
        lh = np.nonzero(d[:h, FI["landed"]])[0][-1] + 1
        left = (storm_end_min(profile, idx, lh * 60) - h * 60) / 60.0
    return np.array([h / H, min(since, 480) / 240.0 if stormed else -1.0, 1.0 if stormed else 0.0, float(ins), left / 120.0, scen], np.float32)


def future_target(d, h):
    """(HOR, 3): pressure (normalised), wind (normalised), storm flag in {-1, 1}."""
    f = d[h:h + HOR]
    return np.stack([(f[:, FI["pressure"]] - 10130) / 100.0, f[:, FI["wind_ms"]] / 10.0, f[:, FI["in_storm"]] * 2.0 - 1.0], -1).astype(np.float32)


def sample_index(nseeds, per_seed, rng, daily=False):
    """(seed, h) pairs; training uses random hours with a full 7-day future; daily=07:00 issue hours."""
    if daily:
        ih = issue_hours()
        return [(s, h) for s in range(nseeds) for h in ih]
    hs = rng.integers(24, H - HOR + 1, (nseeds, per_seed))
    return [(s, int(h)) for s in range(nseeds) for h in hs[s]]
