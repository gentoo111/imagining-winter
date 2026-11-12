"""Climatology, clock-hazard and bot-rule categorical forecasters (fit on training weather)."""
import numpy as np
from common import *

ALPHA = 0.5


def clock_key(d, h, profile):
    idx, after_end, ins = ctx_at(d, h)
    scen = int(idx < scripted_storms(profile))
    if ins:
        lh = np.nonzero(d[:h, FI["landed"]])[0][-1] + 1
        left = (storm_end_min(profile, idx, lh * 60) - h * 60) // 60
        return (scen, 1, int(min(left // 6, 30)))
    if after_end > -(1 << 39):
        return (scen, 0, int(min((h * 60 - after_end) // 360, 80)))
    return (scen, 2, int(min(h // 6, 200)))


def warn(r):
    w = CH["warning"]
    return int(r[FI["sky"]] == 2 or (r[FI["trend"]] <= w["trend"] and r[FI["wind_ms"]] >= w["wind_ms"]))


class Table:
    def __init__(self, keyf):
        self.keyf = keyf; self.t = {}; self.glob = np.full(NB + 1, ALPHA)

    def fit(self, D, profile, hours):
        for s in range(D.shape[0]):
            d = D[s]
            for h in hours:
                c = true_class(d, h); k = self.keyf(d, h, profile)
                self.t.setdefault(k, np.full(NB + 1, ALPHA))[c] += 1
                self.glob[c] += 1
        return self

    def predict(self, d, h, profile):
        k = self.keyf(d, h, profile)
        v = self.t.get(k)
        if v is None:  # back off to the clock key without the extra part
            v = self.glob
        return v / v.sum()


def clim_key(d, h, profile):
    return 0


def bot_key(d, h, profile):
    return clock_key(d, h, profile) + (warn(d[h - 1]),)
