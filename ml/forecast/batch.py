"""Vectorised batches for learned forecasters: history features, context, first-landing class, future path."""
import numpy as np
from common import *
from feats import HIST, NFEAT, NCTX


class Split:
    def __init__(self, profile, noise, split, n=None):
        D = load(profile, noise, split)
        if n:
            D = D[:n]
        self.D = np.ascontiguousarray(D)
        self.profile = profile
        S_, Hh = self.D.shape[:2]
        land = self.D[:, :, FI["landed"]].astype(bool)
        hrs = np.arange(1, Hh + 1)
        # last landing hour up to h (0 = none) and next landing hour after h (big = none); index h-1
        ll = np.where(land, hrs[None], 0)
        self.last_land = np.maximum.accumulate(ll, 1)
        nl = np.where(land, hrs[None], 1 << 30)
        nl_after = np.minimum.accumulate(nl[:, ::-1], 1)[:, ::-1]           # next landing at >= h
        self.next_land = np.concatenate([nl_after[:, 1:], np.full((S_, 1), 1 << 30)], 1)  # strictly after h
        # end minute of the storm that landed at last_land (scenario vs recurring by storm_index then)
        idx = self.D[:, :, FI["storm_index"]]
        nsc = scripted_storms(profile)
        rec_days = BAL["weather_profiles"][profile]["after"]["recurring_storms"]["days"]
        lm = self.last_land.astype(np.int64) * 60
        end_rec = lm + rec_days * DL
        if nsc:
            st = BAL["weather_profiles"][profile]["storms"][0]
            e = lm + len(st["temps_by_day"]) * DL
            if "final_phase" in st:
                e = (e // DL) * DL + int(float(st["final_phase"]["end_h"]) * 60)
            self.cur_end = np.where(idx < nsc, e, end_rec)
        else:
            self.cur_end = end_rec
        self.scen = (idx < nsc).astype(np.float32)

    def batch(self, s, h, want_future=True):
        """s, h: int arrays (h = 1-based issue hour). Returns hist (B,HIST,NFEAT), ctx (B,NCTX), cls (B,), fut (B,HOR,3)."""
        D = self.D
        B = len(s)
        off = np.arange(-HIST + 1, 1)
        hh = h[:, None] + off[None]                  # hours h-167..h
        valid = hh >= 1
        rows = D[s[:, None], np.clip(hh - 1, 0, H - 1)]   # (B,HIST,F)
        x = np.zeros((B, HIST, NFEAT), np.float32)
        x[..., 0] = (rows[..., FI["pressure"]] - 10130) / 100.0
        x[..., 1] = rows[..., FI["trend"]] / 100.0
        x[..., 2] = rows[..., FI["wind_ms"]] / 10.0
        x[..., 3] = rows[..., FI["animals"]] / 2.0
        x[..., 4] = rows[..., FI["sky"]] / 3.0
        x[..., 5] = rows[..., FI["in_storm"]]
        x[..., 6] = rows[..., FI["landed"]]
        x[..., 7] = 1.0
        wd = rows[..., FI["wind_dir"]]
        x[..., 8:16] = (wd[..., None] == np.arange(8)).astype(np.float32)
        x *= valid[..., None]
        r = D[s, h - 1]
        le = r[:, FI["last_end_h"]].astype(np.int64)
        stormed = le > -100000
        since = np.where(stormed, h - le, 0).astype(np.float32)
        ins = r[:, FI["in_storm"]].astype(np.float32)
        left = np.where(ins > 0, (self.cur_end[s, h - 1] - h * 60) / 60.0, 0.0)
        ctx = np.stack([h / H, np.where(stormed, np.minimum(since, 480) / 240.0, -1.0), stormed.astype(np.float32),
                        ins, left / 120.0, self.scen[s, h - 1]], 1).astype(np.float32)
        nxt = self.next_land[s, h - 1]
        k = nxt - h
        cls = np.where(k <= HOR, (k - 1) // 6, NB).astype(np.int64)
        fut = None
        if want_future:
            fh = h[:, None] + np.arange(HOR)[None]          # rows h..h+167 = hours h+1..h+168
            fr = D[s[:, None], np.clip(fh, 0, H - 1)]
            fut = np.stack([(fr[..., FI["pressure"]] - 10130) / 100.0, fr[..., FI["wind_ms"]] / 10.0,
                            fr[..., FI["in_storm"]] * 2.0 - 1.0], -1).astype(np.float32)
        return x, ctx, cls, fut
