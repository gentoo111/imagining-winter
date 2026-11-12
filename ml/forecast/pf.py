"""Bootstrap particle filter oracle for the chaotic weather (exact integer Lorenz, exact observation model).
Usage (module): run_seed(d, profile, noise, N, jitter, reseed, rng) -> dict(probs, paths, ess)."""
import numpy as np
from common import *

L1 = SC["animal_noise_pct"] / 100.0
SKY = {0: SC["halo_calm_pct"] / 100.0, 1: SC["halo_s1_pct"] / 100.0, 2: SC["grey_s2_pct"] / 100.0}


def build_bank(n=200000, seed=0):
    """Attractor states with their own 24 h z history and previous two hourly z, from training weather."""
    d = load("endurance", "murky", "train")[:400]
    rng = np.random.default_rng(seed)
    si = rng.integers(0, d.shape[0], n); hi = rng.integers(30, H, n)
    x = d[si, hi, 1].astype(np.int64); y = d[si, hi, 2].astype(np.int64); z = d[si, hi, 3].astype(np.int64)
    zh = np.stack([d[si, hi - k, 3] for k in range(23, -1, -1)], 1).astype(np.int64)  # z over last 24 h incl. now
    return dict(x=x, y=y, z=z, zh=zh)


def _anim_lik(a, stage):
    # animals = clamp(stage + n), n in {-1:.2, 0:.6, +1:.2}
    p = np.zeros_like(stage, dtype=float)
    for n, pn in ((-1, L1), (0, 1 - 2 * L1), (1, L1)):
        p += pn * (np.clip(stage + n, 0, 2) == a)
    return p


def _sky_lik(s, stage, storm):
    if storm:
        return np.full(stage.shape, 1.0 if s == 3 else 0.0)
    p0 = np.where(stage == 0, 1 - SKY[0], np.where(stage == 1, 1 - SKY[1], 0.0))
    p1 = np.where(stage == 0, SKY[0], np.where(stage == 1, SKY[1], 1 - SKY[2]))
    p2 = np.where(stage == 2, SKY[2], 0.0)
    return {0: p0, 1: p1, 2: p2}.get(s, np.zeros(stage.shape))


def obs_lik(r, x, y, z, zh, storm, sign_hour, noise):
    nzp = NZ[noise]["pressure"]; nzw = NZ[noise]["wind"]
    w = (np.abs(r[FI["pressure"]] - p_mean(z)) <= nzp).astype(float)
    fl = NZ[noise].get("wind_dir_flip", 0.0)
    if fl > 0:  # S1-12 windnoise: P(reported = true) = 1 - p, P(true +- 1 mod 8) = p / 2 each (weather_gen wind_noise)
        dd = (r[FI["wind_dir"]] - wind_dir_of(x, y, storm)) % 8
        w *= np.where(dd == 0, 1.0 - fl, np.where((dd == 1) | (dd == 7), fl / 2, 0.0))
    else:
        w *= (wind_dir_of(x, y, storm) == r[FI["wind_dir"]])
    wm = wind_mean(y, storm); o = r[FI["wind_ms"]]
    if o > 0:
        w *= (np.abs(o - wm) <= nzw)
    else:
        w *= np.clip(-wm + nzw + 1, 0, 2 * nzw + 1) / (2 * nzw + 1) * (2 * nzw + 1)  # count of k with wm+k<=0 (relative)
    if sign_hour:
        zmax = zh.max(1)
        stage = np.where(storm | (zmax >= SC["z2_c"] * S // 100), 2, np.where(zmax >= SC["z1_c"] * S // 100, 1, 0))
        w *= _anim_lik(r[FI["animals"]], stage) * _sky_lik(int(r[FI["sky"]]), stage, storm)
    return w


def forecast(profile, noise, x, y, z, h1, h2, h, idx, after_end, ins, cur_end, rng, n_paths=50):
    """First-landing class distribution over 29 classes + sampled observable paths, from particle states at hour h."""
    N = len(x)
    land_k = np.full(N, -1)
    tierk = np.zeros(N, int)
    in_st = ins == 1
    ae = cur_end if in_st else after_end
    i_idx = idx + 1 if in_st else idx
    pick = rng.choice(N, n_paths, replace=False) if N >= n_paths else rng.choice(N, n_paths)
    zs = np.zeros((HOR, n_paths), np.int64); ys = np.zeros((HOR, n_paths), np.int64)
    for k in range(1, HOR + 1):
        m = (h + k) * 60
        x, y, z = lorenz_hour(x, y, z)
        ok = (land_k < 0) & (h1 > h2) & (h1 >= z) & ((not in_st) | (m > ae))
        if ok.any():
            lnd = ok & lands(profile, i_idx, m, h1, ae)
            land_k[lnd] = k; tierk[lnd] = tier_of(h1[lnd])
        h2 = h1; h1 = z
        zs[k - 1] = z[pick]; ys[k - 1] = y[pick]
    cls = np.where(land_k > 0, (land_k - 1) // 6, NB)
    probs = np.bincount(cls, minlength=NB + 1) / N
    # paths: storm flag from the current storm end and the sampled particle's landing
    hours = (h + np.arange(1, HOR + 1)) * 60
    flag = np.zeros((HOR, n_paths))
    if in_st:
        flag[hours < cur_end] = 1
    for j, p in enumerate(pick):
        if land_k[p] > 0:
            lm = (h + land_k[p]) * 60
            em = storm_end_min(profile, i_idx, lm)
            flag[(hours >= lm) & (hours < em), j] = 1
    st = flag > 0
    pr = (p_mean(zs) + rng.integers(-NZ[noise]["pressure"], NZ[noise]["pressure"] + 1, zs.shape)) / 10.0
    wd = np.maximum(wind_mean(ys, st) + rng.integers(-NZ[noise]["wind"], NZ[noise]["wind"] + 1, ys.shape), 0)
    paths = np.stack([pr, wd, flag], -1).transpose(1, 0, 2)  # (n_paths, 168, 3)
    return probs, paths, land_k


def run_seed(d, profile, noise, N, jitter, reseed, bank, rng, issue=None, want_paths=True):
    issue = set(issue if issue is not None else issue_hours())
    rs = reading_storm(d[None])[0]
    nb = len(bank["x"])
    def draw(n):
        j = rng.integers(0, nb, n)
        return bank["x"][j].copy(), bank["y"][j].copy(), bank["z"][j].copy(), bank["zh"][j].copy()
    x, y, z, zh = draw(N)
    h1 = zh[:, -2].copy(); h2 = zh[:, -3].copy()  # bank zh ends at "now"; treat as state before hour 1's step? (burn-in absorbs)
    out_p, out_paths, ess_log, lost = [], [], [], 0
    cur_land = None
    for h in range(1, H + 1):
        r = d[h - 1]
        h2 = h1; h1 = z
        x, y, z = lorenz_hour(x, y, z)
        zh = np.concatenate([zh[:, 1:], z[:, None]], 1)
        w = obs_lik(r, x, y, z, zh, bool(rs[h - 1]), is_sign_hour(h), noise)
        sw = w.sum()
        if sw <= 0:
            lost += 1
            # full re-seed from the bank, keep those consistent with this reading
            for _ in range(50):
                bx, by, bz, bzh = draw(N * 8)
                bw = obs_lik(r, bx, by, bz, bzh, bool(rs[h - 1]), is_sign_hour(h), noise)
                if bw.sum() > 0:
                    break
            if bw.sum() <= 0:
                bw = np.ones(len(bx))
            idx_ = rng.choice(len(bx), N, p=bw / bw.sum())
            x, y, z, zh = bx[idx_], by[idx_], bz[idx_], bzh[idx_]
            h1 = zh[:, -2].copy(); h2 = zh[:, -3].copy()
            w = np.ones(N); sw = N
        wn = w / sw
        ess_log.append(1.0 / (wn ** 2).sum())
        # systematic resampling
        u = (rng.random() + np.arange(N)) / N
        ii = np.minimum(np.searchsorted(np.cumsum(wn), u), N - 1)
        x, y, z, zh, h1, h2 = x[ii], y[ii], z[ii], zh[ii], h1[ii], h2[ii]
        # LSB jitter (Lorenz units) on x, y, z
        if jitter > 0:
            js = int(jitter * S)
            x = x + rng.integers(-js, js + 1, N); y = y + rng.integers(-js, js + 1, N); z = z + rng.integers(-js, js + 1, N)
            zh[:, -1] = z
        # climatology re-seeding
        if reseed > 0:
            nr = rng.binomial(N, reseed)
            if nr:
                k = rng.choice(N, nr, replace=False)
                bx, by, bz, bzh = draw(nr)
                x[k], y[k], z[k], zh[k] = bx, by, bz, bzh
                h1[k] = bzh[:, -2]; h2[k] = bzh[:, -3]
        if h in issue:
            idx, after_end, ins = ctx_at(d, h)
            cur_end = None
            if ins:
                lh = np.nonzero(d[:h, FI["landed"]])[0][-1] + 1
                cur_end = storm_end_min(profile, idx, lh * 60)
            p, paths, _ = forecast(profile, noise, x.copy(), y.copy(), z.copy(), z.copy(), h1.copy(), h, idx, after_end, ins, cur_end, rng)
            out_p.append(p)
            if want_paths:
                out_paths.append(paths.astype(np.float32))
    return dict(probs=np.array(out_p), paths=np.array(out_paths) if want_paths else None,
                ess=np.array(ess_log), lost=lost)
