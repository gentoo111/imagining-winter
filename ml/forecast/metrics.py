import numpy as np
from common import NB, HOR

LEADS = [12, 24, 48, 72, 96, 120, 144, 168]


def cum(p, L):
    return p[..., :L // 6].sum(-1)


def brier(p, y, L):
    o = (y < L // 6).astype(float)
    return (cum(p, L) - o) ** 2


def crps_cat(p, y):
    """CRPS over ordered landing bins (6 h each, 'none' = bin NB), in hours."""
    F = np.cumsum(p, -1)
    H = (np.arange(NB + 1)[None, :] >= y[:, None]).astype(float)
    return ((F - H) ** 2).sum(-1) * 6.0


def w1(p, q):
    return np.abs(np.cumsum(p, -1) - np.cumsum(q, -1)).sum(-1) * 6.0


def kl(p, q, eps=1e-4):
    p = np.maximum(p, eps); q = np.maximum(q, eps)
    p = p / p.sum(-1, keepdims=True); q = q / q.sum(-1, keepdims=True)
    return (p * np.log(p / q)).sum(-1)


def logloss(p, y):
    return -np.log(np.maximum(p[np.arange(len(y)), y], 1e-4))


def energy_score(X, y):
    """X: (m, D) samples, y: (D,)"""
    a = np.linalg.norm(X - y[None], axis=1).mean()
    b = np.linalg.norm(X[:, None] - X[None], axis=2).mean()
    return a - 0.5 * b


def variogram_score(X, y, p=0.5):
    vy = np.abs(y[:, None] - y[None]) ** p
    vx = (np.abs(X[:, :, None] - X[:, None, :]) ** p).mean(0)
    return ((vy - vx) ** 2).sum()


def path_vec(paths):
    """(..., 168, 3) physical units (hPa, m/s, flag) -> daily means, scaled: pressure/5 hPa, wind/3, flag."""
    s = paths.reshape(*paths.shape[:-2], 7, 24, 3).mean(-2)
    s = s / np.array([5.0, 3.0, 1.0])
    return s.reshape(*s.shape[:-2], 21)


def boot_ci(per_seed_vals, n=1000, seed=0):
    """per_seed_vals: list of arrays (one per seed) -> mean and 95% CI by resampling seeds."""
    rng = np.random.default_rng(seed)
    m = np.array([v.mean() for v in per_seed_vals]); w = np.array([len(v) for v in per_seed_vals])
    tot = (m * w).sum() / w.sum()
    bs = []
    for _ in range(n):
        k = rng.integers(0, len(m), len(m))
        bs.append((m[k] * w[k]).sum() / w[k].sum())
    return tot, np.percentile(bs, 2.5), np.percentile(bs, 97.5)
