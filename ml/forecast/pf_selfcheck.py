"""PF oracle self-check: randomized PIT of the true landing class (should be uniform) and reliability of P(landing <= 72 h)."""
import sys, glob, numpy as np
from common import NB
prof, noise = sys.argv[1], sys.argv[2]
P, Y = [], []
for f in sorted(glob.glob(f"out/pf_{prof}_{noise}_p*.npz")):
    z = np.load(f); P.append(z["P"]); Y.append(z["Y"])
P = np.concatenate(P).reshape(-1, NB + 1); Y = np.concatenate(Y).ravel()
rng = np.random.default_rng(0)
F = np.cumsum(P, 1)
lo = np.where(Y > 0, F[np.arange(len(Y)), np.maximum(Y - 1, 0)], 0.0); hi = F[np.arange(len(Y)), Y]
pit = lo + rng.random(len(Y)) * (hi - lo)
h, _ = np.histogram(pit, bins=10, range=(0, 1))
print("PIT deciles (uniform = 0.10 each):", np.round(h / h.sum(), 3))
p72 = P[:, :12].sum(1); o = (Y < 12).astype(float)
print("reliability P(landing<=72h): bin  mean_p  obs_freq  n")
for a, b in [(0, .05), (.05, .2), (.2, .4), (.4, .6), (.6, .8), (.8, .95), (.95, 1.01)]:
    k = (p72 >= a) & (p72 < b)
    if k.sum():
        print(f"  [{a:.2f},{b:.2f})  {p72[k].mean():.3f}  {o[k].mean():.3f}  {k.sum()}")
