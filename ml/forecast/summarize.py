"""Tables: BSS by lead vs climatology and vs clock, CRPS, logloss, W1/KL to PF, path scores.
  summarize.py <profile> <noise> [methods...]  (uses the common seed subset of the available npz files)
S1-5: a method may be given as key=prefix (e.g. diff=diff3s1: result key diff, files diff3s1_<profile>_<noise>*.npz);
SUM_DIR=<dir> reads the npz files from / writes the summary to <dir> (default out/), SUM_TAG=<s> -> summary_<p>_<n><s>.json;
S1-12: SUM_SEEDS=a:b keeps only test seeds a..b-1 of the common subset (unset = all, unchanged);
npz files with es / vs (evaluate.py PATH_SCORES=1: the same scores, computed where the paths were sampled) use them."""
import sys, os, glob, json
import numpy as np
from common import *
from metrics import *
prof, noise = sys.argv[1], sys.argv[2]
meths = sys.argv[3:] or ["clim", "clock", "bot", "gru", "diff", "pf"]
SRC = dict(a.split("=", 1) if "=" in a else (a, a) for a in meths)  # result key -> npz prefix
OUT = os.environ.get("SUM_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def loadm(m):
    fs = sorted(glob.glob(os.path.join(OUT, f"{m}_{prof}_{noise}*.npz")))
    if not fs:
        return None
    P, Y, S, PA, SC = [], [], [], [], []
    for f in fs:
        z = np.load(f)
        P.append(z["P"]); Y.append(z["Y"]); S.append(z["seeds"]); PA.append(z["paths"] if z["paths"].size else None)
        SC.append((z["es"], z["vs"]) if "es" in z.files else None)
    order = np.argsort(np.concatenate(S))
    P = np.concatenate(P)[order]; Y = np.concatenate(Y)[order]; S = np.concatenate(S)[order]
    PA = np.concatenate(PA)[order] if all(p is not None for p in PA) else None
    sc = (np.concatenate([a for a, _ in SC])[order], np.concatenate([b for _, b in SC])[order]) if all(SC) else None
    return dict(P=P, Y=Y, S=S, paths=PA, scores=sc)


R = {m: loadm(src) for m, src in SRC.items()}
R = {m: v for m, v in R.items() if v is not None}
common = sorted(set.intersection(*[set(v["S"].tolist()) for v in R.values()]))
if os.environ.get("SUM_SEEDS"):
    s0, s1 = map(int, os.environ["SUM_SEEDS"].split(":"))
    common = [s for s in common if s0 <= s < s1]
for m, v in R.items():
    k = np.isin(v["S"], common)
    for key in ("P", "Y", "S"):
        v[key] = v[key][k]
    if v["paths"] is not None:
        v["paths"] = v["paths"][k]
    if v["scores"] is not None:
        v["scores"] = tuple(a[k] for a in v["scores"])
Y = R[next(iter(R))]["Y"]
ns = len(common)
print(f"{prof} {noise}: {ns} test seeds x {Y.shape[1]} daily issues; storms in 7 d window: {(Y < NB).mean():.2f}")
rows = {}
ref = {}
for m, v in R.items():
    P = v["P"]
    rows[m] = {}
    for L in LEADS:
        b = brier(P, Y, L)  # (ns, ni)
        rows[m][f"BS{L}"] = [b[i] for i in range(ns)]
    rows[m]["CRPS"] = [crps_cat(P[i], Y[i]) for i in range(ns)]
    rows[m]["LL"] = [logloss(P[i], Y[i]) for i in range(ns)]
    if "pf" in R and m != "pf":
        rows[m]["W1pf"] = [w1(P[i], R["pf"]["P"][i]) for i in range(ns)]
        rows[m]["KLpf"] = [kl(R["pf"]["P"][i], P[i]) for i in range(ns)]
res = {}
for m in rows:
    res[m] = {}
    for L in LEADS:
        bm = np.array([x.mean() for x in rows[m][f"BS{L}"]])
        for base in ("clim", "clock"):
            if base in rows:
                bb = np.array([x.mean() for x in rows[base][f"BS{L}"]])
                rng = np.random.default_rng(0); bs = []
                for _ in range(1000):
                    k = rng.integers(0, ns, ns); bs.append(1 - bm[k].mean() / bb[k].mean())
                res[m][f"BSS{L}_{base}"] = (1 - bm.mean() / bb.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5))
    for key in ("CRPS", "LL", "W1pf", "KLpf"):
        if key in rows[m]:
            res[m][key] = boot_ci(rows[m][key])
# path scores
for m, v in R.items():
    if v["scores"] is not None:
        res[m]["ES"] = boot_ci(list(v["scores"][0].astype(np.float64))); res[m]["VS"] = boot_ci(list(v["scores"][1].astype(np.float64)))
        continue
    if v["paths"] is None:
        continue
    D = load(prof, noise, "test")
    ih = issue_hours()
    es, vs = [], []
    for i, s in enumerate(v["S"]):
        e1, v1 = [], []
        for j, h in enumerate(ih):
            X = path_vec(v["paths"][i, j].astype(np.float32)); y = path_vec(true_path(D[s], h))
            e1.append(energy_score(X, y)); v1.append(variogram_score(X, y))
        es.append(np.array(e1)); vs.append(np.array(v1))
    res[m]["ES"] = boot_ci(es); res[m]["VS"] = boot_ci(vs)


def f(t, d=3):
    return f"{t[0]:.{d}f} [{t[1]:.{d}f},{t[2]:.{d}f}]"


print("\nBSS vs climatology by lead (h):")
print("method  " + "  ".join(f"{L:>22}" for L in LEADS))
for m in res:
    print(f"{m:6}  " + "  ".join(f"{f(res[m][f'BSS{L}_clim'], 2):>22}" for L in LEADS if f"BSS{L}_clim" in res[m]))
if "clock" in rows:
    print("\nBSS vs clock baseline by lead (h):")
    for m in res:
        print(f"{m:6}  " + "  ".join(f"{f(res[m][f'BSS{L}_clock'], 2):>22}" for L in LEADS))
print("\nCRPS landing time (h) | logloss | W1 to PF (h) | KL(PF||m) | energy score | variogram score")
for m in res:
    print(f"{m:6}  " + " | ".join(f(res[m][k], 2) if k in res[m] else "-" for k in ("CRPS", "LL", "W1pf", "KLpf", "ES", "VS")))
json.dump({m: {k: list(map(float, v)) for k, v in res[m].items()} for m in res},
          open(os.path.join(OUT, f"summary_{prof}_{noise}{os.environ.get('SUM_TAG', '')}.json"), "w"), indent=1)
