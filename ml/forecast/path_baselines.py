"""Reference trajectory forecasts for the energy / variogram scores (RQ1, tab:scores).
Each forecast is 50 real 168-hour futures (observed pressure, wind, storm flag) taken from training weather:
  clim    : 50 training futures issued at the same hour of the episode (calendar climatology);
  clock   : 50 training futures whose clock state (baselines.clock_key: time since the last storm / time left
            in the current one) equals the test issue's (clock analogues);
  gru     : landing class drawn from the GRU's 29-class forecast, then a training future with that first-landing
            class and the same clock state (fallback: same class only). Its event marginal is the GRU's; its
            path structure is real weather, so it is a strong 'event model + analogue' path generator.
Scores on the same test seeds/issues and the same 21-d daily-mean vectors as summarize.py.
  path_baselines.py <n_test_seeds> [n_train_seeds]  -> out/pathbase_endurance_murky_<n>.json
S1-5 options (unset = unchanged): PB_M=<paths per forecast> (default 50; 200 = the diffusion forecasts' ensemble size),
PB_GRU=<GRU npz> (default out/gru_endurance_murky.npz), PB_OUT=<output dir> (default out/)."""
import json, os, sys, time
import numpy as np
from common import *
from metrics import path_vec, energy_score, variogram_score, boot_ci
import baselines as B

PROF, NOISE = "endurance", "murky"
M = int(os.environ.get("PB_M", "50"))
OUTD = os.environ.get("PB_OUT") or os.path.join(HERE, "out")
ns = int(sys.argv[1]); ntr = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
t0 = time.time()
ih = issue_hours()
TR = load(PROF, NOISE, "train")[:ntr]
TE = load(PROF, NOISE, "test")[:ns]
gru = np.load(os.environ.get("PB_GRU") or os.path.join(HERE, "out", f"gru_{PROF}_{NOISE}.npz"))
assert (gru["seeds"][:ns] == np.arange(ns)).all()
# training futures as daily-mean vectors, with clock key and true first-landing class
V = np.zeros((ntr, len(ih), 21), np.float32); K = {}; KC = {}; C = {}
for s in range(ntr):
    for j, h in enumerate(ih):
        V[s, j] = path_vec(true_path(TR[s], h))
        k = B.clock_key(TR[s], h, PROF); c = true_class(TR[s], h)
        K.setdefault(k, []).append((s, j)); KC.setdefault((k, c), []).append((s, j)); C.setdefault(c, []).append((s, j))
K = {k: np.array(v) for k, v in K.items()}; KC = {k: np.array(v) for k, v in KC.items()}; C = {k: np.array(v) for k, v in C.items()}
print(f"train vectors {time.time()-t0:.0f}s, {len(K)} clock keys", flush=True)
rng = np.random.default_rng(0)
res = {m: ([], []) for m in ("clim", "clock", "gru")}
fallback = {"gru_class_only": 0, "gru_any": 0, "clock_any": 0}
for s in range(ns):
    es = {m: [] for m in res}; vs = {m: [] for m in res}
    for j, h in enumerate(ih):
        y = path_vec(true_path(TE[s], h))
        k = B.clock_key(TE[s], h, PROF)
        X = {"clim": V[rng.integers(0, ntr, M), j]}
        if k in K:
            pick = K[k][rng.integers(0, len(K[k]), M)]
        else:
            fallback["clock_any"] += 1; pick = np.c_[rng.integers(0, ntr, M), np.full(M, j)]
        X["clock"] = V[pick[:, 0], pick[:, 1]]
        p = gru["P"][s, j].astype(np.float64); p /= p.sum()
        cls = rng.choice(NB + 1, M, p=p); rows = []
        for c in cls:
            pool = KC.get((k, int(c)))
            if pool is None:
                pool = C.get(int(c)); fallback["gru_class_only"] += 1
            if pool is None:
                pool = K.get(k, C[NB]); fallback["gru_any"] += 1
            rows.append(pool[rng.integers(0, len(pool))])
        rows = np.array(rows); X["gru"] = V[rows[:, 0], rows[:, 1]]
        for m in res:
            es[m].append(energy_score(X[m].astype(np.float64), y)); vs[m].append(variogram_score(X[m].astype(np.float64), y))
    for m in res:
        res[m][0].append(np.array(es[m])); res[m][1].append(np.array(vs[m]))
    if s % 20 == 0:
        print(s, f"{time.time()-t0:.0f}s", flush=True)
out = {m: {"ES": list(map(float, boot_ci(v[0]))), "VS": list(map(float, boot_ci(v[1])))} for m, v in res.items()}
out["fallback"] = fallback; out["n_test_seeds"] = ns; out["m_paths"] = M; out["n_train_seeds"] = ntr; out["secs"] = time.time() - t0
for m in ("clim", "clock", "gru"):
    print(m, "ES %.3f [%.3f,%.3f]" % tuple(out[m]["ES"]), "VS %.2f [%.2f,%.2f]" % tuple(out[m]["VS"]))
print(fallback)
json.dump(out, open(os.path.join(OUTD, f"pathbase_{PROF}_{NOISE}_{ns}.json"), "w"), indent=1)
