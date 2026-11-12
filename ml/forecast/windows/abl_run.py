"""S1-7 v1->v2 ablation, one queue job per variant (jobs\\s17_<n>_abl<X>.cmd):  abl_run.py A|B|C|D
  A = v1; B = A + hour-position / known-storm-remainder channels; C = B + v-prediction; D = C + flag loss x3 = v2
  (models.VARIANTS). Each job: train 12k steps on the GPU (~17 min at 0.087 s/step) -> evaluate 60 test seeds
  (DDIM 25, 50 paths) -> summary_abl<X>.json in summarize.py's layout ({"diff": <this variant>, "clim", "clock"})
  -> update done.json ("runs_done" = variants finished). train.py cannot resume; one job per variant stays far
  below the queue's 3 h cap.
Code: the S1-7 models/train/evaluate.py are deployed to ml\\forecast\\windows\\abl\\ (s1-2's queued jobs keep using
ml\\forecast\\*.py); they run from there with PYTHONPATH=ml\\forecast, checked against abl\\SHA256SUMS.
Writes D:\\LastEmbers\\e1\\out\\ablation\\. ABL_OUT / ABL_NS / ABL_STEPS / ABL_DDIM only for a dry run."""
import hashlib, json, os, shutil, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FC = os.path.dirname(HERE)
ABL = os.path.join(HERE, "abl")
sys.path.insert(0, FC)
from common import NB, issue_hours, load, true_path  # noqa: E402
from metrics import LEADS, boot_ci, brier, crps_cat, energy_score, logloss, path_vec, variogram_score  # noqa: E402

E1 = os.path.abspath(os.path.join(FC, "..", ".."))
OUT = os.environ.get("ABL_OUT", os.path.join(E1, "out", "ablation"))
NS = int(os.environ.get("ABL_NS", "60"))
STEPS = int(os.environ.get("ABL_STEPS", "12000"))
DDIM = int(os.environ.get("ABL_DDIM", "25"))
PROF, NOISE = "endurance", "murky"
ADDS = {"A": "v1", "B": "+ position / remainder channels", "C": "+ v-prediction", "D": "+ flag weight x3 (= v2)"}
X = sys.argv[1]
assert X in ADDS, X
os.makedirs(OUT, exist_ok=True); os.makedirs(os.path.join(ABL, "out"), exist_ok=True)

# the deployed S1-7 code must be exactly the committed one
sums = dict(ln.split()[::-1] for ln in open(os.path.join(ABL, "SHA256SUMS")) if ln.strip())
for f in ("models.py", "train.py", "evaluate.py"):
    h = hashlib.sha256(open(os.path.join(ABL, f), "rb").read()).hexdigest()
    assert sums[f] == h, (f, sums[f], h)
ENV = {**os.environ, "PYTHONPATH": FC, "DIFF_VARIANT": X, "DEVICE": os.environ.get("DEVICE", "cuda"),
       "DDIM": str(DDIM), "PYTHONUTF8": "1"}
started = time.strftime("%Y-%m-%d %H:%M:%S")

# 1. train
pt = os.path.join(OUT, f"abl{X}.pt")
for f in (pt, pt + ".log", os.path.join(OUT, f"summary_abl{X}.json")):
    if os.path.exists(f):
        os.remove(f)  # train.py appends to its log
t0 = time.time()
subprocess.run([sys.executable, os.path.join(ABL, "train.py"), "diff", PROF, NOISE, str(STEPS), pt], cwd=FC, check=True, env=ENV)
train_secs = time.time() - t0
recs = [json.loads(ln) for ln in open(pt + ".log")]
assert recs[-1]["it"] == STEPS, recs[-1]
best = min(recs, key=lambda r: r["val"])


# 2. evaluate (evaluate.py's fixed output name under abl\out -> ours)
def run(meth, name, model=None):
    dst = os.path.join(OUT, f"{name}.npz")
    if meth in ("clim", "clock") and os.path.exists(dst):
        return np.load(dst)
    args = [sys.executable, os.path.join(ABL, "evaluate.py"), meth, PROF, NOISE, str(NS)] + ([model] if model else [])
    t = time.time()
    subprocess.run(args, cwd=FC, check=True, env=ENV)
    shutil.move(os.path.join(ABL, "out", f"{meth}_{PROF}_{NOISE}.npz"), dst)
    print(name, f"{time.time() - t:.0f}s", flush=True)
    return np.load(dst)


Z = {"clim": run("clim", "clim"), "clock": run("clock", "clock")}
t1 = time.time()
Z["diff"] = run("diff", f"abl{X}", pt)
eval_secs = time.time() - t1
for m in ("clock", "diff"):
    assert (Z[m]["seeds"] == Z["clim"]["seeds"]).all() and (Z[m]["Y"] == Z["clim"]["Y"]).all(), m
ns = len(Z["clim"]["seeds"])


# 3. summary, same algorithm as summarize.py (per-seed mean Brier, 1000 seed bootstraps, rng seed 0)
def bss(z, zb, L):
    bm = brier(z["P"], z["Y"], L).mean(1); bb = brier(zb["P"], zb["Y"], L).mean(1)
    rng = np.random.default_rng(0); bs = []
    for _ in range(1000):
        k = rng.integers(0, ns, ns); bs.append(1 - bm[k].mean() / bb[k].mean())
    return [float(1 - bm.mean() / bb.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


res = {}
for m, z in Z.items():
    res[m] = {f"BSS{L}_{b}": bss(z, Z[b], L) for L in LEADS for b in ("clim", "clock")}
    res[m]["CRPS"] = list(map(float, boot_ci([crps_cat(z["P"][i], z["Y"][i]) for i in range(ns)])))
    res[m]["LL"] = list(map(float, boot_ci([logloss(z["P"][i], z["Y"][i]) for i in range(ns)])))
paths = Z["diff"]["paths"].astype(np.float32)  # (ns, issues, 50, 168, 3) physical units
D = load(PROF, NOISE, "test"); ih = issue_hours()
es, vs = [], []
for i, s in enumerate(Z["diff"]["seeds"]):
    e1, v1 = [], []
    for j, h in enumerate(ih):
        Xp = path_vec(paths[i, j]); y = path_vec(true_path(D[s], h))
        e1.append(energy_score(Xp, y)); v1.append(variogram_score(Xp, y))
    es.append(np.array(e1)); vs.append(np.array(v1))
res["diff"]["ES"] = list(map(float, boot_ci(es))); res["diff"]["VS"] = list(map(float, boot_ci(vs)))
# diagnostic only (the paper's 0.31/0.92 and 0.23/0.004 script is not archived): share of sampled paths whose storm
# flag is on in future hour 1, at issue times inside a storm (target ~1) and in calm (target ~0)
from batch import Split  # noqa: E402
sp = Split(PROF, NOISE, "test", NS)
ins = np.stack([sp.batch(np.full(len(ih), s), np.array(ih), want_future=False)[1][:, 3] > 0 for s in Z["diff"]["seeds"]])
h1 = paths[..., 0, 2].mean(-1)  # (ns, issues): share of the 50 paths with flag on at hour 1
h1_rates = {"storm_continues_h1": float(h1[ins].mean()) if ins.any() else None, "n_storm_issues": int(ins.sum()),
            "calm_onset_h1": float(h1[~ins].mean()), "n_calm_issues": int((~ins).sum())}
summary = {**res, "variant": X, "adds": ADDS[X], "seeds": ns, "seed_ids": Z["clim"]["seeds"].tolist(), "ddim": DDIM,
           "paths": int(paths.shape[2]), "h1_rates": h1_rates,
           "train": {"steps": STEPS, "best_step": best["it"], "best_val_loss": best["val"],
                     "final_val_loss": recs[-1]["val"], "sec_per_step": round(train_secs / STEPS, 4),
                     "curve": [{"step": r["it"], "val_loss": r["val"], "train_loss": r["loss"]} for r in recs]},
           "eval_secs": round(eval_secs), "model": f"out/ablation/abl{X}.pt (models.VARIANTS['{X}'], best val)",
           "started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
json.dump(summary, open(os.path.join(OUT, f"summary_abl{X}.json"), "w"), indent=1)
print(X, ADDS[X], "BSS72", res["diff"]["BSS72_clim"], "BSS168", res["diff"]["BSS168_clim"], h1_rates, flush=True)

# 4. done.json: variants with a finished summary
fin = {}
for v in ADDS:
    f = os.path.join(OUT, f"summary_abl{v}.json")
    if os.path.exists(f):
        s = json.load(open(f))
        fin[v] = {"adds": ADDS[v], "BSS72_clim": s["diff"]["BSS72_clim"][0], "BSS168_clim": s["diff"]["BSS168_clim"][0],
                  "finished": s["finished"]}
json.dump({"runs_done": len(fin), "variants": fin, "steps": STEPS, "seeds": ns, "ddim": DDIM,
           "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, open(os.path.join(OUT, "done.json"), "w"), indent=1)
print("runs_done", len(fin), flush=True)
