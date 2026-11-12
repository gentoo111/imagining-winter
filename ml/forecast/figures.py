"""Figures for E1-RESULTS.md: BSS by lead, reliability (72 h), example issue (diffusion vs PF vs truth).
S1-5: E200=1 reads out/eval200/ (summary_<p>_<n>.json with diff = v2, summary_<p>_<n>_v3s1.json for v3 training seed 1;
npz prefixes diff2 / diff3s1) and writes the paper figures docs/paper/figures/e1_bss_<p>_<n>.png and
e1_example_<p>_<n>.png (v3 seed 1, hourly paths `paths0` of test seed 0) plus out/eval200/example_<p>_<n>.json;
the 72 h reliability figure is figures_paper.py reliability."""
import sys, os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
from metrics import LEADS
prof, noise = sys.argv[1], sys.argv[2]
E200 = os.environ.get("E200") == "1"
OUTD = "out/eval200" if E200 else "out"
IMG = os.path.join(ROOT, "docs", "paper" if E200 else "ml", "figures" if E200 else "img"); os.makedirs(IMG, exist_ok=True)
S = json.load(open(f"{OUTD}/summary_{prof}_{noise}.json"))
COL = {"clim": "#999999", "clock": "#7a6f5a", "bot": "#b08a3e", "gru": "#3b6ea8", "diff": "#c0392b", "diff3": "#8e44ad", "pf": "#2e8b57"}
NAME = {"clim": "climatology", "clock": "clock hazard", "bot": "bot sign rule", "gru": "GRU (categorical)", "diff": "diffusion (paths)", "pf": "particle filter (oracle)"}
PREFIX = {}
if E200:
    S["diff3"] = json.load(open(f"{OUTD}/summary_{prof}_{noise}_v3s1.json"))["diff"]
    S = {m: S[m] for m in ("clim", "clock", "bot", "gru", "diff", "diff3", "pf") if m in S}
    NAME.update(diff="diffusion v2 (U-Net, 88k steps)", diff3="diffusion v3 (transformer, training seed 1)")
    PREFIX = {"diff": "diff2", "diff3": "diff3s1"}
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for j, base in enumerate(("clim", "clock")):
    for m in S:
        k = f"BSS{LEADS[0]}_{base}"
        if k not in S[m] or m == base:
            continue
        v = np.array([S[m][f"BSS{L}_{base}"] for L in LEADS])
        ax[j].plot(LEADS, v[:, 0], "-o", color=COL.get(m), label=NAME.get(m, m), ms=4)
        ax[j].fill_between(LEADS, v[:, 1], v[:, 2], color=COL.get(m), alpha=.15)
    ax[j].axhline(0, color="k", lw=.6); ax[j].set_xlabel("lead (h): event = first landing within lead")
    ax[j].set_ylabel(f"Brier skill vs {NAME[base]}"); ax[j].set_xticks(LEADS); ax[j].set_ylim(-0.5, 1.0)
ax[0].legend(fontsize=8)
fig.suptitle(f"E1 storm-landing forecasts, {prof} / {noise} ({'200 test seeds, 200 paths, DDIM 50' if E200 else 'test seeds'}, daily 07:00 issues, 95% CI)")
fig.tight_layout(); fig.savefig(os.path.join(IMG, f"e1_bss_{prof}_{noise}.png"), dpi=130); plt.close(fig)


def loadm(m):
    fs = sorted(glob.glob(f"{OUTD}/{PREFIX.get(m, m)}_{prof}_{noise}*.npz"))
    if not fs:
        return None
    zs = [np.load(f) for f in fs]
    return dict(P=np.concatenate([z["P"] for z in zs]), Y=np.concatenate([z["Y"] for z in zs]),
                S=np.concatenate([z["seeds"] for z in zs]),
                paths=np.concatenate([z["paths"] for z in zs]) if all(z["paths"].size for z in zs) else None,
                paths0=zs[0]["paths0"] if "paths0" in zs[0].files else None)


R = {m: loadm(m) for m in S}
if not E200:
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], "k:", lw=.8)
    edges = np.array([0, .05, .2, .4, .6, .8, .95, 1.0001])
    for m, v in R.items():
        if v is None or m == "clim":
            continue
        p = v["P"][..., :12].sum(-1).ravel(); o = (v["Y"] < 12).ravel()
        xs, ys = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            k = (p >= a) & (p < b)
            if k.sum() >= 20:
                xs.append(p[k].mean()); ys.append(o[k].mean())
        ax.plot(xs, ys, "-o", color=COL.get(m), label=NAME.get(m, m), ms=4)
    ax.set_xlabel("forecast P(landing within 72 h)"); ax.set_ylabel("observed frequency"); ax.legend(fontsize=7)
    ax.set_title(f"Reliability, 72 h ({prof}/{noise})", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(IMG, f"e1_reliability72_{prof}_{noise}.png"), dpi=130); plt.close(fig)
# example issue: pick a test issue with a landing at ~3 days where both have paths
EX = "diff3" if E200 else "diff"  # E200: v3 training seed 1, hourly paths of test seed 0 (paths0)
dv = R.get(EX)
if dv is not None and E200 and dv["paths0"] is not None:
    dv = dict(dv, paths=dv["paths0"][None])  # (1, issues, M, HOR, 3): test seed index 0 only
if dv is not None and dv["paths"] is not None and R.get("pf") is not None and R["pf"]["paths"] is not None:
    D = load(prof, noise, "test"); ih = issue_hours()
    pv = R["pf"]
    i = 0; s = int(dv["S"][i])
    pj = int(np.nonzero(pv["S"] == s)[0][0])
    js = [j for j in range(len(ih)) if 8 <= dv["Y"][i, j] <= 14]
    j = js[0] if js else 10
    h = ih[j]; tp = true_path(D[s], h)
    fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    hrs = np.arange(1, HOR + 1)
    for k in range(20):
        ax[0].plot(hrs, dv["paths"][i, j, k, :, 0], color=COL[EX], alpha=.25, lw=.7)
        ax[0].plot(hrs, pv["paths"][pj, j, k, :, 0], color=COL["pf"], alpha=.25, lw=.7)
    ax[0].plot(hrs, tp[:, 0], "k", lw=1.4, label="truth"); ax[0].set_ylabel("pressure (hPa)")
    pd_ = dv["paths"][i, j, :, :, 2].astype(float).mean(0); pp_ = pv["paths"][pj, j, :, :, 2].astype(float).mean(0)
    ax[1].plot(hrs, pd_, color=COL[EX], label=("diffusion v3" if E200 else "diffusion") + " P(storm)")
    ax[1].plot(hrs, pp_, color=COL["pf"], label="PF P(storm)")
    ax[1].plot(hrs, tp[:, 2], "k", lw=1.2, label="truth"); ax[1].set_ylabel("storm"); ax[1].set_xlabel("hours after issue")
    ax[1].legend(fontsize=8)
    ax[0].set_title(f"Test seed {220000 + s}, issue day {h // 24}: 20 sampled futures "
                    f"({'purple = diffusion v3' if E200 else 'red = diffusion'}, green = PF)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(IMG, f"e1_example_{prof}_{noise}.png"), dpi=130); plt.close(fig)
    if E200:  # the numbers the paper's text quotes about this figure
        land = int(np.argmax(tp[:, 2] > 0)) + 1 if (tp[:, 2] > 0).any() else None
        ex = {"test_seed": 220000 + s, "issue_day": int(h // 24), "issue_hour": int(h), "landing_hours_after_issue": land,
              "model": "diff3s1", "n_paths_diff": int(dv["paths"].shape[2]), "n_paths_pf": int(pv["paths"].shape[2]),
              "peak_p_storm_diff": float(pd_.max()), "peak_p_storm_pf": float(pp_.max()),
              "p_storm_at_landing_diff": float(pd_[land - 1]) if land else None,
              "p_storm_at_landing_pf": float(pp_[land - 1]) if land else None}
        json.dump(ex, open(f"{OUTD}/example_{prof}_{noise}.json", "w"), indent=1)
        print(json.dumps(ex))
print("figures written to", IMG)
