"""S1-10: relative economic value V(C/L) of the storm-landing forecasts (cost-loss model), Endurance/murky.
Inputs: the S1-5 200-seed npz in out/eval200/ (clim, clock, bot, gru, diff2 = v2, diff3s{1,2,3} = v3 training seeds, pf).
Event: landing within L hours of issue, p = P[..., :L//6].sum(-1), o = Y < L//6 (as metrics.brier), L = 24/72/168 h.
Cost-loss model in units of the loss (alpha = C/L): protect (cost alpha) or not (loss 1 if the event occurs).
  E_fc = mean(a*alpha + (1-a)*o), E_clim = min(alpha, sbar), E_perfect = alpha*sbar,
  V = (E_clim - E_fc) / (E_clim - E_perfect).
face value : a = [p >= alpha] (the user takes the stated probability at face value; calibration matters)
envelope   : max over all thresholds t of V with a = [p >= t] (potential value, independent of calibration)
Bootstrap (1000, rng 0) over test seeds for the face-value V at a few alpha.
  value_curve.py  -> out/value_curve.json, docs/paper/figures/value_curve.png"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
from common import HERE, ROOT

SRC = os.path.join(HERE, "out", "eval200")
OUT = os.path.join(HERE, "out", "value_curve.json")
FIG = os.path.join(ROOT, "docs", "paper", "figures", "value_curve.png")
LEADS = (24, 72, 168)
ALPHA = np.exp(np.linspace(np.log(0.01), np.log(0.99), 99))
KEY_ALPHA = (0.02, 0.05, 0.1, 0.2, 0.5)
METHODS = (("clim", "clim"), ("clock", "clock"), ("bot", "bot"), ("gru", "gru"), ("v2", "diff2"),
           ("v3s1", "diff3s1"), ("v3s2", "diff3s2"), ("v3s3", "diff3s3"), ("pf", "pf"))
COL = {"clock": "#7a6f5a", "bot": "#b08a3e", "gru": "#3b6ea8", "v2": "#c0392b", "v3": "#8e44ad", "pf": "#2e8b57"}
NAME = {"clock": "clock hazard", "bot": "bot rule", "gru": "GRU", "v2": "diffusion v2", "v3": "diffusion v3", "pf": "PF oracle"}


def loadm(prefix):
    fs = sorted(glob.glob(os.path.join(SRC, f"{prefix}_endurance_murky*.npz")))
    assert fs, prefix
    zs = [np.load(f) for f in fs]
    S = np.concatenate([z["seeds"] for z in zs]); o = np.argsort(S)
    return dict(P=np.concatenate([z["P"] for z in zs])[o], Y=np.concatenate([z["Y"] for z in zs])[o], S=S[o])


def value(a, o, alpha, sbar):
    """a, o: (n,) protect / occur; alpha scalar; returns V."""
    e_fc = np.mean(a * alpha + (1 - a) * o)
    e_cl = min(alpha, sbar); e_pf = alpha * sbar
    return (e_cl - e_fc) / (e_cl - e_pf)


def curves(p, o):
    """p, o: (n,) flattened. Face-value and envelope V over ALPHA, via hit/false-alarm rates per threshold."""
    sbar = o.mean()
    face = np.array([value((p >= al).astype(float), o, al, sbar) for al in ALPHA])
    # envelope: all distinct thresholds -> (hit rate H, false-alarm rate F); E_fc = F(1-s)al + H s al + (1-H) s
    ts = np.unique(np.concatenate([p, [np.inf]]))
    order = np.argsort(-p); ps = p[order]; os_ = o[order]
    tp = np.cumsum(os_); fp = np.cumsum(1 - os_)
    # threshold t = ps[k]: protect all with p >= ps[k] -> take the last index of each tie group
    last = np.r_[ps[1:] != ps[:-1], True]
    Hh = np.r_[0.0, tp[last] / max(o.sum(), 1)]; Ff = np.r_[0.0, fp[last] / max((1 - o).sum(), 1)]
    env = []
    for al in ALPHA:
        e_fc = Ff * (1 - sbar) * al + Hh * sbar * al + (1 - Hh) * sbar
        e_cl = min(al, sbar); e_pf = al * sbar
        env.append(((e_cl - e_fc) / (e_cl - e_pf)).max())
    return sbar, face, np.array(env)


def main():
    R = {k: loadm(src) for k, src in METHODS}
    common = sorted(set.intersection(*[set(v["S"].tolist()) for v in R.values()]))
    for v in R.values():
        k = np.isin(v["S"], common)
        v["P"], v["Y"], v["S"] = v["P"][k], v["Y"][k], v["S"][k]
    ns, ni = R["pf"]["Y"].shape
    for v in R.values():
        assert (v["Y"] == R["pf"]["Y"]).all()
    out = dict(setting="endurance/murky", test_seeds=ns, issues=ni, leads=list(LEADS), alpha=ALPHA.tolist(),
               key_alpha=list(KEY_ALPHA), decision={"face": "protect iff p >= C/L", "envelope": "max over thresholds"},
               base_rate={}, face={}, envelope={}, key={})
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, ns, ns) for _ in range(1000)]
    for L in LEADS:
        b = L // 6
        o2 = (R["pf"]["Y"] < b).astype(float)
        out["base_rate"][str(L)] = float(o2.mean())
        out["face"][str(L)], out["envelope"][str(L)], out["key"][str(L)] = {}, {}, {}
        for m, v in R.items():
            p2 = v["P"][..., :b].sum(-1).astype(np.float64)
            _, face, env = curves(p2.ravel(), o2.ravel())
            out["face"][str(L)][m] = face.tolist(); out["envelope"][str(L)][m] = env.tolist()
            kk = {}
            for al in KEY_ALPHA:
                a2 = (p2 >= al).astype(float)
                v0 = value(a2.ravel(), o2.ravel(), al, o2.mean())
                bs = [value(a2[i].ravel(), o2[i].ravel(), al, o2[i].mean()) for i in boots]
                kk[str(al)] = [float(v0), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            out["key"][str(L)][m] = kk
        for d in ("face", "envelope"):
            out[d][str(L)]["v3med"] = np.median([out[d][str(L)][f"v3s{s}"] for s in (1, 2, 3)], 0).tolist()
        out["key"][str(L)]["v3med"] = {str(al): float(np.median([out["key"][str(L)][f"v3s{s}"][str(al)][0] for s in (1, 2, 3)]))
                                       for al in KEY_ALPHA}
    # range of C/L where each method has positive face value at 72 h (on the grid)
    out["positive_range_72"] = {}
    for m, f in out["face"]["72"].items():
        pos = ALPHA[np.array(f) > 1e-9]
        out["positive_range_72"][m] = [float(pos.min()), float(pos.max())] if pos.size else None
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7})
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
    for c, L in enumerate(LEADS):
        a = ax[c]; F = out["face"][str(L)]
        a.axhline(0, color="k", lw=.5)
        for m in ("clock", "bot", "gru", "v2"):
            a.plot(ALPHA, F[m], color=COL[m], lw=1.1, label=NAME[m])
        v3 = np.array([F[f"v3s{s}"] for s in (1, 2, 3)])
        a.fill_between(ALPHA, v3.min(0), v3.max(0), color=COL["v3"], alpha=.25, lw=0)
        a.plot(ALPHA, F["v3med"], color=COL["v3"], lw=1.1, label=NAME["v3"] + " (median, range)")
        a.plot(ALPHA, F["pf"], color=COL["pf"], lw=1.4, label=NAME["pf"])
        a.plot(ALPHA, out["envelope"][str(L)]["pf"], color=COL["pf"], lw=.8, ls="--", label="PF envelope")
        a.axvline(out["base_rate"][str(L)], color="#999999", lw=.6, ls=":")
        a.set_xscale("log"); a.set_xlim(0.01, 0.99); a.set_ylim(-0.1, 1.0)
        a.xaxis.set_major_locator(FixedLocator([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]))
        a.xaxis.set_major_formatter(FixedFormatter(["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1"]))
        a.xaxis.set_minor_formatter(NullFormatter())
        a.set_title(f"landing within {L} h (base rate {out['base_rate'][str(L)]:.2f})")
        a.set_xlabel("cost/loss ratio C/L")
    ax[0].set_ylabel("relative economic value V")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(FIG, dpi=200)
    print(f"{ns} seeds x {ni} issues; base rates", {k: round(v, 3) for k, v in out["base_rate"].items()})
    for L in LEADS:
        print(f"\nL={L} h face-value V at C/L " + " ".join(map(str, KEY_ALPHA)))
        for m, kk in out["key"][str(L)].items():
            print(f"  {m:6}", " ".join(f"{(x[0] if isinstance(x, list) else x):6.3f}" for x in kk.values()))
        print("  max face V:", {m: round(max(f), 3) for m, f in out["face"][str(L)].items()})
        print("  max env  V:", {m: round(max(f), 3) for m, f in out["envelope"][str(L)].items()})
    print("\npositive face-value C/L range at 72 h:", out["positive_range_72"])


if __name__ == "__main__":
    main()
