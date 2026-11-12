"""Paper figures/statistics not produced by figures.py:
  reliability : one multi-panel reliability figure, all methods, leads 24/72/168 h, with sample counts
                (S1-5: E200=1 reads out/eval200/, diffusion = v2 (diff2) and v3 training seed 1 (diff3s1))
  tables200   : S1-5: the generated block of main.tex (macros of tab:bss / tab:scores / RQ1 text) from out/eval200/*.json
  pit         : PF oracle PIT histograms (three settings) + rank histograms of PF path ensembles
  testbed     : 120-day weather trace of one test seed (z, observed pressure, wind, storms)
  stats       : storm statistics on the current weather_gen test split
  ladder      : E0 value-ladder bar chart from out/e0_summary_*.json
  storms      : E0 storm-focused metrics over three training seeds (per storm reached, per tier) from out/storm_seeds_*.json
  storms3b    : S1-11c: the generated rows of tab:storms / tab:ladder for rungs 3a' (fcg) and 3b (fcd), their paired rows
                (fcd-fcg, fcd-fc) and the macros of the 3b text, written between the e0b markers of main.tex
  learning_curve : S1-2 v2 diffusion learning curve (100k GPU steps) + BSS by lead, best vs 12k, from out/learning_curve/summary.json
  figures_paper.py [which ...]   -> docs/paper/figures/*.png, out/storm_stats.json"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

FIG = os.path.join(ROOT, "docs", "paper", "figures"); os.makedirs(FIG, exist_ok=True)
OUT = os.path.join(HERE, "out")
E200 = os.environ.get("E200") == "1"
COL = {"clim": "#999999", "clock": "#7a6f5a", "bot": "#b08a3e", "gru": "#3b6ea8", "diff": "#c0392b", "diff3": "#8e44ad", "pf": "#2e8b57"}
NAME = {"clim": "climatology", "clock": "clock hazard", "bot": "bot rule", "gru": "GRU", "diff": "diffusion", "diff3": "diffusion v3 (seed 1)", "pf": "PF oracle"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7})


def loadm(m, prof="endurance", noise="murky", paths=False):
    fs = sorted(glob.glob(os.path.join(OUT, f"{m}_{prof}_{noise}*.npz")))
    if not fs:
        return None
    zs = [np.load(f) for f in fs]
    S = np.concatenate([z["seeds"] for z in zs]); o = np.argsort(S)
    r = dict(P=np.concatenate([z["P"] for z in zs])[o], Y=np.concatenate([z["Y"] for z in zs])[o], S=S[o])
    if paths:
        r["paths"] = np.concatenate([z["paths"] for z in zs])[o]
    return r


def reliability():
    if E200:
        global OUT
        OUT = os.path.join(HERE, "out", "eval200")
        R = {m: loadm(src) for m, src in (("clim", "clim"), ("clock", "clock"), ("bot", "bot"), ("gru", "gru"),
                                          ("diff", "diff2"), ("diff3", "diff3s1"), ("pf", "pf"))}
        NAME["diff"] = "diffusion v2"
    else:
        R = {m: loadm(m) for m in ("clim", "clock", "bot", "gru", "diff", "pf")}
    common = sorted(set.intersection(*[set(v["S"].tolist()) for v in R.values()]))
    edges = np.array([0, .05, .15, .25, .35, .45, .55, .65, .75, .85, .95, 1.0001])
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.0), gridspec_kw={"height_ratios": [3, 1.1]}, sharex=True)
    for c, L in enumerate((24, 72, 168)):
        a, b = ax[0, c], ax[1, c]
        a.plot([0, 1], [0, 1], "k:", lw=.7)
        for m, v in R.items():
            k = np.isin(v["S"], common)
            p = v["P"][k][..., :L // 6].sum(-1).ravel(); o = (v["Y"][k] < L // 6).ravel()
            xs, ys, ns = [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                kk = (p >= lo) & (p < hi)
                ns.append(kk.sum())
                if kk.sum() >= 20:
                    xs.append(p[kk].mean()); ys.append(o[kk].mean())
            a.plot(xs, ys, "-o", color=COL[m], label=NAME[m], ms=2.5, lw=1)
            mid = (edges[:-1] + np.minimum(edges[1:], 1)) / 2
            b.plot(mid, np.maximum(ns, 1), "-", color=COL[m], lw=1, drawstyle="steps-mid")
        a.set_title(f"P(landing within {L} h)"); a.set_xlim(0, 1); a.set_ylim(0, 1)
        b.set_yscale("log"); b.set_xlabel("forecast probability")
        if c == 0:
            a.set_ylabel("observed frequency"); b.set_ylabel("forecasts")
    ax[0, 0].legend(loc="upper left", frameon=False)
    n = len(common) * R["pf"]["Y"].shape[1]
    fig.suptitle(f"Reliability, Endurance/murky, {len(common)} test seeds x 113 issues = {n} forecasts per method (bins with < 20 omitted)", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "reliability_all.png"), dpi=200); plt.close(fig)


def pit_rank():
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 1.9), sharey=False)
    rng = np.random.default_rng(0)
    for c, (prof, noise, lab) in enumerate((("endurance", "murky", "Endurance / murky"), ("endurance", "clear", "Endurance / clear"),
                                             ("a_new_home", "murky", "A New Home / murky"))):
        v = loadm("pf", prof, noise)
        P = v["P"].reshape(-1, NB + 1); Y = v["Y"].ravel()
        F = np.cumsum(P, 1)
        lo = np.where(Y > 0, F[np.arange(len(Y)), np.maximum(Y - 1, 0)], 0.0); hi = F[np.arange(len(Y)), Y]
        pit = lo + rng.random(len(Y)) * (hi - lo)
        h, _ = np.histogram(pit, bins=10, range=(0, 1)); f = h / h.sum()
        band = 1.96 * np.sqrt(0.1 * 0.9 / len(Y))
        ax[c].bar(np.arange(10) / 10 + .05, f, width=.09, color=COL["pf"])
        ax[c].axhspan(.1 - band, .1 + band, color="k", alpha=.15, lw=0); ax[c].axhline(.1, color="k", lw=.6)
        ax[c].set_ylim(0, .14); ax[c].set_title(f"PIT, {lab}\n({len(np.unique(v['S']))} seeds, n={len(Y)})")
        ax[c].set_xlabel("PIT of landing time")
    ax[0].set_ylabel("fraction")
    # rank histogram of the realised day-3 and day-7 mean pressure among the 50 PF paths (murky)
    v = loadm("pf", "endurance", "murky", paths=True)
    D = load("endurance", "murky", "test"); ih = issue_hours()
    for day, col in ((3, "#2e8b57"), (7, "#8fbc8f")):
        ranks = []
        for i, s in enumerate(v["S"]):
            for j, h in enumerate(ih):
                tp = true_path(D[s], h)[(day - 1) * 24:day * 24, 0].mean()
                ens = v["paths"][i, j, :, (day - 1) * 24:day * 24, 0].astype(np.float32).mean(-1)
                ranks.append((ens < tp).sum() + rng.integers(0, (ens == tp).sum() + 1))
        h, _ = np.histogram(ranks, bins=np.arange(0, 52, 5.1)); f = h / h.sum()
        ax[3].plot(np.arange(len(f)) + .5, f, "-o", color=col, ms=2.5, label=f"day {day}")
    ax[3].axhline(1 / 10, color="k", lw=.6); ax[3].set_ylim(0, .2); ax[3].set_xticks([])
    ax[3].set_title("PF rank hist., daily mean\npressure, day 3 / 7 (murky)"); ax[3].set_xlabel("rank of truth (50 paths)")
    ax[3].legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "pf_pit_rank.png"), dpi=200); plt.close(fig)


def testbed(seed_idx=0):
    D = load("endurance", "murky", "test")[seed_idx]
    t = np.arange(1, H + 1) / 24.0
    z = D[:, FI["lz"]] / S; p = D[:, FI["pressure"]] / 10.0; w = D[:, FI["wind_ms"]]; ins = D[:, FI["in_storm"]]
    land = np.nonzero(D[:, FI["landed"]])[0]
    fig, ax = plt.subplots(3, 1, figsize=(7.2, 3.6), sharex=True, gridspec_kw={"height_ratios": [1.2, 1, .8]})
    for a in ax:
        on = np.diff(np.r_[0, ins, 0]); st, en = np.nonzero(on == 1)[0], np.nonzero(on == -1)[0]
        for s0, e0 in zip(st, en):
            a.axvspan(t[s0], t[min(e0, H - 1)], color="#5b8db8", alpha=.25, lw=0)
    ax[0].plot(t, z, color="k", lw=.5); ax[0].axhline(CH["peak_c"] / 100, color="#c0392b", lw=.7, ls="--")
    ax[0].plot(t[land - 1], z[land - 1], "v", color="#c0392b", ms=4)
    ax[0].set_ylabel("Lorenz $z$\n(hidden)"); ax[0].text(t[-1], CH["peak_c"] / 100, " landing\n threshold", va="center", fontsize=6, color="#c0392b")
    ax[1].plot(t, p, color="#2e8b57", lw=.5); ax[1].set_ylabel("pressure\nreading (hPa)")
    ax[2].plot(t, w, color="#b08a3e", lw=.5); ax[2].set_ylabel("wind\n(m/s)"); ax[2].set_xlabel("game day")
    ax[0].set_title(f"Test seed {220000 + seed_idx} (Endurance, murky): {len(land)} storms in 120 days (shaded); "
                    f"triangles = landing at a $z$ peak above {CH['peak_c'] / 100:g}")
    ax[2].set_xlim(0, 120)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "testbed_trace.png"), dpi=200); plt.close(fig)


def stats():
    out = {}
    for prof in ("endurance", "a_new_home"):
        D = load(prof, "murky", "test")
        n60, gaps, first, total = [], [], [], []
        for d in D:
            land = (np.nonzero(d[:, FI["landed"]])[0] + 1) / 24.0
            total.append(len(land)); n60.append(len(land) / 2.0)
            if len(land):
                first.append(land[0])
            gaps += list(np.diff(land))
        out[prof] = dict(seeds=len(D), storms_per_120d=float(np.mean(total)), storms_per_60d=float(np.mean(n60)),
                         gap_p10_50_90=[float(x) for x in np.percentile(gaps, [10, 50, 90])],
                         first_landing_p10_50_90=[float(x) for x in np.percentile(first, [10, 50, 90])],
                         in_storm_fraction=float(D[:, :, FI["in_storm"]].mean()),
                         p72_issue=float(np.mean([[true_class(d, h) < 12 for h in issue_hours()] for d in D[:200]])))
        print(prof, out[prof])
    json.dump(out, open(os.path.join(OUT, "storm_stats.json"), "w"), indent=1)


def ladder():
    fs = os.path.join(OUT, "e0_summary_e0_eval_greedy_1k.json"); fs2 = os.path.join(OUT, "e0_summary_e0_eval_sample_1k.json")
    G = json.load(open(fs)); Sm = json.load(open(fs2)) if os.path.exists(fs2) else {}
    ags = [a for a in ("bot", "start", "none", "signs", "oracle") if a in G]
    lab = {"bot": "scripted\nbot", "start": "stage-11\nstart", "none": "1 no\nsigns", "signs": "2 raw\nsigns", "oracle": "4 oracle\ndate"}
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.2))
    for c, (key, yl) in enumerate((("days", "days survived (cap 120)"), ("deaths_per_storm", "deaths per storm"))):
        for k, (R, off, col, nm) in enumerate(((G, -.18, "#3b6ea8", "greedy"), (Sm, .18, "#c0392b", "sampled"))):
            xs = [i for i, a in enumerate(ags) if a in R]
            if not xs:
                continue
            v = np.array([R[ags[i]][key] for i in xs])
            ax[c].errorbar(np.array(xs) + off, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o", color=col, ms=3, capsize=2, label=nm)
        ax[c].set_xticks(range(len(ags))); ax[c].set_xticklabels([lab[a] for a in ags]); ax[c].set_ylabel(yl)
    ax[0].legend(frameon=False)
    fig.suptitle(f"E0 final checkpoints on {G[ags[0]]['n']} held-out seeds (mean, 95% bootstrap CI; one training seed per rung)", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "e0_ladder.png"), dpi=200); plt.close(fig)


def storms():
    """Three training seeds per rung (storm_seeds.py): per-seed points, across-seed mean; greedy and sampled."""
    G = json.load(open(os.path.join(OUT, "storm_seeds_greedy.json")))
    Sm = json.load(open(os.path.join(OUT, "storm_seeds_sample.json")))
    ags = [a for a in ("start", "none", "signs", "fc", "fcg", "fcd", "oracle") if a in G["rungs"]]
    lab = {"start": "stage-\n11", "none": "1 no\nsigns", "signs": "2 raw\nsigns", "fc": "3a\n+GRU", "fcg": "3a\u2032\nGRU",
           "fcd": "3b\ndiff.", "oracle": "4\noracle"}
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw=dict(width_ratios=[1.25, 1.25, 1]))
    for c, (key, yl) in enumerate((("deaths", "deaths per storm reached"), ("any_death", "P(any death in storm)"))):
        for R, off, col, nm in ((G, -.15, "#3b6ea8", "greedy"), (Sm, .15, "#c0392b", "sampled")):
            for i, a in enumerate(ags):
                if a not in R["rungs"]:
                    continue
                v = R["rungs"][a]["across"][key]
                ax[c].scatter(np.full(len(v["values"]), i + off), v["values"], s=7, color=col, alpha=.55, lw=0)
                ax[c].plot([i + off - .1, i + off + .1], [v["mean"]] * 2, color=col, lw=1.4, label=nm if i == 1 else None)
        ax[c].set_xticks(range(len(ags))); ax[c].set_xticklabels([lab[a] for a in ags], fontsize=6); ax[c].set_ylabel(yl)
    ax[0].legend(frameon=False, loc="upper right")
    tiers = ["shallow", "mid", "deep"]; w = .2
    rs = [a for a in ags if a != "start"]
    cols = {"none": "#999999", "signs": "#3b6ea8", "fc": "#b08a3e", "fcg": "#7a6f5a", "fcd": "#8e44ad", "oracle": "#2e8b57"}
    w = .8 / len(rs)
    for j, a in enumerate(rs):
        t = G["rungs"][a]["across"]["tier"]
        m = np.array([t[k]["mean"] for k in tiers]); lo = np.array([t[k]["min"] for k in tiers]); hi = np.array([t[k]["max"] for k in tiers])
        ax[2].errorbar(np.arange(3) + (j - (len(rs) - 1) / 2) * w, m, yerr=[m - lo, hi - m], fmt="o", ms=3, capsize=2,
                       color=cols[a], label=lab[a].replace("\n", " "))
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels(tiers); ax[2].set_ylabel("deaths per storm (greedy)")
    hi_ = ax[2].get_ylim()[1]; ax[2].set_ylim(-.1, hi_); ax[2].set_yticks([1, 2, 3])
    ax[2].legend(frameon=False, loc="lower center", ncol=2, columnspacing=.6, handletextpad=.2, fontsize=5.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "e0_storms.png"), dpi=200); plt.close(fig)


def _sg(x, d):
    """Signed number for the tables/text: $+$0.22, $-$4.7."""
    return ("$-$" if round(x, d) < 0 else "$+$") + f"{abs(x):.{d}f}"


def _ng(x, d):
    """Unsigned-style number that may be negative: 0.11, $-$0.02."""
    return ("$-$" if round(x, d) < 0 else "") + f"{abs(x):.{d}f}"


def storms3b():
    """S1-11c: rows and macros for rungs 3a' (fcg) and 3b (fcd) from out/storm_seeds_{greedy,sample}.json; replaces the
    text between '% BEGIN e0b<tag>' and '% END e0b<tag>' in main.tex (no hand-copied numbers)."""
    R = {m: json.load(open(os.path.join(OUT, f"storm_seeds_{m}.json"))) for m in ("greedy", "sample")}
    LAB = {"fcg": "3a$^{\\prime}$ Signs + daily GRU", "fcd": "3b Signs + diffusion"}
    PL = {"fcd-fcg": "3b$-$3a$^{\\prime}$", "fcd-fc": "3b$-$3a"}
    DIG = {"deaths": 2, "any_death": 3, "death_rate": 1, "day": 1}

    def rung_row(m, r):
        g = R[m]["rungs"][r]["across"]
        sd = lambda k, d: f"{g[k]['mean']:.{d}f} $\\pm$ {g[k]['sd']:.{d}f}"
        return (f"{LAB[r]} & {g['storms_per_episode']['mean']:.2f} & {sd('deaths', 2)} & {sd('any_death', 2)} & "
                f"{sd('death_rate', 1)} & {sd('day', 1)} \\\\")

    def pair_rows(m, k):
        P = R[m]["paired"][k]
        a = " & ".join(f"{_sg(P[x][0], DIG[x])} [{_ng(P[x][1], DIG[x])}, {_ng(P[x][2], DIG[x])}]" for x in DIG)
        b = " & ".join(f"({P[x + '_n_below0']}/{P[x + '_n_above0']})" for x in DIG)
        return [f"{PL[k]} & & {a} \\\\", f" & & {b} \\\\"]

    def ladder_row(r, label):
        p = R["greedy"]["rungs"][r]["per_seed"]["1"]
        ci = lambda v, d: f"{v[0]:.{d}f} [{v[1]:.{d}f}, {v[2]:.{d}f}]"
        return f"{label} & {ci(p['day_ci'], 1)} & {ci(p['alive_ci'], 1)} & {ci(p['deaths'], 2)} \\\\"

    blocks = {
        "-storms-greedy": [rung_row("greedy", r) for r in ("fcg", "fcd")],
        "-storms-sample": [rung_row("sample", r) for r in ("fcg", "fcd")],
        "-pairs-greedy": pair_rows("greedy", "fcd-fcg") + pair_rows("greedy", "fcd-fc"),
        "-pairs-sample": pair_rows("sample", "fcd-fcg") + pair_rows("sample", "fcd-fc"),
        "-ladder": [ladder_row("fcg", "3a$^{\\prime}$ Signs + daily GRU forecast"), ladder_row("fcd", "3b Signs + diffusion forecast")],
    }
    mac = {}
    MN = {"fcd": "Fcd", "fcg": "Fcg", "fc": "Fc"}
    for m, M in (("greedy", "G"), ("sample", "S")):
        for r in ("fcd", "fcg"):
            g = R[m]["rungs"][r]["across"]
            for k, nm, d in (("deaths", "Deaths", 2), ("day", "Days", 1), ("storms_per_episode", "Storms", 1), ("cap", "Cap", 0),
                             ("exiled", "Exiled", 0)):
                sc = 100 if k in ("cap", "exiled") else 1
                mac[f"eb{MN[r]}{nm}{M}"] = f"{g[k]['mean'] * sc:.{d}f}"
                mac[f"eb{MN[r]}{nm}{M}Min"] = f"{g[k]['min'] * sc:.{d}f}"
                mac[f"eb{MN[r]}{nm}{M}Max"] = f"{g[k]['max'] * sc:.{d}f}"
                mac[f"eb{MN[r]}{nm}{M}Runs"] = ", ".join(f"{v * sc:.{d}f}" for v in g[k]["values"])
            mac[f"eb{MN[r]}Any{M}"] = f"{100 * g['any_death']['mean']:.0f}"
        for k, nm in (("fcd-fcg", "FcdFcg"), ("fcd-fc", "FcdFc")):
            P = R[m]["paired"][k]
            for x, xn, d, sc in (("deaths", "Deaths", 2, 1), ("any_death", "Any", 1, 100), ("death_rate", "Rate", 1, 1), ("day", "Days", 1, 1)):
                mac[f"eb{nm}{xn}{M}"] = _sg(P[x][0] * sc, d)
                mac[f"eb{nm}{xn}{M}CI"] = f"[{_ng(P[x][1] * sc, d)}, {_ng(P[x][2] * sc, d)}]"
                mac[f"eb{nm}{xn}{M}Runs"] = ", ".join(_sg(v * sc, d) for v in P[x][3])
                mac[f"eb{nm}{xn}{M}Below"] = str(P[x + "_n_below0"]); mac[f"eb{nm}{xn}{M}Above"] = str(P[x + "_n_above0"])
    # tab:ladder rows: run 1 of every rung (greedy) and of none / signs / oracle (sampled)
    caps = [R["greedy"]["rungs"][r]["per_seed"]["1"]["cap"] for r in R["greedy"]["rungs"]]
    caps += [R["sample"]["rungs"][r]["per_seed"]["1"]["cap"] for r in ("none", "signs", "oracle")]
    mac["ebCapMaxRunOne"] = f"{100 * max(caps):.0f}"
    # forecast tables (s1-11a: out/fc3b/done.json) and the Windows PPO runs (s1-11b: out/ppo3b_done.json)
    FT = json.load(open(os.path.join(OUT, "fc3b", "done.json")))
    mac["ebIssueHour"] = f"{FT['issue_hour']:02d}:00"; mac["ebTableDdim"] = str(FT["ddim"]); mac["ebTablePaths"] = str(FT["paths"])
    mac["ebTableSeeds"] = f"{FT['train_seeds']:,}".replace(",", "{,}")
    mac["ebTableDiffHours"] = f"{FT['sec']['diff_v3'] / 3600:.1f}"; mac["ebTableGruMin"] = f"{FT['sec']['gru_daily'] / 60:.0f}"
    PD = json.load(open(os.path.join(OUT, "ppo3b_done.json")))
    hrs = [sum(g["sec"] for g in r["segments"]) / 3600 for r in PD["runs"]]  # recorded segments only (lower bound)
    mac["ebPpoHoursMin"] = f"{min(hrs):.0f}"; mac["ebPpoHoursMax"] = f"{max(hrs):.0f}"
    mac["ebStepsPerRound"] = f"{PD['target_steps_per_round']:,}".replace(",", "{,}")
    tex = os.path.join(ROOT, "docs", "paper", "main.tex")
    t = open(tex).read()
    src = "generated by ml/forecast/figures_paper.py storms3b from ml/forecast/out/storm_seeds_{greedy,sample}.json; do not edit"
    blocks[""] = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items()]
    for tag, rows in blocks.items():
        a = t.index(f"% BEGIN e0b{tag} "); a = t.index("\n", a) + 1
        b = t.index(f"% END e0b{tag}\n")
        t = t[:a] + "\n".join(rows) + "\n" + t[b:]
    open(tex, "w").write(t)
    print(f"{tex}: {len(mac)} macros, {sum(len(v) for k, v in blocks.items() if k)} table rows ({src})")


def learning_curve():
    """S1-2: validation loss every 2k steps of the 100k-step GPU run; BSS vs climatology by lead (60 test seeds)
    for the best-validation checkpoint and the paper's 12k-step CPU model."""
    S = json.load(open(os.path.join(OUT, "learning_curve", "summary.json")))
    c = S["curve"]; st = np.array([e["step"] for e in c]) / 1000
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4))
    ax[0].plot(st, [e["train_loss"] for e in c], color="#bbbbbb", lw=.8, label="train loss (last batch)")
    ax[0].plot(st, [e["val_loss"] for e in c], color=COL["diff"], lw=1.2, marker="o", ms=2, label="validation loss")
    b = S["best"]["step"] / 1000
    ax[0].axvline(12, color="#555555", ls=":", lw=.9); ax[0].axvline(b, color=COL["diff"], ls="--", lw=.9)
    ytop = ax[0].get_ylim()[1]
    ax[0].text(12.8, ytop * .97, "12k (paper v2)", fontsize=6.5, va="top", color="#555555")
    ax[0].text(b - .8, ytop * .97, f"best {b:.0f}k", fontsize=6.5, va="top", ha="right", color=COL["diff"])
    ax[0].set_xlabel("training step (thousands)"); ax[0].set_ylabel("denoising loss"); ax[0].legend(frameon=False, loc="center right")
    leads = [12, 24, 48, 72, 96, 120, 144, 168]
    for key, col, nm in (("at12k", "#555555", "12k steps (paper v2)"), ("best", COL["diff"], f"{b:.0f}k steps (best val.)")):
        v = np.array([S[key][f"BSS{L}_clim"] for L in leads])
        ax[1].errorbar(leads, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], color=col, marker="o", ms=3, capsize=2, lw=1, label=nm)
    ax[1].set_xticks(leads); ax[1].set_xlabel("lead (h)"); ax[1].set_ylabel("BSS vs climatology")
    ax[1].legend(frameon=False, loc="lower left")
    fig.suptitle(f"Diffusion v2 (same $3\\times10^5$-parameter model), {S['train']['steps']//1000}k GPU steps; "
                 f"BSS on {S['seeds']} test seeds, DDIM {S['ddim']}, {S['paths']} paths (mean, 95% seed bootstrap)", fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "e1_learning_curve.png"), dpi=200); plt.close(fig)


def tables200():
    """S1-5: write the generated macro block of docs/paper/main.tex (between the BEGIN/END e200 markers) from
    out/eval200/summary_endurance_murky.json (diff = v2), summary_endurance_murky_v3s{1,2,3}.json (v3: median over the
    three training seeds, range = min..max of the point estimates), pathbase_endurance_murky_200.json, done.json and
    example_endurance_murky.json. No number in the block is typed by hand."""
    E = os.path.join(HERE, "out", "eval200")
    J = lambda f: json.load(open(os.path.join(E, f)))
    S = J("summary_endurance_murky.json")
    V3 = [J(f"summary_endurance_murky_v3s{k}.json")["diff"] for k in (1, 2, 3)]
    PB, DONE, EX = J("pathbase_endurance_murky_200.json"), J("done.json"), J("example_endurance_murky.json")
    assert DONE["n_seeds"] == 200 and DONE["n_samples"] == 200 and DONE["ddim_steps"] == 50 and PB["n_test_seeds"] == 200
    leads = (12, 24, 48, 72, 96, 120, 144, 168)
    WORD = {12: "Twelve", 24: "TwentyFour", 48: "FortyEight", 72: "SeventyTwo", 96: "NinetySix", 120: "OneTwenty",
            144: "OneFortyFour", 168: "OneSixtyEight"}

    def f(x, d=2):
        t = f"{x:.{d}f}"
        return "$-$" + t[1:] if t.startswith("-") else t

    med = lambda key: float(np.median([v[key][0] for v in V3]))
    rng = lambda key, d=2: f"({f(min(v[key][0] for v in V3), d)}--{f(max(v[key][0] for v in V3), d)})"
    ci = lambda r, key, d=2: f"[{f(r[key][1], d)}, {f(r[key][2], d)}]"
    mac = {}  # name -> value
    mac["bssGruSeventyTwo"] = f(S["gru"]["BSS72_clim"][0]); mac["bssGruOneSixtyEight"] = f(S["gru"]["BSS168_clim"][0])
    mac["bssDiffTwoSeventyTwo"] = f(S["diff"]["BSS72_clim"][0]); mac["bssDiffTwoOneSixtyEight"] = f(S["diff"]["BSS168_clim"][0])
    mac["bssDiffThreeSeventyTwo"] = f(med("BSS72_clim")); mac["bssDiffThreeOneSixtyEight"] = f(med("BSS168_clim"))
    mac["llDiffThree"] = f(med("LL")); mac["esDiffThree"] = f(med("ES")); mac["vsDiffThree"] = f(med("VS"), 1)
    mac["bssPfSeventyTwo"] = f(S["pf"]["BSS72_clim"][0]); mac["bssPfOneSixtyEight"] = f(S["pf"]["BSS168_clim"][0])
    mac["esGruPaths"] = f(PB["gru"]["ES"][0]); mac["vsGruPaths"] = f(PB["gru"]["VS"][0], 1)
    for m, nm in (("clock", "Clock"), ("bot", "Bot")):
        mac[f"bss{nm}SeventyTwo"] = f(S[m]["BSS72_clim"][0]); mac[f"bss{nm}OneSixtyEight"] = f(S[m]["BSS168_clim"][0])
    mac["bssDiffThreeSeventyTwoRange"] = rng("BSS72_clim", 3); mac["bssDiffThreeOneSixtyEightRange"] = rng("BSS168_clim", 3)
    mac["bssBotOverClockMax"] = f(max(S["bot"][f"BSS{L}_clock"][0] for L in leads))
    for m, nm in (("pf", "Pf"), ("gru", "Gru"), ("diff", "DiffTwo")):  # skill relative to the clock
        mac[f"bssClock{nm}SeventyTwo"] = f(S[m]["BSS72_clock"][0]); mac[f"bssClock{nm}OneSixtyEight"] = f(S[m]["BSS168_clock"][0])
    mac["bssClockDiffThreeSeventyTwo"] = f(med("BSS72_clock")); mac["bssClockDiffThreeOneSixtyEight"] = f(med("BSS168_clock"))
    # tab:bss rows
    row = lambda r: " & ".join(f(r[f"BSS{L}_clim"][0]) + (" " + ci(r, f"BSS{L}_clim") if L in (72, 168) else "") for L in leads)
    mac["bssRowClock"] = row(S["clock"]); mac["bssRowBot"] = row(S["bot"]); mac["bssRowGru"] = row(S["gru"])
    mac["bssRowDiffTwo"] = row(S["diff"]); mac["bssRowPf"] = row(S["pf"])
    mac["bssRowDiffThree"] = " & ".join(f(med(f"BSS{L}_clim")) + (" " + rng(f"BSS{L}_clim", 3) if L in (72, 168) else "")
                                        for L in leads)
    # tab:scores rows: CRPS LL W1 KL ES VS
    sc = lambda r, es=None: " & ".join([f(r["CRPS"][0], 1), f(r["LL"][0]), f(r["W1pf"][0], 1) if "W1pf" in r else "--",
                                        f(r["KLpf"][0]) if "KLpf" in r else "--"] +
                                       ([f(es["ES"][0]), f(es["VS"][0], 1)] if es else ["--", "--"]))
    mac["scRowClim"] = sc(S["clim"], PB["clim"]); mac["scRowClock"] = sc(S["clock"], PB["clock"])
    mac["scRowBot"] = sc(S["bot"]); mac["scRowGru"] = sc(S["gru"], PB["gru"]); mac["scRowDiffTwo"] = sc(S["diff"], S["diff"])
    mac["scRowDiffThree"] = " & ".join([f(med("CRPS"), 1), f(med("LL")), f(med("W1pf"), 1), f(med("KLpf")),
                                        f(med("ES")), f(med("VS"), 1)])
    mac["scRowPf"] = sc(S["pf"], S["pf"])
    rows_es = [PB["clim"], PB["clock"], PB["gru"], S["diff"], S["pf"]] + V3
    mac["esCiMax"] = f(max(max(r["ES"][2] - r["ES"][0], r["ES"][0] - r["ES"][1]) for r in rows_es))
    mac["vsCiMax"] = f(max(max(r["VS"][2] - r["VS"][0], r["VS"][0] - r["VS"][1]) for r in rows_es), 1)
    # single numbers quoted in the RQ1 text
    for m, nm in (("gru", "Gru"), ("diff", "DiffTwo"), ("pf", "Pf")):
        mac[f"crps{nm}"] = f(S[m]["CRPS"][0], 1)
    mac["crpsDiffThree"] = f(med("CRPS"), 1); mac["crpsDiffThreeRange"] = rng("CRPS", 1)
    mac["wGru"] = f(S["gru"]["W1pf"][0], 1); mac["wDiffTwo"] = f(S["diff"]["W1pf"][0], 1); mac["wDiffThree"] = f(med("W1pf"), 1)
    mac["llGru"] = f(S["gru"]["LL"][0]); mac["llDiffTwo"] = f(S["diff"]["LL"][0])
    mac["esDiffTwo"] = f(S["diff"]["ES"][0]); mac["vsDiffTwo"] = f(S["diff"]["VS"][0], 1)
    mac["esDiffThreeRange"] = rng("ES"); mac["vsDiffThreeRange"] = rng("VS", 1)
    mac["esPf"] = f(S["pf"]["ES"][0]); mac["vsPf"] = f(S["pf"]["VS"][0], 1)
    mac["esClimPaths"] = f(PB["clim"]["ES"][0]); mac["vsClimPaths"] = f(PB["clim"]["VS"][0], 1)
    mac["esClockPaths"] = f(PB["clock"]["ES"][0]); mac["vsClockPaths"] = f(PB["clock"]["VS"][0], 1)
    mac["pbFallback"] = f"{PB['fallback']['gru_class_only']:,}".replace(",", "{,}")
    mac["pbDraws"] = f"{PB['n_test_seeds'] * 113 * PB['m_paths']:,}".replace(",", "{,}")
    mac["pathsDiff"] = str(DONE["n_samples"]); mac["ddimSteps"] = str(DONE["ddim_steps"]); mac["pathsAnalogue"] = str(PB["m_paths"])
    gh = {m: s / 3600 for m, s in DONE["secs_per_model"].items()}
    mac["gpuHoursTwo"] = f(gh["v2"], 1)
    v3h = [gh[k] for k in ("v3s1", "v3s2", "v3s3")]
    mac["gpuHoursThreeMin"] = f(min(v3h), 1); mac["gpuHoursThreeMax"] = f(max(v3h), 1)
    mac["exLandHours"] = str(EX["landing_hours_after_issue"]); mac["exIssueDay"] = str(EX["issue_day"])
    mac["exSeed"] = str(EX["test_seed"]); mac["exPeakDiff"] = f(EX["peak_p_storm_diff"])
    mac["exPeakPf"] = f(EX["peak_p_storm_pf"])
    block = ["% BEGIN e200 (generated by ml/forecast/figures_paper.py tables200 from ml/forecast/out/eval200/*.json; do not edit)"]
    block += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items()]
    block += ["% END e200"]
    tex = os.path.join(ROOT, "docs", "paper", "main.tex")
    t = open(tex).read()
    a, b = t.index("% BEGIN e200"), t.index("% END e200") + len("% END e200")
    open(tex, "w").write(t[:a] + "\n".join(block) + t[b:])
    print(f"{tex}: {len(mac)} macros")


if __name__ == "__main__":
    for w in sys.argv[1:] or ["reliability", "pit", "testbed", "stats", "ladder"]:
        {"reliability": reliability, "pit": pit_rank, "testbed": testbed, "stats": stats, "ladder": ladder, "storms": storms,
         "learning_curve": learning_curve, "tables200": tables200, "storms3b": storms3b}[w]()
        print("done", w, flush=True)
