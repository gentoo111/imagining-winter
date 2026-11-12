"""RQ2 storm metrics over the three training seeds per rung (GAPS #2).

Seed 1: out/e0_eval_<mode>_storm.json (runs/e0-<rung> of the main checkout); seeds 2-3: out/e0_eval_<mode>_storm_s<s>_<rung>.json
(e0_queue.py, frozen 4195eb6 rules). Every file holds the same 1,000 held-out seeds 300000-300999.

Per training seed: storm-focused metrics (storm_e0.summarize_agent) and episode metrics (days, outcomes).
Across seeds: mean, s.d. and range of the per-seed values (training variance).
Pooled paired: for rungs A, B the mean over training seeds s of the paired difference A_s - B_s on the storms (seed, k)
both reached; CI = 95% bootstrap over evaluation seeds, resampled jointly for all training seeds (evaluation noise only,
conditional on these three training seeds); n_below0 / n_above0 = training seeds whose own paired CI excludes zero.
Rungs fcg (3a', daily GRU forecast table) and fcd (3b, daily diffusion forecast table), the pairs fcd-fcg / fcd-fc and the
per-episode alive-at-end metric use their own bootstrap RNG (default_rng(1)), so every number of the original rungs and
pairs (default_rng(0), same order as before) is bit-identical to the json before they were added.
  storm_seeds.py [--rungs start,none,signs,oracle,fc,fcg,fcd]  -> out/storm_seeds_<mode>.json + printed tables"""
import argparse, json, os
import numpy as np
from storm_e0 import load, storm_table, summarize_agent, cluster_ci, TIERS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
B = 2000
EP_METRICS = ("day", "cap", "exiled", "exploded", "ep_wiped")
PAIR_M = ("deaths", "any_death", "death_rate", "pop")
NEW_RUNGS = ("fcg", "fcd")  # s1-11c: bootstrap with the separate RNG (see docstring)
PAIRS = (("signs", "none"), ("oracle", "none"), ("oracle", "signs"), ("fc", "signs"), ("fc", "none"), ("oracle", "fc"))
NEW_PAIRS = (("fcd", "fcg"), ("fcd", "fc"))


def files(mode, rung, s):
    if s == 1 and rung in ("none", "signs", "oracle", "start", "bot"):
        return os.path.join(OUT, f"e0_eval_{mode}_storm.json")
    return os.path.join(OUT, f"e0_eval_{mode}_storm_s{s}_{rung}.json")


def episodes(mode, rung, s):
    p = files(mode, rung, s)
    if not os.path.exists(p):
        return None
    return load(p).get(rung)


def ep_table(eps):
    eps = sorted(eps, key=lambda e: e["seed"])
    return dict(seed=np.array([e["seed"] for e in eps]), day=np.array([e["day"] for e in eps], float),
                cap=np.array([e["outcome"] == "cap" for e in eps], float),
                exiled=np.array([e["outcome"] == "exiled" for e in eps], float),
                exploded=np.array([e["outcome"] == "exploded" for e in eps], float),
                ep_wiped=np.array([e["outcome"] == "wiped" for e in eps], float),
                alive=np.array([e["alive"] for e in eps], float))


def pair_rows(TA, TB):
    ka = {(int(s), int(k)): i for i, (s, k) in enumerate(zip(TA["seed"], TA["k"]))}
    kb = {(int(s), int(k)): i for i, (s, k) in enumerate(zip(TB["seed"], TB["k"]))}
    c = sorted(set(ka) & set(kb))
    ia = np.array([ka[x] for x in c]); ib = np.array([kb[x] for x in c])
    return np.array([x[0] for x in c]), {m: TA[m][ia] - TB[m][ib] for m in PAIR_M}


def pooled(diffs, eval_seeds, rng):
    """diffs: list over training seeds of (seed array, value array). Mean over training seeds of per-seed means;
    CI by resampling evaluation seeds jointly."""
    u = np.unique(np.concatenate([s for s, _ in diffs]))
    idx = rng.integers(0, len(u), (B, len(u)))
    cnt = np.zeros((B, len(u)))
    for b in range(B):
        cnt[b] = np.bincount(idx[b], minlength=len(u))
    est, boots = [], []
    for s, v in diffs:
        pos = np.searchsorted(u, s)
        S = np.bincount(pos, v, len(u)); C = np.bincount(pos, minlength=len(u)).astype(float)
        est.append(v.mean())
        boots.append((cnt @ S) / np.maximum(cnt @ C, 1))
    bm = np.mean(boots, 0)
    return [float(np.mean(est)), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5)), [float(x) for x in est]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", default="start,none,signs,oracle,fc,fcg,fcd")
    ap.add_argument("--seeds", default="1,2,3")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    for mode in ("greedy", "sample"):
        rng = np.random.default_rng(0)
        rng1 = np.random.default_rng(1)  # new rungs / pairs / alive only
        res = {"mode": mode, "train_seeds": seeds, "rungs": {}, "paired": {}}
        tabs, etabs = {}, {}
        for r in a.rungs.split(","):
            rg = rng1 if r in NEW_RUNGS else rng
            per = {}
            for s in seeds:
                eps = episodes(mode, r, s)
                if eps is None:
                    continue
                sm, T = summarize_agent(eps, rg)
                E = ep_table(eps)
                tabs[(r, s)], etabs[(r, s)] = T, E
                per[s] = dict(storms_per_episode=sm["storms_per_episode"], n_storms=sm["n_storms"],
                              **{m: sm["all"][m] for m in sm["all"]},
                              tier={t: {m: v[m] for m in ("deaths", "any_death", "death_rate", "n") if m in v} for t, v in sm["tier"].items()},
                              **{m: float(E[m].mean()) for m in EP_METRICS},
                              day_ci=cluster_ci(E["day"], E["seed"], rg))
                per[s]["alive"] = float(E["alive"].mean())
                per[s]["alive_ci"] = cluster_ci(E["alive"], E["seed"], rng1)
            if not per:
                continue
            agg = {}
            for m in ["storms_per_episode", "deaths", "any_death", "death_rate", "pop", "coal_hours", "coal_short",
                      "furnace_at_max", "level_up_48h", "teams_out", "homeless", "food_per_head", "ended"] + list(EP_METRICS) + ["alive"]:
                v = np.array([per[s][m] if not isinstance(per[s][m], list) else per[s][m][0] for s in per], float)
                agg[m] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else None,
                              min=float(v.min()), max=float(v.max()), values=v.tolist())
            agg["tier"] = {}
            for t in TIERS.values():
                v = np.array([per[s]["tier"][t]["deaths"][0] for s in per], float)
                agg["tier"][t] = dict(mean=float(v.mean()), min=float(v.min()), max=float(v.max()),
                                      n=int(sum(per[s]["tier"][t]["n"] for s in per)))
            # coal hours: median of the pooled storms (the mean is dominated by huge stocks)
            ch = np.concatenate([tabs[(r, s)]["coal_hours"] for s in per])
            agg["coal_hours_median"] = float(np.nanmedian(ch)); agg["coal_hours_p10"] = float(np.nanpercentile(ch, 10))
            res["rungs"][r] = dict(per_seed=per, across=agg)
        for A, Bn in PAIRS + NEW_PAIRS:
            rp = rng1 if (A, Bn) in NEW_PAIRS else rng
            ss = [s for s in seeds if (A, s) in tabs and (Bn, s) in tabs]
            if not ss:
                continue
            P = {"train_seeds": ss}
            rows = {s: pair_rows(tabs[(A, s)], tabs[(Bn, s)]) for s in ss}
            P["n_common"] = [int(len(rows[s][0])) for s in ss]
            for m in PAIR_M:
                P[m] = pooled([(rows[s][0], rows[s][1][m]) for s in ss], None, rp)
                per_ci = [cluster_ci(rows[s][1][m], rows[s][0], rp) for s in ss]
                P[m + "_per_seed"] = per_ci
                P[m + "_n_below0"] = int(sum(c[2] < 0 for c in per_ci)); P[m + "_n_above0"] = int(sum(c[1] > 0 for c in per_ci))
            # days survived, paired on evaluation seeds
            dd = [(etabs[(A, s)]["seed"], etabs[(A, s)]["day"] - etabs[(Bn, s)]["day"]) for s in ss]
            assert all((etabs[(A, s)]["seed"] == etabs[(Bn, s)]["seed"]).all() for s in ss)
            P["day"] = pooled(dd, None, rp)
            per_ci = [cluster_ci(v, sd, rp) for sd, v in dd]
            P["day_per_seed"] = per_ci
            P["day_n_below0"] = int(sum(c[2] < 0 for c in per_ci)); P["day_n_above0"] = int(sum(c[1] > 0 for c in per_ci))
            res["paired"][f"{A}-{Bn}"] = P
        json.dump(res, open(os.path.join(OUT, f"storm_seeds_{mode}.json"), "w"), indent=1)
        print(f"\n===== {mode}")
        for r, v in res["rungs"].items():
            g = v["across"]
            print(f"{r:7} " + "  ".join(f"{m} {g[m]['mean']:.3f}±{(g[m]['sd'] or 0):.3f} {['%.2f' % x for x in g[m]['values']]}"
                                        for m in ("storms_per_episode", "deaths", "any_death", "death_rate", "day", "cap", "exiled")))
        for k, P in res["paired"].items():
            print(f"  {k:13} n={P['n_common']} " + "  ".join(
                f"{m} {P[m][0]:+.3f} [{P[m][1]:+.3f},{P[m][2]:+.3f}] seeds {['%+.2f' % x for x in P[m][3]]} (<0 {P[m + '_n_below0']}, >0 {P[m + '_n_above0']})"
                for m in ("deaths", "any_death", "death_rate", "day")))


if __name__ == "__main__":
    main()
