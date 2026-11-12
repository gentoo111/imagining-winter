"""Storm-focused RQ2 metrics from eval_e0.py output (episodes[*].storm_recs: one record per storm the colony reached).

Most E0 episodes end by exile or a furnace explosion, so days survived mixes the value of storm foresight with
failure modes that have little to do with the weather. Here every number is conditioned on a storm being reached
(landing hour with people alive). Weather depends only on the seed, so storm k of seed s is the same storm for every
agent: paired comparisons use the storms (seed, k) that both agents reached. CIs: 95% bootstrap over seeds (cluster).

  storm_e0.py out/e0_eval_greedy_storm.json [...]           -> table + out/storm_summary_<name>.json
"""
import argparse, json, os
import numpy as np

B = 2000
TIERS = {-1: "shallow", 0: "mid", 1: "deep"}
PAIRS = [("signs", "none"), ("oracle", "none"), ("oracle", "signs"), ("none", "start"), ("none", "bot")]


def storm_table(eps):
    """Flatten episodes -> per-storm rows (dict of arrays) with the episode seed attached."""
    rows = []
    for e in eps:
        for r in e.get("storm_recs", []):
            pop = max(r["pop"], 1)
            rows.append(dict(
                seed=e["seed"], k=r["k"], tier=r["tier"], pop=r["pop"], day=r["day"],
                deaths=r["deaths"], deaths24=r["deaths"] + r["post_deaths"], any_death=float(r["deaths"] > 0),
                death_rate=100.0 * r["deaths"] / pop,  # deaths per 100 people alive at landing
                ended=float(r["ended"]),  # the episode ended during the storm or the 24 h after it
                ended_storm=float(r["ended"] and r.get("ended_in") == "storm"),
                wiped=float(r["ended"] and r.get("end_alive", 1) == 0),
                coal=r["coal"], coal_hours=np.nan if r["coal_hours"] is None else min(r["coal_hours"], 1e4),
                coal_short=float(r["coal_short_hours"] > 0), furnace_out=float(r["furnace_out_hours"] > 0),
                furnace_level=r["furnace_level"], furnace_at_max=float(r["furnace_level"] >= r["furnace_max"]),
                heater_max=r["heater_max"], teams_out=float(r["teams_out"] > 0), food_per_head=r["food_per_head"],
                homeless=r["homeless"], hungry=r["hungry"], sick=r["sick"] + r["grave"], overdrive=float(r["overdrive"]),
                discontent=r["discontent"], hope=r["hope"],
                level_up_48h=np.nan if r["level_48h_before"] is None else float(r["furnace_level"] > r["level_48h_before"]),
                coal_gain_48h=np.nan if r["coal_48h_before"] is None else r["coal"] - r["coal_48h_before"]))
    if not rows:
        return {}
    return {k: np.array([x[k] for x in rows], float) for k in rows[0]}


def cluster_ci(vals, seeds, rng, stat=np.nanmean):
    """Mean of per-storm values, CI by resampling seeds (all storms of a seed together)."""
    vals, seeds = np.asarray(vals, float), np.asarray(seeds)
    ok = ~np.isnan(vals)
    vals, seeds = vals[ok], seeds[ok]
    if len(vals) == 0:
        return [None, None, None, 0]
    u, inv = np.unique(seeds, return_inverse=True)
    s = np.bincount(inv, vals, len(u)); c = np.bincount(inv, minlength=len(u)).astype(float)
    idx = rng.integers(0, len(u), (B, len(u)))
    bs = s[idx].sum(1) / np.maximum(c[idx].sum(1), 1)
    return [float(vals.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(len(vals))]


METRICS = ["deaths", "deaths24", "any_death", "death_rate", "ended", "ended_storm", "wiped", "pop",
           "coal_hours", "coal_short", "furnace_out", "furnace_level", "furnace_at_max", "heater_max", "teams_out",
           "food_per_head", "homeless", "hungry", "sick", "overdrive", "discontent", "level_up_48h", "coal_gain_48h"]
PAIR_METRICS = ["deaths", "deaths24", "any_death", "death_rate", "ended", "furnace_level", "coal_hours", "pop"]


def summarize_agent(eps, rng):
    T = storm_table(eps)
    if not T:
        return None
    out = dict(n_storms=int(len(T["seed"])), n_episodes=len(eps),
               storms_per_episode=float(len(T["seed"]) / len(eps)),
               all={m: cluster_ci(T[m], T["seed"], rng) for m in METRICS}, tier={})
    for t, name in TIERS.items():
        sel = T["tier"] == t
        out["tier"][name] = {m: cluster_ci(T[m][sel], T["seed"][sel], rng)
                             for m in ("deaths", "deaths24", "any_death", "death_rate", "ended", "pop")}
        out["tier"][name]["n"] = int(sel.sum())
    # first three storms only (most episodes reach them; less selection by survival)
    sel = T["k"] < 3
    out["first3"] = {m: cluster_ci(T[m][sel], T["seed"][sel], rng) for m in ("deaths", "any_death", "death_rate", "ended")}
    out["first3"]["n"] = int(sel.sum())
    return out, T


def paired(TA, TB, rng, tier=None):
    """Storms (seed, k) reached by both agents: mean difference A-B, CI over seeds."""
    ka = {(int(s), int(k)): i for i, (s, k) in enumerate(zip(TA["seed"], TA["k"]))}
    kb = {(int(s), int(k)): i for i, (s, k) in enumerate(zip(TB["seed"], TB["k"]))}
    common = sorted(set(ka) & set(kb))
    if tier is not None:
        common = [c for c in common if TA["tier"][ka[c]] == tier]
    if not common:
        return None
    ia = np.array([ka[c] for c in common]); ib = np.array([kb[c] for c in common])
    seeds = np.array([c[0] for c in common])
    res = {"n_common": len(common)}
    for m in PAIR_METRICS:
        res[m] = cluster_ci(TA[m][ia] - TB[m][ib], seeds, rng)
    return res


def summarize(eps_by_agent, seed=0):
    rng = np.random.default_rng(seed)
    res, tabs = {}, {}
    for ag, eps in eps_by_agent.items():
        r = summarize_agent(eps, rng)
        if r:
            res[ag], tabs[ag] = r
    res["paired"] = {}
    for a, b in PAIRS:
        if a in tabs and b in tabs:
            res["paired"][f"{a}-{b}"] = dict(all=paired(tabs[a], tabs[b], rng),
                                            **{TIERS[t]: paired(tabs[a], tabs[b], rng, t) for t in TIERS})
    return res


def fmt(t, p=2):
    return "--" if t is None or t[0] is None else f"{t[0]:.{p}f} [{t[1]:.{p}f},{t[2]:.{p}f}]"


def show(res, title):
    print(f"\n== {title}")
    print(f"{'agent':7} {'storms':>6} {'/ep':>5} {'deaths/storm':>20} {'P(death)':>20} {'deaths/100':>20} {'ended':>18} "
          f"{'pop':>8} {'coal h':>8} {'F lvl':>6} {'F@max':>6} {'teams':>6} {'homeless':>8} {'fout':>6}")
    for ag, v in res.items():
        if ag == "paired":
            continue
        A = v["all"]
        print(f"{ag:7} {v['n_storms']:6d} {v['storms_per_episode']:5.2f} {fmt(A['deaths']):>20} {fmt(A['any_death']):>20} "
              f"{fmt(A['death_rate']):>20} {fmt(A['ended']):>18} {A['pop'][0]:8.1f} {np.nan if A['coal_hours'][0] is None else A['coal_hours'][0]:8.0f} "
              f"{A['furnace_level'][0]:6.2f} {A['furnace_at_max'][0]:6.2f} {A['teams_out'][0]:6.2f} {A['homeless'][0]:8.2f} {A['furnace_out'][0]:6.3f}")
        print("        tiers: " + "  ".join(f"{n} n={t['n']} d={fmt(t['deaths'])} P={fmt(t['any_death'])}" for n, t in v["tier"].items()))
    for k, p in res["paired"].items():
        a = p["all"]
        if a is None:
            continue
        print(f"  paired {k:13} n={a['n_common']:5d} deaths {fmt(a['deaths'])}  P(death) {fmt(a['any_death'], 3)}  "
              f"per100 {fmt(a['death_rate'])}  ended {fmt(a['ended'], 3)}  pop {fmt(a['pop'], 1)}")
        for t in TIERS.values():
            if p[t]:
                print(f"      {t:8} n={p[t]['n_common']:5d} deaths {fmt(p[t]['deaths'])}  P(death) {fmt(p[t]['any_death'], 3)}")


def load(path, rename=None):
    d = json.load(open(path)); by = {}
    for e in d["episodes"]:
        by.setdefault((rename or {}).get(e["agent"], e["agent"]), []).append(e)
    return by


def drift(paths, out):
    """Checkpoint drift: storm metrics and paired rung differences per checkpoint file, then mean/sd/range over files."""
    per = []
    for p in paths:
        res = summarize(load(p))
        per.append(res)
    D = {"files": paths, "agents": {}, "paired": {}}
    for ag in ("none", "signs", "oracle"):
        D["agents"][ag] = {}
        for m in ("deaths", "any_death", "death_rate", "ended", "pop", "furnace_level"):
            v = np.array([r[ag]["all"][m][0] for r in per])
            D["agents"][ag][m] = dict(values=v.tolist(), mean=float(v.mean()), sd=float(v.std(ddof=1)), min=float(v.min()), max=float(v.max()))
        D["agents"][ag]["storms_per_episode"] = [r[ag]["storms_per_episode"] for r in per]
    for k in ("signs-none", "oracle-none", "oracle-signs"):
        D["paired"][k] = {}
        for m in ("deaths", "any_death", "death_rate"):
            v = np.array([r["paired"][k]["all"][m][0] for r in per])
            lo = np.array([r["paired"][k]["all"][m][1] for r in per]); hi = np.array([r["paired"][k]["all"][m][2] for r in per])
            D["paired"][k][m] = dict(values=v.tolist(), lo=lo.tolist(), hi=hi.tolist(), mean=float(v.mean()), sd=float(v.std(ddof=1)),
                                     n_below0=int((hi < 0).sum()), n_above0=int((lo > 0).sum()))
    print("\n== checkpoint drift over", len(paths), "files")
    for ag, v in D["agents"].items():
        print(f"{ag:7} " + "  ".join(f"{m} {x['mean']:.2f}±{x['sd']:.2f} [{x['min']:.2f},{x['max']:.2f}]" for m, x in v.items() if m != "storms_per_episode"))
    for k, v in D["paired"].items():
        print(f"  {k:13} " + "  ".join(f"{m} {x['mean']:+.3f}±{x['sd']:.3f} (CI<0: {x['n_below0']}, CI>0: {x['n_above0']} of {len(paths)})" for m, x in v.items()))
    json.dump(D, open(out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--drift", default="", help="write a checkpoint-drift summary over the files to this path")
    a = ap.parse_args()
    if a.drift:
        return drift(a.files, a.drift)
    groups = [(os.path.splitext(os.path.basename(p))[0], load(p)) for p in a.files]
    for name, by in groups:
        res = summarize(by)
        show(res, name)
        json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(a.files[0])), f"storm_summary_{name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
