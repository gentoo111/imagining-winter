"""E0 ladder table from eval_e0.py output: per agent mean [95% bootstrap CI over seeds] and paired rung differences.
  summarize_e0.py out/e0_eval_greedy_1k.json [more json ...]  -> printed table + out/e0_summary_<name>.json"""
import json, os, sys
import numpy as np

B = 2000


def ci(v, rng):
    v = np.asarray(v, float); n = len(v)
    bs = v[rng.integers(0, n, (B, n))].mean(1)
    return [float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def ratio_ci(num, den, rng):
    num, den = np.asarray(num, float), np.asarray(den, float); n = len(num)
    k = rng.integers(0, n, (B, n))
    bs = num[k].sum(1) / np.maximum(den[k].sum(1), 1)
    return [float(num.sum() / max(den.sum(), 1)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    for path in sys.argv[1:]:
        d = json.load(open(path))
        rng = np.random.default_rng(0)
        by = {}
        for r in d["episodes"]:
            by.setdefault(r["agent"], []).append(r)
        res = {}
        print(f"\n{path}: seeds {d['seeds']}, {d['cond']}")
        print(f"{'agent':7} {'n':>4} {'days':>22} {'alive@end':>20} {'cap%':>6} {'wiped%':>6} {'lost%':>6} {'storms':>6} "
              f"{'storm deaths':>20} {'deaths/storm':>20} {'storm share':>11}")
        for ag, rs in by.items():
            rs = sorted(rs, key=lambda r: r["seed"])
            g = lambda k: [r[k] for r in rs]
            res[ag] = dict(n=len(rs), seeds=g("seed"), days=ci(g("day"), rng), alive_end=ci(g("alive_counted"), rng),
                           alive_raw=ci(g("alive"), rng),
                           cap=float(np.mean([r["outcome"] == "cap" for r in rs])), wiped=float(np.mean([r["outcome"] == "wiped" for r in rs])),
                           lost=float(np.mean([r["outcome"] not in ("cap", "wiped") for r in rs])), exiled=float(np.mean([r["outcome"] == "exiled" for r in rs])), exploded=float(np.mean([r["outcome"] == "exploded" for r in rs])), post_storm_deaths=ci([r.get("post_storm_deaths", 0) for r in rs], rng), storms=ci(g("storms"), rng),
                           storm_deaths=ci(g("storm_deaths"), rng), deaths=ci(g("deaths"), rng),
                           deaths_per_storm=ratio_ci(g("storm_deaths"), g("storms"), rng),
                           storm_share=float(np.sum(g("storm_deaths")) / max(np.sum(g("deaths")), 1)),
                           storm_hour_share=float(np.sum(g("storm_hours")) / max(np.sum(g("hours")), 1)),
                           day_values=g("day"), storm_death_values=g("storm_deaths"), secs=rs[0].get("secs_total"))
            v = res[ag]; f = lambda t, p=1: f"{t[0]:.{p}f} [{t[1]:.{p}f},{t[2]:.{p}f}]"
            print(f"{ag:7} {v['n']:4d} {f(v['days']):>22} {f(v['alive_end'], 2):>20} {100*v['cap']:6.1f} {100*v['wiped']:6.1f} {100*v['lost']:6.1f} "
                  f"{v['storms'][0]:6.2f} {f(v['storm_deaths']):>20} {f(v['deaths_per_storm'], 2):>20} {v['storm_share']:11.2f}")
        # paired differences on identical seeds
        pairs = [("signs", "none"), ("oracle", "none"), ("oracle", "signs"), ("none", "bot"), ("none", "start"),
                 ("signs", "start"), ("oracle", "start")]
        res["paired"] = {}
        for a, b in pairs:
            if a in res and b in res and res[a]["seeds"] == res[b]["seeds"]:
                for key in ("days", "storm_deaths"):
                    vals = "day_values" if key == "days" else "storm_death_values"
                    dv = np.array(res[a][vals]) - np.array(res[b][vals])
                    t = ci(dv, rng)
                    res["paired"][f"{a}-{b}:{key}"] = t
                    print(f"  paired {a}-{b} {key}: {t[0]:+.2f} [{t[1]:+.2f},{t[2]:+.2f}]")
        name = os.path.splitext(os.path.basename(path))[0]
        json.dump(res, open(os.path.join(os.path.dirname(path), f"e0_summary_{name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
