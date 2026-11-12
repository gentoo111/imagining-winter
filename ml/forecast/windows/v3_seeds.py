"""S1-4 v3 x 3 training seeds (RESEARCH-PLAN S1-4) of the S1-3 choice (cfgC, lr 1e-3; ml/forecast/out/v3/select.json).
  v3_seeds.py jobs [jobdir]              (main Mac) write s14_s1..s14_s3.cmd + s14_z_summary.cmd
  v3_seeds.py summary                    (Windows queue job s14_z_summary.cmd) -> D:\\LastEmbers\\e1\\out\\v3\\done.json
  v3_seeds.py export <done.json> <outdir> (main Mac) -> <outdir>/train_s<k>.json
Each seed: SEED=k train_chain.py diff endurance murky 100000 out\\v3\\diff3_endurance_murky_s<k>.pt (segments <= 10200 s),
VAL_EVERY 1000, TF32. CHAIN_ENV lists itself so segment 3+ keeps V3_CFG / LR / TF32 / SEED (train_chain.py re-exports only
PASS_ENV + the names in CHAIN_ENV; see E1-RESULTS S1-3). The .pt is each run's best-validation checkpoint (train.py)."""
import json, os, statistics, sys, time

SEEDS = (1, 2, 3)
CFG, LR, STEPS, VAL_EVERY = "C", "1e-3", 100000, 1000
E1W = r"D:\LastEmbers\e1"
FCW = E1W + r"\ml\forecast"
OUTW = E1W + r"\out\v3"


def pt_name(k):
    return f"diff3_endurance_murky_s{k}"


def jobs(jobdir):
    os.makedirs(jobdir, exist_ok=True)
    for k in SEEDS:
        job = f"s14_s{k}"
        lines = ["@echo off",
                 f"rem S1-4 v3 training seed {k}: cfg{CFG}, lr {LR}, {STEPS} steps chained through the queue (train_chain.py",
                 f"rem re-queues {job}.cmd until done). Writes {OUTW}\\{pt_name(k)}.pt(.log/.state/.chain.json), {OUTW}\\runs\\s{k}.json.",
                 "rem console/locale is GBK; sources and balance.json are UTF-8",
                 "set PYTHONUTF8=1", "set DEVICE=cuda", f"set SEED={k}", f"set V3_CFG={CFG}", f"set LR={LR}", "set TF32=1",
                 f"set VAL_EVERY={VAL_EVERY}", "set CHAIN_ENV=V3_CFG,LR,TF32,SEED,CHAIN_ENV", f"cd /d {FCW}",
                 f"if not exist {OUTW}\\runs mkdir {OUTW}\\runs",
                 f"{E1W}\\.venv\\Scripts\\python.exe -u windows\\train_chain.py diff endurance murky {STEPS} "
                 f"{OUTW}\\{pt_name(k)}.pt --job {job} --done {OUTW}\\runs\\s{k}.json"]
        with open(os.path.join(jobdir, job + ".cmd"), "w", newline="\r\n") as fh:
            fh.write("\n".join(lines) + "\n")
    lines = ["@echo off", "rem S1-4 summary (after the 3 seeds): writes " + OUTW + "\\done.json",
             "set PYTHONUTF8=1", f"cd /d {FCW}", f"{E1W}\\.venv\\Scripts\\python.exe -u windows\\v3_seeds.py summary"]
    with open(os.path.join(jobdir, "s14_z_summary.cmd"), "w", newline="\r\n") as fh:
        fh.write("\n".join(lines) + "\n")


def summary():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(os.path.abspath(os.path.join(here, "..", "..", "..")), "out", "v3")
    runs = []
    for k in SEEDS:
        n = pt_name(k)
        d = json.load(open(os.path.join(out, "runs", f"s{k}.json")))
        ch = json.load(open(os.path.join(out, n + ".pt.chain.json")))
        log = [json.loads(ln) for ln in open(os.path.join(out, n + ".pt.log"))]
        b = min(log, key=lambda r: r["val"])
        assert d["steps"] == STEPS and log[-1]["it"] == STEPS, (n, d["steps"], log[-1]["it"])
        runs.append({"seed": k, "cfg": "cfg" + CFG, "lr": float(LR), "steps": d["steps"], "best_step": b["it"],
                     "best_val_loss": b["val"], "final_val_loss": log[-1]["val"],
                     "sec_per_step": round(log[-1]["secs"] / STEPS, 4), "segments": ch["segments"],
                     "curve": [[r["it"], r["val"]] for r in log], "pt": os.path.join(out, n + ".pt")})
    v = [r["best_val_loss"] for r in runs]
    done = {"seeds_done": len(runs), "cfg": "cfg" + CFG, "lr": float(LR), "steps": STEPS, "val_every": VAL_EVERY,
            "best_val_loss_mean": statistics.mean(v), "best_val_loss_sd": statistics.stdev(v), "runs": runs,
            "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(done, open(os.path.join(out, "done.json"), "w"), indent=1)
    print(json.dumps({r["seed"]: (r["best_val_loss"], r["best_step"]) for r in runs}), flush=True)


def export(donef, outdir):
    d = json.load(open(donef))
    os.makedirs(outdir, exist_ok=True)
    for r in d["runs"]:
        json.dump({k: r[k] for k in ("seed", "cfg", "lr", "steps", "best_step", "best_val_loss", "final_val_loss",
                                     "sec_per_step")} | {"segments": len(r["segments"])},
                  open(os.path.join(outdir, f"train_s{r['seed']}.json"), "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "jobs":
        jobs(a[1] if len(a) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs"))
    elif a and a[0] == "summary":
        summary()
    elif a and a[0] == "export":
        export(a[1], a[2])
    else:
        sys.exit(__doc__)
