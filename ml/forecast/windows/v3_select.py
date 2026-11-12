"""S1-3 v3 selection (RESEARCH-PLAN S1-3: <= 3 configs x 3 learning rates, 100k steps each, pick by validation loss).
  v3_select.py jobs <tf32 0|1> [jobdir]   (main Mac) write the 9 chained training jobs + the summary job
  v3_select.py select                      (Windows queue job s13_z_select.cmd) -> D:\\LastEmbers\\e1\\out\\v3sel\\done.json
  v3_select.py export <done.json> <outdir> (main Mac) -> <outdir>/select.json, <outdir>/select/cfg<X>.json
Each training: train_chain.py diff endurance murky 100000 out\\v3sel\\cfg<X>_lr<lr>.pt (segments <= 10200 s), VAL_EVERY
1000, V3_CFG / LR / TF32 passed on to later segments through CHAIN_ENV (CHAIN_ENV lists itself: train_chain.py
re-exports only PASS_ENV + the CHAIN_ENV names, so without it segment 3+ would lose V3_CFG / LR / TF32). A run's best_val_loss is the lowest
validation loss in its log (= its best checkpoint, which train.py keeps as the .pt); a config's best_val_loss is the
lowest of its 3 learning rates; chosen = the config with the lowest. v2 reference: S1-2 out\\lc\\done.json."""
import json, os, sys, time

CFGS = ("A", "B", "C")
LRS = ("1e-4", "3e-4", "1e-3")
STEPS, VAL_EVERY = 100000, 1000
E1W = r"D:\LastEmbers\e1"
FCW = E1W + r"\ml\forecast"
OUTW = E1W + r"\out\v3sel"


def name(cfg, lr):
    return f"cfg{cfg}_lr{lr}"


def jobs(tf32, jobdir):
    os.makedirs(jobdir, exist_ok=True)
    for i, cfg in enumerate(CFGS):
        for j, lr in enumerate(LRS):
            job = f"s13_{cfg}{j + 1}_lr{lr}"
            lines = ["@echo off",
                     f"rem S1-3 v3 selection: config {cfg} (models.V3_CONFIGS), peak lr {lr}, {STEPS} steps chained through the queue",
                     f"rem (train_chain.py re-queues {job}.cmd until done). Writes {OUTW}\\{name(cfg, lr)}.pt(.log/.state/.chain.json),",
                     f"rem {OUTW}\\runs\\{name(cfg, lr)}.json. console/locale is GBK; sources and balance.json are UTF-8",
                     "set PYTHONUTF8=1", "set DEVICE=cuda", f"set V3_CFG={cfg}", f"set LR={lr}", f"set TF32={tf32}",
                     f"set VAL_EVERY={VAL_EVERY}", "set CHAIN_ENV=V3_CFG,LR,TF32,CHAIN_ENV", f"cd /d {FCW}",
                     f"if not exist {OUTW}\\runs mkdir {OUTW}\\runs",
                     f"{E1W}\\.venv\\Scripts\\python.exe -u windows\\train_chain.py diff endurance murky {STEPS} "
                     f"{OUTW}\\{name(cfg, lr)}.pt --job {job} --done {OUTW}\\runs\\{name(cfg, lr)}.json"]
            with open(os.path.join(jobdir, job + ".cmd"), "w", newline="\r\n") as fh:
                fh.write("\n".join(lines) + "\n")
    lines = ["@echo off", "rem S1-3 v3 selection summary (after the 9 trainings): writes " + OUTW + "\\done.json",
             "set PYTHONUTF8=1", f"cd /d {FCW}", f"{E1W}\\.venv\\Scripts\\python.exe -u windows\\v3_select.py select"]
    with open(os.path.join(jobdir, "s13_z_select.cmd"), "w", newline="\r\n") as fh:
        fh.write("\n".join(lines) + "\n")


def select():
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(here))
    from models import v3_net
    out = os.path.join(os.path.abspath(os.path.join(here, "..", "..", "..")), "out", "v3sel")
    configs = {}
    for cfg in CFGS:
        runs = []
        for lr in LRS:
            n = name(cfg, lr)
            d = json.load(open(os.path.join(out, "runs", n + ".json")))
            ch = json.load(open(os.path.join(out, n + ".pt.chain.json")))
            log = [json.loads(ln) for ln in open(os.path.join(out, n + ".pt.log"))]
            b = min(log, key=lambda r: r["val"])
            assert d["steps"] == STEPS and log[-1]["it"] == STEPS, (n, d["steps"], log[-1]["it"])
            runs.append({"lr": float(lr), "steps": d["steps"], "best_val_loss": b["val"], "best_step": b["it"],
                         "final_val_loss": log[-1]["val"], "train_secs": log[-1]["secs"],
                         "sec_per_step": round(log[-1]["secs"] / STEPS, 4), "segments": ch["segments"],
                         "curve": [[r["it"], r["val"]] for r in log], "pt": os.path.join(out, n + ".pt")})
        br = min(runs, key=lambda r: r["best_val_loss"])
        configs["cfg" + cfg] = {"params": sum(p.numel() for p in v3_net(cfg).parameters()), "lrs": [float(x) for x in LRS],
                                "best_val_loss": br["best_val_loss"], "best_lr": br["lr"], "runs": runs}
    chosen = min(configs, key=lambda k: configs[k]["best_val_loss"])
    ref = os.path.join(out, "..", "lc", "done.json")
    v2 = None
    if os.path.exists(ref):
        r = json.load(open(ref))
        v2 = {"best_val_loss": r["best_val_loss"], "best_step": r["best_step"], "val_every": r["val_every"],
              "sec_per_step": r["sec_per_step"], "params": 298931}
    done = {"configs": configs, "chosen": chosen, "v2_100k": v2, "steps": STEPS, "val_every": VAL_EVERY,
            "rule": "per run min validation loss (best checkpoint); per config min over 3 lrs; chosen = argmin over configs",
            "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(done, open(os.path.join(out, "done.json"), "w"), indent=1)
    print(json.dumps({k: (v["best_val_loss"], v["best_lr"]) for k, v in configs.items()}), "chosen", chosen, flush=True)


def export(donef, outdir):
    d = json.load(open(donef))
    os.makedirs(os.path.join(outdir, "select"), exist_ok=True)
    for k, c in d["configs"].items():
        json.dump({"params": c["params"], "best_val_loss": c["best_val_loss"], "best_lr": c["best_lr"],
                   "lr_sweep": [{"lr": r["lr"], "best_val_loss": r["best_val_loss"], "best_step": r["best_step"],
                                 "final_val_loss": r["final_val_loss"], "sec_per_step": r["sec_per_step"],
                                 "segments": len(r["segments"])} for r in c["runs"]]},
                  open(os.path.join(outdir, "select", k + ".json"), "w"), indent=1)
    json.dump({"chosen": d["chosen"], "params": {k: c["params"] for k, c in d["configs"].items()},
               "best_val_loss": {k: c["best_val_loss"] for k, c in d["configs"].items()},
               "best_lr": {k: c["best_lr"] for k, c in d["configs"].items()}, "v2_100k": d.get("v2_100k"),
               "rule": d["rule"], "finished": d["finished"]},
              open(os.path.join(outdir, "select.json"), "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "jobs":
        jobs(a[1], a[2] if len(a) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs"))
    elif a[0] == "select":
        select()
    elif a[0] == "export":
        export(a[1], a[2])
    else:
        sys.exit(__doc__)
