"""S1-2 learning curve, queue job 2 (lc_eval.cmd, after lc_train.cmd): 60 test seeds (0-59), endurance/murky,
DDIM 25, 50 paths (the paper's v2 settings, E1-RESULTS.md) for
  best   = lowest-validation-loss checkpoint of the 100k run (selected on val, not test)
  at12k  = the paper's 12k-step CPU model runs/diff2_endurance_murky.pt, re-scored here with the same sampler
  run12k = the 100k run's own 12k-step checkpoint (reference only: its LR schedule is the 100k one)
BSS vs climatology exactly as summarize.py (per-seed mean Brier, 1000 seed bootstraps, rng seed 0).
Writes D:\\LastEmbers\\e1\\out\\lc\\summary.json (+ the npz files)."""
import json, os, shutil, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FC = os.path.dirname(HERE)
sys.path.insert(0, FC)
from metrics import LEADS, brier  # noqa: E402

E1 = os.path.abspath(os.path.join(FC, "..", ".."))
OUT = os.environ.get("LC_OUT", os.path.join(E1, "out", "lc"))  # LC_OUT / LC_NS only for a dry run
NS, DDIM = int(os.environ.get("LC_NS", "60")), 25
done = json.load(open(os.path.join(OUT, "done.json")))
started = time.strftime("%Y-%m-%d %H:%M:%S")


def run(meth, name, pt=None):
    dst = os.path.join(OUT, f"{name}.npz")
    if os.path.exists(dst):
        os.remove(dst)
    args = [sys.executable, "evaluate.py", meth, "endurance", "murky", str(NS)] + ([pt] if pt else [])
    t = time.time()
    subprocess.run(args, cwd=FC, check=True, env={**os.environ, "DEVICE": os.environ.get("DEVICE", "cuda"), "DDIM": str(DDIM)})
    shutil.move(os.path.join(FC, "out", f"{meth}_endurance_murky.npz"), dst)  # evaluate.py's fixed name -> ours
    print(name, f"{time.time() - t:.0f}s", flush=True)
    return np.load(dst)


def bss(z, zc, L):
    bm = brier(z["P"], z["Y"], L).mean(1); bb = brier(zc["P"], zc["Y"], L).mean(1)
    rng = np.random.default_rng(0); n = len(bm); bs = []
    for _ in range(1000):
        k = rng.integers(0, n, n); bs.append(1 - bm[k].mean() / bb[k].mean())
    return [float(1 - bm.mean() / bb.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


zc = run("clim", "clim")
models = {"best": os.path.join(OUT, "ckpt", f"step_{done['best_step']}.pt"),
          "at12k": os.path.join(FC, "runs", "diff2_endurance_murky.pt"),
          "run12k": os.path.join(OUT, "ckpt", "step_12000.pt")}
val = {r["step"]: r["val_loss"] for r in done["curve"]}
res = {}
for name, pt in models.items():
    z = run("diff", name, pt)
    assert (z["seeds"] == zc["seeds"]).all() and (z["Y"] == zc["Y"]).all()
    res[name] = {f"BSS{L}_clim": bss(z, zc, L) for L in LEADS}
res["best"].update(step=done["best_step"], val_loss=done["best_val_loss"], model=f"out/lc/ckpt/step_{done['best_step']}.pt")
res["at12k"].update(step=12000, model="runs/diff2_endurance_murky.pt (paper v2, CPU 12k steps)",
                    paper_BSS72_clim_cpu_sampler=0.6714)
res["run12k"].update(step=12000, val_loss=val.get(12000), model="out/lc/ckpt/step_12000.pt")
summary = {"curve": [{"step": r["step"], "val_loss": r["val_loss"], "train_loss": r["train_loss"]} for r in done["curve"]],
           **res, "seeds": NS, "seed_ids": zc["seeds"].tolist(), "ddim": DDIM, "paths": 50,
           "train": {k: done[k] for k in ("steps", "val_every", "best_step", "best_val_loss", "sec_per_step", "final_val_loss")},
           "started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
for n in ("best", "at12k", "run12k"):
    print(n, res[n]["step"], "BSS72", res[n]["BSS72_clim"], "BSS168", res[n]["BSS168_clim"])
