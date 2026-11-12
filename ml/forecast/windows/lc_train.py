"""S1-2 learning curve, queue job 1 (lc_train.cmd): the unchanged v2 diffusion model (UNet1D, ~3e5 params),
endurance/murky, 100k steps on the 4060, validation + checkpoint every 2k steps.
Writes D:\\LastEmbers\\e1\\out\\lc\\{lc_best.pt, lc_best.pt.log, ckpt\\step_<it>.pt, done.json}.
~0.087 s/step measured in S1-1b -> ~2.4 h, inside the queue's 3 h cap (train.py cannot resume)."""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
FC = os.path.dirname(HERE)
E1 = os.path.abspath(os.path.join(FC, "..", ".."))
OUT = os.path.join(E1, "out", "lc")
STEPS, VAL_EVERY = 100000, 2000
os.makedirs(OUT, exist_ok=True)
pt = os.path.join(OUT, "lc_best.pt")
for f in (pt, pt + ".log", os.path.join(OUT, "done.json")):
    if os.path.exists(f):
        os.remove(f)                                         # train.py appends to its log
started = time.strftime("%Y-%m-%d %H:%M:%S"); t = time.time()
subprocess.run([sys.executable, "train.py", "diff", "endurance", "murky", str(STEPS), pt], cwd=FC, check=True,
               env={**os.environ, "DEVICE": os.environ.get("DEVICE", "cuda"), "VAL_EVERY": str(VAL_EVERY),
                    "CKPT_DIR": os.path.join(OUT, "ckpt")})
recs = [json.loads(ln) for ln in open(pt + ".log")]
curve = [{"step": r["it"], "val_loss": r["val"], "train_loss": r["loss"], "secs": r["secs"]} for r in recs]
b = min(curve, key=lambda r: r["val_loss"])
assert curve[-1]["step"] == STEPS, curve[-1]
missing = [r["step"] for r in curve if not os.path.exists(os.path.join(OUT, "ckpt", f"step_{r['step']}.pt"))]
assert not missing, missing
import torch  # noqa: E402
done = {"steps": STEPS, "val_every": VAL_EVERY, "best_step": b["step"], "best_val_loss": b["val_loss"],
        "sec_per_step": round(curve[-1]["secs"] / STEPS, 4), "wall_secs": round(time.time() - t),
        "final_val_loss": curve[-1]["val_loss"], "curve": curve,
        "model": "UNet1D v2 (models.py unchanged)", "profile": "endurance", "noise": "murky", "batch": 256,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "torch": torch.__version__,
        "started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
json.dump(done, open(os.path.join(OUT, "done.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in done.items() if k != "curve"}, indent=1))
