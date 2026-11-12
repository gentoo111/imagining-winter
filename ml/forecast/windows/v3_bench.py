"""S1-3 v3 speed check on the 4060 (queue job s13_0_bench.cmd, ~15 min): 600 train.py steps of v2 and of each v3
candidate (models.V3_CONFIGS A..C), with TF32 off and on. Only timing (s/step) and the step-600 validation loss are
kept; checkpoints are deleted. Writes D:\\LastEmbers\\e1\\out\\v3sel\\bench.json. Used to decide TF32 and the ETA of
the 9 x 100k-step selection runs; no selection is made from it."""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
FC = os.path.dirname(HERE)
sys.path.insert(0, FC)
E1 = os.path.abspath(os.path.join(FC, "..", ".."))
OUT = os.path.join(E1, "out", "v3sel", "bench")
os.makedirs(OUT, exist_ok=True)
STEPS = 600
from models import UNet1D, v3_net  # noqa: E402
import torch  # noqa: E402
res = []
for cfg in ("v2", "A", "B", "C"):
    n = UNet1D() if cfg == "v2" else v3_net(cfg)
    params = sum(p.numel() for p in n.parameters())
    for tf32 in ("0", "1"):
        pt = os.path.join(OUT, f"{cfg}_tf{tf32}.pt")
        for f in (pt, pt + ".log"):
            if os.path.exists(f):
                os.remove(f)
        env = {**os.environ, "DEVICE": os.environ.get("DEVICE", "cuda"), "VAL_EVERY": str(STEPS), "TF32": tf32}
        if cfg != "v2":
            env["V3_CFG"] = cfg
        t = time.time()
        subprocess.run([sys.executable, "train.py", "diff", "endurance", "murky", str(STEPS), pt], cwd=FC, check=True, env=env)
        r = [json.loads(ln) for ln in open(pt + ".log")][-1]
        rec = {"cfg": cfg, "tf32": tf32 == "1", "params": params, "steps": r["it"], "val": r["val"],
               "train_secs": r["secs"], "sec_per_step": round(r["secs"] / r["it"], 4), "wall_secs": round(time.time() - t, 1)}
        print(json.dumps(rec), flush=True)
        res.append(rec)
        os.remove(pt); os.remove(pt + ".log")
json.dump({"runs": res, "steps": STEPS, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "torch": torch.__version__, "finished": time.strftime("%Y-%m-%d %H:%M:%S")},
          open(os.path.join(E1, "out", "v3sel", "bench.json"), "w"), indent=1)
