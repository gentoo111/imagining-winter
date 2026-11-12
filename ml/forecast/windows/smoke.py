"""S1-1b GPU smoke + speed measurement on the Windows 4060 (run as queue job smoke.cmd).
Writes D:\\LastEmbers\\e1\\out\\smoke\\done.json. Runs, all endurance/murky, same seeds as train.py:
  diff 200 steps on cuda (the smoke itself) and on cpu (loss reference, same seed)
  diff 1000 steps on cuda, gru 1000 on cuda / 500 on cpu (s/step; 200-step whole-second `secs` is too coarse)
  diffusion sampling with evaluate.py (DDIM 20, 50 paths) on a few test seeds, cuda and cpu.
Exit code 1 if the GPU/CPU val-loss ratio at step 200 is outside 0.5-2."""
import json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
FC = os.path.dirname(HERE)                                   # ...\ml\forecast
E1 = os.path.abspath(os.path.join(FC, "..", ".."))           # D:\LastEmbers\e1
OUT = os.path.join(E1, "out", "smoke")
os.makedirs(OUT, exist_ok=True)
PY = sys.executable
started = time.strftime("%Y-%m-%d %H:%M:%S")


def train(kind, steps, dev, name):
    pt = os.path.join(OUT, name + ".pt")
    for f in (pt, pt + ".log"):
        if os.path.exists(f):
            os.remove(f)                                     # train.py appends to its log
    t = time.time()
    subprocess.run([PY, "train.py", kind, "endurance", "murky", str(steps), pt], cwd=FC, check=True,
                   env={**os.environ, "DEVICE": dev})
    recs = [json.loads(ln) for ln in open(pt + ".log")]
    r = recs[-1]
    print(name, dev, r, flush=True)
    return {"it": r["it"], "loss": r["loss"], "val": r["val"], "train_secs": r["secs"],
            "wall_secs": round(time.time() - t, 1), "sec_per_step": r["secs"] / r["it"], "pt": pt}


def sample(dev, n, pt):
    npz = os.path.join(FC, "out", "diff_endurance_murky.npz")
    p = subprocess.run([PY, "evaluate.py", "diff", "endurance", "murky", str(n), pt], cwd=FC, check=True,
                       capture_output=True, text=True, env={**os.environ, "DEVICE": dev, "DDIM": "20"})
    cum = [float(m.group(1)) for m in re.finditer(r"^\d+ \d+ (\d+)s$", p.stdout, re.M)]
    import numpy as np
    nih = int(np.load(npz)["P"].shape[1])
    os.remove(npz)                                           # timing only: never leave a result-looking npz
    per = (cum[-1] - cum[0]) / (len(cum) - 1) if len(cum) > 1 else cum[-1]
    print("sample", dev, n, cum, flush=True)
    return {"seeds": n, "issue_hours_per_seed": nih, "paths": 50, "ddim": 20, "cum_secs": cum,
            "sec_per_seed": round(per, 2)}


import torch  # noqa: E402

g200 = train("diff", 200, "cuda", "diff_gpu200")
c200 = train("diff", 200, "cpu", "diff_cpu200")
g1k = train("diff", 1000, "cuda", "diff_gpu1000")
gg = train("gru", 1000, "cuda", "gru_gpu1000")
gc = train("gru", 500, "cpu", "gru_cpu500")
sg = sample("cuda", 10, g200["pt"])
sc = sample("cpu", 2, g200["pt"])
ratio = g200["val"] / c200["val"]
done = {
    "steps": 200, "device": "cuda",
    "sec_per_step": round(g1k["sec_per_step"], 4),
    "loss_first200_gpu": g200["val"], "loss_first200_cpu": c200["val"],
    "loss_cpu_ratio_first200": round(ratio, 4),
    "loss_first200_train_batch_gpu": g200["loss"], "loss_first200_train_batch_cpu": c200["loss"],
    "diff_train_sec_per_step": {"cuda": round(g1k["sec_per_step"], 4), "cpu": round(c200["sec_per_step"], 4),
                                "cuda_200": round(g200["sec_per_step"], 4)},
    "gru_train_sec_per_step": {"cuda": round(gg["sec_per_step"], 4), "cpu": round(gc["sec_per_step"], 4)},
    "diff_sample_sec_per_seed": {"cuda": sg["sec_per_seed"], "cpu": sc["sec_per_seed"]},
    "sampling": {"cuda": sg, "cpu": sc},
    "runs": {"diff_gpu200": g200, "diff_cpu200": c200, "diff_gpu1000": g1k, "gru_gpu1000": gg, "gru_cpu500": gc},
    "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "torch_threads_cpu": int(os.environ.get("TORCH_THREADS", "4")),
    "started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
}
json.dump(done, open(os.path.join(OUT, "done.json"), "w"), indent=1)
print(json.dumps(done, indent=1))
sys.exit(0 if 0.5 <= ratio <= 2 else 1)
