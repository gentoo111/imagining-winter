"""S1-11a: offline daily 07:00 forecast tables for 3b (v3 diffusion, seed 1) and 3a' (frozen GRU, daily issue).
  fc3b.py weather                       (main Mac) weather_gen -> <out>/weather_{train,eval}.bin + seeds.json
  fc3b.py jobs <cpu jobdir> <gpu jobdir> (main Mac) the CPU-lane GRU job and the GPU-queue diffusion job (.cmd)
  fc3b.py gru-all --procs P --threads T (CPU lane) P parallel `gru --part k/P` processes, then gru-merge
  fc3b.py gru --part k/n                (one process: every n-th of the 2000 pool seeds starting at k)
  fc3b.py gru-merge                     -> <out>/features_gru_daily.npz
  fc3b.py diff --job NAME               (GPU queue, CUDA) chunks of 50 seeds until the segment budget, re-queues itself;
                                        when all 40 chunks exist -> features_diff_v3.npz, then done.json
Pools: train = 500000..500999 (fresh weather; disjoint from the forecasters' data seeds 200000-/210000-/220000- and
GAPS 400000-), eval = 300000..300999 (the storm evaluation pool of eval_e0 / e0_queue). Issue hours h = 24d + 7,
d = 0..119 (1-based game hours as feats / batch / fc_obs.FcState: hour h's row is D[h-1]). Features: fc_obs 6-d
(P(landing <= 24/48/72/120/168 h), expected hours / 240). GRU: fc_obs.GRU_CKPT, inputs = batch.Split (== the online
history / context, fc_obs.py --check / --check-table). Diffusion: models.v3_net("C") with the S1-4 seed-1 checkpoint,
DDIM 25, 50 paths per issue (the paper's v2 evaluation setting), landing = storm flag on for >= 3 h (evaluate.py),
29-class frequencies -> fc_obs.summarize_probs; noise generator seeded with the pool seed (chunking-independent).
prior (6,) = the forecast before the first 07:00 (empty history, hour-0 context), same rule for both forecasters.
npz keys: train (1000, 120, 6), eval (1000, 120, 6), prior (6,), seeds_train, seeds_eval, issue_hours.
Out dir: D:\\LastEmbers\\e1\\out\\fc3b on Windows, ml/forecast/out/fc3b on the Mac (FC3B_OUT overrides)."""
import argparse, json, os, subprocess, sys, time

T_START = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
FC = os.path.dirname(HERE)
E1 = os.path.abspath(os.path.join(FC, "..", ".."))
sys.path.insert(0, FC)
OUT = os.environ.get("FC3B_OUT") or (os.path.join(E1, "out", "fc3b") if os.name == "nt" else os.path.join(FC, "out", "fc3b"))
PARTS = os.path.join(OUT, "parts")
QUEUE = os.environ.get("CHAIN_QUEUE", os.path.join(E1, "queue"))
N, TRAIN0, EVAL0, DAYS, ISSUE_H = 1000, 500000, 300000, 120, 7
PROFILE, NOISE = "endurance", "murky"
DDIM, M, CHUNK = 25, 50, 50
NCHUNK = 2 * N // CHUNK
V3_CFG = "C"
V3_CKPT = os.environ.get("V3_CKPT", r"D:\LastEmbers\e1\out\v3\diff3_endurance_murky_s1.pt")
BUDGET = 10200  # s per segment (queue cap 10800 s, as train_chain.py)
HOURS = [24 * d + ISSUE_H for d in range(DAYS)]
POOLS = ("train", "eval")
E1W = r"D:\LastEmbers\e1"
PYW = E1W + r"\.venv\Scripts\python.exe"
FCW = E1W + r"\ml\forecast"


def seeds_of(pool):
    return list(range(TRAIN0 if pool == "train" else EVAL0, (TRAIN0 if pool == "train" else EVAL0) + N))


def log(**kw):
    print(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}), flush=True)


def weather():
    wg = os.environ.get("WEATHER_GEN", os.path.expanduser("~/Documents/Games/last-embers-e1/simrs/target/release/weather_gen"))
    data = os.path.join(FC, "data", "sim_339afa0", "data")
    os.makedirs(OUT, exist_ok=True)
    for pool in POOLS:
        subprocess.run([wg, data, os.path.join(OUT, f"weather_{pool}.bin"), str(seeds_of(pool)[0]), str(N), str(DAYS),
                        PROFILE, NOISE], check=True)
    json.dump({"train": seeds_of("train"), "eval": seeds_of("eval"), "days": DAYS, "issue_hour": ISSUE_H,
               "issue_hours": HOURS, "profile": PROFILE, "noise": NOISE,
               "weather": {p: f"weather_{p}.bin" for p in POOLS},
               "weather_gen": "e1 simrs weather_gen <sim_339afa0 data> <out> <seed_from> 1000 120 endurance murky",
               "rule": "train = fresh seeds 500000.. (disjoint from forecaster data 200000-/210000-/220000-/400000- and the "
                       "eval pool); eval = storm evaluation pool 300000:301000 (eval_e0 / e0_queue)"},
              open(os.path.join(OUT, "seeds.json"), "w"), indent=1)
    log(weather="ok", out=OUT)


def split(pool):
    """batch.Split over the pool's weather file (same features as training / the online FcState)."""
    import numpy as np
    import batch
    from common import H, F
    path = os.path.join(OUT, f"weather_{pool}.bin")
    batch.load = lambda profile, noise, sp: np.fromfile(path, dtype="<i4").reshape(-1, H, len(F))
    sp = batch.Split(PROFILE, NOISE, pool)
    assert sp.D.shape[0] == N, sp.D.shape
    return sp


def prior_inputs():
    """Hour-0 inputs: empty history, fc_obs.FcState.ctx at hour 0 (never stormed)."""
    import numpy as np
    from feats import HIST, NFEAT
    return np.zeros((1, HIST, NFEAT), np.float32), np.array([[0.0, -1.0, 0.0, 0.0, 0.0, 0.0]], np.float32)


def gru_part(k, n, threads):
    import numpy as np, torch
    torch.set_num_threads(threads)
    import fc_obs
    net = fc_obs.gru()
    hrs = np.array(HOURS)
    os.makedirs(PARTS, exist_ok=True)
    jobs = [(p, i) for p in POOLS for i in range(N)][k::n]
    res = {p: ([], []) for p in POOLS}
    t0 = time.time()
    for pool in POOLS:
        idx = [i for p, i in jobs if p == pool]
        if not idx:
            continue
        sp = split(pool)
        for a in range(0, len(idx), 16):
            b = idx[a:a + 16]
            s = np.repeat(np.array(b), len(hrs)); h = np.tile(hrs, len(b))
            x, c, _, _ = sp.batch(s, h, want_future=False)
            with torch.inference_mode():
                f = fc_obs.summarize(net(torch.from_numpy(x), torch.from_numpy(c)))
            res[pool][0].extend(b); res[pool][1].append(f.reshape(len(b), len(hrs), fc_obs.N_FC))
    out = os.path.join(PARTS, f"gru_{k}of{n}.npz")
    np.savez(out + ".tmp.npz", **{f"{p}_idx": np.array(res[p][0], int) for p in POOLS},
             **{f"{p}_feat": (np.concatenate(res[p][1]) if res[p][1] else np.zeros((0, DAYS, fc_obs.N_FC), np.float32)) for p in POOLS})
    os.replace(out + ".tmp.npz", out)
    log(part=f"{k}/{n}", seeds=len(jobs), secs=round(time.time() - t0, 1))


def gru_all(procs, threads):
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads), "CUDA_VISIBLE_DEVICES": "-1",
           "DEVICE": "cpu"}
    t0 = time.time()
    ps = [subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), "gru", "--part", f"{k}/{procs}",
                            "--threads", str(threads)], env=env) for k in range(procs)]
    rc = [p.wait() for p in ps]
    assert not any(rc), rc
    gru_merge(procs, time.time() - t0)


def _npz_out(name, tabs, prior, meta):
    """Probability columns are clipped to [0, 1]: the float32 GRU softmax summed in float64 can reach 1 + 1.2e-7."""
    import numpy as np
    clip = lambda x: np.concatenate([np.clip(x[..., :5], 0.0, 1.0), x[..., 5:]], -1).astype(np.float32)
    tabs = {k: clip(v) for k, v in tabs.items()}; prior = clip(np.asarray(prior))
    out = os.path.join(OUT, name)
    np.savez(out + ".tmp.npz", train=tabs["train"], eval=tabs["eval"], prior=prior.astype(np.float32),
             seeds_train=np.array(seeds_of("train")), seeds_eval=np.array(seeds_of("eval")), issue_hours=np.array(HOURS),
             meta=np.array(json.dumps(meta)))
    os.replace(out + ".tmp.npz", out)
    return out


def gru_merge(procs, secs=None):
    import numpy as np
    import fc_obs
    tabs = {p: np.full((N, DAYS, fc_obs.N_FC), np.nan, np.float32) for p in POOLS}
    for k in range(procs):
        z = np.load(os.path.join(PARTS, f"gru_{k}of{procs}.npz"))
        for p in POOLS:
            tabs[p][z[f"{p}_idx"]] = z[f"{p}_feat"]
    assert all(np.isfinite(t).all() for t in tabs.values()), "missing GRU rows"
    st = fc_obs.FcState(1); st.reset(0)
    x0, c0 = prior_inputs()
    import torch
    with torch.inference_mode():
        pr = fc_obs.summarize(fc_obs.gru()(torch.from_numpy(x0), torch.from_numpy(c0)))[0]
    assert np.array_equal(pr, st.prior), (pr, st.prior)
    gd = os.path.join(OUT, "gru_done.json")
    if secs is None:  # a re-merge of existing parts keeps the original compute time
        secs = json.load(open(gd))["secs"] if os.path.exists(gd) else 0.0
    out = _npz_out("features_gru_daily.npz", tabs, pr, {"forecaster": "gru", "ckpt": fc_obs.GRU_CKPT, "procs": procs,
                                                        "secs": round(secs, 1)})
    json.dump({"secs": round(secs, 1), "procs": procs, "finished": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(os.path.join(OUT, "gru_done.json"), "w"))
    log(merged=out, secs=round(secs, 1))


def diff_sample(net, dif, x, c, gen, batch):
    """(B issues) -> (B, NB+1) landing-class frequencies over M paths (evaluate.py's rule)."""
    import numpy as np, torch
    from common import NB, HOR
    x = torch.tensor(x); c = torch.tensor(c)
    P = []
    per = max(1, batch // M)
    for a in range(0, len(x), per):
        xb, cb = x[a:a + per], c[a:a + per]
        nb = len(xb)
        with torch.no_grad():
            cv = tuple(t.repeat_interleave(M, 0) for t in net.cond_vec(xb, cb))
            yy = dif.sample(net, cv, steps=DDIM, gen=gen).cpu().numpy()
        yy = yy.reshape(nb, M, 3, HOR).transpose(0, 1, 3, 2)
        ins0 = cb[:, 3].cpu().numpy() > 0
        raw = yy[..., 2] > 0
        fl = raw & np.concatenate([raw[..., 1:], raw[..., -1:]], -1) & np.concatenate([raw[..., 2:], raw[..., -2:]], -1)
        prev = np.concatenate([np.broadcast_to(ins0[:, None, None], (nb, M, 1)), fl[..., :-1]], -1)
        edge = fl & ~prev
        first = np.where(edge.any(-1), edge.argmax(-1) + 1, 0)
        cls = np.where(first > 0, (first - 1) // 6, NB)
        P.append(np.stack([np.bincount(cls[i], minlength=NB + 1) / M for i in range(nb)]))
    return np.concatenate(P)


def diff(job, budget):
    import numpy as np
    dev = os.environ.get("DEVICE", "cuda")
    import torch
    if dev != "cpu":
        torch.set_default_device(dev)  # before importing models (Diffusion tables / noise follow the default device)
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    from models import v3_net, Diffusion
    import fc_obs
    net = v3_net(V3_CFG)
    net.load_state_dict(torch.load(V3_CKPT, map_location=dev, weights_only=True)); net.eval()
    dif = Diffusion()
    bs = int(os.environ.get("FC3B_BATCH", "6000"))
    os.makedirs(PARTS, exist_ok=True)
    hrs = np.array(HOURS)
    sps = {}
    chunk_secs = []
    for ci in range(NCHUNK):
        out = os.path.join(PARTS, f"diff_c{ci:02d}.npz")
        if os.path.exists(out):
            continue
        est = max(chunk_secs) if chunk_secs else 0
        if time.time() - T_START + est * 1.15 > budget:
            break
        pool = POOLS[ci * CHUNK // N]
        i0 = ci * CHUNK % N
        sp = sps.get(pool) or sps.setdefault(pool, split(pool))
        t = time.time()
        feats = np.zeros((CHUNK, DAYS, fc_obs.N_FC), np.float32)
        for j in range(CHUNK):
            i = i0 + j
            x, c, _, _ = sp.batch(np.full(len(hrs), i), hrs, want_future=False)
            g = torch.Generator(device=dev).manual_seed(seeds_of(pool)[i])
            feats[j] = fc_obs.summarize_probs(diff_sample(net, dif, x, c, g, bs).astype(np.float64))
        np.savez(out + ".tmp.npz", pool=pool, idx=np.arange(i0, i0 + CHUNK), feat=feats)
        os.replace(out + ".tmp.npz", out)
        chunk_secs.append(time.time() - t)
        log(chunk=ci, pool=pool, seeds=f"{i0}..{i0 + CHUNK - 1}", secs=round(chunk_secs[-1], 1),
            seed_secs=round(chunk_secs[-1] / CHUNK, 2))
    left = [ci for ci in range(NCHUNK) if not os.path.exists(os.path.join(PARTS, f"diff_c{ci:02d}.npz"))]
    seg = {"sec": round(time.time() - T_START), "chunks": len(chunk_secs), "left": len(left),
           "chunk_secs": [round(s, 1) for s in chunk_secs]}
    segf = os.path.join(OUT, "diff_segments.json")
    segs = json.load(open(segf)) if os.path.exists(segf) else []
    segs.append(seg); json.dump(segs, open(segf, "w"), indent=1)
    if left:
        assert chunk_secs, "no chunk fits the segment budget"
        requeue(job, budget)
        return
    tabs = {p: np.full((N, DAYS, fc_obs.N_FC), np.nan, np.float32) for p in POOLS}
    for ci in range(NCHUNK):
        z = np.load(os.path.join(PARTS, f"diff_c{ci:02d}.npz"))
        tabs[str(z["pool"])][z["idx"]] = z["feat"]
    assert all(np.isfinite(t).all() for t in tabs.values())
    sec = sum(s["sec"] for s in segs)
    _npz_out("features_diff_v3.npz", tabs, gru_prior(), diff_meta(sec))
    finish(sec)


def gru_prior():
    """Prior of both tables (hours 0-6, before the first 07:00 issue) = the GRU table's hour-0 forecast: the observer-
    known 'no storm possible yet' (P = 0, expected 240 h). Driver decision 2026-11-06: the diffusion model sampled at
    hour 0 (empty history, never seen in training: it trains on hours >= 24) gave P72 = 0.90, a spurious signal; with a
    shared prior the 3b and 3a' rungs differ only in the 07:00 issues."""
    import numpy as np
    return np.load(os.path.join(OUT, "features_gru_daily.npz"))["prior"]


def diff_meta(sec):
    return {"forecaster": "diff_v3", "cfg": V3_CFG, "ckpt": V3_CKPT, "ddim": DDIM, "paths": M, "tf32": True, "secs": sec,
            "prior": "GRU table prior (observer-known no-storm-yet; driver 2026-11-06), not a diffusion sample"}


def fix_prior():
    """Rewrite features_diff_v3.npz's prior with gru_prior() (tables unchanged; the old prior kept as prior_diff_hour0)."""
    import numpy as np
    p = os.path.join(OUT, "features_diff_v3.npz")
    z = dict(np.load(p))
    old = z["prior"].copy()
    tabs = {"train": z["train"], "eval": z["eval"]}
    m0 = json.loads(str(z["meta"]))
    meta = {**m0, **diff_meta(m0["secs"]), "prior_diff_hour0": m0.get("prior_diff_hour0", old.tolist())}
    _npz_out("features_diff_v3.npz", tabs, gru_prior(), meta)
    log(fixed_prior=gru_prior().tolist(), old=old.tolist())


def finish(diff_sec):
    import fc_obs
    gd = os.path.join(OUT, "gru_done.json")
    assert os.path.exists(os.path.join(OUT, "features_gru_daily.npz")) and os.path.exists(gd), "GRU table missing (CPU lane)"
    g = json.load(open(gd))
    done = {"train_seeds": N, "eval_seeds": N, "issue_hour": ISSUE_H, "features": ["diff_v3", "gru_daily"],
            "v3_ckpt": V3_CKPT, "v3_cfg": V3_CFG, "gru_ckpt": fc_obs.GRU_CKPT, "days": DAYS, "ddim": DDIM, "paths": M,
            "sec": {"diff_v3": diff_sec, "gru_daily": g["secs"]}, "gru_procs": g["procs"],
            "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(done, open(os.path.join(OUT, "done.json"), "w"))  # one line: the pipeline check reads it with a regex
    log(done=done)


def requeue(job, budget):
    """train_chain.py convention: write queue\\pending\\<job>.cmd re-running this segment with the same environment."""
    keep = ["DEVICE", "TORCH_THREADS", "FC3B_BATCH", "V3_CKPT", "CHAIN_QUEUE", "FC3B_OUT"]
    lines = ["@echo off", f"rem fc3b.py diff: next segment of {job} (S1-11a diffusion table)", "set PYTHONUTF8=1"] + \
            [f"set {k}={os.environ[k]}" for k in keep if os.environ.get(k)] + \
            [f"cd /d {FC}", subprocess.list2cmdline([sys.executable, "-u", os.path.abspath(__file__), "diff", "--job", job,
                                                     "--budget", str(budget)])]
    os.makedirs(os.path.join(QUEUE, "pending"), exist_ok=True)
    p = os.path.join(QUEUE, "pending", job + ".cmd")
    assert not os.path.exists(p), f"{p} already queued"
    with open(p + ".tmp", "w", newline="\r\n") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(p + ".tmp", p)
    log(requeued=p)


def jobs(cpu_dir, gpu_dir, procs=9, threads=2):
    os.makedirs(cpu_dir, exist_ok=True); os.makedirs(gpu_dir, exist_ok=True)
    cpu = ["@echo off",
           f"rem S1-11a CPU lane: frozen GRU daily 07:00 table for the 2000 pool seeds, {procs} processes x {threads} threads",
           "rem (18 of 20 logical threads; GPU hidden). Writes out\\fc3b\\features_gru_daily.npz + gru_done.json",
           "set PYTHONUTF8=1", "set CUDA_VISIBLE_DEVICES=-1", "set DEVICE=cpu", f"set OMP_NUM_THREADS={threads}",
           f"set TORCH_THREADS={threads}", f"cd /d {FCW}",
           f"{PYW} -u windows\\fc3b.py gru-all --procs {procs} --threads {threads}"]
    with open(os.path.join(cpu_dir, "c1_fc3b_gru.cmd"), "w", newline="\r\n") as fh:
        fh.write("\n".join(cpu) + "\n")
    bench = ["@echo off",
             "rem S1-11a CPU lane: Windows frozen-sim rung-3 PPO throughput for s1-11b (k = 1, 2, 3, 6 concurrent runs,",
             "rem 3a hyperparameters, gru_daily table, 18 threads split per run; nothing kept). Writes out\\fc3b\\ppo_bench.json",
             "set PYTHONUTF8=1", "set CUDA_VISIBLE_DEVICES=-1", "set DEVICE=cpu", f"cd /d {FCW}",
             f"{PYW} -u windows\\ppo_bench.py --ks 1,2,3,6 --secs 600 --threads 18"]
    with open(os.path.join(cpu_dir, "c2_ppo_bench.cmd"), "w", newline="\r\n") as fh:
        fh.write("\n".join(bench) + "\n")
    bench1 = bench[:3] + ["rem variant: one rayon thread per run's VecEnv (8 envs; less thread sync)"] + bench[3:-1] + [
        f"{PYW} -u windows\\ppo_bench.py --ks 1,6 --secs 600 --threads 18 --rayon 1 --tag _rayon1"]
    with open(os.path.join(cpu_dir, "c3_ppo_bench_rayon1.cmd"), "w", newline="\r\n") as fh:
        fh.write("\n".join(bench1) + "\n")
    gpu = ["@echo off",
           "rem S1-11a GPU: v3 (cfgC, S1-4 seed 1) daily 07:00 forecast table, DDIM 25 x 50 paths, 2000 pool seeds in 40",
           "rem chunks; segments <= 10200 s, re-queues itself until done; then features_diff_v3.npz + out\\fc3b\\done.json.",
           "rem Named to run right after s14_s3 (before s14_z_summary / s16_*).",
           "set PYTHONUTF8=1", "set DEVICE=cuda", "set TORCH_THREADS=2", "set FC3B_BATCH=3000", f"cd /d {FCW}",
           f"{PYW} -u windows\\fc3b.py diff --job s14_t_fc3b_diff"]
    with open(os.path.join(gpu_dir, "s14_t_fc3b_diff.cmd"), "w", newline="\r\n") as fh:
        fh.write("\n".join(gpu) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("args", nargs="*")
    ap.add_argument("--part", default="0/1"); ap.add_argument("--procs", type=int, default=9)
    ap.add_argument("--threads", type=int, default=2); ap.add_argument("--job", default="s14_t_fc3b_diff")
    ap.add_argument("--budget", type=float, default=BUDGET)
    a = ap.parse_args()
    if a.cmd == "weather":
        weather()
    elif a.cmd == "jobs":
        jobs(a.args[0], a.args[1], a.procs, a.threads)
    elif a.cmd == "gru":
        k, n = map(int, a.part.split("/")); gru_part(k, n, a.threads)
    elif a.cmd == "gru-all":
        gru_all(a.procs, a.threads)
    elif a.cmd == "gru-merge":
        gru_merge(a.procs)
    elif a.cmd == "fix-prior":
        fix_prior()
    elif a.cmd == "diff":
        assert 0 < a.budget <= 10800
        diff(a.job, a.budget)
    else:
        sys.exit(__doc__)
