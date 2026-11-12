"""S1-5 full evaluation (RESEARCH-PLAN S1-5 / C3): diffusion v2 and v3 x 3 training seeds on all 200 test seeds
(Endurance/murky, 113 daily 07:00 issues), 200 paths per forecast, DDIM 50; path baselines on the same 200 seeds.
  eval200.py jobs [jobdir]       (main Mac) GPU queue jobs s16_zz_e200_{1_v2,2_v3s1,3_v3s2,4_v3s3,9_done}.cmd (named to
                                 run right after S1-6's s16_*) and the CPU-lane job c6_e200_pathbase.cmd
  eval200.py deploy              (main Mac) scp evaluate.py / path_baselines.py -> windows\\eval200\\ (+ SHA256SUMS), this
                                 script -> windows\\, gru npz -> out\\eval200\\; checks the v3 checkpoints' sha256 on Windows
  eval200.py verify              (Windows) deployed windows\\eval200\\ files == SHA256SUMS
  eval200.py run <model> --job J (GPU queue) chunks of 10 test seeds (parts\\<model>_<a>.npz) until the segment budget,
                                 then re-queues queue\\pending\\J.cmd; when all 20 chunks exist -> <name>_endurance_murky.npz
  eval200.py done                (GPU queue, last) -> out\\eval200\\done.json
  eval200.py pathbase            (CPU lane) PB_M=200 path_baselines.py 200 -> out\\eval200\\pathbase_endurance_murky_200.json
  eval200.py export <dir>        (main Mac) scp the merged npz / json files back to <dir>
Models: v2 = S1-2's best-validation checkpoint of the 100k-step v2 run (out\\lc\\ckpt\\step_88000.pt, 88k steps, the file S1-2's lc_eval.py scored; fp32 as trained),
v3s<k> = S1-4 seed k (out\\v3\\diff3_endurance_murky_s<k>.pt, V3_CFG=C, TF32 as trained). evaluate.py PATH_SCORES=1: per-issue
energy / variogram scores of all 200 paths are computed here (same metrics.py code as summarize.py); the merged npz keeps
P, Y, seeds, es, vs and the first seed's hourly paths (paths0); the 21-d daily-mean paths stay on Windows (pvec_<name>.npz).
Noise generator seeded with the chunk's first test seed, so a resumed segment reproduces the same samples.
HTCondor pool (env CJOB_STOP set by the lane wrapper): at the segment budget, or once the wrapper's STOP file exists, the
segment exits 85 (restarted in place, resuming from the parts on disk) instead of re-queueing; successors come from the
job's [after] jobs (ml/forecast/windows/condor/)."""
import hashlib, json, os, re, shutil, subprocess, sys, time

T_START = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
FC = os.path.dirname(HERE)
E1 = os.environ.get("E200_E1") or os.path.abspath(os.path.join(FC, "..", ".."))  # E200_* overrides: dry runs only
CODE = os.path.join(HERE, "eval200")
OUT = os.environ.get("E200_OUT", os.path.join(E1, "out", "eval200"))
PARTS = os.path.join(OUT, "parts")
QUEUE = os.environ.get("CHAIN_QUEUE", os.path.join(E1, "queue"))
NS, NSAMP, DDIM, CHUNK = (int(os.environ.get(k, v)) for k, v in
                          (("E200_NS", "200"), ("E200_NSAMP", "200"), ("E200_DDIM", "50"), ("E200_CHUNK", "10")))
PROF, NOISE = "endurance", "murky"
BUDGET = float(os.environ.get("E200_BUDGET", "10200"))  # s per segment (queue cap 10800 s, as train_chain.py / fc3b.py)
CJOB_STOP = os.environ.get("CJOB_STOP")  # HTCondor lane wrapper's STOP file: segment end = exit 85, no legacy re-queue
FILES = ("evaluate.py", "path_baselines.py")
FC_MODS = ("common.py", "metrics.py", "models.py", "batch.py", "baselines.py")  # imported from ml\forecast (data paths)
E1W = r"D:\LastEmbers\e1"
FCW = E1W + r"\ml\forecast"
PYW = E1W + r"\.venv\Scripts\python.exe"
MODELS = {  # model -> (npz name, checkpoint relative to E1, extra env)
    "v2": ("diff2", ("out", "lc", "ckpt", "step_88000.pt"), {}),
    "v3s1": ("diff3s1", ("out", "v3", "diff3_endurance_murky_s1.pt"), {"V3_CFG": "C", "TF32": "1"}),
    "v3s2": ("diff3s2", ("out", "v3", "diff3_endurance_murky_s2.pt"), {"V3_CFG": "C", "TF32": "1"}),
    "v3s3": ("diff3s3", ("out", "v3", "diff3_endurance_murky_s3.pt"), {"V3_CFG": "C", "TF32": "1"}),
}
GPU_JOBS = {"v2": "s16_zz_e200_1_v2", "v3s1": "s16_zz_e200_2_v3s1", "v3s2": "s16_zz_e200_3_v3s2", "v3s3": "s16_zz_e200_4_v3s3"}
DONE_JOB, CPU_JOB = "s16_zz_e200_9_done", "c6_e200_pathbase"
FIRST_EST = {"v2": 300.0, "v3s1": 2600.0, "v3s2": 2600.0, "v3s3": 2600.0}  # s per 10-seed chunk before any is measured


def log(**kw):
    print(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}), flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def job_text(name):
    head = ["@echo off"]
    pre = ["set PYTHONUTF8=1"]
    if name in GPU_JOBS.values():
        m = [k for k, v in GPU_JOBS.items() if v == name][0]
        head += [f"rem S1-5: {m} on test seeds 0-{NS - 1}, {NSAMP} paths, DDIM {DDIM}, {CHUNK}-seed chunks until the segment budget;",
                 f"rem re-queues {name}.cmd until all chunks exist -> out\\eval200\\{MODELS[m][0]}_endurance_murky.npz",
                 "rem Named to run right after S1-6's s16_* jobs. Code: windows\\eval200\\ (SHA256SUMS)."]
        pre += ["set DEVICE=cuda", "set TORCH_THREADS=2"]
        cmd = f"{PYW} -u windows\\eval200.py run {m} --job {name}"
    elif name == DONE_JOB:
        head += ["rem S1-5: after the 4 model jobs -> out\\eval200\\done.json"]
        cmd = f"{PYW} -u windows\\eval200.py done"
    else:
        head += ["rem S1-5 CPU lane: path baselines (clim / clock / GRU + analogue paths) on the 200 test seeds,",
                 "rem 200 paths per forecast -> out\\eval200\\pathbase_endurance_murky_200.json (CPU only)"]
        pre += ["set CUDA_VISIBLE_DEVICES=-1", "set DEVICE=cpu", "set OMP_NUM_THREADS=4", "set MKL_NUM_THREADS=4",
                "set OPENBLAS_NUM_THREADS=4", "set TORCH_THREADS=4"]
        cmd = f"{PYW} -u windows\\eval200.py pathbase"
    lines = head + pre + [f"cd /d {FCW}", f"if not exist {E1W}\\out\\eval200\\parts mkdir {E1W}\\out\\eval200\\parts",
                          f"{PYW} -u windows\\eval200.py verify", "if errorlevel 1 exit /b 1", cmd]
    return "\n".join(lines) + "\n"


def jobs(jobdir):
    os.makedirs(jobdir, exist_ok=True)
    for name in list(GPU_JOBS.values()) + [DONE_JOB, CPU_JOB]:
        with open(os.path.join(jobdir, name + ".cmd"), "w", newline="\r\n") as fh:
            fh.write(job_text(name))
        print("wrote", os.path.join(jobdir, name + ".cmd"))


def deploy():
    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "s15_eval200_deploy")
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    with open(os.path.join(tmp, "SHA256SUMS"), "w") as fh:
        for f in FILES:
            shutil.copy(os.path.join(FC, f), tmp); fh.write(f"{sha(os.path.join(FC, f))}  {f}\n")
    with open(os.path.join(tmp, "FC_SHA256SUMS"), "w") as fh:  # the imported modules as deployed (checked, not frozen)
        for f in FC_MODS:
            fh.write(f"{sha(os.path.join(FC, f))}  {f}\n")
    subprocess.run(["ssh", "w", rf"if not exist {FCW}\windows\eval200 mkdir {FCW}\windows\eval200 & "
                                rf"if not exist {E1W}\out\eval200\parts mkdir {E1W}\out\eval200\parts"], check=True)
    subprocess.run(["scp", "-q"] + [os.path.join(tmp, f) for f in FILES + ("SHA256SUMS", "FC_SHA256SUMS")] +
                   ["w:D:/LastEmbers/e1/ml/forecast/windows/eval200/"], check=True)
    subprocess.run(["scp", "-q", os.path.abspath(__file__), "w:D:/LastEmbers/e1/ml/forecast/windows/"], check=True)
    subprocess.run(["scp", "-q", os.path.join(FC, "out", "gru_endurance_murky.npz"), "w:D:/LastEmbers/e1/out/eval200/"], check=True)
    for m, (_, rel, _) in MODELS.items():
        o = subprocess.run(["ssh", "w", "certutil -hashfile " + E1W + "\\" + "\\".join(rel) + " SHA256"],
                           check=True, capture_output=True).stdout.decode("ascii", "replace")
        r = [ln.strip().replace(" ", "") for ln in o.splitlines() if re.fullmatch(r"[0-9a-f]{64}", ln.strip().replace(" ", ""))][0]
        loc = os.path.join(FC, "runs", os.path.basename(rel[-1])) if m != "v2" else None
        if loc and os.path.exists(loc):
            assert sha(loc) == r, (m, sha(loc), r)
        print(m, "\\".join(rel), r, "== main Mac" if loc and os.path.exists(loc) else "(Windows only)")
    print(open(os.path.join(tmp, "SHA256SUMS")).read())


def verify():
    sums = dict(ln.split()[::-1] for ln in open(os.path.join(CODE, "SHA256SUMS")) if ln.strip())
    for f in FILES:
        h = sha(os.path.join(CODE, f))
        assert sums[f] == h, (f, sums[f], h)
    print("eval200 code ok", flush=True)
    return fc_mods_changed()


def fc_mods_changed():
    """ml\forecast modules that differ from deploy time (another step re-copied sources): recorded, not fatal."""
    sums = dict(ln.split()[::-1] for ln in open(os.path.join(CODE, "FC_SHA256SUMS")) if ln.strip())
    bad = [f for f in FC_MODS if sha(os.path.join(FC, f)) != sums[f]]
    if bad:
        print("WARN ml\\forecast modules changed since deploy:", bad, flush=True)
    return bad


def part_file(m, a):
    return os.path.join(PARTS, f"{m}_{a:03d}.npz")


def requeue(job, m, nxt):
    if CJOB_STOP:  # HTCondor restarts the job in place (checkpoint_exit_code = 85)
        log(event="segment_end", job=job, model=m, next_seed=nxt, exit=85)
        sys.exit(85)
    os.makedirs(os.path.join(QUEUE, "pending"), exist_ok=True)
    p = os.path.join(QUEUE, "pending", job + ".cmd")
    assert not os.path.exists(p), f"{p} already queued"
    with open(p + ".tmp", "w", newline="\r\n") as fh:
        fh.write(job_text(job).replace("@echo off\n", f"@echo off\nrem re-queued: {m} from test seed {nxt}\n", 1))
    os.replace(p + ".tmp", p)  # the runner only picks *.cmd
    log(event="requeued", job=job, model=m, next_seed=nxt)


def run(m, job):
    import numpy as np
    changed = verify()
    os.makedirs(PARTS, exist_ok=True)
    name, rel, extra = MODELS[m]
    ckpt = os.path.join(E1, *rel)
    hist_f = os.path.join(PARTS, f"{m}.jsonl")
    hist = [json.loads(ln) for ln in open(hist_f)] if os.path.exists(hist_f) else []
    seg = 1 + len({h["segment_start"] for h in hist}); ran = 0
    for a in range(0, NS, CHUNK):
        dst = part_file(m, a)
        if os.path.exists(dst):
            continue
        est = max([h["secs"] for h in hist] or [FIRST_EST[m]])
        if ran and (time.time() - T_START + 1.1 * est > BUDGET or  # >= 1 chunk per segment (a chunk is << 3 h)
                    (CJOB_STOP and os.path.exists(CJOB_STOP))):
            requeue(job, m, a)
            return
        env = {**os.environ, "PYTHONPATH": FC, "DEVICE": os.environ.get("DEVICE", "cuda"), "PYTHONUTF8": "1",
               "NSAMP": str(NSAMP), "DDIM": str(DDIM), "PATH_SCORES": "1", "SEEDS": f"{a}:{min(a + CHUNK, NS)}",
               "GEN_SEED": str(a), "EVAL_CHUNK": os.environ.get("EVAL_CHUNK", "15"), "EVAL_OUT": dst + ".tmp.npz"}
        for k in ("V3_CFG", "TF32", "PART", "DIFF_VARIANT"):
            env.pop(k, None)
        env.update(extra)
        t = time.time()
        subprocess.run([sys.executable, os.path.join(CODE, "evaluate.py"), "diff", PROF, NOISE, str(NS), ckpt],
                       cwd=FC, check=True, env=env)
        os.replace(dst + ".tmp.npz", dst)
        rec = {"chunk": a, "secs": round(time.time() - t, 1), "segment": seg, "segment_start": round(T_START),
               "fc_mods_changed": changed}
        hist.append(rec)
        with open(hist_f, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        log(model=m, **rec); ran += 1
    merge(m)


def merge(m):
    import numpy as np
    name, rel, extra = MODELS[m]
    zs = [np.load(part_file(m, a)) for a in range(0, NS, CHUNK)]
    S = np.concatenate([z["seeds"] for z in zs])
    assert (S == np.arange(NS)).all(), S
    for z in zs:
        assert int(z["nsamp"]) == NSAMP and int(z["ddim"]) == DDIM
    cat = {k: np.concatenate([z[k] for z in zs]) for k in ("P", "Y", "es", "vs")}
    np.savez_compressed(os.path.join(OUT, f"{name}_{PROF}_{NOISE}.npz"), seeds=S, paths=np.zeros(0),
                        paths0=zs[0]["paths0"], nsamp=NSAMP, ddim=DDIM, **cat)
    np.savez_compressed(os.path.join(OUT, f"pvec_{name}.npz"), seeds=S, pvec=np.concatenate([z["pvec"] for z in zs]))
    hist = [json.loads(ln) for ln in open(os.path.join(PARTS, f"{m}.jsonl"))]
    segs = {}
    for h in hist:
        segs.setdefault(h["segment_start"], 0.0); segs[h["segment_start"]] += h["secs"]
    d = {"model": m, "name": name, "ckpt": "\\".join(rel), "ckpt_sha256": sha(os.path.join(E1, *rel)), **extra,
         "n_seeds": NS, "n_samples": NSAMP, "ddim_steps": DDIM, "secs": round(sum(h["secs"] for h in hist), 1),
         "secs_per_seed": round(sum(h["secs"] for h in hist) / NS, 1), "segments": [round(v, 1) for v in segs.values()],
         "fc_mods_changed": sorted({f for h in hist for f in h.get("fc_mods_changed", [])}),
         "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(d, open(os.path.join(OUT, f"{m}.json"), "w"), indent=1)
    log(event="merged", **d)


def done():
    res = {m: json.load(open(os.path.join(OUT, f"{m}.json"))) for m in MODELS if os.path.exists(os.path.join(OUT, f"{m}.json"))}
    d = {"n_seeds": NS, "n_samples": NSAMP, "ddim_steps": DDIM, "models_done": len(res),
         "models": list(res), "secs_per_model": {m: r["secs"] for m, r in res.items()},
         "ckpt_sha256": {m: r["ckpt_sha256"] for m, r in res.items()}, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    assert len(res) == len(MODELS), f"missing models: {set(MODELS) - set(res)}"
    with open(os.path.join(OUT, "done.json"), "w") as fh:
        fh.write(json.dumps(d) + "\n")
    print(json.dumps(d), flush=True)


def pathbase():
    verify()
    os.makedirs(OUT, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": FC, "PYTHONUTF8": "1", "PB_M": str(NSAMP), "PB_OUT": OUT,
           "PB_GRU": os.path.join(OUT, f"gru_{PROF}_{NOISE}.npz")}
    subprocess.run([sys.executable, os.path.join(CODE, "path_baselines.py"), str(NS), "2000"], cwd=FC, check=True, env=env)


def export(dst):
    os.makedirs(dst, exist_ok=True)
    fs = [f"{n}_{PROF}_{NOISE}.npz" for n, _, _ in MODELS.values()] + [f"{m}.json" for m in MODELS] + \
         ["done.json", f"pathbase_{PROF}_{NOISE}_{NS}.json"]
    for f in fs:
        subprocess.run(["scp", "-q", f"w:D:/LastEmbers/e1/out/eval200/{f}", dst], check=True)
        print("copied", f)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "jobs":
        jobs(sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "jobs"))
    elif cmd == "deploy":
        deploy()
    elif cmd == "verify":
        verify()
    elif cmd == "run":
        assert sys.argv[3] == "--job"
        run(sys.argv[2], sys.argv[4])
    elif cmd == "done":
        done()
    elif cmd == "pathbase":
        pathbase()
    elif cmd == "export":
        export(sys.argv[2])
    else:
        sys.exit(__doc__)
