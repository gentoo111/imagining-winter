"""S1-12 (b) observation-model ablation (RESEARCH-PLAN S1-12 / GAPS #20): Endurance on the s1-12a windnoise data
(weather_gen wind_noise=0.2: the reported wind sector shifted +-1 in 20% of hours; everything else as murky), 60 test seeds.
  PF oracle with the flip-aware wind-direction likelihood (pf.py, common.NZ["murky_windnoise"]) + climatology  -> CPU lane
  GRU retrained on the windnoise training set, 16k steps (main-table recipe), then evaluated                      -> CPU lane
  diffusion retrained with the selected-v3 recipe (cfgC, lr 1e-3, SEED 1, 100k steps, TF32), then 200 paths, DDIM 50 -> GPU
Commands:
  windnoise.py jobs [jobdir]   (main Mac) c8_wn_a_pf.cmd, c8_wn_b_gru.cmd (CPU lane queue-cpu), s1c_wn_a_train.cmd (GPU queue,
                               after s16_* / s18_* / s19_*); the eval job s1c_wn_b_eval.cmd is queued by queue_eval
  windnoise.py deploy          (main Mac) scp common/pf/evaluate/summarize/train.py -> windows\\windnoise\\ (+ SHA256SUMS), this
                               script -> windows\\, the three windnoise .bin files -> ml\\forecast\\data\\ (sha256 checked)
  windnoise.py verify          (Windows) deployed files == SHA256SUMS, data sha256 == DATA_SHA256
  windnoise.py pf              (CPU lane) clim + PF (10-seed chunks, re-queues itself on queue-cpu) -> out\\windnoise\\
                               {clim,pf}_endurance_murky_windnoise.npz, mark pf
  windnoise.py gru_eval        (CPU lane, train_chain --then) GRU on the 60 seeds -> gru_endurance_murky_windnoise.npz, mark gru
  windnoise.py queue_eval      (GPU queue, train_chain --then) writes queue\\pending\\s1c_wn_b_eval.cmd
  windnoise.py diff --job J    (GPU queue) 10-seed chunks until the segment budget (re-queues J), then merge, mark diff
  windnoise.py mark            done.json {"pf", "gru", "diff": "done"|"pending"}; when all are done, summarize.py first
  windnoise.py export          (main Mac) scp results -> ml/forecast/out/windnoise/, summary_endurance_murky.json
  windnoise.py ref <eval200>   (main Mac, CPU seconds) the same methods on the same 60 test seeds of the original murky data:
                               summarize.py SUM_SEEDS=0:60 over S1-5's eval200 npz (clim, pf, gru16k, diff = v3 seed 1)
                               -> ml/forecast/out/windnoise/ref60/summary_endurance_murky.json
  windnoise.py tex             (main Mac) generated macro block '% BEGIN wn' .. '% END wn' of docs/paper/main.tex
                               (\\wn* macros, tab:windnoise rows) from the two summaries; no number typed by hand
The deployed copies run with PYTHONPATH=ml\\forecast (batch / models / metrics / baselines / feats, checked against
FC_SHA256SUMS, warn only) and E1_FC=ml\\forecast (common.py's data and balance.json paths); a script's own directory is
first on sys.path, so evaluate.py / train.py / summarize.py in windows\\windnoise\\ import the deployed common.py and pf.py.
HTCondor pool (env CJOB_STOP set by the lane wrapper): a chunked segment exits 85 at the budget or once the wrapper's STOP
file exists (restarted in place, resuming from the parts on disk) instead of re-queueing, and queue_eval writes nothing:
s1c_wn_b_eval is its own Condor job after s1c_wn_a_train (docs/ml/windows/condor/ on e1/s1-5-full-eval-200)."""
import hashlib, json, os, re, shutil, subprocess, sys, time

T_START = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
FC = os.path.dirname(HERE)
E1 = os.path.abspath(os.path.join(FC, "..", ".."))
CODE = os.path.join(HERE, "windnoise")
OUT = os.path.join(E1, "out", "windnoise")
PARTS = os.path.join(OUT, "parts")
QUEUE = os.path.join(E1, "queue")
PROF, NOISE, NS = "endurance", "murky_windnoise", 60
NSAMP, DDIM, CHUNK = 200, 50, 10
BUDGET = 10200.0  # s per segment (queue cap 10800 s)
CJOB_STOP = os.environ.get("CJOB_STOP")  # HTCondor lane wrapper's STOP file: segment end = exit 85, no legacy re-queue
FIRST_EST = 2600.0  # s per 10-seed chunk before one is measured (S1-5's v3 estimate, 200 paths)
FILES = ("common.py", "pf.py", "evaluate.py", "summarize.py", "train.py")
FC_MODS = ("batch.py", "models.py", "metrics.py", "baselines.py", "feats.py")
DATA = [f"{PROF}_{NOISE}_{s}.bin" for s in ("train", "val", "test")]
E1W = r"D:\LastEmbers\e1"
FCW = E1W + r"\ml\forecast"
PYW = E1W + r"\.venv\Scripts\python.exe"
GRU_PT = E1W + r"\out\windnoise\gru16k_windnoise.pt"
DIFF_PT = E1W + r"\out\windnoise\diff3_windnoise_s1.pt"
PARTS_DONE = ("pf", "gru", "diff")
J_PF, J_GRU, J_TRAIN, J_EVAL = "c8_wn_a_pf", "c8_wn_b_gru", "s1c_wn_a_train", "s1c_wn_b_eval"
BASE_ENV = ["set PYTHONUTF8=1", f"set PYTHONPATH={FCW}", f"set E1_FC={FCW}"]
CPU_ENV = ["set CUDA_VISIBLE_DEVICES=-1", "set DEVICE=cpu", "set OMP_NUM_THREADS=4", "set MKL_NUM_THREADS=4",
           "set OPENBLAS_NUM_THREADS=4", "set TORCH_THREADS=4"]


def log(**kw):
    print(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}), flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def job_text(name):
    pre = ["@echo off"]
    if name == J_PF:
        pre += ["rem S1-12b CPU lane: climatology + PF oracle (flip-aware wind likelihood) on the 60 windnoise test seeds",
                "rem -> out\\windnoise\\{clim,pf}_endurance_murky_windnoise.npz. Code: windows\\windnoise\\ (SHA256SUMS)."]
        env, cmd = BASE_ENV + CPU_ENV, f"{PYW} -u windows\\windnoise.py pf"
    elif name == J_GRU:
        pre += ["rem S1-12b CPU lane: GRU retrained on the windnoise training set, 16000 steps (main-table recipe), chained",
                "rem through queue-cpu (train_chain.py re-queues c8_wn_b_gru.cmd), then evaluated on the 60 test seeds."]
        env = BASE_ENV + CPU_ENV + [f"set CHAIN_QUEUE={E1W}\\queue-cpu", f"set CHAIN_TRAIN_PY={FCW}\\windows\\windnoise\\train.py",
                                    "set CHAIN_ENV=CUDA_VISIBLE_DEVICES,OMP_NUM_THREADS,MKL_NUM_THREADS,OPENBLAS_NUM_THREADS,"
                                    "E1_FC,PYTHONUTF8,CHAIN_ENV"]
        cmd = (f"{PYW} -u windows\\train_chain.py gru {PROF} {NOISE} 16000 {GRU_PT} --job {J_GRU} "
               f"--done {E1W}\\out\\windnoise\\train_gru.json --then \"windows\\windnoise.py gru_eval\"")
    elif name == J_TRAIN:
        pre += ["rem S1-12b GPU: diffusion retrained on the windnoise training set with the selected-v3 recipe (cfgC, lr 1e-3,",
                "rem SEED 1, 100000 steps, TF32, VAL_EVERY 1000) chained through the queue; when done queues s1c_wn_b_eval.cmd.",
                "rem Named to run after S1-5/6 (s16_*), S1-8 (s18_*) and S1-9 (s19_*)."]
        env = BASE_ENV + ["set DEVICE=cuda", "set SEED=1", "set V3_CFG=C", "set LR=1e-3", "set TF32=1", "set VAL_EVERY=1000",
                          f"set CHAIN_TRAIN_PY={FCW}\\windows\\windnoise\\train.py", "set CHAIN_ENV=V3_CFG,LR,TF32,SEED,E1_FC,PYTHONUTF8,CHAIN_ENV"]
        cmd = (f"{PYW} -u windows\\train_chain.py diff {PROF} {NOISE} 100000 {DIFF_PT} --job {J_TRAIN} "
               f"--done {E1W}\\out\\windnoise\\train_diff.json --then \"windows\\windnoise.py queue_eval\"")
    elif name == J_EVAL:
        pre += ["rem S1-12b GPU: the windnoise-trained v3 on the 60 windnoise test seeds, 200 paths, DDIM 50, 10-seed chunks",
                "rem until the segment budget (re-queues itself); then merge, mark diff (summary when all three are done)."]
        env, cmd = BASE_ENV + ["set DEVICE=cuda", "set TORCH_THREADS=2"], f"{PYW} -u windows\\windnoise.py diff --job {J_EVAL}"
    else:
        raise KeyError(name)
    lines = pre + env + [f"cd /d {FCW}", f"if not exist {E1W}\\out\\windnoise\\parts mkdir {E1W}\\out\\windnoise\\parts",
                         f"{PYW} -u windows\\windnoise.py verify", "if errorlevel 1 exit /b 1", cmd]
    return "\n".join(lines) + "\n"


def jobs(jobdir):
    os.makedirs(jobdir, exist_ok=True)
    for name in (J_PF, J_GRU, J_TRAIN):
        with open(os.path.join(jobdir, name + ".cmd"), "w", newline="\r\n") as fh:
            fh.write(job_text(name))
        print("wrote", os.path.join(jobdir, name + ".cmd"))


def deploy():
    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "s112b_windnoise_deploy")
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    for f in FILES:
        shutil.copy(os.path.join(FC, f), tmp)
    with open(os.path.join(tmp, "SHA256SUMS"), "w") as fh:
        for f in FILES:
            fh.write(f"{sha(os.path.join(FC, f))}  {f}\n")
    with open(os.path.join(tmp, "FC_SHA256SUMS"), "w") as fh:
        for f in FC_MODS:
            fh.write(f"{sha(os.path.join(FC, f))}  {f}\n")
    with open(os.path.join(tmp, "DATA_SHA256"), "w") as fh:
        for f in DATA:
            fh.write(f"{sha(os.path.join(FC, 'data', f))}  {f}\n")
    subprocess.run(["ssh", "w", rf"if not exist {FCW}\windows\windnoise mkdir {FCW}\windows\windnoise"], check=True)
    subprocess.run(["ssh", "w", rf"if not exist {E1W}\out\windnoise\parts mkdir {E1W}\out\windnoise\parts"], check=True)
    subprocess.run(["scp", "-q"] + [os.path.join(tmp, f) for f in FILES + ("SHA256SUMS", "FC_SHA256SUMS", "DATA_SHA256")] +
                   ["w:D:/LastEmbers/e1/ml/forecast/windows/windnoise/"], check=True)
    subprocess.run(["scp", "-q", os.path.abspath(__file__), "w:D:/LastEmbers/e1/ml/forecast/windows/"], check=True)
    for f in DATA:
        o = subprocess.run(["ssh", "w", rf"if exist {FCW}\data\{f} certutil -hashfile {FCW}\data\{f} SHA256"],
                           capture_output=True).stdout.decode("ascii", "replace")
        have = [ln.strip().replace(" ", "") for ln in o.splitlines() if re.fullmatch(r"[0-9a-f]{64}", ln.strip().replace(" ", ""))]
        if have and have[0] == sha(os.path.join(FC, "data", f)):
            print("data ok", f)
            continue
        subprocess.run(["scp", "-q", os.path.join(FC, "data", f), f"w:D:/LastEmbers/e1/ml/forecast/data/{f}"], check=True)
        print("copied", f)
    print(open(os.path.join(tmp, "SHA256SUMS")).read())


def verify():
    sums = dict(ln.split()[::-1] for ln in open(os.path.join(CODE, "SHA256SUMS")) if ln.strip())
    for f in FILES:
        h = sha(os.path.join(CODE, f))
        assert sums[f] == h, (f, sums[f], h)
    ds = dict(ln.split()[::-1] for ln in open(os.path.join(CODE, "DATA_SHA256")) if ln.strip())
    for f in DATA:
        h = sha(os.path.join(FC, "data", f))
        assert ds[f] == h, (f, ds[f], h)
    print("windnoise code + data ok", flush=True)
    fs = dict(ln.split()[::-1] for ln in open(os.path.join(CODE, "FC_SHA256SUMS")) if ln.strip())
    bad = [f for f in FC_MODS if sha(os.path.join(FC, f)) != fs[f]]
    if bad:
        print("WARN ml\\forecast modules changed since deploy:", bad, flush=True)
    return bad


def env_for(extra):
    env = {**os.environ, "PYTHONPATH": FC, "E1_FC": FC, "PYTHONUTF8": "1"}
    for k in ("V3_CFG", "TF32", "PART", "DIFF_VARIANT", "PATH_SCORES", "SEEDS", "NSAMP", "GEN_SEED", "EVAL_OUT", "SEED", "LR"):
        env.pop(k, None)
    env.update(extra)
    return env


def evaluate(meth, extra, ckpt=None):
    t = time.time()
    subprocess.run([sys.executable, "-u", os.path.join(CODE, "evaluate.py"), meth, PROF, NOISE, str(NS)] + ([ckpt] if ckpt else []),
                   cwd=FC, check=True, env=env_for(extra))
    return round(time.time() - t, 1)


def npz(meth):
    return os.path.join(OUT, f"{meth}_{PROF}_{NOISE}.npz")


def record(part, d):
    with open(os.path.join(OUT, f"{part}.json"), "w") as fh:
        json.dump({**d, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=1)


def pf(job=J_PF):
    """clim (seconds), then the PF in 10-seed chunks (resumable; re-queues itself on queue-cpu near the segment budget)."""
    changed = verify()
    os.makedirs(PARTS, exist_ok=True)
    if not os.path.exists(npz("clim")):
        evaluate("clim", {"EVAL_OUT": npz("clim") + ".tmp.npz"}); os.replace(npz("clim") + ".tmp.npz", npz("clim"))
    hist = chunks("pf", None, {}, job, os.path.join(E1, "queue-cpu"), 1500.0, changed)
    if hist is None:
        return
    import numpy as np
    zs = [np.load(os.path.join(PARTS, f"pf_{a:03d}.npz")) for a in range(0, NS, CHUNK)]
    S = np.concatenate([z["seeds"] for z in zs])
    assert (S == np.arange(NS)).all(), S
    np.savez_compressed(npz("pf"), P=np.concatenate([z["P"] for z in zs]), Y=np.concatenate([z["Y"] for z in zs]), seeds=S,
                        paths=np.concatenate([z["paths"] for z in zs]))
    # N 4096, jitter 0.03, reseed 0.01: evaluate.py defaults, as the main table
    record("pf", {"n_seeds": NS, "secs": round(sum(h["secs"] for h in hist), 1), "fc_mods_changed": changed})
    mark()


def chunks(meth, ckpt, extra, job, queue, first_est, changed):
    """Run evaluate.py <meth> on 10-seed chunks -> parts/<meth>_<a>.npz; re-queue `job` on `queue` before the segment
    budget would be exceeded (>= 1 chunk per segment). Returns the chunk history when all chunks exist, else None."""
    hist_f = os.path.join(PARTS, f"{meth}.jsonl")
    hist = [json.loads(ln) for ln in open(hist_f)] if os.path.exists(hist_f) else []
    seg = 1 + len({h["segment_start"] for h in hist}); ran = 0
    for a in range(0, NS, CHUNK):
        dst = os.path.join(PARTS, f"{meth}_{a:03d}.npz")
        if os.path.exists(dst):
            continue
        est = max([h["secs"] for h in hist] or [first_est])
        if ran and (time.time() - T_START + 1.1 * est > BUDGET or (CJOB_STOP and os.path.exists(CJOB_STOP))):
            if CJOB_STOP:  # HTCondor restarts the job in place (checkpoint_exit_code = 85)
                log(event="segment_end", job=job, meth=meth, next_seed=a, exit=85)
                sys.exit(85)
            requeue(job, f"re-queued: from test seed {a}", queue)
            return None
        s = evaluate(meth, {**extra, "SEEDS": f"{a}:{min(a + CHUNK, NS)}", "GEN_SEED": str(a), "EVAL_OUT": dst + ".tmp.npz"}, ckpt)
        os.replace(dst + ".tmp.npz", dst)
        rec = {"chunk": a, "secs": s, "segment": seg, "segment_start": round(T_START), "fc_mods_changed": changed}
        hist.append(rec)
        with open(hist_f, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        log(meth=meth, **rec); ran += 1
    return hist


def gru_eval():
    changed = verify()
    s = evaluate("gru", {"EVAL_OUT": npz("gru"), "DEVICE": "cpu"}, GRU_PT)
    record("gru", {"n_seeds": NS, "ckpt": GRU_PT, "ckpt_sha256": sha(GRU_PT), "secs": s,
                   "train": json.load(open(os.path.join(OUT, "train_gru.json"))), "fc_mods_changed": changed})
    mark()


def requeue(job, note, queue=QUEUE):
    os.makedirs(os.path.join(queue, "pending"), exist_ok=True)
    p = os.path.join(queue, "pending", job + ".cmd")
    assert not os.path.exists(p), f"{p} already queued"
    with open(p + ".tmp", "w", newline="\r\n") as fh:
        fh.write(job_text(job).replace("@echo off\n", f"@echo off\nrem {note}\n", 1))
    os.replace(p + ".tmp", p)  # the runner only picks *.cmd
    log(event="queued", job=job, note=note)


def queue_eval():
    assert os.path.exists(DIFF_PT), DIFF_PT
    if CJOB_STOP:  # under HTCondor the eval is its own job ([after] s1c_wn_a_train)
        log(event="not_queued", job=J_EVAL, note="HTCondor: runs from its [after] dependency")
        return
    requeue(J_EVAL, "queued by s1c_wn_a_train after training")


def diff(job):
    changed = verify()
    os.makedirs(PARTS, exist_ok=True)
    extra = {"DEVICE": os.environ.get("DEVICE", "cuda"), "NSAMP": str(NSAMP), "DDIM": str(DDIM), "PATH_SCORES": "1",
             "EVAL_CHUNK": "15", "V3_CFG": "C", "TF32": "1"}
    hist = chunks("diff", DIFF_PT, extra, job, QUEUE, FIRST_EST, changed)
    if hist is None:
        return
    import numpy as np
    zs = [np.load(os.path.join(PARTS, f"diff_{a:03d}.npz")) for a in range(0, NS, CHUNK)]
    S = np.concatenate([z["seeds"] for z in zs])
    assert (S == np.arange(NS)).all(), S
    assert all(int(z["nsamp"]) == NSAMP and int(z["ddim"]) == DDIM for z in zs)
    cat = {k: np.concatenate([z[k] for z in zs]) for k in ("P", "Y", "es", "vs")}
    np.savez_compressed(npz("diff"), seeds=S, paths=np.zeros(0), paths0=zs[0]["paths0"], nsamp=NSAMP, ddim=DDIM, **cat)
    record("diff", {"n_seeds": NS, "n_samples": NSAMP, "ddim_steps": DDIM, "ckpt": DIFF_PT, "ckpt_sha256": sha(DIFF_PT),
                    "V3_CFG": "C", "TF32": "1", "secs": round(sum(h["secs"] for h in hist), 1),
                    "train": json.load(open(os.path.join(OUT, "train_diff.json"))),
                    "fc_mods_changed": sorted({f for h in hist for f in h.get("fc_mods_changed", [])})})
    mark()


def mark():
    st = {p: "done" if os.path.exists(os.path.join(OUT, f"{p}.json")) else "pending" for p in PARTS_DONE}
    if all(v == "done" for v in st.values()):
        subprocess.run([sys.executable, "-u", os.path.join(CODE, "summarize.py"), PROF, NOISE, "clim", "pf", "gru", "diff"],
                       cwd=FC, check=True, env=env_for({"SUM_DIR": OUT}))
        assert os.path.exists(os.path.join(OUT, f"summary_{PROF}_{NOISE}.json"))
    d = {**st, "n_seeds": NS, "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(OUT, "done.json.tmp"), "w") as fh:
        fh.write(json.dumps(d) + "\n")
    os.replace(os.path.join(OUT, "done.json.tmp"), os.path.join(OUT, "done.json"))
    log(event="mark", **d)


def export():
    dst = os.path.join(FC, "out", "windnoise")
    os.makedirs(dst, exist_ok=True)
    fs = [f"summary_{PROF}_{NOISE}.json", "done.json", "pf.json", "gru.json", "diff.json", "train_gru.json", "train_diff.json"] + \
         [f"{m}_{PROF}_{NOISE}.npz" for m in ("clim", "pf", "gru", "diff")]
    for f in fs:
        subprocess.run(["scp", "-q", f"w:D:/LastEmbers/e1/out/windnoise/{f}", dst], check=True)
        print("copied", f)
    shutil.copy(os.path.join(dst, f"summary_{PROF}_{NOISE}.json"), os.path.join(dst, f"summary_{PROF}_murky.json"))
    print("wrote", os.path.join(dst, f"summary_{PROF}_murky.json"))


REF = os.path.join(FC, "out", "windnoise", "ref60")
REF_NPZ = {"clim": "clim", "pf": "pf", "gru": "gru", "diff": "diff3s1"}  # result key -> S1-5 eval200 npz prefix (v3 seed 1)


def ref(eval200):
    os.makedirs(REF, exist_ok=True)
    for pre in REF_NPZ.values():
        fs = [f for f in os.listdir(eval200) if f.startswith(f"{pre}_{PROF}_murky") and f.endswith(".npz")]
        assert fs, pre
        for f in fs:
            d = os.path.join(REF, f)
            if os.path.lexists(d):
                os.remove(d)
            os.symlink(os.path.join(os.path.abspath(eval200), f), d)
    subprocess.run([sys.executable, "-u", os.path.join(FC, "summarize.py"), PROF, "murky"] +
                   [k if k == v else f"{k}={v}" for k, v in REF_NPZ.items()],
                   cwd=FC, check=True, env={**os.environ, "SUM_DIR": REF, "SUM_SEEDS": f"0:{NS}"})


def tex():
    W = json.load(open(os.path.join(FC, "out", "windnoise", f"summary_{PROF}_{NOISE}.json")))
    R = json.load(open(os.path.join(REF, f"summary_{PROF}_murky.json")))
    D = json.load(open(os.path.join(FC, "out", "windnoise", "done.json")))
    DF = json.load(open(os.path.join(FC, "out", "windnoise", "diff.json")))
    assert all(D[p] == "done" for p in PARTS_DONE) and D["n_seeds"] == NS == DF["n_seeds"]
    assert DF["n_samples"] == NSAMP and DF["ddim_steps"] == DDIM and DF["V3_CFG"] == "C"
    f = lambda x, d=2: (lambda s: "$-$" + s[1:] if s.startswith("-") else s)(f"{x:.{d}f}")
    ci = lambda r, k: f"{f(r[k][0])} [{f(r[k][1])}, {f(r[k][2])}]"
    mac = {}
    for m, nm in (("pf", "Pf"), ("gru", "Gru"), ("diff", "Diff")):
        for L, w in ((72, "SeventyTwo"), (168, "OneSixtyEight")):
            mac[f"wn{nm}Bss{w}"] = f(W[m][f"BSS{L}_clim"][0])
            mac[f"wnRef{nm}Bss{w}"] = f(R[m][f"BSS{L}_clim"][0])
        mac[f"wnRow{nm}"] = " & ".join(ci(S[m], f"BSS{L}_clim") for L in (72, 168) for S in (R, W))
    mac["wnSeeds"] = str(NS); mac["wnPaths"] = str(DF["n_samples"]); mac["wnDdim"] = str(DF["ddim_steps"])
    mac["wnDiffSteps"] = f"{DF['train']['steps']:,}".replace(",", "{,}")
    mac["wnGpuHours"] = f((DF["train"]["wall_secs"] + DF["secs"]) / 3600, 1)  # v3 retraining + 60-seed sampling
    G = json.load(open(os.path.join(FC, "out", "windnoise", "gru.json")))
    P = json.load(open(os.path.join(FC, "out", "windnoise", "pf.json")))
    mac["wnCpuHours"] = f((G["train"]["wall_secs"] + G["secs"] + P["secs"]) / 3600, 1)  # GRU training + eval, clim + PF
    block = ["% BEGIN wn (generated by ml/forecast/windows/windnoise.py tex from ml/forecast/out/windnoise/"
             "{summary_endurance_murky_windnoise,ref60/summary_endurance_murky,done,diff,gru,pf}.json; do not edit)"]
    block += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items()]
    block += ["% END wn"]
    p = os.path.join(E1, "docs", "paper", "main.tex")
    t = open(p).read()
    a, b = t.index("% BEGIN wn "), t.index("% END wn") + len("% END wn")
    open(p, "w").write(t[:a] + "\n".join(block) + t[b:])
    print(f"{p}: {len(mac)} macros")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "jobs":
        jobs(sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "jobs"))
    elif cmd in ("deploy", "verify", "pf", "gru_eval", "queue_eval", "mark", "export", "tex"):
        globals()[cmd]()
    elif cmd == "ref":
        ref(sys.argv[2])
    elif cmd == "diff":
        assert sys.argv[2] == "--job"
        diff(sys.argv[3])
    else:
        sys.exit(__doc__)
