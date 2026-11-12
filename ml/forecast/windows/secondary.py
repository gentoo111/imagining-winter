"""S1-8 secondary settings (RESEARCH-PLAN S1-8, GAPS #9): diffusion v3 (S1-3's cfgC, lr 1e-3, 100k steps, VAL_EVERY 1000,
TF32, best-validation checkpoint; training seed 1 as S1-4's first seed) trained once on each of Endurance / clear and
A New Home / murky, then evaluated on 200 test seeds exactly as S1-5 (200 paths, DDIM 50, TF32, 10-seed chunks,
generator seeded with the chunk's first test seed, EVAL_CHUNK 15, PATH_SCORES=1).
  secondary.py jobs [jobdir]       (main Mac) GPU queue jobs s18_a/b_train_<s>, s18_c/d_eval_<s>, s18_z_done (named to run
                                   after S1-5's s16_zz_e200_*)
  secondary.py deploy              (main Mac) scp train.py / evaluate.py -> windows\\secondary\\ (+ SHA256SUMS), this script
                                   -> windows\\; checks the setting data on Windows == main Mac (sha256)
  secondary.py verify              (Windows) deployed windows\\secondary\\ files == SHA256SUMS
  secondary.py run <setting> --job J  (GPU queue) as eval200.py run: chunks until the segment budget, then re-queues J
  secondary.py done                (GPU queue, last) -> out\\secondary\\done.json {"endurance_clear": "done", ...}
  secondary.py export <dir> [runs] (main Mac) scp merged npz / json back to <dir>, checkpoints + logs to [runs]
  secondary.py tex [sumdir]        (main Mac) tab:secondary rows (between the e18-rows markers) and the e18 macro block of
                                   docs/paper/main.tex from <sumdir>/summary_<setting>.json (default ml/forecast/out/eval200)
Training: train_chain.py with CHAIN_TRAIN_PY = the deployed train.py and CHAIN_ENV listing itself (E1-RESULTS S1-3 note),
out\\secondary\\diff3_<setting>_s1.pt (best-validation checkpoint, .log, .state, .chain.json), done file out\\secondary\\runs\\<s>.json.
HTCondor pool (env CJOB_STOP set by the lane wrapper): at the segment budget, or once the wrapper's STOP file exists, the
segment exits 85 (restarted in place, resuming from the parts on disk) instead of re-queueing; successors come from the
job's [after] jobs (docs/ml/windows/condor/ on e1/s1-5-full-eval-200)."""
import hashlib, json, os, re, shutil, subprocess, sys, time

T_START = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
FC = os.path.dirname(HERE)
E1 = os.environ.get("SEC_E1") or os.path.abspath(os.path.join(FC, "..", ".."))  # SEC_* overrides: dry runs only
CODE = os.environ.get("SEC_CODE", os.path.join(HERE, "secondary"))
OUT = os.environ.get("SEC_OUT", os.path.join(E1, "out", "secondary"))
PARTS = os.path.join(OUT, "parts")
QUEUE = os.environ.get("CHAIN_QUEUE", os.path.join(E1, "queue"))
NS, NSAMP, DDIM, CHUNK = (int(os.environ.get(k, v)) for k, v in
                          (("SEC_NS", "200"), ("SEC_NSAMP", "200"), ("SEC_DDIM", "50"), ("SEC_CHUNK", "10")))
BUDGET = float(os.environ.get("SEC_BUDGET", "10200"))  # s per segment (queue cap 10800 s)
CJOB_STOP = os.environ.get("CJOB_STOP")  # HTCondor lane wrapper's STOP file: segment end = exit 85, no legacy re-queue
SETTINGS = {"endurance_clear": ("endurance", "clear"), "a_new_home_murky": ("a_new_home", "murky")}
SEED, CFG, LR, STEPS, VAL_EVERY = 1, "C", "1e-3", 100000, 1000
TRAIN_JOBS = {"endurance_clear": "s18_a_train_clear", "a_new_home_murky": "s18_b_train_newhome"}
EVAL_JOBS = {"endurance_clear": "s18_c_eval_clear", "a_new_home_murky": "s18_d_eval_newhome"}
DONE_JOB = "s18_z_done"
FILES = ("train.py", "evaluate.py")
FC_MODS = ("common.py", "metrics.py", "models.py", "batch.py", "baselines.py", "windows/train_chain.py")
E1W = r"D:\LastEmbers\e1"
FCW = E1W + r"\ml\forecast"
OUTW = E1W + r"\out\secondary"
PYW = E1W + r"\.venv\Scripts\python.exe"
FIRST_EST = 2600.0  # s per 10-seed chunk before one is measured (S1-5: ~224 s / seed)


def log(**kw):
    print(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}), flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pt_name(s):
    return f"diff3_{s}_s{SEED}"


def ckpt(s):
    return os.path.join(OUT, pt_name(s) + ".pt")


def job_text(name):
    head, pre = ["@echo off"], ["set PYTHONUTF8=1", "set DEVICE=cuda"]
    tail = [f"cd /d {FCW}", f"if not exist {OUTW}\\runs mkdir {OUTW}\\runs", f"if not exist {OUTW}\\parts mkdir {OUTW}\\parts",
            f"{PYW} -u windows\\secondary.py verify", "if errorlevel 1 exit /b 1"]
    if name in TRAIN_JOBS.values():
        s = [k for k, v in TRAIN_JOBS.items() if v == name][0]
        p, n = SETTINGS[s]
        head += [f"rem S1-8: v3 cfg{CFG} lr {LR} seed {SEED} on {p}/{n}, {STEPS} steps chained (train_chain.py re-queues {name}.cmd);",
                 f"rem best-validation checkpoint {OUTW}\\{pt_name(s)}.pt. Code: windows\\secondary\\train.py (SHA256SUMS)."]
        pre += [f"set SEED={SEED}", f"set V3_CFG={CFG}", f"set LR={LR}", "set TF32=1", f"set VAL_EVERY={VAL_EVERY}",
                f"set PYTHONPATH={FCW}", f"set CHAIN_TRAIN_PY={FCW}\\windows\\secondary\\train.py",
                "set CHAIN_ENV=V3_CFG,LR,TF32,SEED,CHAIN_ENV"]
        cmd = (f"{PYW} -u windows\\train_chain.py diff {p} {n} {STEPS} {OUTW}\\{pt_name(s)}.pt --job {name} "
               f"--done {OUTW}\\runs\\{s}.json")
    elif name in EVAL_JOBS.values():
        s = [k for k, v in EVAL_JOBS.items() if v == name][0]
        head += [f"rem S1-8: {s} v3 on test seeds 0-{NS - 1}, {NSAMP} paths, DDIM {DDIM} (as S1-5), {CHUNK}-seed chunks;",
                 f"rem re-queues {name}.cmd until all chunks exist -> {OUTW}\\diff3_{s}.npz"]
        pre += ["set TORCH_THREADS=2"]
        cmd = f"{PYW} -u windows\\secondary.py run {s} --job {name}"
    else:
        head += [f"rem S1-8: after the 2 trainings + 2 evaluations -> {OUTW}\\done.json"]
        cmd = f"{PYW} -u windows\\secondary.py done"
    return "\n".join(head + pre + tail + [cmd]) + "\n"


def jobs(jobdir):
    os.makedirs(jobdir, exist_ok=True)
    for name in list(TRAIN_JOBS.values()) + list(EVAL_JOBS.values()) + [DONE_JOB]:
        with open(os.path.join(jobdir, name + ".cmd"), "w", newline="\r\n") as fh:
            fh.write(job_text(name))
        print("wrote", os.path.join(jobdir, name + ".cmd"))


def deploy():
    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "s18_secondary_deploy")
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    with open(os.path.join(tmp, "SHA256SUMS"), "w") as fh:
        for f in FILES:
            shutil.copy(os.path.join(FC, f), tmp); fh.write(f"{sha(os.path.join(FC, f))}  {f}\n")
    with open(os.path.join(tmp, "FC_SHA256SUMS"), "w") as fh:  # imported modules as deployed (checked, not frozen)
        for f in FC_MODS:
            fh.write(f"{sha(os.path.join(FC, f))}  {f}\n")
    subprocess.run(["ssh", "w", rf"if not exist {FCW}\windows\secondary mkdir {FCW}\windows\secondary & "
                                rf"if not exist {OUTW}\parts mkdir {OUTW}\parts & if not exist {OUTW}\runs mkdir {OUTW}\runs"],
                   check=True)
    subprocess.run(["scp", "-q"] + [os.path.join(tmp, f) for f in FILES + ("SHA256SUMS", "FC_SHA256SUMS")] +
                   ["w:D:/LastEmbers/e1/ml/forecast/windows/secondary/"], check=True)
    subprocess.run(["scp", "-q", os.path.abspath(__file__), "w:D:/LastEmbers/e1/ml/forecast/windows/"], check=True)
    for f in FC_MODS:  # the deployed modules must be the ones this branch was tested with
        o = subprocess.run(["ssh", "w", f"certutil -hashfile {FCW}\\{f.replace('/', chr(92))} SHA256"], check=True,
                           capture_output=True).stdout.decode("ascii", "replace")
        r = [x.strip().replace(" ", "") for x in o.splitlines() if re.fullmatch(r"[0-9a-f]{64}", x.strip().replace(" ", ""))][0]
        assert r == sha(os.path.join(FC, f)), (f, r)
    for s in SETTINGS:
        for split in ("train", "val", "test"):
            f = f"{s}_{split}.bin"
            o = subprocess.run(["ssh", "w", f"certutil -hashfile {FCW}\\data\\{f} SHA256"], check=True,
                               capture_output=True).stdout.decode("ascii", "replace")
            r = [x.strip().replace(" ", "") for x in o.splitlines() if re.fullmatch(r"[0-9a-f]{64}", x.strip().replace(" ", ""))][0]
            assert r == sha(os.path.join(FC, "data", f)), (f, r)
            print(f, r, "== main Mac")
    print(open(os.path.join(tmp, "SHA256SUMS")).read())


def verify():
    sums = dict(ln.split()[::-1] for ln in open(os.path.join(CODE, "SHA256SUMS")) if ln.strip())
    for f in FILES:
        h = sha(os.path.join(CODE, f))
        assert sums[f] == h, (f, sums[f], h)
    print("secondary code ok", flush=True)
    fsums = dict(ln.split()[::-1] for ln in open(os.path.join(CODE, "FC_SHA256SUMS")) if ln.strip())
    bad = [f for f in fsums if sha(os.path.join(FC, *f.split("/"))) != fsums[f]]
    if bad:  # another step re-copied ml\forecast sources: recorded, not fatal (as eval200.py)
        print("WARN ml\\forecast modules changed since deploy:", bad, flush=True)
    return bad


def part_file(s, a):
    return os.path.join(PARTS, f"{s}_{a:03d}.npz")


def requeue(job, s, nxt):
    if CJOB_STOP:  # HTCondor restarts the job in place (checkpoint_exit_code = 85)
        log(event="segment_end", job=job, setting=s, next_seed=nxt, exit=85)
        sys.exit(85)
    os.makedirs(os.path.join(QUEUE, "pending"), exist_ok=True)
    p = os.path.join(QUEUE, "pending", job + ".cmd")
    assert not os.path.exists(p), f"{p} already queued"
    with open(p + ".tmp", "w", newline="\r\n") as fh:
        fh.write(job_text(job).replace("@echo off\n", f"@echo off\nrem re-queued: {s} from test seed {nxt}\n", 1))
    os.replace(p + ".tmp", p)  # the runner only picks *.cmd
    log(event="requeued", job=job, setting=s, next_seed=nxt)


def run(s, job):
    changed = verify()
    os.makedirs(PARTS, exist_ok=True)
    prof, noise = SETTINGS[s]
    ck = ckpt(s)
    assert os.path.exists(ck), f"no checkpoint {ck}"
    hist_f = os.path.join(PARTS, f"{s}.jsonl")
    hist = [json.loads(ln) for ln in open(hist_f)] if os.path.exists(hist_f) else []
    seg = 1 + len({h["segment_start"] for h in hist}); ran = 0
    for a in range(0, NS, CHUNK):
        dst = part_file(s, a)
        if os.path.exists(dst):
            continue
        est = max([h["secs"] for h in hist] or [FIRST_EST])
        if ran and (time.time() - T_START + 1.1 * est > BUDGET or  # >= 1 chunk per segment (a chunk is << 3 h)
                    (CJOB_STOP and os.path.exists(CJOB_STOP))):
            requeue(job, s, a)
            return
        env = {**os.environ, "PYTHONPATH": FC, "DEVICE": os.environ.get("DEVICE", "cuda"), "PYTHONUTF8": "1",
               "NSAMP": str(NSAMP), "DDIM": str(DDIM), "PATH_SCORES": "1", "SEEDS": f"{a}:{min(a + CHUNK, NS)}",
               "GEN_SEED": str(a), "EVAL_CHUNK": os.environ.get("EVAL_CHUNK", "15"), "EVAL_OUT": dst + ".tmp.npz"}
        for k in ("V3_CFG", "TF32", "PART", "DIFF_VARIANT"):
            env.pop(k, None)
        env.update({"V3_CFG": CFG, "TF32": "1"})
        t = time.time()
        subprocess.run([sys.executable, os.path.join(CODE, "evaluate.py"), "diff", prof, noise, str(NS), ck],
                       cwd=FC, check=True, env=env)
        os.replace(dst + ".tmp.npz", dst)
        rec = {"chunk": a, "secs": round(time.time() - t, 1), "segment": seg, "segment_start": round(T_START),
               "fc_mods_changed": changed}
        hist.append(rec)
        with open(hist_f, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        log(setting=s, **rec); ran += 1
    merge(s)


def merge(s):
    import numpy as np
    zs = [np.load(part_file(s, a)) for a in range(0, NS, CHUNK)]
    S = np.concatenate([z["seeds"] for z in zs])
    assert (S == np.arange(NS)).all(), S
    for z in zs:
        assert int(z["nsamp"]) == NSAMP and int(z["ddim"]) == DDIM
    cat = {k: np.concatenate([z[k] for z in zs]) for k in ("P", "Y", "es", "vs")}
    np.savez_compressed(os.path.join(OUT, f"diff3_{s}.npz"), seeds=S, paths=np.zeros(0), paths0=zs[0]["paths0"],
                        nsamp=NSAMP, ddim=DDIM, **cat)
    np.savez_compressed(os.path.join(OUT, f"pvec_diff3_{s}.npz"), seeds=S, pvec=np.concatenate([z["pvec"] for z in zs]))
    hist = [json.loads(ln) for ln in open(os.path.join(PARTS, f"{s}.jsonl"))]
    segs = {}
    for h in hist:
        segs.setdefault(h["segment_start"], 0.0); segs[h["segment_start"]] += h["secs"]
    d = {"setting": s, "name": "diff3", "ckpt": "out\\secondary\\" + pt_name(s) + ".pt", "ckpt_sha256": sha(ckpt(s)),
         "V3_CFG": CFG, "TF32": "1", "n_seeds": NS, "n_samples": NSAMP, "ddim_steps": DDIM,
         "secs": round(sum(h["secs"] for h in hist), 1), "secs_per_seed": round(sum(h["secs"] for h in hist) / NS, 1),
         "segments": [round(v, 1) for v in segs.values()],
         "fc_mods_changed": sorted({f for h in hist for f in h.get("fc_mods_changed", [])}),
         "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(d, open(os.path.join(OUT, f"eval_{s}.json"), "w"), indent=1)
    log(event="merged", **d)


def train_summary(s):
    n = pt_name(s)
    d = json.load(open(os.path.join(OUT, "runs", f"{s}.json")))
    ch = json.load(open(os.path.join(OUT, n + ".pt.chain.json")))
    lg = [json.loads(ln) for ln in open(os.path.join(OUT, n + ".pt.log"))]
    b = min(lg, key=lambda r: r["val"])
    assert d["steps"] == STEPS and lg[-1]["it"] == STEPS, (n, d["steps"], lg[-1]["it"])
    return {"seed": SEED, "cfg": "cfg" + CFG, "lr": float(LR), "steps": STEPS, "best_step": b["it"], "best_val_loss": b["val"],
            "final_val_loss": lg[-1]["val"], "sec_per_step": round(lg[-1]["secs"] / STEPS, 4),
            "segments": ch["segments"], "train_secs": sum(g["sec"] for g in ch["segments"]),
            "curve": [[r["it"], r["val"]] for r in lg]}


def done():
    res = {}
    for s in SETTINGS:
        ev = json.load(open(os.path.join(OUT, f"eval_{s}.json")))
        assert ev["n_seeds"] == NS and ev["n_samples"] == NSAMP and ev["ddim_steps"] == DDIM, ev
        assert ev["ckpt_sha256"] == sha(ckpt(s)), s
        res[s] = {"train": train_summary(s), "eval": ev}
    d = {s: "done" for s in SETTINGS}
    d.update({"n_seeds": NS, "n_samples": NSAMP, "ddim_steps": DDIM, "settings": res,
              "gpu_secs": round(sum(r["train"]["train_secs"] + r["eval"]["secs"] for r in res.values()), 1),
              "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    with open(os.path.join(OUT, "done.json"), "w") as fh:
        fh.write(json.dumps(d) + "\n")
    print(json.dumps({s: (r["train"]["best_val_loss"], r["train"]["best_step"], r["eval"]["secs"]) for s, r in res.items()}),
          flush=True)


def export(dst, runs=None):
    os.makedirs(dst, exist_ok=True)
    for f in [f"diff3_{s}.npz" for s in SETTINGS] + [f"eval_{s}.json" for s in SETTINGS] + ["done.json"]:
        subprocess.run(["scp", "-q", f"w:D:/LastEmbers/e1/out/secondary/{f}", dst], check=True)
        print("copied", f)
    if runs:
        for s in SETTINGS:
            for ext in (".pt", ".pt.log"):
                subprocess.run(["scp", "-q", f"w:D:/LastEmbers/e1/out/secondary/{pt_name(s)}{ext}", runs], check=True)
                print("copied", pt_name(s) + ext)


def tex(sumdir=None):
    """Literal tab:secondary rows (the claims check reads numbers in the rows) and the text macros, both from the summary
    json files (summarize.py ... diff=diff3), so no number is hand-copied."""
    sumdir = sumdir or os.path.join(FC, "out", "eval200")
    label = {"endurance_clear": ("Endurance / clear", "secClear"), "a_new_home_murky": ("A New Home / murky", "secHome")}
    meth = {"clock": "Clock", "gru": "Gru", "diff": "Diff", "pf": "Pf"}
    lead = {72: "SeventyTwo", 168: "OneSixtyEight"}
    f = lambda x: f"{x:.2f}"
    rows, mac = [], {}
    for s, (name, pre) in label.items():
        R = json.load(open(os.path.join(sumdir, f"summary_{s}.json")))
        cells = [" / ".join(f(R[m][f"BSS{L}_clim"][0]) for L in lead) for m in meth]
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
        for m, M in meth.items():
            for L, W in lead.items():
                mac[f"{pre}{M}{W}"] = f(R[m][f"BSS{L}_clim"][0])
            mac[f"{pre}{M}Crps"] = f(R[m]["CRPS"][0])
        for m in ("diff", "pf"):
            mac[f"{pre}{meth[m]}Es"] = f(R[m]["ES"][0])
    p = os.path.join(E1, "docs", "paper", "main.tex")
    t = open(p).read()
    src = "generated by ml/forecast/windows/secondary.py tex from ml/forecast/out/eval200/summary_{endurance_clear,a_new_home_murky}.json; do not edit"
    for tag, body in (("e18", [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items()]), ("e18-rows", rows)):
        a = t.index(f"% BEGIN {tag} ")
        b = t.index(f"% END {tag}\n")
        t = t[:a] + f"% BEGIN {tag} ({src})" + "\n" + "\n".join(body) + "\n" + t[b:]
    open(p, "w").write(t)
    print(f"{p}: {len(mac)} macros, {len(rows)} rows")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
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
    elif cmd == "export":
        export(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    elif cmd == "tex":
        tex(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        sys.exit(__doc__)
