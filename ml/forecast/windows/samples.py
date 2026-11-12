"""S1-9 sample-count study (RESEARCH-PLAN S1-9 / GAPS #10): the selected v3 (S1-4 seed 1, out\\v3\\diff3_endurance_murky_s1.pt,
V3_CFG=C, TF32 as trained) on test seeds 0-59 (Endurance/murky, 113 daily 07:00 issues), DDIM 50 (as S1-5), 500 paths per
forecast. The 50- and 200-sample readouts are the first 50 / 200 of the same 500 paths (evaluate.py P_SUBSETS=50,200): the
paths of one forecast are i.i.d. given its conditioning (independent noise per path), so the first k are a valid k-sample
draw; the three readouts are nested, which removes between-draw noise from the comparison across sample counts.
  samples.py jobs [jobdir]   (main Mac) GPU queue job s19_a_samples.cmd (named to run after S1-8's s18_*)
  samples.py deploy          (main Mac) scp evaluate.py -> windows\\samples\\ (+ SHA256SUMS), this script -> windows\\;
                             checks the checkpoint's sha256 on Windows against the main-Mac copy
  samples.py verify          (Windows) deployed windows\\samples\\ files == SHA256SUMS
  samples.py run --job J     (GPU queue) chunks of 5 test seeds (parts\\<a>.npz) until the segment budget, then re-queues
                             queue\\pending\\J.cmd; when all 12 chunks exist -> diff3s1_n500_endurance_murky.npz + done.json
  samples.py export          (main Mac) scp the merged npz / done.json to ml/forecast/out/samples/, split it into
                             diffn{50,200,500}_endurance_murky.npz and write summary_n{50,200,500}.json with summarize.py
                             (method key diff, against climatology on the same 60 seeds).
  samples.py tables          (main Mac) out/samples/sample_count.json (granularity vs model-error split, see tables())
                             and the generated macro block "% BEGIN samples" in docs/paper/main.tex.
HTCondor pool (env CJOB_STOP set by the lane wrapper): at the segment budget, or once the wrapper's STOP file exists, the
segment exits 85 (restarted in place, resuming from the parts on disk) instead of re-queueing; successors come from the
job's [after] jobs (docs/ml/windows/condor/ on e1/s1-5-full-eval-200)."""
import hashlib, json, os, re, shutil, subprocess, sys, time

T_START = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
FC = os.path.dirname(HERE)
E1 = os.environ.get("SMP_E1") or os.path.abspath(os.path.join(FC, "..", ".."))  # SMP_* overrides: dry runs only
CODE = os.path.join(HERE, "samples")
OUT = os.environ.get("SMP_OUT", os.path.join(E1, "out", "samples"))
PARTS = os.path.join(OUT, "parts")
QUEUE = os.environ.get("CHAIN_QUEUE", os.path.join(E1, "queue"))
NS, NSAMP, DDIM, CHUNK = (int(os.environ.get(k, v)) for k, v in
                          (("SMP_NS", "60"), ("SMP_NSAMP", "500"), ("SMP_DDIM", "50"), ("SMP_CHUNK", "5")))
SUBS = tuple(int(k) for k in os.environ.get("SMP_SUBS", "50,200").split(","))
PROF, NOISE = "endurance", "murky"
BUDGET = float(os.environ.get("SMP_BUDGET", "10200"))  # s per segment (queue cap 10800 s)
CJOB_STOP = os.environ.get("CJOB_STOP")  # HTCondor lane wrapper's STOP file: segment end = exit 85, no legacy re-queue
GEN0 = 9000  # noise generator seed = GEN0 + chunk's first test seed (S1-5 uses the bare chunk start)
CKPT = tuple(os.environ.get("SMP_CKPT", "out/v3/diff3_endurance_murky_s1.pt").split("/"))
EXTRA = {"V3_CFG": "C", "TF32": "1"}
EVAL_CHUNK = "6"  # issue hours per sampling call: 6 x 500 = 3000 paths, as S1-5's 15 x 200
JOB = "s19_a_samples"
NAME = f"diff3s1_n{NSAMP}_{PROF}_{NOISE}.npz"
FIRST_EST = 3300.0  # s per 5-seed chunk before one is measured (S1-5 estimate 2600 s / 10 seeds x 200 paths, x 2.5 / 2)
E1W = r"D:\LastEmbers\e1"
FCW = E1W + r"\ml\forecast"
PYW = E1W + r"\.venv\Scripts\python.exe"
FC_MODS = ("common.py", "metrics.py", "models.py", "batch.py")


def log(**kw):
    print(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}), flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def job_text():
    lines = ["@echo off",
             f"rem S1-9: v3 seed 1 on test seeds 0-{NS - 1}, {NSAMP} paths (50/200 = first-k subsets), DDIM {DDIM},",
             f"rem {CHUNK}-seed chunks until the segment budget; re-queues {JOB}.cmd until all chunks exist",
             f"rem -> out\\samples\\{NAME} + done.json. Named to run after S1-8's s18_*. Code: windows\\samples\\ (SHA256SUMS).",
             "set PYTHONUTF8=1", "set DEVICE=cuda", "set TORCH_THREADS=2",
             f"cd /d {FCW}", f"if not exist {E1W}\\out\\samples\\parts mkdir {E1W}\\out\\samples\\parts",
             f"{PYW} -u windows\\samples.py verify", "if errorlevel 1 exit /b 1",
             f"{PYW} -u windows\\samples.py run --job {JOB}"]
    return "\n".join(lines) + "\n"


def jobs(jobdir):
    os.makedirs(jobdir, exist_ok=True)
    with open(os.path.join(jobdir, JOB + ".cmd"), "w", newline="\r\n") as fh:
        fh.write(job_text())
    print("wrote", os.path.join(jobdir, JOB + ".cmd"))


def deploy():
    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "s19_samples_deploy")
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    shutil.copy(os.path.join(FC, "evaluate.py"), tmp)
    with open(os.path.join(tmp, "SHA256SUMS"), "w") as fh:
        fh.write(f"{sha(os.path.join(FC, 'evaluate.py'))}  evaluate.py\n")
    with open(os.path.join(tmp, "FC_SHA256SUMS"), "w") as fh:  # imported modules as deployed (checked, not frozen)
        for f in FC_MODS:
            fh.write(f"{sha(os.path.join(FC, f))}  {f}\n")
    subprocess.run(["ssh", "w", rf"if not exist {FCW}\windows\samples mkdir {FCW}\windows\samples & "
                                rf"if not exist {E1W}\out\samples\parts mkdir {E1W}\out\samples\parts"], check=True)
    subprocess.run(["scp", "-q"] + [os.path.join(tmp, f) for f in ("evaluate.py", "SHA256SUMS", "FC_SHA256SUMS")] +
                   ["w:D:/LastEmbers/e1/ml/forecast/windows/samples/"], check=True)
    subprocess.run(["scp", "-q", os.path.abspath(__file__), "w:D:/LastEmbers/e1/ml/forecast/windows/"], check=True)
    o = subprocess.run(["ssh", "w", "certutil -hashfile " + E1W + "\\" + "\\".join(CKPT) + " SHA256"],
                       check=True, capture_output=True).stdout.decode("ascii", "replace")
    r = [ln.strip().replace(" ", "") for ln in o.splitlines() if re.fullmatch(r"[0-9a-f]{64}", ln.strip().replace(" ", ""))][0]
    loc = os.path.join(FC, "runs", CKPT[-1])
    assert sha(loc) == r, (sha(loc), r)
    print("checkpoint", "\\".join(CKPT), r, "== main Mac")
    print(open(os.path.join(tmp, "SHA256SUMS")).read())


def verify():
    sums = dict(ln.split()[::-1] for ln in open(os.path.join(CODE, "SHA256SUMS")) if ln.strip())
    h = sha(os.path.join(CODE, "evaluate.py"))
    assert sums["evaluate.py"] == h, (sums["evaluate.py"], h)
    print("samples code ok", flush=True)
    fs = dict(ln.split()[::-1] for ln in open(os.path.join(CODE, "FC_SHA256SUMS")) if ln.strip())
    bad = [f for f in FC_MODS if sha(os.path.join(FC, f)) != fs[f]]
    if bad:
        print("WARN ml\\forecast modules changed since deploy:", bad, flush=True)
    return bad


def part_file(a):
    return os.path.join(PARTS, f"{a:03d}.npz")


def requeue(job, nxt):
    if CJOB_STOP:  # HTCondor restarts the job in place (checkpoint_exit_code = 85)
        log(event="segment_end", job=job, next_seed=nxt, exit=85)
        sys.exit(85)
    os.makedirs(os.path.join(QUEUE, "pending"), exist_ok=True)
    p = os.path.join(QUEUE, "pending", job + ".cmd")
    assert not os.path.exists(p), f"{p} already queued"
    with open(p + ".tmp", "w", newline="\r\n") as fh:
        fh.write(job_text().replace("@echo off\n", f"@echo off\nrem re-queued: from test seed {nxt}\n", 1))
    os.replace(p + ".tmp", p)  # the runner only picks *.cmd
    log(event="requeued", job=job, next_seed=nxt)


def run(job):
    changed = verify()
    os.makedirs(PARTS, exist_ok=True)
    hist_f = os.path.join(PARTS, "hist.jsonl")
    hist = [json.loads(ln) for ln in open(hist_f)] if os.path.exists(hist_f) else []
    seg = 1 + len({h["segment_start"] for h in hist}); ran = 0
    for a in range(0, NS, CHUNK):
        dst = part_file(a)
        if os.path.exists(dst):
            continue
        est = max([h["secs"] for h in hist] or [FIRST_EST])
        if ran and (time.time() - T_START + 1.1 * est > BUDGET or  # >= 1 chunk per segment (a chunk is << 3 h)
                    (CJOB_STOP and os.path.exists(CJOB_STOP))):
            requeue(job, a)
            return
        env = {**os.environ, "PYTHONPATH": FC, "DEVICE": os.environ.get("DEVICE", "cuda"), "PYTHONUTF8": "1",
               "NSAMP": str(NSAMP), "DDIM": str(DDIM), "P_SUBSETS": ",".join(map(str, SUBS)), "NO_PATHS": "1",
               "SEEDS": f"{a}:{min(a + CHUNK, NS)}", "GEN_SEED": str(GEN0 + a), "EVAL_CHUNK": EVAL_CHUNK,
               "EVAL_OUT": dst + ".tmp.npz"}
        for k in ("V3_CFG", "TF32", "PART", "DIFF_VARIANT", "PATH_SCORES"):
            env.pop(k, None)
        env.update(EXTRA)
        t = time.time()
        subprocess.run([sys.executable, os.path.join(CODE, "evaluate.py"), "diff", PROF, NOISE, str(NS), os.path.join(E1, *CKPT)],
                       cwd=FC, check=True, env=env)
        os.replace(dst + ".tmp.npz", dst)
        rec = {"chunk": a, "secs": round(time.time() - t, 1), "segment": seg, "segment_start": round(T_START),
               "fc_mods_changed": changed}
        hist.append(rec)
        with open(hist_f, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        log(**rec); ran += 1
    merge()


def merge():
    import numpy as np
    zs = [np.load(part_file(a)) for a in range(0, NS, CHUNK)]
    S = np.concatenate([z["seeds"] for z in zs])
    assert (S == np.arange(NS)).all(), S
    for z in zs:
        assert int(z["nsamp"]) == NSAMP and int(z["ddim"]) == DDIM
    keys = ["P", "Y"] + [f"P_n{k}" for k in SUBS]
    cat = {k: np.concatenate([z[k] for z in zs]) for k in keys}
    np.savez_compressed(os.path.join(OUT, NAME), seeds=S, paths=np.zeros(0), nsamp=NSAMP, ddim=DDIM, **cat)
    hist = [json.loads(ln) for ln in open(os.path.join(PARTS, "hist.jsonl"))]
    segs = {}
    for h in hist:
        segs.setdefault(h["segment_start"], 0.0); segs[h["segment_start"]] += h["secs"]
    d = {"model": "v3s1", "ckpt": "\\".join(CKPT), "ckpt_sha256": sha(os.path.join(E1, *CKPT)), **EXTRA,
         "n_seeds": NS, "n_samples_done": sorted(SUBS + (NSAMP,)), "nested": True, "ddim_steps": DDIM,
         "gen_seed": f"{GEN0} + chunk start", "secs": round(sum(h["secs"] for h in hist), 1),
         "secs_per_seed": round(sum(h["secs"] for h in hist) / NS, 1), "segments": [round(v, 1) for v in segs.values()],
         "fc_mods_changed": sorted({f for h in hist for f in h.get("fc_mods_changed", [])}),
         "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(OUT, "done.json"), "w") as fh:
        fh.write(json.dumps(d) + "\n")
    log(event="merged", **d)


def export(src=None):
    import numpy as np
    dst = os.environ.get("SMP_EXPORT") or os.path.join(FC, "out", "samples")
    os.makedirs(dst, exist_ok=True)
    for f in (NAME, "done.json"):
        if src:  # dry run: local files
            shutil.copy(os.path.join(src, f), dst)
        else:
            subprocess.run(["scp", "-q", f"w:D:/LastEmbers/e1/out/samples/{f}", dst], check=True)
        print("copied", f)
    z = np.load(os.path.join(dst, NAME))
    assert int(z["nsamp"]) == NSAMP and int(z["ddim"]) == DDIM and (z["seeds"] == np.arange(NS)).all()
    for k in SUBS + (NSAMP,):
        P = z["P"] if k == NSAMP else z[f"P_n{k}"]
        np.savez_compressed(os.path.join(dst, f"diffn{k}_{PROF}_{NOISE}.npz"), P=P, Y=z["Y"], seeds=z["seeds"], paths=np.zeros(0))
    clim = os.path.join(dst, f"clim_{PROF}_{NOISE}.npz")
    assert os.path.exists(clim), f"{clim} missing (symlink the frozen clim npz there: BSS is against climatology)"
    py = os.environ.get("PYTHON", sys.executable)
    for k in SUBS + (NSAMP,):
        env = {**os.environ, "SUM_DIR": dst, "SUM_TAG": f"_n{k}"}
        subprocess.run([py, "summarize.py", PROF, NOISE, "clim", f"diff=diffn{k}"], cwd=FC, check=True, env=env)
        shutil.copy(os.path.join(dst, f"summary_{PROF}_{NOISE}_n{k}.json"), os.path.join(dst, f"summary_n{k}.json"))
        print("wrote", os.path.join(dst, f"summary_n{k}.json"))

def tables(tex=None):
    """(main Mac, after export) sample_count.json + the generated macro block in docs/paper/main.tex.
    Separates the readout's probability granularity (1/k) from model error: (a) the share of forecasts whose observed class
    got none of the k paths (floored at 1e-4) and its log-loss contribution; (b) the log loss expected from sampling alone,
    E[-log max(c/k, 1e-4)] with c ~ Binomial(k, q500) (q500 = the 500-path probability of the observed class); (c) the
    Brier score at 72 h minus its finite-sample variance term p(1-p)/(k-1) (unbiased for the k -> infinity score)."""
    import numpy as np
    from math import lgamma
    sys.path.insert(0, FC)
    from common import NB
    from metrics import logloss, brier, cum
    src = os.environ.get("SMP_EXPORT") or os.path.join(FC, "out", "samples")
    tex = tex or os.path.join(E1, "docs", "paper", "main.tex")
    K = SUBS + (NSAMP,)
    z = {k: np.load(os.path.join(src, f"diffn{k}_{PROF}_{NOISE}.npz")) for k in K}
    S = z[NSAMP]["seeds"]
    for k in K:
        assert (z[k]["seeds"] == S).all() and (z[k]["Y"] == z[NSAMP]["Y"]).all()
    c = np.load(os.path.join(src, f"clim_{PROF}_{NOISE}.npz"))
    ci = np.array([np.where(c["seeds"] == s)[0][0] for s in S])
    Y = z[NSAMP]["Y"].reshape(-1); ns = len(S); ar = np.arange(len(Y))
    bs_ref = brier(c["P"][ci].reshape(-1, NB + 1), Y, 72).mean()
    q500 = z[NSAMP]["P"].reshape(-1, NB + 1)[ar, Y].astype(np.float64)
    floor = -np.log(1e-4)
    rng = np.random.default_rng(0); boot = rng.integers(0, ns, (1000, ns))
    per_seed_ll = {}
    R = {"n_seeds": int(ns), "n_forecasts": int(len(Y)), "ddim_steps": DDIM, "nested": True, "k": {}}
    for k in K:
        P = z[k]["P"].reshape(-1, NB + 1); q = P[ar, Y]
        ll = logloss(P, Y); hit = q == 0
        p72 = cum(P, 72); bs = brier(P, Y, 72); bs_db = bs - p72 * (1 - p72) / (k - 1)
        n = np.arange(k + 1)
        lc = np.array([lgamma(k + 1) - lgamma(i + 1) - lgamma(k - i + 1) for i in n])
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = (lc[None] + np.where(n[None] > 0, n[None] * np.log(q500[:, None]), 0.0)
                  + np.where((k - n)[None] > 0, (k - n)[None] * np.log1p(-q500[:, None]), 0.0))
        pm = np.nan_to_num(np.exp(lp))
        ll_pred = (pm * -np.log(np.maximum(n / k, 1e-4))[None]).sum(1)
        per_seed_ll[k] = ll.reshape(ns, -1).mean(1)
        R["k"][str(k)] = {"LL": float(ll.mean()), "floor_hit_frac": float(hit.mean()), "floor_hits": int(hit.sum()),
                          "LL_floor_part": float((hit * floor).mean()), "LL_rest": float((ll * ~hit).mean()),
                          "LL_pred_binomial_from_500": float(ll_pred.mean()),
                          "floor_hit_frac_pred_binomial_from_500": float(pm[:, 0].mean()),
                          "BSS72_clim": float(1 - bs.mean() / bs_ref),
                          "BSS72_clim_debiased": float(1 - bs_db.mean() / bs_ref)}
    for k in SUBS:
        dd = per_seed_ll[k] - per_seed_ll[NSAMP]; b = dd[boot].mean(1)
        R["k"][str(k)]["dLL_vs_500"] = [float(dd.mean()), *map(float, np.percentile(b, [2.5, 97.5]))]
    with open(os.path.join(src, "sample_count.json"), "w") as fh:
        json.dump(R, fh, indent=1)
    sm = {k: json.load(open(os.path.join(src, f"summary_n{k}.json")))["diff"] for k in K}
    word = {50: "Fifty", 200: "TwoHundred", 500: "FiveHundred"}
    f3 = lambda v: f"{v:.3f}"
    m = []
    for k in K:
        w, r = word[k], R["k"][str(k)]
        ll, bss = sm[k]["LL"], sm[k]["BSS72_clim"]
        m += [(f"llN{w}", f3(ll[0])), (f"llN{w}CI", f"[{ll[1]:.2f}, {ll[2]:.2f}]"),
              (f"bssN{w}", f3(bss[0])), (f"bssN{w}CI", f"[{bss[1]:.2f}, {bss[2]:.2f}]"),
              (f"bssDbN{w}", f3(r["BSS72_clim_debiased"])),
              (f"floorPctN{w}", f"{100 * r['floor_hit_frac']:.2f}"), (f"llFloorN{w}", f3(r["LL_floor_part"])),
              (f"llPredN{w}", f3(r["LL_pred_binomial_from_500"]))]
        if k in SUBS:
            d = r["dLL_vs_500"]
            m += [(f"dllN{w}", f3(d[0])), (f"dllN{w}CI", f"[{d[1]:.3f}, {d[2]:.3f}]")]
    m += [("floorHitsNFiveHundred", str(R["k"][str(NSAMP)]["floor_hits"])), ("nForecastsSamples", f"{len(Y):,}".replace(",", "{,}")),
          ("nSeedsSamples", str(ns))]
    block = ["% BEGIN samples (generated by ml/forecast/windows/samples.py tables from ml/forecast/out/samples/*.json; do not edit)"]
    block += [f"\\newcommand{{\\{a}}}{{{b}}}" for a, b in m] + ["% END samples"]
    t = open(tex, encoding="utf-8").read()
    if "% BEGIN samples" in t:
        t = re.sub(r"% BEGIN samples.*?% END samples", lambda _: "\n".join(block), t, flags=re.S)
    else:
        t = t.replace("% S1-5 200-seed evaluation", "% S1-9 sample-count study (tab:samples)\n" + "\n".join(block)
                      + "\n% S1-5 200-seed evaluation", 1)
    open(tex, "w", encoding="utf-8").write(t)
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "jobs":
        jobs(sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "jobs"))
    elif cmd == "deploy":
        deploy()
    elif cmd == "verify":
        verify()
    elif cmd == "run":
        assert sys.argv[2] == "--job"
        run(sys.argv[3])
    elif cmd == "export":
        export(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "tables":
        tables()
    else:
        sys.exit(__doc__)
