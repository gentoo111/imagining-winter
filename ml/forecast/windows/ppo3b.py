"""S1-11b: 3a' (fcg, GRU daily table) and 3b (fcd, v3 daily table) x training seeds 1-3, 200M PPO steps each, on the
Windows CPU lane (D:\\LastEmbers\\e1\\queue-cpu), chained in <= 3 h segments.
  ppo3b.py jobs <jobdir>        (main Mac) write p1_ppo3b.cmd (one segment job; it re-queues itself)
  ppo3b.py segment              (CPU lane) one segment: every ready run concurrently, then storm evaluations, re-queue
  ppo3b.py status               print per-run progress
A run is ready when its table exists (fcg: features_gru_daily.npz, fcd: features_diff_v3.npz in out\\fc3b) and it has
not reached 200M. Each run = `fc_ppo.py <seed> learn ...` with exactly e0_queue.train_cmd("fc", seed)'s hyperparameters
(only the Windows START path, port 23000 + 100 i, --out and --no-mlflow differ), table mode (FC_TABLE), chained by
FC_STATE (policy + Adam + RNG + counters, fc_ppo.py) with FC_BUDGET 10200 s per segment, and FC_TARGET_STEPS = 42,082
(3a's mean steps per PPO round: 600,045,256 steps / 14,259 rounds of e0-fc-s{1,2,3}/rounds.jsonl) so every rollout has
3a's size although Windows collects ~3-4x slower. Threads per run: <= 3 concurrent RAYON 3 / torch 3, else RAYON 2 /
torch 1 (s1-11a ppo_bench). A trained run is evaluated in the next segment, alongside the training runs:
eval_e0.py greedy + sample on the 1000 storm-evaluation seeds 300000:301000 (table mode, 2 threads), then
e0_queue.py --report -> <out>\\e0-<rung>-s<seed>.storm.json (with steps). When all 6 are trained and evaluated:
<out>\\done.json {fcd_done, fcg_done, runs:[{rung, seed, steps, t_final, segments:[{sec, steps_from, steps_to}],
storm_json}]}. Out: D:\\LastEmbers\\e1\\out\\ppo3b (run dirs e0-<rung>-s<seed>, .state, .log, .segments.json)."""
import json, os, subprocess, sys, time

T_START = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
FC = os.path.dirname(HERE)
E1 = os.path.abspath(os.path.join(FC, "..", ".."))
sys.path.insert(0, FC)
OUT = os.path.join(E1, "out", "ppo3b")
os.environ.setdefault("E0_RUNS", OUT)
import e0_queue  # noqa: E402

FC3B = os.path.join(E1, "out", "fc3b")
TABLES = {"fcg": os.path.join(FC3B, "features_gru_daily.npz"), "fcd": os.path.join(FC3B, "features_diff_v3.npz")}
RUNS = [("fcg", 1), ("fcg", 2), ("fcg", 3), ("fcd", 1), ("fcd", 2), ("fcd", 3)]
STEPS = e0_queue.STEPS
TARGET_STEPS = 42082
BUDGET = 10200
FROZEN_ROOT = os.path.join(E1, "ppo", "frozen")
SITE = os.path.join(E1, "sim", "site")
PYDEPS = os.path.join(E1, "ppo", "pydeps")
START = os.path.join(E1, "ppo", "start", "model_1755066122.pt")
QUEUE = os.path.join(E1, "queue-cpu")
JOB = "p1_ppo3b"
E1W = r"D:\LastEmbers\e1"


def log(**kw):
    print(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}), flush=True)


def name(r, s):
    return f"e0-{r}-s{s}"


def state_t(r, s):
    p = os.path.join(OUT, name(r, s) + ".state")
    if not os.path.exists(p):
        return 0
    import torch
    return int(torch.load(p, weights_only=False)["t"])


def trained(r, s):
    return e0_queue.final_model(os.path.join(OUT, name(r, s))) is not None


def evaluated(r, s):
    return os.path.exists(os.path.join(OUT, name(r, s) + ".storm.json"))


def base_env():
    return {**os.environ, "PYTHONUTF8": "1", "CUDA_VISIBLE_DEVICES": "-1", "DEVICE": "cpu",
            "PYTHONPATH": os.pathsep.join([SITE, PYDEPS, FC]), "FC_FROZEN_ROOT": FROZEN_ROOT, "FC_FROZEN_SITE": SITE,
            "FC_SIM_DATA": os.path.join(FROZEN_ROOT, "data"), "FC3B_OUT": FC3B, "E0_RUNS": OUT}


def learn_cmd(r, s, i):
    cmd = e0_queue.train_cmd("fc", s, i)
    assert cmd[:3] == ["nice", "-n", "10"] and cmd[4] == e0_queue.FC_PPO and cmd[5] == str(s), cmd[:6]
    a = cmd[6:]
    a[a.index("--resume") + 1] = START
    a[a.index("--out") + 1] = os.path.join(OUT, name(r, s))
    assert a[a.index("--workers") + 1] == f"local:{23000 + 100 * i}:8:rust"
    return [sys.executable, "-u", os.path.join(FC, "fc_ppo.py"), str(s)] + a + ["--no-mlflow"]


def eval_cmds(r, s):
    out = []
    for c in e0_queue.eval_cmds(r, s):
        c = c[4:]  # drop nice -n 10 <mac python>; keep eval_e0.py and its arguments
        assert c[0] == os.path.join(FC, "eval_e0.py"), c[0]
        c[c.index("--threads") + 1] = "2"
        out.append([sys.executable, "-u"] + c + ["--fc-tables", f"{r}={TABLES[r]}"])
    return out


def segment():
    os.makedirs(OUT, exist_ok=True)
    ready = [(i, r, s) for i, (r, s) in enumerate(RUNS) if os.path.exists(TABLES[r]) and not trained(r, s)]
    to_eval = [(r, s) for r, s in RUNS if trained(r, s) and not evaluated(r, s)]
    k = len(ready) + len(to_eval)
    rayon, torch_t = (3, 3) if k <= 3 else (2, 1)
    procs = []
    for i, r, s in ready:
        env = {**base_env(), "FC_TABLE": TABLES[r], "FC_STATE": os.path.join(OUT, name(r, s) + ".state"),
               "FC_BUDGET": str(BUDGET), "FC_TARGET_STEPS": str(TARGET_STEPS), "RAYON_NUM_THREADS": str(rayon),
               "FC_TORCH_THREADS": str(torch_t), "OMP_NUM_THREADS": str(torch_t), "MKL_NUM_THREADS": str(torch_t)}
        lf = open(os.path.join(OUT, name(r, s) + ".log"), "a")
        t0 = state_t(r, s)
        p = subprocess.Popen(learn_cmd(r, s, i), cwd=os.path.join(FROZEN_ROOT, "ml"), env=env, stdout=lf,
                             stderr=subprocess.STDOUT)
        procs.append({"r": r, "s": s, "p": p, "lf": lf, "from": t0, "t": time.time(), "kind": "train"})
        log(start=name(r, s), steps_from=t0, pid=p.pid, rayon=rayon, torch=torch_t)
    for r, s in to_eval:  # one process per run: greedy, sample, then the storm report
        env = {**base_env(), "OMP_NUM_THREADS": "1", "RAYON_NUM_THREADS": "2"}
        sh = " && ".join(subprocess.list2cmdline(c) for c in eval_cmds(r, s) + [
            [sys.executable, "-u", os.path.join(FC, "e0_queue.py"), "--report", r, str(s)]])
        lf = open(os.path.join(OUT, name(r, s) + ".eval.log"), "a")
        p = subprocess.Popen(sh, shell=True, cwd=FC, env=env, stdout=lf, stderr=subprocess.STDOUT)
        procs.append({"r": r, "s": s, "p": p, "lf": lf, "t": time.time(), "kind": "eval"})
        log(start_eval=name(r, s), pid=p.pid)
    if not procs:  # nothing ready yet (3b waits for its table): idle a while instead of spinning the lane
        log(idle="no run ready", fcd_table=os.path.exists(TABLES["fcd"]))
        time.sleep(1800)
    bad = []
    for q in procs:
        rc = q["p"].wait(); q["lf"].close()
        sec = round(time.time() - q["t"])
        if q["kind"] == "train":
            segf = os.path.join(OUT, name(q["r"], q["s"]) + ".segments.json")
            segs = json.load(open(segf)) if os.path.exists(segf) else []
            to = state_t(q["r"], q["s"])
            segs.append({"sec": sec, "steps_from": q["from"], "steps_to": to, "rc": rc})
            json.dump(segs, open(segf, "w"), indent=1)
            log(end=name(q["r"], q["s"]), rc=rc, sec=sec, steps_to=to, trained=trained(q["r"], q["s"]))
            if rc or to <= q["from"]:
                bad.append(name(q["r"], q["s"]))
        else:
            log(end_eval=name(q["r"], q["s"]), rc=rc, sec=sec, storm=evaluated(q["r"], q["s"]))
            if rc:
                bad.append(name(q["r"], q["s"]) + " eval")
    assert not bad, f"failed: {bad}"
    if all(trained(r, s) and evaluated(r, s) for r, s in RUNS):
        write_done()
    else:
        requeue()


def write_done():
    runs = []
    for r, s in RUNS:
        fm = e0_queue.final_model(os.path.join(OUT, name(r, s)))
        segs = json.load(open(os.path.join(OUT, name(r, s) + ".segments.json")))
        runs.append({"rung": r, "seed": s, "steps": STEPS, "t_final": int(os.path.basename(fm)[6:15]),
                     "segments": [{k: g[k] for k in ("sec", "steps_from", "steps_to")} for g in segs],
                     "storm_json": os.path.join(OUT, name(r, s) + ".storm.json")})
    done = {"fcd_done": sum(r["rung"] == "fcd" for r in runs), "fcg_done": sum(r["rung"] == "fcg" for r in runs),
            "runs": runs, "target_steps_per_round": TARGET_STEPS, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(done, open(os.path.join(OUT, "done.json"), "w"))  # one line: the pipeline check reads it with a regex
    log(done={k: done[k] for k in ("fcd_done", "fcg_done")})


def requeue():
    lines = ["@echo off", f"rem ppo3b.py segment: next segment of {JOB} (S1-11b 3b / 3a' PPO, CPU lane)",
             "set PYTHONUTF8=1", "set CUDA_VISIBLE_DEVICES=-1", "set DEVICE=cpu", f"cd /d {FC}",
             subprocess.list2cmdline([sys.executable, "-u", os.path.abspath(__file__), "segment"])]
    p = os.path.join(QUEUE, "pending", JOB + ".cmd")
    assert not os.path.exists(p), f"{p} already queued"
    with open(p + ".tmp", "w", newline="\r\n") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(p + ".tmp", p)
    log(requeued=p)


def status():
    for r, s in RUNS:
        segf = os.path.join(OUT, name(r, s) + ".segments.json")
        print(name(r, s), "table" if os.path.exists(TABLES[r]) else "no-table", state_t(r, s), "trained" if trained(r, s) else "",
              "evaluated" if evaluated(r, s) else "", len(json.load(open(segf))) if os.path.exists(segf) else 0, "segments")


def jobs(jobdir):
    os.makedirs(jobdir, exist_ok=True)
    lines = ["@echo off",
             "rem S1-11b CPU lane: one <= 3 h segment of the 6 PPO runs (3a' fcg x3 now, 3b fcd x3 once features_diff_v3.npz",
             "rem exists), each resumed from its own FC_STATE; trained runs get their storm evaluation; re-queues itself.",
             "set PYTHONUTF8=1", "set CUDA_VISIBLE_DEVICES=-1", "set DEVICE=cpu", f"cd /d {E1W}\\ml\\forecast",
             f"{E1W}\\.venv\\Scripts\\python.exe -u windows\\ppo3b.py segment"]
    with open(os.path.join(jobdir, JOB + ".cmd"), "w", newline="\r\n") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "segment":
        segment()
    elif a and a[0] == "status":
        status()
    elif a and a[0] == "jobs":
        jobs(a[1])
    else:
        sys.exit(__doc__)
