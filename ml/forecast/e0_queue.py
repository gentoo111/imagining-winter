"""E0 extra training seeds (GAPS #2): two more PPO runs per rung (none / signs / oracle), then the storm-focused
evaluation of each final checkpoint, logged to MLflow experiment last-embers-diffusion.

Same config as runs/e0-{none,signs,oracle} (their MLflow params): learn --workers local:<port>:8:rust, resume from
stage-11 model_1755066122.pt, 200M steps, eval every 5M, 10 s rounds, gamma 0.9995, lambda 0.98, lr 3e-4, 4 epochs,
batch 4096, clip 0.2, ent 0.01, HARDCORE, chaotic, murky, max_days 120, eval backend rust.

Frozen rules: training runs from ROOT, a `git archive 4195eb6 ml data` checkout (simrs/ and data/ identical to 339afa0,
ml/ identical to main's current ml/), with ml/.venv symlinked to the main venv (its lastembers_sim was built from
4195eb6; never rebuilt). dppo.py takes data/ from its own parent directory (workers: --root dirname(ml); learner and
evaluate: ../data), so the runs never read main's data/ (FP eating rule since 863bfff).

dppo.py has no --seed option: worker env seeds come from port*100+time and the learner's torch RNG is unseeded, so
every run is an independent draw. For the record the learner's torch/numpy RNGs are seeded with the run's seed
number through a runpy wrapper (dppo.py itself is not modified).

Each job runs under main_gate (memory lease, 4 GB) and nice -n 10. Parallelism is read from ROOT/ml/runs/e0_queue.conf
({"parallel": N}, default 3 = the budget of the original three concurrent E0 runs; each run uses about 1.4 cores
and ~5.9k steps/s on the shared M5 Pro) before each new job starts, so it can be changed while the queue runs.
A restarted queue skips jobs whose main_gate lease is still held (still running from the earlier queue).

  nohup ~/Documents/Games/last-embers/ml/.venv/bin/python e0_queue.py > ROOT/ml/runs/e0_queue.log 2>&1 &
  cat ROOT/ml/runs/e0_queue_status.json

Frozen module (2026-11-02): the main venv's lastembers_sim may be rebuilt with newer rules at any time, so every job
runs with PYTHONPATH=ROOT/ml/frozen_site (a copy of the 4195eb6 build) and asserts that lastembers_sim resolves there
(training wrapper, fc_ppo.py workers and eval_e0.py). The finished seed-2/3 queue ran before this guard, on the
then-unchanged main venv module (same .so md5 as frozen_site, checked 2026-11-02).

Rung 3 (signs + GRU forecast, fc_obs.py / fc_ppo.py), three training seeds, its own status/log/conf files:
  nohup PY e0_queue.py --name fc_queue --jobs fc:1,fc:2,fc:3 > ROOT/ml/runs/fc_queue.log 2>&1 &
  cat ROOT/ml/runs/fc_queue_status.json
"""
import json, os, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/Documents/Games/last-embers-e0frozen")
ML = os.path.join(ROOT, "ml")
RUNS = os.environ.get("E0_RUNS") or os.path.join(ML, "runs")  # S1-11b on Windows: D:\\LastEmbers\\e1\\out\\ppo3b
PY = os.path.join(ML, ".venv", "bin", "python")
GATE = os.path.expanduser("~/Documents/AI-Automation/old-mac-runtime/production/main_gate.py")
START = os.path.expanduser("~/Documents/Games/last-embers/ml/runs/stage11/model_1755066122.pt")
STEPS = 200_000_000
JOBS = [(r, s) for s in (2, 3) for r in ("none", "signs", "oracle")]
FLAGS = {"none": [], "signs": ["--precursor-obs"], "oracle": ["--precursor-obs", "--oracle-obs"], "fc": ["--precursor-obs"],
         "fcd": ["--precursor-obs"], "fcg": ["--precursor-obs"]}  # fcd / fcg: S1-11b table-mode rungs (fc_ppo.py)
QNAME = "e0_queue"
STATUS = os.path.join(RUNS, "e0_queue_status.json")
CONF = os.path.join(RUNS, "e0_queue.conf")
EXP = "last-embers-diffusion"
FROZEN_SITE = os.path.join(ML, "frozen_site")
FC_PPO = os.path.join(HERE, "fc_ppo.py")

WRAP = ("import sys, runpy, numpy, torch, os; import lastembers_sim; "
        f"assert os.path.realpath(lastembers_sim.__file__).startswith({os.path.realpath(FROZEN_SITE)!r}), lastembers_sim.__file__; "
        "s = int(sys.argv[1]); torch.manual_seed(s); numpy.random.seed(s); "
        "sys.argv = ['dppo.py'] + sys.argv[2:]; runpy.run_path('dppo.py', run_name='__main__')")


def log(**kw):
    print(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}), flush=True)


def name(r, s):
    return f"e0-{r}-s{s}"


def final_model(d):
    if not os.path.isdir(d):
        return None
    fs = sorted(f for f in os.listdir(d) if f.startswith("model_"))
    return os.path.join(d, fs[-1]) if fs and int(fs[-1][6:15]) >= STEPS else None


def eval_files(r, s):
    return {m: os.path.join(HERE, "out", f"e0_eval_{m}_storm_s{s}_{r}.json") for m in ("greedy", "sample")}


def train_cmd(r, s, i):
    port = 23000 + 100 * i
    learn = ["learn", "--workers", f"local:{port}:8:rust", "--resume", START, "--out", f"runs/{name(r, s)}",
             "--steps", str(STEPS), "--eval-every", "5000000", "--seconds", "10", "--lr", "0.0003", "--epochs", "4",
             "--batch", "4096", "--clip", "0.2", "--ent", "0.01", "--gamma", "0.9995", "--lam", "0.98",
             "--difficulty", "HARDCORE", "--eval-backend", "rust", "--weather", "chaotic", "--weather-noise", "murky",
             "--max-days", "120"] + FLAGS[r]
    if r == "fc":
        return ["nice", "-n", "10", PY, FC_PPO, str(s)] + learn
    return ["nice", "-n", "10", PY, "-c", WRAP, str(s)] + learn


def eval_cmds(r, s):
    d = os.path.join(RUNS, name(r, s))
    base = ["nice", "-n", "10", PY, os.path.join(HERE, "eval_e0.py"), "--seeds", "300000:301000", "--agents", r,
            "--threads", "3", "--probe-every", "0", "--run-dirs", f"{r}={d}", "--tag", f"_storm_s{s}_{r}"]
    return [base + ["--mode", "greedy"], base + ["--mode", "sample"]]


def job_script(r, s, i):
    """One shell job: train (unless already finished), then evaluate; all under one main_gate lease."""
    d = os.path.join(RUNS, name(r, s))
    q = lambda c: " ".join("'" + x.replace("'", "'\\''") + "'" for x in c)
    parts = [f"cd {q([ML])}", f"export PYTHONPATH={q([FROZEN_SITE])}"]
    if not final_model(d):
        parts.append(f"{q(train_cmd(r, s, i))} > {q([d + '.log'])} 2>&1")
    for c in eval_cmds(r, s):
        parts.append(f"{q(c)} >> {q([d + '.eval.log'])} 2>&1")
    parts.append(f"{q([PY, os.path.abspath(__file__), '--report', r, str(s)])} >> {q([d + '.eval.log'])} 2>&1")
    script = " && ".join(parts)
    return [sys.executable, GATE, "run", "--holder", name(r, s), "--gb", "4", "--wait", str(7 * 86400), "--",
            "/bin/sh", "-c", script]


def report(r, s):
    """Storm summary of this run's final checkpoint -> MLflow (experiment last-embers-diffusion)."""
    sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
    from storm_e0 import summarize, load
    files = eval_files(r, s)
    metrics, out = {}, {}
    for m, p in files.items():
        by = load(p)
        eps = by[r]
        res = summarize({r: eps})[r]
        A = res["all"]
        day = float(sum(e["day"] for e in eps) / len(eps))
        out[m] = dict(mean_day=day, storms=res["n_storms"], **{k: A[k] for k in ("deaths", "any_death", "death_rate", "ended")})
        metrics.update({f"{m}_mean_day": day, f"{m}_storms_per_ep": res["storms_per_episode"],
                        f"{m}_deaths_per_storm": A["deaths"][0], f"{m}_p_storm_death": A["any_death"][0],
                        f"{m}_deaths_per100": A["death_rate"][0], f"{m}_ended_in_storm": A["ended"][0],
                        f"{m}_exiled": sum(e["outcome"] == "exiled" for e in eps) / len(eps),
                        f"{m}_exploded": sum(e["outcome"] == "exploded" for e in eps) / len(eps)})
        for t, v in res["tier"].items():
            if v["deaths"][0] is not None:
                metrics[f"{m}_deaths_per_storm_{t}"] = v["deaths"][0]
    fm = final_model(os.path.join(RUNS, name(r, s)))
    if r in ("fcd", "fcg"):  # S1-11b: the configured training length and the final checkpoint's step
        out.update(steps=STEPS, t_final=int(os.path.basename(fm)[6:15]) if fm else None)
    json.dump(out, open(os.path.join(RUNS, name(r, s) + ".storm.json"), "w"), indent=1)
    try:
        from mlflow_live import _post, _get
        try:
            eid = _get("experiments/get-by-name", {"experiment_name": EXP})["experiment"]["experiment_id"]
        except Exception:
            eid = _post("experiments/create", {"name": EXP})["experiment_id"]
        ts = int(time.time() * 1000)
        run = _post("runs/create", {"experiment_id": eid, "run_name": f"{name(r, s)}-storm-eval", "start_time": ts,
                                    "tags": [{"key": k, "value": v} for k, v in
                                             (("project", "last-embers"), ("stage", "E0"), ("rung", r), ("train_seed", str(s)),
                                              ("ckpt", final_model(os.path.join(RUNS, name(r, s))) or ""),
                                              ("sim", "lastembers_sim@4195eb6 + data 339afa0 (frozen)"),
                                              ("eval_seeds", "300000-300999"))]})["run"]["info"]["run_id"]
        _post("runs/log-batch", {"run_id": run, "metrics": [{"key": k, "value": float(v), "timestamp": ts, "step": 0}
                                                            for k, v in metrics.items() if v is not None]})
        _post("runs/update", {"run_id": run, "status": "FINISHED", "end_time": ts})
        log(report=name(r, s), mlflow_run=run)
    except Exception as e:
        log(report=name(r, s), mlflow_error=repr(e))
    print(json.dumps(out), flush=True)


def parallel():
    try:
        return max(0, int(json.load(open(CONF))["parallel"]))
    except Exception:
        return 3


def main():
    global JOBS, QNAME, STATUS, CONF
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="", help="rung:seed,... (default: the seed-2/3 E0 jobs)")
    ap.add_argument("--name", default="e0_queue", help="status/conf file prefix in ROOT/ml/runs")
    ap.add_argument("--port-base", type=int, default=0, help="job i uses port 23000 + 100 * (port_base + i)")
    a = ap.parse_args()
    if a.jobs:
        JOBS = [(j.split(":")[0], int(j.split(":")[1])) for j in a.jobs.split(",")]
    QNAME = a.name
    STATUS = os.path.join(RUNS, f"{QNAME}_status.json"); CONF = os.path.join(RUNS, f"{QNAME}.conf")
    os.makedirs(RUNS, exist_ok=True)
    if not os.path.exists(CONF):
        json.dump({"parallel": 3}, open(CONF, "w"))
    pending = [(r, s, a.port_base + i) for i, (r, s) in enumerate(JOBS)
               if not all(os.path.exists(p) for p in eval_files(r, s).values())
               or not os.path.exists(os.path.join(RUNS, name(r, s) + ".storm.json"))]
    # jobs of an earlier queue process may still be running (they hold their main_gate lease): leave them alone
    sys.path.insert(0, os.path.dirname(GATE))
    import main_gate
    busy = {lease["holder"] for lease in main_gate.leases()}
    if busy & {name(r, s) for r, s, _ in pending}:
        log(skipped_running=sorted(busy & {name(r, s) for r, s, _ in pending}))
    pending = [(r, s, i) for r, s, i in pending if name(r, s) not in busy]
    for r, s, _ in pending:  # a run that died half-way cannot resume its optimiser: keep it aside, start again
        d = os.path.join(RUNS, name(r, s))
        if os.path.isdir(d) and not final_model(d):
            shutil.move(d, d + f".partial-{int(time.time())}")
    running, done = {}, []
    log(queue="start", pending=[name(r, s) for r, s, _ in pending], root=ROOT)
    while pending or running:
        for k, (p, t0) in list(running.items()):
            if p.poll() is not None:
                log(job=k, exit=p.returncode, hours=round((time.time() - t0) / 3600, 2))
                done.append({"job": k, "exit": p.returncode, "hours": round((time.time() - t0) / 3600, 2)})
                del running[k]
        while pending and len(running) < parallel():
            r, s, i = pending.pop(0)
            p = subprocess.Popen(job_script(r, s, i), stdin=subprocess.DEVNULL)
            running[name(r, s)] = (p, time.time())
            log(job=name(r, s), started=p.pid)
        st = {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "parallel": parallel(),
              "running": {}, "pending": [name(r, s) for r, s, _ in pending], "done": done}
        for k in running:
            try:
                last = open(os.path.join(RUNS, k, "rounds.jsonl")).readlines()[-1]
                st["running"][k] = {"t": json.loads(last)["t"], "of": STEPS}
            except Exception:
                st["running"][k] = {"t": None, "phase": "starting or evaluating"}
        json.dump(st, open(STATUS + ".tmp", "w"), indent=1); os.replace(STATUS + ".tmp", STATUS)
        time.sleep(60)
    log(queue="finished", done=done)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--report":
        report(sys.argv[2], int(sys.argv[3]))
    else:
        main()
