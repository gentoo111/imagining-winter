"""S1-11a: Windows CPU throughput of the frozen-sim rung-3 PPO for planning s1-11b (CPU lane job, no training kept).
  ppo_bench.py [--ks 1,2,3,6] [--secs 480] [--threads 18] [--table <npz>] [--rayon R] [--tag T]
For each k, k learners `fc_ppo.py <i> learn ...` run concurrently with exactly e0_queue.train_cmd("fc", ...)'s
hyperparameters (one local rust worker with 8 envs, 10 s rounds, lr 3e-4, 4 epochs, batch 4096, ...; only the port,
--out, the Windows START path and --no-mlflow differ) in table mode (FC_TABLE, default the gru_daily table), on the
S2-0 Windows frozen module (PYTHONPATH D:\\LastEmbers\\e1\\sim\\site) and a copy of the frozen 4195eb6 ml/ + data/
(FC_FROZEN_ROOT D:\\LastEmbers\\e1\\ppo\\frozen). Threads: each run gets T = threads // k, split into
RAYON_NUM_THREADS = max(1, T // 2) for the worker's VecEnv pool and FC_TORCH_THREADS = OMP_NUM_THREADS = max(1, T - rayon)
for the learner; the GPU is hidden. After --secs wall seconds the runs are killed and their dirs deleted.
Per run: steady steps/s = (t_last - t_first) / (wall_last - wall_first) over the observed rounds (collection + PPO
update), mean steps per round, and the start-up time (process start -> first round, includes the t=0 evaluation).
Note: dppo rounds are wall-clock (--seconds 10), so the rollout per PPO update (steps per round) scales with speed;
mean_steps_per_round is reported so it can be compared with 3a's (~45k on the M5 Pro).
Writes D:\\LastEmbers\\e1\\out\\fc3b\\ppo_bench<tag>.json."""
import argparse, json, os, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
FC = os.path.dirname(HERE)
E1 = os.path.abspath(os.path.join(FC, "..", ".."))
sys.path.insert(0, FC)
import e0_queue  # noqa: E402  (train_cmd = the 3a hyperparameters)

FROZEN_ROOT = os.path.join(E1, "ppo", "frozen")
SITE = os.path.join(E1, "sim", "site")
PYDEPS = os.path.join(E1, "ppo", "pydeps")      # gymnasium for the frozen env.py (not in the e1 venv)
START = os.path.join(E1, "ppo", "start", "model_1755066122.pt")
OUT = os.path.join(E1, "out", "fc3b")
BENCH = os.path.join(OUT, "bench")


def learn_args(i, out, port):
    cmd = e0_queue.train_cmd("fc", i + 1, 0)
    assert cmd[:3] == ["nice", "-n", "10"] and cmd[4] == e0_queue.FC_PPO, cmd[:6]
    a = cmd[6:]
    a[a.index("--workers") + 1] = f"local:{port}:8:rust"
    a[a.index("--resume") + 1] = START
    a[a.index("--out") + 1] = out
    return [str(i + 1)] + a + ["--no-mlflow"]


def rounds(path):
    try:
        return [json.loads(ln) for ln in open(path) if ln.strip()]
    except (OSError, ValueError):
        return None


def bench(k, secs, threads, table, rayon_fixed=0):
    T = max(1, threads // k)
    rayon = rayon_fixed or max(1, T // 2); tt = max(1, T - rayon)
    env = {**os.environ, "PYTHONUTF8": "1", "CUDA_VISIBLE_DEVICES": "-1", "DEVICE": "cpu",
           "PYTHONPATH": os.pathsep.join([SITE, PYDEPS, FC]), "FC_FROZEN_ROOT": FROZEN_ROOT, "FC_FROZEN_SITE": SITE,
           "FC_TABLE": table, "RAYON_NUM_THREADS": str(rayon), "FC_TORCH_THREADS": str(tt), "OMP_NUM_THREADS": str(tt),
           "MKL_NUM_THREADS": str(tt)}
    runs = []
    for i in range(k):
        out = os.path.join(BENCH, f"k{k}_{i}")
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(out)
        cmd = [sys.executable, "-u", os.path.join(FC, "fc_ppo.py")] + learn_args(i, out, 24000 + 100 * i)
        lf = open(os.path.join(BENCH, f"k{k}_{i}.log"), "w")
        p = subprocess.Popen(cmd, cwd=os.path.join(FROZEN_ROOT, "ml"), env=env, stdout=lf, stderr=subprocess.STDOUT)
        runs.append({"i": i, "p": p, "out": out, "t0": time.time(), "seen": [], "lf": lf})
    t_end = time.time() + secs
    while time.time() < t_end:
        time.sleep(2)
        now = time.time()
        for r in runs:
            rr = rounds(os.path.join(r["out"], "rounds.jsonl"))
            if rr and len(rr) > len(r["seen"]):
                r["seen"] += [(now, x["t"], x["steps"], x["secs"]) for x in rr[len(r["seen"]):]]
            if r["p"].poll() is not None and "rc" not in r:
                r["rc"] = r["p"].returncode
    res = []
    for r in runs:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(r["p"].pid)], capture_output=True)
        r["p"].wait(); r["lf"].close()
        s = r["seen"]
        rec = {"run": r["i"], "rounds": len(s), "rc_early": r.get("rc"),
               "startup_secs": round(s[0][0] - r["t0"], 1) if s else None}
        if len(s) >= 3:
            rec["steps_per_s"] = round((s[-1][1] - s[0][1]) / (s[-1][0] - s[0][0]), 1)
            rec["mean_steps_per_round"] = round(sum(x[2] for x in s[1:]) / (len(s) - 1))
            rec["mean_collect_secs"] = round(sum(x[3] for x in s[1:]) / (len(s) - 1), 2)
            rec["round_wall_secs"] = round((s[-1][0] - s[0][0]) / (len(s) - 1), 2)
        res.append(rec)
        shutil.rmtree(r["out"], ignore_errors=True)
    tot = sum(x.get("steps_per_s", 0) for x in res)
    d = {"k": k, "threads_per_run": T, "rayon": rayon, "torch": tt, "secs": secs, "runs": res,
         "total_steps_per_s": round(tot, 1), "per_run_steps_per_s": round(tot / k, 1)}
    print(json.dumps(d), flush=True)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="1,2,3,6"); ap.add_argument("--secs", type=float, default=480)
    ap.add_argument("--threads", type=int, default=18)
    ap.add_argument("--table", default=os.path.join(OUT, "features_gru_daily.npz"))
    ap.add_argument("--rayon", type=int, default=0, help="fixed RAYON_NUM_THREADS per run (0 = T // 2)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(BENCH, exist_ok=True)
    assert os.path.exists(START) and os.path.exists(a.table), (START, a.table)
    t0 = time.time()
    res = [bench(int(k), a.secs, a.threads, a.table, a.rayon) for k in a.ks.split(",")]
    steps = 200_000_000
    best = max(res, key=lambda d: d["total_steps_per_s"])
    plan = {}
    for d in res:
        if d["per_run_steps_per_s"] > 0:
            waves = -(-6 // d["k"])
            ups = [r["startup_secs"] for r in d["runs"] if r.get("startup_secs")]
            ev = (sum(ups) / len(ups)) * (steps // 5_000_000 + 1) if ups else 0.0  # ~ eval at t=0 and every 5M steps
            hrs = (steps / d["per_run_steps_per_s"] + ev) / 3600
            plan[f"k{d['k']}"] = {"waves": waves, "hours_per_run": round(hrs, 1), "hours_all_6": round(waves * hrs, 1),
                                  "eval_overhead_hours_per_run": round(ev / 3600, 2)}
    json.dump({"bench": res, "six_runs_200M": plan, "best_total_k": best["k"], "hyperparameters": "e0_queue.train_cmd('fc')",
               "threads_budget": a.threads, "rayon_fixed": a.rayon, "wall_secs": round(time.time() - t0),
               "finished": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(os.path.join(OUT, f"ppo_bench{a.tag}.json"), "w"), indent=1)
    print(json.dumps(plan), flush=True)
