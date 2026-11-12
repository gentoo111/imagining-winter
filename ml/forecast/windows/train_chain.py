"""S1-3a chained training through the Windows queue (each queue job is capped at 180 min = 10800 s).
  train_chain.py <gru|diff> <profile> <noise> <steps> <out.pt> --job <name> [--budget SEC] [--done FILE] [--then "script args"]
One call = one segment: train.py runs with RESUME_STATE=<out>.state and TIME_BUDGET = what is left of --budget
(default 10200 s, never above 10800 s, so the segment ends inside the queue cap), saves its full state and exits.
The segment {sec, steps_from, steps_to} is appended to <out>.chain.json. If the target step is not reached, this
script writes queue\\pending\\<job>.cmd re-running itself with the same arguments and environment (same queue
convention: PYTHONUTF8=1, DEVICE, cd, venv python; the runner still waits for the PhD queue first) and exits 0.
At the target it writes --done (default <out dir>\\done.json: segments list, steps, wall seconds) and runs --then
(python script relative to ml\\forecast) if given. A killed segment resumes from the state of its last validation
when the job is queued again. CHAIN_TRAIN_PY points at a deployed train.py copy (default ml\\forecast\\train.py).
HTCondor pool (env CJOB_STOP set by the lane wrapper): a segment that has not reached the target exits 85 instead of
writing a queue job (HTCondor restarts the same command in place, so the environment is kept without CHAIN_ENV); a
successor is a separate Condor job with [after] (--then still runs at the target)."""
import argparse, json, os, shlex, subprocess, sys, time

T_START = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
FC = os.path.dirname(HERE)
E1 = os.path.abspath(os.path.join(FC, "..", ".."))
QUEUE = os.environ.get("CHAIN_QUEUE", os.path.join(E1, "queue"))
MAX_BUDGET = 10800  # queue runner $MaxMinutes = 180
PASS_ENV = ["DEVICE", "PYTHONPATH", "CHAIN_TRAIN_PY", "CHAIN_QUEUE", "TORCH_THREADS", "VAL_EVERY", "CKPT_DIR",
            "DIFF_VARIANT"] + [k for k in os.environ.get("CHAIN_ENV", "").split(",") if k]

ap = argparse.ArgumentParser()
ap.add_argument("kind"); ap.add_argument("profile"); ap.add_argument("noise"); ap.add_argument("steps", type=int)
ap.add_argument("out"); ap.add_argument("--job", required=True)
ap.add_argument("--budget", type=float, default=10200); ap.add_argument("--done"); ap.add_argument("--then")
a = ap.parse_args()
if not 0 < a.budget <= MAX_BUDGET:
    sys.exit(f"--budget {a.budget} must be in (0, {MAX_BUDGET}] s (queue cap)")
out = os.path.abspath(a.out)
state, chainf = out + ".state", out + ".chain.json"
done = os.path.abspath(a.done) if a.done else os.path.join(os.path.dirname(out), "done.json")
train_py = os.environ.get("CHAIN_TRAIN_PY", os.path.join(FC, "train.py"))
os.environ.setdefault("DEVICE", "cuda")
import torch  # noqa: E402


def step_now():
    return int(torch.load(state, map_location="cpu", weights_only=False)["it"]) if os.path.exists(state) else 0


chain = json.load(open(chainf)) if os.path.exists(chainf) else {
    "job": a.job, "kind": a.kind, "profile": a.profile, "noise": a.noise, "steps": a.steps, "out": out, "segments": []}
assert (chain["kind"], chain["profile"], chain["noise"], chain["steps"]) == (a.kind, a.profile, a.noise, a.steps), chain
frm = step_now()
if frm < a.steps:
    left = a.budget - (time.time() - T_START)
    t = time.time()
    subprocess.run([sys.executable, "-u", train_py, a.kind, a.profile, a.noise, str(a.steps), out], cwd=FC, check=True,
                   env={**os.environ, "RESUME_STATE": state, "TIME_BUDGET": str(max(left, 1.0))})
    to = step_now()
    assert to > frm, f"no progress: state at {to}"
    chain["segments"].append({"sec": round(time.time() - t), "steps_from": frm, "steps_to": to})
    json.dump(chain, open(chainf, "w"), indent=1)
    print(json.dumps(chain["segments"][-1]), flush=True)
    if to < a.steps and os.environ.get("CJOB_STOP"):  # HTCondor: restart in place (checkpoint_exit_code = 85)
        print(f"segment end at step {to} of {a.steps}: exit 85 (HTCondor restarts in place)", flush=True)
        sys.exit(85)
    if to < a.steps:
        args = [a.kind, a.profile, a.noise, str(a.steps), out, "--job", a.job, "--budget", str(a.budget), "--done", done]
        if a.then:
            args += ["--then", a.then]
        lines = ["@echo off", f"rem train_chain.py segment {len(chain['segments']) + 1}: {a.job} from step {to} of {a.steps}",
                 "set PYTHONUTF8=1"] + [f"set {k}={os.environ[k]}" for k in PASS_ENV if os.environ.get(k)] + [
                 f"cd /d {FC}", subprocess.list2cmdline([sys.executable, "-u", os.path.abspath(__file__)] + args)]
        os.makedirs(os.path.join(QUEUE, "pending"), exist_ok=True)
        job = os.path.join(QUEUE, "pending", a.job + ".cmd")
        assert not os.path.exists(job), f"{job} already queued"
        with open(job + ".tmp", "w", newline="\r\n") as fh:
            fh.write("\n".join(lines) + "\n")
        os.replace(job + ".tmp", job)  # the runner only picks *.cmd
        print(f"re-queued {job} (resume from step {to})", flush=True)
        sys.exit(0)
d = {"job": a.job, "kind": a.kind, "profile": a.profile, "noise": a.noise, "steps": step_now(),
     "segments": chain["segments"], "wall_secs": sum(s["sec"] for s in chain["segments"]), "out": out,
     "device": torch.cuda.get_device_name(0) if os.environ["DEVICE"] != "cpu" and torch.cuda.is_available() else "cpu",
     "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
json.dump(d, open(done, "w"), indent=1)
print(json.dumps(d), flush=True)
if a.then:
    subprocess.run([sys.executable, "-u"] + shlex.split(a.then, posix=False), cwd=FC, check=True)
