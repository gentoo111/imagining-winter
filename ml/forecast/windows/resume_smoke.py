"""S1-3a GPU resume smoke (queue job jobs\\s3a_resume_smoke.cmd, minutes):  resume_smoke.py start | compare
start:   v2 diff endurance/murky 400 steps in one go, twice (no resume variables): once.pt with DETERMINISTIC=1
         and repeat.pt without it (the GPU's run-to-run noise floor; two plain CUDA runs differ by ~9e-4, above the
         1e-4 tolerance, so the resume comparison runs with deterministic kernels); then train_chain.py (also
         DETERMINISTIC=1) with a time budget of 0.75 x once's loop seconds: segment 1 stops on the budget and
         re-queues itself as s3a_resume_smoke_chain.cmd; the queue runs segment 2 to 400, then `compare`.
compare: final parameters once vs chained -> D:\\LastEmbers\\e1\\out\\resume_smoke\\done.json
         {segments, steps, max_abs_param_diff, val_once, val_chained, ...}.
Code: this step's train.py is deployed to windows\\resume\\ (checked against resume\\SHA256SUMS) and run with
PYTHONPATH=ml\\forecast, so the s1-2 / s1-7 jobs queued before it keep their own ml\\forecast\\*.py."""
import hashlib, json, os, shutil, subprocess, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FC = os.path.dirname(HERE)
E1 = os.path.abspath(os.path.join(FC, "..", ".."))
RES = os.path.join(HERE, "resume")
OUT = os.path.join(E1, "out", "resume_smoke")
STEPS = 400
ARGS = ["diff", "endurance", "murky", str(STEPS)]
for ln in open(os.path.join(RES, "SHA256SUMS")):
    h, f = ln.split()
    assert hashlib.sha256(open(os.path.join(RES, f), "rb").read()).hexdigest() == h, f"{f} is not the deployed copy"
env = {**os.environ, "PYTHONUTF8": "1", "DEVICE": os.environ.get("DEVICE", "cuda"), "PYTHONPATH": FC,
       "CHAIN_TRAIN_PY": os.path.join(RES, "train.py"), "CHAIN_ENV": "DETERMINISTIC"}
for k in ("RESUME_STATE", "TIME_BUDGET", "STOP_AT", "VAL_EVERY", "CKPT_DIR", "DIFF_VARIANT", "DETERMINISTIC"):
    env.pop(k, None)
once, chained, repeat = (os.path.join(OUT, f) for f in ("once.pt", "chained.pt", "repeat.pt"))


def maxdiff(A, B):
    assert A.keys() == B.keys()
    d = {k: float((A[k].double() - B[k].double()).abs().max()) for k in A if A[k].numel()}
    w = max(d, key=d.get)
    return d[w], w, all(torch.equal(A[k], B[k]) for k in A)


def last(log):
    return [json.loads(ln) for ln in open(log) if ln.strip()][-1]


if sys.argv[1] == "start":
    shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)
    subprocess.run([sys.executable, "-u", env["CHAIN_TRAIN_PY"], *ARGS, repeat], cwd=FC, env=env, check=True)
    env["DETERMINISTIC"] = "1"
    subprocess.run([sys.executable, "-u", env["CHAIN_TRAIN_PY"], *ARGS, once], cwd=FC, env=env, check=True)
    secs = last(once + ".log")["secs"]
    budget = max(1.0, round(0.75 * secs, 1))
    json.dump({"once_secs": secs, "budget_sec": budget}, open(os.path.join(OUT, "meta.json"), "w"))
    subprocess.run([sys.executable, "-u", os.path.join(HERE, "train_chain.py"), *ARGS, chained,
                    "--job", "s3a_resume_smoke_chain", "--budget", str(budget),
                    "--done", os.path.join(OUT, "chain_done.json"), "--then", "windows\\resume_smoke.py compare"],
                   cwd=FC, env=env, check=True)
else:
    ch, meta = json.load(open(os.path.join(OUT, "chain_done.json"))), json.load(open(os.path.join(OUT, "meta.json")))
    st = torch.load(chained + ".state", map_location="cpu", weights_only=False)
    A = torch.load(once, map_location="cpu")
    md, worst, eq = maxdiff(A, st["net"])
    rd, rworst, req = maxdiff(A, torch.load(repeat, map_location="cpu"))
    d = {"segments": len(ch["segments"]), "steps": int(st["it"]), "max_abs_param_diff": md,
         "worst_tensor": worst, "bit_equal": eq, "deterministic": True,
         "noise_floor_nondeterministic_max_abs_param_diff": rd, "noise_floor_worst_tensor": rworst,
         "val_once": last(once + ".log")["val"], "val_chained": last(chained + ".log")["val"],
         "once_secs": meta["once_secs"], "budget_sec": meta["budget_sec"], "chain_segments": ch["segments"],
         "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "torch": torch.__version__,
         "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(d, open(os.path.join(OUT, "done.json"), "w"), indent=1)
    print(json.dumps(d, indent=1))
