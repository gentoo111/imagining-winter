"""dppo.py of the frozen root, run with the rung-3 forecast observation (fc_obs.py): same CLI and training loop,
lastembers_sim.VecEnv / Env replaced by the forecast wrappers in the learner (periodic evaluation) and in every worker.
dppo.py itself is not modified: its source is executed as __main__ with one substitution, so that local workers are
started through this script (and so get the wrappers too).

  cd <frozen root>/ml && PY fc_ppo.py <train seed> learn --workers local:<port>:8:rust --precursor-obs ...   (learner)
  (workers: fc_ppo.py - worker ...; started by the learner)
The learner's torch/numpy RNGs are seeded with <train seed> (as e0_queue.py does for the other rungs); "-" = unseeded.
S1-11a: FC_TABLE=<npz> = table mode (fc_obs.py; 3b / 3a'), inherited by the workers. On Windows (os.name == "nt") the
local worker command `nice -n 10 <root>/ml/.venv/bin/python` is replaced by this interpreter (second substitution);
FC_TORCH_THREADS=<n> sets the learner's torch threads (unset = torch default, as before). PPO hyperparameters unchanged.
S1-11b (all unset = unchanged; the hooks below are then identities):
  FC_STATE=<file>    chained training: the learner restores policy, Adam, numpy/torch global RNG, t, next_eval, round,
                     warm, good and the adaptive seconds from the file if it exists, saves them every FC_SAVE_EVERY s
                     (default 600) and at the last round, and stops cleanly (state saved, workers told to quit, exit 0)
                     when FC_BUDGET s since start are nearly used or FC_STOP_ROUND rounds are done.
  FC_TARGET_STEPS=n  dppo's rounds are wall-clock (the `seconds` sent to the workers); they are set to
                     n / (smoothed measured steps per second) so every PPO rollout has ~n steps, as 3a had on the Mac.
                     The CLI --seconds stays as given and is only the starting value.
  FC_DET_ROLLOUT=k   (tests only) each worker round reseeds torch/numpy from the received weights, restarts every
                     episode and collects exactly k vec-steps, so the same learner state gives the same data.
Resuming the 128-d stage-11 checkpoint into the 140-d policy zero-initialises the 12 new input columns (dppo.load_resized),
so rung 3 starts exactly where the signs rung started."""
import hashlib, json, os, sys, time

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fc_obs  # noqa: E402

ls = fc_obs.install()
ML = os.path.join(fc_obs.FROZEN, "ml")
sys.path.insert(1, ML)
SRC = os.path.join(ML, "dppo.py")
src = open(SRC).read()
OLD = 'os.path.join(HERE, "dppo.py"), "worker"'
assert src.count(OLD) == 1
src = src.replace(OLD, f'{os.path.abspath(__file__)!r}, "-", "worker"')
if os.name == "nt":
    OLDW = 'cmd = ["nice", "-n", "10", py, '
    assert src.count(OLDW) == 1
    src = src.replace(OLDW, 'cmd = [sys.executable, ')

# S1-11b hooks
A_RESUME = '    log = open(os.path.join(a.out, "rounds.jsonl"), "a")\n'
A_ROUND = '        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\\n"); log.flush()\n'
A_SEC = '{"weights": w, "seconds": a.seconds,'
for x in (A_RESUME, A_ROUND, A_SEC):
    assert src.count(x) == 1, x
src = src.replace(A_RESUME, "    t, next_eval, rnd, warm, good = _fc_resume(model, opt, t, next_eval, rnd, warm, good)\n" + A_RESUME)
src = src.replace(A_ROUND, A_ROUND + "        _fc_after_round(model, opt, links, t, next_eval, rnd, warm, good, n)\n")
src = src.replace(A_SEC, '{"weights": w, "seconds": _fc_seconds(a.seconds),')
i0 = src.index("def worker_rust(")
i1 = src.index("\ndef ", i0 + 10)
wr = src[i0:i1]
W_BUDGET = '        budget, t0 = float(msg["seconds"]), time.time()\n'
W_LOOP = '        while time.time() - t0 < budget:\n'
for x in (W_BUDGET, W_LOOP):
    assert wr.count(x) == 1, x
DET = ('        if _FC_DET:  # tests: the same learner state gives the same rollout\n'
       '            _h = int.from_bytes(hashlib.sha256(msg["weights"]).digest()[:4], "little")\n'
       '            torch.manual_seed(_h); rng = np.random.default_rng(_h)\n'
       '            seeds = [new_seed() for _ in range(n)]; steps_taken = [0] * n\n'
       '            obs, mask = venv.reset(seeds, True, warm, True, diff, weather, max_days, noise or "murky")\n')
wr = wr.replace(W_BUDGET, DET + W_BUDGET)
wr = wr.replace(W_LOOP, '        while (len(cols["act"]) < _FC_DET) if _FC_DET else (time.time() - t0 < budget):\n')
src = src[:i0] + wr + src[i1:]

STATE = os.environ.get("FC_STATE")
BUDGET = float(os.environ.get("FC_BUDGET", "0"))
STOP_ROUND = int(os.environ.get("FC_STOP_ROUND", "0"))
SAVE_EVERY = float(os.environ.get("FC_SAVE_EVERY", "600"))
TARGET = int(os.environ.get("FC_TARGET_STEPS", "0"))
_FC_DET = int(os.environ.get("FC_DET_ROLLOUT", "0"))
FC = {"sec": None, "rate": None, "saved": time.time(), "last": time.time(), "round_wall": 0.0}


def _fc_seconds(cli):
    if not TARGET:
        return cli
    if FC["sec"] is None:
        FC["sec"] = float(cli)
    return FC["sec"]


def _fc_save(model, opt, t, next_eval, rnd, warm, good):
    import numpy, torch
    st = {"policy": model.state_dict(), "opt": opt.state_dict(), "np_rng": numpy.random.get_state(),
          "torch_rng": torch.get_rng_state(), "t": t, "next_eval": next_eval, "rnd": rnd, "warm": warm, "good": good,
          "fc_sec": FC["sec"], "fc_rate": FC["rate"], "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
    torch.save(st, STATE + ".tmp")
    os.replace(STATE + ".tmp", STATE)
    FC["saved"] = time.time()


def _fc_resume(model, opt, t, next_eval, rnd, warm, good):
    if not (STATE and os.path.exists(STATE)):
        return t, next_eval, rnd, warm, good
    import numpy, torch
    st = torch.load(STATE, weights_only=False)
    model.load_state_dict(st["policy"]); opt.load_state_dict(st["opt"])
    numpy.random.set_state(st["np_rng"]); torch.set_rng_state(st["torch_rng"])
    FC["sec"], FC["rate"] = st["fc_sec"], st["fc_rate"]
    print(json.dumps({"fc_resumed": st["t"], "round": st["rnd"], "seconds": FC["sec"], "state": STATE}), flush=True)
    return st["t"], st["next_eval"], st["rnd"], st["warm"], st["good"]


def _fc_after_round(model, opt, links, t, next_eval, rnd, warm, good, n):
    now = time.time()
    FC["round_wall"] = now - FC["last"]; FC["last"] = now
    if TARGET and FC["sec"]:
        rate = n / FC["sec"]
        FC["rate"] = rate if FC["rate"] is None else 0.8 * FC["rate"] + 0.2 * rate
        FC["sec"] = min(max(TARGET / FC["rate"], 0.5 * FC["sec"]), 2.0 * FC["sec"])
    if TARGET or STATE:
        print(json.dumps({"fc_round": rnd, "t": t, "steps": n, "seconds": round(FC["sec"] or 0, 2),
                          "round_wall": round(FC["round_wall"], 1)}), flush=True)
    if not STATE:
        return
    steps = g["a"].steps
    stop = bool((STOP_ROUND and rnd >= STOP_ROUND) or (BUDGET and now - T0 + 1.5 * FC["round_wall"] > BUDGET))
    if stop or now - FC["saved"] > SAVE_EVERY or t >= steps:
        _fc_save(model, opt, t, next_eval, rnd, warm, good)
    if stop and t < steps:
        for l in links:
            try:
                g["send_msg"](l.s, {"quit": True})
            except Exception:
                pass
        print(json.dumps({"fc_segment_end": t, "round": rnd, "secs": round(now - T0)}), flush=True)
        sys.stdout.flush()
        os._exit(0)


if os.environ.get("FC_TORCH_THREADS"):
    import torch
    torch.set_num_threads(int(os.environ["FC_TORCH_THREADS"]))

seed = sys.argv[1]
if seed != "-":
    import numpy, torch
    torch.manual_seed(int(seed)); numpy.random.seed(int(seed))
sys.argv = [SRC] + sys.argv[2:]
print({"fc_obs": True, "every_h": fc_obs.EVERY, "gru": fc_obs.GRU_CKPT, "FC_TABLE": fc_obs.FC_TABLE,
       "lastembers_sim": ls.__file__}, flush=True)
if os.environ.get("FC_DUMP_SRC"):  # tests: the substituted dppo source, then exit
    open(os.environ["FC_DUMP_SRC"], "w").write(src)
    sys.exit(0)
g = {"__name__": "__main__", "__file__": SRC, "_fc_resume": _fc_resume, "_fc_after_round": _fc_after_round,
     "_fc_seconds": _fc_seconds, "_FC_DET": _FC_DET, "hashlib": hashlib}
exec(compile(src, SRC, "exec"), g)
