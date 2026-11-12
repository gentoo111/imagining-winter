"""S1-11b CPU check: fc_ppo.py chained training (FC_STATE) is bit-exact, and without FC_* the dppo source only gains
identity hooks.
  PYTHONPATH=<e0frozen>/ml/frozen_site ml/.venv/bin/python ml/forecast/tests/test_ppo_resume.py
Frozen sim, table mode (fc3b GRU table), one local rust worker with 4 envs, FC_DET_ROLLOUT=32 (the worker reseeds from
the received weights and collects exactly 32 vec-steps = 128 steps per round), batch 64, 2 epochs, --steps 1024:
A = 8 PPO updates in one process; B = FC_STOP_ROUND=4 (state saved, exit), then a second process resumes from the
state file and does updates 5..8. The final state files (policy, Adam state, numpy/torch RNG, t, round) must be equal
bit for bit. Then: the substituted dppo source differs from dppo.py only in lines carrying the _fc_/_FC_DET hooks."""
import difflib, os, shutil, subprocess, sys, tempfile
import numpy as np
import torch

FC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN = os.path.expanduser("~/Documents/Games/last-embers-e0frozen")
START = os.path.expanduser("~/Documents/Games/last-embers/ml/runs/stage11/model_1755066122.pt")
TABLE = os.environ.get("FC3B_TABLE", os.path.join(FC, "out", "fc3b", "features_gru_daily.npz"))
fails, passed = [], 0


def ok(name, cond, detail=""):
    global passed
    if cond:
        passed += 1
    else:
        fails.append(f"{name}: {detail}")


def learn(out, port, **env):
    e = {k: v for k, v in os.environ.items() if not k.startswith("FC_")}
    e.update(FC_TABLE=TABLE, FC_DET_ROLLOUT="32", FC_TORCH_THREADS="1", RAYON_NUM_THREADS="1", **env)
    cmd = [sys.executable, os.path.join(FC, "fc_ppo.py"), "1", "learn", "--workers", f"local:{port}:4:rust",
           "--resume", START, "--out", out, "--steps", "1024", "--eval-every", "1000000000", "--seconds", "10",
           "--lr", "0.0003", "--epochs", "2", "--batch", "64", "--clip", "0.2", "--ent", "0.01", "--gamma", "0.9995",
           "--lam", "0.98", "--difficulty", "HARDCORE", "--eval-backend", "rust", "--weather", "chaotic",
           "--weather-noise", "murky", "--max-days", "120", "--precursor-obs", "--no-mlflow"]
    r = subprocess.run(cmd, cwd=os.path.join(FROZEN, "ml"), env=e, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
    return r


def same(name, a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(f"{name}.{k}", a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same(name, x, y) for x, y in zip(a, b))
    return a == b


def test_resume():
    d = tempfile.mkdtemp()
    sa, sb = os.path.join(d, "A.state"), os.path.join(d, "B.state")
    ra = learn(os.path.join(d, "A"), 26100, FC_STATE=sa)
    ok("A ran", ra.returncode == 0 and os.path.exists(sa), ra.returncode)
    rb1 = learn(os.path.join(d, "B"), 26200, FC_STATE=sb, FC_STOP_ROUND="4")
    st = torch.load(sb, weights_only=False) if os.path.exists(sb) else {}
    ok("B stopped at round 4", rb1.returncode == 0 and st.get("rnd") == 4 and "fc_segment_end" in rb1.stdout,
       (rb1.returncode, st.get("rnd")))
    rb2 = learn(os.path.join(d, "B"), 26300, FC_STATE=sb)
    ok("B resumed", rb2.returncode == 0 and '"fc_resumed": 512' in rb2.stdout, rb2.returncode)
    A = torch.load(sa, weights_only=False); B = torch.load(sb, weights_only=False)
    ok("t / round", (A["t"], A["rnd"]) == (B["t"], B["rnd"]) == (1024, 8), (A["t"], A["rnd"], B["t"], B["rnd"]))
    ok("policy bit-equal", same("policy", A["policy"], B["policy"]))
    ok("optimizer bit-equal", same("opt", A["opt"], B["opt"]))
    ok("torch RNG equal", torch.equal(A["torch_rng"], B["torch_rng"]))
    ok("numpy RNG equal", same("np", list(A["np_rng"]), list(B["np_rng"])))
    p0 = torch.load(START, weights_only=False)
    moved = any(not torch.equal(v, A["policy"][k]) for k, v in p0.items() if k in A["policy"] and v.shape == A["policy"][k].shape)
    ok("policy actually trained", moved)
    fa = sorted(f for f in os.listdir(os.path.join(d, "A")) if f.startswith("model_"))
    fb = sorted(f for f in os.listdir(os.path.join(d, "B")) if f.startswith("model_"))
    ok("final checkpoints equal", fa[-1] == fb[-1] and same("ckpt", torch.load(os.path.join(d, "A", fa[-1])),
                                                             torch.load(os.path.join(d, "B", fb[-1]))), (fa, fb))
    shutil.rmtree(d, ignore_errors=True)


def test_hooks_only():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "src.py")
    e = {k: v for k, v in os.environ.items() if not k.startswith("FC_")}
    e["FC_DUMP_SRC"] = p
    subprocess.run([sys.executable, os.path.join(FC, "fc_ppo.py"), "1", "learn"], cwd=os.path.join(FROZEN, "ml"),
                   env=e, check=True, capture_output=True)
    new = open(p).read().splitlines(); old = open(os.path.join(FROZEN, "ml", "dppo.py")).read().splitlines()
    added = [ln[1:] for ln in difflib.unified_diff(old, new, lineterm="", n=0) if ln.startswith("+") and not ln.startswith("+++")]
    removed = [ln[1:] for ln in difflib.unified_diff(old, new, lineterm="", n=0) if ln.startswith("-") and not ln.startswith("---")]
    hook = lambda ln: "_fc_" in ln or "_FC_DET" in ln or "hashlib" in ln or "fc_ppo.py" in ln or ln.strip().startswith(
        ("torch.manual_seed(_h)", "seeds = [new_seed()", "obs, mask = venv.reset(seeds"))
    ok("only hook lines added", all(hook(ln) for ln in added), [ln for ln in added if not hook(ln)])
    ok("removed lines replaced by hooks", len(removed) <= 4, removed)
    shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    assert os.path.exists(TABLE), TABLE
    test_hooks_only()
    test_resume()
    for f in fails:
        print("FAIL", f)
    print(f"{passed} passed, {len(fails)} failed")
    sys.exit(1 if fails else 0)
