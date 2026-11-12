"""S1-3a CPU check: train.py resume is bit-exact, and without resume variables train.py is unchanged.
  DEVICE=cpu ml/.venv/bin/python ml/forecast/tests/test_train_resume.py
A = `train.py diff endurance murky 40` in one process; B = 20 steps (RESUME_STATE + STOP_AT=20), then a second
process continues from the state file to 40. Final net tensors, optimizer state and the .log (it, loss, val)
records must be equal (VAL_EVERY=10 so both segments log). Then the pre-change train.py (commit d199c1e) and the
new one, both without resume variables, must give the same .log records and best checkpoint."""
import json, os, shutil, subprocess, sys, tempfile
import torch

FC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.abspath(os.path.join(FC, "..", ".."))
OLD_REV = "d199c1e"  # last train.py on the base before S1-3a (defaults identical after s1-2 / s1-7)
ARGS = ["diff", "endurance", "murky", "40"]
fails = []


def run(script, out, **env):
    e = {k: v for k, v in os.environ.items() if k not in ("RESUME_STATE", "TIME_BUDGET", "STOP_AT", "VAL_EVERY", "CKPT_DIR")}
    e.update(DEVICE="cpu", PYTHONPATH=FC, **env)
    subprocess.run([sys.executable, script, *ARGS, out], cwd=FC, env=e, check=True, stdout=subprocess.DEVNULL)


def recs(out):
    return [(r["it"], r["loss"], r["val"]) for r in map(json.loads, open(out + ".log"))]


def same(name, a, b):
    """Recursive bit-equality for tensors / dicts / lists; records the first difference."""
    if isinstance(a, torch.Tensor):
        ok = isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    elif isinstance(a, dict):
        ok = isinstance(b, dict) and a.keys() == b.keys() and all(same(f"{name}.{k}", a[k], b[k]) for k in a)
        return ok
    elif isinstance(a, (list, tuple)):
        ok = len(a) == len(b) and all(same(f"{name}[{i}]", x, y) for i, (x, y) in enumerate(zip(a, b)))
        return ok
    else:
        ok = a == b
    if not ok:
        fails.append(f"first unequal: {name}")
    return ok


tmp = tempfile.mkdtemp(prefix="s13a_")
try:
    new = os.path.join(FC, "train.py")
    a, b = os.path.join(tmp, "a.pt"), os.path.join(tmp, "b.pt")
    sa, sb = a + ".state", b + ".state"
    run(new, a, RESUME_STATE=sa, VAL_EVERY="10")
    run(new, b, RESUME_STATE=sb, VAL_EVERY="10", STOP_AT="20")
    mid = torch.load(sb, weights_only=False)
    assert mid["it"] == 20, mid["it"]
    assert [r[0] for r in recs(b)] == [10, 20], recs(b)
    run(new, b, RESUME_STATE=sb, VAL_EVERY="10")  # separate process continues 21..40
    A, Bs = torch.load(sa, weights_only=False), torch.load(sb, weights_only=False)
    assert A["it"] == Bs["it"] == 40
    same("net", A["net"], Bs["net"]) and same("opt", A["opt"], Bs["opt"]) and same("sched", A["sched"], Bs["sched"])
    same("np_rng", A["np_rng"], Bs["np_rng"]) and same("torch_rng", A["torch_rng"], Bs["torch_rng"])
    same("best.pt", torch.load(a), torch.load(b))
    if recs(a) != recs(b):
        fails.append(f"val records differ: {recs(a)} vs {recs(b)}")
    print("resume: records", recs(a))

    old = os.path.join(tmp, "train_old.py")
    with open(old, "wb") as fh:
        fh.write(subprocess.run(["git", "show", f"{OLD_REV}:ml/forecast/train.py"], cwd=REPO, check=True,
                                capture_output=True).stdout)
    o, n = os.path.join(tmp, "o.pt"), os.path.join(tmp, "n.pt")
    run(old, o); run(new, n)
    if recs(o) != recs(n):
        fails.append(f"no-resume log differs from {OLD_REV}: {recs(o)} vs {recs(n)}")
    same("unchanged.best.pt", torch.load(o), torch.load(n))
    assert not os.path.exists(n + ".state")
    print("unchanged: records", recs(n))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

if fails:
    print("FAIL", *fails, sep="\n")
    sys.exit(1)
print("PASS: resumed 20+20 == 40 (net, optimizer, scheduler, RNG, val log); no-resume == " + OLD_REV)
