"""S1-11a CPU checks of the fc_obs table mode (FC_TABLE) and of 3a non-regression.
  PYTHONPATH=<e0frozen>/ml/frozen_site ml/.venv/bin/python -B ml/forecast/tests/test_fc3b.py
1. summarize_probs(softmax) == summarize(logits) bit for bit (3a features unchanged by the refactor).
2. Without a table FcState gives exactly the features of the base branch's fc_obs.py (git show e1-forecast:...).
3. Table mode: prior before the first 07:00, table[row, d] from hour 24d + 7, held in between, last day held.
4. Table.seed_row: eval-pool / train-pool seeds keep their rows, other seeds map to training-pool seed seed % N.
5. FcVecEnv / FcEnv in table mode on the frozen sim: a random reset seed runs the pool seed's weather (obs equal to a
   plain VecEnv reset with the pool seed) and shows the prior."""
import importlib.util, os, subprocess, sys, tempfile
import numpy as np

FC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.abspath(os.path.join(FC, "..", ".."))
sys.path.insert(0, FC)
import torch  # noqa: E402
import fc_obs  # noqa: E402
from fc_obs import FcState, Table, N_FC, IN_STORM, ACT_HOUR, SIGNS  # noqa: E402

fails, passed = [], 0


def ok(name, cond, detail=""):
    global passed
    if cond:
        passed += 1
    else:
        fails.append(f"{name}: {detail}")


def fake_obs(rng, n, ins):
    o = np.zeros((n, 136), np.float32)
    o[:, IN_STORM] = ins
    o[:, SIGNS:SIGNS + 6] = rng.random((n, 6)).astype(np.float32)
    return o


def make_table(tmp, days=120, n=1000):
    rng = np.random.default_rng(1)
    t = np.sort(rng.random((n, days, 5)), -1).astype(np.float32)
    tr = np.concatenate([t, rng.random((n, days, 1)).astype(np.float32)], -1)
    ev = tr[::-1].copy() * 0.5
    p = os.path.join(tmp, "t.npz")
    np.savez(p, train=tr, eval=ev, prior=np.arange(6, dtype=np.float32) / 10, seeds_train=np.arange(500000, 500000 + n),
             seeds_eval=np.arange(300000, 300000 + n), issue_hours=np.arange(days) * 24 + 7)
    return p


def test_summarize():
    g = torch.Generator().manual_seed(0)
    lg = torch.randn(64, 29, generator=g) * 3
    a = fc_obs.summarize(lg)
    b = fc_obs.summarize_probs(torch.softmax(lg, -1).numpy().astype(np.float64))
    ok("summarize_probs == summarize", np.array_equal(a, b))


def test_no_table_unchanged():
    src = subprocess.run(["git", "show", "e1-forecast:ml/forecast/fc_obs.py"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    d = tempfile.mkdtemp()
    p = os.path.join(d, "fc_obs_base.py")
    open(p, "w").write(src)
    spec = importlib.util.spec_from_file_location("fc_obs_base", p)
    base = importlib.util.module_from_spec(spec); sys.path.insert(0, FC); spec.loader.exec_module(base)
    base.GRU_CKPT = fc_obs.GRU_CKPT
    n = 4
    a, b = FcState(n), base.FcState(n)
    a.reset(np.arange(n)); b.reset(np.arange(n))
    rng = np.random.default_rng(2)
    same = np.array_equal(a.feat, b.feat)
    ins = np.zeros(n)
    for h in range(200):
        if h % 37 == 5:
            ins = 1 - ins
        o = fake_obs(rng, n, ins)
        adv = rng.random(n) < 0.8
        a.advance(adv, o); b.advance(adv, o)
        same &= np.array_equal(a.feat, b.feat) and np.array_equal(a.hist, b.hist)
    ok("no table: FcState == base fc_obs.FcState", same)


def test_table_switch(tab):
    n = 3
    rows = np.array([0, 1500, 999])
    st = FcState(n, tab); st.reset(np.arange(n), rows)
    ok("prior at reset", np.array_equal(st.feat, np.repeat(tab.prior[None], n, 0)))
    rng = np.random.default_rng(3)
    good = True
    want = np.repeat(tab.prior[None], n, 0)
    for h in range(1, 24 * 121 + 1):
        st.advance(np.ones(n, bool), fake_obs(rng, n, 0))
        if h % 24 == 7 and (h - 7) // 24 < tab.days:
            want = tab.feat[rows, (h - 7) // 24]
        good &= np.array_equal(st.feat, want)
        if h == 6:
            ok("hour 6 still prior", np.array_equal(st.feat[0], tab.prior))
        if h == 7:
            ok("hour 7 = day 0", np.array_equal(st.feat[1], tab.eval[500, 0]))
        if h == 30:
            ok("hour 30 holds day 0", np.array_equal(st.feat[2], tab.train[999, 0]))
        if h == 31:
            ok("hour 31 = day 1", np.array_equal(st.feat[0], tab.train[0, 1]))
    ok("table switch/hold over 121 days", good)
    ok("last day held after day 119", np.array_equal(st.feat[0], tab.train[0, 119]))
    st.reset(1, 3)
    ok("reset_at one env -> prior", np.array_equal(st.feat[1], tab.prior) and st.hour[1] == 0 and st.row[1] == 3)


def test_seed_row(tab):
    ok("eval seed", tab.seed_row(300007) == (300007, 1007))
    ok("train seed", tab.seed_row(500123) == (500123, 123))
    ok("other seed -> train pool", tab.seed_row(1234567) == (500567, 567))
    ok("small seed -> train pool", tab.seed_row(1001) == (500001, 1))


def test_wrappers(path):
    fc_obs.FC_TABLE = path; fc_obs._TABLE = None
    ls = fc_obs.install()
    sim_data = os.path.join(FC, "data", "sim_339afa0", "data")
    cond = (True, 0.0, True, "HARDCORE", "chaotic", 120, "murky")
    v = ls.VecEnv(2, sim_data, 2, True, False)
    obs, _ = v.reset([1234567, 300004], *cond)
    ref = fc_obs._ORIG_VEC(2, sim_data, 2, True, False)
    r, _ = ref.reset([500567, 300004], *cond)
    ok("FcVecEnv: random seed runs the pool seed", np.array_equal(obs[:, :134], r) and obs.shape[1] == 140)
    ok("FcVecEnv: prior shown", np.array_equal(obs[:, 134:], np.repeat(v.tab.prior[None], 2, 0)))
    ok("FcVecEnv rows", list(v.fc.row) == [567, 1004])
    o1, _ = v.reset_at(0, 987654321, *cond)
    r1, _ = ref.reset_at(0, 500000 + 987654321 % 1000, *cond)
    ok("FcVecEnv.reset_at maps seed", np.array_equal(o1[:134], r1) and v.fc.row[0] == 987654321 % 1000)
    e = ls.Env(sim_data)
    eo, _, _ = e.reset(42, False, 0.0, True, "HARDCORE", "chaotic", True, False, 120, "murky")
    re_ = fc_obs._ORIG_ENV(sim_data)
    ro, _, _ = re_.reset(500042, False, 0.0, True, "HARDCORE", "chaotic", True, False, 120, "murky")
    ok("FcEnv maps seed", np.array_equal(eo[:134], ro) and e.fc.row[0] == 42)


if __name__ == "__main__":
    test_summarize()
    test_no_table_unchanged()
    tmp = tempfile.mkdtemp()
    p = make_table(tmp)
    tab = Table(p)
    test_table_switch(tab)
    test_seed_row(tab)
    test_wrappers(p)
    for f in fails:
        print("FAIL", f)
    print(f"{passed} passed, {len(fails)} failed")
    sys.exit(1 if fails else 0)
