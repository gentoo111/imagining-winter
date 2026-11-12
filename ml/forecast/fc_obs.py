"""RQ2 rung 3 (signs + forecast features): the frozen RQ1 GRU landing-time classifier run online on the precursor
history an agent has seen, appended to the raw-signs observation (134 + 6 = 140 entries).

The 168 h history and the observer context are rebuilt hour by hour from the signs observation itself (pressure,
trend, wind, animals, sky and the in-storm flag are exact integer readouts in the 134-d obs); `--check` verifies that
they equal feats.hist_feats / ctx_feats on the same seeds' weather from weather_gen, so the GRU sees exactly the inputs
it was trained on. Features (6): P(landing within 24 / 48 / 72 / 120 / 168 h) and the expected hours to the next
landing / 240 (no landing within 168 h counted as 240 h), from the 29-class GRU output. The forecast is re-issued every
EVERY game hours (and at hour 0); between issues the last one is shown unchanged.

Frozen rules: every process that uses this module must import lastembers_sim from the frozen 4195eb6 build
(use_frozen() puts ml/frozen_site of the frozen root first on sys.path and asserts it).

  fc_obs.py --check [n_seeds]       reconstruction check against weather_gen + timing
  fc_obs.py --check-table <npz> [n] S1-11a: a gru_daily table == the online GRU at every 07:00 on frozen-sim replays

Table mode (S1-11a, 3b / 3a'): FC_TABLE=<npz> (windows/fc3b.py: keys train/eval (1000, days, 6), prior (6,),
seeds_train, seeds_eval) replaces the online GRU: the forecast shown is `prior` until the first 07:00 and from every
game hour h = 24d + 7 the table's day-d entry, held until the next 07:00. Episode seeds from the evaluation pool use
the eval rows; any other seed (dppo workers draw random ones) is replaced by training-pool seed number seed % 1000 and
the sim is reset with that pool seed, so training weather only comes from the fixed pool. FC_TABLE unset = 3a exactly.
FC_FROZEN_ROOT / FC_FROZEN_SITE override the frozen root / module directory (Windows copies; unset = unchanged).
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.environ.get("FC_FROZEN_ROOT") or os.path.expanduser("~/Documents/Games/last-embers-e0frozen")
FROZEN_SITE = os.environ.get("FC_FROZEN_SITE") or os.path.join(FROZEN, "ml", "frozen_site")
FC_TABLE = os.environ.get("FC_TABLE")              # table mode (S1-11a); unset = online GRU (3a)
ISSUE_H = 7                                        # table mode: daily issue at 07:00 (common.is_sign_hour)
GRU_CKPT = os.path.join(HERE, "runs", "gru16k_endurance_murky.pt")
EVERY = int(os.environ.get("FC_EVERY", "3"))   # re-issue the forecast every EVERY game hours
N_FC = 6
LEADS = (24, 48, 72, 120, 168)
IN_STORM, ACT_HOUR, SIGNS = 4, 122, 128          # obs indices (339afa0 layout): in-storm flag, actions this hour, signs
STORM_H = 48                                     # Endurance recurring storm length (balance.json, 2 days)


def use_frozen():
    if sys.path[0] != FROZEN_SITE:
        sys.path.insert(0, FROZEN_SITE)
    import lastembers_sim
    f = os.path.realpath(lastembers_sim.__file__)
    assert f.startswith(os.path.realpath(FROZEN_SITE)), f"lastembers_sim not frozen: {f}"
    return lastembers_sim


sys.path.insert(0, HERE)
import torch  # noqa: E402
from models import GRUCat  # noqa: E402
from feats import HIST, NFEAT  # noqa: E402
from common import H, NB  # noqa: E402

_NET = None


def gru():
    global _NET
    if _NET is None:
        with torch.random.fork_rng():  # building the net must not consume the caller's RNG (dppo.evaluate seeds it)
            _NET = GRUCat()
        _NET.load_state_dict(torch.load(GRU_CKPT, weights_only=True)); _NET.eval()
    return _NET


def summarize(logits):
    return summarize_probs(torch.softmax(logits, -1).numpy().astype(np.float64))


def summarize_probs(p):
    """(B, NB+1) class probabilities (float64) -> (B, 6) features."""
    cum = np.cumsum(p[:, :NB], 1)                      # bin b = landing in hours 6b+1 .. 6b+6
    out = np.zeros((len(p), N_FC), np.float32)
    for j, L in enumerate(LEADS):
        out[:, j] = cum[:, L // 6 - 1]
    mid = np.arange(NB) * 6 + 3.5
    out[:, 5] = ((p[:, :NB] * mid).sum(1) + p[:, NB] * 240.0) / 240.0
    return out


class Table:
    """S1-11a forecast table (FC_TABLE npz): per-seed daily 07:00 features of one forecaster."""

    def __init__(self, path):
        z = np.load(path)
        self.path = os.path.abspath(path)
        self.train, self.eval, self.prior = z["train"].astype(np.float32), z["eval"].astype(np.float32), z["prior"].astype(np.float32)
        self.seeds_train, self.seeds_eval = [int(s) for s in z["seeds_train"]], [int(s) for s in z["seeds_eval"]]
        assert self.train.shape == (len(self.seeds_train), self.train.shape[1], N_FC) and self.eval.shape[0] == len(self.seeds_eval)
        self.days = self.train.shape[1]
        self.feat = np.concatenate([self.train, self.eval])          # row r: train rows first, then eval rows
        self._row = {s: i for i, s in enumerate(self.seeds_train)}
        self._row.update({s: len(self.seeds_train) + i for i, s in enumerate(self.seeds_eval)})

    def seed_row(self, seed):
        """(sim seed, table row): pool seeds keep theirs; any other seed -> training-pool seed number seed % N."""
        seed = int(seed)
        if seed in self._row:
            return seed, self._row[seed]
        s = self.seeds_train[seed % len(self.seeds_train)]
        return s, self._row[s]


_TABLE = None


def table():
    global _TABLE
    if _TABLE is None and FC_TABLE:
        _TABLE = Table(FC_TABLE)
    return _TABLE


class FcState:
    """Per-env precursor history (rolling 168 h) and observer context, rebuilt from the signs observation.
    With a Table (table mode) the features come from the table instead: prior until 07:00, then the day's entry."""

    def __init__(self, n, tab=None):
        self.n = n
        self.tab = tab
        self.row = np.zeros(n, int)
        self.hist = np.zeros((n, HIST, NFEAT), np.float32)
        self.hour = np.zeros(n, int); self.ins = np.zeros(n, bool)
        self.last_end = np.full(n, -10**6); self.land_h = np.full(n, -10**6)
        self.feat = np.zeros((n, N_FC), np.float32)
        self.prior = None

    def ctx(self, i):
        h = self.hour[i]
        stormed = self.last_end[i] > -10**5
        since = h - self.last_end[i]
        left = (self.land_h[i] + STORM_H - h) if self.ins[i] else 0.0
        return np.array([h / H, min(since, 480) / 240.0 if stormed else -1.0, 1.0 if stormed else 0.0,
                         float(self.ins[i]), left / 120.0, 0.0], np.float32)

    def reset(self, idx, rows=None):
        idx = np.atleast_1d(idx)
        self.hist[idx] = 0; self.hour[idx] = 0; self.ins[idx] = False
        self.last_end[idx] = -10**6; self.land_h[idx] = -10**6
        if self.tab is not None:
            assert rows is not None, "table mode needs the table rows of the reset envs"
            self.row[idx] = rows
            self.feat[idx] = self.tab.prior
            return
        if self.prior is None:  # hour 0: empty history, same for every seed
            with torch.inference_mode():
                self.prior = summarize(gru()(torch.zeros(1, HIST, NFEAT), torch.tensor(self.ctx(idx[0])[None])))[0]
        self.feat[idx] = self.prior

    def advance(self, adv, obs):
        """adv: bool (n,) envs whose game hour advanced in this step; obs: their new observation (>= 134 entries)."""
        idx = np.nonzero(adv)[0]
        if not len(idx):
            return
        o = obs[idx].astype(np.float64)
        ins = o[:, IN_STORM] > 0.5
        self.hour[idx] += 1
        h = self.hour[idx]
        if self.tab is not None:  # table mode: switch at 07:00 to the day's entry, hold otherwise; no history needed
            due = (h % 24 == ISSUE_H) & ((h - ISSUE_H) // 24 < self.tab.days)
            if due.any():
                self.feat[idx[due]] = self.tab.feat[self.row[idx[due]], (h[due] - ISSUE_H) // 24]
            return
        landed = ins & ~self.ins[idx]
        ended = ~ins & self.ins[idx]
        self.land_h[idx] = np.where(landed, h, self.land_h[idx])
        self.last_end[idx] = np.where(ended, h, self.last_end[idx])
        self.ins[idx] = ins
        row = np.zeros((len(idx), NFEAT), np.float32)
        row[:, 0] = np.rint(o[:, SIGNS] * 300) / 100.0
        row[:, 1] = np.rint(o[:, SIGNS + 1] * 300) / 100.0
        row[:, 2] = np.rint(o[:, SIGNS + 3] * 25) / 10.0
        row[:, 3] = np.rint(o[:, SIGNS + 4] * 2) / 2.0
        row[:, 4] = np.rint(o[:, SIGNS + 5] * 3) / 3.0
        row[:, 5] = ins; row[:, 6] = landed; row[:, 7] = 1.0
        row[np.arange(len(idx)), 8 + np.rint(o[:, SIGNS + 2] * 7).astype(int)] = 1.0
        self.hist[idx] = np.roll(self.hist[idx], -1, 1)
        self.hist[idx, -1] = row
        due = idx[(h % EVERY) == 0]
        if len(due):
            c = np.stack([self.ctx(i) for i in due])
            with torch.inference_mode():
                self.feat[due] = summarize(gru()(torch.from_numpy(self.hist[due]), torch.from_numpy(c)))


def _sim():
    return use_frozen()


class FcVecEnv:
    """lastembers_sim.VecEnv with the forecast features appended (training workers; needs precursor_obs)."""

    def __init__(self, n, data, threads, precursor_obs, oracle_obs):
        assert precursor_obs and not oracle_obs, "rung 3 = signs + forecast"
        self.v = _ORIG_VEC(n, data, threads, precursor_obs, oracle_obs)
        self.n = n
        self.obs_size = self.v.obs_size + N_FC
        self.n_actions, self.action_names = self.v.n_actions, self.v.action_names
        self.tab = table()
        self.fc = FcState(n, self.tab)

    def _cat(self, obs, idx=None):
        f = self.fc.feat if idx is None else self.fc.feat[idx]
        return np.concatenate([obs, f], -1).astype(np.float32)

    def reset(self, seeds, *a):
        rows = None
        if self.tab is not None:
            seeds, rows = map(list, zip(*[self.tab.seed_row(s) for s in seeds]))
        obs, mask = self.v.reset(seeds, *a)
        self.fc.reset(np.arange(self.n), rows)
        return self._cat(obs), mask

    def reset_at(self, i, seed, *a):
        row = None
        if self.tab is not None:
            seed, row = self.tab.seed_row(seed)
        obs, mask = self.v.reset_at(i, seed, *a)
        self.fc.reset(i, row)
        return self._cat(obs, i), mask

    def step(self, acts):
        obs, mask, rew, done, day, alive = self.v.step(acts)
        self.fc.advance((obs[:, ACT_HOUR] == 0) & ~done, obs)
        return self._cat(obs), mask, rew, done, day, alive


class FcEnv:
    """lastembers_sim.Env with the forecast features appended (the learner's periodic evaluation, env.LastEmbersEnv)."""

    def __init__(self, data):
        self.e = _ORIG_ENV(data)
        self.action_names = self.e.action_names
        self.tab = table()
        self.fc = FcState(1, self.tab)
        self.obs_size = self.e.obs_size + N_FC  # the base size here; LastEmbersEnv only reads action names / bounds

    def reset(self, seed, perturb, warm, no_floor, diff, weather, precursor_obs, oracle_obs, max_days, noise):
        assert precursor_obs and not oracle_obs
        row = None
        if self.tab is not None:
            seed, row = self.tab.seed_row(seed)
        obs, mask, info = self.e.reset(seed, perturb, warm, no_floor, diff, weather, precursor_obs, oracle_obs, max_days, noise)
        self.fc.reset(0, row)
        return np.concatenate([obs, self.fc.feat[0]]).astype(np.float32), mask, info

    def step(self, a):
        obs, mask, rew, done, info = self.e.step(a)
        if not done:
            self.fc.advance(np.array([obs[ACT_HOUR] == 0]), np.asarray(obs)[None])
        return np.concatenate([obs, self.fc.feat[0]]).astype(np.float32), mask, rew, done, info


_ORIG_VEC = _ORIG_ENV = None


def install():
    """Replace lastembers_sim.VecEnv / Env by the forecast wrappers (in this process)."""
    global _ORIG_VEC, _ORIG_ENV
    ls = use_frozen()
    if _ORIG_VEC is None:
        _ORIG_VEC, _ORIG_ENV = ls.VecEnv, ls.Env
        ls.VecEnv, ls.Env = FcVecEnv, FcEnv
    return ls


def check(n_seeds=6):
    """Rebuilt history/context == feats.hist_feats/ctx_feats on weather_gen data for the same seeds; timing."""
    import subprocess, time, tempfile
    from feats import hist_feats, ctx_feats
    ls = use_frozen()
    sim_data = os.path.join(HERE, "data", "sim_339afa0", "data")
    seeds = list(range(300000, 300000 + n_seeds))
    tmp = os.path.join(tempfile.mkdtemp(), "w.bin")
    subprocess.run([os.path.join(HERE, "..", "..", "simrs", "target", "release", "weather_gen"), sim_data, tmp,
                    str(seeds[0]), str(n_seeds), "120", "endurance", "murky"], check=True)
    D = np.fromfile(tmp, dtype="<i4").reshape(n_seeds, H, 15).astype(np.int64)
    v = ls.VecEnv(n_seeds, sim_data, 2, True, True)
    obs, mask = v.reset(seeds, False, 0.0, True, "HARDCORE", "chaotic", 120, "murky")
    st = FcState(n_seeds); st.reset(np.arange(n_seeds))
    done = np.zeros(n_seeds, bool); bad = 0; checked = 0
    global EVERY
    ev, EVERY = EVERY, 1
    t_gru = 0.0; t0 = time.time(); steps = 0
    from dppo import Policy
    net = Policy(134, v.n_actions)
    net.load_state_dict(torch.load(os.path.join(FROZEN, "ml", "runs", "e0-signs-s2", sorted(
        f for f in os.listdir(os.path.join(FROZEN, "ml", "runs", "e0-signs-s2")) if f.startswith("model_"))[-1]), weights_only=True))
    while not done.all():
        with torch.no_grad():
            dist, _ = net(torch.tensor(obs[:, :134]), torch.tensor(mask))
        acts = dist.probs.argmax(-1).numpy().astype(np.int64); acts[done] = 0
        obs, mask, rew, d2, day, alive = v.step(acts); steps += 1
        adv = (obs[:, ACT_HOUR] == 0) & ~done
        t1 = time.time(); st.advance(adv, obs); t_gru += time.time() - t1
        for i in np.nonzero(adv)[0]:
            h = st.hour[i]
            if h < 1 or h > H:
                continue
            hf = hist_feats(D[i], h); cf = ctx_feats(D[i], h, "endurance")
            ok = np.allclose(hf, st.hist[i], atol=1e-6) and np.allclose(cf, st.ctx(i), atol=1e-6)
            # the oracle entry must agree too (same weather): hours to the next landing at this hour
            checked += 1
            if not ok:
                bad += 1
                if bad <= 5:
                    j = np.nonzero(~np.isclose(hf, st.hist[i]).all(1))[0]
                    print("mismatch seed", seeds[i], "hour", h, "hist rows", j[-3:], hf[j[-1]] if len(j) else None,
                          st.hist[i][j[-1]] if len(j) else None, "ctx", cf, st.ctx(i))
        done |= d2
    EVERY = ev
    print(f"checked {checked} hours of {n_seeds} episodes (days {day.tolist()}): {bad} mismatches; "
          f"{steps} vec-steps, {time.time() - t0:.1f}s total, forecast {t_gru:.1f}s (EVERY=1)")
    return bad == 0


def _policy(obs_dim, n_actions):
    from dppo import Policy
    net = Policy(obs_dim, n_actions)
    d = os.path.join(FROZEN, "ml", "runs", "e0-signs-s2")
    net.load_state_dict(torch.load(os.path.join(d, sorted(f for f in os.listdir(d) if f.startswith("model_"))[-1]), weights_only=True))
    return net


def check_table(path, n_seeds=6, atol=1e-5):
    """S1-11a: replay n evaluation-pool seeds (perturb off) and n//3 training-pool seeds (perturb on) in the frozen sim
    (greedy e0-signs-s2 policy, as check()); at every 07:00 the online GRU (EVERY=1) must equal the table entry, and a
    table-mode FcState must show prior before the first 07:00, the day's entry from 07:00 and hold it otherwise."""
    global EVERY
    ls = use_frozen()
    tab = Table(path)
    sim_data = os.path.join(HERE, "data", "sim_339afa0", "data")
    ev, EVERY = EVERY, 1
    bad = issues = held = 0; maxdiff = 0.0
    for seeds, perturb in ((tab.seeds_eval[:n_seeds], False), (tab.seeds_train[:max(1, n_seeds // 3)], True)):
        k = len(seeds)
        v = ls.VecEnv(k, sim_data, 2, True, True)
        obs, mask = v.reset(list(seeds), perturb, 0.0, True, "HARDCORE", "chaotic", 120, "murky")
        rows = np.array([tab.seed_row(s)[1] for s in seeds])
        on = FcState(k); on.reset(np.arange(k))
        ts = FcState(k, tab); ts.reset(np.arange(k), rows)
        want = np.repeat(tab.prior[None], k, 0)
        net = _policy(134, v.n_actions)
        done = np.zeros(k, bool)
        while not done.all():
            with torch.no_grad():
                dist, _ = net(torch.tensor(obs[:, :134]), torch.tensor(mask))
            acts = dist.probs.argmax(-1).numpy().astype(np.int64); acts[done] = 0
            obs, mask, rew, d2, day, alive = v.step(acts)
            adv = (obs[:, ACT_HOUR] == 0) & ~done
            on.advance(adv, obs); ts.advance(adv, obs)
            for i in np.nonzero(adv)[0]:
                h = on.hour[i]
                if h % 24 == ISSUE_H and (h - ISSUE_H) // 24 < tab.days:
                    want[i] = tab.feat[rows[i], (h - ISSUE_H) // 24]
                    diff = float(np.abs(on.feat[i] - want[i]).max()); maxdiff = max(maxdiff, diff); issues += 1
                    if diff > atol:
                        bad += 1
                        if bad <= 5:
                            print("online GRU != table: seed", seeds[i], "hour", h, on.feat[i], want[i])
                else:
                    held += 1
                if not np.array_equal(ts.feat[i], want[i]):
                    bad += 1
                    if bad <= 5:
                        print("table-mode FcState wrong: seed", seeds[i], "hour", h, ts.feat[i], want[i])
            done |= d2
        print(f"seeds {list(seeds)} perturb={perturb}: days {day.tolist()}", flush=True)
    EVERY = ev
    print(f"check-table {path}: {issues} issues (07:00) compared, max |online - table| {maxdiff:.2e}, "
          f"{held} held hours checked, {bad} mismatches")
    return bad == 0 and issues > 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check-table":
        sys.path.insert(1, os.path.join(FROZEN, "ml"))
        ok = check_table(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6)
        sys.exit(0 if ok else 1)
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        sys.path.insert(1, os.path.join(FROZEN, "ml"))
        ok = check(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
        sys.exit(0 if ok else 1)
