"""Multi-host PPO with action masking: rollout workers on every host, one learner.

Each round the learner sends the current weights to every worker. Each worker plays
its local sim servers for a fixed wall-clock budget with that policy (so a slow host
just contributes fewer steps instead of holding everyone back), then returns the
segment. The learner computes GAE per segment and runs a standard clipped PPO update.

  worker:  python dppo.py worker --port 17500 --envs 4 --godot <path> --root <game dir>
           python dppo.py worker --port 17500 --envs 64 --root <game dir> --backend rust [--threads 8]
  learner: python dppo.py learn --workers local:17500,o:17600,w:17700 --resume <sb3 zip> --out runs/stage3
           (a worker entry host:port:envs:rust runs that worker on the in-process Rust sim, simrs/;
            --eval-backend rust evaluates on it too; results are identical to the Godot server)

Network layout mirrors SB3's MlpPolicy (64-64 tanh, separate policy/value nets), so a
MaskablePPO checkpoint from stages 1-2 loads directly.
"""
import argparse, io, json, os, pickle, socket, struct, subprocess, sys, threading, time
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- model

class Policy(nn.Module):
    def __init__(self, obs_dim, n_act, hidden=64):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.vf = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.action_net = nn.Linear(hidden, n_act)
        self.value_net = nn.Linear(hidden, 1)

    def forward(self, obs, mask):
        logits = self.action_net(self.pi(obs))
        logits = torch.where(mask, logits, torch.full_like(logits, -1e8))
        return torch.distributions.Categorical(logits=logits), self.value_net(self.vf(obs)).squeeze(-1)

    def load_sb3(self, zip_path):
        from sb3_contrib import MaskablePPO
        sd = MaskablePPO.load(zip_path, device="cpu").policy.state_dict()
        m = {"pi.0": "mlp_extractor.policy_net.0", "pi.2": "mlp_extractor.policy_net.2",
             "vf.0": "mlp_extractor.value_net.0", "vf.2": "mlp_extractor.value_net.2",
             "action_net": "action_net", "value_net": "value_net"}
        own = {}
        for k, v in m.items():
            own[k + ".weight"] = sd[v + ".weight"]
            own[k + ".bias"] = sd[v + ".bias"]
        self.load_state_dict(own)


# Action-table changes, newest last: (new action names, new observation tail entries).
# 2026-10-29: generator overdrive + its sacrifice event, two FP techs (tail: overdrive, pressure, scenario_won).
# 2026-10-29b: FP Infirmary (building, its job, its tech).
ACTION_CHANGES = [
    (["overdrive:on", "overdrive:off", "research:overdrive_couplings",
      "research:generator_safety_bypass", "choice:sacrifice_core", "choice:sacrifice_child"], 3),
    (["build:infirmary", "add:infirmary", "remove:infirmary", "research:infirmary"], 0),
]
NEW_ACTION_BIAS = -5.0
OBS_TAIL_AFTER_LAWS = 7  # teams, incoming, visited, choice pending, actions this hour, hope, discontent


def _obs_layout(names, obs_size, tail):
    """Observation index per name: [prefix] + 2 per job + 2 per building + tech + law + tail."""
    jobs = [n[4:] for n in names if n.startswith("add:")]
    builds = [n[6:] for n in names if n.startswith("build:")]
    techs = [n for n in names if n.startswith("research:")]
    laws = [n for n in names if n.startswith("law:")]
    keys = []
    keys += [("job", j, k) for j in jobs for k in (0, 1)]
    keys += [("build", j, k) for j in builds for k in (0, 1)]
    keys += [("tech", t) for t in techs] + [("law", l) for l in laws]
    prefix = obs_size - tail - len(keys)
    return ([("prefix", i) for i in range(prefix)] + keys + [("tail", i) for i in range(tail)])


def load_resized(model, path, action_names):
    """Load a checkpoint into a model whose action list / observation grew.
    Old weights are copied to their (by-name) positions; columns for new inputs and
    rows for new actions start at zero, with the new actions' bias at NEW_ACTION_BIAS
    so the old policy is nearly unchanged at first but still explores them (bias 0
    made it pick overdrive often enough to blow the furnace: eval 42 -> 26 days)."""
    sd = torch.load(path, weights_only=True)
    own = model.state_dict()
    if all(sd[k].shape == own[k].shape for k in own):
        model.load_state_dict(sd)
        return
    if (sd["action_net.weight"].shape == own["action_net.weight"].shape
            and sd["pi.0.weight"].shape[1] < own["pi.0.weight"].shape[1]
            and all(sd[k].shape == own[k].shape for k in own if k not in ("pi.0.weight", "vf.0.weight"))):
        # same actions, extra observation entries appended at the end (--precursor-obs, then
        # --oracle-obs): their input columns start at zero. Assumes the checkpoint's own extras
        # are a prefix of the new ones (e.g. none -> precursor, precursor -> precursor+oracle).
        n_old = sd["pi.0.weight"].shape[1]
        new = dict(sd)
        for k in ("pi.0.weight", "vf.0.weight"):
            new[k] = torch.zeros_like(own[k]); new[k][:, :n_old] = sd[k]
        model.load_state_dict(new)
        print(json.dumps({"resized_from": path, "obs": [n_old, own["pi.0.weight"].shape[1]], "actions": "same"}), flush=True)
        return
    n_old_act, n_old_obs = sd["action_net.weight"].shape[0], sd["pi.0.weight"].shape[1]
    obs_new = own["pi.0.weight"].shape[1]
    tail_new = OBS_TAIL_AFTER_LAWS + sum(t for _, t in ACTION_CHANGES)
    removed, tail_old = set(), tail_new
    for added, tail in reversed(ACTION_CHANGES):  # undo changes newest first until the checkpoint fits
        removed |= set(added); tail_old -= tail
        old_names = [n for n in action_names if n not in removed]
        if len(old_names) == n_old_act:
            break
    assert len(old_names) == n_old_act, "checkpoint action list does not match"
    act_idx = [action_names.index(n) for n in old_names]
    lay_new = _obs_layout(action_names, obs_new, tail_new)
    lay_old = _obs_layout(old_names, n_old_obs, tail_old)
    # the newest tail entries are appended after the older ones
    pos_new = {k: i for i, k in enumerate(lay_new)}
    obs_idx = [pos_new[k] for k in lay_old]
    assert lay_old[0] == ("prefix", 0) and len(obs_idx) == n_old_obs, "checkpoint observation size does not match"
    new = {k: torch.zeros_like(v) for k, v in own.items()}
    for k in own:
        if k in ("pi.0.weight", "vf.0.weight"):
            new[k][:, obs_idx] = sd[k]
        elif k.startswith("action_net."):
            new[k][act_idx] = sd[k]
            if k == "action_net.bias":
                new_rows = [i for i in range(len(action_names)) if i not in act_idx]
                new[k][new_rows] = NEW_ACTION_BIAS
        else:
            new[k] = sd[k]
    model.load_state_dict(new)
    print(json.dumps({"resized_from": path, "actions": [len(old_names), len(action_names)],
                      "obs": [len(obs_idx), obs_new]}), flush=True)


# ---------------------------------------------------------------- wire format

def send_msg(sock, obj):
    b = pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack("!Q", len(b)) + b)


def recv_msg(sock):
    def exact(n):
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(min(1 << 20, n - len(buf)))
            if not chunk:
                raise ConnectionError("peer closed")
            buf += chunk
        return bytes(buf)
    (n,) = struct.unpack("!Q", exact(8))
    return pickle.loads(exact(n))


# ---------------------------------------------------------------- sim server (TCP mode, works on every OS)

class Sim:
    def __init__(self, godot, root, port, seed_base):
        self.p = subprocess.Popen([godot, "--headless", "--path", root, "-s", "res://tools/rl_server.gd", "--", "--port", str(port)],
                                  stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(600):
            try:
                self.s = socket.create_connection(("127.0.0.1", port), timeout=5)
                break
            except OSError:
                time.sleep(0.2)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.f = self.s.makefile("r", encoding="utf-8")
        self.rng = np.random.default_rng(seed_base)
        spec = self.call({"cmd": "spec"})
        self.n_act, self.obs_dim = len(spec["actions"]), spec["obs_size"]
        self.action_names = spec["actions"]

    def call(self, c):
        self.s.sendall((json.dumps(c) + "\n").encode())
        return json.loads(self.f.readline())

    warm = 0.0
    difficulty = "STANDARD"
    extra = {}  # non-default reset options: weather, weather_noise, precursor_obs, oracle_obs, max_days

    def reset(self):
        self.seed = int(self.rng.integers(1, 2**31 - 1))
        self.actions = []
        return self.call({"cmd": "reset", "seed": self.seed, "perturb": True, "warm": Sim.warm, "difficulty": Sim.difficulty,
                          **Sim.extra})

    def close(self):
        try:
            self.call({"cmd": "quit"})
        except Exception:
            pass
        self.p.kill()


# ---------------------------------------------------------------- worker

def reset_extra(weather, max_days, precursor_obs, oracle_obs, weather_noise=""):
    """Non-default reset options (older commands send none, so they reproduce)."""
    o = {}
    if weather != "scripted":
        o["weather"] = weather
    if weather_noise:
        o["weather_noise"] = weather_noise
    if precursor_obs:
        o["precursor_obs"] = True
    if oracle_obs:
        o["oracle_obs"] = True
    if max_days:
        o["max_days"] = int(max_days)
    return o


def worker(a):
    if os.name == "nt":  # below-normal priority; the Godot children inherit it
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    Sim.extra = reset_extra("scripted", 0, a.precursor_obs, a.oracle_obs)  # obs size is fixed per worker
    sims = [Sim(a.godot, a.root, a.port + 1 + i, seed_base=a.port * 100 + i + int(time.time())) for i in range(a.envs)]
    obs_dim, n_act = sims[0].obs_dim, sims[0].n_act
    model = Policy(obs_dim, n_act)
    torch.set_num_threads(1)
    cur = [s.reset() for s in sims]
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", a.port)); srv.listen(1)
    print(f"worker ready {a.port} envs={a.envs}", flush=True)
    conn, _ = srv.accept()
    send_msg(conn, {"obs_dim": obs_dim, "n_act": n_act, "envs": a.envs, "actions": sims[0].action_names})
    while True:
        msg = recv_msg(conn)
        if msg.get("quit"):
            break
        model.load_state_dict(torch.load(io.BytesIO(msg["weights"]), weights_only=True))
        Sim.warm = float(msg.get("warm", 0.0))
        Sim.difficulty = str(msg.get("difficulty", "STANDARD"))
        ex = reset_extra(str(msg.get("weather", "scripted")), int(msg.get("max_days", 0)), a.precursor_obs, a.oracle_obs,
                         str(msg.get("weather_noise", "")))
        if ex != Sim.extra:  # weather / episode length changed: restart every episode under it
            Sim.extra = ex
            cur = [s.reset() for s in sims]
        budget, t0 = float(msg["seconds"]), time.time()
        segs = [{k: [] for k in ("obs", "mask", "act", "logp", "val", "rew", "done")} for _ in sims]
        episodes = []
        while time.time() - t0 < budget:
            obs = torch.tensor(np.array([np.clip(c["obs"], -5, 5) for c in cur], np.float32))
            mask = torch.tensor(np.array([c["mask"] for c in cur], bool))
            with torch.no_grad():
                dist, val = model(obs, mask)
                act = dist.sample()
                logp = dist.log_prob(act)
            for i, s in enumerate(sims):
                r = s.call({"cmd": "step", "action": int(act[i])})
                s.actions.append(int(act[i]))
                g = segs[i]
                g["obs"].append(obs[i].numpy()); g["mask"].append(mask[i].numpy()); g["act"].append(int(act[i]))
                g["logp"].append(float(logp[i])); g["val"].append(float(val[i])); g["rew"].append(float(r["reward"]))
                g["done"].append(bool(r["done"]))
                if r["done"]:
                    episodes.append({"seed": s.seed, "day": r["info"]["day"], "alive": r["info"]["alive"], "steps": len(s.actions)})
                    r = s.reset()
                cur[i] = r
        # bootstrap value for each unfinished segment end
        obs = torch.tensor(np.array([np.clip(c["obs"], -5, 5) for c in cur], np.float32))
        mask = torch.tensor(np.array([c["mask"] for c in cur], bool))
        with torch.no_grad():
            _, last_val = model(obs, mask)
        out = []
        for i, g in enumerate(segs):
            if not g["act"]:
                continue
            out.append({"obs": np.array(g["obs"], np.float32), "mask": np.array(g["mask"], bool),
                        "act": np.array(g["act"], np.int64), "logp": np.array(g["logp"], np.float32),
                        "val": np.array(g["val"], np.float32), "rew": np.array(g["rew"], np.float32),
                        "done": np.array(g["done"], bool), "last_val": float(last_val[i])})
        send_msg(conn, {"segments": out, "episodes": episodes, "secs": time.time() - t0})
    for s in sims:
        s.close()


def worker_rust(a):
    """Same protocol and segments as worker(), but all envs live in one in-process
    Rust VecEnv (lastembers_sim) stepped in parallel; no Godot servers."""
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    import lastembers_sim
    venv = lastembers_sim.VecEnv(a.envs, os.path.join(a.root, "data"), a.threads, a.precursor_obs, a.oracle_obs)
    obs_dim, n_act, n = venv.obs_size, venv.n_actions, a.envs
    rng = np.random.default_rng(a.port * 100 + int(time.time()))
    new_seed = lambda: int(rng.integers(1, 2**31 - 1))
    warm, diff, weather, max_days, noise = 0.0, "STANDARD", "scripted", 0, ""
    seeds = [new_seed() for _ in range(n)]
    steps_taken = [0] * n
    obs, mask = venv.reset(seeds, True, warm)
    model = Policy(obs_dim, n_act)
    torch.set_num_threads(1)
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", a.port)); srv.listen(1)
    print(f"worker ready {a.port} envs={n} backend=rust", flush=True)
    conn, _ = srv.accept()
    send_msg(conn, {"obs_dim": obs_dim, "n_act": n_act, "envs": n, "actions": list(venv.action_names)})
    while True:
        msg = recv_msg(conn)
        if msg.get("quit"):
            break
        model.load_state_dict(torch.load(io.BytesIO(msg["weights"]), weights_only=True))
        warm = float(msg.get("warm", 0.0))
        nd = str(msg.get("difficulty", "STANDARD"))
        nw, nm, nn = str(msg.get("weather", "scripted")), int(msg.get("max_days", 0)), str(msg.get("weather_noise", ""))
        if (nd, nw, nm, nn) != (diff, weather, max_days, noise):  # restart all episodes under the new difficulty / weather
            diff, weather, max_days, noise = nd, nw, nm, nn
            seeds = [new_seed() for _ in range(n)]; steps_taken = [0] * n
            obs, mask = venv.reset(seeds, True, warm, True, diff, weather, max_days, noise or "murky")
        budget, t0 = float(msg["seconds"]), time.time()
        cols = {k: [] for k in ("obs", "mask", "act", "logp", "val", "rew", "done")}
        episodes = []
        while time.time() - t0 < budget:
            o = torch.tensor(np.clip(obs, -5, 5)); m = torch.tensor(mask)
            with torch.no_grad():
                dist, val = model(o, m)
                act = dist.sample()
                logp = dist.log_prob(act)
            obs2, mask2, rew, done, day, alive = venv.step(act.numpy().astype(np.int64))
            cols["obs"].append(o.numpy()); cols["mask"].append(mask); cols["act"].append(act.numpy())
            cols["logp"].append(logp.numpy()); cols["val"].append(val.numpy()); cols["rew"].append(rew.astype(np.float32))
            cols["done"].append(done)
            for i in range(n):
                steps_taken[i] += 1
            for i in np.flatnonzero(done):
                episodes.append({"seed": seeds[i], "day": int(day[i]), "alive": int(alive[i]), "steps": steps_taken[i]})
                seeds[i], steps_taken[i] = new_seed(), 0
                obs2[i], mask2[i] = venv.reset_at(int(i), seeds[i], True, warm, True, diff, weather, max_days, noise or "murky")
            obs, mask = obs2, mask2
        with torch.no_grad():
            _, last_val = model(torch.tensor(np.clip(obs, -5, 5)), torch.tensor(mask))
        out = []
        if cols["act"]:
            st = {k: np.stack(v, 1) for k, v in cols.items()}  # [env, time, ...]
            for i in range(n):
                out.append({"obs": st["obs"][i].astype(np.float32), "mask": st["mask"][i].astype(bool),
                            "act": st["act"][i].astype(np.int64), "logp": st["logp"][i].astype(np.float32),
                            "val": st["val"][i].astype(np.float32), "rew": st["rew"][i].astype(np.float32),
                            "done": st["done"][i].astype(bool), "last_val": float(last_val[i])})
        send_msg(conn, {"segments": out, "episodes": episodes, "secs": time.time() - t0})


# ---------------------------------------------------------------- learner

REMOTE = {
    "o": "cd ~/Games/last-embers/ml && nice -n 15 .venv/bin/python dppo.py worker --port {port} --envs {envs} --godot /opt/homebrew/bin/godot --root ~/Games/last-embers",
    "w": r'D:\LastEmbers\venv\Scripts\python.exe D:\LastEmbers\game\ml\dppo.py worker --port {port} --envs {envs} --godot D:\LastEmbers\bin\Godot_v4.7.2-stable_win64_console.exe --root D:\LastEmbers\game',
}


class Link:
    """One worker: started locally or over ssh -L, then driven over TCP."""

    def __init__(self, host, port, envs, backend="godot", obs_flags=()):
        self.host, self.port, self.envs, self.backend, self.obs_flags = host, port, envs, backend, list(obs_flags)
        self.start()

    def start(self):
        host, port, envs = self.host, self.port, self.envs
        py = os.path.join(HERE, ".venv", "bin", "python")
        extra = (["--backend", "rust"] if self.backend == "rust" else []) + self.obs_flags
        if host == "local":
            cmd = ["nice", "-n", "10", py, os.path.join(HERE, "dppo.py"), "worker", "--port", str(port), "--envs", str(envs),
                   "--godot", "godot", "--root", os.path.dirname(HERE)] + extra
        else:
            cmd = ["ssh", "-o", "ExitOnForwardFailure=yes", "-L", f"{port}:127.0.0.1:{port}", host,
                   REMOTE[host].format(port=port, envs=envs) + " ".join([""] + extra)]
        self.p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(900):
            try:
                self.s = socket.create_connection(("127.0.0.1", port), timeout=10)
                self.info = recv_msg(self.s)
                self.s.settimeout(None)  # rounds take a minute; the connect timeout must not apply
                return
            except (OSError, ConnectionError, struct.error):
                time.sleep(0.5)
        raise RuntimeError(f"worker {host}:{port} did not start")

    def restart(self):
        """A home machine can sleep, drop off the network or kill the worker: start a fresh one."""
        try:
            self.s.close()
        except Exception:
            pass
        self.p.kill()
        self.port += 50  # fresh ports, in case the old server is still holding them
        self.start()


def gae(seg, gamma, lam):
    rew, val, done = seg["rew"], seg["val"], seg["done"]
    n = len(rew)
    adv = np.zeros(n, np.float32)
    last = 0.0
    for t in reversed(range(n)):
        nxt = seg["last_val"] if t == n - 1 else val[t + 1]
        nonterm = 0.0 if done[t] else 1.0
        delta = rew[t] + gamma * nxt * nonterm - val[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + val


def evaluate(model, out_dir, t, seeds=(1001, 1002, 1003, 1004, 1005), backend="godot", difficulty="STANDARD",
             weather="scripted", precursor_obs=False, oracle_obs=False, max_days=0, weather_noise=""):
    sys.path.insert(0, HERE)
    from env import LastEmbersEnv
    recs = []
    torch.manual_seed(t)  # sampled like training, but reproducible; replays store actions anyway
    env = LastEmbersEnv(perturb=False, seeds=list(seeds), record=recs.append, backend=backend, difficulty=difficulty,
                        weather=weather, precursor_obs=precursor_obs, oracle_obs=oracle_obs, max_days=max_days,
                        weather_noise=weather_noise)
    for _ in seeds:
        obs, _ = env.reset(); done = False
        while not done:
            with torch.no_grad():
                dist, _ = model(torch.tensor(obs)[None], torch.tensor(env.action_masks())[None])
            obs, _, done, _, _ = env.step(int(dist.sample()))
    env.close()
    line = {"t": t, "time": time.time(), "mean_day": float(np.mean([r["day"] for r in recs])),
            "difficulty": difficulty, "days": [r["day"] for r in recs], "alive": [r["alive"] for r in recs]}
    if weather != "scripted" or max_days:
        line.update(weather=weather, max_days=max_days)
    if weather_noise:
        line.update(weather_noise=weather_noise)
    print(json.dumps(line), flush=True)
    with open(os.path.join(out_dir, "eval.jsonl"), "a") as f:
        f.write(json.dumps(line) + "\n")
    if LIVE:
        LIVE.metric("eval_mean_day", line["mean_day"], t)
        LIVE.metric("eval_min_day", min(line["days"]), t)
        LIVE.tag("difficulty", difficulty)
    with open(os.path.join(out_dir, f"replays_{t:09d}.json"), "w") as f:
        json.dump(recs, f)
    torch.save(model.state_dict(), os.path.join(out_dir, f"model_{t:09d}.pt"))


LIVE = None  # live MLflow logger (ml/mlflow_live.py); training never depends on it


def learn(a):
    global LIVE
    os.makedirs(a.out, exist_ok=True)
    if not a.no_mlflow:
        try:
            from mlflow_live import LiveLogger
            LIVE = LiveLogger(a.out, {k: v for k, v in vars(a).items() if k != "func"})
        except Exception as e:
            print(json.dumps({"mlflow_live_disabled": repr(e)}), flush=True)
    links = []
    obs_flags = (["--precursor-obs"] if a.precursor_obs else []) + (["--oracle-obs"] if a.oracle_obs else [])
    for part in a.workers.split(","):
        host, port, envs, *backend = part.split(":")
        links.append(Link(host, int(port), int(envs), backend[0] if backend else "godot", obs_flags))
    ev = dict(backend=a.eval_backend, difficulty=a.difficulty, weather=a.weather, precursor_obs=a.precursor_obs,
              oracle_obs=a.oracle_obs, max_days=a.max_days, weather_noise=a.weather_noise)
    obs_dim, n_act = links[0].info["obs_dim"], links[0].info["n_act"]
    model = Policy(obs_dim, n_act)
    if a.resume.endswith(".zip"):
        model.load_sb3(a.resume)
    elif a.resume:
        names = links[0].info.get("actions")
        if names is None:
            import lastembers_sim
            names = list(lastembers_sim.Env(os.path.join(os.path.dirname(HERE), "data")).action_names)
        load_resized(model, a.resume, names)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    t, next_eval, rnd = a.start_t, a.start_t, 0
    warm, good = a.warm, 0  # curriculum: start warmer, step colder once training runs survive
    log = open(os.path.join(a.out, "rounds.jsonl"), "a")
    while t < a.steps:
        if t >= next_eval:
            evaluate(model, a.out, t, **ev)
            next_eval = t + a.eval_every
        buf = io.BytesIO(); torch.save(model.state_dict(), buf); w = buf.getvalue()
        results = [None] * len(links)

        def run(i):
            try:
                send_msg(links[i].s, {"weights": w, "seconds": a.seconds, "warm": warm, "difficulty": a.difficulty,
                                      "weather": a.weather, "max_days": a.max_days, "weather_noise": a.weather_noise})
                results[i] = recv_msg(links[i].s)
            except Exception as e:
                results[i] = None
                print(json.dumps({"worker_lost": f"{links[i].host}:{links[i].port}", "error": repr(e)}), flush=True)
        th = [threading.Thread(target=run, args=(i,)) for i in range(len(links))]
        t0 = time.time()
        for x in th: x.start()
        for x in th: x.join()
        for i, r in enumerate(results):
            if r is None:
                try:
                    links[i].restart()
                    print(json.dumps({"worker_restarted": f"{links[i].host}:{links[i].port}"}), flush=True)
                except Exception as e:
                    print(json.dumps({"worker_restart_failed": links[i].host, "error": repr(e)}), flush=True)
        ok = [(l, r) for l, r in zip(links, results) if r is not None]
        if not ok:
            time.sleep(30)
            continue
        segs = [s for _, r in ok for s in r["segments"]]
        obs = np.concatenate([s["obs"] for s in segs]); mask = np.concatenate([s["mask"] for s in segs])
        act = np.concatenate([s["act"] for s in segs]); oldlp = np.concatenate([s["logp"] for s in segs])
        advs, rets = zip(*[gae(s, a.gamma, a.lam) for s in segs])
        adv = np.concatenate(advs); ret = np.concatenate(rets)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(act)
        O, M, A, L, AD, R = map(torch.tensor, (obs, mask, act, oldlp, adv, ret))
        for _ in range(a.epochs):
            perm = torch.randperm(n)
            for k in range(0, n, a.batch):
                idx = perm[k:k + a.batch]
                dist, v = model(O[idx], M[idx])
                ratio = torch.exp(dist.log_prob(A[idx]) - L[idx])
                pg = -torch.min(ratio * AD[idx], torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * AD[idx]).mean()
                loss = pg + 0.5 * ((v - R[idx]) ** 2).mean() - a.ent * dist.entropy().mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()
        ent_last, pg_last = float(dist.entropy().mean()), float(pg)
        t += n; rnd += 1
        per_host = {l.host: int(sum(len(s["act"]) for s in r["segments"])) for l, r in ok}
        eps = [e for _, r in ok for e in r["episodes"]]
        mean_day = float(np.mean([e["day"] for e in eps])) if eps else 0.0
        good = good + 1 if mean_day >= a.curriculum_day else 0
        if warm > 0 and good >= 3:
            warm, good = max(0.0, warm - 10.0), 0
        rec = {"warm": warm, "round": rnd, "t": t, "steps": n, "secs": round(time.time() - t0, 1), "per_worker": per_host,
               "episodes": len(eps), "train_mean_day": float(np.mean([e["day"] for e in eps])) if eps else None}
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        if LIVE:
            LIVE.metric("train_mean_day", rec["train_mean_day"], t)
            LIVE.metric("steps_per_sec", n / max(1e-6, time.time() - t0), t)
            LIVE.metric("warm_c", warm, t)
            LIVE.metric("entropy", ent_last, t)
            LIVE.metric("policy_loss", pg_last, t)
            for h, k in per_host.items():
                LIVE.metric(f"steps_{h}", k, t)
    evaluate(model, a.out, t, **ev)
    for l in links:
        send_msg(l.s, {"quit": True})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--port", type=int, required=True); w.add_argument("--envs", type=int, default=4)
    w.add_argument("--godot", default="godot"); w.add_argument("--root", required=True)
    w.add_argument("--backend", default="godot", choices=["godot", "rust"]); w.add_argument("--threads", type=int, default=0)
    w.add_argument("--precursor-obs", action="store_true"); w.add_argument("--oracle-obs", action="store_true")
    l = sub.add_parser("learn")
    l.add_argument("--no-mlflow", action="store_true"); l.add_argument("--workers", required=True); l.add_argument("--resume", default=""); l.add_argument("--out", default="runs/stage3")
    l.add_argument("--steps", type=int, default=50_000_000); l.add_argument("--start-t", type=int, default=0)
    l.add_argument("--eval-every", type=int, default=200_000); l.add_argument("--seconds", type=float, default=60.0)
    l.add_argument("--lr", type=float, default=3e-4); l.add_argument("--epochs", type=int, default=4)
    l.add_argument("--batch", type=int, default=4096); l.add_argument("--clip", type=float, default=0.2)
    l.add_argument("--ent", type=float, default=0.01); l.add_argument("--gamma", type=float, default=0.995)
    l.add_argument("--lam", type=float, default=0.95)
    l.add_argument("--warm", type=float, default=0.0); l.add_argument("--curriculum-day", type=float, default=30.0)
    l.add_argument("--difficulty", default="STANDARD", choices=["GENTLE", "STANDARD", "HARSH", "HARDCORE"])
    l.add_argument("--eval-backend", default="godot", choices=["godot", "rust"])
    # weather experiments (docs/ml/DIFFUSION-PROPOSAL.md E0); defaults reproduce the older stages
    l.add_argument("--weather", default="scripted", choices=["scripted", "chaotic"])
    l.add_argument("--weather-noise", default="", choices=["", "murky", "clear"],
                   help="chaotic weather observation noise (default: the game's level, murky)")
    l.add_argument("--precursor-obs", action="store_true", help="append the visible weather signs to the observation")
    l.add_argument("--oracle-obs", action="store_true", help="chaotic weather: append the true next landing (upper bound)")
    l.add_argument("--max-days", type=int, default=0, help="episode cap in days (0 = 60)")
    a = ap.parse_args()
    if a.mode == "worker":
        worker_rust(a) if a.backend == "rust" else worker(a)
    else:
        learn(a)
