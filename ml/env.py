"""Gymnasium env wrapping the sim: the Godot server (tools/rl_server.gd) or, with
backend="rust", the in-process Rust port (simrs/, module lastembers_sim), which
returns exactly the same values (checked by simrs/parity/)."""
import json, subprocess, os
import numpy as np
import gymnasium as gym

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# How each remote host starts a sim server listening on 127.0.0.1:{port} (low priority).
HOSTS = {
    "o": "nice -n 15 /opt/homebrew/bin/godot --headless --path ~/Games/last-embers -s res://tools/rl_server.gd -- --port {port}",
    "w": r'cmd /c start "" /b /wait /belownormal D:\LastEmbers\bin\Godot_v4.7.2-stable_win64_console.exe --headless --path D:\LastEmbers\game -s res://tools/rl_server.gd -- --port {port}',
}


class LastEmbersEnv(gym.Env):
    def __init__(self, seed_base=0, perturb=True, difficulty="STANDARD", seeds=None, record=None, host="local", port=None,
                 backend="godot", weather="scripted", precursor_obs=False, oracle_obs=False, max_days=0,
                 weather_noise=""):
        # reset options (defaults reproduce the older runs): weather scripted/chaotic,
        # precursor_obs / oracle_obs (extra observation entries), max_days (0 = the env's 60)
        self.weather, self.precursor_obs, self.oracle_obs, self.max_days = weather, bool(precursor_obs), bool(oracle_obs), int(max_days)
        self.weather_noise = str(weather_noise or "")  # chaotic weather: "" = the game's level (murky), or "clear"
        self.host, self.backend = host, backend
        self.sim = None
        if backend == "rust":
            import lastembers_sim
            self.sim = lastembers_sim.Env(os.path.join(ROOT, "data"))
            self.p = self.sock = None
            spec = {"actions": self.sim.action_names, "obs_size": self.sim.obs_size}
        elif host == "local":
            self.p = subprocess.Popen(["godot", "--headless", "--path", ROOT, "-s", "res://tools/rl_server.gd"],
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
            self.sock = None
        else:
            self._open_remote(host, port)
        if self.sim is None:
            spec = self._call({"cmd": "spec"})
        self.action_names = spec["actions"]
        self.action_space = gym.spaces.Discrete(len(self.action_names))
        self.observation_space = gym.spaces.Box(-5, 5, (spec["obs_size"],), np.float32)
        self.rng = np.random.default_rng(seed_base)
        self.perturb, self.difficulty, self.seeds, self.record = perturb, difficulty, seeds, record
        self._mask = np.ones(len(self.action_names), bool)
        self.k = 0

    def _open_remote(self, host, port):
        import socket, time
        cmd = HOSTS[host].format(port=port)
        # one ssh both runs the sim server on the remote host and forwards the port
        self.p = subprocess.Popen(["ssh", "-o", "ExitOnForwardFailure=yes", "-L", f"{port}:127.0.0.1:{port}", host, cmd],
                                  stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(600):
            try:
                self.sock = socket.create_connection(("127.0.0.1", port), timeout=2)
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.rf = self.sock.makefile("r", encoding="utf-8")
                # ssh accepts the forward before the server listens; a spec round-trip proves it is up
                self.sock.sendall(b'{"cmd":"spec"}\n')
                if self.rf.readline().startswith("{"):
                    return
                raise OSError("not ready")
            except OSError:
                try:
                    self.sock.close()
                except Exception:
                    pass
                time.sleep(0.3)
        raise RuntimeError(f"cannot reach sim server on {host}:{port}")

    def _call(self, c):
        if self.sock is not None:
            self.sock.sendall((json.dumps(c) + "\n").encode())
            line = self.rf.readline()
            if not line:
                raise RuntimeError(f"sim server on {self.host} died")
            return json.loads(line)
        self.p.stdin.write(json.dumps(c) + "\n")
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("sim server died")
            if line.startswith("{"):
                return json.loads(line)

    def _out(self, r):
        self._mask = np.array(r["mask"], bool)
        return np.clip(np.array(r["obs"], np.float32), -5, 5)

    def reset(self, seed=None, options=None):
        if self.seeds is not None:
            s = int(self.seeds[self.k % len(self.seeds)]); self.k += 1
        else:
            s = int(self.rng.integers(1, 2**31 - 1))
        self.cur_seed, self.actions_taken = s, []
        if self.sim is not None:
            obs, mask, info = self.sim.reset(s, bool(self.perturb), 0.0, True, self.difficulty, self.weather,
                                             self.precursor_obs, self.oracle_obs, self.max_days, self.weather_noise or "murky")
            self._mask = mask
            return np.clip(obs, -5, 5), info
        r = self._call({"cmd": "reset", "seed": s, "perturb": self.perturb, "difficulty": self.difficulty, **self._opts()})
        return self._out(r), r["info"]

    def step(self, a):
        self.actions_taken.append(int(a))
        if self.sim is not None:
            obs, mask, rew, done, info = self.sim.step(int(a))
            r = {"done": done, "info": info, "reward": rew}
            self._mask = mask
            obs = np.clip(obs, -5, 5)
        else:
            r = self._call({"cmd": "step", "action": int(a)})
            obs = self._out(r)
        info = r["info"]
        if r["done"]:
            info["episode_seed"] = self.cur_seed
            if self.record is not None:
                rec = {"seed": self.cur_seed, "perturb": self.perturb, "actions": self.actions_taken,
                       "day": info["day"], "alive": info["alive"]}
                rec.update(self._opts())  # replays need the weather mode to reproduce
                if self.difficulty != "STANDARD":
                    rec["difficulty"] = self.difficulty
                self.record(rec)
        return obs, float(r["reward"]), bool(r["done"]), False, info

    def _opts(self):
        """Non-default reset options only, so older servers and replays see the same messages."""
        o = {}
        if self.weather != "scripted":
            o["weather"] = self.weather
        if self.precursor_obs:
            o["precursor_obs"] = True
        if self.oracle_obs:
            o["oracle_obs"] = True
        if self.max_days:
            o["max_days"] = self.max_days
        if self.weather_noise:
            o["weather_noise"] = self.weather_noise
        return o

    def action_masks(self):
        return self._mask

    def close(self):
        if self.sim is not None:
            return
        try:
            self._call({"cmd": "quit"})
        except Exception:
            pass
        if self.sock is not None:
            self.sock.close()
        self.p.kill()
