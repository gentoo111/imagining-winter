"""Stage-1 training: MaskablePPO on parallel sim servers. Saves checkpoints and,
every eval, replayable episode records (seed + action list) for later video."""
import argparse, json, os, time
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from env import LastEmbersEnv

EVAL_SEEDS = [1001, 1002, 1003, 1004, 1005]  # held out: never used in training (training seeds are random 31-bit)


def make(i, host, port):
    return lambda: LastEmbersEnv(seed_base=i, perturb=True, host=host, port=port)


def env_fns(spec):
    """spec like "local:6,o:4,w:8" -> one sim server per slot, spread across hosts."""
    fns, i = [], 0
    for part in spec.split(","):
        host, n = part.split(":")
        for k in range(int(n)):
            port = {"local": 0, "o": 17300, "w": 17400}[host] + k
            fns.append(make(i, host, port)); i += 1
    return fns


class Eval(BaseCallback):
    def __init__(self, out, every):
        super().__init__(); self.out, self.every, self.next = out, every, 0

    def _on_step(self):
        if self.num_timesteps < self.next:
            return True
        self.next = self.num_timesteps + self.every
        recs = []
        env = LastEmbersEnv(perturb=False, seeds=EVAL_SEEDS, record=recs.append)
        for _ in EVAL_SEEDS:
            obs, _ = env.reset(); done = False
            while not done:
                a, _ = self.model.predict(obs, action_masks=env.action_masks(), deterministic=True)
                obs, r, done, _, info = env.step(a)
        env.close()
        days = [r["day"] for r in recs]
        line = {"t": self.num_timesteps, "time": time.time(), "mean_day": float(np.mean(days)), "days": days,
                "alive": [r["alive"] for r in recs]}
        print(json.dumps(line), flush=True)
        with open(os.path.join(self.out, "eval.jsonl"), "a") as f:
            f.write(json.dumps(line) + "\n")
        with open(os.path.join(self.out, f"replays_{self.num_timesteps:09d}.json"), "w") as f:
            json.dump(recs, f)
        self.model.save(os.path.join(self.out, f"model_{self.num_timesteps:09d}"))
        return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=6)
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--eval-every", type=int, default=50_000)
    ap.add_argument("--out", default="runs/stage1")
    ap.add_argument("--hosts", default="")
    ap.add_argument("--resume", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fns = env_fns(a.hosts) if a.hosts else [make(i, "local", None) for i in range(a.envs)]
    venv = SubprocVecEnv(fns)
    if a.resume:
        model = MaskablePPO.load(a.resume, env=venv, device="cpu")
    else:
        model = MaskablePPO("MlpPolicy", venv, n_steps=1024, batch_size=1024, gamma=0.995, ent_coef=0.01, verbose=0, device="cpu")
    model.learn(a.steps, callback=Eval(a.out, a.eval_every), reset_num_timesteps=not a.resume)
