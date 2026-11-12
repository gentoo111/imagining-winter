#!/usr/bin/env python3
"""Step-by-step trace of the frozen Rust sim (lastembers_sim, commit 4195eb6) for cross-host parity (S2-0).

  python -B sim_trace.py --site <dir holding lastembers_sim> --data <frozen data/> --out trace.json [--seeds 1-60]

Each seed runs one episode. Reset options copy `config()` of 4195eb6 simrs/parity/run.py
(policy mode, perturb, warm, no_floor, difficulty, scripted/chaotic weather, precursor_obs).
Actions come from a pure-integer splitmix64 hash of (seed, step) over the current legal
actions, so two hosts choose the same action as long as their masks agree. Every step
hashes what Env.step returns (float32 obs, mask, reward f64 bytes, done, day, alive) and,
separately, Env.raw_obs() (the f64 obs before the float32 cast); the trace keeps all
per-step digests so sim_parity.py can name the first mismatch of each.
The module is only imported (python -B: no __pycache__ written next to a frozen copy).
"""
import argparse, hashlib, json, os, platform, struct, sys, time

MASK64 = (1 << 64) - 1


def splitmix64(x):
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


def config(s):
    """Reset options of 4195eb6 simrs/parity/run.py config() for seeds < 2000."""
    mode = ["random", "bot", "mixed"][s % 3]
    perturb = (s // 3) % 2
    no_floor = 1 if (s // 6) % 2 == 0 else 0
    warm = [0.0, 0.0, 10.0, 30.0][(s // 12) % 4]
    diff = "STANDARD"
    if s % 7 == 0:
        diff = "HARSH"
    elif s % 11 == 0:
        diff = "GENTLE"
    elif s % 13 == 0:
        diff = "HARDCORE"
    weather = "chaotic" if s % 2 == 0 else "scripted"
    precursor_obs = 1 if s % 5 == 0 else 0
    return dict(seed=s, mode=mode, perturb=perturb, no_floor=no_floor, warm=warm, difficulty=diff,
                weather=weather, precursor_obs=precursor_obs)


def pick(seed, t, mode, mask):
    legal = [i for i, ok in enumerate(mask) if ok]
    h = splitmix64((seed << 32) ^ t)
    if mode == "bot" and mask[1]:
        return 1
    if mode == "mixed" and mask[1] and (h & 1):
        return 1
    return legal[(h >> 1) % len(legal)]


def digest(obs, mask, reward, done, info):
    """What Env.step returns (the PPO's whole view): float32 obs, mask, reward f64, done, day, alive."""
    h = hashlib.sha256()
    h.update(obs.astype("<f4").tobytes())
    h.update(mask.astype("u1").tobytes())
    h.update(struct.pack("<d?qq", reward, bool(done), int(info["day"]), int(info["alive"])))
    return h.hexdigest()[:32]


def raw_digest(raw):
    """Env.raw_obs(): the f64 observation before the float32 cast (diagnostic, not returned by step)."""
    return hashlib.sha256(struct.pack("<%dd" % len(raw), *raw)).hexdigest()[:32]


def run(m, data, s):
    c = config(s)
    e = m.Env(data)
    obs, mask, info = e.reset(s, bool(c["perturb"]), float(c["warm"]), bool(c["no_floor"]), c["difficulty"],
                              c["weather"], bool(c["precursor_obs"]), False, 0, "murky")
    steps = [digest(obs, mask, 0.0, False, info)]
    raws = [raw_digest(e.raw_obs())]
    actions, ret, done, t = [], 0.0, False, 0
    while not done:
        a = pick(s, t, c["mode"], mask)
        obs, mask, r, done, info = e.step(a)
        actions.append(a)
        ret += r
        steps.append(digest(obs, mask, r, done, info))
        raws.append(raw_digest(e.raw_obs()))
        t += 1
        if t > 200000:
            raise RuntimeError(f"seed {s}: episode did not end")
    whole = hashlib.sha256("".join(steps).encode()).hexdigest()
    return dict(seed=s, config=c, steps=len(actions), day=int(info["day"]), alive=int(info["alive"]),
                ret=repr(ret), episode_sha256=whole, actions=actions, step_digests=steps, raw_obs_digests=raws)


def seed_list(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="1-60")
    a = ap.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, a.site)
    import lastembers_sim as m
    so = m.lastembers_sim.__file__ if hasattr(m, "lastembers_sim") else m.__file__
    t0 = time.time()
    eps = [run(m, a.data, s) for s in seed_list(a.seeds)]
    meta = dict(host=platform.node(), platform=platform.platform(), python=platform.python_version(),
                module=so, module_sha256=hashlib.sha256(open(so, "rb").read()).hexdigest(),
                seconds=round(time.time() - t0, 1), episodes=eps)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f)
    os.replace(tmp, a.out)
    print(f"{len(eps)} episodes, {sum(e['steps'] for e in eps)} steps, {meta['seconds']} s -> {a.out}")


if __name__ == "__main__":
    main()
