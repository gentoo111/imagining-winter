#!/usr/bin/env python3
"""Parity harness: generate GDScript traces (tools/rl_trace.gd) and replay them in Rust.

  python3 simrs/parity/run.py --seeds 1-300 --jobs 4          # generate missing traces + check
  python3 simrs/parity/run.py --seeds 5 --full                 # one trace with full state dumps
  python3 simrs/parity/run.py --check-only
  python3 simrs/parity/run.py --seeds 2000-2099           # chaotic weather: all difficulties, noise levels, oracle/max_days

Each seed gets a fixed config (policy mode, perturb, warm, no_floor, difficulty, weather)
so the suite covers random, bot-heavy and mixed play with and without every
reset option. Traces go to simrs/parity/traces/ (git-ignored).
"""
import argparse, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TRACES = os.path.join(HERE, "traces")
CHECKER = os.path.join(ROOT, "simrs", "target", "release", "parity" + (".exe" if os.name == "nt" else ""))
GODOT = os.environ.get("GODOT", "godot")  # e.g. the Windows console exe


def config(s):
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
    # even seeds: chaotic weather (the game default); odd: FP scripted storms (training)
    weather = "chaotic" if s % 2 == 0 else "scripted"
    precursor_obs = 1 if s % 5 == 0 else 0
    c = dict(seed=s, mode=mode, perturb=perturb, no_floor=no_floor, warm=warm, difficulty=diff,
             weather=weather, precursor_obs=precursor_obs)
    if s >= 2000:
        # chaotic weather with every experiment option: seeds 2000+ (older seeds keep their configs)
        c.update(weather="chaotic", difficulty=["HARDCORE", "HARSH", "STANDARD", "GENTLE"][s % 4],
                 weather_noise="clear" if s % 6 == 5 else "murky",
                 precursor_obs=1 if s % 3 else 0, oracle_obs=1 if s % 4 < 2 else 0,
                 max_days=[0, 90, 120][s % 3], warm=[0.0, 10.0, 30.0, 30.0][(s // 3) % 4],
                 no_floor=0 if s % 5 == 4 else c["no_floor"])
    return c


def trace_path(s, full=False):
    return os.path.join(TRACES, f"s{s:05d}{'_full' if full else ''}.jsonl")


def gen(s, full=False, godot=None):
    godot = godot or GODOT
    c = config(s)
    out = trace_path(s, full)
    tmp = out + ".tmp"
    args = [godot, "--headless", "--path", ROOT, "-s", "res://tools/rl_trace.gd", "--"]
    for k, v in c.items():
        args += [f"--{k}", str(v)]
    args += ["--out", tmp, "--full", "1" if full else "0"]
    r = subprocess.run(([] if os.name == "nt" else ["nice", "-n", "10"]) + args, capture_output=True, text=True)
    if r.returncode != 0 or "SCRIPT ERROR" in r.stdout + r.stderr:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise RuntimeError(f"trace {s} failed")
    os.replace(tmp, out)
    return out


def summary(files):
    """Coverage: how far the traced episodes got and which actions they used."""
    import json
    days, steps, kinds = [], 0, {}
    for f in files:
        with open(f) as fh:
            head = json.loads(fh.readline())
            last = None
            for line in fh:
                r = json.loads(line); steps += 1; last = r
                kinds[r["a"]] = kinds.get(r["a"], 0) + 1
        days.append(last["day"])
    days.sort()
    print(f"{len(files)} episodes, {steps} steps; days min {days[0]} median {days[len(days)//2]} max {days[-1]}; "
          f">=30: {sum(d >= 30 for d in days)}, >=42: {sum(d >= 42 for d in days)}, 60: {sum(d >= 60 for d in days)}; "
          f"distinct actions used {len(kinds) - 1}")


def seeds_arg(x):
    out = []
    for part in x.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1-300")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--regen", action="store_true")
    a = ap.parse_args()
    os.makedirs(TRACES, exist_ok=True)
    seeds = seeds_arg(a.seeds)
    if not a.check_only:
        todo = [s for s in seeds if a.regen or a.full or not os.path.exists(trace_path(s, a.full))]
        with ThreadPoolExecutor(a.jobs) as ex:
            for i, p in enumerate(ex.map(lambda s: gen(s, a.full), todo)):
                if (i + 1) % 20 == 0:
                    print(f"generated {i + 1}/{len(todo)}", flush=True)
    files = [trace_path(s, a.full) for s in seeds if os.path.exists(trace_path(s, a.full))]
    summary(files)
    r = subprocess.run([CHECKER, os.path.join(ROOT, "data")] + files)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
