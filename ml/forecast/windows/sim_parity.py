#!/usr/bin/env python3
"""Compare two sim_trace.py traces (main-Mac frozen .so vs Windows build) and write parity.json (S2-0).

  python sim_parity.py --ref trace_mac.json --test trace_win.json --out parity.json \
      --commit <full sha> --toolchain "<rustc -V>" --hours <effort> [--notes ...]

A seed counts as bitwise identical only when every per-step digest of what Env.step returns
(float32 obs, mask, reward f64 bytes, done, day, alive) and the action sequence match; parity
is "ok" only when all compared seeds are identical. Env.raw_obs() (f64 obs before the float32
cast, a diagnostic the PPO never sees) is compared separately and reported as
raw_obs_f64_identical / raw_obs_mismatches, never folded into parity.
"""
import argparse, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--toolchain", default="")
    ap.add_argument("--hours", type=float, default=0.0)
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    ref, test = json.load(open(a.ref)), json.load(open(a.test))
    tmap = {e["seed"]: e for e in test["episodes"]}
    same, steps, bad, raw_same, raw_bad = 0, 0, [], 0, []
    for r in ref["episodes"]:
        t = tmap.get(r["seed"])
        if t is None:
            bad.append({"seed": r["seed"], "why": "missing in test trace"})
            continue
        rr, tr = r["raw_obs_digests"], t["raw_obs_digests"]
        rf = next((i for i in range(min(len(rr), len(tr))) if rr[i] != tr[i]), None)
        if rf is None and len(rr) == len(tr):
            raw_same += 1
        else:
            raw_bad.append({"seed": r["seed"], "first_mismatch_step": rf})
        rd, td = r["step_digests"], t["step_digests"]
        first = next((i for i in range(min(len(rd), len(td))) if rd[i] != td[i]), None)
        if first is None and (len(rd) != len(td) or r["actions"] != t["actions"]):
            first = min(len(rd), len(td))
        steps += min(len(rd), len(td)) if first is None else first
        if first is None:
            same += 1
        else:
            bad.append({"seed": r["seed"], "first_mismatch_step": first, "ref_steps": r["steps"], "test_steps": t["steps"]})
    n = len(ref["episodes"])
    out = {
        "commit": a.commit,
        "parity": "ok" if same == n and n > 0 and not bad else "failed",
        "seeds_bitwise_identical": same,
        "seeds": n,
        "steps_compared": steps,
        "mismatches": bad,
        "raw_obs_f64_identical": raw_same,
        "raw_obs_mismatches": raw_bad,
        "mac_so_sha256": ref["module_sha256"],
        "win_pyd_sha256": test["module_sha256"],
        "ref_host": f'{ref["host"]} {ref["platform"]} py{ref["python"]}',
        "test_host": f'{test["host"]} {test["platform"]} py{test["python"]}',
        "toolchain": a.toolchain,
        "hours_spent": a.hours,
        "notes": a.notes,
    }
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, a.out)
    print(f'parity {out["parity"]}: {same}/{n} seeds bitwise identical, {steps} steps compared; raw f64 obs identical {raw_same}/{n}')
    return 0 if out["parity"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
