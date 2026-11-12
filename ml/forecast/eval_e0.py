"""E0 final evaluation (RQ2 value ladder): final PPO checkpoints of the three rungs + the scripted bot
on held-out seeds, under the E0 training conditions (HARDCORE, chaotic weather, murky signs, 120-day cap,
perturb=False like dppo.evaluate). Also records per-hour probe data (raw obs, policy/value hidden layers,
oracle hours-to-landing) for probe_e0.py.

Runs on the frozen 4195eb6 Rust module (fc_obs.use_frozen, asserted) and the checkpoints (read-only):
  nice -n 10 ~/Documents/Games/last-embers/ml/.venv/bin/python eval_e0.py [--seeds 300000:300200] [--mode greedy|sample]
Out: out/e0_eval_<mode>.json (per-episode records) and out/e0_probe_<rung>.npz (greedy only).

All envs are created with precursor+oracle observation entries on (136-d); a rung sees only its own prefix
(128 / 134 / 136), which is exactly its training observation: the extras are appended at the end and are
pure readouts (the oracle is computed by the weather regardless of the flag)."""
import argparse, json, os, sys, time
import numpy as np
import torch

MAIN = os.path.expanduser("~/Documents/Games/last-embers")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fc_obs  # noqa: E402
lastembers_sim = fc_obs.use_frozen()  # the frozen 4195eb6 build (asserted); never the main venv's module
sys.path.insert(1, os.path.join(fc_obs.FROZEN, "ml"))
from dppo import Policy  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
RUNS = os.path.join(MAIN, "ml", "runs")
RUNGS = {"none": 128, "signs": 134, "oracle": 136, "start": 128, "fc": 140,  # fc: 134 signs + 6 forecast (fc_obs.py)
         "fcd": 140, "fcg": 140}  # S1-11b: 3b / 3a' = the same 140-d obs, forecast read from a daily 07:00 table
FC3B = os.environ.get("FC3B_OUT") or (os.path.join(HERE, "..", "..", "..", "out", "fc3b") if os.name == "nt"
                                      else os.path.join(HERE, "out", "fc3b"))
FC_TABLES = {"fcd": os.path.join(FC3B, "features_diff_v3.npz"), "fcg": os.path.join(FC3B, "features_gru_daily.npz")}
IN_STORM, ACT_HOUR, ORACLE_H = 4, 122, 134   # obs indices: in-storm flag, actions this hour, oracle (h/240)
# Game data the E0 policies were trained on: the Rust module in the main venv was built at 05:09 on 2026-11-01
# from 4195eb6 (simrs/ and data/ identical to 339afa0); training started 05:15. Do not rebuild
# that module. Main's data/ changed at 863bfff (FP eating rule), so the data are frozen here (git archive 339afa0 data) instead of read from the main checkout.
SIM_DATA = os.environ.get("FC_SIM_DATA") or os.path.join(HERE, "data", "sim_339afa0", "data")  # Windows: frozen data copy
COND = dict(perturb=False, warm=0.0, no_floor=True, difficulty="HARDCORE", weather="chaotic", max_days=120, weather_noise="murky")


CKPT_IDX = -1  # which checkpoint of each run (-1 = final); --ckpt-idx evaluates earlier ones
RUN_DIRS = {}  # rung -> run directory override (--run-dirs), e.g. the extra training seeds of e0_queue.py

# Storm-focused metrics (RQ2): base-observation indices (128-d layout of simrs/core/src/env.rs obs_into, data 339afa0)
F_LEVEL, F_LIT, F_MAX, H_MAX, COAL, RAW_FOOD, RATIONS, ALIVE = 8, 9, 10, 11, 12, 15, 16, 18
SICK, GRAVE, HYPO, HUNGRY, HOMELESS = 22, 23, 24, 25, 26
TECH0, TEAMS, HOPE, DISCONTENT, OVERDRIVE, PRESSURE = 75, 118, 123, 124, 125, 126
GEN_EFF = ((12, 0.9), (13, 0.8642))  # techs generator_efficiency_1/2: furnace coal multipliers
COAL_PER_LEVEL_H = 6.0 * 0.125      # balance.json furnace.coal_per_level_hour x k
ORACLE_TIER = 135
POST_H = 24                          # deaths counted this many hours after a storm ends


def res(o, k):
    return np.exp(6.0 * o[..., k].astype(np.float64)) - 1.0  # obs = ln(1+x)/6


def furnace_need(o):
    """Coal per hour the furnace burns at its current level (heaters excluded: their tiers are not observed)."""
    mult = np.ones(o.shape[:-1])
    for t, m in GEN_EFF:
        mult = np.where(o[..., TECH0 + t] > 0.99, mult * m, mult)
    return np.rint(o[..., F_LEVEL] * 4) * COAL_PER_LEVEL_H * mult


def final_ckpt(r):
    if r == "start":  # the common stage-11 starting point of all three rungs
        return os.path.join(RUNS, "stage11", "model_1755066122.pt")
    d = RUN_DIRS.get(r, os.path.join(RUNS, f"e0-{r}"))
    fs = sorted(f for f in os.listdir(d) if f.startswith("model_"))
    return os.path.join(d, fs[CKPT_IDX])


def run(agent, seeds, mode, threads, probe_every=0, torch_seed=0):
    """agent: rung name or 'bot'. Returns per-episode records (and probe arrays if probe_every)."""
    n = len(seeds)
    venv = lastembers_sim.VecEnv(n, SIM_DATA, threads, True, True)
    obs, mask = venv.reset(list(map(int, seeds)), **COND)
    net = None
    if agent != "bot":
        d = RUNGS[agent]
        net = Policy(d, venv.n_actions)
        net.load_state_dict(torch.load(final_ckpt(agent), weights_only=True)); net.eval()
    g = torch.Generator().manual_seed(torch_seed)
    fc = None
    if agent == "fc":
        fc = fc_obs.FcState(n); fc.reset(np.arange(n))
    elif agent in FC_TABLES:  # table mode: rows of these (evaluation-pool) seeds; the sim keeps the same seeds
        tab = fc_obs.Table(FC_TABLES[agent])
        assert all(tab.seed_row(int(s))[0] == int(s) for s in seeds), "table mode evaluates pool seeds only"
        fc = fc_obs.FcState(n, tab); fc.reset(np.arange(n), [tab.seed_row(int(s))[1] for s in seeds])
    done = np.zeros(n, bool)
    alive0 = None
    hours = np.zeros(n, int); storm_h = np.zeros(n, int)
    deaths = np.zeros(n, int); storm_deaths = np.zeros(n, int)
    storms = np.zeros(n, int); prev_storm = obs[:, IN_STORM] > 0.5
    per_storm = [[] for _ in range(n)]
    day = np.zeros(n, int); alive = np.zeros(n, int)
    post_deaths = np.zeros(n, int); since_end = np.full(n, 10**6)  # deaths within 48 h after a storm ended
    end_obs = np.zeros((n, 128), np.float32)
    # storm-focused records: one dict per storm the colony reaches (landing hour with people alive)
    storm_recs = [[] for _ in range(n)]
    open_rec = [None] * n                 # storm in progress or in its POST_H-hour aftermath
    ring_coal = np.zeros((n, 49)); ring_lvl = np.zeros((n, 49)); ring_need = np.zeros((n, 49))
    prev_tier = obs[:, ORACLE_TIER].copy(); prev_orc = obs[:, ORACLE_H] * 240.0
    probe = {k: [] for k in ("seed", "hour", "obs", "h1", "h2", "v1", "v2", "in_storm", "orc")}
    steps = 0
    while not done.all():
        o = torch.tensor(obs)
        acts = np.ones(n, np.int64)  # bot
        new_hour = obs[:, ACT_HOUR] == 0
        if net is not None:
            x = o[:, :RUNGS[agent]] if fc is None else torch.cat([o[:, :134], torch.from_numpy(fc.feat)], 1)
            with torch.no_grad():
                h1 = net.pi[1](net.pi[0](x)); h2 = net.pi[3](net.pi[2](h1))
                logits = net.action_net(h2)
                logits = torch.where(torch.tensor(mask), logits, torch.full_like(logits, -1e8))
                acts = (logits.argmax(-1) if mode == "greedy" else
                        torch.multinomial(torch.softmax(logits, -1), 1, generator=g)[:, 0]).numpy().astype(np.int64)
                if probe_every:
                    k = np.nonzero(new_hour & ~done & (hours % probe_every == 0))[0]
                    if len(k):
                        v1 = net.vf[1](net.vf[0](x[k])); v2 = net.vf[3](net.vf[2](v1))
                        probe["seed"].append(seeds[k]); probe["hour"].append(hours[k].copy())
                        probe["obs"].append(obs[k, :136].astype(np.float16)); probe["h1"].append(h1[k].numpy().astype(np.float16))
                        probe["h2"].append(h2[k].numpy().astype(np.float16)); probe["v1"].append(v1.numpy().astype(np.float16))
                        probe["v2"].append(v2.numpy().astype(np.float16)); probe["in_storm"].append(obs[k, IN_STORM] > 0.5)
                        probe["orc"].append(obs[k, ORACLE_H] * 240.0)
        acts[done] = 0
        obs2, mask2, rew, d2, day2, alive2 = venv.step(acts)
        steps += int((~done).sum())
        if alive0 is None:
            alive0 = alive2.copy()  # alive count before the first hour (no deaths without an hour advance)
            alive_prev = alive2.copy()
        adv = (obs2[:, ACT_HOUR] == 0) & ~done  # an hour advanced in this step
        if fc is not None:
            fc.advance(adv, obs2)
        ins = obs2[:, IN_STORM] > 0.5
        # deaths in the advanced hour, exactly, from the reward: r = (alive/24 - 10*deaths - 10*alive*[lost])/10 - cost
        cost = np.where((acts != 0) & (acts != 1), 0.002, 0.0)
        lostf = d2 & (alive2 > 0) & (day2 < 120)
        dd = np.where(adv, np.rint((alive2 / 24.0 - 10.0 * (rew + cost)) / 10.0 - lostf * alive2).astype(int), 0)
        # storm hour: the camp was in a storm at the start or the end of the advanced hour
        sh = adv & (ins | prev_storm)
        hours += adv; storm_h += sh
        deaths += np.maximum(dd, 0); storm_deaths += np.where(sh, np.maximum(dd, 0), 0)
        onset = adv & ins & ~prev_storm
        for i in np.nonzero(onset)[0]:
            storms[i] += 1; per_storm[i].append(0)
        for i in np.nonzero(sh & (dd > 0))[0]:
            if per_storm[i]:
                per_storm[i][-1] += int(dd[i])
        since_end = np.where(adv & prev_storm & ~ins, 0, np.where(adv, since_end + 1, since_end))
        post_deaths += np.where(adv & ~sh & (since_end <= 48), np.maximum(dd, 0), 0)
        # ---- storm-focused bookkeeping (hour boundaries only)
        coal2 = res(obs2, COAL); need2 = furnace_need(obs2)
        ring_coal = np.where(adv[:, None], np.roll(ring_coal, 1, 1), ring_coal); ring_coal[adv, 0] = coal2[adv]
        ring_lvl = np.where(adv[:, None], np.roll(ring_lvl, 1, 1), ring_lvl); ring_lvl[adv, 0] = np.rint(obs2[adv, F_LEVEL] * 4)
        ring_need = np.where(adv[:, None], np.roll(ring_need, 1, 1), ring_need); ring_need[adv, 0] = need2[adv]
        for i in np.nonzero(adv)[0]:
            rec = open_rec[i]
            ddi = max(int(dd[i]), 0)
            if onset[i]:
                if rec is not None:  # a new storm inside the previous one's aftermath: close it (truncated)
                    rec["post_truncated"] = True; storm_recs[i].append(rec)
                o = obs2[i]; hrs = 48 if hours[i] >= 48 else 0
                food = float(res(o, RAW_FOOD) + res(o, RATIONS)); pop = int(alive_prev[i])
                need = float(need2[i])
                rec = dict(k=len(storm_recs[i]), day=float(o[0] * 120), hour=int(hours[i]), tier=int(round(prev_tier[i])),
                           pop=pop, coal=float(coal2[i]), coal_need_h=need,
                           coal_hours=float(coal2[i] / need) if need > 0 else None,
                           furnace_level=int(round(o[F_LEVEL] * 4)), furnace_lit=bool(o[F_LIT] > 0.5),
                           furnace_max=int(round(o[F_MAX] * 4)), heater_max=int(round(o[H_MAX] * 3)),
                           teams_out=int(round(o[TEAMS])), food=food, food_per_head=food / max(pop, 1),
                           sick=float(o[SICK]), grave=float(o[GRAVE]), hypo=float(o[HYPO]), hungry=float(o[HUNGRY]),
                           homeless=float(o[HOMELESS]), hope=float(o[HOPE]), discontent=float(o[DISCONTENT]),
                           overdrive=bool(o[OVERDRIVE] > 0.5), pressure=float(o[PRESSURE]),
                           coal_48h_before=float(ring_coal[i, hrs]) if hrs > 0 else None,
                           level_48h_before=int(ring_lvl[i, hrs]) if hrs > 0 else None,
                           storm_hours=0, deaths=0, post_deaths=0, furnace_out_hours=0, min_coal=float(coal2[i]),
                           coal_short_hours=0, ended=False, phase="storm")
                open_rec[i] = rec
            if rec is None or open_rec[i] is None:
                continue
            rec = open_rec[i]
            if rec["phase"] == "storm":
                rec["storm_hours"] += 1; rec["deaths"] += ddi
                lit = obs2[i, F_LIT] > 0.5
                rec["furnace_out_hours"] += int(not lit)
                rec["coal_short_hours"] += int(coal2[i] < need2[i] and need2[i] > 0)
                rec["min_coal"] = min(rec["min_coal"], float(coal2[i]))
                if not ins[i]:
                    rec["phase"] = "post"; rec["post_h"] = 0
            else:
                rec["post_h"] += 1; rec["post_deaths"] += ddi
                if rec["post_h"] >= POST_H:
                    rec.pop("phase"); rec.pop("post_h"); storm_recs[i].append(rec); open_rec[i] = None
        prev_tier = np.where(adv, obs2[:, ORACLE_TIER], prev_tier)
        prev_storm = np.where(adv, ins, prev_storm)
        alive_prev = np.where(~done, alive2, alive_prev)
        fin = d2 & ~done
        day[fin] = day2[fin]; alive[fin] = alive2[fin]; end_obs[fin] = obs2[fin, :128]
        for i in np.nonzero(fin)[0]:
            if open_rec[i] is not None:  # the episode ended during the storm or its aftermath
                rec = open_rec[i]; rec["ended"] = True; rec["ended_in"] = rec.pop("phase"); rec.pop("post_h", None)
                rec["end_alive"] = int(alive2[i]); storm_recs[i].append(rec); open_rec[i] = None
        done |= d2
        obs, mask = obs2, mask2
    recs = []
    for i in range(n):
        # lost with people alive: the furnace exploded (pressure pinned at its maximum, furnace out) or the camp
        # exiled the leader (failed ultimatum; discontent ~1). A cold furnace alone does not end the game.
        outcome = "cap" if day[i] >= 120 else ("wiped" if alive[i] == 0 else
                                               ("exploded" if end_obs[i, 126] >= 0.99 and end_obs[i, 9] < 0.5 else "exiled"))
        recs.append(dict(agent=agent, mode=mode if agent != "bot" else "bot", seed=int(seeds[i]), day=int(day[i]), alive=int(alive[i]),
                         alive_counted=int(alive[i]) if outcome == "cap" else 0, outcome=outcome, start_pop=int(alive0[i]),
                         hours=int(hours[i]), storm_hours=int(storm_h[i]), storms=int(storms[i]), deaths=int(deaths[i]),
                         storm_deaths=int(storm_deaths[i]), post_storm_deaths=int(post_deaths[i]), per_storm=per_storm[i],
                         end_hope=float(end_obs[i, 123]), end_discontent=float(end_obs[i, 124]), end_pressure=float(end_obs[i, 126]),
                         end_overdrive=float(end_obs[i, 125]), storm_recs=storm_recs[i]))
    pr = {k: np.concatenate(v) for k, v in probe.items()} if probe_every and probe["seed"] else None
    return recs, pr, steps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="300000:300200")
    ap.add_argument("--mode", default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--agents", default="bot,none,signs,oracle")
    ap.add_argument("--probe-every", type=int, default=1)
    ap.add_argument("--ckpt-idx", type=int, default=-1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--run-dirs", default="", help="rung=dir,... override the e0-<rung> run directories")
    ap.add_argument("--fc-tables", default="", help="fcd=<npz>,fcg=<npz>: forecast tables of the table-mode rungs")
    a = ap.parse_args()
    CKPT_IDX = a.ckpt_idx
    RUN_DIRS.update(dict(kv.split("=", 1) for kv in a.run_dirs.split(",") if kv))
    FC_TABLES.update(dict(kv.split("=", 1) for kv in a.fc_tables.split(",") if kv))
    torch.set_num_threads(1)
    lo, hi = map(int, a.seeds.split(":"))
    seeds = np.arange(lo, hi)
    assert not ((seeds >= 200000) & (seeds < 220500)).any() and not np.isin(seeds, [1001, 1002, 1003, 1004, 1005]).any()
    os.makedirs(OUT, exist_ok=True)
    allrecs = []
    for ag in a.agents.split(","):
        if ag == "bot" and a.mode == "sample":
            continue
        t0 = time.time()
        recs, pr, steps = run(ag, seeds, a.mode, a.threads, probe_every=a.probe_every if (a.mode == "greedy" and ag != "bot") else 0)
        dt = time.time() - t0
        print(json.dumps({"agent": ag, "mode": a.mode, "secs": round(dt, 1), "steps": steps,
                          "mean_day": float(np.mean([r["day"] for r in recs])),
                          "ckpt": final_ckpt(ag) if ag != "bot" else None}), flush=True)
        for r in recs:
            r["secs_total"] = round(dt, 1)
        allrecs += recs
        if pr is not None:
            np.savez_compressed(os.path.join(OUT, f"e0_probe_{ag}{a.tag}.npz"), **pr)
    with open(os.path.join(OUT, f"e0_eval_{a.mode}{a.tag}.json"), "w") as f:
        json.dump(dict(seeds=[lo, hi], cond=COND, ckpts={r: final_ckpt(r) for r in a.agents.split(",") if r != "bot"}, episodes=allrecs), f)
