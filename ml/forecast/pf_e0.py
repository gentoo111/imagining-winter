"""PF oracle (N=1024, jitter 0.01, re-seed 0.01, as in the paper) on the E0 RL evaluation seeds, for the
probe ceiling. Weather from weather_gen (game seed == weather seed):
  simrs/target/release/weather_gen data ml/forecast/data/endurance_murky_e0rl.bin 300000 200 120 endurance murky
  pf_e0.py <first index> <last index>   -> out/pf_e0rl_<a>_<b>.npz  (P over the 29 classes at daily 07:00 issues)"""
import sys, os, time
import numpy as np
from common import *
import pf
a, b = int(sys.argv[1]), int(sys.argv[2])
D = load("endurance", "murky", "e0rl")
ih = issue_hours()
bank = pf.build_bank(100000)
rng = np.random.default_rng(5000 + a)
P = np.zeros((b - a, len(ih), NB + 1), np.float32); Y = np.zeros((b - a, len(ih)), np.int64)
t0 = time.time()
for j, s in enumerate(range(a, b)):
    o = pf.run_seed(D[s].astype(np.int64), "endurance", "murky", 1024, 0.01, 0.01, bank, rng, issue=ih, want_paths=False)
    P[j] = o["probs"]; Y[j] = [true_class(D[s], h) for h in ih]
    print(j, s, f"{time.time()-t0:.0f}s lost={o['lost']}", flush=True)
np.savez_compressed(os.path.join(HERE, "out", f"pf_e0rl_{a}_{b}.npz"), P=P, Y=Y, seeds=np.arange(a, b) + 300000, ih=np.array(ih))
