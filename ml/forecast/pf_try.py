import sys, time
from pf import *
prof, noise, N, jit, rs, ns = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
d = load(prof, noise, "val")[:ns].astype(np.int64)
bank = build_bank(100000)
rng = np.random.default_rng(1)
t0 = time.time(); ll = []; br72 = []; hit = []
for s in range(ns):
    o = run_seed(d[s], prof, noise, N, jit, rs, bank, rng, want_paths=False)
    ih = issue_hours()
    tc = np.array([true_class(d[s], h) for h in ih])
    p = o["probs"]
    ll.append(-np.log(np.maximum(p[np.arange(len(tc)), tc], 1e-4)))
    p72 = p[:, :12].sum(1); y72 = (tc < 12).astype(float)
    br72.append((p72 - y72) ** 2); hit.append(y72)
ll = np.concatenate(ll); br = np.concatenate(br72); y = np.concatenate(hit)
clim = y.mean()
print(f"{prof} {noise} N={N} jit={jit} rs={rs}: logloss {ll.mean():.3f}  BS72 {br.mean():.4f} BSS72(clim) {1-br.mean()/((clim-y)**2).mean():.3f}  {time.time()-t0:.0f}s", flush=True)
