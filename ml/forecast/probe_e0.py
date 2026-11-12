"""Implicit-forecast probe on the E0 PPO policies (RQ2): can hours-to-next-landing / landing-within-72h be
read linearly from the policy / value hidden layers, beyond what is linearly available in the raw input?

Data: out/e0_probe_<rung>.npz from eval_e0.py (greedy rollouts on seeds 300000-300199, one row per game
hour the agent was alive). Target: the oracle's true next landing (obs entry, h/240 capped at 240 h).
Rows: calm hours only (during a storm the next landing is trivially >= storm + 4-day gap).
Split by seed: 300000-300139 train, 300140-300199 test. Within train, the ridge / L2 strength is chosen on a
seed-split validation fold (300000-300109 vs 300110-300139), then refit on all train seeds.
Controls: (i) linear probe on the raw input observation; (ii) a 2x128 MLP on the raw input (the best any
memoryless policy could compute from its input, approximately); (iii) the same probes on the no-signs
policy; (iv) the oracle-date policy (positive control: the answer is in its input). Ceiling: the particle
filter with full history (pf_e0.py), compared on the same daily 07:00 calm rows.
  probe_e0.py  -> out/probe_e0.json and a printed table"""
import json, os, sys
import numpy as np
import torch
from metrics import boot_ci

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
torch.set_num_threads(2)
TR_END, VAL_FROM = 300140, 300110
OBS_D = {"none": 128, "signs": 134, "oracle": 136}


def standardize(Xtr, *Xs):
    # near-constant columns (e.g. flags that never fire in training rows) are only centred, and standardized
    # values are clipped, so a rare test value cannot blow up a fitted weight
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-3, 1.0, sd)
    return [np.clip((Xtr - mu) / sd, -10, 10)] + [np.clip((X - mu) / sd, -10, 10) for X in Xs]


def ridge(Xtr, ytr, Xte, lam):
    Xa = np.c_[Xtr, np.ones(len(Xtr))]; I = np.eye(Xa.shape[1]); I[-1, -1] = 0
    w = np.linalg.solve(Xa.T @ Xa + lam * len(Xtr) * I, Xa.T @ ytr)
    return np.c_[Xte, np.ones(len(Xte))] @ w


def logistic(Xtr, ytr, Xte, lam, iters=200):
    X = torch.tensor(np.c_[Xtr, np.ones(len(Xtr))], dtype=torch.float64); y = torch.tensor(ytr, dtype=torch.float64)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=iters, line_search_fn="strong_wolfe")
    def clo():
        opt.zero_grad()
        l = torch.nn.functional.binary_cross_entropy_with_logits(X @ w, y) + lam * (w[:-1] ** 2).sum()
        l.backward(); return l
    opt.step(clo)
    Xt = torch.tensor(np.c_[Xte, np.ones(len(Xte))], dtype=torch.float64)
    return torch.sigmoid(Xt @ w).detach().numpy()


def mlp(Xtr, ytr, Xte, kind, sd_tr, steps=4000, seed=0):
    """2x128 ReLU MLP, early-stopped on the held-out validation seeds of the training split (without it the
    MLP memorises training episodes through the camp-state inputs and generalises worse than chance)."""
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 128), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                              torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(128, 1))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    tr, va = sd_tr < VAL_FROM, sd_tr >= VAL_FROM
    X = torch.tensor(Xtr[tr], dtype=torch.float32); y = torch.tensor(ytr[tr], dtype=torch.float32)
    Xv = torch.tensor(Xtr[va], dtype=torch.float32); yv = torch.tensor(ytr[va], dtype=torch.float32)
    lossf = (lambda o, t: torch.nn.functional.binary_cross_entropy_with_logits(o, t)) if kind == "cls" else (lambda o, t: ((o - t) ** 2).mean())
    g = torch.Generator().manual_seed(seed)
    best, state = float("inf"), None
    for it in range(steps):
        net.train()
        k = torch.randint(0, len(X), (512,), generator=g)
        l = lossf(net(X[k])[:, 0], y[k])
        opt.zero_grad(); l.backward(); opt.step()
        if it % 100 == 99:
            net.eval()
            with torch.no_grad():
                lv = float(lossf(net(Xv)[:, 0], yv))
            if lv < best:
                best, state = lv, {k2: v.clone() for k2, v in net.state_dict().items()}
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        o = net(torch.tensor(Xte, dtype=torch.float32))[:, 0]
        return (torch.sigmoid(o) if kind == "cls" else o).numpy()


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def auc(y, p):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bss(y, p, base):
    return 1 - ((p - y) ** 2).mean() / ((base - y) ** 2).mean()


def boot(fn, seeds, *arrs, n=1000):
    """95% CI of fn(*arrs) by resampling test seeds."""
    us = np.unique(seeds); idx = {s: np.nonzero(seeds == s)[0] for s in us}
    rng = np.random.default_rng(0); vals = []
    for _ in range(n):
        k = np.concatenate([idx[s] for s in rng.choice(us, len(us))])
        vals.append(fn(*[a[k] for a in arrs]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def fit_eval(name, Xtr, Xte, ytr_h, yte_h, ytr_c, yte_c, sd_tr, sd_te, base, use_mlp=False):
    Xtr, Xte = standardize(Xtr, Xte)
    if use_mlp:
        ph = mlp(Xtr, ytr_h / 240.0, Xte, "reg", sd_tr) * 240.0
        pc = mlp(Xtr, ytr_c.astype(float), Xte, "cls", sd_tr)
    else:
        tr, va = sd_tr < VAL_FROM, sd_tr >= VAL_FROM
        lams = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
        lh = max(lams, key=lambda l: r2(ytr_h[va], ridge(Xtr[tr], ytr_h[tr], Xtr[va], l)))
        lc = min(lams, key=lambda l: ((logistic(Xtr[tr], ytr_c[tr], Xtr[va], l) - ytr_c[va]) ** 2).mean())
        ph = ridge(Xtr, ytr_h, Xte, lh); pc = logistic(Xtr, ytr_c, Xte, lc)
    res = dict(R2=float(r2(yte_h, ph)), R2_ci=boot(r2, sd_te, yte_h, ph), BSS72=float(bss(yte_c, pc, base)),
               BSS72_ci=boot(lambda y, p: bss(y, p, base), sd_te, yte_c, pc), AUC72=float(auc(yte_c, pc)),
               AUC72_ci=boot(auc, sd_te, yte_c, pc))
    print(f"  {name:28s} R2 {res['R2']:.3f} [{res['R2_ci'][0]:.3f},{res['R2_ci'][1]:.3f}]  BSS72 {res['BSS72']:.3f} "
          f"[{res['BSS72_ci'][0]:.3f},{res['BSS72_ci'][1]:.3f}]  AUC {res['AUC72']:.3f}", flush=True)
    return res, pc


def main():
    out = {}
    pfz = [np.load(os.path.join(OUT, f)) for f in sorted(os.listdir(OUT)) if f.startswith("pf_e0rl_")]
    pf = {}
    for z in pfz:
        for i, s in enumerate(z["seeds"]):
            for j, h in enumerate(z["ih"]):
                pf[(int(s), int(h))] = z["P"][i, j, :12].sum()
    for rung in ("signs", "none", "oracle"):
        z = np.load(os.path.join(OUT, f"e0_probe_{rung}.npz"))
        calm = ~z["in_storm"]
        sd, hr, y = z["seed"][calm], z["hour"][calm], z["orc"][calm].astype(np.float64)
        F = {"raw obs": z["obs"][calm, :OBS_D[rung]].astype(np.float64)}
        for k, nm in (("h1", "policy hidden 1"), ("h2", "policy hidden 2"), ("v1", "value hidden 1"), ("v2", "value hidden 2")):
            F[nm] = z[k][calm].astype(np.float64)
        F["policy h1+h2"] = np.c_[F["policy hidden 1"], F["policy hidden 2"]]
        yc = (y <= 72).astype(np.float64)
        tr, te = sd < TR_END, sd >= TR_END
        base = yc[tr].mean()
        print(f"{rung}: {tr.sum()} train / {te.sum()} test calm hours, {len(np.unique(sd[tr]))}/{len(np.unique(sd[te]))} seeds, "
              f"P(landing<=72h) train {base:.3f} test {yc[te].mean():.3f}", flush=True)
        out[rung] = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "base72": float(base)}
        preds = {}
        for nm, X in F.items():
            out[rung][nm], preds[nm] = fit_eval(nm, X[tr], X[te], y[tr], y[te], yc[tr], yc[te], sd[tr], sd[te], base)
        out[rung]["raw obs MLP"], preds["raw obs MLP"] = fit_eval("raw obs MLP (memoryless)", F["raw obs"][tr], F["raw obs"][te],
                                                                  y[tr], y[te], yc[tr], yc[te], sd[tr], sd[te], base, use_mlp=True)
        # daily 07:00 calm test rows, where the PF forecast exists: probe vs PF on identical rows
        if pf:
            k7 = np.array([(int(s), int(h)) in pf for s, h in zip(sd[te], hr[te])])
            if k7.sum():
                ppf = np.array([pf[(int(s), int(h))] for s, h in zip(sd[te][k7], hr[te][k7])])
                yy = yc[te][k7]; ss = sd[te][k7]
                d7 = {"n": int(k7.sum()), "PF": dict(BSS72=float(bss(yy, ppf, base)), BSS72_ci=boot(lambda y, p: bss(y, p, base), ss, yy, ppf),
                                                     AUC72=float(auc(yy, ppf)))}
                for nm, p in preds.items():
                    d7[nm] = dict(BSS72=float(bss(yy, p[k7], base)), AUC72=float(auc(yy, p[k7])))
                out[rung]["daily07"] = d7
                print(f"  07:00 rows (n={k7.sum()}): PF BSS72 {d7['PF']['BSS72']:.3f} AUC {d7['PF']['AUC72']:.3f}; "
                      + "; ".join(f"{nm} {v['BSS72']:.3f}" for nm, v in d7.items() if nm not in ("n", "PF")), flush=True)
    json.dump(out, open(os.path.join(OUT, "probe_e0.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
