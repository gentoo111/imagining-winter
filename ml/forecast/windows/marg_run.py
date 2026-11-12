"""S1-6 marginal control (RESEARCH-PLAN S1-6 / C1): the v3 backbone with an independent output per lead time, trained once,
evaluated with the selected v3 on the same 60 test seeds (0-59) for event BSS and path energy / variogram scores.
  marg_run.py selftest          (main Mac CPU, seconds) lead-time independence of the marginal net, event readout
  marg_run.py jobs [jobdir]     (main Mac) write the 4 queue jobs s16_*.cmd
  marg_run.py deploy            (main Mac) scp models/train/evaluate.py -> windows\\marg\\ (+ SHA256SUMS) and this script
  marg_run.py verify            (Windows) deployed windows\\marg\\ files == SHA256SUMS
  marg_run.py eval v3|marg      (Windows queue) 60 seeds, DDIM 25, 50 paths -> out\\marginal\\{v3,marg}.npz (+ clim, clock)
  marg_run.py summary           (Windows queue) -> out\\marginal\\summary_endurance_murky.json, done.json
  marg_run.py paired            (Windows, CPU minutes, after summary) paired seed-bootstrap differences -> out\\marginal\\paired.json
  marg_run.py export <dir>      (main Mac) scp summary / done / marg.pt(.log) -> <dir>
Marginal control = models.V3_CONFIGS["M"]: cfgC (4-layer adaLN-zero transformer, d 160, 5 heads) with attention restricted
to the diagonal (no information between lead hours; initial noise i.i.d. per hour -> given the condition each hour is
sampled independently) and a 4th per-lead event channel models.event_channel (event P(landing <= 6k) read at hour 6k,
models.marg_event_probs). Same diffusion, loss, optimiser, 100k steps, lr 1e-3, TF32, VAL_EVERY 1000, seed 0 as v3.
summary keys (summarize.py layout and algorithms): marg, diff (= selected v3 out\\v3sel\\cfgC_lr1e-3.pt), diff_shuf
(diagnostic: v3's own 50 paths with the sample index permuted independently per hour, rng 0 -> v3's exact per-hour
marginals without its joint structure), clim, clock. The deployed code lives in windows\\marg\\ so S1-4's queued jobs
keep using ml\\forecast\\*.py."""
import hashlib, json, os, shutil, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FC = os.path.dirname(HERE)
MARGD = os.path.join(HERE, "marg")
sys.path.insert(0, FC)
E1 = os.path.abspath(os.path.join(FC, "..", ".."))
OUT = os.environ.get("MARG_OUT", os.path.join(E1, "out", "marginal"))
NS = int(os.environ.get("MARG_NS", "60"))
DDIM, STEPS = 25, 100000
PROF, NOISE = "endurance", "murky"
E1W = r"D:\LastEmbers\e1"
FCW = E1W + r"\ml\forecast"
OUTW = E1W + r"\out\marginal"
V3PT = os.path.join(E1, "out", "v3sel", "cfgC_lr1e-3.pt")
FILES = ("models.py", "train.py", "evaluate.py")
PY = E1W + r"\.venv\Scripts\python.exe"


def selftest():
    import torch
    from models import v3_net, event_channel, marg_event_probs
    torch.manual_seed(0)
    from common import HOR, NB
    for cfg, want_indep in (("M", True), ("C", False)):
        net = v3_net(cfg).eval()
        for n_, q in net.named_parameters():  # adaLN-zero starts with zero gates: randomise so attention is live
            if "ada" in n_:
                torch.nn.init.normal_(q, std=0.05)
        nch = 4 if cfg == "M" else 3
        x = torch.randn(2, 168, 16); c = torch.rand(2, 6)
        cv = net.cond_vec(x, c)
        y = torch.randn(2, nch, HOR, requires_grad=True)
        out = net(y, torch.tensor([50, 120]), cv)
        assert out.shape == (2, nch, HOR), out.shape
        k = 100
        g, = torch.autograd.grad(out[:, :, k].sum(), y)
        other = g.abs().sum((0, 1)); other[k] = 0
        cross = float(other.sum())
        print(cfg, "grad of hour", k, "w.r.t. other hours:", cross, "own:", float(g[:, :, k].abs().sum()))
        assert (cross == 0.0) == want_indep, (cfg, cross)
    e = event_channel(np.array([0, 5, NB]))[..., 0]
    assert e.shape == (3, HOR) and e[0].min() == 1 and e[2].max() == -1 and e[1, 35] == 1 and e[1, 29] == -1
    rng = np.random.default_rng(0)
    p = marg_event_probs(rng.choice([-1.0, 1.0], (7, 50, HOR)))
    assert p.shape == (7, NB + 1) and (p >= -1e-12).all() and np.allclose(p.sum(-1), 1)
    p = marg_event_probs(np.broadcast_to(e[:, None], (3, 50, HOR)))
    assert np.allclose(p, np.eye(NB + 1)[[0, 5, NB]]), p.argmax(-1)
    print("selftest ok; params M", sum(q.numel() for q in v3_net("M").parameters()),
          "C", sum(q.numel() for q in v3_net("C").parameters()))


def jobs(jobdir):
    os.makedirs(jobdir, exist_ok=True)
    head = ["@echo off", "set PYTHONUTF8=1", "set DEVICE=cuda", f"cd /d {FCW}", f"if not exist {OUTW} mkdir {OUTW}",
            f"{PY} -u windows\\marg_run.py verify", "if errorlevel 1 exit /b 1"]
    J = {
        "s16_m_train": ["rem S1-6 marginal control training: models.V3_CONFIGS['M'] (cfgC, diagonal attention, per-lead event channel),",
                        f"rem lr 1e-3, {STEPS} steps chained through the queue (train_chain.py re-queues s16_m_train.cmd until done;",
                        "rem CHAIN_ENV lists itself, see E1-RESULTS S1-3). Code: windows\\marg\\train.py. Writes out\\marginal\\marg.pt(.log/.state/.chain.json)",
                        "set V3_CFG=M", "set LR=1e-3", "set TF32=1", "set VAL_EVERY=1000", f"set PYTHONPATH={FCW}",
                        f"set CHAIN_TRAIN_PY={FCW}\\windows\\marg\\train.py", "set CHAIN_ENV=V3_CFG,LR,TF32,CHAIN_ENV"],
        "s16_v_eval_v3": ["rem S1-6: selected v3 (out\\v3sel\\cfgC_lr1e-3.pt) on 60 test seeds, DDIM 25, 50 paths -> out\\marginal\\v3.npz (+ clim, clock)"],
        "s16_w_eval_marg": ["rem S1-6: marginal control (out\\marginal\\marg.pt) on the same 60 test seeds -> out\\marginal\\marg.npz"],
        "s16_z_summary": ["rem S1-6 summary -> out\\marginal\\summary_endurance_murky.json, done.json"],
    }
    cmd = {"s16_m_train": f"{PY} -u windows\\train_chain.py diff {PROF} {NOISE} {STEPS} {OUTW}\\marg.pt --job s16_m_train --done {OUTW}\\train_done.json",
           "s16_v_eval_v3": f"{PY} -u windows\\marg_run.py eval v3",
           "s16_w_eval_marg": f"{PY} -u windows\\marg_run.py eval marg",
           "s16_z_summary": f"{PY} -u windows\\marg_run.py summary"}
    for job, rem in J.items():
        lines = [head[0]] + rem + head[1:] + [cmd[job]]
        with open(os.path.join(jobdir, job + ".cmd"), "w", newline="\r\n") as fh:
            fh.write("\n".join(lines) + "\n")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def deploy():
    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "s16_marg_deploy")
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    with open(os.path.join(tmp, "SHA256SUMS"), "w") as fh:
        for f in FILES:
            shutil.copy(os.path.join(FC, f), tmp); fh.write(f"{sha(os.path.join(FC, f))}  {f}\n")
    subprocess.run(["ssh", "w", r"if not exist D:\LastEmbers\e1\ml\forecast\windows\marg\out mkdir D:\LastEmbers\e1\ml\forecast\windows\marg\out"], check=True)
    subprocess.run(["scp", "-q"] + [os.path.join(tmp, f) for f in FILES + ("SHA256SUMS",)] + ["w:D:/LastEmbers/e1/ml/forecast/windows/marg/"], check=True)
    subprocess.run(["scp", "-q", os.path.abspath(__file__), "w:D:/LastEmbers/e1/ml/forecast/windows/"], check=True)
    print(open(os.path.join(tmp, "SHA256SUMS")).read())


def verify():
    sums = dict(ln.split()[::-1] for ln in open(os.path.join(MARGD, "SHA256SUMS")) if ln.strip())
    for f in FILES:
        h = sha(os.path.join(MARGD, f))
        assert sums[f] == h, (f, sums[f], h)
    print("marg code ok", flush=True)


def ev(meth, name, model=None, extra=None):
    dst = os.path.join(OUT, f"{name}.npz")
    env = {**os.environ, "PYTHONPATH": FC, "DEVICE": os.environ.get("DEVICE", "cuda"), "DDIM": str(DDIM), "PYTHONUTF8": "1"}
    env.pop("V3_CFG", None)
    env.update(extra or {})
    os.makedirs(os.path.join(MARGD, "out"), exist_ok=True)
    args = [sys.executable, os.path.join(MARGD, "evaluate.py"), meth, PROF, NOISE, str(NS)] + ([model] if model else [])
    t = time.time()
    subprocess.run(args, cwd=FC, check=True, env=env)
    shutil.move(os.path.join(MARGD, "out", f"{meth}_{PROF}_{NOISE}.npz"), dst)
    secs = round(time.time() - t)
    print(name, f"{secs}s", flush=True)
    return secs


def evaluate(which):
    verify()
    os.makedirs(OUT, exist_ok=True)
    tim = {}
    if which == "v3":
        for b in ("clim", "clock"):
            tim[b] = ev(b, b)
        tim["v3"] = ev("diff", "v3", V3PT, {"V3_CFG": "C"})
    else:
        tim["marg"] = ev("diff", "marg", os.path.join(OUT, "marg.pt"), {"V3_CFG": "M"})
    f = os.path.join(OUT, "eval_secs.json")
    d = json.load(open(f)) if os.path.exists(f) else {}
    d.update(tim); json.dump(d, open(f, "w"), indent=1)


def summary():
    from common import issue_hours, load, true_path
    from metrics import LEADS, boot_ci, brier, crps_cat, energy_score, logloss, path_vec, variogram_score
    sys.path.insert(0, MARGD)  # the deployed models.py (V3_CONFIGS["M"])
    verify()
    from models import v3_net
    pt = os.path.join(OUT, "marg.pt")
    ch = json.load(open(pt + ".chain.json"))
    recs = [json.loads(ln) for ln in open(pt + ".log")]
    trained = os.path.exists(pt) and ch["segments"][-1]["steps_to"] == STEPS and recs[-1]["it"] == STEPS
    best = min(recs, key=lambda r: r["val"])
    Z = {m: dict(np.load(os.path.join(OUT, f"{f}.npz"))) for m, f in (("marg", "marg"), ("diff", "v3"), ("clim", "clim"), ("clock", "clock"))}
    for m in Z:
        assert (Z[m]["seeds"] == Z["clim"]["seeds"]).all() and (Z[m]["Y"] == Z["clim"]["Y"]).all(), m
    ns = len(Z["clim"]["seeds"])
    evaluated = ns == NS and Z["marg"]["paths"].shape[:3] == Z["diff"]["paths"].shape[:3] == (NS, len(issue_hours()), 50)

    def bss(z, zb, L):  # summarize.py: per-seed mean Brier, 1000 seed bootstraps, rng seed 0
        bm = brier(z["P"], z["Y"], L).mean(1); bb = brier(zb["P"], zb["Y"], L).mean(1)
        rng = np.random.default_rng(0); bs = []
        for _ in range(1000):
            k = rng.integers(0, ns, ns); bs.append(1 - bm[k].mean() / bb[k].mean())
        return [float(1 - bm.mean() / bb.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    res = {"marg": {}, "diff": {}, "diff_shuf": {}}
    for m in ("marg", "diff", "clim", "clock"):
        z = Z[m]
        res[m] = {f"BSS{L}_{b}": bss(z, Z[b], L) for L in LEADS for b in ("clim", "clock")}
        res[m]["CRPS"] = list(map(float, boot_ci([crps_cat(z["P"][i], z["Y"][i]) for i in range(ns)])))
        res[m]["LL"] = list(map(float, boot_ci([logloss(z["P"][i], z["Y"][i]) for i in range(ns)])))
    D = load(PROF, NOISE, "test"); ih = issue_hours()
    rng = np.random.default_rng(0)
    sc = {"marg": ([], []), "diff": ([], []), "diff_shuf": ([], [])}
    for i, s in enumerate(Z["clim"]["seeds"]):
        ys = [path_vec(true_path(D[s], h)) for h in ih]
        pm = Z["marg"]["paths"][i].astype(np.float32); pd = Z["diff"]["paths"][i].astype(np.float32)  # (ni, 50, 168, 3)
        perm = np.argsort(rng.random(pd.shape[:3]), axis=1)                                            # per issue, per hour
        psh = np.take_along_axis(pd, perm[..., None], axis=1)
        for m, pp in (("marg", pm), ("diff", pd), ("diff_shuf", psh)):
            e1, v1 = [], []
            for j in range(len(ih)):
                X = path_vec(pp[j]); e1.append(energy_score(X, ys[j])); v1.append(variogram_score(X, ys[j]))
            sc[m][0].append(np.array(e1)); sc[m][1].append(np.array(v1))
        print("paths", i, s, flush=True)
    for m, (es, vs) in sc.items():
        res[m]["ES"] = list(map(float, boot_ci(es))); res[m]["VS"] = list(map(float, boot_ci(vs)))
    # diff_shuf: same per-hour samples as diff, so its event probabilities would only change through the flag-run readout;
    # it is a path-score diagnostic only (BSS keys copied from diff for the table layout would mislead, so none)
    secs = json.load(open(os.path.join(OUT, "eval_secs.json")))
    summary = {**res, "seeds": ns, "seed_ids": Z["clim"]["seeds"].tolist(), "ddim": DDIM, "paths": 50,
               "params": {"marg": sum(q.numel() for q in v3_net("M").parameters()), "diff": sum(q.numel() for q in v3_net("C").parameters())},
               "models": {"marg": "out/marginal/marg.pt (models.V3_CONFIGS['M'], best val)",
                          "diff": "out/v3sel/cfgC_lr1e-3.pt (S1-3 selected v3, best val at 91k)",
                          "diff_shuf": "diff's 50 paths, sample index permuted independently per issue and hour (rng 0)"},
               "train": {"steps": STEPS, "best_step": best["it"], "best_val_loss": best["val"], "final_val_loss": recs[-1]["val"],
                         "train_secs": recs[-1]["secs"], "sec_per_step": round(recs[-1]["secs"] / STEPS, 4), "segments": ch["segments"],
                         "note": "val loss includes the event channel (4 channels): not comparable with v3's 0.1551",
                         "curve": [[r["it"], r["val"]] for r in recs]},
               "eval_secs": secs, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(summary, open(os.path.join(OUT, f"summary_{PROF}_{NOISE}.json"), "w"), indent=1)
    done = {"trained": bool(trained), "steps": recs[-1]["it"], "best_step": best["it"], "evaluated": bool(evaluated), "seeds": ns,
            "marg": {k: res["marg"][k][0] for k in ("BSS72_clim", "BSS168_clim", "ES", "VS")},
            "diff": {k: res["diff"][k][0] for k in ("BSS72_clim", "BSS168_clim", "ES", "VS")},
            "diff_shuf": {k: res["diff_shuf"][k][0] for k in ("ES", "VS")}, "finished": summary["finished"]}
    json.dump(done, open(os.path.join(OUT, "done.json"), "w"), indent=1)
    print(json.dumps(done), flush=True)


def paired():
    """Paired differences on the same 60 seeds (summary CIs are per method): per-seed mean Brier / ES / VS, 1000 seed
    bootstraps (rng 0, same resampled seeds for both methods). BSS difference = BSS(a) - BSS(b) vs climatology."""
    from common import issue_hours, load, true_path
    from metrics import brier, energy_score, path_vec, variogram_score
    Z = {m: dict(np.load(os.path.join(OUT, f"{f}.npz"))) for m, f in (("marg", "marg"), ("diff", "v3"), ("clim", "clim"))}
    ns = len(Z["clim"]["seeds"]); D = load(PROF, NOISE, "test"); ih = issue_hours()
    bm = {m: {L: brier(Z[m]["P"], Z[m]["Y"], L).mean(1) for L in (72, 168)} for m in Z}
    rng = np.random.default_rng(0)
    sc = {m: ([], []) for m in ("marg", "diff", "diff_shuf")}
    for i, s in enumerate(Z["clim"]["seeds"]):  # identical to summary(): same rng stream for the shuffle
        ys = [path_vec(true_path(D[s], h)) for h in ih]
        pm = Z["marg"]["paths"][i].astype(np.float32); pd = Z["diff"]["paths"][i].astype(np.float32)
        perm = np.argsort(rng.random(pd.shape[:3]), axis=1)
        psh = np.take_along_axis(pd, perm[..., None], axis=1)
        for m, pp in (("marg", pm), ("diff", pd), ("diff_shuf", psh)):
            sc[m][0].append(np.mean([energy_score(path_vec(pp[j]), ys[j]) for j in range(len(ih))]))
            sc[m][1].append(np.mean([variogram_score(path_vec(pp[j]), ys[j]) for j in range(len(ih))]))
    sc = {m: (np.array(e), np.array(v)) for m, (e, v) in sc.items()}
    rng = np.random.default_rng(0); K = [rng.integers(0, ns, ns) for _ in range(1000)]

    def ci(f):
        b = [f(k) for k in K]
        return [float(f(np.arange(ns))), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)),
                float(np.mean(np.array(b) > 0))]
    out = {"note": "[point, 2.5%, 97.5%, share of bootstraps > 0]; a - b on the same seeds", "seeds": ns}
    for L in (72, 168):
        out[f"BSS{L}_marg_minus_diff"] = ci(lambda k: (1 - bm["marg"][L][k].mean() / bm["clim"][L][k].mean())
                                            - (1 - bm["diff"][L][k].mean() / bm["clim"][L][k].mean()))
    for t, x in (("ES", 0), ("VS", 1)):
        for a in ("marg", "diff_shuf"):
            out[f"{t}_{a}_minus_diff"] = ci(lambda k: sc[a][x][k].mean() - sc["diff"][x][k].mean())
        out[f"{t}_marg_minus_diff_shuf"] = ci(lambda k: sc["marg"][x][k].mean() - sc["diff_shuf"][x][k].mean())
    json.dump(out, open(os.path.join(OUT, "paired.json"), "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)


def export(dst):
    os.makedirs(dst, exist_ok=True)
    for f in ("summary_endurance_murky.json", "done.json", "paired.json", "eval_secs.json", "train_done.json", "marg.pt", "marg.pt.log", "marg.pt.chain.json"):
        subprocess.run(["scp", "-q", f"w:D:/LastEmbers/e1/out/marginal/{f}", os.path.join(dst, f)], check=True)
    print("exported to", dst)


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    if a[0] == "selftest":
        selftest()
    elif a[0] == "jobs":
        jobs(a[1] if len(a) > 1 else os.path.join(HERE, "jobs"))
    elif a[0] == "deploy":
        deploy()
    elif a[0] == "verify":
        verify()
    elif a[0] == "eval":
        assert a[1] in ("v3", "marg"), a[1]
        evaluate(a[1])
    elif a[0] == "summary":
        summary()
    elif a[0] == "paired":
        paired()
    elif a[0] == "export":
        export(a[1])
    else:
        sys.exit(__doc__)
