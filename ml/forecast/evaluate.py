"""E1 evaluation on test seeds: PF oracle, climatology, clock, bot rule, GRU, diffusion.
  evaluate.py <method> <profile> <noise> <n_seeds> [extra]  -> out/<method>_<profile>_<noise>.npz
method in: pf, clim, clock, bot, gru, diff
S1-5 options (all unset = unchanged): NSAMP=<paths per forecast> (default 50), SEEDS=a:b (test seeds a..b-1 only),
GEN_SEED=<diffusion noise generator seed> (default 0), EVAL_CHUNK=<issue hours sampled at a time> (always on for V3_CFG),
TF32=1 (allow TF32 on CUDA), PATH_SCORES=1 (store per-issue energy / variogram scores of all paths on the 21-d daily means,
the daily-mean paths `pvec` (float16) and the hourly paths of the first seed `paths0` instead of all hourly paths),
EVAL_OUT=<npz path>.
diff with V3_CFG=A|B|C|M (S1-6; unset = v2 UNet1D, unchanged): loads models.v3_net(V3_CFG) and samples in chunks of
EVAL_CHUNK issue hours (default 38); M = marginal control: event probabilities from its per-lead event channel
(models.marg_event_probs), paths = its first 3 channels.
S1-9 options: P_SUBSETS=50,200 (also store P_n<k> = the readout from the first k of the NSAMP paths of each forecast; the
paths are i.i.d. given the forecast's conditioning, so the first k are a k-sample draw, nested in the full one),
NO_PATHS=1 (store no hourly paths)."""
import sys, os, time
import numpy as np
from common import *
from metrics import *
meth, prof, noise, ns = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
part = os.environ.get("PART")  # "k/n" splits seeds for parallel workers
D = load(prof, noise, "test")[:ns]
seeds = np.arange(ns)
if part:
    k, n = map(int, part.split("/")); seeds = seeds[k::n]
if os.environ.get("SEEDS"):
    assert not part, "SEEDS and PART are exclusive"
    a_, b_ = map(int, os.environ["SEEDS"].split(":")); assert 0 <= a_ < b_ <= ns; seeds = np.arange(a_, b_)
ih = issue_hours()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", f"{meth}_{prof}_{noise}" + (f"_p{part.replace('/', 'of')}" if part else "") + ".npz")
out = os.environ.get("EVAL_OUT") or out
PSC = os.environ.get("PATH_SCORES") == "1"
SUBS = [int(k) for k in os.environ.get("P_SUBSETS", "").split(",") if k]
NOP = os.environ.get("NO_PATHS") == "1"
t0 = time.time()
P = np.zeros((len(seeds), len(ih), NB + 1), np.float32); Y = np.zeros((len(seeds), len(ih)), np.int64)
PATHS = None
for j, s in enumerate(seeds):
    Y[j] = [true_class(D[s], h) for h in ih]
if meth in ("clim", "clock", "bot"):
    import baselines as B
    tr = load(prof, noise, "train")[:2000]
    kf = {"clim": B.clim_key, "clock": B.clock_key, "bot": B.bot_key}[meth]
    tab = B.Table(kf).fit(tr, prof, ih)
    for j, s in enumerate(seeds):
        P[j] = [tab.predict(D[s], h, prof) for h in ih]
elif meth == "pf":
    import pf
    N = int(os.environ.get("PF_N", "4096")); jit = float(os.environ.get("PF_JIT", "0.03")); rs = float(os.environ.get("PF_RS", "0.01"))
    bank = pf.build_bank(100000)
    rng = np.random.default_rng(1000 + int(seeds[0]))
    PATHS = np.zeros((len(seeds), len(ih), 50, HOR, 3), np.float16)
    for j, s in enumerate(seeds):
        o = pf.run_seed(D[s].astype(np.int64), prof, noise, N, jit, rs, bank, rng, issue=ih)
        P[j] = o["probs"]; PATHS[j] = o["paths"]
        print(j, s, f"{time.time()-t0:.0f}s lost={o['lost']}", flush=True)
elif meth in ("gru", "diff"):
    import torch
    DEVICE = os.environ.get("DEVICE", "cpu")  # cpu = unchanged behaviour
    if DEVICE != "cpu":
        torch.set_default_device(DEVICE)  # before importing models (Diffusion tables, noise)
    from batch import Split
    from models import GRUCat, UNet1D, Diffusion, diff_variant, v3_net, MARG_CFG, marg_event_probs
    uk, dk = diff_variant()  # DIFF_VARIANT=A..D (S1-7 ablation); unset = D = v2
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
    if os.environ.get("TF32") == "1":
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    sp = Split(prof, noise, "test", ns)
    V3_CFG = os.environ.get("V3_CFG") if meth == "diff" else None
    MARG = V3_CFG == MARG_CFG
    if V3_CFG:
        uk, dk = {}, ({"nch": 4} if MARG else {})  # v3 / marginal control: v2 loss and sampler settings
    net = GRUCat() if meth == "gru" else (v3_net(V3_CFG) if V3_CFG else UNet1D(**uk))
    net.load_state_dict(torch.load(sys.argv[5], map_location=DEVICE)); net.eval()
    ihs = np.array(ih)
    if meth == "diff":
        dif = Diffusion(**dk); g = torch.Generator(device=DEVICE).manual_seed(int(os.environ.get("GEN_SEED", "0")))
        M = int(os.environ.get("NSAMP", "50"))
        assert all(0 < k < M for k in SUBS), (SUBS, M)
        PSUB = {k: np.zeros((len(seeds), len(ih), NB + 1), np.float32) for k in SUBS}
        if PSC:
            ES_ = np.zeros((len(seeds), len(ih)), np.float64); VS_ = np.zeros_like(ES_)
            PVEC = np.zeros((len(seeds), len(ih), M, 21), np.float16); PATHS0 = None
        elif not NOP:
            PATHS = np.zeros((len(seeds), len(ih), M, HOR, 3), np.float16)
    for j, s in enumerate(seeds):
        x, c, cl, _ = sp.batch(np.full(len(ih), s), ihs, want_future=False)
        x = torch.tensor(x); c = torch.tensor(c)
        with torch.no_grad():
            if meth == "gru":
                P[j] = torch.softmax(net(x, c), -1).cpu().numpy()
            else:
                steps_ = int(os.environ.get("DDIM", "20"))
                if V3_CFG or os.environ.get("EVAL_CHUNK"):  # chunks of issue hours (8 GB GPU); one generator, chunks in issue order
                    ck = int(os.environ.get("EVAL_CHUNK", "38")); parts = []
                    for a0 in range(0, len(ih), ck):
                        cv = tuple(None if a is None else a.repeat_interleave(M, 0) for a in net.cond_vec(x[a0:a0 + ck], c[a0:a0 + ck]))
                        parts.append(dif.sample(net, cv, steps=steps_, gen=g).cpu().numpy())
                    yy = np.concatenate(parts)
                else:
                    cv = tuple(None if a is None else a.repeat_interleave(M, 0) for a in net.cond_vec(x, c))  # None: v1 has no extra channels
                    yy = dif.sample(net, cv, steps=steps_, gen=g).cpu().numpy()  # (n*M,3,HOR)
                yy = yy.reshape(len(ih), M, dif.nch, HOR).transpose(0, 1, 3, 2)
                ins0 = c[:, 3].cpu().numpy() > 0
                # a landing = the flag turns on and stays on for >= 3 h (storms last 48 h); single-hour blips are noise
                raw = yy[..., 2] > 0
                fl = raw & np.concatenate([raw[..., 1:], raw[..., -1:]], -1) & np.concatenate([raw[..., 2:], raw[..., -2:]], -1)
                prev = np.concatenate([np.broadcast_to(ins0[:, None, None], (len(ih), M, 1)), fl[..., :-1]], -1)
                edge = fl & ~prev
                first = np.where(edge.any(-1), edge.argmax(-1) + 1, 0)
                cls = np.where(first > 0, (first - 1) // 6, NB)
                for i in range(len(ih)):
                    P[j, i] = np.bincount(cls[i], minlength=NB + 1) / M
                    for k in SUBS:
                        PSUB[k][j, i] = np.bincount(cls[i, :k], minlength=NB + 1) / k
                if MARG:  # no path to read the landing from: per-lead event channel
                    P[j] = marg_event_probs(yy[..., 3])
                phys = np.stack([yy[..., 0] * 10.0 + 1013.0, yy[..., 1] * 10.0, (yy[..., 2] > 0).astype(np.float32)], -1)
                if PSC:  # summarize.py's path scores, here so the 200 x 200 hourly paths need not be stored
                    pv = path_vec(phys.astype(np.float16).astype(np.float32))  # (ni, M, 21); float16 as stored paths
                    for i, h in enumerate(ih):
                        y_ = path_vec(true_path(D[s], h))
                        ES_[j, i] = energy_score(pv[i], y_); VS_[j, i] = variogram_score(pv[i], y_)
                    PVEC[j] = pv
                    if PATHS0 is None:
                        PATHS0 = phys.astype(np.float16)
                elif not NOP:
                    PATHS[j] = phys
        print(j, s, f"{time.time()-t0:.0f}s", flush=True)
if meth == "diff" and PSC:
    np.savez_compressed(out, P=P, Y=Y, seeds=seeds, paths=np.zeros(0), es=ES_, vs=VS_, pvec=PVEC, paths0=PATHS0,
                        nsamp=M, ddim=steps_)
elif meth == "diff" and (SUBS or NOP):
    np.savez_compressed(out, P=P, Y=Y, seeds=seeds, paths=PATHS if PATHS is not None else np.zeros(0), nsamp=M, ddim=steps_,
                        **{f"P_n{k}": v for k, v in PSUB.items()})
else:
    np.savez_compressed(out, P=P, Y=Y, seeds=seeds, paths=PATHS if PATHS is not None else np.zeros(0))
print("saved", out, f"{time.time()-t0:.0f}s")
