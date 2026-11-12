"""Train GRU categorical baseline or the diffusion forecaster.
  train.py gru|diff <profile> <noise> <steps> <out.pt>
Resume (S1-3a; all unset = exactly the old behaviour): RESUME_STATE=<file> keeps the full training state
(net, optimizer, scheduler, numpy + torch/CUDA RNG, validation indices, step, best, elapsed) after every
validation and at a stop, and continues from it when the file exists. TIME_BUDGET=<sec> / STOP_AT=<step>
end the process cleanly (state saved, exit 0) after that many loop seconds / that step; windows/train_chain.py
chains such segments through the queue.
S1-3 v3 (all unset = unchanged): V3_CFG=A|B|C trains models.V3_CONFIGS[<cfg>] instead of UNet1D (v2 loss, DIFF_VARIANT
ignored); LR=<float> overrides the peak learning rate; TF32=1 allows TF32 matmul/conv on CUDA.
S1-4 (unset = 0 = unchanged): SEED=<int> sets the torch init seed and, after the fixed validation indices are drawn
from default_rng(0), the training-batch numpy RNG (default_rng(SEED)); the validation set is the same for every seed.
S1-6: V3_CFG=M = the marginal control (models.V3_CONFIGS["M"]): 4 channels, the future target gets the per-lead
event channel models.event_channel(class) appended (training and validation); every other setting is cfgC's."""
import sys, time, json, os
import numpy as np, torch
DEVICE = os.environ.get("DEVICE", "cpu")  # e.g. DEVICE=cuda on the Windows GPU; cpu leaves everything as before
if DEVICE != "cpu":
    torch.set_default_device(DEVICE)  # before importing models: Diffusion's tables and noise follow the default device
from batch import Split
from models import GRUCat, UNet1D, Diffusion, diff_variant, v3_net, MARG_CFG, event_channel
from common import H, HOR, NB
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
VAL_EVERY = int(os.environ.get("VAL_EVERY", "500"))  # validation interval (default unchanged)
CKPT_DIR = os.environ.get("CKPT_DIR")  # if set: also save every validated checkpoint as <dir>/step_<it>.pt (learning curve)
if CKPT_DIR:
    os.makedirs(CKPT_DIR, exist_ok=True)
kind, prof, noise, steps, outp = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
tr = Split(prof, noise, "train"); va = Split(prof, noise, "val")
SEED = int(os.environ.get("SEED", "0"))
rng = np.random.default_rng(0); torch.manual_seed(SEED)
B = 256
uk, dk = diff_variant()  # DIFF_VARIANT=A..D (S1-7 ablation); unset = D = v2
V3_CFG = os.environ.get("V3_CFG")
if V3_CFG:
    assert kind == "diff", "V3_CFG is for the diffusion forecaster"
    uk, dk = {}, {}  # v3 keeps the v2 loss (v-prediction, storm flag x3) so validation losses compare
if os.environ.get("TF32") == "1":
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
net = GRUCat() if kind == "gru" else (v3_net(V3_CFG) if V3_CFG else UNet1D(**uk))
MARG = V3_CFG == MARG_CFG  # S1-6 marginal control
if MARG:
    dk = {"nch": 4}
dif = Diffusion(**dk) if kind == "diff" else None


def with_event(b):
    """(x, c, cls, fut): S1-6 appends the per-lead event channel to the future target; otherwise unchanged."""
    x, c, cl, f = b
    return (x, c, cl, np.concatenate([f, event_channel(cl)], -1) if MARG and f is not None else f)
LR = float(os.environ.get("LR", "0")) or (1e-3 if kind == "gru" else 3e-4)
opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=opt.param_groups[0]["lr"], total_steps=steps, pct_start=0.05)
vs = rng.integers(0, va.D.shape[0], 2048); vh = rng.integers(24, H - HOR + 1, 2048)
vx, vc, vcl, vf = (torch.tensor(a) if a is not None else None for a in with_event(va.batch(vs, vh)))
if SEED:
    rng = np.random.default_rng(SEED)  # training batches; validation indices above stay those of seed 0
if os.environ.get("DETERMINISTIC") == "1":  # opt-in (S1-3a GPU resume smoke): bit-reproducible CUDA kernels (before the first matmul)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False
RESUME_STATE = os.environ.get("RESUME_STATE")
TIME_BUDGET = float(os.environ.get("TIME_BUDGET", "0"))
STOP_AT = int(os.environ.get("STOP_AT", "0"))
assert RESUME_STATE or not (TIME_BUDGET or STOP_AT), "TIME_BUDGET / STOP_AT need RESUME_STATE"
start, elapsed, best = 1, 0.0, 1e9
if RESUME_STATE and os.path.exists(RESUME_STATE):
    st = torch.load(RESUME_STATE, map_location="cpu", weights_only=False)
    assert (st["kind"], st["prof"], st["noise"], st["steps"]) == (kind, prof, noise, steps), "state is for another run"
    assert np.array_equal(st["vs"], vs) and np.array_equal(st["vh"], vh)
    net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
    rng.bit_generator.state = st["np_rng"]; torch.set_rng_state(st["torch_rng"])
    if "cuda_rng" in st:
        torch.cuda.set_rng_state(st["cuda_rng"])
    start, elapsed, best = st["it"] + 1, st["elapsed"], st["best"]
    print(json.dumps({"resumed": st["it"], "elapsed": round(elapsed)}), flush=True)


def save_state(it):
    st = {"it": it, "kind": kind, "prof": prof, "noise": noise, "steps": steps, "net": net.state_dict(),
          "opt": opt.state_dict(), "sched": sched.state_dict(), "np_rng": rng.bit_generator.state,
          "torch_rng": torch.get_rng_state(), "vs": vs, "vh": vh, "best": best, "elapsed": elapsed + time.time() - t0}
    if DEVICE != "cpu" and torch.cuda.is_available():
        st["cuda_rng"] = torch.cuda.get_rng_state()
    torch.save(st, RESUME_STATE + ".tmp"); os.replace(RESUME_STATE + ".tmp", RESUME_STATE)


log = open(outp + ".log", "a"); t0 = time.time()
for it in range(start, steps + 1):
    s = rng.integers(0, tr.D.shape[0], B); h = rng.integers(24, H - HOR + 1, B)
    x, c, cl, f = with_event(tr.batch(s, h, want_future=(kind == "diff")))
    x = torch.tensor(x); c = torch.tensor(c)
    if kind == "gru":
        loss = torch.nn.functional.cross_entropy(net(x, c), torch.tensor(cl))
    else:
        loss = dif.loss(net, torch.tensor(f).transpose(1, 2), net.cond_vec(x, c))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    if it % VAL_EVERY == 0 or it == steps:
        net.eval()
        with torch.no_grad():
            if kind == "gru":
                vl = float(torch.nn.functional.cross_entropy(net(vx, vc), vcl))
            else:
                g = torch.Generator().manual_seed(1)
                vl = float(np.mean([float(dif.loss(net, vf.transpose(1, 2), net.cond_vec(vx, vc))) for _ in range(4)]))
        net.train()
        rec = {"it": it, "loss": float(loss), "val": vl, "secs": round(elapsed + time.time() - t0)}
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        if CKPT_DIR:
            torch.save(net.state_dict(), os.path.join(CKPT_DIR, f"step_{it}.pt"))
        if vl < best:
            best = vl; torch.save(net.state_dict(), outp)
        if RESUME_STATE:
            save_state(it)
    if RESUME_STATE and it < steps and ((STOP_AT and it >= STOP_AT) or (TIME_BUDGET and time.time() - t0 >= TIME_BUDGET)):
        save_state(it)
        print(json.dumps({"stopped": it, "secs": round(elapsed + time.time() - t0)}), flush=True)
        break
