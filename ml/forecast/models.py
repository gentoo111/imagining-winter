import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from feats import NFEAT, NCTX
from common import NB, HOR


class Encoder(nn.Module):
    """GRU over the precursor history + context -> vector."""
    def __init__(self, hid=96, out=128):
        super().__init__()
        self.inp = nn.Linear(NFEAT, hid)
        self.gru = nn.GRU(hid, hid, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid + NCTX, out), nn.SiLU(), nn.Linear(out, out))

    def forward(self, x, c):
        h, _ = self.gru(Fn.silu(self.inp(x)))
        return self.head(torch.cat([h[:, -1], c], -1))


class ConvEncoder(nn.Module):
    """Strided 1-D conv over the 168 h history (fast on CPU) + context -> vector."""
    def __init__(self, out=96):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(NFEAT, 32, 5, stride=2, padding=2), nn.SiLU(),
                                 nn.Conv1d(32, 48, 5, stride=2, padding=2), nn.SiLU(),
                                 nn.Conv1d(48, 64, 5, stride=2, padding=2), nn.SiLU(),
                                 nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.SiLU())   # 168->11
        self.head = nn.Sequential(nn.Linear(64 * 11 + NCTX + NFEAT * 6, out), nn.SiLU(), nn.Linear(out, out))

    def forward(self, x, c):
        h = self.net(x.transpose(1, 2)).flatten(1)
        last = x[:, -6:].flatten(1)   # the last 6 hours verbatim
        return self.head(torch.cat([h, c, last], -1))


class GRUCat(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(128, NB + 1))

    def forward(self, x, c):
        return self.out(self.enc(x, c))


def temb(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    a = t[:, None].float() * f[None]
    return torch.cat([a.sin(), a.cos()], -1)


class Res(nn.Module):
    def __init__(self, ch, cond):
        super().__init__()
        self.n1 = nn.GroupNorm(8, ch); self.c1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(8, ch); self.c2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.film = nn.Linear(cond, 2 * ch)

    def forward(self, x, e):
        h = self.c1(Fn.silu(self.n1(x)))
        sc, sh = self.film(e).chunk(2, -1)
        h = self.n2(h) * (1 + sc[..., None]) + sh[..., None]
        return x + self.c2(Fn.silu(h))


class UNet1D(nn.Module):
    """Denoiser over the 7-day hourly future (3 channels, length 168), conditioned on the encoder vector.
    extra=False is the v1 denoiser (no hour-position / known-storm-remainder input channels; S1-7 ablation)."""
    def __init__(self, ch=(32, 48, 64), cond=96, extra=True):
        super().__init__()
        self.extra = extra
        self.enc = ConvEncoder(out=cond)
        self.tproj = nn.Sequential(nn.Linear(cond, cond), nn.SiLU(), nn.Linear(cond, cond))
        self.inc = nn.Conv1d(5 if extra else 3, ch[0], 3, padding=1)
        self.d0 = Res(ch[0], cond); self.down0 = nn.Conv1d(ch[0], ch[1], 4, stride=2, padding=1)   # 168->84
        self.d1 = Res(ch[1], cond); self.down1 = nn.Conv1d(ch[1], ch[2], 4, stride=2, padding=1)   # 84->42
        self.m1 = Res(ch[2], cond)
        self.up1 = nn.ConvTranspose1d(ch[2], ch[1], 4, stride=2, padding=1); self.u1 = Res(ch[1], cond)
        self.up0 = nn.ConvTranspose1d(ch[1], ch[0], 4, stride=2, padding=1); self.u0 = Res(ch[0], cond)
        self.out = nn.Sequential(nn.GroupNorm(8, ch[0]), nn.SiLU(), nn.Conv1d(ch[0], 3, 3, padding=1))
        self.cond = cond

    def cond_vec(self, x, c):
        """(vector, extra channels): extra = hour position 1..168 (scaled) and the known rest of a storm under way."""
        if not self.extra:
            return self.enc(x, c), None
        B = x.shape[0]
        k = torch.arange(1, HOR + 1, dtype=torch.float32)[None].expand(B, HOR)
        left = (c[:, 4] * 120.0)[:, None]                      # hours left in the current storm (0 if calm)
        known = (k < left).float() * 2 - 1
        extra = torch.stack([k / HOR * 2 - 1, known], 1)
        return self.enc(x, c), extra

    def forward(self, y, t, cv):
        cvec, extra = cv
        e = cvec + self.tproj(temb(t, self.cond))
        h0 = self.d0(self.inc(y if extra is None else torch.cat([y, extra], 1)), e)
        h1 = self.d1(self.down0(h0), e)
        m = self.m1(self.down1(h1), e)
        u1 = self.u1(self.up1(m) + h1, e)
        u0 = self.u0(self.up0(u1) + h0, e)
        return self.out(u0)


def _cond_extra(x, c):
    """Extra denoiser input channels shared by v2 / v3: hour position 1..168 (scaled) and the known rest of a storm."""
    B = x.shape[0]
    k = torch.arange(1, HOR + 1, dtype=torch.float32)[None].expand(B, HOR)
    left = (c[:, 4] * 120.0)[:, None]
    known = (k < left).float() * 2 - 1
    return torch.stack([k / HOR * 2 - 1, known], 1)


class Attn1D(nn.Module):
    """Pre-norm multi-head self-attention over the time axis of a (B, ch, L) feature map, residual."""
    def __init__(self, ch, heads=4):
        super().__init__()
        self.n = nn.GroupNorm(8, ch)
        self.a = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x):
        h = self.n(x).transpose(1, 2)
        return x + self.a(h, h, h, need_weights=False)[0].transpose(1, 2)


class UNet1Dv3(nn.Module):
    """S1-3 v3 U-Net denoiser: same 3-level layout as UNet1D (168->84->42), wider channels, `nres` residual blocks per
    level and self-attention in the middle (42 positions). Same inputs/outputs as UNet1D (extra channels on)."""
    def __init__(self, ch=(64, 96, 128), cond=256, nres=2, heads=4):
        super().__init__()
        self.cond = cond
        self.enc = ConvEncoder(out=cond)
        self.tproj = nn.Sequential(nn.Linear(cond, cond), nn.SiLU(), nn.Linear(cond, cond))
        self.inc = nn.Conv1d(5, ch[0], 3, padding=1)
        R = lambda c: nn.ModuleList([Res(c, cond) for _ in range(nres)])
        self.d0, self.d1 = R(ch[0]), R(ch[1])
        self.down0 = nn.Conv1d(ch[0], ch[1], 4, stride=2, padding=1)
        self.down1 = nn.Conv1d(ch[1], ch[2], 4, stride=2, padding=1)
        self.m0, self.m1 = Res(ch[2], cond), Res(ch[2], cond)
        self.attn = Attn1D(ch[2], heads)
        self.up1 = nn.ConvTranspose1d(ch[2], ch[1], 4, stride=2, padding=1); self.u1 = R(ch[1])
        self.up0 = nn.ConvTranspose1d(ch[1], ch[0], 4, stride=2, padding=1); self.u0 = R(ch[0])
        self.out = nn.Sequential(nn.GroupNorm(8, ch[0]), nn.SiLU(), nn.Conv1d(ch[0], 3, 3, padding=1))

    def cond_vec(self, x, c):
        return self.enc(x, c), _cond_extra(x, c)

    @staticmethod
    def _run(blocks, h, e):
        for b in blocks:
            h = b(h, e)
        return h

    def forward(self, y, t, cv):
        cvec, extra = cv
        e = cvec + self.tproj(temb(t, self.cond))
        h0 = self._run(self.d0, self.inc(torch.cat([y, extra], 1)), e)
        h1 = self._run(self.d1, self.down0(h0), e)
        m = self.m1(self.attn(self.m0(self.down1(h1), e)), e)
        u1 = self._run(self.u1, self.up1(m) + h1, e)
        u0 = self._run(self.u0, self.up0(u1) + h0, e)
        return self.out(u0)


class DiTBlock(nn.Module):
    """Transformer block with adaLN-zero conditioning (Peebles & Xie 2023): shift/scale/gate from the cond vector."""
    def __init__(self, d, heads, cond, ff=4, diag=False):
        super().__init__()
        self.diag = diag  # S1-6 marginal control: each token attends only to itself (no information across lead times)
        self.n1 = nn.LayerNorm(d, elementwise_affine=False); self.n2 = nn.LayerNorm(d, elementwise_affine=False)
        self.a = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, ff * d), nn.GELU(), nn.Linear(ff * d, d))
        self.ada = nn.Linear(cond, 6 * d)
        nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)

    def forward(self, h, e):
        s1, b1, g1, s2, b2, g2 = self.ada(Fn.silu(e))[:, None].chunk(6, -1)
        x = self.n1(h) * (1 + s1) + b1
        m = None
        if self.diag:
            L = h.shape[1]
            m = ~torch.eye(L, dtype=torch.bool, device=h.device)  # True = blocked: only the diagonal is open
        h = h + g1 * self.a(x, x, x, need_weights=False, attn_mask=m)[0]
        return h + g2 * self.mlp(self.n2(h) * (1 + s2) + b2)


class TransDenoiser(nn.Module):
    """S1-3 v3 transformer denoiser: one token per future hour (168 tokens: 3 noisy channels + 2 extra channels),
    learned position embedding, `layers` adaLN-zero blocks conditioned on encoder vector + diffusion time.
    S1-6 marginal control (defaults = v3 unchanged): nch=4 adds a per-hour event channel, diag=True blocks attention
    between hours, so given the condition every lead time is denoised (and sampled) independently of the others."""
    def __init__(self, d=192, layers=4, heads=6, cond=256, nch=3, diag=False):
        super().__init__()
        self.cond = cond
        self.enc = ConvEncoder(out=cond)
        self.tproj = nn.Sequential(nn.Linear(cond, cond), nn.SiLU(), nn.Linear(cond, cond))
        self.inp = nn.Linear(nch + 2, d)
        self.pos = nn.Parameter(torch.zeros(1, HOR, d)); nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([DiTBlock(d, heads, cond, diag=diag) for _ in range(layers)])
        self.nf = nn.LayerNorm(d, elementwise_affine=False)
        self.ada = nn.Linear(cond, 2 * d); nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)
        self.out = nn.Linear(d, nch)

    def cond_vec(self, x, c):
        return self.enc(x, c), _cond_extra(x, c)

    def forward(self, y, t, cv):
        cvec, extra = cv
        e = cvec + self.tproj(temb(t, self.cond))
        h = self.inp(torch.cat([y, extra], 1).transpose(1, 2)) + self.pos
        for b in self.blocks:
            h = b(h, e)
        s, sh = self.ada(Fn.silu(e))[:, None].chunk(2, -1)
        return self.out(self.nf(h) * (1 + s) + sh).transpose(1, 2)


# S1-3 v3 candidates (RESEARCH-PLAN S1-3: channels 64/96/128 + mid attention, or a 4-layer transformer; at most 3).
V3_CONFIGS = {
    "A": (UNet1Dv3, dict(ch=(64, 96, 128), cond=256, nres=3)),
    "B": (UNet1Dv3, dict(ch=(96, 144, 192), cond=256, nres=1)),
    "C": (TransDenoiser, dict(d=160, layers=4, heads=5, cond=256)),
    # S1-6 marginal control: cfgC unchanged except attention restricted to the diagonal + per-lead event channel
    "M": (TransDenoiser, dict(d=160, layers=4, heads=5, cond=256, nch=4, diag=True)),
}
MARG_CFG = "M"


def event_channel(cls):
    """S1-6: per-hour event target (B,) landing class -> (B, HOR, 1) float32, +1 at future hour j (1-based) if the next
    landing falls in a 6 h bin <= (j-1)//6 (i.e. by the end of hour j's bin), else -1. At j = 6k it is [class < k]."""
    b = np.arange(HOR)[None] // 6
    return np.where(np.asarray(cls)[:, None] <= b, 1.0, -1.0).astype(np.float32)[..., None]


def marg_event_probs(e):
    """S1-6: (n, M, HOR) sampled event channel -> (n, NB+1) class probabilities. P(landing <= 6k) = share of the M
    samples with the channel > 0 at hour 6k, made monotone in k by a running max; class k-1 = increment, NB = rest."""
    cum = (e[..., 5::6] > 0).mean(1)                      # (n, NB): leads 6, 12, ..., 168
    cum = np.maximum.accumulate(cum, -1)
    p = np.diff(np.concatenate([np.zeros_like(cum[:, :1]), cum], -1), axis=-1)
    return np.concatenate([p, 1.0 - cum[:, -1:]], -1)


def v3_net(name):
    cls, kw = V3_CONFIGS[name]
    return cls(**kw)


class Diffusion:
    """Cosine-schedule DDPM, DDIM sampling. pred="v" (v2, default) or "eps" (v1); flag_w = storm-flag loss weight
    (v2: 3, v1: 1). pred="eps" reproduces the v1 sampler: x0 from eps, clamp x0, eps not recomputed."""
    def __init__(self, T=200, pred="v", flag_w=3.0, nch=3):
        assert pred in ("v", "eps"), pred
        self.T, self.pred, self.nch = T, pred, nch
        # v2: storm-flag channel weighted x3; S1-6 nch=4: the per-lead event channel (also binary) gets the same weight
        self.W = torch.tensor([1.0, 1.0] + [float(flag_w)] * (nch - 2))[None, :, None]
        s = 0.008
        t = torch.linspace(0, T, T + 1) / T
        ac = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        ac = ac / ac[0]
        self.ab = ac[1:].clamp(1e-5, 1.0)  # alpha_bar at t = 1..T (index t-1)

    def loss(self, net, y0, cv):
        """v-prediction (Salimans & Ho 2022), or epsilon prediction for v1."""
        B = y0.shape[0]
        t = torch.randint(1, self.T + 1, (B,))
        ab = self.ab[t - 1][:, None, None]
        eps = torch.randn_like(y0)
        yt = ab.sqrt() * y0 + (1 - ab).sqrt() * eps
        if self.pred == "eps":
            return (((net(yt, t, cv) - eps) ** 2) * self.W).mean()
        v = ab.sqrt() * eps - (1 - ab).sqrt() * y0
        return (((net(yt, t, cv) - v) ** 2) * self.W).mean()

    @torch.no_grad()
    def sample(self, net, cv, steps=20, gen=None):
        B = cv[0].shape[0]
        y = torch.randn(B, self.nch, HOR, generator=gen)
        ts = torch.linspace(self.T, 1, steps).round().long()
        for i, t in enumerate(ts):
            ab = self.ab[t - 1]
            out = net(y, torch.full((B,), int(t)), cv)
            if self.pred == "eps":
                eps = out
                y0 = (y - (1 - ab).sqrt() * eps) / ab.sqrt()
                y0 = torch.stack([y0[:, 0].clamp(-4, 4), y0[:, 1].clamp(-0.5, 4), y0[:, 2].clamp(-1, 1)], 1)
            else:
                y0 = ab.sqrt() * y - (1 - ab).sqrt() * out
                eps = (1 - ab).sqrt() * y + ab.sqrt() * out
                y0 = torch.stack([y0[:, 0].clamp(-4, 4), y0[:, 1].clamp(-0.5, 4)] + [y0[:, k].clamp(-1, 1) for k in range(2, self.nch)], 1)
                eps = (y - ab.sqrt() * y0) / (1 - ab).sqrt().clamp(min=1e-4)
            if i + 1 < len(ts):
                abn = self.ab[ts[i + 1] - 1]
                y = abn.sqrt() * y0 + (1 - abn).sqrt() * eps
            else:
                y = y0
        return y


# S1-7 ablation: the v1 -> v2 fixes added one at a time (D = v2 = the defaults above).
VARIANTS = {
    "A": (dict(extra=False), dict(pred="eps", flag_w=1.0)),   # v1
    "B": (dict(extra=True), dict(pred="eps", flag_w=1.0)),    # + hour-position / known-storm-remainder channels
    "C": (dict(extra=True), dict(pred="v", flag_w=1.0)),      # + v-prediction
    "D": (dict(extra=True), dict(pred="v", flag_w=3.0)),      # + storm-flag loss x3 = v2
}


def diff_variant():
    """(UNet1D kwargs, Diffusion kwargs) for env DIFF_VARIANT (default D = v2, i.e. unchanged behaviour)."""
    return VARIANTS[os.environ.get("DIFF_VARIANT", "D")]
