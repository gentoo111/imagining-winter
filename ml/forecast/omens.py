"""Folk omens vs pressure signs in the frozen weather (339afa0 / 4195eb6 rules), Endurance murky.

Data: weather_gen (e1 branch, same weather as the frozen RL simulator), 200 fresh seeds x 120 days:
  simrs/target/release/weather_gen data ml/forecast/data/endurance_murky_omen.bin 400000 200 120 endurance murky
For every calm hour (not in a storm) at which a sign is showing, the rate at which a storm lands within the next
24 / 72 h, against the base rate over all calm hours. Daily signs (animals, sky) are set at 07:00 and held all day;
they are scored at the 07:00 issue (rows `07:00`) and at every calm hour (rows `hourly`).
  omens.py [file] -> out/omens.json"""
import json, os, sys
import numpy as np
from common import FI, H, HERE

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data", "endurance_murky_omen.bin")
D = np.fromfile(path, dtype="<i4").reshape(-1, H, 15).astype(np.int64)
n = D.shape[0]
land = D[..., FI["landed"]]; ins = D[..., FI["in_storm"]]
p = D[..., FI["pressure"]] / 10.0; trend = D[..., FI["trend"]] / 10.0; wind = D[..., FI["wind_ms"]]
sky = D[..., FI["sky"]]; ani = D[..., FI["animals"]]
hour = np.arange(1, H + 1)
seven = (hour * 60) % 1440 == 7 * 60


def within(L):
    """landing in hours h+1 .. h+L (row index t = h-1: rows t+1 .. t+L)."""
    c = np.concatenate([np.zeros((n, 1), int), np.cumsum(land, 1)], 1)  # c[:, t] = landings in rows < t
    out = np.full((n, H), -1, int)
    t = np.arange(H - L)
    out[:, t] = (c[:, t + 1 + L] - c[:, t + 1]) > 0
    return out


p24min = np.full_like(p, np.nan)
for t in range(H):
    p24min[:, t] = p[:, max(0, t - 23):t + 1].min(1)
signs = {
    "halo (sky=1)": sky == 1,
    "leaden sky (sky=2)": sky == 2,
    "animal tracks fewer (animals=1)": ani == 1,
    "animals gone (animals=2)": ani == 2,
    "game warning rule": (sky == 2) | ((trend <= -8) & (wind >= 9)),
    "pressure < 997 hPa": p < 997,
    "24 h trend <= -8 hPa": trend <= -8,
    "24 h-low rebound (p >= 24 h min + 4 hPa, min < 1005)": (p >= p24min + 4) & (p24min < 1005),
}
res = {"file": path, "seeds": n, "rates": {}}
for L in (24, 72):
    w = within(L)
    calm = (ins == 0) & (w >= 0)
    for scope, sel in (("07:00", calm & seven[None]), ("hourly", calm)):
        base = w[sel].mean()
        rows = {"base": [float(base), int(sel.sum())]}
        for k, s in signs.items():
            m = sel & s
            rows[k] = [float(w[m].mean()) if m.any() else None, int(m.sum())]
        res["rates"][f"{L}h {scope}"] = rows
        print(f"\n== storm lands within {L} h, calm {scope} hours (n={sel.sum()}): base {100 * base:.1f}%")
        for k, (v, c) in rows.items():
            if k != "base":
                print(f"  {k:52} {('%.1f%%' % (100 * v)) if v is not None else '--':>7}  (n={c}, x{(v or 0) / base:.2f})")
json.dump(res, open(os.path.join(HERE, "out", "omens.json"), "w"), indent=1)
