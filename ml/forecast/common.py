"""E1 shared: data loading, exact integer Lorenz, weather rules (mirror of sim/weather.gd chaotic mode)."""
import json, os
import numpy as np

HERE = os.environ.get("E1_FC") or os.path.dirname(os.path.abspath(__file__))  # E1_FC: ml/forecast when this is a deployed copy (S1-12)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(HERE, "data")
F = ["minute", "lx", "ly", "lz", "pressure", "trend", "wind_dir", "wind_ms", "animals", "sky",
     "in_storm", "landed", "tier", "storm_index", "last_end_h"]
FI = {k: i for i, k in enumerate(F)}
DAYS = 120
H = DAYS * 24
S = 65536
BAL = json.load(open(os.path.join(ROOT, "data", "balance.json")))
CH = BAL["weather"]["chaos"]
LOR = CH["lorenz"]
DL = 1440
NB = 28          # 6 h bins over 7 days; class NB = no landing within 7 days
HOR = 168


def load(profile, noise, split):
    a = np.fromfile(os.path.join(DATA, f"{profile}_{noise}_{split}.bin"), dtype="<i4")
    return a.reshape(-1, H, len(F))


def tdiv(a, b):
    """Integer division truncating toward zero (GDScript / Rust), b > 0 scalar."""
    q = np.abs(a) // b
    return np.where(a < 0, -q, q)


def lorenz_step(x, y, z):
    dt = LOR["dt_div"]; bn = LOR["beta_num"]; bd = LOR["beta_den"]; sg = LOR["sigma"]; rho = LOR["rho"] * S
    dx = tdiv(sg * (y - x), dt)
    dy = tdiv(tdiv(x * (rho - z), S) - y, dt)
    dz = tdiv(bd * tdiv(x * y, S) - bn * z, bd * dt)
    return x + dx, y + dy, z + dz


def lorenz_hour(x, y, z):
    for _ in range(LOR["steps_per_hour"]):
        x, y, z = lorenz_step(x, y, z)
    return x, y, z


def scripted_storms(profile):
    return len(BAL["weather_profiles"][profile].get("storms", []))


def lands(profile, idx, minute, zpk, after_end_min):
    """Vectorised _lands: would a peak of height zpk (1/65536) one hour before `minute` land storm idx."""
    if idx < scripted_storms(profile):
        sc = CH["scenario"]
        return (minute >= sc["force_day"] * DL) | ((minute >= sc["open_day"] * DL) & (zpk > sc["peak_c"] * S // 100))
    return (zpk > CH["peak_c"] * S // 100) & (minute >= CH["first_day"] * DL) & (minute >= after_end_min + CH["min_gap_days"] * DL)


def tier_of(z):
    return np.where(z < CH["tier_shallow_below_c"] * S // 100, -1, np.where(z >= CH["tier_deep_from_c"] * S // 100, 1, 0))


def storm_minutes(profile, idx):
    """Duration in minutes of storm number idx (approx for scenario storm: days*1440; end handled from data)."""
    pr = BAL["weather_profiles"][profile]
    sc = pr.get("storms", [])
    if idx < len(sc):
        return len(sc[idx]["temps_by_day"]) * DL
    return pr["after"]["recurring_storms"]["days"] * DL


def first_landing_class(d, t):
    """Class of the first landing strictly after hour index t within 168 h: bin (h-1)//6, or NB."""
    land = d[t + 1:t + 1 + HOR, FI["landed"]]
    k = np.nonzero(land)[0]
    if len(k) == 0 or t + 1 + HOR > H:
        return NB if len(k) == 0 else int(k[0]) // 6
    return int(k[0]) // 6


def landing_hours_after(d, t):
    land = d[t + 1:t + 1 + HOR, FI["landed"]]
    k = np.nonzero(land)[0]
    return int(k[0]) + 1 if len(k) else None

NZ = CH["noise_levels"]
# S1-12 observation-model ablation, offline data only (weather_gen wind_noise=0.2; the game is unchanged): the reported wind
# sector is shifted one step on the 8-sector ring (+1 or -1, 0.1 each) in 20% of hours; pressure / wind speed as murky.
NZ = {**NZ, "murky_windnoise": {**NZ["murky"], "wind_dir_flip": 0.2}}
PC = CH["pressure"]; WC = CH["wind"]; SC = CH["signs"]


def p_mean(z):
    return PC["base"] - tdiv((z - PC["z_mid"] * S) * PC["z_gain"], S)


def wind_mean(y, storm):
    return np.where(storm, WC["storm_base"], WC["base"]) + tdiv(np.abs(y), S * WC["y_div"])


def wind_dir_of(x, y, storm):
    calm = np.where(x >= 0, np.where(y >= 0, 6, 5), np.where(y > 8 * S, 7, np.where(y < -8 * S, 1, 0)))
    return np.where(storm, np.where(y >= 0, 0, 1), calm)


def reading_storm(d):
    """Phase STORM at the moment readings were taken in hour h (before an exact-hour storm end)."""
    ins = d[..., FI["in_storm"]]; land = d[..., FI["landed"]]; le = d[..., FI["last_end_h"]]
    prev = np.concatenate([np.zeros_like(ins[..., :1]), ins[..., :-1]], -1)
    hh = np.arange(1, H + 1)
    return (land == 1) | (ins == 1) | ((prev == 1) & (le == hh))


def is_sign_hour(h):
    return (h * 60) % DL == 7 * 60


def storm_end_min(profile, idx, land_min):
    pr = BAL["weather_profiles"][profile]
    sc = pr.get("storms", [])
    if idx < len(sc):
        st = sc[idx]
        end = land_min + len(st["temps_by_day"]) * DL
        if "final_phase" in st:
            end = (end // DL) * DL + int(float(st["final_phase"]["end_h"]) * 60)
        return end
    return land_min + pr["after"]["recurring_storms"]["days"] * DL


def issue_hours():
    """Hour numbers h (1-based, minute = 60h) of the daily 07:00 issue times whose 7-day window fits."""
    return [h for h in range(1, H + 1) if is_sign_hour(h) and h + HOR <= H]


def ctx_at(d, h):
    """Observer-known storm bookkeeping at hour h (after processing): (storm_index, after_end_min, in_storm, storm_end_min or None)."""
    r = d[h - 1]
    idx = int(r[FI["storm_index"]]); ins = int(r[FI["in_storm"]]); le = int(r[FI["last_end_h"]])
    after_end = le * 60 if le > -100000 else -(1 << 40)
    return idx, after_end, ins


def true_class(d, h):
    """First landing at hour h+k (k = 1..168) -> bin (k-1)//6, else NB."""
    land = d[h:h + HOR, FI["landed"]]
    k = np.nonzero(land)[0]
    return int(k[0]) // 6 if len(k) else NB


def true_path(d, h):
    """Observable future path (168 x 3): pressure hPa, wind m/s, in_storm."""
    f = d[h:h + HOR]
    return np.stack([f[:, FI["pressure"]] / 10.0, f[:, FI["wind_ms"]].astype(float), f[:, FI["in_storm"]].astype(float)], -1)
