from common import *
for noise in ("murky", "clear"):
    d = load('endurance', noise, 'test')[:100].astype(np.int64)
    st = reading_storm(d)
    x, y, z = d[..., 1], d[..., 2], d[..., 3]
    r = d[..., 4] - p_mean(z); nzp = NZ[noise]["pressure"]
    print(noise, 'pressure resid min/max', r.min(), r.max(), 'bound', nzp)
    wm = wind_mean(y, st); wr = d[..., 7] - wm; nzw = NZ[noise]["wind"]
    ok = ((wr >= -nzw) & (wr <= nzw)) | ((d[..., 7] == 0) & (wm - nzw <= 0))
    print(' wind ok frac', ok.mean(), 'dir ok', (wind_dir_of(x, y, st) == d[..., 6]).mean())
    # trend = p - p 24h ago (25-entry history)
    tr = d[:, 24:, 4] - d[:, :-24, 4]
    print(' trend ok', (tr == d[:, 24:, 5]).mean())
    # daily signs at 07:00: stage from max z over last 24 readings
    hs = [h for h in range(1, H + 1) if is_sign_hour(h)]
    bad = 0; n = 0
    for h in hs[1:]:
        i = h - 1
        zmax = z[:, i - 23:i + 1].max(1)
        stage = np.where(st[:, i] | (zmax >= SC["z2_c"] * S // 100), 2, np.where(zmax >= SC["z1_c"] * S // 100, 1, 0))
        a = d[:, i, 8]; bad += int((np.abs(a - stage) > 1).sum()); n += len(a)
    print(' animals within +-1 of stage', 1 - bad / n)
