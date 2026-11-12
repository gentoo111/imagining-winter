from common import *
d = load('endurance', 'murky', 'test')[:50]
print(d.shape, 'storms/seed(120d)', d[:, :, FI['landed']].sum(1).mean(), 'per60', d[:, :1440, FI['landed']].sum(1).mean(), flush=True)
x, y, z = (d[:, 0, i].astype(np.int64) for i in (1, 2, 3))
bad = 0
for h in range(1, H):
    x, y, z = lorenz_hour(x, y, z)
    bad += int((x != d[:, h, 1]).sum() + (y != d[:, h, 2]).sum() + (z != d[:, h, 3]).sum())
print('lorenz mismatches', bad)
