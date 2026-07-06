"""C 고정밀 국소 데모: 북사면 평탄툰드라에 PolSAR(30m 물리)+공변량 학습 모델로 고해상 ALT 필드.
 (1) PolSAR 원자료(물리 base) vs 우리 모델(보정) vs Diffusion 불확실성 — 3패널 고해상
 (2) Area-of-Applicability: 평탄툰드라 조건(slope<2,elev<150) 밖은 회색(모델 유효범위 명시).
학습=평탄툰드라 실측(dl_dataset_polsar), 16피처(공변량14+polsar_alt+polsar_std). GPU 1장.
산출: maps/local_demo_alt_field.png, maps/local_demo_uncertainty.png
"""
import sys, os, calendar
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import uniform_filter
sys.path.insert(0, "src")
from polar.plotstyle import use_polar, CMAP, add_cbar, style_geo, BAD
from polar.outputs import mappath
plt = use_polar()
from sklearn.ensemble import HistGradientBoostingRegressor
import torch, torch.nn as nn

SMOKE = os.environ.get("SMOKE", "0") == "1"
dev = "cuda" if torch.cuda.is_available() else "cpu"
N, S, Wl, E = 70.9, 69.7, -153.6, -150.8     # 북사면 평탄 툰드라 창
RES = 0.004 if SMOKE else 0.0025             # ~250m
CLIP = (np.log1p(1), np.log1p(600))
E5F = ["e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
DEMF = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough"]
FEAT = DEMF + E5F + ["polsar_alt", "polsar_std"]

# ---------------- 학습: 평탄툰드라 실측 ----------------
d = pd.read_csv("data/processed/dl_dataset_polsar.csv")
d = d[d.polsar_valid == 1].copy()
for c in ["polsar_alt", "polsar_std"]:
    d[c] = d[c].fillna(d[c].median())
tundra = d[(d.dem_slope < 2) & (d.dem_elev < 150)].reset_index(drop=True)
print(f"평탄툰드라 학습점 {len(tundra):,}")
mu = tundra[FEAT].mean().values.astype(np.float32); sd = tundra[FEAT].std().values.astype(np.float32) + 1e-6
Xo = ((tundra[FEAT].values - mu) / sd).astype(np.float32)
ylog = np.log1p(tundra.alt_cm.values).astype(np.float32); ymu, ysd = ylog.mean(), ylog.std() + 1e-6

# ---------------- 격자 좌표 ----------------
glat = np.arange(S, N, RES); glon = np.arange(Wl, E, RES)
LO, LA = np.meshgrid(glon, glat)
H, W = LA.shape
print(f"격자 {H}×{W} = {H*W:,}")

# ERA5 보간
ds = xr.open_dataset("data/raw/era5land/nh_monthly_2015-2020.nc")
tn = "valid_time" if "valid_time" in ds.coords else "time"
clim = ds.assign_coords(month=ds[tn].dt.month).groupby("month").mean(tn).sel(
    latitude=slice(N + 0.3, S - 0.3), longitude=slice(Wl - 0.3, E + 0.3))
fine = clim.interp(latitude=glat, longitude=glon, method="linear")
days = np.array([calendar.monthrange(2019, m)[1] for m in range(1, 13)])[:, None, None]
t = fine["t2m"].values - 273.15; stl = fine["stl1"].values - 273.15; sdp = fine["sd"].values
tdd = np.nansum(np.clip(t, 0, None) * days, 0); fdd = np.nansum(np.clip(-t, 0, None) * days, 0)
G = {"e5_maat": np.nanmean(t, 0), "e5_tdd": tdd, "e5_fdd": fdd, "e5_sqrt_tdd": np.sqrt(tdd),
     "e5_twarm": np.nanmax(t, 0), "e5_tcold": np.nanmin(t, 0),
     "e5_stl1": np.nanmean(stl, 0), "e5_swe": np.nanmean(sdp, 0)}

# DEM 30m → 지형특징(타일별)
D = {c: np.full(H * W, np.nan, np.float32) for c in DEMF}
mperdeg = 111320.0; W33 = 33
flat_lat, flat_lon = LA.ravel(), LO.ravel()
for a in range(int(np.floor(S)), int(np.floor(N)) + 1):
    for b in range(int(np.floor(-E)), int(np.floor(-Wl)) + 1):
        path = f"data/raw/dem/Copernicus_DSM_COG_10_N{a:02d}_00_W{b:03d}_00_DEM.tif"
        if not os.path.exists(path): continue
        m = (np.floor(flat_lat).astype(int) == a) & (np.floor(flat_lon).astype(int) == -b)
        if not m.any(): continue
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            if src.nodata is not None: arr[arr == src.nodata] = np.nan
            fc, fr = (~src.transform) * (flat_lon[m], flat_lat[m])
            rr = np.clip(np.floor(fr).astype(int), 0, arr.shape[0] - 1)
            cc = np.clip(np.floor(fc).astype(int), 0, arr.shape[1] - 1)
            dy = mperdeg * abs(src.transform.e); dx = mperdeg * abs(src.transform.a) * max(np.cos(np.radians(a + 0.5)), 0.05)
        arrf = np.nan_to_num(arr, nan=float(np.nanmean(arr)))
        gy, gx = np.gradient(arrf, dy, dx)
        slope = np.degrees(np.arctan(np.hypot(gy, gx))); asp = np.arctan2(-gy, gx)
        m1 = uniform_filter(arrf, W33); m2 = uniform_filter(arrf ** 2, W33)
        std = np.sqrt(np.clip(m2 - m1 ** 2, 0, None))
        D["dem_elev"][m] = arr[rr, cc]; D["dem_slope"][m] = slope[rr, cc]
        D["dem_aspect_sin"][m] = np.sin(asp[rr, cc]); D["dem_aspect_cos"][m] = np.cos(asp[rr, cc])
        D["dem_tpi"][m] = arr[rr, cc] - m1[rr, cc]; D["dem_rough"][m] = std[rr, cc]

# PolSAR 30m → 격자 재투영(2015 대표), window std
ps_year = "data/raw/polsar_alt/upscaled_alt_2015.tif"
dst_tr = rasterio.transform.from_bounds(Wl, S, E, N, W, H)
polsar = np.full((H, W), np.nan, np.float32)
with rasterio.open(ps_year) as src:
    reproject(source=rasterio.band(src, 1), destination=polsar,
              src_transform=src.transform, src_crs=src.crs,
              dst_transform=dst_tr, dst_crs="EPSG:4326", resampling=Resampling.bilinear,
              src_nodata=src.nodata, dst_nodata=np.nan)
polsar = polsar * 100.0                        # m→cm
polsar[(polsar <= 1) | (polsar > 600)] = np.nan
m1 = uniform_filter(np.nan_to_num(polsar, nan=float(np.nanmean(polsar))), 5)
m2 = uniform_filter(np.nan_to_num(polsar, nan=float(np.nanmean(polsar))) ** 2, 5)
polsar_std = np.sqrt(np.clip(m2 - m1 ** 2, 0, None))
polsar = polsar[::-1]; polsar_std = polsar_std[::-1]   # dst_tr는 north-up이라 배열 상하 정렬 맞춤
print(f"PolSAR 격자: 유효 {np.isfinite(polsar).sum():,}/{H*W:,}")

# 피처 조립
Gflat = {k: v.ravel() for k, v in G.items()}
cols = {}
for f in DEMF: cols[f] = D[f]
for f in E5F: cols[f] = Gflat[f]
cols["polsar_alt"] = polsar.ravel(); cols["polsar_std"] = polsar_std.ravel()
Xg = np.column_stack([cols[f] for f in FEAT]).astype(np.float32)
# Area of Applicability: 평탄툰드라 조건 + 전 피처 유효
aoa = (cols["dem_slope"] < 2) & (cols["dem_elev"] < 150) & np.isfinite(Xg).all(1)
print(f"AoA(평탄툰드라 유효) {aoa.sum():,}/{H*W:,}")
Xz = ((Xg[aoa] - mu) / sd).astype(np.float32)


class CondNet(nn.Module):
    def __init__(self, dd):
        super().__init__()
        self.xemb = nn.Sequential(nn.Linear(dd, 128), nn.SiLU(), nn.Linear(128, 128))
        self.net = nn.Sequential(nn.Linear(130, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                                 nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))
    def forward(self, yt, t, x):
        return self.net(torch.cat([yt.unsqueeze(-1), t.unsqueeze(-1), self.xemb(x)], -1)).squeeze(-1)


# 학습(GBM + Diffusion)
gbm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(Xo, ylog)
T = 100; S_n = 40 if not SMOKE else 12
betas = torch.linspace(1e-4, 0.02, T, device=dev); acp = torch.cumprod(1 - betas, 0)
net = CondNet(len(FEAT)).to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
Xt = torch.tensor(Xo); yz = torch.tensor((ylog - ymu) / ysd)
for ep in range(12 if SMOKE else 70):
    net.train(); idx = torch.randperm(len(Xt))
    for k in range(0, len(Xt), 8192):
        b = idx[k:k + 8192]; xb = Xt[b].to(dev); y0 = yz[b].to(dev)
        ti = torch.randint(0, T, (len(b),), device=dev); a = acp[ti]
        eps = torch.randn_like(y0); ytt = torch.sqrt(a) * y0 + torch.sqrt(1 - a) * eps
        opt.zero_grad(); ((net(ytt, ti.float() / T, xb) - eps) ** 2).mean().backward(); opt.step()
print("모델 학습 완료")

# 격자 예측(GBM + Diffusion 앙상블 + 불확실성)
pg = np.expm1(np.clip(gbm.predict(Xz), *CLIP))
net.eval(); mean_pred = np.full(aoa.sum(), np.nan); pi = np.full(aoa.sum(), np.nan); off = 0
with torch.no_grad():
    for k in range(0, len(Xz), 4096):
        xb = torch.tensor(Xz[k:k + 4096]).to(dev)
        xe = xb.unsqueeze(0).expand(S_n, -1, -1).reshape(S_n * len(xb), -1)
        y = torch.randn(S_n * len(xb), device=dev)
        for ti in reversed(range(T)):
            a = acp[ti]; b_ = betas[ti]; tt = torch.full((S_n * len(xb),), ti / T, device=dev)
            eps = net(y, tt, xe); mn = (y - b_ / torch.sqrt(1 - a) * eps) / torch.sqrt(1 - b_)
            y = mn + (torch.sqrt(b_) * torch.randn_like(y) if ti > 0 else 0)
        cm = np.expm1(np.clip(y.reshape(S_n, len(xb)).cpu().numpy() * ysd + ymu, *CLIP))
        mean_pred[off:off + len(xb)] = cm.mean(0)
        pi[off:off + len(xb)] = np.percentile(cm, 95, 0) - np.percentile(cm, 5, 0); off += len(xb)
ens = 0.5 * pg + 0.5 * mean_pred

alt_field = np.full(H * W, np.nan); alt_field[aoa] = ens; alt_field = alt_field.reshape(H, W)
unc_field = np.full(H * W, np.nan); unc_field[aoa] = pi; unc_field = unc_field.reshape(H, W)
ps_field = np.where(aoa.reshape(H, W), polsar, np.nan)
print(f"예측 ALT {np.nanmin(alt_field):.0f}~{np.nanmax(alt_field):.0f}cm, 불확실성폭 중앙 {np.nanmedian(unc_field):.0f}cm")

# ---------------- (1) 3패널: PolSAR / 우리모델 / 불확실성 ----------------
obs_t = tundra[(tundra.lat.between(S, N)) & (tundra.lon.between(Wl, E))]
fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
for ax, Z, ttl, cmap, vlim, lab in [
    (axes[0], ps_field, "PolSAR 원자료 (P-band 물리, 30m)", CMAP.alt, (20, 70), "ALT (cm)"),
    (axes[1], alt_field, "우리 모델 (PolSAR+공변량, 앙상블 ~13cm)", CMAP.alt, (20, 70), "ALT (cm)"),
    (axes[2], unc_field, "Diffusion 불확실성 (90% 구간폭)", CMAP.err, (0, np.nanpercentile(unc_field, 98)), "cm")]:
    ax.set_facecolor(BAD)
    mesh = ax.pcolormesh(glon, glat, Z, cmap=cmap, vmin=vlim[0], vmax=vlim[1], shading="auto")
    if ax is not axes[2]:
        ax.scatter(obs_t.lon, obs_t.lat, c=obs_t.alt_cm, s=10, cmap=CMAP.alt, vmin=20, vmax=70,
                   edgecolors="#111", linewidths=0.3, zorder=5)
    add_cbar(fig, mesh, ax, lab)
    ax.set_title(ttl, fontsize=11); ax.set_xlabel("경도 (°E)"); ax.set_ylabel("위도 (°N)")
    ax.set_xlim(Wl, E); ax.set_ylim(S, N)
fig.suptitle("고정밀 국소 데모 — 알래스카 북사면 평탄 툰드라 (회색=모델 유효범위 밖, 테두리점=실측)",
             fontsize=14, weight="bold")
fig.tight_layout(); fig.savefig(mappath("local_demo_alt_field")); plt.close(fig)
print("saved", mappath("local_demo_alt_field"))
