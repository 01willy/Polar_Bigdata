"""토너먼트 상위 모델을 전량 실측으로 학습 → 알래스카 북사면 격자에 배포 예측(직관 2D 레이어).
 (1) maps/deploy_alt_gbm_vs_diffusion.png — ALT 예측 표면 GBM vs Diffusion(평균), 실측 오버레이
 (2) maps/deploy_diffusion_uncertainty.png — Diffusion 90% 예측구간 폭(공간 불확실성) — 생성모델 고유 산출
엔진: GBM(sklearn) + 조건부 DDPM(torch, model_tournament와 동일 구현). GPU 1장.
"""
import sys, os, glob, calendar
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from scipy.ndimage import uniform_filter
sys.path.insert(0, "src")
from polar.plotstyle import use_polar, CMAP, add_cbar, style_geo, FROZEN
from polar.outputs import mappath
plt = use_polar()
from sklearn.ensemble import HistGradientBoostingRegressor
import torch
import torch.nn as nn

SMOKE = os.environ.get("SMOKE", "0") == "1"
dev = "cuda" if torch.cuda.is_available() else "cpu"
N, S, Wl, E = 72.0, 68.0, -162.0, -143.0
RES = 0.05 if SMOKE else 0.02
FEAT = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough",
        "e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
E5F = [f for f in FEAT if f.startswith("e5_")]
CLIP = (np.log1p(1), np.log1p(600))
print("device:", dev, "| SMOKE:", SMOKE)

obs = pd.read_csv("data/processed/dl_dataset.csv").dropna(subset=FEAT + ["alt_cm"]).reset_index(drop=True)
if SMOKE:
    obs = obs.sample(20000, random_state=0).reset_index(drop=True)
mu = obs[FEAT].mean().values.astype(np.float32); sd = obs[FEAT].std().values.astype(np.float32) + 1e-6
Xo = ((obs[FEAT].values - mu) / sd).astype(np.float32)
ylog = np.log1p(obs.alt_cm.values).astype(np.float32)
ymu, ysd = ylog.mean(), ylog.std() + 1e-6

# ---------------- 격자 공변량(ERA5 보간 + DEM 타일) — b0와 동일 절차 ----------------
glat = np.arange(S, N, RES); glon = np.arange(Wl, E, RES)
ds = xr.open_dataset("data/raw/era5land/nh_monthly_2015-2020.nc")
tn = "valid_time" if "valid_time" in ds.coords else "time"
clim = ds.assign_coords(month=ds[tn].dt.month).groupby("month").mean(tn)
clim = clim.sel(latitude=slice(N + 0.3, S - 0.3), longitude=slice(Wl - 0.3, E + 0.3))
fine = clim.interp(latitude=glat, longitude=glon, method="linear")
days = np.array([calendar.monthrange(2019, m)[1] for m in range(1, 13)])[:, None, None]
t = fine["t2m"].values - 273.15; stl = fine["stl1"].values - 273.15; sdp = fine["sd"].values
tdd = np.nansum(np.clip(t, 0, None) * days, 0); fdd = np.nansum(np.clip(-t, 0, None) * days, 0)
G = {"e5_maat": np.nanmean(t, 0), "e5_tdd": tdd, "e5_fdd": fdd, "e5_sqrt_tdd": np.sqrt(tdd),
     "e5_twarm": np.nanmax(t, 0), "e5_tcold": np.nanmin(t, 0),
     "e5_stl1": np.nanmean(stl, 0), "e5_swe": np.nanmean(sdp, 0)}
LA, LO = np.meshgrid(glat, glon, indexing="ij")
flat_lat, flat_lon = LA.ravel(), LO.ravel()
D = {c: np.full(flat_lat.shape, np.nan, dtype=np.float32) for c in
     ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough"]}
mperdeg, W33 = 111320.0, 33
tl = np.floor(flat_lat).astype(int); tn2 = np.floor(flat_lon).astype(int)
for a in np.unique(tl):
    for b in np.unique(tn2):
        m = (tl == a) & (tn2 == b)
        if not m.any(): continue
        path = f"data/raw/dem/Copernicus_DSM_COG_10_N{abs(a):02d}_00_W{abs(b):03d}_00_DEM.tif"
        if not os.path.exists(path): continue
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            if src.nodata is not None: arr[arr == src.nodata] = np.nan
            fc, fr = (~src.transform) * (flat_lon[m], flat_lat[m])
            rows = np.clip(np.floor(fr).astype(int), 0, arr.shape[0] - 1)
            cols = np.clip(np.floor(fc).astype(int), 0, arr.shape[1] - 1)
            dy = mperdeg * abs(src.transform.e)
            dx = mperdeg * abs(src.transform.a) * max(np.cos(np.radians(a + 0.5)), 0.05)
        arrf = np.nan_to_num(arr, nan=float(np.nanmean(arr)))
        gy, gx = np.gradient(arrf, dy, dx)
        slope = np.degrees(np.arctan(np.hypot(gy, gx))); asp = np.arctan2(-gy, gx)
        m1 = uniform_filter(arrf, W33); m2 = uniform_filter(arrf ** 2, W33)
        std = np.sqrt(np.clip(m2 - m1 ** 2, 0, None))
        D["dem_elev"][m] = arr[rows, cols]; D["dem_slope"][m] = slope[rows, cols]
        D["dem_aspect_sin"][m] = np.sin(asp[rows, cols]); D["dem_aspect_cos"][m] = np.cos(asp[rows, cols])
        D["dem_tpi"][m] = arr[rows, cols] - m1[rows, cols]; D["dem_rough"][m] = std[rows, cols]
Xg = np.column_stack([D[f] if f in D else G[f].ravel().astype(np.float32) for f in FEAT])
land = np.isfinite(Xg).all(1)
Zg = ((Xg[land] - mu) / sd).astype(np.float32)
print(f"격자 {len(glat)}×{len(glon)}, 육지 {land.sum():,}")


# ---------------- GBM ----------------
gbm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(Xo, ylog)
alt_gbm = np.full(len(flat_lat), np.nan)
alt_gbm[land] = np.expm1(np.clip(gbm.predict(Zg), *CLIP))
alt_gbm = alt_gbm.reshape(len(glat), len(glon))


# ---------------- Diffusion(조건부 DDPM) ----------------
class CondNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.xemb = nn.Sequential(nn.Linear(d, 128), nn.SiLU(), nn.Linear(128, 128))
        self.net = nn.Sequential(nn.Linear(130, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                                 nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))
    def forward(self, yt, t, x):
        return self.net(torch.cat([yt.unsqueeze(-1), t.unsqueeze(-1), self.xemb(x)], -1)).squeeze(-1)

T = 100
betas = torch.linspace(1e-4, 0.02, T, device=dev); acp = torch.cumprod(1 - betas, 0)
net = CondNet(len(FEAT)).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
yz = torch.tensor((ylog - ymu) / ysd); Xt = torch.tensor(Xo)
EPO = 8 if SMOKE else 70
for ep in range(EPO):
    net.train(); idx = torch.randperm(len(Xt))
    for k in range(0, len(Xt), 8192):
        bx = idx[k:k + 8192]; xb = Xt[bx].to(dev); y0 = yz[bx].to(dev)
        ti = torch.randint(0, T, (len(bx),), device=dev); a = acp[ti]
        eps = torch.randn_like(y0); ytt = torch.sqrt(a) * y0 + torch.sqrt(1 - a) * eps
        opt.zero_grad(); ((net(ytt, ti.float() / T, xb) - eps) ** 2).mean().backward(); opt.step()
print("Diffusion 학습 완료")

net.eval(); Sn = 48
q05 = np.full(len(flat_lat), np.nan); q50 = np.full(len(flat_lat), np.nan); q95 = np.full(len(flat_lat), np.nan)
with torch.no_grad():
    res_mean = np.full(land.sum(), np.nan); res_lo = np.full(land.sum(), np.nan); res_hi = np.full(land.sum(), np.nan)
    off = 0
    for k in range(0, len(Zg), 4096):
        xb = torch.tensor(Zg[k:k + 4096]).to(dev)
        xb_e = xb.unsqueeze(0).expand(Sn, -1, -1).reshape(Sn * len(xb), -1)
        y = torch.randn(Sn * len(xb), device=dev)
        for ti in reversed(range(T)):
            a = acp[ti]; b_ = betas[ti]; tt = torch.full((Sn * len(xb),), ti / T, device=dev)
            eps = net(y, tt, xb_e); mean = (y - b_ / torch.sqrt(1 - a) * eps) / torch.sqrt(1 - b_)
            y = mean + (torch.sqrt(b_) * torch.randn_like(y) if ti > 0 else 0)
        samp = (y.reshape(Sn, len(xb)).cpu().numpy() * ysd + ymu)
        cm = np.expm1(np.clip(samp, *CLIP))
        res_mean[off:off + len(xb)] = cm.mean(0)
        res_lo[off:off + len(xb)] = np.percentile(cm, 5, axis=0)
        res_hi[off:off + len(xb)] = np.percentile(cm, 95, axis=0)
        off += len(xb)
alt_diff = np.full(len(flat_lat), np.nan); alt_diff[land] = res_mean
unc = np.full(len(flat_lat), np.nan); unc[land] = res_hi - res_lo
alt_diff = alt_diff.reshape(len(glat), len(glon)); unc = unc.reshape(len(glat), len(glon))
print(f"Diffusion 배포: ALT {np.nanmin(alt_diff):.0f}~{np.nanmax(alt_diff):.0f}, 불확실성폭 중앙 {np.nanmedian(unc):.0f}cm")

# ---------------- 렌더 ----------------
o = obs[(obs.lat.between(S, N)) & (obs.lon.between(Wl, E))].groupby("loc_id").agg(
    lat=("lat", "mean"), lon=("lon", "mean"), alt=("alt_cm", "mean")).reset_index()

# (1) GBM vs Diffusion 표면
fig, axes = plt.subplots(2, 1, figsize=(13, 11), sharex=True)
for ax, Zm, ttl in [(axes[0], alt_gbm, "GBM (기존 최강 baseline)"),
                    (axes[1], alt_diff, "Diffusion (조건부 DDPM, 샘플 평균) — 생성모델")]:
    mesh = ax.pcolormesh(glon, glat, Zm, cmap=CMAP.alt, vmin=20, vmax=90, shading="auto")
    ax.scatter(o.lon, o.lat, c=o.alt, s=16, cmap=CMAP.alt, vmin=20, vmax=90,
               edgecolors="#111", linewidths=0.4, zorder=5)
    add_cbar(fig, mesh, ax, "예측 ALT (cm)")
    ax.set_ylabel("위도 (°N)"); ax.set_xlim(Wl, E); ax.set_ylim(S, N); ax.set_title(ttl)
axes[1].set_xlabel("경도 (°E)")
fig.suptitle("알래스카 북사면 ALT 예측 표면 — 전량 학습 배포, 테두리점=실측", fontsize=14, weight="bold")
fig.tight_layout(); fig.savefig(mappath("deploy_alt_gbm_vs_diffusion")); plt.close(fig)
print("saved", mappath("deploy_alt_gbm_vs_diffusion"))

# (2) Diffusion 공간 불확실성
fig, ax = plt.subplots(figsize=(13, 6.2))
mesh = ax.pcolormesh(glon, glat, unc, cmap=CMAP.err, vmin=0, vmax=np.nanpercentile(unc, 98), shading="auto")
ax.scatter(o.lon, o.lat, s=10, c="white", edgecolors="#111", linewidths=0.5, zorder=5,
           label=f"실측 위치({len(o)})")
add_cbar(fig, mesh, ax, "Diffusion 90% 예측구간 폭 (cm) — 넓을수록 불확실")
style_geo(ax, "Diffusion 공간 불확실성 지도 — 생성모델 고유 산출\n"
              "(실측 밀집부에서 좁고, 외삽 영역에서 넓어짐을 기대)")
ax.legend(loc="lower left")
fig.tight_layout(); fig.savefig(mappath("deploy_diffusion_uncertainty")); plt.close(fig)
print("saved", mappath("deploy_diffusion_uncertainty"))
