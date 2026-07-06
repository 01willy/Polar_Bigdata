"""B0 결과 직관 시각화(절대원칙): 북사면 연속 ALT 예측 표면 3-패널 + fold 오차 지도.
 (1) DL(사전학습+미세조정) vs GBM vs ESA CCI — 같은 색축으로 나란히, 실측 오버레이
 (2) 공간블록 CV fold별 |오차| 지도 — DL과 GBM 비교 + 개선(Δ) 지도
입력: outputs/models/b0_mlp_pretrained.pt, data/processed/b0_fold_predictions.csv
산출: outputs/maps/alt_surface_b0_comparison.png, b0_fold_error_map.png
GPU: CUDA_VISIBLE_DEVICES 1장.
"""
import sys, os, glob, calendar
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from scipy.ndimage import uniform_filter
sys.path.insert(0, "src")
from polar.plotstyle import use_polar, CMAP, tnorm, add_cbar
from polar.outputs import mappath
plt = use_polar()
# 스타일 안전장치: 일부 mpl/cmcrameri 버전 조합에서 image.cmap 기본값이
# 정규화되지 않아(batlow→cmc.batlow) mappable 생성 시 조회 실패할 수 있음.
# 명시적 CMAP 객체는 항상 넘기지만, 기본값도 유효 컬러맵으로 보정(순수 스타일).
import matplotlib as _mpl
if plt.rcParams["image.cmap"] not in _mpl.colormaps:
    plt.rcParams["image.cmap"] = "cmc.batlow" if "cmc.batlow" in _mpl.colormaps else "viridis"
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor

N, S, Wl, E = 72.0, 68.0, -162.0, -143.0     # 북사면
RES = 0.02
dev = "cuda" if torch.cuda.is_available() else "cpu"

ck = torch.load("outputs/models/b0_mlp_pretrained.pt", map_location="cpu")
FEAT, mu, sd, ymu, ysd = ck["feat"], ck["mu"], ck["sd"], ck["ymu"], ck["ysd"]
E5F = [f for f in FEAT if f.startswith("e5_")]


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.1),
            nn.Linear(128, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(1)


# ---------------- 전체 실측으로 미세조정(최종 배포형 모델) ----------------
obs = pd.read_csv("data/processed/dl_dataset.csv")
Xo = ((obs[FEAT].values - mu) / sd).astype(np.float32)
yo = ((obs["y"].values - ymu) / ysd).astype(np.float32)
net = MLP(len(FEAT)).to(dev); net.load_state_dict(ck["state"])
opt = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-5)
lossf = nn.SmoothL1Loss()
vs = np.random.RandomState(0).rand(len(Xo)) < 0.1
Xt = torch.tensor(Xo[~vs]); yt = torch.tensor(yo[~vs])
Xv = torch.tensor(Xo[vs]).to(dev); yv = yo[vs]
best, best_state, pat = 1e9, None, 0
for ep in range(60):
    net.train(); idx = torch.randperm(len(Xt))
    for k in range(0, len(Xt), 1024):
        b = idx[k:k + 1024]
        xb, yb = Xt[b].to(dev), yt[b].to(dev)
        opt.zero_grad(); l = lossf(net(xb), yb); l.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        v = float(np.mean((net(Xv).cpu().numpy() - yv) ** 2))
    if v < best - 1e-4:
        best, best_state, pat = v, {k2: t.cpu().clone() for k2, t in net.state_dict().items()}, 0
    else:
        pat += 1
        if pat >= 5: break
net.load_state_dict(best_state)
torch.save(dict(state=best_state, mu=mu, sd=sd, ymu=ymu, ysd=ysd, feat=FEAT),
           "outputs/models/b0_mlp_finetuned_full.pt")
print("미세조정(전체 실측) 완료")

# ---------------- 격자 공변량: ERA5 보간 ----------------
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
print("ERA5 격자 완료", G["e5_maat"].shape)

# ---------------- 격자 공변량: DEM 타일별 ----------------
LA, LO = np.meshgrid(glat, glon, indexing="ij")
flat_lat, flat_lon = LA.ravel(), LO.ravel()
D = {c: np.full(flat_lat.shape, np.nan, dtype=np.float32)
     for c in ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough"]}
mperdeg = 111320.0
W33 = 33
tl = np.floor(flat_lat).astype(int); tn2 = np.floor(flat_lon).astype(int)
for a in np.unique(tl):
    for b in np.unique(tn2):
        m = (tl == a) & (tn2 == b)
        if not m.any(): continue
        path = f"data/raw/dem/Copernicus_DSM_COG_10_N{abs(a):02d}_00_W{abs(b):03d}_00_DEM.tif"
        if not os.path.exists(path): continue
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            fc, fr = (~src.transform) * (flat_lon[m], flat_lat[m])
            rows = np.clip(np.floor(fr).astype(int), 0, arr.shape[0] - 1)
            cols = np.clip(np.floor(fc).astype(int), 0, arr.shape[1] - 1)
            dy = mperdeg * abs(src.transform.e)
            dx = mperdeg * abs(src.transform.a) * max(np.cos(np.radians(a + 0.5)), 0.05)
        arrf = np.nan_to_num(arr, nan=float(np.nanmean(arr)))
        gy, gx = np.gradient(arrf, dy, dx)
        slope = np.degrees(np.arctan(np.hypot(gy, gx)))
        asp = np.arctan2(-gy, gx)
        m1 = uniform_filter(arrf, W33); m2 = uniform_filter(arrf ** 2, W33)
        std = np.sqrt(np.clip(m2 - m1 ** 2, 0, None))
        D["dem_elev"][m] = arr[rows, cols]
        D["dem_slope"][m] = slope[rows, cols]
        D["dem_aspect_sin"][m] = np.sin(asp[rows, cols])
        D["dem_aspect_cos"][m] = np.cos(asp[rows, cols])
        D["dem_tpi"][m] = arr[rows, cols] - m1[rows, cols]
        D["dem_rough"][m] = std[rows, cols]
print(f"DEM 격자 완료: 유효 {np.isfinite(D['dem_elev']).sum():,}/{len(flat_lat):,}")

Xg = np.column_stack([D[f].astype(np.float32) if f in D else G[f].ravel().astype(np.float32)
                      for f in FEAT])
land = np.isfinite(Xg).all(1)
Xz = ((Xg[land] - mu) / sd).astype(np.float32)

# DL 표면
net.eval(); preds = []
with torch.no_grad():
    for k in range(0, len(Xz), 65536):
        preds.append(net(torch.tensor(Xz[k:k + 65536]).to(dev)).cpu().numpy())
pdl = np.concatenate(preds) * ysd + ymu
alt_dl = np.full(len(flat_lat), np.nan)
alt_dl[land] = np.expm1(np.clip(pdl, np.log1p(1), np.log1p(600)))
alt_dl = alt_dl.reshape(len(glat), len(glon))

# GBM 표면(14피처 동일 조건)
gbm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(Xo, obs["y"])
pg = gbm.predict(((Xg[land] - mu) / sd))
alt_gbm = np.full(len(flat_lat), np.nan)
alt_gbm[land] = np.expm1(np.clip(pg, np.log1p(1), np.log1p(600)))
alt_gbm = alt_gbm.reshape(len(glat), len(glon))

# CCI 다년평균 표면
cci_grid = np.full((len(glat), len(glon)), np.nan)
ssum = cnt = None
for f in sorted(glob.glob("data/raw/cci_alt/*.nc")):
    try:
        c = xr.open_dataset(f)
        sub = c["ALT"].isel(time=0).sel(lat=slice(S, N), lon=slice(Wl, E))
        a = sub.values.astype(np.float32)
        if ssum is None:
            ssum = np.zeros_like(a); cnt = np.zeros_like(a)
            cla, clo = sub.lat.values, sub.lon.values
        mm = np.isfinite(a) & (a > 0) & (a < 10)
        ssum[mm] += a[mm]; cnt[mm] += 1
        c.close()
    except Exception:
        pass
if ssum is not None:
    Am = np.where(cnt > 0, ssum / np.maximum(cnt, 1) * 100.0, np.nan)
    iy = np.clip(np.searchsorted(cla, LA.ravel()), 0, len(cla) - 1)
    ix = np.clip(np.searchsorted(clo, LO.ravel()), 0, len(clo) - 1)
    cci_grid = Am[iy, ix].reshape(len(glat), len(glon))
print("표면 3종 완료")

# ---------------- 그림 1: 3-패널 비교 ----------------
o = obs[(obs.lat.between(S, N)) & (obs.lon.between(Wl, E))].groupby("loc_id").agg(
    lat=("lat", "mean"), lon=("lon", "mean"), alt=("alt_cm", "mean")).reset_index()
fig, axes = plt.subplots(3, 1, figsize=(13, 15), sharex=True)
panels = [(alt_dl, f"DL 사전학습(weak 4.0M)+미세조정 — 우리 모델"),
          (alt_gbm, "GBM (같은 14개 공변량) — 기존 최강 baseline"),
          (cci_grid, "ESA CCI 위성 물리모델 (1km, 2003-2019 평균) — 외부 제품")]
for ax, (Z, title) in zip(axes, panels):
    mesh = ax.pcolormesh(glon, glat, Z, cmap=CMAP.alt, vmin=20, vmax=90, shading="auto")
    ax.scatter(o.lon, o.lat, c=o.alt, s=16, cmap=CMAP.alt, vmin=20, vmax=90,
               edgecolors="k", linewidths=0.4, zorder=5)
    add_cbar(fig, mesh, ax, "활성층 두께 ALT (cm)", shrink=0.9, pad=0.015)
    ax.set_ylabel("위도 (°N)"); ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlim(Wl, E); ax.set_ylim(S, N)
axes[-1].set_xlabel("경도 (°E)")
fig.suptitle("알래스카 북사면 ALT 예측 표면 비교 — 동일 색축, 테두리 점=실측", fontsize=14, weight="bold", y=0.995)
fig.tight_layout()
fig.savefig(mappath("alt_surface_b0_comparison")); plt.close(fig)
print("saved", mappath("alt_surface_b0_comparison"))

# ---------------- 그림 2: fold 오차 지도 ----------------
fp = pd.read_csv("data/processed/b0_fold_predictions.csv")
fp["e_dl"] = (fp.pred_dl - fp.alt_cm).abs()
fp["e_gbm"] = (fp.pred_gbm - fp.alt_cm).abs()
loc = fp.groupby([fp.lat.round(2), fp.lon.round(2)]).agg(
    e_dl=("e_dl", "mean"), e_gbm=("e_gbm", "mean"), fold=("fold", "first")).reset_index()
fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
for ax, col, title in [(axes[0], "e_dl", "DL |오차|"), (axes[1], "e_gbm", "GBM |오차|")]:
    sc = ax.scatter(loc.lon, loc.lat, c=loc[col], s=9, cmap=CMAP.err, vmin=0, vmax=40,
                    edgecolors="none", rasterized=True)
    add_cbar(fig, sc, ax, "|오차| (cm)")
    ax.set_title(f"{title} (공간블록 CV, 위치평균)", weight="bold")
    ax.set_xlabel("경도 (°E)"); ax.set_ylabel("위도 (°N)"); ax.grid(alpha=0.25)
d = loc.e_gbm - loc.e_dl
sc = axes[2].scatter(loc.lon, loc.lat, c=d, s=9, cmap=CMAP.diff, norm=tnorm(-15, 15),
                     edgecolors="none", rasterized=True)
add_cbar(fig, sc, axes[2], "GBM-DL (cm, 청색=DL 우세)")
axes[2].set_title("개선 지도: 청색=DL이 더 정확", weight="bold")
axes[2].set_xlabel("경도 (°E)"); axes[2].grid(alpha=0.25)
fig.suptitle("B0 공간블록 CV 오차의 공간 분포 — 어디서 DL이 이기는가", fontsize=13, weight="bold")
fig.tight_layout()
fig.savefig(mappath("b0_fold_error_map")); plt.close(fig)
print("saved", mappath("b0_fold_error_map"))
