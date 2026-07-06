"""Phase C: 알래스카 전역(60-72N) 3D 지중 열구조 — 조건장(게이트 승자 엔진)으로 T(x,y,z) 큐브 생성.
 (1) volumes_3d/thermal_cube_alaska.png — 깊이 슬라이스 층층 3D 뷰
 (2) maps/magt_alaska_2m_20m.png — 연평균 지중온도(MAGT) 2m/20m 지도 + 0°C 등고선(영구동토 경계) + 실측
 (3) animations/thermal_depth_slices.gif — 0→20m 깊이 내려가는 애니메이션
엔진: data/processed/b1b_results.csv에서 지역전이 최저 RMSE 모델 자동 선택(NF 앙상블 or GBM).
주의: 연평균 장(MAGT)이므로 계절 최대융해깊이(ALT)와 다름 — ALT 지도는 B0(maps/alt_surface_b0_*) 담당.
"""
import sys, os, glob, calendar
import numpy as np
import pandas as pd
import xarray as xr
sys.path.insert(0, "src")
from polar.plotstyle import use_polar, CMAP, tnorm, FROZEN
from polar.outputs import mappath, volpath, animpath
plt = use_polar()
plt.rcParams["image.cmap"] = "cmc.batlow"  # 이 matplotlib 버전은 'batlow' 미등록 → 정식명으로 보정
from matplotlib import cm, colors
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.ensemble import HistGradientBoostingRegressor
import torch
import torch.nn as nn

N, S, Wl, E = 72.0, 60.0, -166.0, -141.0   # 알래스카 전역: 온난(남)~한랭(북) 대비
DEPTHS = np.arange(0.25, 20.01, 0.25)
E5F = ["e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
dev = "cuda" if torch.cuda.is_available() else "cpu"

def fourier(dm):
    dn = (dm / 30.0).astype(np.float32)
    out = []
    for k in range(5):
        out += [np.sin(2 ** k * np.pi * dn), np.cos(2 ** k * np.pi * dn)]
    return np.column_stack(out)

# ---------------- 학습 데이터(지중온도) + ERA5 ----------------
g = pd.read_csv("data/processed/ground_temp_all.csv")
g = g[(g.depth_m > 0) & (g.depth_m <= 30) & (g.temp_c > -25) & (g.temp_c < 25)].reset_index(drop=True)
ds = xr.open_dataset("data/raw/era5land/nh_monthly_2015-2020.nc")
tn = "valid_time" if "valid_time" in ds.coords else "time"
clim0 = ds.assign_coords(month=ds[tn].dt.month).groupby("month").mean(tn)
elat, elon = clim0["latitude"].values, clim0["longitude"].values
days = np.array([calendar.monthrange(2019, m)[1] for m in range(1, 13)])[:, None, None]
def derive(c):
    t = c["t2m"].values - 273.15; stl = c["stl1"].values - 273.15; sdp = c["sd"].values
    tdd = np.nansum(np.clip(t, 0, None) * days, 0); fdd = np.nansum(np.clip(-t, 0, None) * days, 0)
    return dict(e5_maat=np.nanmean(t, 0), e5_tdd=tdd, e5_fdd=fdd, e5_sqrt_tdd=np.sqrt(tdd),
                e5_twarm=np.nanmax(t, 0), e5_tcold=np.nanmin(t, 0),
                e5_stl1=np.nanmean(stl, 0), e5_swe=np.nanmean(sdp, 0))
E5 = derive(clim0)
iy = np.clip(np.searchsorted(-elat, -g.lat.values), 0, len(elat) - 1)
ix = np.clip(np.searchsorted(elon, g.lon.values), 0, len(elon) - 1)
for k, gr in E5.items():
    g[k] = gr[iy, ix].astype(np.float32)
g = g.dropna(subset=["e5_maat"]).reset_index(drop=True)
g["logd"] = np.log1p(g.depth_m)
FFtr = fourier(g.depth_m.values)
FFn = [f"ff{i}" for i in range(FFtr.shape[1])]
for i, n in enumerate(FFn):
    g[n] = FFtr[:, i]
FEAT = E5F + ["depth_m", "logd"] + FFn

# ---------------- 엔진 선택(게이트 승자) ----------------
engine = "GBM"
try:
    r = pd.read_csv("data/processed/b1b_results.csv")
    tr_ = r[r.split == "지역전이"].sort_values("rmse")
    if tr_.iloc[0].model.startswith("NF"):
        engine = "NF"
except Exception:
    pass
print("엔진:", engine)


class NF(nn.Module):
    def __init__(self, d, w=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, w), nn.SiLU(), nn.Linear(w, w), nn.SiLU(),
            nn.Linear(w, w // 2), nn.SiLU(), nn.Linear(w // 2, 1))
    def forward(self, x): return self.net(x).squeeze(1)

if engine == "NF":
    ck = torch.load("outputs/models/b1b_neural_field.pt", map_location="cpu")
    mu, sd, ymu, ysd = ck["mu"], ck["sd"], ck["ymu"], ck["ysd"]
    nets = []
    for st in ck["states"]:
        n_ = NF(len(FEAT)).to(dev); n_.load_state_dict(st); n_.eval(); nets.append(n_)
    def predict(Xf):
        Xz = ((Xf - mu) / sd).astype(np.float32)
        ps = []
        with torch.no_grad():
            for n_ in nets:
                out = []
                for k in range(0, len(Xz), 65536):
                    out.append(n_(torch.tensor(Xz[k:k + 65536]).to(dev)).cpu().numpy())
                ps.append(np.concatenate(out) * ysd + ymu)
        return np.mean(ps, 0)
else:
    gbm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0)
    gbm.fit(g[FEAT].values, g.temp_c.values)
    def predict(Xf): return gbm.predict(Xf)

# ---------------- 북사면 격자 T(x,y,z) 큐브 ----------------
sub = clim0.sel(latitude=slice(N, S), longitude=slice(Wl, E))
Gs = derive(sub)
glat, glon = sub["latitude"].values, sub["longitude"].values
NLAT, NLON = Gs["e5_maat"].shape
base = np.column_stack([Gs[f].ravel() for f in E5F])
land = np.isfinite(base).all(1)
print(f"격자 {NLAT}×{NLON}, 육지 {land.sum():,}")
cube = np.full((len(DEPTHS), NLAT * NLON), np.nan, dtype=np.float32)
for i, d in enumerate(DEPTHS):
    dm = np.full(land.sum(), d, dtype=np.float32)
    Xf = np.column_stack([base[land], dm, np.log1p(dm), fourier(dm)])
    cube[i, land] = predict(Xf)
cube = cube.reshape(len(DEPTHS), NLAT, NLON)
np.save("data/processed/thermal_cube_alaska.npy", cube)
print(f"큐브 완료: T {np.nanmin(cube):.1f}~{np.nanmax(cube):.1f}°C")

# ---------------- (1) 3D 층층 슬라이스 ----------------
from mpl_toolkits.mplot3d import Axes3D  # noqa
fig = plt.figure(figsize=(13.5, 9))
ax = fig.add_subplot(111, projection="3d")
LON2, LAT2 = np.meshgrid(glon, glat)
norm = tnorm(-8, 4)
for d in [0.5, 2, 5, 10, 15, 20]:
    i = int(np.argmin(np.abs(DEPTHS - d)))
    Z = cube[i]
    fc = CMAP.temp(norm(Z))
    fc[..., 3] = np.where(np.isfinite(Z), 0.92, 0.0)
    ax.plot_surface(LON2, LAT2, np.full_like(Z, -d), facecolors=fc,
                    rstride=1, cstride=2, shade=False, linewidth=0)
    ax.text(E + 0.3, S, -d, f"{d:g} m", fontsize=9)
m = cm.ScalarMappable(norm=norm, cmap=CMAP.temp); m.set_array([])
cb = fig.colorbar(m, ax=ax, shrink=0.6, pad=0.06); cb.set_label("지중온도 (°C) — 파랑=동결")
ax.set_xlabel("경도"); ax.set_ylabel("위도"); ax.set_zlabel("깊이 (m)")
ax.set_zlim(-21, 0); ax.view_init(elev=22, azim=-60)
ax.set_title(f"알래스카 3D 지중 열구조 (엔진: {engine} 조건장, 깊이 0.5~20m 슬라이스)",
             fontsize=13, weight="bold")
fig.tight_layout()
fig.savefig(volpath("thermal_cube_alaska")); plt.close(fig)
print("saved", volpath("thermal_cube_alaska"))

# ---------------- (2) MAGT 2m/20m 지도 + 0°C 등고선(영구동토 경계) + 실측 ----------------
def slice_at(d):
    return cube[int(np.argmin(np.abs(DEPTHS - d)))]

bh = g[(g.lat.between(S, N)) & (g.lon.between(Wl, E))]
fig, axes = plt.subplots(2, 1, figsize=(13, 11), sharex=True)
for ax_, d, dlo, dhi in [(axes[0], 2, 1.0, 3.5), (axes[1], 20, 12, 28)]:
    Z = slice_at(d)
    mesh = ax_.pcolormesh(glon, glat, Z, cmap=CMAP.temp, norm=tnorm(-8, 4), shading="auto")
    cs = ax_.contour(glon, glat, Z, levels=[0], colors=FROZEN, linewidths=1.6)
    ax_.clabel(cs, fmt="0°C", fontsize=9)
    ob = bh[bh.depth_m.between(dlo, dhi)].groupby("site").agg(
        lat=("lat", "first"), lon=("lon", "first"), t=("temp_c", "mean")).reset_index()
    ax_.scatter(ob.lon, ob.lat, c=ob.t, s=55, cmap=CMAP.temp, norm=tnorm(-8, 4),
                edgecolors="k", linewidths=0.8, zorder=5)
    fig.colorbar(mesh, ax=ax_, shrink=0.9).set_label("연평균 지중온도 (°C)")
    ax_.set_ylabel("위도"); ax_.set_xlim(Wl, E); ax_.set_ylim(S, N)
    ax_.set_title(f"깊이 {d}m — 파란 0°C 등고선=영구동토 경계, 테두리점=시추공 실측({len(ob)}곳)",
                  fontsize=12, weight="bold")
axes[1].set_xlabel("경도")
fig.suptitle(f"알래스카 연평균 지중온도(MAGT) 지도 — 엔진 {engine} 조건장\n"
             "남부(온난·비동토) ↔ 북사면(한랭·연속 영구동토) 대비", fontsize=14, weight="bold")
fig.tight_layout()
fig.savefig(mappath("magt_alaska_2m_20m")); plt.close(fig)
print("saved", mappath("magt_alaska_2m_20m"))

# ---------------- (3) 깊이 내려가는 GIF ----------------
fig, ax = plt.subplots(figsize=(11, 5.1))
i0 = 0
mesh = ax.pcolormesh(glon, glat, cube[i0], cmap=CMAP.temp, norm=tnorm(-8, 4), shading="auto")
cb = fig.colorbar(mesh, ax=ax, shrink=0.9); cb.set_label("지중온도 (°C) — 파랑=동결")
tt = ax.set_title("깊이 0.00 m — 지중온도 수평 슬라이스", fontsize=12, weight="bold")
ax.set_xlabel("경도 (°E)"); ax.set_ylabel("위도 (°N)")
fig.subplots_adjust(top=0.88, bottom=0.11, left=0.07, right=0.99)  # GIF는 tight bbox 미적용 → 제목 여백 확보
frames = list(range(0, len(DEPTHS), 2))
def update(fi):
    mesh.set_array(cube[fi].ravel())
    tt.set_text(f"깊이 {DEPTHS[fi]:.2f} m — 지중온도 수평 슬라이스 (엔진 {engine})")
    return mesh, tt
ani = FuncAnimation(fig, update, frames=frames, interval=120)
ani.save(animpath("thermal_depth_slices"), writer=PillowWriter(fps=8), dpi=100)
plt.close(fig)
print("saved", animpath("thermal_depth_slices"))
