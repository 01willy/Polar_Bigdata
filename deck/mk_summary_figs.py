# -*- coding: utf-8 -*-
"""중간보고 슬림 덱(10p)용 신규 그림.
- connection.png     : ALT(main) + 3D 지중온도장(증강) + Stefan(물리) 연결 개념도 (슬라이드 3)
- uncertainty_map.png: 예측 불확실성(구간 폭) 지도 + 적용범위(AOA) 지도 (슬라이드 5)
빌드: (ROOT) python3 deck/mk_summary_figs.py
"""
import os, sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

for _f in ["Regular", "Medium", "SemiBold", "Bold", "ExtraBold"]:
    try: fm.fontManager.addfont(f"/home/willy010313/.fonts/Pretendard-{_f}.otf")
    except Exception: pass
FP  = fm.FontProperties(fname="/home/willy010313/.fonts/Pretendard-SemiBold.otf")
FPB = fm.FontProperties(fname="/home/willy010313/.fonts/Pretendard-ExtraBold.otf")
FPM = fm.FontProperties(fname="/home/willy010313/.fonts/Pretendard-Medium.otf")
plt.rcParams.update({"axes.unicode_minus": False})

OUT = "deck/assets/mid"; os.makedirs(OUT, exist_ok=True)
INK="#18181B"; SLATE="#4A5158"; MUTE="#8A9096"; PAPER="#FFFFFF"
SLATEBG="#E7ECEF"; TEALBG="#DCEBEA"; NAVY="#12324B"; TEAL="#0E7C86"; GOLD="#B9822E"
NAVYBG="#12324B"; GOLDBG="#F3E7CF"

# ================================================================= 1. 연결 개념도
def box(ax, x, y, w, h, title, body, fc, ec, tc="#18181B", bc="#4A5158", ts=13, bs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.6, zorder=3))
    ax.text(x+w/2, y+h-0.30, title, ha="center", va="top", fontsize=ts,
            fontproperties=FPB, color=tc, zorder=4)
    if body:
        ax.text(x+w/2, y+h-0.66, body, ha="center", va="top", fontsize=bs,
                fontproperties=FPM, color=bc, zorder=4, linespacing=1.35)

def arrow(ax, x0, y0, x1, y1, color=TEAL, lw=2.2, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=16,
                                 color=color, lw=lw, zorder=2,
                                 shrinkA=2, shrinkB=2))

fig, ax = plt.subplots(figsize=(13.2, 5.9), dpi=200)
fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
ax.set_xlim(0, 13.2); ax.set_ylim(0, 5.9); ax.axis("off")

# 열 좌표
c1x, c2x, c3x = 0.25, 4.35, 9.15
bw1, bw2, bw3 = 3.7, 4.35, 3.8
bh = 1.18
r = [4.35, 2.78, 1.21]   # 상·중·하 행 y(하단)

# 컬럼 1: 데이터 상황
box(ax, c1x, r[0], bw1, bh, "알래스카", "ALT 프로브 관측 (약 22만점)", SLATEBG, "#9FB2BC")
box(ax, c1x, r[1], bw1, bh, "러시아·유럽", "시추공 지중온도 (260사이트)", SLATEBG, "#9FB2BC")
box(ax, c1x, r[2], bw1, bh, "광역 (대부분)", "지하 관측 없음\n지표 기후(ERA5)만 존재", SLATEBG, "#9FB2BC")

# 컬럼 2: 처리 경로
box(ax, c2x, r[0], bw2, bh, "직접 학습", "관측 ALT를 라벨로 사용", TEALBG, TEAL, tc=NAVY, bc=TEAL)
box(ax, c2x, r[1], bw2, bh, "3D 지중온도장 (증강)",
    "0°C 등온면 깊이 → ALT 라벨 생성", TEALBG, TEAL, tc=NAVY, bc=TEAL)
box(ax, c2x, r[2], bw2, bh, "Stefan 물리",
    "기후(TDD) → ALT 예측 · 잔차 사전지식", GOLDBG, GOLD, tc="#6B4F1D", bc=GOLD)

# 컬럼 3: 통합 → 산출
cy = r[2] + (r[0]+bh - r[2]) / 2 - 0.62
ax.add_patch(FancyBboxPatch((c3x, cy), bw3, 1.24, boxstyle="round,pad=0.02,rounding_size=0.14",
                            fc=NAVYBG, ec=NAVY, lw=1.8, zorder=3))
ax.text(c3x+bw3/2, cy+0.86, "ALT 예측 모델", ha="center", va="center", fontsize=15,
        fontproperties=FPB, color="white", zorder=4)
ax.text(c3x+bw3/2, cy+0.40, "main 산출", ha="center", va="center", fontsize=11,
        fontproperties=FPM, color="#BFD3DE", zorder=4)
# 산출물 배너(하단)
oy = 0.10
ax.add_patch(FancyBboxPatch((c3x, oy), bw3, 0.78, boxstyle="round,pad=0.02,rounding_size=0.10",
                            fc=GOLDBG, ec=GOLD, lw=1.5, zorder=3))
ax.text(c3x+bw3/2, oy+0.39, "산출: ALT 2D 지도 + 불확실성", ha="center", va="center",
        fontsize=12.5, fontproperties=FPB, color="#6B4F1D", zorder=4)

# 화살표: 컬럼1 → 컬럼2 (행 대응)
for ry in r:
    arrow(ax, c1x+bw1, ry+bh/2, c2x, ry+bh/2, color="#7C8A92", lw=2.0)
# 컬럼2 → 통합(ALT 모델)
for ry in r:
    col = GOLD if ry == r[2] else TEAL
    arrow(ax, c2x+bw2, ry+bh/2, c3x, cy+0.62, color=col, lw=2.0)
# ALT 모델 → 산출
arrow(ax, c3x+bw3/2, cy, c3x+bw3/2, oy+0.78, color=NAVY, lw=2.4)

# 공유 경계 주석(하단 좌측)
ax.text(0.25, 0.30, "ALT = 3D 지중온도장이 0°C를 지나는 깊이. 세 경로가 이 경계를 공유한다.",
        ha="left", va="center", fontsize=11.5, fontproperties=FP, color=SLATE)
# 열 라벨
for lx, lab in [(c1x+bw1/2, "데이터 상황"), (c2x+bw2/2, "ALT 확보 경로"), (c3x+bw3/2, "통합·산출")]:
    ax.text(lx, 5.72, lab, ha="center", va="center", fontsize=11.5, fontproperties=FPB,
            color=MUTE)
fig.savefig(f"{OUT}/connection.png", dpi=200, facecolor=PAPER, bbox_inches="tight")
plt.close(fig); print("saved connection.png")

# ================================================================= 2. 불확실성·적용범위 지도
from polar.plotstyle import use_polar, CMAP, add_cbar, style_geo, BAD
o = pd.read_csv("data/processed/alt_cell_best_oof.csv")
o = o.dropna(subset=["lo_cqr", "hi_cqr", "lat", "lon"]).copy()
o["width"] = (o.hi_cqr - o.lo_cqr).clip(lower=0)
# 배경 육지 마스크(ERA5-Land)
import xarray as xr
ds = xr.open_dataset("data/raw/era5land/nh_monthly_2015-2020.nc")
tn = "valid_time" if "valid_time" in ds.coords else "time"
t0 = ds["t2m"].isel({tn: 0})
land = np.isfinite(t0.values)
elat = ds["latitude"].values; elon = ds["longitude"].values
LO, HI, SO, NO = o.lon.min()-1.5, o.lon.max()+1.5, o.lat.min()-1.0, o.lat.max()+1.0

plt.rcParams.update({"font.family": FP.get_name()})
fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), dpi=200)
fig.patch.set_facecolor(PAPER)
lon2, lat2 = np.meshgrid(elon, elat)
for ax in axes:
    ax.set_facecolor("white")
    ax.pcolormesh(elon, elat, np.where(land, 0.0, np.nan), cmap=matplotlib.colors.ListedColormap(["#eef1f3"]),
                  shading="auto", zorder=0)
    ax.set_xlim(LO, HI); ax.set_ylim(SO, NO)

# (a) 예측 불확실성(90% 구간 폭)
vmax = float(np.nanpercentile(o.width, 96))
sc0 = axes[0].scatter(o.lon, o.lat, c=o.width, s=8, cmap=CMAP.err, vmin=0, vmax=vmax,
                      linewidths=0, zorder=3)
cb0 = fig.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.02, extend="max")
cb0.set_label("90% 예측구간 폭 (cm)", fontsize=15, fontproperties=FP)
cb0.ax.tick_params(labelsize=13)
for t in cb0.ax.get_yticklabels(): t.set_fontproperties(FPM)
axes[0].set_title("(a) 예측 불확실성 지도", fontsize=18, fontproperties=FPB, color=INK, loc="left", pad=8)

# (b) 실측 대비 예측 오차 |pred-obs| (공간블록 OOF, 전 셀 계산)
o["aerr"] = (o["pred"] - o["alt_cm"]).abs()
emax = float(np.nanpercentile(o.aerr, 96))
sc1 = axes[1].scatter(o.lon, o.lat, c=o.aerr, s=8, cmap=CMAP.err, vmin=0, vmax=emax,
                      linewidths=0, zorder=3)
cb1 = fig.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.02, extend="max")
cb1.set_label("|예측 − 관측| (cm)", fontsize=15, fontproperties=FP)
cb1.ax.tick_params(labelsize=13)
for t in cb1.ax.get_yticklabels(): t.set_fontproperties(FPM)
axes[1].set_title("(b) 실측 대비 예측 오차 지도", fontsize=18, fontproperties=FPB, color=INK, loc="left", pad=8)

for ax in axes:
    ax.set_xlabel("경도 (°E)", fontsize=14, fontproperties=FP)
    ax.set_ylabel("위도 (°N)", fontsize=14, fontproperties=FP)
    ax.tick_params(labelsize=12.5)
    for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontproperties(FPM)
    for sp in ax.spines.values(): sp.set_color("#C9CDD2"); sp.set_linewidth(0.9)
    ax.grid(color="#E6E8EB", lw=0.6, zorder=1); ax.set_axisbelow(True)
fig.text(0.01, -0.01, "자료: alt_cell_best_oof.csv · 알래스카 셀 · (a) 90% 예측구간 폭, (b) 공간블록 OOF 예측 오차. 둘이 정합하면 잘 보정된 것",
         fontsize=9.5, color=MUTE, fontproperties=FPM, ha="left", va="top")
fig.tight_layout()
fig.savefig(f"{OUT}/uncertainty_map.png", dpi=200, facecolor=PAPER, bbox_inches="tight")
plt.close(fig); print("saved uncertainty_map.png")

# ================================================================= 3. MAGT 2m/20m 지도(대형 폰트·냉색·em-dash 없음)
# 슬라이드 7 전용. vol_thermal_field_alaska.py의 학습 로직을 재사용해 지중온도 격자를 만들고
# 2m·20m 두 깊이만 큰 폰트로 조판한다(원 산출물의 소형 폰트·em-dash 문제 해소).
import calendar
from sklearn.ensemble import HistGradientBoostingRegressor
from polar.plotstyle import tnorm as _tnorm
E5F = ["e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
def _fourier(dm):
    dn = (dm / 30.0).astype(np.float32); out = []
    for k in range(5): out += [np.sin(2**k*np.pi*dn), np.cos(2**k*np.pi*dn)]
    return np.column_stack(out)
gt = pd.read_csv("data/processed/ground_temp_all.csv")
gt = gt[(gt.depth_m > 0) & (gt.depth_m <= 30) & (gt.temp_c > -25) & (gt.temp_c < 25)].reset_index(drop=True)
ds2 = xr.open_dataset("data/raw/era5land/nh_monthly_2015-2020.nc")
tn2 = "valid_time" if "valid_time" in ds2.coords else "time"
clim = ds2.assign_coords(month=ds2[tn2].dt.month).groupby("month").mean(tn2)
_days = np.array([calendar.monthrange(2019, m)[1] for m in range(1, 13)])[:, None, None]
def _derive(c):
    t = c["t2m"].values - 273.15; stl = c["stl1"].values - 273.15; sdp = c["sd"].values
    tdd = np.nansum(np.clip(t, 0, None)*_days, 0); fdd = np.nansum(np.clip(-t, 0, None)*_days, 0)
    return dict(e5_maat=np.nanmean(t, 0), e5_tdd=tdd, e5_fdd=fdd, e5_sqrt_tdd=np.sqrt(tdd),
                e5_twarm=np.nanmax(t, 0), e5_tcold=np.nanmin(t, 0),
                e5_stl1=np.nanmean(stl, 0), e5_swe=np.nanmean(sdp, 0))
E5 = _derive(clim)
elat2, elon2 = clim["latitude"].values, clim["longitude"].values
iy = np.clip(np.searchsorted(-elat2, -gt.lat.values), 0, len(elat2)-1)
ix = np.clip(np.searchsorted(elon2, gt.lon.values), 0, len(elon2)-1)
for k, gr in E5.items(): gt[k] = gr[iy, ix].astype(np.float32)
gt = gt.dropna(subset=["e5_maat"]).reset_index(drop=True)
FEATm = E5F + ["depth_m", "logd"] + [f"ff{i}" for i in range(10)]
gt["logd"] = np.log1p(gt.depth_m)
FF = _fourier(gt.depth_m.values)
for i in range(10): gt[f"ff{i}"] = FF[:, i]
gbm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(gt[FEATm].values, gt.temp_c.values)
Nb, Sb, Wb, Eb = 72.0, 60.0, -166.0, -141.0
sub = clim.sel(latitude=slice(Nb, Sb), longitude=slice(Wb, Eb))
Gs = _derive(sub); glat, glon = sub["latitude"].values, sub["longitude"].values
NLAT, NLON = Gs["e5_maat"].shape
base = np.column_stack([Gs[f].ravel() for f in E5F]); landm = np.isfinite(base).all(1)
def _pred_depth(d):
    z = np.full(NLAT*NLON, np.nan, np.float32); dm = np.full(landm.sum(), d, np.float32)
    Xf = np.column_stack([base[landm], dm, np.log1p(dm), _fourier(dm)])
    z[landm] = gbm.predict(Xf); return z.reshape(NLAT, NLON)
fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), dpi=200); fig.patch.set_facecolor(PAPER)
norm = _tnorm(-9, 3)
for ax, d in zip(axes, [2.0, 20.0]):
    Z = _pred_depth(d)
    ax.set_facecolor(BAD)
    mesh = ax.pcolormesh(glon, glat, Z, cmap=CMAP.temp, norm=norm, shading="auto", zorder=1)
    cs = ax.contour(glon, glat, Z, levels=[0], colors="#12324B", linewidths=1.6, zorder=2)
    ax.clabel(cs, fmt="0°C", fontsize=12)
    # 실측점은 예측 격자(알래스카)와 같은 범위만 표시(전지구 시추공이 축을 늘려 깨지는 문제 방지)
    ob = gt[(gt.depth_m.between(d-0.5, d+0.5)) & (gt.lon.between(Wb, Eb)) & (gt.lat.between(Sb, Nb))]
    if len(ob):
        ax.scatter(ob.lon, ob.lat, c=ob.temp_c, cmap=CMAP.temp, norm=norm, s=30,
                   edgecolors="#111", linewidths=0.5, zorder=3)
    ax.set_xlim(Wb, Eb); ax.set_ylim(Sb, Nb)
    cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02, extend="both")
    cb.set_label("연평균 지중온도 (°C)", fontsize=14, fontproperties=FP)
    cb.ax.tick_params(labelsize=12.5)
    for t in cb.ax.get_yticklabels(): t.set_fontproperties(FPM)
    ax.set_title(f"깊이 {d:.0f} m", fontsize=18, fontproperties=FPB, color=INK, loc="left", pad=8)
    ax.set_xlabel("경도 (°E)", fontsize=14, fontproperties=FP)
    ax.set_ylabel("위도 (°N)", fontsize=14, fontproperties=FP)
    ax.tick_params(labelsize=12.5)
    for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontproperties(FPM)
    for sp in ax.spines.values(): sp.set_color("#C9CDD2"); sp.set_linewidth(0.9)
fig.suptitle("알래스카 연평균 지중온도(MAGT). 0°C 등온선이 영구동토 상단 경계",
             fontsize=17, fontproperties=FPB, color=INK, x=0.01, ha="left", y=1.02)
fig.text(0.01, -0.01, "자료: ground_temp_all.csv 학습 GBM 조건장 · ERA5-Land 격자 예측 · 테두리점=시추공 실측",
         fontsize=9.5, color=MUTE, fontproperties=FPM, ha="left", va="top")
fig.tight_layout()
fig.savefig(f"{OUT}/magt_clean.png", dpi=200, facecolor=PAPER, bbox_inches="tight")
plt.close(fig); print("saved magt_clean.png")
