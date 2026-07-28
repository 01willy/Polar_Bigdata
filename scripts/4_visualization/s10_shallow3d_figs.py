"""S10 시각화: 얕은 3D 온도장(0-3 m) 재렌더 · 깊이별 지도·남북 단면·검증.

그림(스펙: figures/figure_spec.json의 s10_* 항목)
1. outputs/figures/s10_shallow3d/s10_tmax_depth_maps.{png,pdf}
   깊이 0.5/1/2/3 m t_max 지도 4패널(cartopy Albers, vik 0°C 중심, hexbin median,
   mask_ocean, inset locator+스케일바).
2. outputs/figures/s10_shallow3d/s10_tmax_section_thaw.{png,pdf}
   (a) 남북 단면(Dalton 회랑, 위도x깊이, 0°C 등온선) + (b) 0°C 교차 깊이 지도
   (지원반경 15 km 격자, 관측 ALT 오버레이).
3. outputs/figures/s10_shallow3d/s10_validation.{png,pdf}
   (a) 깊이밴드 RMSE(단조 vs 무제약) + (b) 유도 ALT vs 실측 산점(OOF, r=0.28, R2<0).

입력: shallow3d_alaska_{grid,results,altmatch}.csv(재현 확인분) + s10_shallow3d_volume.npz.
기존 shallow3d_alaska_figs.py는 수정하지 않는다(신규 파일).

실행: /home/anaconda3/bin/python scripts/4_visualization/s10_shallow3d_figs.py  (ROOT, CPU)
"""
import sys, os
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm
from polar.plotstyle import use_polar, CMAP, tnorm, BAD
from polar.geomap import (make_ax, ALASKA, hexbin_map, field_map, mask_ocean,
                          add_inset_locator, add_scalebar, scatter_map)
import cartopy.crs as ccrs

plt = use_polar()
PROC = "data/processed"
OUTDIR = "outputs/figures/s10_shallow3d"
os.makedirs(OUTDIR, exist_ok=True)

grid = pd.read_csv(os.path.join(PROC, "shallow3d_alaska_grid.csv"))
res = pd.read_csv(os.path.join(PROC, "shallow3d_alaska_results.csv"))
altm = pd.read_csv(os.path.join(PROC, "shallow3d_alaska_altmatch.csv"))
vol = np.load(os.path.join(PROC, "s10_shallow3d_volume.npz"))
lats, lons, depths = vol["lats"], vol["lons"], vol["depths_m"]
tvol, thaw = vol["tmax_c"], vol["thaw_cm"]

PROJ = ccrs.AlbersEqualArea(central_longitude=-154.0, central_latitude=65.5,
                            standard_parallels=(62.0, 69.0))


def save(fig, name):
    png = os.path.join(OUTDIR, f"{name}.png")
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {png}")


# ---------- 그림 1: 깊이별 t_max 지도(4패널) ----------
DEPTH_PANELS = [0.5, 1.0, 2.0, 3.0]
vmax = max(1.0, float(np.nanpercentile(np.abs(grid.tmax_pred), 99)))
norm_t = tnorm(-vmax, vmax, 0.0)
fig = plt.figure(figsize=(11.6, 9.2), constrained_layout=True)
axes = []
for i, dm in enumerate(DEPTH_PANELS):
    ax = fig.add_subplot(2, 2, i + 1, projection=PROJ)
    make_ax(ALASKA, ax=ax, fig=fig)
    g = grid[np.isclose(grid.depth_m, dm)]
    hb = hexbin_map(ax, g.lon.values, g.lat.values, g.tmax_pred.values,
                    gridsize=58, cmap=CMAP.temp, norm=norm_t)
    mask_ocean(ax)
    ax.set_title(f"({chr(97+i)}) 깊이 {dm:g} m", fontsize=11.5)
    axes.append(ax)
add_inset_locator(fig, axes[0], ALASKA)
add_scalebar(axes[0], loc="lower right")
cb = fig.colorbar(hb, ax=axes, shrink=0.74, pad=0.02, aspect=30, extend="both")
cb.set_label("연최대 지중온도 (°C)  ·  0°C 중심(청=동결 유지, 갈=여름 융해)", fontsize=10)
fig.suptitle("알래스카 얕은 지중 연최대온도 t_max 깊이별 지도 (GBM 조건장, 0-3 m)\n"
             "hexbin 셀 median(보간 없음) · 사이트블록 검증 $R^2$ 0.47 (온도 예측)",
             fontsize=13, fontweight="bold")
save(fig, "s10_tmax_depth_maps")

# ---------- 그림 2: 남북 단면 + 0°C 교차 깊이 지도 ----------
LON0, LON1 = -150.5, -147.5   # Dalton 회랑(Deadhorse-Fairbanks)
mlon = (lons >= LON0) & (lons <= LON1)
sect = np.nanmedian(tvol[:, :, mlon], axis=2)      # (ndepth, nlat)
has = np.isfinite(sect).any(axis=0)
la0, la1 = lats[has].min() - 0.2, lats[has].max() + 0.2
svmax = max(1.0, float(np.nanpercentile(np.abs(sect), 99)))
norm_s = tnorm(-svmax, svmax, 0.0)

# 갭(지원반경 밖) 색을 vik 근-0°C 밝은색과 구분되도록 진한 회색으로(이 패널 한정 사본).
GAP_GRAY = "#b0b0b0"
cmap_sect = CMAP.temp.copy()
cmap_sect.set_bad(GAP_GRAY)

fig = plt.figure(figsize=(13.2, 5.4))
axa = fig.add_subplot(1, 2, 1)
pm = axa.pcolormesh(lats, depths, np.ma.masked_invalid(sect), cmap=cmap_sect,
                    norm=norm_s, shading="auto", rasterized=True)
fin = np.isfinite(sect)
if fin.any():
    # 0°C 등온선은 실제 부호변화(음↔양)가 있는 열에서만 그린다. 지원반경 밖(갭)이나
    # 창(0-3 m) 내 부호변화 없는 열은 masked 처리해 등온선을 끊어 수직 스파이크를 제거한다.
    cmin = np.nanmin(np.where(fin, sect, np.nan), axis=0)
    cmax = np.nanmax(np.where(fin, sect, np.nan), axis=0)
    cross_col = fin.any(axis=0) & (cmin < 0) & (cmax > 0)   # 위도별 실제 교차 여부
    csrc = np.where(fin, sect, np.nan)
    csrc[:, ~cross_col] = np.nan
    axa.contour(lats, depths, np.ma.masked_invalid(csrc), levels=[0.0],
                colors="k", linewidths=1.4)
axa.plot([], [], color="k", lw=1.4, label="0°C 등온선(교차 구간만)")
axa.set_xlim(la0, la1)
axa.set_ylim(depths.max(), 0)
axa.set_xlabel("위도 (°N)")
axa.set_ylabel("깊이 (m)")
axa.set_title(f"(a) 남북 온도 단면  ·  경도 {LON0:g}~{LON1:g}° 중앙값(Dalton 회랑)")
axa.legend(loc="lower left")
cba = fig.colorbar(pm, ax=axa, shrink=0.9, pad=0.02, extend="both")
cba.set_label("연최대 지중온도 (°C)", fontsize=10)
axa.text(0.99, 0.02, "진회색=지원반경(15 km) 밖(결측)", transform=axa.transAxes,
         ha="right", va="bottom", fontsize=8, color="0.25")

axb = fig.add_subplot(1, 2, 2, projection=PROJ)
make_ax(ALASKA, ax=axb, fig=fig)
lon2d, lat2d = np.meshgrid(lons, lats)
LEV = [30, 45, 60, 75, 90, 105, 120, 135, 150]
bn = BoundaryNorm(LEV, CMAP.alt.N, extend="both")
fm = field_map(axb, lon2d, lat2d, thaw, cmap=CMAP.alt, norm=bn)
sc = scatter_map(axb, altm.lon.values, altm.lat.values, altm.alt_obs_cm.values,
                 cmap=CMAP.alt, norm=bn, s=46, edge=True)
mask_ocean(axb)
axb.add_patch(mpatches.Rectangle((LON0, la0), LON1 - LON0, la1 - la0,
                                 transform=ccrs.PlateCarree(), fill=False,
                                 edgecolor="#2d2d2d", linewidth=1.2, linestyle="--", zorder=6))
both = altm.dropna(subset=["alt_pred_cm"])
r_oof = float(np.corrcoef(both.alt_obs_cm, both.alt_pred_cm)[0, 1])
axb.set_title("(b) 0°C 교차 깊이(전량학습 산출)  ·  테두리 점=실측 ALT")
cbb = fig.colorbar(fm, ax=axb, shrink=0.82, pad=0.02, extend="both")
cbb.set_label("0°C 교차 깊이 · 실측 ALT (cm)", fontsize=10)
axb.text(0.02, 0.02, f"검증(OOF 유도 ALT): r={r_oof:.2f}, $R^2$<0 (절대 정합 미성립)\n점선=단면 (a) 회랑",
         transform=axb.transAxes, fontsize=8.5, va="bottom",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9))
fig.suptitle("얕은 온도장 단면과 0°C 등온선 깊이  ·  격자는 관측셀 지원반경 15 km 한정(연속장 아님)",
             fontsize=12.5, fontweight="bold")
save(fig, "s10_tmax_section_thaw")

# ---------- 그림 3: 검증(밴드 RMSE + ALT 산점) ----------
band_order = ["0.02-0.5m", "0.5-1m", "1-2m", "2-3m", "all"]
sb = res[res.cv == "site_block6"]
mono = sb[sb.model == "gbm_mono"].set_index("band").reindex(band_order)
free = sb[sb.model == "gbm_free"].set_index("band").reindex(band_order)
fig, (axr, axs) = plt.subplots(1, 2, figsize=(12.4, 5.0), constrained_layout=True)
x = np.arange(len(band_order))
w = 0.38
axr.bar(x - w / 2, mono.rmse_cm, w, label="단조 제약", color="#2166ac")
axr.bar(x + w / 2, free.rmse_cm, w, label="무제약", color="#88a8c8")
for xi, v in zip(x - w / 2, mono.rmse_cm):
    axr.text(xi, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
axr.set_xticks(x)
axr.set_xticklabels(band_order, rotation=12)
axr.set_ylabel("RMSE (°C)")
axr.set_title("(a) 깊이밴드별 t_max RMSE  ·  사이트블록 6-fold")
axr.legend(loc="upper right")

am = res[(res.cv == "alt_match") & (res.band == "0C_crossing")].iloc[0]
trunc = both.alt_obs_near_maxdepth.astype(bool).values
lim = [0, max(both.alt_obs_cm.max(), both.alt_pred_cm.max()) * 1.08]
axs.plot(lim, lim, "--", color="#444", lw=1.0, label="1:1")
axs.scatter(both.alt_obs_cm[~trunc], both.alt_pred_cm[~trunc], s=56, c="#2166ac",
            edgecolors="white", linewidths=0.6, zorder=3, label="시추공")
axs.scatter(both.alt_obs_cm[trunc], both.alt_pred_cm[trunc], s=56, facecolors="none",
            edgecolors="#2166ac", linewidths=1.4, zorder=3, label="절단 후보(최심깊이 근접)")
axs.set_xlim(lim)
axs.set_ylim(lim)
axs.set_xlabel("실측 시추공 envelope ALT (cm)")
axs.set_ylabel("OOF 유도 ALT (cm)")
axs.set_title("(b) 0°C 교차 유도 ALT vs 실측  ·  OOF 포락선")
axs.text(0.03, 0.97, f"r = {r_oof:.2f},  RMSE = {am.rmse_cm:.0f} cm,  $R^2$ = {am.r2:.2f},  n = {int(am.n)}\n"
         "$R^2$<0: 자기평균보다 나쁨(절대 정합 미성립)",
         transform=axs.transAxes, va="top", fontsize=9.5,
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))
axs.legend(loc="lower right", fontsize=8.5)
fig.suptitle("S10 재현 검증  ·  온도 예측은 성립($R^2$ 0.47), ALT 유도 정합은 미성립($R^2$<0)",
             fontsize=12.5, fontweight="bold")
save(fig, "s10_validation")
print("[done] 3개 그림(PNG+PDF) 저장")


# ============================================================
# 지적6 대응: 얕은 3D warp 대신 논문 관례(깊이-슬라이스 평면·fence 단면·융해깊이 2D·프로파일)
# 재작성. 아래는 신규 추가 함수/그림만. 위 기존 코드/함수는 무수정.
# 논문 근거: Obu2019·Ran·CCI 관례는 깊이=색 평면지도, 트럼펫 다이어그램, 깊이-공간 등온선
# 컨투어. S10 볼륨은 지원반경 15 km 회랑(격자의 4.7%)이라 3D warp 판독이 나쁘다. 온도장
# 예측은 site-block R2 0.47로 성립, 유도 ALT는 r=0.28(정성 보조, 절대 정합 미성립).
# ============================================================
lon2d_v, lat2d_v = np.meshgrid(lons, lats)


def save300(fig, name):
    """보고서용 고해상 저장(dpi 300). 기존 save()는 무수정으로 두고 신규 그림에만 적용."""
    png = os.path.join(OUTDIR, f"{name}.png")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig300] {png}")


# ---------- 그림 4(주력): 깊이-슬라이스 4패널 평면지도(0.5/1/1.5/2 m) ----------
# 3D warp의 논문형 대체. 각 깊이 평면을 지도에 색면(온도 vik, 0°C 중심 대칭)으로 깔고
# 0°C 등온선을 검은 윤곽으로 겹친다. 볼륨(지원반경 15 km 격자)에서 슬라이스를 취해
# 4패널 모두 동일 좌표·동일 clim을 공유한다(프레임 간 색한계 고정).
SLICE_DEPTHS = [0.5, 1.0, 1.5, 2.0]        # m — 지적6이 지정한 대표 깊이
CLIM_SLICE = 4.0                            # ±4°C 대칭(요구 예시), 0°C 중심 diverging
norm_slice = tnorm(-CLIM_SLICE, CLIM_SLICE, 0.0)
cmap_slice = CMAP.temp.copy()
cmap_slice.set_bad(BAD)                     # 지원반경 밖 = 중립 회색

fig = plt.figure(figsize=(11.6, 9.4), constrained_layout=True)
sax = []
for i, dm in enumerate(SLICE_DEPTHS):
    ax = fig.add_subplot(2, 2, i + 1, projection=PROJ)
    make_ax(ALASKA, ax=ax, fig=fig)
    di = int(np.argmin(np.abs(depths - dm)))
    sl = tvol[di]                          # (nlat, nlon), 지원 밖 NaN
    pm = field_map(ax, lon2d_v, lat2d_v, sl, cmap=cmap_slice, norm=norm_slice)
    # 0°C 등온선(동결/융해 경계). 유한값 있는 셀에서만 그린다.
    if np.isfinite(sl).any():
        ax.contour(lon2d_v, lat2d_v, np.ma.masked_invalid(sl), levels=[0.0],
                   colors="k", linewidths=1.2, transform=ccrs.PlateCarree(), zorder=4)
    mask_ocean(ax)
    ax.set_title(f"({chr(97+i)}) 깊이 {dm:g} m", fontsize=11.5)
    sax.append((ax, pm))
sax[0][0].plot([], [], color="k", lw=1.2, label="0°C 등온선")
sax[0][0].legend(loc="lower left", fontsize=8.0, framealpha=0.85)
add_inset_locator(fig, sax[0][0], ALASKA)
add_scalebar(sax[0][0], loc="lower right")
cb = fig.colorbar(sax[-1][1], ax=[a for a, _ in sax], shrink=0.74, pad=0.02,
                  aspect=30, extend="both")
cb.set_label("연최대 지중온도 (°C)  ·  0°C 중심(청=동결 유지, 갈=여름 융해)", fontsize=10)
fig.suptitle("알래스카 얕은 지중 연최대온도 깊이-슬라이스 지도 (0.5·1·1.5·2 m 평면)\n"
             "3D 대체(논문형 깊이 평면지도) · 색면=온도(vik ±4°C), 검은선=0°C 등온선 · 사이트블록 $R^2$ 0.47\n"
             "격자=관측셀 지원반경 15 km 회랑 한정(회색=지원 밖, 연속 온도장 아님)",
             fontsize=12.5, fontweight="bold")
save300(fig, "s10_depth_slices")


# ---------- 그림 5(보조A): 융해깊이(0°C 등온면 깊이) 2D 색면 ----------
# thaw_cm = 연최대온도 기준 0°C 교차 깊이(융해 최대 깊이 대리). oslo_r 순차형,
# clim은 데이터 분위(약 45-140 cm)에 맞춰 좁힌다.
tq = np.nanpercentile(thaw, [5, 95])       # 약 45, 145 cm
TH_LO, TH_HI = 45.0, 140.0
LEV_TH = [45, 60, 75, 90, 105, 120, 140]
bn_th = BoundaryNorm(LEV_TH, CMAP.alt.N, extend="both")

fig = plt.figure(figsize=(8.2, 7.4), constrained_layout=True)
axt = fig.add_subplot(1, 1, 1, projection=PROJ)
make_ax(ALASKA, ax=axt, fig=fig)
fmt = field_map(axt, lon2d_v, lat2d_v, thaw, cmap=CMAP.alt, norm=bn_th)
# 실측 ALT 시추공 오버레이(테두리 점) — 동일 색척도로 정성 비교
sct = scatter_map(axt, altm.lon.values, altm.lat.values, altm.alt_obs_cm.values,
                  cmap=CMAP.alt, norm=bn_th, s=48, edge=True)
mask_ocean(axt)
add_inset_locator(fig, axt, ALASKA)
add_scalebar(axt, loc="lower right")
cbt = fig.colorbar(fmt, ax=axt, shrink=0.82, pad=0.02, extend="both")
cbt.set_label("0°C 등온면 깊이 및 실측 ALT, 동일 색척도 (cm)", fontsize=10)
both = altm.dropna(subset=["alt_pred_cm"])
r_oof = float(np.corrcoef(both.alt_obs_cm, both.alt_pred_cm)[0, 1])
axt.text(0.02, 0.02,
         f"테두리 점=실측 시추공 ALT(동일 색척도)\n실측 ALT와 상관 r={r_oof:.2f} (정성 보조)\n"
         "절대 정합은 미성립($R^2$<0)",
         transform=axt.transAxes, fontsize=8.8, va="bottom",
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", alpha=0.92))
fig.suptitle("0°C 등온면 깊이 (융해깊이 대리) 2D 색면  ·  지원반경 15 km 회랑\n"
             "연최대온도 프로파일의 0°C 교차 깊이 · oslo_r 순차형(cm)",
             fontsize=12.0, fontweight="bold")
save300(fig, "s10_thaw_depth_map")


# ---------- 그림 6(보조B): fence 단면 2매(경도-깊이 / 위도-깊이) ----------
# matplotlib 2D fence. 세로축=실제 깊이(m, 과장 없음). 온도 vik(0°C 중심) + 0°C 등온선.
# 회랑을 가로지르는 두 대표 절단면을 나란히 둔다.
GAP_GRAY6 = "#b0b0b0"
cmap_fence = CMAP.temp.copy()
cmap_fence.set_bad(GAP_GRAY6)


def _isotherm_cols(field2d):
    """부호변화(음↔양)가 실제로 있는 열에서만 0°C선을 그리기 위한 소스(그 외 NaN)."""
    fin = np.isfinite(field2d)
    cmn = np.nanmin(np.where(fin, field2d, np.nan), axis=0)
    cmx = np.nanmax(np.where(fin, field2d, np.nan), axis=0)
    cross = fin.any(axis=0) & (cmn < 0) & (cmx > 0)
    src = np.where(fin, field2d, np.nan)
    src[:, ~cross] = np.nan
    return src


# 절단면 A: 위도 68.5°N 부근(North Slope 툰드라) 경도-깊이
LAT_FENCE = 68.5
jlat = int(np.argmin(np.abs(lats - LAT_FENCE)))
sect_lon = tvol[:, jlat, :]                # (ndepth, nlon)
# 절단면 B: 경도 -148.7° 부근(Dalton 회랑) 위도-깊이
LON_FENCE = -148.7
klon = int(np.argmin(np.abs(lons - LON_FENCE)))
sect_lat = tvol[:, :, klon]                # (ndepth, nlat)

fvmax = max(1.0, float(np.nanpercentile(np.abs(np.r_[sect_lon[np.isfinite(sect_lon)].ravel(),
                                                      sect_lat[np.isfinite(sect_lat)].ravel()]), 99)))
norm_f = tnorm(-fvmax, fvmax, 0.0)

fig = plt.figure(figsize=(13.4, 5.2), constrained_layout=True)

# (a) 경도-깊이
axf1 = fig.add_subplot(1, 2, 1)
hasA = np.isfinite(sect_lon).any(axis=0)
loA0, loA1 = lons[hasA].min() - 0.3, lons[hasA].max() + 0.3
pmA = axf1.pcolormesh(lons, depths, np.ma.masked_invalid(sect_lon), cmap=cmap_fence,
                      norm=norm_f, shading="auto", rasterized=True)
srcA = _isotherm_cols(sect_lon)
if np.isfinite(srcA).any():
    axf1.contour(lons, depths, np.ma.masked_invalid(srcA), levels=[0.0],
                 colors="k", linewidths=1.4)
axf1.plot([], [], color="k", lw=1.4, label="0°C 등온선(교차 열만)")
axf1.set_xlim(loA0, loA1)
axf1.set_ylim(depths.max(), 0.0)           # 깊이 아래 방향, 실제 m(과장 없음)
axf1.set_xlabel("경도 (°E)")
axf1.set_ylabel("깊이 (m)")
axf1.set_title(f"(a) 경도-깊이 fence  ·  위도 {LAT_FENCE:g}°N(North Slope)")
axf1.legend(loc="lower left", fontsize=8.5)
cbfA = fig.colorbar(pmA, ax=axf1, shrink=0.9, pad=0.02, extend="both")
cbfA.set_label("연최대 지중온도 (°C)", fontsize=10)

# (b) 위도-깊이
axf2 = fig.add_subplot(1, 2, 2)
hasB = np.isfinite(sect_lat).any(axis=0)
laB0, laB1 = lats[hasB].min() - 0.2, lats[hasB].max() + 0.2
pmB = axf2.pcolormesh(lats, depths, np.ma.masked_invalid(sect_lat), cmap=cmap_fence,
                      norm=norm_f, shading="auto", rasterized=True)
srcB = _isotherm_cols(sect_lat)
if np.isfinite(srcB).any():
    axf2.contour(lats, depths, np.ma.masked_invalid(srcB), levels=[0.0],
                 colors="k", linewidths=1.4)
axf2.plot([], [], color="k", lw=1.4, label="0°C 등온선(교차 열만)")
axf2.set_xlim(laB0, laB1)
axf2.set_ylim(depths.max(), 0.0)
axf2.set_xlabel("위도 (°N)")
axf2.set_ylabel("깊이 (m)")
axf2.set_title(f"(b) 위도-깊이 fence  ·  경도 {LON_FENCE:g}°E(Dalton 회랑)")
axf2.legend(loc="lower left", fontsize=8.5)
cbfB = fig.colorbar(pmB, ax=axf2, shrink=0.9, pad=0.02, extend="both")
cbfB.set_label("연최대 지중온도 (°C)", fontsize=10)
for _axf in (axf1, axf2):
    _axf.text(0.99, 0.02, "진회색=지원반경 15 km 밖(관측 희소 회랑)",
              transform=_axf.transAxes, ha="right", va="bottom", fontsize=8, color="0.25")

fig.suptitle("얕은 온도장 fence 단면 (2D, 세로축=실제 깊이 m, 과장 없음)  ·  3D warp 대체\n"
             "온도 vik(0°C 중심) + 검은 0°C 등온선(부호변화 있는 열만)\n"
             "진회색 영역(약 3/4)=지원반경 15 km 밖(관측 희소 회랑, 무예측) · 색면=예측 온도장",
             fontsize=12.0, fontweight="bold")
save300(fig, "s10_fence_sections")


# ---------- 그림 7(보조C): 대표 지점 T(z) 프로파일(트럼펫형) 예측 vs 실측 ----------
# 남→북 열구배를 대표하도록 4지점 선정(Fairbanks 내륙 온난, Franklin Bluff·Sagwon 툰드라,
# Barrow 최북 한랭). 실측 t_max(z)는 라벨 원본(ground_temp_gtnp_global.csv), 예측 T(z)는
# 볼륨의 최근접 지원노드 프로파일. 이는 시각화용 전량학습 산출이며 검증 지표가 아니다.
GT_PATH = os.path.join(PROC, "ground_temp_gtnp_global.csv")
gt_all = pd.read_csv(GT_PATH)
ak_lab = gt_all[(gt_all.lat >= 54) & (gt_all.lat <= 72) &
                (gt_all.lon >= -170) & (gt_all.lon <= -129)].dropna(subset=["depth", "t_max"])
ak_lab = ak_lab[(ak_lab.depth >= 0.02) & (ak_lab.depth <= 3.0)]

# (borehole_id, 표시명) — 깊이 표본이 풍부한 대표 4지점
PROFILE_SITES = [(610, "Fairbanks (64.95°N, 내륙)"),
                 (116, "Sagwon (69.43°N, 툰드라)"),
                 (106, "Franklin Bluff (69.66°N, 툰드라)"),
                 (33, "Barrow (71.31°N, 최북 한랭)")]

fig, axp = plt.subplots(1, len(PROFILE_SITES), figsize=(13.6, 4.8),
                        constrained_layout=True, sharey=True)
for ax, (bid, name) in zip(axp, PROFILE_SITES):
    sub = ak_lab[ak_lab.borehole_id == bid].sort_values("depth")
    la, lo = float(sub.lat.iloc[0]), float(sub.lon.iloc[0])
    j = int(np.argmin(np.abs(lats - la)))
    k = int(np.argmin(np.abs(lons - lo)))
    pred = tvol[:, j, k]
    ax.axvline(0.0, color="0.55", lw=0.9, ls=":", zorder=1)   # 0°C 동결/융해 경계
    ax.plot(pred, depths, color="#b2182b", lw=1.8, zorder=3, label="예측 T(z)")
    ax.scatter(sub.t_max.values, sub.depth.values, s=22, color="#2166ac",
               edgecolors="white", linewidths=0.4, zorder=4, label="실측 $t_{max}$")
    ax.set_ylim(3.0, 0.0)
    ax.set_xlim(-8, 12)
    ax.set_xlabel("연최대 지중온도 (°C)")
    ax.set_title(name, fontsize=9.8)
    ax.grid(alpha=0.25, lw=0.5)
axp[0].set_ylabel("깊이 (m)")
axp[0].legend(loc="lower left", fontsize=8.0, framealpha=0.9)
fig.suptitle("대표 지점 깊이별 T(z) 프로파일  ·  예측(전량학습) vs 실측 $t_{max}$  ·  0°C=동결/융해 경계\n"
             "남→북 열구배(내륙 온난 → 최북 한랭) · 시각화용 산출(검증 지표는 site-block $R^2$ 0.47)",
             fontsize=12.0, fontweight="bold")
save300(fig, "s10_profiles_trumpet")

print("[done] 지적6 대체 그림 4종(깊이슬라이스·융해깊이·fence·프로파일) 추가 저장")
