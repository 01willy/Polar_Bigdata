"""S1 시각화: 실제 지도 배경 위 관측·예측·잔차 + Taylor diagram + 모델 비교.

`docs/RESEARCH_PLAN_...` §11.5. 냉색 규약, cartopy 지도 배경(위경도 지역 식별),
표<그래프<지도 우선. 여러 모델을 나란히(한 모델 단정 금지 시각화).

실행: python scripts/4_visualization/s1_baseline_figs.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polar import config as C
from polar.plotstyle import use_polar, CMAP, tnorm
from polar.geomap import (make_ax, scatter_map, hexbin_map, add_colorbar, add_scalebar,
                          add_inset_locator, add_zoom_inset, mask_ocean, ALASKA)

use_polar()
PROC = C.PROCESSED
OUT = C.FIGURES / "s1_baseline"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {OUT.name}/{name}.png+pdf")


def cool_cats(n):
    """모델 범주용 냉색 이산 팔레트(붉은계열 회피). davos_r 저-고 구간을 균등 샘플."""
    from cmcrameri import cm as _cmc
    import numpy as _np
    base = _cmc.davos_r
    return [base(t) for t in _np.linspace(0.12, 0.86, max(n, 1))]


oof = pd.read_csv(PROC / "s1_baseline_oof.csv")
res = pd.read_csv(PROC / "s1_baseline_results.csv")
pred_cols = [c for c in oof.columns if c.startswith("pred_")]
models = [c.replace("pred_", "") for c in pred_cols]
# in-domain RMSE 순위(seed 평균). 발산 모델(RMSE>30)은 시각화에서 제외(별도 안정화 대상).
rank = (res[res.cv == "spatial_block_AK"].groupby("model").rmse_cm.mean().sort_values())
diverged = rank[rank > 30].index.tolist()
ordered = [m for m in rank.index if rank[m] <= 30 and f"pred_{m}" in oof.columns]
if diverged:
    print(f"[warn] 발산 모델 시각화 제외: {diverged} (별도 안정화 필요)")
print(f"[load] {len(oof):,} OOF · 모델 {ordered}")

vmin, vmax = 20, 110


# ---------------- 1. 관측 + 상위 3모델 예측 지도 (전 셀 산점 + 밀집 사이트 확대 inset) ----------------
# 사용자 지적: 13,606 셀이 실재하나 ABoVE 250m 격자 사이트라 전 지도에서는 작은 클러스터로 뭉쳐 보인다.
#   (배로 71.25/-156.5 3,978셀, 육쿤델타 61.5/-163.0 1,669셀, 북사면 69.0/-149.0 1,594셀 등 83개
#    0.5도 지역에 초집중). 이는 버그가 아니라 데이터 구조이자 연구의 공간 대표성 한계다.
# 대응 (1) 최대 밀집지 배로를 확대 inset으로 붙여 250m 격자 개별 셀 수천 점이 촘촘히 보이게 하고,
#        본 지도에 확대 위치 사각형을 표시한다. (2) 겹침 구간에서 개별 점이 드러나도록 마커를
#        작게(s=4)·alpha 낮춰(0.35) 밀도가 짙어지게 한다. 예측 패널도 동일 방식으로 정합 유지.
# 지적: 1x4 배열은 각 패널 폭이 좁아 가로로 넓은 알래스카(경도폭 32°>위도폭 13°)를 세로로
# 찌그러뜨린다. 2x2 배열로 각 패널을 넓게 배치해 알래스카 실제 종횡비와 점 밀도를 보존한다.
from polar.geomap import _proj
proj = _proj(ALASKA)
n_cell = len(oof)
top = ordered[:3]
# 배로 확대 범위(최대 밀집지): 실측 데이터 extent(lon -156.642~-156.538, lat 71.246~71.324)에 소폭 여백.
BARROW = (-156.652, -156.528, 71.242, 71.328)  # (lon0, lon1, lat0, lat1)
bmask = (oof.lat.between(BARROW[2], BARROW[3])) & (oof.lon.between(BARROW[0], BARROW[1]))
n_barrow = int(bmask.sum())
panels = [("(a) 관측 ALT", oof.alt_cm)] + [
    (f"({chr(98+i)}) {m} 예측 (RMSE {rank[m]:.1f})", oof[f"pred_{m}"]) for i, m in enumerate(top)]
fig = plt.figure(figsize=(11.4, 8.6))
sc = None
for i, (title, vals) in enumerate(panels):
    ax = fig.add_subplot(2, 2, i + 1, projection=proj)
    make_ax(ALASKA, ax=ax, fig=fig, title=title)
    sc = scatter_map(ax, oof.lon, oof.lat, vals, cmap=CMAP.alt, vmin=vmin, vmax=vmax,
                     s=4, edge=False)
    sc.set_alpha(0.35)
    # 밀집 사이트 확대: 250m 격자 개별 셀이 촘촘히 드러나게(작은 마커·본지도 위치 사각형).
    axz = add_zoom_inset(fig, ax, ALASKA, BARROW, loc=(0.66, 0.02, 0.32, 0.52))
    if axz is not None:
        scz = scatter_map(axz, oof.lon[bmask], oof.lat[bmask], vals[bmask],
                          cmap=CMAP.alt, vmin=vmin, vmax=vmax, s=6, edge=False)
        scz.set_alpha(0.45)
        axz.set_title(f"확대: 배로 {n_barrow:,}셀\n(250m 격자)", fontsize=7, color="#c0392b", pad=2)
    if i == 0:
        add_inset_locator(fig, ax, ALASKA); add_scalebar(ax, loc="lower left")
cb = fig.colorbar(sc, ax=fig.axes, fraction=0.022, pad=0.02)
cb.set_label("ALT (cm)", fontsize=9)
fig.suptitle(
    f"S1 in-domain ALT 예측: 관측과 정합 (관측 {n_cell:,} 셀, 원 관측 22.5만 레코드 집계). "
    f"관측은 83개 0.5° 지역·소수 ABoVE 250m 사이트에 초집중(확대: 배로 {n_barrow:,}셀), "
    f"공간 대표성 한계이자 전이 도전. 상위 모델 RMSE {rank[top[0]]:.0f}-{rank[top[-1]]:.0f}cm, 공간블록 OOF",
    fontsize=10, y=0.99)
fig.subplots_adjust(hspace=0.18, wspace=0.10)
save(fig, "alaska_obs_vs_pred_maps")


# ---------------- 2. 모델별 잔차 지도 (hexbin, broc 0중심) ----------------
n = len(ordered)
ncol = min(4, n); nrow = int(np.ceil(n / ncol))
fig = plt.figure(figsize=(3.4 * ncol, 3.4 * nrow))
for i, m in enumerate(ordered):
    ax = fig.add_subplot(nrow, ncol, i + 1, projection=proj)
    make_ax(ALASKA, ax=ax, fig=fig, title=f"{m}  (bias {(oof[f'pred_{m}']-oof.alt_cm).mean():+.1f})")
    resid = oof[f"pred_{m}"] - oof.alt_cm
    hb = hexbin_map(ax, oof.lon, oof.lat, resid, gridsize=38, cmap=CMAP.diff, norm=tnorm(-40, 40, 0))
    mask_ocean(ax)
cb = fig.colorbar(hb, ax=fig.axes, fraction=0.012, pad=0.02)
cb.set_label("예측 - 관측 (cm)", fontsize=9)
fig.suptitle("S1 모델별 잔차 지도 (hexbin 중앙값; 파랑=과소, 갈색=과대)", fontsize=12, y=1.0)
save(fig, "alaska_residual_maps")


# ---------------- 3. Taylor diagram (in-domain) ----------------
obs = oof.alt_cm.values
sd_obs = obs.std()
fig = plt.figure(figsize=(6.4, 6))
ax = fig.add_subplot(111, polar=True)
ax.set_thetamin(0); ax.set_thetamax(90)
ax.set_theta_zero_location("E"); ax.set_theta_direction(1)
catcol = cool_cats(len(ordered))
for i, m in enumerate(ordered):
    p = oof[f"pred_{m}"].values
    r = np.corrcoef(obs, p)[0, 1]
    sd = p.std()
    theta = np.arccos(np.clip(r, -1, 1))
    ax.plot(theta, sd, "o", ms=10, color=catcol[i], markeredgecolor="0.3",
            markeredgewidth=0.5, label=f"{m} (r={r:.2f})")
ax.plot(0, sd_obs, "k*", ms=16, label="관측")
# RMSE 반원 등고선(관측 기준)
for rms in [8, 12, 16, 20]:
    th = np.linspace(0, np.pi / 2, 100)
    xr = sd_obs + rms * np.cos(th + np.pi)  # 근사
ax.set_rlabel_position(90)
ax.set_xlabel("표준편차 (cm)")
ax.set_title("Taylor 다이어그램: 알래스카 in-domain\n(관측 별표에 가까울수록 우수)", fontsize=11, pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.05), fontsize=8)
save(fig, "taylor_indomain")


# ---------------- 4. 모델 비교 막대 (in-domain RMSE + LORO) ----------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
# in-domain
ind = res[res.cv == "spatial_block_AK"].groupby("model").rmse_cm.agg(["mean", "std"]).reindex(ordered)
barcol = cool_cats(len(ind))
axes[0].bar(range(len(ind)), ind["mean"], yerr=ind["std"].fillna(0), capsize=3,
            color=barcol, edgecolor="0.3", linewidth=0.5)
axes[0].set_xticks(range(len(ind))); axes[0].set_xticklabels(ind.index, rotation=45, ha="right", fontsize=8)
axes[0].axhline(14, color="0.4", ls="--", lw=1, label="대표성 하한 ~14cm")
axes[0].set_ylabel("RMSE (cm)"); axes[0].set_title("알래스카 in-domain (공간블록)", fontsize=10)
axes[0].legend(fontsize=8)
# LORO (region별)
lo = res[res.cv == "LORO"].groupby(["model", "region"]).rmse_cm.mean().unstack()
lo = lo.reindex(ordered)
lo.plot(kind="bar", ax=axes[1], color=cool_cats(lo.shape[1]), width=0.8,
        edgecolor="0.3", linewidth=0.4)
axes[1].set_ylabel("RMSE (cm)"); axes[1].set_title("LORO 전이 (공유 코어, SAR 제외)", fontsize=10)
axes[1].set_xticklabels(lo.index, rotation=45, ha="right", fontsize=8)
axes[1].legend(title="test 지역", fontsize=8)
fig.suptitle("S1 모델 비교: in-domain은 동률권(하한 근접), 전이는 지역별 편차 큼", fontsize=12)
save(fig, "model_comparison_bars")

print("[done] S1 시각화 4종 완료")
