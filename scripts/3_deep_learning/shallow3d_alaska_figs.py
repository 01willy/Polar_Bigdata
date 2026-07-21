"""트랙 3 시각화 — 얕은 3D 알래스카 지중온도장·검증·0°C→ALT 정합.

그림
1. outputs/maps/shallow3d_alaska_tmax_depths.{png,pdf}
   깊이 0.5·1·2·3m 알래스카 연최대 지중온도(t_max) 지도(vik, 0°C 중심 tnorm).
2. outputs/figures/14_shallow3d/shallow3d_band_rmse.{png,pdf}
   깊이밴드별 RMSE·skill 막대(단조제약 vs 무제약, 필드 검증).
3. outputs/figures/14_shallow3d/shallow3d_alt_scatter.{png,pdf}
   얕은장 유도 ALT vs 실측 ALT 산점(1:1), 심부판 r 참조.
보조. outputs/figures/14_shallow3d/shallow3d_loro_site.{png,pdf}
   LORO 사이트별 RMSE·skill(전이 검증 정직 보고).

실행: /home/anaconda3/bin/python scripts/3_deep_learning/shallow3d_alaska_figs.py (ROOT)
"""
import sys, os, json
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from polar.plotstyle import use_polar, CMAP, tnorm, add_cbar, style_geo, BAD, lon_formatter, lat_formatter
from polar.outputs import figpath, mappath
plt = use_polar()

PROC = "data/processed"
grid = pd.read_csv(os.path.join(PROC, "shallow3d_alaska_grid.csv"))
res = pd.read_csv(os.path.join(PROC, "shallow3d_alaska_results.csv"))
altm = pd.read_csv(os.path.join(PROC, "shallow3d_alaska_altmatch.csv"))
loro = pd.read_csv(os.path.join(PROC, "shallow3d_alaska_loro.csv"))
meta = json.load(open(os.path.join(PROC, "shallow3d_alaska_meta.json")))


def save(fig, path_png):
    fig.savefig(path_png, dpi=260, bbox_inches="tight")
    fig.savefig(path_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------- 그림 1: 깊이별 t_max 지도 ----------
depths = [0.5, 1.0, 2.0, 3.0]
tv = grid.tmax_pred.values
vmax = float(np.nanpercentile(np.abs(tv), 99))
vmax = max(vmax, 1.0)
norm = tnorm(-vmax, vmax, 0.0)
fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.3), constrained_layout=True)
for ax, dm in zip(axes, depths):
    g = grid[np.isclose(grid.depth_m, dm)]
    sc = ax.scatter(g.lon, g.lat, c=g.tmax_pred, s=3.2, cmap=CMAP.temp, norm=norm,
                    linewidths=0, rasterized=True)
    ax.set_title(f"깊이 {dm:g} m", fontsize=12)
    style_geo(ax, xlabel="경도", ylabel=("위도" if dm == depths[0] else ""))
    ax.xaxis.set_major_formatter(lon_formatter())
    ax.yaxis.set_major_formatter(lat_formatter())
    ax.set_aspect(1.0 / np.cos(np.radians(66)))
cb = fig.colorbar(sc, ax=axes, shrink=0.82, pad=0.012, aspect=32)
cb.set_label("연최대 지중온도 t_max (°C)  ·  청=동결 유지, 적=여름 융해", fontsize=10)
cb.ax.tick_params(labelsize=9)
fig.suptitle("알래스카 얕은 지중 연최대온도장 (GBM 조건장, 0-3 m)", fontsize=13.5, fontweight="bold")
save(fig, mappath("shallow3d_alaska_tmax_depths"))
print("[fig1] shallow3d_alaska_tmax_depths 저장")


# ---------- 그림 2: 깊이밴드 RMSE·skill 막대 ----------
band_order = ["0.02-0.5m", "0.5-1m", "1-2m", "2-3m", "all"]
sb = res[res.cv == "site_block6"].copy()
mono = sb[sb.model == "gbm_mono"].set_index("band").reindex(band_order)
free = sb[sb.model == "gbm_free"].set_index("band").reindex(band_order)
x = np.arange(len(band_order))
w = 0.38
fig, (axr, axs) = plt.subplots(1, 2, figsize=(12.2, 4.5), constrained_layout=True)
axr.bar(x - w / 2, mono.rmse_cm, w, label="단조 제약", color="#2166ac")
axr.bar(x + w / 2, free.rmse_cm, w, label="무제약", color="#88a8c8")
for xi, v in zip(x - w / 2, mono.rmse_cm):
    axr.text(xi, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
axr.set_xticks(x); axr.set_xticklabels(band_order, rotation=12)
axr.set_ylabel("RMSE (°C)")
axr.set_title("깊이밴드별 필드 RMSE (사이트블록 6-fold)")
axr.legend(loc="upper right")
axr.set_axisbelow(True)

axs.bar(x - w / 2, mono.skill_over_mean, w, label="단조 제약", color="#1b7837")
axs.bar(x + w / 2, free.skill_over_mean, w, label="무제약", color="#a6dba0")
axs.axhline(0, color="#444", lw=0.8)
axs.set_xticks(x); axs.set_xticklabels(band_order, rotation=12)
axs.set_ylabel("skill-over-mean (1 - RMSE/SD)")
axs.set_title("깊이밴드별 skill  ·  0 = 평균예측기 수준")
axs.legend(loc="upper right")
axs.set_axisbelow(True)
fig.suptitle("얕은 지중온도장 검증  ·  단조 제약이 깊은 밴드 skill을 보전", fontsize=12.5, fontweight="bold")
save(fig, figpath("14_shallow3d", "shallow3d_band_rmse"))
print("[fig2] shallow3d_band_rmse 저장")


# ---------- 그림 3: 유도 ALT vs 실측 ALT 산점 ----------
both = altm.dropna(subset=["alt_pred_cm"])
am = res[res.cv == "alt_match"].iloc[0]
r_alt = float(np.corrcoef(both.alt_obs_cm, both.alt_pred_cm)[0, 1]) if len(both) >= 3 else np.nan
ref = meta["phase1_deep_reference"]
fig, ax = plt.subplots(figsize=(6.2, 6.0), constrained_layout=True)
lim = [0, max(both.alt_obs_cm.max(), both.alt_pred_cm.max()) * 1.08]
ax.plot(lim, lim, "--", color="#444", lw=1.0, label="1:1")
ax.scatter(both.alt_obs_cm, both.alt_pred_cm, s=56, c="#2166ac",
           edgecolors="white", linewidths=0.6, zorder=3)
for _, r in both.iterrows():
    ax.annotate(str(r.site)[:10], (r.alt_obs_cm, r.alt_pred_cm),
                fontsize=6.5, color="#555", xytext=(3, 3), textcoords="offset points")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("실측 시추공 envelope ALT (cm)")
ax.set_ylabel("얕은장 유도 ALT (cm)")
txt = (f"얕은판(0-3 m): r = {r_alt:.2f},  RMSE = {am.rmse_cm:.0f} cm,  n = {int(am.n)}\n"
       f"심부판(참조):  r = {ref['r']:.2f},  RMSE = {ref['rmse_cm']:.0f} cm")
ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))
ax.set_title("0°C 교차 유도 ALT vs 실측 ALT", fontsize=12.5, fontweight="bold")
ax.legend(loc="lower right")
save(fig, figpath("14_shallow3d", "shallow3d_alt_scatter"))
print("[fig3] shallow3d_alt_scatter 저장")


# ---------- 보조 그림: LORO 사이트별 ----------
lr = loro[loro.n >= 10].sort_values("rmse_cm")
fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
y = np.arange(len(lr))
colors = ["#1b7837" if s > 0 else "#b2182b" for s in lr.skill_over_mean]
ax.barh(y, lr.rmse_cm, color=colors, alpha=0.9)
for yi, (rmse, sk) in enumerate(zip(lr.rmse_cm, lr.skill_over_mean)):
    ax.text(rmse + 0.03, yi, f"skill={sk:+.2f}", va="center", fontsize=8, color="#333")
ax.set_yticks(y); ax.set_yticklabels(lr.site)
ax.set_xlabel("LORO RMSE (°C)  ·  사이트 제외 학습")
ax.set_title("사이트별 전이(LORO) 필드 성능  ·  녹색=평균예측 우세, 적색=열세",
             fontsize=12, fontweight="bold")
ax.set_axisbelow(True)
save(fig, figpath("14_shallow3d", "shallow3d_loro_site"))
print("[fig4] shallow3d_loro_site 저장")
print("[done] 4개 그림(PNG+PDF) 저장 완료")
