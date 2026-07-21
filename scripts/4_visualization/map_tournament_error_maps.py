"""모델 토너먼트 ALT 예측·오차 지도 (위치 동등가중).

재료: data/processed/model_tournament_predictions.csv (CV fold 예측, 점 단위)
집계: (lat, lon) 0.0001도 반올림 후 groupby mean → 위치 동등가중(셀 프로토콜과 일치,
      점 단위 pseudo-replication 방지). N=14,348 위치.

그림 A tournament_pred_maps : 2x4 = 관측 ALT + 7모델 예측 (CMAP.alt, 20-90 cm 공통,
      컬러바 양끝 삼각형 + 절단 비율 캡션)
그림 B tournament_error_maps: 2x4 = 7모델 오차(예측-관측, 0중심 ±40 cm, 절단 비율 캡션)
      + RMSE·skill 막대. 패널 라벨 (a)-(h).

출력: outputs/maps/tournament_pred_maps.png/.pdf, tournament_error_maps.png/.pdf
"""
import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from polar.plotstyle import use_polar, CMAP, BAD, tnorm, lon_formatter, lat_formatter
from polar.outputs import mappath

plt = use_polar()
plt.rcParams["pdf.fonttype"] = 42   # PDF 텍스트를 TrueType로 임베딩(검색·편집 가능, 세 그림 공통)

# ---------------------------------------------------------------- 집계
df = pd.read_csv("data/processed/model_tournament_predictions.csv")
df["lat"] = df["lat"].round(4)
df["lon"] = df["lon"].round(4)
g = df.groupby(["lat", "lon"], as_index=False).mean(numeric_only=True)
n_loc = len(g)

MODELS = [
    ("pred_GBM", "GBM"),
    ("pred_MLP", "MLP"),
    ("pred_FT-Transformer", "FT-Transformer"),
    ("pred_TabM", "TabM"),
    ("pred_Flow matching", "Flow matching"),
    ("pred_Diffusion", "Diffusion"),
    ("pred_앙상블(GBM+FT-T)", "앙상블(GBM+FT-T)"),
]
obs = g["alt_cm"].values
sd_obs = obs.std(ddof=0)
metrics = {}
for col, name in MODELS:
    rmse = float(np.sqrt(np.mean((g[col].values - obs) ** 2)))
    metrics[name] = {"rmse": rmse, "skill": 1.0 - rmse / sd_obs}
    print(f"{name:22s} RMSE {rmse:6.2f} cm  skill {metrics[name]['skill']:+.3f}")
print(f"locations={n_loc:,}  sd(obs)={sd_obs:.2f} cm")

# ---------------------------------------------------------------- 배경(육지)
PAD = 1.5
x0, x1 = g.lon.min() - PAD, g.lon.max() + PAD
y0, y1 = g.lat.min() - PAD, g.lat.max() + PAD

ds = xr.open_dataset("data/raw/era5land/nh_monthly_2015-2020.nc")
sub = ds["t2m"].isel(valid_time=0).sel(latitude=slice(y1, y0), longitude=slice(x0, x1)).load()
land_img = np.where(np.isfinite(sub.values), 1.0, np.nan)
land_ext = (float(sub.longitude.min()), float(sub.longitude.max()),
            float(sub.latitude.min()), float(sub.latitude.max()))
ds.close()
# 규약(plotstyle.BAD): 해양/결측=옅은 회색, 육지=백색
cm_land = ListedColormap(["#ffffff"])
cm_land.set_bad(BAD)

ASPECT = 1.0 / np.cos(np.deg2rad(g.lat.mean()))   # 물리 종횡비(위도 보정)


from matplotlib.ticker import MultipleLocator


def draw_base(ax):
    ax.imshow(land_img, cmap=cm_land, origin="upper", extent=land_ext,
              aspect="auto", interpolation="nearest", zorder=0)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect(ASPECT, adjustable="box")
    ax.xaxis.set_major_formatter(lon_formatter())
    ax.yaxis.set_major_formatter(lat_formatter())
    ax.xaxis.set_major_locator(MultipleLocator(15))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.grid(alpha=0.3, lw=0.4, color="#bbbbbb")
    ax.tick_params(length=3)


def finish_axes(fig, axes):
    for i, ax in enumerate(axes.ravel()):
        if i % 4 == 0:
            ax.set_ylabel("위도")
        else:
            ax.set_yticklabels([])
        if i // 4 == 1:
            ax.set_xlabel("경도")
        else:
            ax.set_xticklabels([])


def side_cbar(fig, sm, label, extend="neither"):
    """공통 세로 컬러바. extend='both'는 색 범위 밖 값(절단)을 양끝 삼각형으로 표시."""
    fig.subplots_adjust(left=0.05, right=0.905, top=0.88, bottom=0.115,
                        wspace=0.08, hspace=0.16)
    cax = fig.add_axes([0.925, 0.18, 0.012, 0.60])
    cb = fig.colorbar(sm, cax=cax, extend=extend)
    cb.set_label(label, fontsize=13)
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(labelsize=12, length=2.5)
    return cb


def panel_label(ax, s):
    """패널 알파벳 라벨 (a)-(h): 축 좌상단 바깥(캡션 참조용)."""
    ax.text(0.0, 1.02, s, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=13, fontweight="bold", color="#1b2a41")


# ============================ 그림 A: 예측 지도 ============================
# 색 범위 20-90 cm 근거: 관측 분포 퍼센타일(하한=p1, 상한=p97 근사). 절단 비율은 캡션에 표기.
VLO, VHI = 20.0, 90.0
q_lo = (obs <= VLO).mean() * 100                    # 20 cm 이하 누적 백분위
q_hi = (obs <= VHI).mean() * 100                    # 90 cm 이하 누적 백분위
obs_hi = (obs > VHI).mean() * 100
obs_lo = (obs < VLO).mean() * 100
pred_hi = float(np.mean([(g[c].values > VHI).mean() * 100 for c, _ in MODELS]))

fig, axes = plt.subplots(2, 4, figsize=(18, 8.6))
panels = [("alt_cm", "관측 ALT")] + MODELS
for i, (ax, (col, name)) in enumerate(zip(axes.ravel(), panels)):
    draw_base(ax)
    ax.scatter(g.lon, g.lat, c=g[col], s=9, cmap=CMAP.alt, vmin=VLO, vmax=VHI,
               lw=0, rasterized=True, zorder=4)
    panel_label(ax, f"({chr(ord('a') + i)})")
    if col == "alt_cm":
        ax.set_title(name, fontsize=14, fontweight="bold")
    else:
        ax.set_title(f"{name} · RMSE {metrics[name]['rmse']:.1f} cm", fontsize=14, fontweight="bold")
finish_axes(fig, axes)
side_cbar(fig, ScalarMappable(norm=Normalize(VLO, VHI), cmap=CMAP.alt), "ALT (cm)",
          extend="both")
fig.suptitle(f"모델별 ALT 예측 지도 (위치 동등가중, {n_loc:,} 위치, 알래스카·서캐나다)",
             fontsize=18, fontweight="bold")
fig.text(0.05, 0.025,
         f"색 범위 {VLO:.0f}-{VHI:.0f} cm는 관측 p{q_lo:.0f}-p{q_hi:.0f} 구간이다. "
         f"범위 밖 절단 비율: 관측 {obs_hi:.1f}% (>{VHI:.0f} cm), {obs_lo:.1f}% (<{VLO:.0f} cm), "
         f"7모델 예측 평균 {pred_hi:.1f}% (>{VHI:.0f} cm). 컬러바 양끝 삼각형은 범위 밖 값을 표시한다.",
         ha="left", va="bottom", fontsize=9, color="#555555")
pngA = mappath("tournament_pred_maps")
pdfA = mappath("tournament_pred_maps", ext="pdf")
fig.savefig(pngA); fig.savefig(pdfA); plt.close(fig)
print(f"[saved] {pngA}")

# ============================ 그림 B: 오차 지도 ============================
ELIM = 40.0
norm_err = tnorm(-ELIM, ELIM)
clip_pct = {name: float((np.abs(g[col].values - obs) > ELIM).mean() * 100)
            for col, name in MODELS}                 # ±40 cm 범위 밖 절단 비율(모델별)
fig, axes = plt.subplots(2, 4, figsize=(18, 8.6))
axs = axes.ravel()
for i, (ax, (col, name)) in enumerate(zip(axs[:7], MODELS)):
    draw_base(ax)
    err = g[col].values - obs
    # 얇은 회색 테두리: 0 근처 오차(발산맵 중앙=백색)가 백색 육지에서 사라지지 않게 한다
    ax.scatter(g.lon, g.lat, c=err, s=9, cmap=CMAP.diff, norm=norm_err,
               edgecolors="#8f99a6", linewidths=0.2, rasterized=True, zorder=4)
    panel_label(ax, f"({chr(ord('a') + i)})")
    ax.set_title(f"{name} · RMSE {metrics[name]['rmse']:.1f} cm", fontsize=14, fontweight="bold")
for i, ax in enumerate(axs[:7]):
    if i % 4 == 0:
        ax.set_ylabel("위도")
    else:
        ax.set_yticklabels([])
    if i // 4 == 1:
        ax.set_xlabel("경도")
    else:
        ax.set_xticklabels([])
side_cbar(fig, ScalarMappable(norm=norm_err, cmap=CMAP.diff), "예측 - 관측 (cm)",
          extend="both")

# 8번째 패널: 위치가중 RMSE·skill 가로 막대
axb = axs[7]
axb.grid(axis="x", alpha=0.4, lw=0.5)
axb.grid(axis="y", visible=False)
SHORT = {"GBM": "GBM", "MLP": "MLP", "FT-Transformer": "FT-T", "TabM": "TabM",
         "Flow matching": "Flow", "Diffusion": "Diffusion",
         "앙상블(GBM+FT-T)": "앙상블"}
order = sorted(metrics, key=lambda k: metrics[k]["rmse"])       # RMSE 오름차순
ypos = np.arange(len(order))[::-1]                              # 최상단=최저 RMSE
vals = [metrics[k]["rmse"] for k in order]
axb.barh(ypos, vals, height=0.62, color=CMAP.count(0.66), edgecolor="#33518e", lw=0.6)
for y, k, v in zip(ypos, order, vals):
    axb.text(v + 0.3, y, f"{v:.1f} cm · skill {metrics[k]['skill']:+.2f}",
             va="center", fontsize=11.5, color="#1b2a41")
axb.set_yticks(ypos)
axb.set_yticklabels([SHORT[k] for k in order], fontsize=12)
axb.set_xlim(0, max(vals) * 1.52)
axb.set_xlabel(f"위치가중 RMSE (cm)\nskill = 1 - RMSE/σ(관측), σ = {sd_obs:.1f} cm",
               fontsize=12)
axb.set_title("모델별 RMSE·skill", fontsize=14, fontweight="bold")
panel_label(axb, "(h)")

worst = max(clip_pct, key=clip_pct.get)
fig.suptitle(f"모델별 ALT 오차 지도 (예측-관측, 위치 동등가중, {n_loc:,} 위치, 알래스카·서캐나다)",
             fontsize=18, fontweight="bold")
fig.text(0.05, 0.025,
         f"오차 색 범위 ±{ELIM:.0f} cm 밖 절단 비율: 7모델 평균 "
         f"{np.mean(list(clip_pct.values())):.1f}%, 최대 {clip_pct[worst]:.1f}% ({SHORT[worst]}). "
         f"컬러바 양끝 삼각형은 범위 밖 값을 표시한다.",
         ha="left", va="bottom", fontsize=9, color="#555555")
pngB = mappath("tournament_error_maps")
pdfB = mappath("tournament_error_maps", ext="pdf")
fig.savefig(pngB); fig.savefig(pdfB); plt.close(fig)
print(f"[saved] {pngB}")
