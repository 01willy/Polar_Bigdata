"""E1 시각화: 정통 공간보간(OK·RK·IDW·MEAN) vs GBM·Stefan 물리앵커 비교.

`e1_kriging_baseline.py` 산출을 3종 그림으로. 컬러맵 규약(ALT=oslo_r)·mask_ocean·축단위 준수.
  (i)   방법별 RMSE 막대: in-domain vs LORO 게이트, y축 0시작, 블록부트스트랩 CI.
  (ii)  OK variogram: 경험 semivariance + 적합 spherical(range·sill·nugget 주석).
  (iii) 대표 지역(알래스카) 예측 지도 비교: GBM vs RK vs STEF(실제 배경·mask_ocean·oslo_r).

실행: python scripts/4_visualization/e1_kriging_figs.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from polar import config as C
from polar.plotstyle import use_polar, CMAP
from polar.geomap import make_ax, hexbin_map, mask_ocean, add_scalebar, ALASKA, _proj

use_polar()
PROC = C.PROCESSED
OUT = C.FIGURES / "e1_kriging"
OUT.mkdir(parents=True, exist_ok=True)

# 방법 표기(한글). 물리·ML(우리)과 정통보간 시각 구분용 색.
METHODS = ["MEAN", "IDW", "OK", "RK", "GBM", "STEF"]
LABELS = {"MEAN": "지역평균", "IDW": "IDW(거리역가중)", "OK": "OK(정규크리깅)",
          "RK": "RK(회귀크리깅)", "GBM": "GBM(공변량)", "STEF": "Stefan 물리앵커"}
# 정통 순수보간=회색계열, 공변량ML=청, 물리앵커=강조 남색
COLORS = {"MEAN": "#b0b0b0", "IDW": "#8f8f8f", "OK": "#6f7f8f",
          "RK": "#4a90b8", "GBM": "#2f6f9f", "STEF": "#1f4e79"}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {OUT.name}/{name}.png+pdf")


comp = pd.read_csv(PROC / "e1_kriging_comparison_table.csv").set_index("method")
boot = pd.read_csv(PROC / "e1_kriging_bootstrap.csv").set_index("method")
vario = pd.read_csv(PROC / "e1_kriging_variograms.csv")
oof = pd.read_csv(PROC / "e1_kriging_oof.csv")


# ============================================================
# (i) 방법별 RMSE 막대: in-domain vs LORO 게이트 (y축 0시작·부트스트랩 CI)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0))
order = ["MEAN", "IDW", "OK", "RK", "GBM", "STEF"]
xpos = np.arange(len(order))
cols = [COLORS[m] for m in order]

# 좌: in-domain. 막대=대표 RMSE(비교표 rmse; STOCH는 seed평균), 오차막대=블록부트스트랩 CI.
#   과거 앙상블 RMSE(seed bias 상쇄로 부당히 낮음)를 헤드라인에 쓰지 않도록 seed평균으로 통일.
ax = axes[0]
vals = [float(comp.loc[m, "rmse"]) for m in order]
lo = [float(max(0.0, comp.loc[m, "rmse"] - boot.loc[m, "ci_lo"])) for m in order]
hi = [float(max(0.0, boot.loc[m, "ci_hi"] - comp.loc[m, "rmse"])) for m in order]
ax.bar(xpos, vals, color=cols, edgecolor="#333", linewidth=0.6, width=0.68, zorder=2)
ax.errorbar(xpos, vals, yerr=[lo, hi], fmt="none", ecolor="#333", elinewidth=1.1,
            capsize=3.5, zorder=3)
for x, v in zip(xpos, vals):
    ax.text(x, v + max(hi) * 0.12, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
ax.set_title("알래스카 in-domain (공간블록 6-fold)")
ax.set_ylabel("RMSE (cm)")
ax.set_ylim(0, max([boot.loc[m, "ci_hi"] for m in order]) * 1.12)
ax.set_xticks(xpos); ax.set_xticklabels([LABELS[m] for m in order], rotation=20, ha="right", fontsize=8.5)
ax.grid(axis="y", alpha=0.4)

# 우: LORO 게이트 (비가중평균)
ax = axes[1]
gate = [float(comp.loc[m, "LORO_gate_rmse"]) for m in order]
ax.bar(xpos, gate, color=cols, edgecolor="#333", linewidth=0.6, width=0.68, zorder=2)
for x, v in zip(xpos, gate):
    ax.text(x, v + max(gate) * 0.015, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
ax.set_title("LORO 전이 게이트 (비가중평균: 알래스카·레나·캐나다)")
ax.set_ylabel("RMSE (cm)")
ax.set_ylim(0, max(gate) * 1.15)
ax.set_xticks(xpos); ax.set_xticklabels([LABELS[m] for m in order], rotation=20, ha="right", fontsize=8.5)
ax.grid(axis="y", alpha=0.4)
ax.axhline(gate[order.index("STEF")], color=COLORS["STEF"], ls="--", lw=1.0, alpha=0.7, zorder=1)

fig.suptitle("E1 인터폴레이션 vs 물리+ML: in-domain은 공간보간 경쟁, 전이는 물리앵커만 방어",
             fontsize=12.5, y=1.02)
save(fig, "e1_rmse_bars")


# ============================================================
# (ii) OK variogram (경험 semivariance + 적합 spherical). 대표: 알래스카 in-domain fold.
# ============================================================
def spherical(h, psill, rng, nugget):
    h = np.asarray(h, float)
    g = np.where(h < rng, psill * (1.5 * h / rng - 0.5 * (h / rng) ** 3), psill)
    return nugget + g


# in-domain OK variogram 파라미터(3 fold 중앙값 사용) + 경험 semivariance 재계산(알래스카)
ok_ind = vario[(vario.cv == "spatial_block_AK") & (vario.method == "OK")]
p_med = ok_ind[["psill", "range_km", "nugget", "sill"]].median()
psill_m, rng_m, nug_m = float(p_med.psill), float(p_med.range_km), float(p_med.nugget)

# 경험 variogram: 알래스카 OOF 관측을 서브샘플로 skgstat
ak_oof = oof[oof.region.isin(["ABoVE_AK", "United States (Alaska)"])].dropna(subset=["alt_cm"])
rng_seed = np.random.default_rng(20260727)
sub = rng_seed.choice(len(ak_oof), min(1200, len(ak_oof)), replace=False)
lat0 = float(ak_oof.lat.mean())
xk = ak_oof.lon.values[sub] * np.cos(np.deg2rad(lat0)) * 111.0
yk = ak_oof.lat.values[sub] * 111.0
zk = ak_oof.alt_cm.values[sub]

fig, ax = plt.subplots(figsize=(7.2, 5.0))
try:
    import skgstat as skg
    # maxlag=800km: sill 포화 근방까지만(원거리 표본 잡음이 적합을 끌지 않게). nugget 추정 허용.
    V = skg.Variogram(np.column_stack([xk, yk]), zk, model="spherical",
                      n_lags=12, maxlag=800, normalize=False, use_nugget=True)
    bins = V.bins
    exp = V.experimental
    ax.plot(bins, exp, "o", color="#2f6f9f", ms=6, label="경험 semivariance (관측)", zorder=3)
except Exception as e:
    print(f"[vario] skgstat 실패({str(e)[:60]}), 경험 semivariance 생략")

# 파이프라인 실사용(pykrige) 적합선을 기본 적합으로 표기(6 in-domain fold 중앙값).
hh = np.linspace(0, 800, 200)
ax.plot(hh, spherical(hh, psill_m, rng_m, nug_m), "-", color="#1f4e79", lw=2.2,
        label=(f"적합 spherical (파이프라인·pykrige)\n"
               f"range={rng_m:.0f}km · sill={psill_m + nug_m:.0f} · nugget={nug_m:.0f} cm²"), zorder=2)
ax.axhline(psill_m + nug_m, color="0.55", ls=":", lw=1.0, zorder=1)
ax.axvline(rng_m, color="0.55", ls=":", lw=1.0, zorder=1)
ax.set_xlabel("거리 h (km)")
ax.set_ylabel("semivariance γ(h) (cm²)")
ax.set_title("OK variogram (알래스카 in-domain): 공간 자기상관 구조")
ax.set_xlim(0, 800)
ax.legend(fontsize=8.5, loc="lower right")
ax.grid(alpha=0.4)
ax.text(0.02, 0.97, "range 밖에서 sill 포화 = 공간상관 소멸 거리\nnugget = 미세규모 변동·측정잡음",
        transform=ax.transAxes, fontsize=8, va="top", color="0.3",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.85))
save(fig, "e1_ok_variogram")


# ============================================================
# (iii) 대표 지역(알래스카) 예측 지도 비교: GBM vs RK vs STEF (mask_ocean·oslo_r)
# ============================================================
proj = _proj(ALASKA)
panels = [("관측 ALT", "alt_cm"), ("GBM(공변량)", "pred_GBM"),
          ("RK(회귀크리깅)", "pred_RK"), ("Stefan 물리앵커", "pred_STEF")]
vmin, vmax = 20, 110
fig = plt.figure(figsize=(14.5, 4.2))
sc = None
for i, (title, col) in enumerate(panels):
    ax = fig.add_subplot(1, 4, i + 1, projection=proj)
    make_ax(ALASKA, ax=ax, fig=fig, title=title)
    d = ak_oof.dropna(subset=[col])
    sc = hexbin_map(ax, d.lon.values, d.lat.values, d[col].values, gridsize=42,
                    cmap=CMAP.alt, vmin=vmin, vmax=vmax, mincnt=1)
    mask_ocean(ax)
    add_scalebar(ax, loc="lower right")
cb = fig.colorbar(sc, ax=fig.axes, fraction=0.011, pad=0.02)
cb.set_label("ALT (cm)", fontsize=9)
fig.suptitle("E1 알래스카 in-domain 예측 비교: GBM·RK·Stefan (셀 중앙값 hexbin, 실제 배경)",
             fontsize=12.5, y=1.03)
save(fig, "e1_pred_maps_alaska")

# ============================================================
# (iii-보조) LORO 전이 붕괴: 방법별 지역 RMSE (kriging의 Lena 붕괴 가시화)
# ============================================================
res = pd.read_csv(PROC / "e1_kriging_results.csv")
loro = res[res.cv == "LORO"].groupby(["method", "region"])["rmse_cm"].mean().unstack()
regions = ["Alaska", "Lena", "Canada"]
loro = loro.reindex(index=order, columns=regions)
# 지역 3종: 냉색 계열(붉은/주황 금지)로 통일 + 해칭 병행(색맹안전, 색으로만 구분 금지).
#   Alaska=짙은 남색, Lena=중청, Canada=옅은 청록.
REG_COLOR = {"Alaska": "#1f4e79", "Lena": "#4a90b8", "Canada": "#7fbfc7"}
REG_HATCH = {"Alaska": "", "Lena": "//", "Canada": ".."}
fig, ax = plt.subplots(figsize=(9.0, 5.0))
w = 0.26
for j, reg in enumerate(regions):
    ax.bar(np.arange(len(order)) + (j - 1) * w, loro[reg].values, width=w,
           label=reg, color=REG_COLOR[reg], hatch=REG_HATCH[reg],
           edgecolor="#222", linewidth=0.6, zorder=2)
ax.set_xticks(np.arange(len(order)))
ax.set_xticklabels([LABELS[m] for m in order], rotation=20, ha="right", fontsize=8.5)
ax.set_ylabel("RMSE (cm)")
ax.set_ylim(0, float(np.nanmax(loro.values)) * 1.12)
# Stefan 전역 안정 기준선(전 지역 최대 RMSE 대비 낮음)을 점선으로 보조 표기.
stef_worst = float(np.nanmax(loro.loc["STEF"].values))
ax.axhline(stef_worst, color=REG_COLOR["Alaska"], ls="--", lw=1.0, alpha=0.7, zorder=1)
ax.text(0.015, stef_worst / (float(np.nanmax(loro.values)) * 1.12) + 0.015,
        f"Stefan 최악 지역 {stef_worst:.0f} cm", transform=ax.transAxes,
        fontsize=8, color=REG_COLOR["Alaska"], va="bottom")
ax.set_title("LORO 지역별 전이 RMSE: 순수 보간(IDW·OK)은 지역 편차 큼, Stefan만 전역 안정")
ax.legend(title="전이 대상 지역", fontsize=9)
ax.grid(axis="y", alpha=0.4)
save(fig, "e1_loro_by_region")

print("[done] E1 그림 4종 저장:", OUT)
