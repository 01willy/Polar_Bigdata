"""횡단(2) — transfer AOA: 환경 비유사도(DI)가 커질수록 오차↑·커버리지↓ (정직한 외삽 경고).

Meyer(2021) dissimilarity index(DI)를 LORO 전이에 적용:
 held-out region의 각 점이 학습 환경공간에서 얼마나 먼가(DI) → DI 분위수별 RMSE·coverage.
 AOA = DI ≤ 임계(train DI의 Q75+1.5IQR). AOA 밖은 신뢰하지 말라고 지도에 표기.

실행: python3 scripts/2_evaluation/aoa_transfer.py
"""
import sys, os
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from polar.eval_metrics import rmse, coverage
from polar.outputs import figpath, mappath
from polar.plotstyle import use_polar, CMAP, lon_formatter, lat_formatter, despine

plt = use_polar()
# figs-only: decile 곡선은 저장된 CSV를 읽고, 지도용 DI만 재계산(NearestNeighbors, 수 초).
# 무거운 GBM 분위 재적합은 건너뛴다. (--figs-only 로 활성화)
FIGS_ONLY = "--figs-only" in sys.argv
PROC = "data/processed"
CLIP = (np.log1p(1), np.log1p(600))
A = 0.10
BASE14 = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough",
          "e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
base = pd.read_csv(os.path.join(PROC, "dl_dataset.csv"))
X = base[BASE14].values; y = base.alt_cm.values; ylog = np.log1p(y)
mu, sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-9
Xs = (X - mu) / sd
reg = base.region.values
to_cm = lambda p: np.expm1(np.clip(p, *CLIP))
gbm = lambda q=None: (HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=63,
                      l2_regularization=1.0, early_stopping=(q is None), random_state=0)
                      if q is None else HistGradientBoostingRegressor(loss="quantile", quantile=q,
                      max_iter=300, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=1.0, random_state=0))

di = np.full(len(base), np.nan); inside = np.full(len(base), np.nan)
pt = np.full(len(base), np.nan); lo = np.full(len(base), np.nan); hi = np.full(len(base), np.nan)
thr_list = []
for r in pd.unique(reg):
    te = np.where(reg == r)[0]; tr = np.where(reg != r)[0]
    if len(te) < 200:
        continue
    # 참조점 = 학습영역 '고유 위치'(pseudo-replication의 중복 제거 → DI 정규화 왜곡 방지)
    uniq_idx = base.iloc[tr].drop_duplicates(subset="loc_id").index.values
    Xref = (X[uniq_idx] - mu) / sd
    nn = NearestNeighbors(n_neighbors=2).fit(Xref)
    d_ref, _ = nn.kneighbors(Xref); d_bar = d_ref[:, 1].mean()
    di_tr = d_ref[:, 1] / d_bar
    q75, q25 = np.quantile(di_tr, [0.75, 0.25]); thr = q75 + 1.5 * (q75 - q25); thr_list.append(thr)
    d_te, _ = nn.kneighbors(Xs[te], n_neighbors=1)
    di[te] = d_te[:, 0] / d_bar
    inside[te] = (di[te] <= thr).astype(float)
    if FIGS_ONLY:
        continue
    pt[te] = to_cm(gbm().fit(X[tr], ylog[tr]).predict(X[te]))
    lo[te] = to_cm(gbm(A / 2).fit(X[tr], ylog[tr]).predict(X[te]))
    hi[te] = to_cm(gbm(1 - A / 2).fit(X[tr], ylog[tr]).predict(X[te]))

THR = float(np.median(thr_list))
DEC_CSV = os.path.join(PROC, "alt_aoa_transfer_results.csv")
if FIGS_ONLY:
    # 그림용: 지도 DI(재계산)만 있으면 됨. decile 곡선은 저장된 결과를 읽는다.
    m = np.isfinite(di)
    dec = pd.read_csv(DEC_CSV)
    print(f"[figs-only] held-out 점 {m.sum()}, AOA 임계(median)={THR:.2f}, "
          f"AOA 안 비율={100*np.nanmean(inside[m]):.1f}%  (decile: {DEC_CSV} 읽음)")
else:
    m = np.isfinite(di) & np.isfinite(pt)
    print(f"held-out 점 {m.sum()}, AOA 임계(median)={THR:.2f}, AOA 안 비율={100*np.nanmean(inside[m]):.1f}%")

    # DI 분위수(decile)별 RMSE·coverage
    dd = pd.DataFrame({"di": di[m], "y": y[m], "pt": pt[m], "lo": lo[m], "hi": hi[m]})
    dd["dec"] = pd.qcut(dd["di"], 10, labels=False, duplicates="drop")
    rows = []
    for d, g in dd.groupby("dec"):
        rows.append({"di_decile": int(d) + 1, "di_mid": round(g.di.median(), 2), "n": len(g),
                     "rmse_cm": round(rmse(g.y, g.pt), 2),
                     "coverage_pct": round(100 * coverage(g.y, g.lo, g.hi), 1)})
    dec = pd.DataFrame(rows)
    print(dec.to_string(index=False))
    dec.to_csv(DEC_CSV, index=False)

    # 전역 inside/outside
    mi = m & (inside == 1); mo = m & (inside == 0)
    if mo.sum() > 50:
        print(f"AOA 안: RMSE {rmse(y[mi],pt[mi]):.1f} cov {100*coverage(y[mi],lo[mi],hi[mi]):.0f}%  |  "
              f"밖: RMSE {rmse(y[mo],pt[mo]):.1f} cov {100*coverage(y[mo],lo[mo],hi[mo]):.0f}%")

# ---------- Fig: DI-decile gradient + 대표 지역 지도 (제목 없음, 캡션이 대체) ----------
C_RMSE = "#1f4e79"   # RMSE (진청)
C_COV = "#0b7285"    # coverage (청록)
C_REF = "#495057"    # 90% 기준선 (청회색, 강조색 아님)
fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
ax = axes[0]
x = dec.di_decile.values
ax.plot(x, dec.rmse_cm, "o-", color=C_RMSE, lw=2, ms=6, label="RMSE (cm)")
ax.set_xlabel("환경 비유사도 DI 분위수 (1=낮음 → 10=높음)", fontsize=12)
ax.set_ylabel("RMSE (cm)", color=C_RMSE, fontsize=12)
ax.tick_params(axis="y", labelcolor=C_RMSE, labelsize=10.5)
ax.tick_params(axis="x", labelsize=10.5)
ax.set_xticks(x)
ax2 = ax.twinx()
ax2.plot(x, dec.coverage_pct, "s--", color=C_COV, lw=1.8, ms=6, label="coverage (%)")
ax2.axhline(90, color=C_REF, ls=":", lw=1.2, label="목표 커버리지 90%")
ax2.set_ylabel("90% 구간 커버리지 (%)", color=C_COV, fontsize=12); ax2.set_ylim(0, 100)
ax2.tick_params(axis="y", labelcolor=C_COV, labelsize=10.5)
ax.set_title("(a)", loc="left", fontsize=13, fontweight="bold")
despine(ax)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
# 범례: 좌측 중앙(데이터 공백 영역), 1열 — 8cm 축소 게재 시에도 판독 가능, x축 라벨과 비중첩
ax.legend(h1 + h2, l1 + l2, fontsize=10, loc="center left", ncol=1, framealpha=0.9)

ax = axes[1]
# 대표: DI 변동이 큰 region
rr = pd.Series(reg[m]).value_counts().index
best = max(rr, key=lambda R: np.nanstd(di[(reg == R) & m]))
sel = (reg == best) & m
# 산점 수가 적으면 완전 벡터 PDF(rasterized=False), 많으면 파일 크기 때문에 래스터 유지
n_pts = int(sel.sum())
scv = ax.scatter(base.lon[sel], base.lat[sel], c=di[sel], s=24, cmap=CMAP.count,
                 vmin=np.nanpercentile(di[sel], 5), vmax=np.nanpercentile(di[sel], 95),
                 rasterized=(n_pts > 5000))
out = sel & (inside == 0)
# AOA 밖(외삽) 표시: 흰 후광 + 진청회색 테두리 원 — 짙은(고DI) 점 위와 범례 흰 배경에서 모두 판독,
# 강조색(적/주황) 미사용
ax.scatter(base.lon[out], base.lat[out], s=84, marker="o", facecolors="none",
           edgecolors="white", linewidths=2.4, zorder=5)
ax.scatter(base.lon[out], base.lat[out], s=84, marker="o", facecolors="none",
           edgecolors="#1f2d3d", linewidths=1.1, zorder=6, label="AOA 밖 (외삽 영역)")
# 컬러바: 색 범위를 DI p5-p95로 클리핑 → extend 화살표 + 명시 주석으로 표기
cb = fig.colorbar(scv, ax=ax, fraction=0.046, pad=0.04, extend="both")
cb.set_label("환경 비유사도 DI (무차원)", fontsize=11, rotation=270, labelpad=16)
cb.ax.tick_params(labelsize=9.5)
cb.ax.text(0.5, -0.045, "p5-p95\n클리핑", transform=cb.ax.transAxes,
           ha="center", va="top", fontsize=8.5, color="#495057")
ax.set_xlabel("경도 (°)", fontsize=12); ax.set_ylabel("위도 (°)", fontsize=12)
ax.tick_params(labelsize=10.5)
ax.xaxis.set_major_formatter(lon_formatter()); ax.yaxis.set_major_formatter(lat_formatter())
ax.set_aspect(1.0 / np.cos(np.deg2rad(60))); despine(ax)
ax.set_title("(b)", loc="left", fontsize=13, fontweight="bold")
ax.legend(fontsize=9.5, loc="best")
fig.tight_layout()
fig.savefig(mappath("alt_aoa_mask"), dpi=300, bbox_inches="tight")
fig.savefig(mappath("alt_aoa_mask", ext="pdf"), bbox_inches="tight")
print(f"저장: {mappath('alt_aoa_mask')} | {DEC_CSV} | 대표 region: {best} | "
      f"지도 산점 {n_pts}개 (rasterized={n_pts > 5000})")
