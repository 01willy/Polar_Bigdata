"""L2-2 잔차 공간 구조 — Moran's I 와 경험 변동도.

잔차 = 예측 − 관측. 알래스카 6-fold OOF: Stefan(λ=0)·Stefan+ridge(λ=0.75)·catboost_lo 직접. 정보 없음 전이 레나·캐나다:
Stefan·Stefan+CCI. 거리 = haversine km.
  - Moran's I: 가중 = k=8 최근접(행 표준화) 및 거리대역 5·20·50·100 km(이진 대칭, 자기 제외). 순열 검정 --nperm(기본 999),
    p = (1 + #{I_perm ≥ I_obs}) / (nperm + 1)(양의 자기상관 단측), z = (I − mean)/sd(순열 분포). E[I] = −1/(n−1).
  - 경험 변동도: γ(h) = Σ(r_i − r_j)² / (2 N(h)), 대역 0–300 km(로그 간격). 잔차 분산(문턱 참조)과 블록 간 분산 비율 병기.
산출 data/processed/m1/l2_residual_spatial_moran.csv · l2_residual_spatial_variogram.csv · l2_residual_spatial_summary.csv ·
     l2_residual_spatial_meta.json  그림 outputs/figures/m1/l2_residual_spatial.{png,pdf}
실행 python3 scripts/2_evaluation/l2_residual_spatial.py [--nperm 999]
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_common as L                                                     # noqa: E402
from polar.fidelity import TARGET                                        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--nperm", type=int, default=999)
ap.add_argument("--chunk", type=int, default=1000)
args = ap.parse_args()
t0 = time.time()
BANDS = [5.0, 20.0, 50.0, 100.0]
KNN = 8
VBINS = np.array([0, 0.5, 1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300], float)
METHODS_AK = ["stefan", "stefan_ridge075", "catboost_lo"]
METHODS_TR = ["stefan", "stefan_cci"]

df = L.load()
print("[oof] 알래스카 6-fold", flush=True)
oof = L.run_oof(df, L.alaska_folds(df), METHODS_AK)
print("[noinfo] 레나·캐나다", flush=True)
tr, _ = L.run_noinfo(df, ["Lena", "Canada"], METHODS_TR)
SETS = [("Alaska_OOF", oof, METHODS_AK)] + [(r, tr[tr.macro == r].reset_index(drop=True), METHODS_TR) for r in ["Lena", "Canada"]]


def neighbours(lat, lon, chunk):
    """최대 대역(100 km) 안의 정렬 쌍(i≠j) 목록(rows, cols, dist)과 kNN 이웃(n×k). 변동도 누적도 같은 통과에서 수행."""
    n = len(lat); rows, cols, dist = [], [], []; knn = np.zeros((n, KNN), int)
    vsum = {}; vcnt = np.zeros(len(VBINS) - 1)
    return n, rows, cols, dist, knn, vsum, vcnt


def analyse(name, d, methods):
    lat, lon = d.lat.values, d.lon.values; n = len(d)
    R = {m: d[f"pred_{m}"].values - d[TARGET].values for m in methods}
    Z = {m: R[m] - R[m].mean() for m in methods}
    rows, cols, dist = [], [], []; knn = np.zeros((n, KNN), int)
    vsum = {m: np.zeros(len(VBINS) - 1) for m in methods}; vcnt = np.zeros(len(VBINS) - 1)
    bmax = max(BANDS)
    for s in range(0, n, args.chunk):
        e = min(n, s + args.chunk)
        D = L.haversine_km(lat[s:e, None], lon[s:e, None], lat[None, :], lon[None, :])
        ii = np.arange(s, e)
        D[np.arange(e - s), ii] = np.inf                                   # 자기 제외
        knn[s:e] = np.argpartition(D, KNN, axis=1)[:, :KNN]
        r_, c_ = np.nonzero(D <= bmax)
        rows.append((r_ + s).astype(np.int32)); cols.append(c_.astype(np.int32)); dist.append(D[r_, c_].astype(np.float32))
        # 변동도(정렬 쌍, 300 km 이내)
        m300 = D <= VBINS[-1]
        r3, c3 = np.nonzero(m300)
        b = np.digitize(D[r3, c3], VBINS) - 1
        vcnt += np.bincount(b, minlength=len(VBINS) - 1)[:len(VBINS) - 1]
        for m in methods:
            dz = R[m][r3 + s] - R[m][c3]
            vsum[m] += np.bincount(b, weights=dz * dz, minlength=len(VBINS) - 1)[:len(VBINS) - 1]
    rows, cols, dist = map(np.concatenate, (rows, cols, dist))
    print(f"  [{name}] n {n:,} · 100 km 이내 정렬 쌍 {len(rows):,} · 거리 계산 {time.time()-t0:.0f}s", flush=True)
    # 가중 행렬
    Ws = {}
    for b in BANDS:
        k = dist <= b
        Ws[f"band{int(b)}km"] = sp.csr_matrix((np.ones(k.sum(), np.float32), (rows[k], cols[k])), shape=(n, n))
    Ws[f"knn{KNN}"] = sp.csr_matrix((np.full(n * KNN, 1.0 / KNN, np.float32), (np.repeat(np.arange(n), KNN), knn.ravel())), shape=(n, n))
    # 순열(모든 방법·가중 공통)
    rng = np.random.RandomState(0)
    P = np.stack([rng.permutation(n) for _ in range(args.nperm)], axis=1) if args.nperm > 0 else None
    mrows = []
    for wname, W in Ws.items():
        S0 = float(W.sum()); nnz = int(W.nnz)
        for m in methods:
            z = Z[m].astype(np.float32); zz = float(z @ z)
            I = (n / S0) * float(z @ (W @ z)) / zz
            EI = -1.0 / (n - 1)
            if P is not None:
                Zp = z[P]                                              # n × nperm
                WZ = W @ Zp
                Ip = (n / S0) * (Zp * WZ).sum(0) / zz
                p = (1 + int((Ip >= I).sum())) / (args.nperm + 1)
                zsc = (I - Ip.mean()) / Ip.std(ddof=1)
            else:
                p, zsc = np.nan, np.nan
            mrows.append(dict(set=name, method=m, weight=wname, n=n, n_pairs=nnz, S0=S0, moran_I=I, E_I=EI, z_perm=zsc, p_perm=p,
                              n_perm=args.nperm, resid_var=float(R[m].var()), mean_neighbours=nnz / n))
        print(f"    {wname}: nnz {nnz:,} · " + " · ".join(f"{r['method']} I={r['moran_I']:.3f} p={r['p_perm']:.3g}" for r in mrows[-len(methods):])
              + f" · {time.time()-t0:.0f}s", flush=True)
    vrows = []
    for i in range(len(VBINS) - 1):
        for m in methods:
            vrows.append(dict(set=name, method=m, h_lo_km=VBINS[i], h_hi_km=VBINS[i + 1], h_mid_km=np.sqrt(max(VBINS[i], 0.25) * VBINS[i + 1]),
                              n_pairs=int(vcnt[i] / 2), gamma_cm2=(vsum[m][i] / (2 * vcnt[i]) if vcnt[i] > 0 else np.nan),
                              resid_var_cm2=float(R[m].var())))
    srows = []
    for m in methods:
        g = pd.DataFrame(dict(r=R[m], b=d.block.values)).groupby("b").r
        bm, bn = g.mean(), g.size()
        between = float(np.average((bm - R[m].mean()) ** 2, weights=bn))
        srows.append(dict(set=name, method=m, n=n, n_blocks=int(d.block.nunique()), resid_mean=float(R[m].mean()), resid_sd=float(R[m].std()),
                          rmse=L.rmse(d[TARGET].values, d[f"pred_{m}"].values), between_block_var_frac=between / float(R[m].var()),
                          block_mean_abs_resid_median=float(bm.abs().median())))
    return mrows, vrows, srows


M, V, S = [], [], []
for name, d, methods in SETS:
    m_, v_, s_ = analyse(name, d, methods)
    M += m_; V += v_; S += s_
moran, vario, summ = pd.DataFrame(M), pd.DataFrame(V), pd.DataFrame(S)
moran.to_csv(L.OUT / "l2_residual_spatial_moran.csv", index=False)
vario.to_csv(L.OUT / "l2_residual_spatial_variogram.csv", index=False)
summ.to_csv(L.OUT / "l2_residual_spatial_summary.csv", index=False)

# ---------------------------------------------------------------- 그림(변동도 2패널)
from polar.plotstyle import use_polar, despine                            # noqa: E402
plt = use_polar()
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)
ax = axes[0]
for m in METHODS_AK:
    v = vario[(vario.set == "Alaska_OOF") & (vario.method == m)]
    ax.plot(v.h_mid_km, v.gamma_cm2, marker=L.METHOD_MARKER[m], color=L.METHOD_COLOR[m], lw=1.6, ms=5, label=L.METHOD_LABEL[m])
    ax.axhline(v.resid_var_cm2.iloc[0], color=L.METHOD_COLOR[m], lw=0.8, ls=":")
ax.set_xscale("log"); ax.set_xlabel("거리 h (km)"); ax.set_ylabel("반분산 γ(h) (cm²)")
ax.set_title("(a) 알래스카 지역 내 6-fold OOF 잔차 (n=13,606)", loc="left"); ax.legend(fontsize=9); despine(ax)
ax.text(0.02, 0.03, "점선: 잔차 분산(문턱 참조)", transform=ax.transAxes, fontsize=8.5, color="#555555")
ax = axes[1]
ls = {"Lena": "-", "Canada": "--"}
for reg in ["Lena", "Canada"]:
    for m in METHODS_TR:
        v = vario[(vario.set == reg) & (vario.method == m)]
        n_ = int(summ[(summ.set == reg) & (summ.method == m)].n.iloc[0])
        ax.plot(v.h_mid_km, v.gamma_cm2, marker=L.METHOD_MARKER[m], color=L.METHOD_COLOR[m], lw=1.6, ms=5, ls=ls[reg],
                label=f"{reg} · {L.METHOD_LABEL[m]} (n={n_:,})")
ax.set_xscale("log"); ax.set_xlabel("거리 h (km)"); ax.set_ylabel("반분산 γ(h) (cm²)")
ax.set_title("(b) 정보 없음 전이 잔차 (실선 레나 · 파선 캐나다)", loc="left"); ax.legend(fontsize=8.5); despine(ax)
exports = L.savefig(fig, "l2_residual_spatial")

L.write_meta(L.OUT / "l2_residual_spatial_meta.json", script="scripts/2_evaluation/l2_residual_spatial.py", n_perm=args.nperm,
             bands_km=BANDS, knn=KNN, variogram_bins_km=VBINS.tolist(), sets={n: int(len(d)) for n, d, _ in SETS},
             methods_alaska=METHODS_AK, methods_transfer=METHODS_TR, seeds=L.SEEDS,
             note="잔차 = 예측 − 관측. 대역 가중 이진 대칭·자기 제외, kNN 행 표준화. 순열 p 단측(양의 자기상관). 변동도는 정렬 쌍 합/(2N).",
             elapsed_s=round(time.time() - t0, 1), exports=exports)
print(moran[["set", "method", "weight", "moran_I", "z_perm", "p_perm", "mean_neighbours"]].round(4).to_string(index=False))
print(summ.round(3).to_string(index=False))
print(f"done {time.time()-t0:.0f}s")
