"""L2-3 오차 층화 표 — 층별 RMSE·bias·n 과 블록 짝지은 부트스트랩 CI.

집합: 알래스카 6-fold OOF(Stefan λ=0 · Stefan+ridge λ=0.75 · catboost_lo 직접), 정보 없음 전이 레나·캐나다·러시아 W/E
(Stefan · Stefan+CCI · catboost_lo 직접, 알래스카 전 셀 학습). 층: 관측 ALT 5분위(집합 내), 고도 4대역(<100·100–300·300–600·>600 m),
MAAT 3계급(<−6·−6~−2·>−2 °C), SOC 0–5 cm 3분위(집합 내), 경사 3계급(<0.5·0.5–2·≥2°), 위도 2° 대역.
CI = 층 내 0.5° 블록 재표집 1,000회(같은 재표집으로 전 방법 채점 → Δ는 짝지음), 블록 8개 미만이면 NaN.
산출 data/processed/m1/l2_stratified_error.csv · l2_stratified_error_meta.json  그림 outputs/figures/m1/l2_stratified_error.{png,pdf}
실행 python3 scripts/2_evaluation/l2_stratified_error.py [--nboot 1000]
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_common as L                                                     # noqa: E402
from polar.fidelity import TARGET                                        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--nboot", type=int, default=1000)
args = ap.parse_args()
t0 = time.time()
METHODS_AK = ["stefan", "stefan_ridge075", "catboost_lo"]
METHODS_TR = ["stefan", "stefan_cci", "catboost_lo"]
REGIONS = ["Lena", "Canada", "Russia_W", "Russia_E"]

df = L.load()
print("[oof] 알래스카 6-fold", flush=True)
oof = L.run_oof(df, L.alaska_folds(df), METHODS_AK)
print("[noinfo] 전이 4지역", flush=True)
tr, _ = L.run_noinfo(df, REGIONS, METHODS_TR)
SETS = [("Alaska_OOF", oof, METHODS_AK)] + [(r, tr[tr.macro == r].reset_index(drop=True), METHODS_TR) for r in REGIONS]


def qlabels(x, k, unit, fmt="{:.0f}"):
    """집합 내 k분위 구간 라벨. 반환 (라벨 배열, 경계)."""
    q = np.nanquantile(x, np.linspace(0, 1, k + 1))
    b = np.clip(np.searchsorted(q, x, side="right") - 1, 0, k - 1)
    lab = [f"Q{i+1} [{fmt.format(q[i])}–{fmt.format(q[i+1])}{unit}]" for i in range(k)]
    return np.array(lab)[b], lab


def fixed(x, edges, labels):
    b = np.clip(np.digitize(x, edges) - 1, 0, len(labels) - 1)
    return np.array(labels)[b], labels


def strata_of(d):
    out = {}
    out["ALT 5분위"] = qlabels(d[TARGET].values, 5, " cm")
    out["고도"] = fixed(d.dem_elev.values, [-np.inf, 100, 300, 600, np.inf], ["<100 m", "100–300 m", "300–600 m", ">600 m"])
    out["MAAT"] = fixed(d.e5_maat.values, [-np.inf, -6, -2, np.inf], ["<−6 °C", "−6~−2 °C", ">−2 °C"])
    out["SOC 0–5 cm 3분위"] = qlabels(d.sg_soc_0_5.values, 3, "", fmt="{:.0f}")
    out["경사"] = fixed(d.dem_slope.values, [-np.inf, 0.5, 2.0, np.inf], ["<0.5°", "0.5–2°", "≥2°"])
    lo = (np.floor(d.lat.values / 2) * 2).astype(int)
    out["위도 2° 대역"] = (np.array([f"{a}–{a+2}°N" for a in lo]), [f"{a}–{a+2}°N" for a in sorted(set(lo))])
    return out


rows = []
for name, d, methods in SETS:
    y, blocks = d[TARGET].values.astype(float), d.block.values
    strata = strata_of(d)
    strata["전체"] = (np.array(["전체"] * len(d)), ["전체"])
    for var, (lab, order) in strata.items():
        for oi, s in enumerate(order):
            m = lab == s
            if m.sum() == 0:
                continue
            preds = {mm: d[f"pred_{mm}"].values[m] for mm in methods}
            bb = L.block_boot(y[m], preds, blocks[m], nboot=args.nboot, seed=hash((name, var, s)) % (2 ** 31), ref="stefan")
            nb = len(np.unique(blocks[m]))
            for mm in methods:
                r = bb[mm]
                rows.append(dict(set=name, stratum_var=var, stratum=s, order=oi, n=int(m.sum()), n_blocks=nb, y_mean=float(y[m].mean()),
                                 y_sd=float(y[m].std()), method=mm, rmse=r["rmse"], rmse_lo=r["rmse_lo"], rmse_hi=r["rmse_hi"], bias=r["bias"],
                                 bias_lo=r["bias_lo"], bias_hi=r["bias_hi"], delta_vs_stefan=r["delta"], delta_lo=r["delta_lo"],
                                 delta_hi=r["delta_hi"], ci_flag="ok" if nb >= L.MIN_BLOCKS_CI else f"blocks<{L.MIN_BLOCKS_CI}"))
    print(f"  [{name}] n {len(d):,} · 층 {sum(len(o) for _, o in strata.values())} · {time.time()-t0:.0f}s", flush=True)
tab = pd.DataFrame(rows)
tab.to_csv(L.OUT / "l2_stratified_error.csv", index=False)

# ---------------------------------------------------------------- 그림: ALT 5분위 × 방법 (RMSE·bias) · 알래스카/레나/캐나다
from polar.plotstyle import use_polar, despine                            # noqa: E402
plt = use_polar()
panels = [("Alaska_OOF", METHODS_AK, "알래스카 지역 내 OOF"), ("Lena", METHODS_TR, "레나 (정보 없음)"), ("Canada", METHODS_TR, "캐나다 (정보 없음)")]
HATCH = {"stefan": "", "stefan_ridge075": "//", "catboost_lo": "..", "stefan_cci": "\\\\"}
fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0), constrained_layout=True, sharex="col")
for j, (name, methods, title) in enumerate(panels):
    sub = tab[(tab.set == name) & (tab.stratum_var == "ALT 5분위")]
    order = sub.sort_values("order").stratum.unique()
    x = np.arange(len(order)); w = 0.8 / len(methods)
    for row, (col, lo, hi, ylab) in enumerate([("rmse", "rmse_lo", "rmse_hi", "RMSE (cm)"), ("bias", "bias_lo", "bias_hi", "bias = 예측 − 관측 (cm)")]):
        ax = axes[row, j]
        for i, m in enumerate(methods):
            s = sub[sub.method == m].set_index("stratum").reindex(order)
            err = np.vstack([s[col] - s[lo], s[hi] - s[col]])
            err = np.where(np.isfinite(err), err, 0)
            ax.bar(x + (i - (len(methods) - 1) / 2) * w, s[col], w, yerr=err, color=L.METHOD_COLOR[m], edgecolor="#333333", lw=0.5,
                   hatch=HATCH[m], label=L.METHOD_LABEL[m], error_kw=dict(lw=0.7, capsize=2))
        if row == 1:
            ax.axhline(0, color="#444444", lw=0.7)
            ax.set_xticks(x); ax.set_xticklabels([o.replace(" [", "\n[") for o in order], fontsize=8.5)
            ax.set_xlabel("관측 ALT 5분위 (cm)")
        else:
            n_ = int(sub[sub.method == methods[0]].n.sum())
            ax.set_title(f"{title} (n={n_:,})", loc="left")
        if j == 0:
            ax.set_ylabel(ylab)
        despine(ax)
    axes[0, j].legend(fontsize=8.5, loc="upper left")
fig.text(0.005, 0.995, "오차막대: 층 내 0.5° 블록 재표집 95% CI(블록 8개 미만 미표시)", fontsize=8.5, color="#555555", va="top")
exports = L.savefig(fig, "l2_stratified_error")

L.write_meta(L.OUT / "l2_stratified_error_meta.json", script="scripts/2_evaluation/l2_stratified_error.py", n_boot=args.nboot,
             sets={n: int(len(d)) for n, d, _ in SETS}, methods_alaska=METHODS_AK, methods_transfer=METHODS_TR, seeds=L.SEEDS,
             strata=dict(alt="집합 내 5분위", elev_m=[100, 300, 600], maat_c=[-6, -2], soc="집합 내 3분위(sg_soc_0_5, SoilGrids 단위)",
                         slope_deg=[0.5, 2.0], lat="2° 대역"),
             note="CI: 층 내 블록 재표집 1,000회, 같은 재표집으로 전 방법 채점(Δ 짝지음). 블록 8개 미만 NaN.",
             elapsed_s=round(time.time() - t0, 1), exports=exports)
pv = tab[tab.stratum_var.isin(["ALT 5분위", "전체"])].pivot_table(index=["set", "stratum"], columns="method", values="rmse").round(2)
print(pv.to_string())
print(f"done {time.time()-t0:.0f}s")
