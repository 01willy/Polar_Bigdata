"""E4.4: 라벨 정의 민감도 (W8) — 관측 기기(GPR vs 탐침)·관측 시기가 알래스카 라벨과 전이 결론에 미치는 영향.

발견(2026-09-08): ABoVE 알래스카 점관측 219,089개 중 GPR 유도 ALT 86%, 탐침 7.7%(그 외 결측). 8–9월 관측 97%.
전이 대상(CALM·ALLena)은 탐침·계절 최대 기준이므로 학습원(알래스카)과 라벨 정의가 다르다.
계획 `docs/EXPERIMENT_PLAN_PAPER_2026-09-08.md` §4 E4.4.

방법
----
1) ABoVE 원자료를 좌표 키(round 4, 셀 정의와 동일)로 집계해 셀별 GPR 비율·탐침 비율·8–9월 비율·평균 관측월을 만든다.
2) 알래스카 셀을 '탐침 우세(탐침 ≥ 50%)' / 'GPR 우세' / 'CALM(탐침)' 으로 나눈다.
3) 지역 내(6-fold): Stefan E를 전체·GPR·탐침 셀 각각에서 적합(fold 내부) → E 차이와 그 함의(√TDD 중앙값에서의 ALT 차이, cm).
   전체 학습 Stefan·CatBoost 잔차의 부분집합별 편향(GPR 셀 vs 탐침 셀).
4) 전이: E를 (a) 전체 셀, (b) 탐침 셀만으로 적합해 주 전이 집합 6지역(정보 없음, 전체 셀)에서 Stefan·Stefan+CCI RMSE 비교.
   → "학습 라벨을 탐침으로 통일하면 전이 결론이 바뀌는가"에 답한다.

입력: data/raw/above/ABoVE_Soil_ThawDepth_Moisture_Validation_V2.csv, data/processed/fidelity_base_v2.csv
산출: data/processed/e4_label_sensitivity.csv, e4_label_cell_meta.csv, e4_label_sensitivity_meta.json
실행(ROOT, CPU): python3 scripts/3_deep_learning/e4_label_sensitivity.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import add_group_keys, macro_region, spatial_block_splits, TARGET, TRANSFER_MAIN  # noqa: E402

PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "above" / "ABoVE_Soil_ThawDepth_Moisture_Validation_V2.csv"
KEY = 4

# ---------------------------------------------------------------- 1) 셀별 기기·시기 메타
raw = pd.read_csv(RAW, usecols=["latitude", "longitude", "date", "ALT_instrument", "ALT"], low_memory=False)
raw = raw[(raw.ALT != -9999) & raw.ALT.notna() & (raw.ALT > 0) & (raw.ALT < 300) & (raw.latitude != -9999)]
raw["month"] = pd.to_datetime(raw.date, errors="coerce").dt.month
raw["klat"], raw["klon"] = raw.latitude.round(KEY), raw.longitude.round(KEY)
raw["is_gpr"] = (raw.ALT_instrument == "GPR").astype(float)
raw["is_probe"] = (raw.ALT_instrument == "Probe").astype(float)
raw["late"] = raw.month.isin([8, 9]).astype(float)
cm = (raw.groupby(["klat", "klon"], as_index=False)
         .agg(n_pts=("ALT", "size"), frac_gpr=("is_gpr", "mean"), frac_probe=("is_probe", "mean"),
              frac_late=("late", "mean"), month_mean=("month", "mean")))

base = add_group_keys(pd.read_csv(PROC / "fidelity_base_v2.csv", low_memory=False))
base["macro"] = macro_region(base)
base = base[base[TARGET].notna() & (base.source_id == "F4_direct")].reset_index(drop=True)
base["klat"], base["klon"] = base.lat.round(KEY), base.lon.round(KEY)
base = base.merge(cm, on=["klat", "klon"], how="left")
ak = base[base.macro == "Alaska"].copy()
ak["label_class"] = np.where(ak.region == "United States (Alaska)", "CALM(탐침)",
                     np.where(ak.frac_probe >= 0.5, "ABoVE 탐침 우세",
                     np.where(ak.frac_gpr >= 0.5, "ABoVE GPR 우세", "ABoVE 기타/결측")))
print("[알래스카 셀 라벨 분류]")
print(ak.groupby("label_class").agg(n=("loc_id", "size"), alt=("alt_cm", "mean"), sqrt_tdd=("e5_sqrt_tdd", "mean"),
                                    late=("frac_late", "mean")).round(2).to_string())
ak[["loc_id", "region", "label_class", "n_pts", "frac_gpr", "frac_probe", "frac_late", "month_mean"]].to_csv(
    PROC / "e4_label_cell_meta.csv", index=False)


def fit_E(y, s):
    m = np.isfinite(y) & np.isfinite(s) & (s > 0)
    return float((s[m] @ y[m]) / (s[m] @ s[m]))


rows = []
# ---------------------------------------------------------------- 3) 지역 내: 부분집합별 E와 편향
folds = spatial_block_splits(ak.reset_index(drop=True), n_splits=6)
akr = ak.reset_index(drop=True)
y, s = akr[TARGET].values.astype(float), akr.e5_sqrt_tdd.values.astype(float)
cls = akr.label_class.values
oof = {k: np.full(len(akr), np.nan) for k in ["E_all", "E_gpr", "E_probe"]}
Es = []
for fi, (tr, te) in enumerate(folds):
    sel = {"E_all": tr, "E_gpr": tr[cls[tr] == "ABoVE GPR 우세"],
           "E_probe": tr[np.isin(cls[tr], ["ABoVE 탐침 우세", "CALM(탐침)"])]}
    for k, idx in sel.items():
        E = fit_E(y[idx], s[idx]) if len(idx) >= 3 else np.nan
        oof[k][te] = E * s[te]
        Es.append(dict(fold=fi, fit_on=k, E=E, n=len(idx)))
Es = pd.DataFrame(Es)
print("\n[fold별 E]"); print(Es.pivot_table(index="fold", columns="fit_on", values="E").round(4).to_string())
E_tab = Es.groupby("fit_on").E.agg(["mean", "std"]).round(4)
s_med = float(np.nanmedian(s))
print(f"\n√TDD 중앙값 {s_med:.2f} 에서 E 차이의 ALT 함의: (E_probe−E_gpr)·√TDD = "
      f"{(E_tab.loc['E_probe','mean'] - E_tab.loc['E_gpr','mean']) * s_med:+.2f} cm")
for k in oof:
    for c in ["ABoVE GPR 우세", "ABoVE 탐침 우세", "CALM(탐침)", "ALL"]:
        m = np.ones(len(akr), bool) if c == "ALL" else (cls == c)
        m &= np.isfinite(oof[k])
        if m.sum() < 5:
            continue
        r = oof[k][m] - y[m]
        rows.append(dict(axis="indomain_AK", fit_on=k, eval_subset=c, n=int(m.sum()),
                         rmse=float(np.sqrt(np.mean(r ** 2))), bias=float(np.mean(r))))
# ---------------------------------------------------------------- 4) 전이: E_all vs E_probe
E_all = fit_E(y, s)
probe_m = np.isin(cls, ["ABoVE 탐침 우세", "CALM(탐침)"])
E_probe = fit_E(y[probe_m], s[probe_m])
gpr_m = cls == "ABoVE GPR 우세"
E_gpr = fit_E(y[gpr_m], s[gpr_m])
print(f"\n[전체 적합 E] all {E_all:.4f} · probe {E_probe:.4f} (n={probe_m.sum()}) · gpr {E_gpr:.4f} (n={gpr_m.sum()})")
for tg in TRANSFER_MAIN:
    t = base[(base.macro == tg) & base.cci_alt.notna()]
    if len(t) < 3:
        continue
    yt, st, ct = t[TARGET].values.astype(float), t.e5_sqrt_tdd.values.astype(float), t.cci_alt.values.astype(float)
    for name, E in [("E_all", E_all), ("E_probe", E_probe), ("E_gpr", E_gpr)]:
        p = E * st
        rows.append(dict(axis="transfer_noinfo", fit_on=name, eval_subset=tg, n=len(t),
                         rmse=float(np.sqrt(np.mean((p - yt) ** 2))), bias=float(np.mean(p - yt)),
                         rmse_stefan_cci=float(np.sqrt(np.mean((0.5 * (p + ct) - yt) ** 2)))))
res = pd.DataFrame(rows)
res.to_csv(PROC / "e4_label_sensitivity.csv", index=False)
pd.set_option("display.width", 200)
print("\n=== 지역 내: 적합 라벨 × 평가 부분집합 (OOF, 6-fold) ===")
print(res[res.axis == "indomain_AK"].pivot_table(index="eval_subset", columns="fit_on", values=["rmse", "bias"]).round(2).to_string())
print("\n=== 전이(정보 없음, 주 집합): Stefan RMSE by E 적합 라벨 ===")
tr = res[res.axis == "transfer_noinfo"]
print(tr.pivot_table(index="eval_subset", columns="fit_on", values="rmse").round(2).to_string())
print("\n[Stefan+CCI]"); print(tr.pivot_table(index="eval_subset", columns="fit_on", values="rmse_stefan_cci").round(2).to_string())
print("\n[지역 비가중 평균]"); print(tr.groupby("fit_on")[["rmse", "rmse_stefan_cci"]].mean().round(2).to_string())
(PROC / "e4_label_sensitivity_meta.json").write_text(json.dumps(dict(
    stage="E4.4", n_ak_cells=int(len(ak)), label_class_counts=ak.label_class.value_counts().to_dict(),
    E_all=E_all, E_probe=E_probe, E_gpr=E_gpr, sqrt_tdd_median=s_med,
    raw_counts=dict(n_pts=int(len(raw)), gpr=int(raw.is_gpr.sum()), probe=int(raw.is_probe.sum()), late=int(raw.late.sum())),
    plan="docs/EXPERIMENT_PLAN_PAPER_2026-09-08.md §4 E4.4"), ensure_ascii=False, indent=1))
print("saved: e4_label_sensitivity.csv · e4_label_cell_meta.csv")
