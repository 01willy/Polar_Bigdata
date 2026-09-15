"""E4.1: S12 185조합 탐색의 winner's curse 해소 — 중첩(leave-one-target-out) 선택 재평가.

배경
----
tab:s12의 22.92(정보 없음)·21.32(공변량만)는 185구성 × λ 5단계를 평가 셀에서 채점한 뒤 최소값을
보고한 것이다(선택과 채점이 같은 표본). 선택이 개입하지 않은 수치는 등가중 앵커(λ=0)뿐이다.
계획 `docs/EXPERIMENT_PLAN_PAPER_2026-09-08.md` §4 E4.1.

방법
----
대상 지역이 레나델타·캐나다 둘이므로 중첩 선택은 leave-one-target-out이다.
  선택: 지역 X에서 seed 평균 RMSE 최소 구성을 고른다.
  채점: 그 구성을 지역 Y에서 채점한다. (X,Y) = (레나, 캐나다), (캐나다, 레나).
  보고: 두 채점값의 비가중 평균 = "선택 편향이 제거된 탐색 결과".
비교 기준(같은 CSV, 같은 프로토콜):
  R0 표 값(in-sample 최소)           : 선택·채점 동일 표본(winner's curse 포함)
  R1 물리식 단독(CCI 유효 셀)          : anchor=stefan_cci_w (W_SC=1.0 → Stefan on CCI-valid cells)
  R2 사전 지정 H1: 등가중 앵커 λ=0    : anchor=stefan_cci, resid=none
  R3 사전 지정 H2: 앵커+ridge λ=0.25 : anchor=stefan_cci, resid=ridge, pseudo=none, lam=0.25
  R4 중첩 선택(전 구성)
  R5 중첩 선택(anchored 계열만)
계열별 중첩 선택도 함께 낸다.

입력: data/processed/s12_hybrid_transfer_shard{0..3}.csv, s12_hybrid_transfer_meta_shard0.json
산출: data/processed/e4_nested_selection.csv, e4_nested_selection_meta.json
실행: python3 scripts/3_deep_learning/e4_nested_selection.py
"""
from __future__ import annotations
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed"

CFG = ["family", "anchor", "pseudo", "resid", "r", "lam"]
TARGETS = ["Lena", "Canada"]

files = sorted(glob.glob(str(OUT / "s12_hybrid_transfer_shard*.csv")))
raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
meta0 = json.loads((OUT / "s12_hybrid_transfer_meta_shard0.json").read_text())
W_SC = float(meta0["w_stefan_cci"])
print(f"[load] {len(files)} shard · {len(raw)}행 · w_stefan={W_SC:.3f}")

ok = raw[raw.rmse_cm.notna()].copy()
ok["r"] = ok["r"].astype(float)
ok["lam"] = ok["lam"].astype(float)

# seed 평균 (해석적 구성은 seed=-1 단일행)
agg = (ok.groupby(["proto", "target"] + CFG, as_index=False)
         .agg(rmse=("rmse_cm", "mean"), bias=("bias_cm", "mean"),
              n_cells=("n", "first"), n_seed=("seed", "nunique")))

# 두 지역 모두 산출된 구성만(지역 수 확인 후 집계: 07-29 감사의 집계 함정 방지)
both = (agg.groupby(["proto"] + CFG).target.nunique() == 2).reset_index(name="both")
agg = agg.merge(both[both.both][["proto"] + CFG], on=["proto"] + CFG)
print(f"[cfg] 두 지역 모두 산출된 (proto, 구성·λ) 조합: "
      f"{agg.groupby('proto').apply(lambda g: g[CFG].drop_duplicates().shape[0]).to_dict()}")

wide = agg.pivot_table(index=["proto"] + CFG, columns="target", values="rmse").reset_index()
wide["mean2"] = wide[TARGETS].mean(axis=1)


def row(proto, label, lena, canada, note=""):
    return dict(proto=proto, reference=label, rmse_lena=lena, rmse_canada=canada,
                rmse_mean2=np.nanmean([lena, canada]), note=note)


def pick(w, **kw):
    m = np.ones(len(w), bool)
    for k, v in kw.items():
        m &= (w[k] == v).values
    return w[m]


rows = []
for proto in ["loro", "half"]:
    w = wide[wide.proto == proto].copy()

    # R0 in-sample 최소(표 값 재현)
    best = w.loc[w.mean2.idxmin()]
    rows.append(row(proto, "R0 in-sample 최소(표 tab:s12)", best.Lena, best.Canada,
                    note=" ".join(f"{k}={best[k]}" for k in CFG)))

    # R1 물리식 단독(CCI 유효 셀)
    r1 = pick(w, anchor="stefan_cci_w", resid="none")
    assert len(r1) == 1
    rows.append(row(proto, "R1 물리식 단독(CCI 유효 셀)", r1.Lena.item(), r1.Canada.item(),
                    note=f"anchor=stefan_cci_w(W_SC={W_SC:.2f})"))

    # R2 H1 등가중 앵커 λ=0
    r2 = pick(w, anchor="stefan_cci", resid="none")
    assert len(r2) == 1
    rows.append(row(proto, "R2 H1 사전지정: Stefan+CCI 등가중 앵커(λ=0)", r2.Lena.item(), r2.Canada.item()))

    # R3 H2 앵커+ridge λ=0.25, 증강 없음
    r3 = pick(w, anchor="stefan_cci", resid="ridge", pseudo="none", lam=0.25)
    if len(r3) == 1:
        rows.append(row(proto, "R3 H2 사전지정: Stefan+CCI 앵커 + ridge(λ=0.25), 증강 없음",
                        r3.Lena.item(), r3.Canada.item()))
    else:
        rows.append(row(proto, "R3 H2 사전지정", np.nan, np.nan, note=f"행 {len(r3)}개(산출 없음)"))

    # R4 중첩 선택(전 구성) — X에서 선택, Y에서 채점
    def nested(sub, label):
        sel_on_lena = sub.loc[sub.Lena.idxmin()]      # 레나에서 선택 → 캐나다 채점
        sel_on_can = sub.loc[sub.Canada.idxmin()]     # 캐나다에서 선택 → 레나 채점
        lena_score = sel_on_can.Lena
        can_score = sel_on_lena.Canada
        note = ("레나선택→캐나다채점: " + " ".join(f"{k}={sel_on_lena[k]}" for k in CFG)
                + " | 캐나다선택→레나채점: " + " ".join(f"{k}={sel_on_can[k]}" for k in CFG))
        return row(proto, label, lena_score, can_score, note=note)

    rows.append(nested(w, "R4 중첩 선택(전 구성)"))
    rows.append(nested(w[w.family == "anchored"], "R5 중첩 선택(anchored 계열)"))
    rows.append(nested(w[w.family == "direct"], "R6 중첩 선택(direct 계열)"))
    rows.append(nested(w[w.family == "analytic"], "R7 중첩 선택(analytic 계열)"))
    rows.append(nested(w[(w.family == "anchored") & (w.pseudo == "none")],
                       "R8 중첩 선택(anchored, 증강 없음)"))

res = pd.DataFrame(rows)
res["delta_vs_R1"] = np.nan
for proto in ["loro", "half"]:
    base = res[(res.proto == proto) & res.reference.str.startswith("R1")].rmse_mean2.item()
    res.loc[res.proto == proto, "delta_vs_R1"] = res.loc[res.proto == proto, "rmse_mean2"] - base

pd.set_option("display.width", 200)
print(res[["proto", "reference", "rmse_lena", "rmse_canada", "rmse_mean2", "delta_vs_R1"]]
      .round(2).to_string(index=False))
print("\n[선택된 구성]")
for _, r in res.iterrows():
    if r.note:
        print(f"  {r.proto:4s} {r.reference[:28]:28s} {r.note}")

res.to_csv(OUT / "e4_nested_selection.csv", index=False)
(OUT / "e4_nested_selection_meta.json").write_text(json.dumps(dict(
    stage="E4.1", inputs=[Path(f).name for f in files], w_stefan_cci=W_SC,
    method="leave-one-target-out: 지역 X에서 seed 평균 RMSE 최소 구성 선택, 지역 Y에서 채점. "
           "두 지역 모두 산출된 구성만 사용. R1은 CCI 유효 셀 한정 Stefan(anchor=stefan_cci_w, W_SC=1.0)",
    plan="docs/EXPERIMENT_PLAN_PAPER_2026-09-08.md §4 E4.1, §7 H1·H2",
), ensure_ascii=False, indent=1))
print(f"\nsaved: {OUT / 'e4_nested_selection.csv'}")
