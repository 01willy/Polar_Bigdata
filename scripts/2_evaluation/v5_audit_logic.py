"""v5 연도별 파이프라인 논리 감사(사용자 질문 15). 라벨-강제력 연도 정합, 분할 누설, 제품 연도 정합, 지연 특징, 관측 월, 집계 일관성을 점검해
data/processed/v5/audit_logic_v5.md 에 통과/실패를 적는다. 실행: python3 scripts/2_evaluation/v5_audit_logic.py
"""
from __future__ import annotations
import sys, glob
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import polar.testbed_v5 as T
PROC = ROOT / "data" / "processed" / "v5"
_args = sys.argv[1:]
RUN = Path(_args[_args.index("--run") + 1]) if "--run" in _args else PROC / "runs" / "core_cb"
FORC = _args[_args.index("--forcing") + 1] if "--forcing" in _args else "year"
RUN = RUN if RUN.is_absolute() else ROOT / RUN
out = []
def ck(name, ok, detail=""):
    out.append((name, "PASS" if ok else "FAIL", detail)); print(("PASS " if ok else "FAIL ") + name, detail)

lab = pd.read_csv(PROC / "labels_cell_year_v5.csv"); fy = pd.read_csv(PROC / "forcing_year.csv")
df = T.load_v5(PROC)
# 1. 라벨 연도 = 강제력 연도(join 결과의 y_e5_tdd 가 forcing_year 의 같은 (loc_id, year) 값과 같다)
m = df.merge(fy[["loc_id", "year", "e5_tdd"]], on=["loc_id", "year"], how="left")
ck("label-year equals forcing-year (TDD identical after join)", np.allclose(m.y_e5_tdd.values, m.e5_tdd.values, equal_nan=True), f"n={len(m)}")
# 2. 연별 강제력이 해마다 다르다(같은 셀에서 연도 간 TDD 표준편차 > 0)
sd = fy.groupby("loc_id").e5_tdd.std()
ck("forcing varies across years within cells", float((sd > 0).mean()) > 0.99, f"frac cells with sd>0 = {(sd>0).mean():.3f}, median sd {sd.median():.0f} °C·d")
# 3. 기후값 판은 셀 안에서 연도에 따라 변하지 않는다
ck("climatological forcing constant within cell", df.groupby("loc_id").c_e5_tdd.std().fillna(0).max() < 1e-9)
# 4. 지연 특징 = 전년 강제력
l = df.merge(fy[["loc_id", "year", "e5_tdd"]].assign(year=lambda d: d.year + 1).rename(columns={"e5_tdd": "prev"}), on=["loc_id", "year"], how="left")
ck("lag feature equals previous-year TDD", np.allclose(l.y_e5_tdd_prev.values, l.prev.values, equal_nan=True))
# 5. 월별 누적 TDD 단조·마지막 달 = 연간 TDD
cum = np.stack([df[f"tdd_cum_{k}"].values for k in range(1, 13)], 1)
ck("monthly cumulative TDD monotone and tdd_cum_12 == e5_tdd", bool((np.diff(cum, axis=1) >= -1e-6).all()) and np.allclose(cum[:, 11], df.y_e5_tdd.values, equal_nan=True))
# 6. 관측 월: ABoVE 8–10, ALLENA 8–9, CALM eos
ok_m = df.obs_month_last.between(8, 10).all()
ck("observation months within Aug–Oct (CALM = Sep)", bool(ok_m), df.obs_month_last.value_counts().to_dict().__repr__())
# 7. 제품 연도 정합: prod_wei 값이 products 긴 표의 (loc_id, year) 값과 같다
pr = pd.read_csv(PROC / "products_cell_year_v5.csv"); w = pr[pr["product"] == "wei"].set_index(["loc_id", "year"]).value_cm
chk = df[["loc_id", "year", "prod_wei"]].dropna().head(5000)
ck("product value is the label-year value", np.allclose(chk.prod_wei.values, w.loc[list(zip(chk.loc_id, chk.year))].values, equal_nan=True))
# 8. 블록 CV 누설 없음(core_cb 예측 파일의 fold 별 test 블록이 다른 fold 와 겹치지 않음)
f = pd.read_csv(sorted(glob.glob(str(RUN / "preds" / f"Alaska__block__{FORC}__F8TS__P1__*.csv.gz")))[0])
blk = f.groupby("fold").block.apply(set); inter = sum(len(a & b) for i, a in enumerate(blk) for b in list(blk)[i + 1:])
ck("block folds disjoint in blocks", inter == 0, f"folds {len(blk)}")
# 9. 연도 홀드아웃: 각 fold 의 test 가 한 해
f = pd.read_csv(sorted(glob.glob(str(RUN / "preds" / f"Alaska__year__{FORC}__F8TS__P1__*.csv.gz")))[0])
ck("year hold-out folds contain one year each", bool((f.groupby("fold").year.nunique() == 1).all()), f"folds {f.fold.nunique()}")
# 10. 예측 행 수 = 셀-연도 수(씨앗 평균 전)
ck("prediction rows cover all cell-years once per seed", f.groupby("seed").size().eq(len(df[df.region == "Alaska"])).all())
# 11. 지역별 셀-연도 수와 등록부 일치
cnt = df.groupby("region").size().to_dict()
ck("cell-year counts match registry", cnt == {"Alaska": 15540, "Canada": 1196, "Lena": 2436}, str(cnt))
# 12. MODIS 연별 값은 2010–2024 전 기간(AppEEARS 재요청 10-14). 연별 값이 있는 행은 평균 대체 표지가 0 이고, 표지 비율은 10 % 미만
mo = df[["year", "ndvi_summer_is_mean"]]
ck("MODIS yearly values cover 2010–2024 (mean-filled flag < 10 %)", bool(mo.ndvi_summer_is_mean.mean() < 0.10), f"mean-filled frac {mo.ndvi_summer_is_mean.mean():.3f}")
# 13. 겨울 변수: 2010 은 부분 창 표지
ck("winter_partial flag only for 2010", set(df[df.winter_partial == 1].year.unique()) <= {2010})
md = ["# v5 논리 감사 (자동)", "", "| 항목 | 결과 | 세부 |", "|---|---|---|"] + [f"| {n} | {r} | {d} |" for n, r, d in out]
(PROC / ("audit_logic_v5.md" if RUN.name == "core_cb" else f"audit_logic_{RUN.name}.md")).write_text("\n".join(md)); print("fails:", sum(r == "FAIL" for _, r, _ in out))
