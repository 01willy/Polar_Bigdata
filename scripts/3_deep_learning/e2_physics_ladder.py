"""E2: 물리식 사다리 — 같은 보정 자유도(k)에서의 물리·경험식 비교.

배경
----
tab:physics(S2)는 Stefan(라벨로 E 1개 보정)과 무보정 pedotransfer 4종(edaphic·TTOP·Kudryavtsev·λ보정)을
나란히 두었다. "Stefan만 정확"은 식의 형태와 보정 유무가 교락된 결론이다. 5종 상관 0.93–1.00.
계획 `docs/EXPERIMENT_PLAN_PAPER_2026-09-08.md` §2, 사전 등록 H4(같은 k에서 비교)·H5(멱지수 b).

식 계열 × k (k = 학습 fold 라벨로 적합하는 계수 수)
  F1 Stefan          k1: E·s            k2: a + E·s                 (s = √TDD_air)
  F2 멱법칙          k2: a·TDD^b                                     (b≈0.5 이면 √ 형태 지지)
  F3 TDD 선형        k2: a + b·TDD                                   (저복잡도 대조, W6)
  F4 MAAT 선형       k2: a + b·MAAT                                  (대조)
  F5 edaphic Stefan  k0: p2 raw        k1: c·p2      k2: a + c·p2   (Johansen 열전도도, 토양)
  F6 Kudryavtsev     k0: p4 raw        k1: c·p4      k2: a + c·p4   (눈 절연·오프셋)
  F7 지표강제 Stefan k1: E·s_soil      k2: a + E·s_soil             (s_soil = √TDD_stl1, 신규 공변량)
  F8 2층 Stefan      k2: (h, E_m)  유기층 h·f_om/med(f_om) 위 광물층, E_o = 0.5·E_m 고정
  F9 Stefan+FDD      k2: E·s + γ·√FDD                              (동결 성분)
  F10 Stefan×눈      k2: E·s·(1 + γ·SWE)                           (눈 변조)

평가
----
  지역 내: 알래스카 0.5° 공간블록 6-fold. 계수는 train fold에서만 적합. OOF 채점.
  전이   : LORO(Alaska·Lena·Canada; 셀 100 이상). 계수는 학습 지역에서 적합. 지역별 채점 + 비가중 평균.
  모든 행에 같은 셀의 F1k1(Stefan E) 오차를 짝지어 기록(delta, 0.5° 블록 짝지은 부트스트랩 95% CI).
  식마다 산출 가능 셀이 다르므로(F6은 Ku 동토 판정 셀만) n을 기록하고 짝지은 Stefan으로 비교한다.
  계수(E, a, b, c, h, γ)는 fold/지역별로 저장(H5: b의 분포).
  지역 내 OOF 예측의 식 간 Pearson·Spearman 상관행렬(다양성 진단).

입력: data/processed/fidelity_base.csv, e5_soil_tdd.csv
산출: data/processed/e2_physics_ladder_results.csv, e2_physics_ladder_coefs.csv, e2_physics_corr.csv, e2_physics_ladder_meta.json
실행(ROOT, CPU): python3 scripts/3_deep_learning/e2_physics_ladder.py
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.fidelity import add_group_keys, macro_region, spatial_block_splits, loro_splits, TARGET  # noqa: E402
from polar.physics import physics_ensemble, load_physics_inputs                                   # noqa: E402
from polar.eval_metrics import all_metrics                                                         # noqa: E402
import argparse                                                                                    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="fidelity_base.csv", help="fidelity_base_v2.csv 로 E3 신규 지역 포함 실행")
ap.add_argument("--min-test", type=int, default=100, help="LORO 대상 지역 최소 셀 수")
ap.add_argument("--regions", default="", help="LORO 대상 매크로 지역 제한(쉼표). 비우면 min-test 충족 전부")
ap.add_argument("--tag", default="")
ap.add_argument("--exclude", default="", help="학습·평가에서 제외할 매크로 지역(쉼표). 심부 레짐 분리용")
args = ap.parse_args()
TAG = args.tag

PROC = ROOT / "data" / "processed"
NBOOT = 400
RNG = np.random.RandomState(0)

# ---------------------------------------------------------------- 자료
df = add_group_keys(pd.read_csv(PROC / args.base, low_memory=False))
soil = pd.read_csv(PROC / "e5_soil_tdd.csv")
s1 = soil[soil.loc_id >= 0][["loc_id", "e5_sqrt_tdd_soil", "e5_tdd_soil"]]
df = df.merge(s1, on="loc_id", how="left")
if df.e5_sqrt_tdd_soil.isna().any():                       # 신규 CALM 셀: 좌표로 병합
    s2 = soil[soil.loc_id < 0][["lat", "lon", "e5_sqrt_tdd_soil", "e5_tdd_soil"]].copy()
    s2["klat"], s2["klon"] = s2.lat.round(4), s2.lon.round(4)
    df["klat"], df["klon"] = df.lat.round(4), df.lon.round(4)
    df = df.merge(s2[["klat", "klon", "e5_sqrt_tdd_soil", "e5_tdd_soil"]].rename(
        columns={"e5_sqrt_tdd_soil": "_a", "e5_tdd_soil": "_b"}), on=["klat", "klon"], how="left")
    df["e5_sqrt_tdd_soil"] = df.e5_sqrt_tdd_soil.fillna(df["_a"])
    df["e5_tdd_soil"] = df.e5_tdd_soil.fillna(df["_b"])
    df = df.drop(columns=["klat", "klon", "_a", "_b"])
df["macro"] = macro_region(df)
df = df[df[TARGET].notna() & (df.source_id == "F4_direct")].reset_index(drop=True)
REGION_FILTER = [r for r in args.regions.split(",") if r]
EXCLUDE = [r for r in args.exclude.split(",") if r]
if EXCLUDE:
    df = df[~df.macro.isin(EXCLUDE)].reset_index(drop=True)
print(f"[data] {len(df):,}셀 · macro {df.macro.value_counts().to_dict()}")

# 공변량 전용 물리 산출(라벨 미사용). E는 p1에만 쓰이며 여기서는 p2·p4만 사용.
phys = physics_ensemble(df, E=1.0)
pin = load_physics_inputs(df)
X = dict(
    s=df["e5_sqrt_tdd"].values.astype(float),
    tdd=np.clip(df["e5_tdd"].values.astype(float), 0, None),
    maat=df["e5_maat"].values.astype(float),
    fdd=np.clip(df["e5_fdd"].values.astype(float), 0, None),
    swe=np.clip(df["e5_swe"].values.astype(float), 0, 0.2),
    p2=phys["p2_edaphic"], p4=phys["p4_kudryavtsev"],
    s_soil=df["e5_sqrt_tdd_soil"].values.astype(float),
    f_om=pin["f_om"],
)
y_all = df[TARGET].values.astype(float)
blocks_all = df["block"].values


# ---------------------------------------------------------------- 적합기
def _ols(A, y):
    m = np.isfinite(A).all(axis=1) & np.isfinite(y)
    coef, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
    return coef


def fit_F1k1(tr):
    s, y = X["s"][tr], y_all[tr]
    m = np.isfinite(s) & np.isfinite(y) & (s > 0)
    E = float((s[m] @ y[m]) / (s[m] @ s[m]))
    return dict(E=E), lambda te: E * X["s"][te]


def fit_F1k2(tr):
    c = _ols(np.c_[np.ones(len(tr)), X["s"][tr]], y_all[tr])
    return dict(a=c[0], E=c[1]), lambda te: c[0] + c[1] * X["s"][te]


def fit_F2k2(tr):
    t, y = X["tdd"][tr], y_all[tr]
    m = (t > 1) & (y > 0) & np.isfinite(y)
    lc = _ols(np.c_[np.ones(m.sum()), np.log(t[m])], np.log(y[m]))     # 로그공간 초기값
    a0, b0 = float(np.exp(lc[0])), float(lc[1])
    res = least_squares(lambda p: p[0] * t[m] ** p[1] - y[m], x0=[a0, b0],
                        bounds=([1e-6, 0.05], [1e4, 1.5]))
    a, b = map(float, res.x)
    return dict(a=a, b=b, b_log=b0), lambda te: a * X["tdd"][te] ** b


def fit_F3k2(tr):
    c = _ols(np.c_[np.ones(len(tr)), X["tdd"][tr]], y_all[tr])
    return dict(a=c[0], b=c[1]), lambda te: c[0] + c[1] * X["tdd"][te]


def fit_F4k2(tr):
    c = _ols(np.c_[np.ones(len(tr)), X["maat"][tr]], y_all[tr])
    return dict(a=c[0], b=c[1]), lambda te: c[0] + c[1] * X["maat"][te]


def _scale_family(key):
    def k0(tr):
        return dict(), lambda te: X[key][te]

    def k1(tr):
        p, y = X[key][tr], y_all[tr]
        m = np.isfinite(p) & np.isfinite(y)
        c = float((p[m] @ y[m]) / (p[m] @ p[m]))
        return dict(c=c), lambda te: c * X[key][te]

    def k2(tr):
        c = _ols(np.c_[np.ones(len(tr)), X[key][tr]], y_all[tr])
        return dict(a=c[0], c=c[1]), lambda te: c[0] + c[1] * X[key][te]
    return k0, k1, k2


F5k0, F5k1, F5k2 = _scale_family("p2")
F6k0, F6k1, F6k2 = _scale_family("p4")


def fit_F7k1(tr):
    s, y = X["s_soil"][tr], y_all[tr]
    m = np.isfinite(s) & np.isfinite(y) & (s > 0)
    E = float((s[m] @ y[m]) / (s[m] @ s[m]))
    return dict(E=E), lambda te: E * X["s_soil"][te]


def fit_F7k2(tr):
    c = _ols(np.c_[np.ones(len(tr)), X["s_soil"][tr]], y_all[tr])
    return dict(a=c[0], E=c[1]), lambda te: c[0] + c[1] * X["s_soil"][te]


def _two_layer(h, Em, tdd, fom_rel, rho=0.5):
    """유기층 두께 h_o = h·(f_om/median_train) [cm], E_o = rho·E_m. 유기층 관통 도일 TDD_o=(h_o/E_o)^2."""
    Eo = rho * Em
    h_o = h * fom_rel
    tdd_o = (h_o / max(Eo, 1e-6)) ** 2
    shallow = Eo * np.sqrt(np.clip(tdd, 0, None))
    deep = h_o + Em * np.sqrt(np.clip(tdd - tdd_o, 0, None))
    return np.where(tdd <= tdd_o, shallow, deep)


def fit_F8k2(tr):
    t, y, f = X["tdd"][tr], y_all[tr], X["f_om"][tr]
    med = float(np.nanmedian(f))
    m = np.isfinite(t) & np.isfinite(y) & np.isfinite(f)
    E0 = float((np.sqrt(t[m]) @ y[m]) / (np.sqrt(t[m]) @ np.sqrt(t[m])))
    res = least_squares(lambda p: _two_layer(p[0], p[1], t[m], f[m] / med) - y[m],
                        x0=[10.0, E0], bounds=([0.0, 0.1], [100.0, 10.0]))
    h, Em = map(float, res.x)
    return dict(h=h, E=Em, fom_med=med), lambda te: _two_layer(h, Em, X["tdd"][te], X["f_om"][te] / med)


def fit_F9k2(tr):
    c = _ols(np.c_[X["s"][tr], np.sqrt(X["fdd"][tr])], y_all[tr])
    return dict(E=c[0], gamma=c[1]), lambda te: c[0] * X["s"][te] + c[1] * np.sqrt(X["fdd"][te])


def fit_F10k2(tr):
    c = _ols(np.c_[X["s"][tr], X["s"][tr] * X["swe"][tr]], y_all[tr])
    E, Eg = float(c[0]), float(c[1])
    return dict(E=E, gamma=Eg / E if E else np.nan), lambda te: E * X["s"][te] + Eg * X["s"][te] * X["swe"][te]


FORMULAS = [  # (id, 계열, k, 적합기, 설명)
    ("F1k1", "F1 Stefan", 1, fit_F1k1, "E·√TDD"),
    ("F1k2", "F1 Stefan", 2, fit_F1k2, "a+E·√TDD"),
    ("F2k2", "F2 멱법칙", 2, fit_F2k2, "a·TDD^b"),
    ("F3k2", "F3 TDD 선형", 2, fit_F3k2, "a+b·TDD"),
    ("F4k2", "F4 MAAT 선형", 2, fit_F4k2, "a+b·MAAT"),
    ("F5k0", "F5 edaphic", 0, F5k0, "pedotransfer raw"),
    ("F5k1", "F5 edaphic", 1, F5k1, "c·edaphic"),
    ("F5k2", "F5 edaphic", 2, F5k2, "a+c·edaphic"),
    ("F6k0", "F6 Kudryavtsev", 0, F6k0, "raw"),
    ("F6k1", "F6 Kudryavtsev", 1, F6k1, "c·Ku"),
    ("F6k2", "F6 Kudryavtsev", 2, F6k2, "a+c·Ku"),
    ("F7k1", "F7 지표강제 Stefan", 1, fit_F7k1, "E·√TDD_stl1"),
    ("F7k2", "F7 지표강제 Stefan", 2, fit_F7k2, "a+E·√TDD_stl1"),
    ("F8k2", "F8 2층 Stefan", 2, fit_F8k2, "유기층(h·f_om) + E_m√(TDD−TDD_o)"),
    ("F9k2", "F9 Stefan+FDD", 2, fit_F9k2, "E·√TDD+γ·√FDD"),
    ("F10k2", "F10 Stefan×눈", 2, fit_F10k2, "E·√TDD·(1+γ·SWE)"),
]


# ---------------------------------------------------------------- 채점
def paired_boot(y, p, p_ref, blocks, n_boot=NBOOT):
    """같은 셀에서 RMSE(p) − RMSE(p_ref)의 0.5° 블록 짝지은 부트스트랩 95% CI."""
    ub = np.unique(blocks)
    if len(ub) < 3:
        return np.nan, np.nan
    out = []
    for _ in range(n_boot):
        pick = RNG.choice(ub, len(ub), replace=True)
        idx = np.concatenate([np.where(blocks == b)[0] for b in pick])
        out.append(np.sqrt(np.mean((y[idx] - p[idx]) ** 2)) - np.sqrt(np.mean((y[idx] - p_ref[idx]) ** 2)))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def score(axis, region, fid, fam, k, desc, te, pred, pred_ref):
    y = y_all[te]
    m = np.isfinite(pred) & np.isfinite(y) & np.isfinite(pred_ref)
    if m.sum() < 3:
        return dict(axis=axis, region=region, formula=fid, family=fam, k=k, desc=desc, n=int(m.sum()))
    mt = all_metrics(y[m], pred[m])
    mr = all_metrics(y[m], pred_ref[m])
    lo, hi = paired_boot(y[m], pred[m], pred_ref[m], blocks_all[te][m])
    return dict(axis=axis, region=region, formula=fid, family=fam, k=k, desc=desc, n=int(m.sum()),
                rmse_cm=mt["rmse_cm"], mae_cm=mt["mae_cm"], bias_cm=mt["bias_cm"], r2=mt["r2"],
                skill=mt["skill_over_mean"], rmse_stefan_paired=mr["rmse_cm"],
                delta_vs_stefan=mt["rmse_cm"] - mr["rmse_cm"], ci_lo=lo, ci_hi=hi)


rows, coefs = [], []
t0 = time.time()

# --- 지역 내(알래스카 6-fold) ---
ak = np.where(df.macro.values == "Alaska")[0]
folds = spatial_block_splits(df, n_splits=6, sub_idx=ak)
oof = {fid: np.full(len(df), np.nan) for fid, *_ in FORMULAS}
for fi, (tr, te) in enumerate(folds):
    for fid, fam, k, fitter, desc in FORMULAS:
        c, pred_fn = fitter(tr)
        oof[fid][te] = pred_fn(te)
        coefs.append(dict(axis="indomain_AK", fold=fi, region="Alaska", formula=fid, **c))
for fid, fam, k, _, desc in FORMULAS:
    rows.append(score("indomain_AK", "Alaska", fid, fam, k, desc, ak, oof[fid][ak], oof["F1k1"][ak]))
print(f"[indomain] 완료 {time.time()-t0:.0f}s")

# 식 간 상관(지역 내 OOF, 전부 유한한 셀)
P = pd.DataFrame({fid: oof[fid][ak] for fid, *_ in FORMULAS})
ok = P.notna().all(axis=1)
corr_p = P[ok].corr(method="pearson").round(3)
corr_s = P[ok].corr(method="spearman").round(3)
corr = pd.concat([corr_p.assign(method="pearson"), corr_s.assign(method="spearman")])
corr.to_csv(PROC / f"e2_physics_corr{TAG}.csv")
print(f"[corr] 공통 유한 셀 {int(ok.sum()):,} · F1k1 대비 Pearson: "
      + " ".join(f"{c}={corr_p.loc['F1k1', c]:.2f}" for c in corr_p.columns if c != 'F1k1'))

# --- 전이(LORO) ---
per_region = {}
for region, tr, te in loro_splits(df, min_test=args.min_test):
    if REGION_FILTER and region not in REGION_FILTER:
        continue
    preds = {}
    for fid, fam, k, fitter, desc in FORMULAS:
        c, pred_fn = fitter(tr)
        preds[fid] = pred_fn(te)
        coefs.append(dict(axis="LORO", fold=-1, region=region, formula=fid, **c))
    for fid, fam, k, _, desc in FORMULAS:
        r = score("LORO", region, fid, fam, k, desc, te, preds[fid], preds["F1k1"])
        rows.append(r)
        per_region.setdefault(fid, []).append(r.get("rmse_cm", np.nan))
for fid, fam, k, _, desc in FORMULAS:
    v = per_region.get(fid, [])
    rows.append(dict(axis="LORO_gate", region="UNWEIGHTED_MEAN", formula=fid, family=fam, k=k, desc=desc,
                     n=len(v), rmse_cm=float(np.nanmean(v)) if v else np.nan))
print(f"[LORO] 완료 {time.time()-t0:.0f}s")

res = pd.DataFrame(rows)
res.to_csv(PROC / f"e2_physics_ladder_results{TAG}.csv", index=False)
pd.DataFrame(coefs).to_csv(PROC / f"e2_physics_ladder_coefs{TAG}.csv", index=False)

pd.set_option("display.width", 220)
show = ["formula", "k", "n", "rmse_cm", "bias_cm", "r2", "rmse_stefan_paired", "delta_vs_stefan", "ci_lo", "ci_hi"]
print("\n=== 지역 내(알래스카 6-fold) ===")
print(res[res.axis == "indomain_AK"][show].round(2).to_string(index=False))
print("\n=== 전이 LORO 지역별 ===")
piv = res[res.axis == "LORO"].pivot_table(index=["formula", "k"], columns="region", values="rmse_cm").round(2)
piv["mean3"] = res[res.axis == "LORO_gate"].set_index("formula").loc[piv.index.get_level_values(0), "rmse_cm"].values.round(2)
print(piv.to_string())
cb = pd.DataFrame(coefs)
b = cb[(cb.formula == "F2k2")][["axis", "region", "fold", "b", "b_log"]]
print("\n=== H5 멱지수 b (F2k2) ===")
print(b.round(3).to_string(index=False))

(PROC / f"e2_physics_ladder_meta{TAG}.json").write_text(json.dumps(dict(
    stage="E2", n_cells=int(len(df)), formulas=[dict(id=f, family=fam, k=k, desc=d) for f, fam, k, _, d in FORMULAS],
    indomain="Alaska 0.5° block 6-fold, coefs fitted on train fold only",
    loro=[r for r, _, _ in loro_splits(df, min_test=args.min_test) if not REGION_FILTER or r in REGION_FILTER], base=args.base, exclude=EXCLUDE, min_test=args.min_test,
    paired_reference="F1k1 (Stefan E least squares) on identical cells; block paired bootstrap 400",
    inputs=["fidelity_base.csv", "e5_soil_tdd.csv"],
    plan="docs/EXPERIMENT_PLAN_PAPER_2026-09-08.md §2, H4·H5",
    elapsed_s=round(time.time() - t0, 1),
), ensure_ascii=False, indent=1))
print(f"\nsaved: e2_physics_ladder_results.csv · e2_physics_ladder_coefs.csv · e2_physics_corr.csv  ({time.time()-t0:.0f}s)")
