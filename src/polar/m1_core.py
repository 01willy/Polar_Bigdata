"""M1 마스터 요인 실험 공용 코어 — 데이터·분할·계수·앵커·유사라벨·모델 (계획 docs/EXPERIMENT_PLAN_MASTER_2026-09-16.md).

E1 하네스(scripts/3_deep_learning/e1_unified_factorial.py)를 일반화한다. 모든 축은 여기서 정의하고
하네스(m1_master_factorial.py)는 열거·실행·저장만 한다.

축과 수준
  C 조건    labels / covonly / noinfo / deploy
  D 학습집합 alaska / loro(알래스카+대상 외 전 지역) / loro_main(알래스카+대상 외 주 집합)
  X 입력    x25(SHARED_CORE) / x14(지형+기후) / x34(COVARIATE_CORE, 알래스카 지역 내만)
  P 유사라벨 none / stefan / stefan_k2 / stefan_soil / ku_cal / ku_k2 / edaphic_k2 / tddlin / const / const_t / shuffle / cci
  K 앵커    none / stefan / stefan_k2 / cci_cal / cci_raw / stefan_cci / stefan_cci_cal / stefan_soil / stefan_soil_cci /
            ku_cal / stefan_ku / stefan_ku_cci / emap_ridge / emap_cb
  G 모델    ridge / catboost_lo / catboost / mlp / ftt / tabm / realmlp / cfm / ddpm / nflow  (앙상블은 분석 단계에서 g 평균)
  λ        사후 스윕 {0, .25, .5, .75, 1}
누설 규약: 계수·E 지도·표준화 통계는 학습 실측(라벨 있음이면 A블록 실측 포함)에서만. 대상 라벨은 채점 전용.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from polar.fidelity import (SHARED_CORE, COVARIATE_CORE, TERRAIN, CLIMATE, SOIL, TARGET,
                            TRANSFER_MAIN, TRANSFER_DEEP, add_group_keys, macro_region, spatial_block_splits)
from polar.preprocessing import fold_prep
from polar.physics import physics_ensemble

INPUT_SETS = {"x25": list(SHARED_CORE), "x14": list(TERRAIN + CLIMATE), "x34": list(COVARIATE_CORE),
              "x17": list(CLIMATE + SOIL)}
ANCHORS_ALL = ["none", "stefan", "stefan_med", "stefan_k2", "cci_cal", "cci_raw", "stefan_cci", "stefan_cci_cal",
               "stefan_soil", "stefan_soil_cci", "ku_cal", "stefan_ku", "stefan_ku_cci", "emap_ridge", "emap_cb"]
PSEUDOS_ALL = ["none", "stefan", "stefan_k2", "stefan_soil", "ku_cal", "ku_k2", "edaphic_k2", "tddlin",
               "const", "const_t", "shuffle", "cci"]
MODELS_ALL = ["ridge", "catboost_lo", "catboost", "mlp", "ftt", "tabm", "realmlp", "cfm", "ddpm", "nflow"]
NAN_NATIVE = {"catboost_lo", "catboost"}
LAM_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
EMAP_FEATS = list(SOIL + CLIMATE)      # 물리 계수 지도 입력(토양 9 + 기후 8)


# ---------------------------------------------------------------- 자료
def load_base(proc, base="fidelity_base_v3.csv", soil="e5_soil_tdd_v3.csv", sources=("F4_direct",)):
    """fidelity_base + 토양 도일 병합, 매크로 지역, Ku 원식(p4, 라벨 미사용) 부착. source_id 필터."""
    df = add_group_keys(pd.read_csv(proc / base, low_memory=False))
    sd = pd.read_csv(proc / soil)
    s1 = sd[sd.loc_id >= 0][["loc_id", "e5_sqrt_tdd_soil", "e5_tdd_soil"]]
    df = df.merge(s1, on="loc_id", how="left")
    if df.e5_sqrt_tdd_soil.isna().any():                    # 신규 CALM 셀: 좌표 병합
        s2 = sd[sd.loc_id < 0][["lat", "lon", "e5_sqrt_tdd_soil", "e5_tdd_soil"]].copy()
        s2["klat"], s2["klon"] = s2.lat.round(4), s2.lon.round(4)
        df["klat"], df["klon"] = df.lat.round(4), df.lon.round(4)
        df = df.merge(s2[["klat", "klon", "e5_sqrt_tdd_soil", "e5_tdd_soil"]].rename(
            columns={"e5_sqrt_tdd_soil": "_a", "e5_tdd_soil": "_b"}), on=["klat", "klon"], how="left")
        df["e5_sqrt_tdd_soil"] = df.e5_sqrt_tdd_soil.fillna(df["_a"])
        df["e5_tdd_soil"] = df.e5_tdd_soil.fillna(df["_b"])
        df = df.drop(columns=["klat", "klon", "_a", "_b"])
    df["macro"] = macro_region(df)
    df = df[df[TARGET].notna() & df.source_id.isin(list(sources))].reset_index(drop=True)
    pe = physics_ensemble(df, E=1.0)                        # 공변량 전용(라벨 미사용). _fallback은 전체 중앙값(공변량 통계, 라벨 아님)
    df["p4_ku"] = pe["p4_kudryavtsev"]
    df["p2_edaphic"] = pe["p2_edaphic"]
    return df


def eval_mask(d: pd.DataFrame) -> np.ndarray:
    """평가 셀 = y 유효 ∩ CCI 유효 ∩ 토양 도일 유효 (모든 방법 동일 셀)."""
    return (d[TARGET].notna() & d.cci_alt.notna() & d.e5_sqrt_tdd_soil.notna()).values


# ---------------------------------------------------------------- 분할
def half_split_blocks(df: pd.DataFrame, idx: np.ndarray, split_seed: int):
    """대상 지역 셀 idx를 0.5° 블록 단위로 A/B 2분할. split_seed=0 은 E1·S3·S12 규약(GroupKFold n=2, fold0:
    A=te, B=tr)과 동일. split_seed>0 은 블록을 무작위 순열 후 셀 수 균형 탐욕 배정(반복 분할 분산 추정용)."""
    if split_seed == 0:
        folds = spatial_block_splits(df, n_splits=2, sub_idx=idx)
        return folds[0][1], folds[0][0]
    blocks = df["block"].values[idx]
    ub, cnt = np.unique(blocks, return_counts=True)
    rng = np.random.RandomState(split_seed)
    order = rng.permutation(len(ub))
    A, B, nA, nB = [], [], 0, 0
    for j in order:                                          # 셀 수가 적은 쪽에 배정(균형)
        if nA <= nB:
            A.append(ub[j]); nA += cnt[j]
        else:
            B.append(ub[j]); nB += cnt[j]
    inA = np.isin(blocks, A)
    return idx[inA], idx[~inA]


def train_region_set(df: pd.DataFrame, target: str, dset: str) -> np.ndarray:
    """학습 지역 집합 D. 대상 지역은 항상 제외."""
    m = df.macro.values
    if dset == "alaska":
        return np.where(m == "Alaska")[0]
    if dset == "loro":
        return np.where(m != target)[0]
    if dset == "loro_main":
        keep = set(["Alaska"] + [r for r in TRANSFER_MAIN if r != target])
        return np.where(np.isin(m, list(keep)))[0]
    raise ValueError(dset)


# ---------------------------------------------------------------- 계수 (학습 실측 라벨에서만)
def _ls_scale(p, y):
    m = np.isfinite(y) & np.isfinite(p) & (p > 0)
    return float((p[m] @ y[m]) / (p[m] @ p[m])) if m.sum() >= 3 else np.nan


def _ls_affine(p, y):
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 3:
        return np.nan, np.nan
    b, a = np.polyfit(p[m], y[m], 1)
    return float(a), float(b)


def fit_coefs(train: pd.DataFrame, emap: bool = False) -> dict:
    y = train[TARGET].values.astype(float)
    s = train["e5_sqrt_tdd"].values.astype(float)
    ss = train["e5_sqrt_tdd_soil"].values.astype(float)
    c = train["cci_alt"].values.astype(float)
    t = train["e5_tdd"].values.astype(float)
    p4 = train["p4_ku"].values.astype(float)
    p2 = train["p2_edaphic"].values.astype(float)
    mm = np.isfinite(y) & np.isfinite(s) & (s > 0)
    k = dict(E=_ls_scale(s, y), E_med=float(np.median(y[mm] / s[mm])) if mm.sum() >= 3 else np.nan,
             E_soil=_ls_scale(ss, y), c_ku=_ls_scale(p4, y), c_ed=_ls_scale(p2, y),
             ymean=float(np.nanmean(y)), n_train_real=int(np.isfinite(y).sum()))
    k["st_a"], k["st_E"] = _ls_affine(s, y)
    k["cci_a"], k["cci_b"] = _ls_affine(c, y)
    k["tdd_a"], k["tdd_b"] = _ls_affine(t, y)
    k["ku_a"], k["ku_b"] = _ls_affine(p4, y)
    k["ed_a"], k["ed_b"] = _ls_affine(p2, y)
    if emap:
        k["emap"] = fit_emap(train)
    return k


def fit_emap(train: pd.DataFrame, min_cells: int = 3) -> dict:
    """T0 물리 계수 지도: 학습 실측의 0.5° 블록별 E(최소제곱) → 블록 평균 공변량(토양 9 + 기후 8)으로 회귀.
    ridge(α=10, 표준화)와 catboost_lo 둘 다 적합. 블록 수를 기록."""
    d = train[np.isfinite(train[TARGET].values) & (train.e5_sqrt_tdd.values > 0)]
    g = d.groupby("block")
    rows = []
    for b, sub in g:
        if len(sub) < min_cells:
            continue
        y, s = sub[TARGET].values.astype(float), sub.e5_sqrt_tdd.values.astype(float)
        rows.append(dict(block=b, E=float((s @ y) / (s @ s)), n=len(sub),
                         **{f: float(np.nanmean(sub[f].values.astype(float))) for f in EMAP_FEATS}))
    bt = pd.DataFrame(rows)
    X = bt[EMAP_FEATS].values.astype(float)
    med = np.nanmedian(X, 0); med = np.where(np.isfinite(med), med, 0.0)
    X = np.where(np.isnan(X), med, X)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    from sklearn.linear_model import Ridge
    rd = Ridge(alpha=10.0).fit((X - mu) / sd, bt.E.values, sample_weight=np.sqrt(bt.n.values))
    out = dict(n_blocks=int(len(bt)), E_blocks_sd=float(bt.E.std()), med=med, mu=mu, sd=sd, ridge=rd,
               E_lo=float(np.percentile(bt.E, 2)), E_hi=float(np.percentile(bt.E, 98)))
    try:
        from catboost import CatBoostRegressor
        cb = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=3, l2_leaf_reg=3.0, random_seed=0,
                               verbose=0, allow_writing_files=False, thread_count=4)
        cb.fit(bt[EMAP_FEATS].values.astype(float), bt.E.values, sample_weight=np.sqrt(bt.n.values))
        out["cb"] = cb
    except Exception:                                        # noqa: BLE001
        out["cb"] = None
    return out


def emap_predict(em: dict, d: pd.DataFrame, kind: str) -> np.ndarray:
    X = d[EMAP_FEATS].values.astype(float)
    if len(X) == 0:                                          # 평가 셀 0개(그린란드 A/B 등): sklearn predict가 0행을 거부
        return np.zeros(0)
    if kind == "emap_ridge" or em.get("cb") is None:
        Xz = (np.where(np.isnan(X), em["med"], X) - em["mu"]) / em["sd"]
        E = em["ridge"].predict(Xz)
    else:
        E = em["cb"].predict(X)
    return np.clip(E, em["E_lo"], em["E_hi"])               # 학습 블록 E의 2–98% 범위로 클립(외삽 폭주 방지)


# ---------------------------------------------------------------- 앵커·유사라벨
def anchor_pred(kind: str, d: pd.DataFrame, k: dict):
    if kind == "none":
        return None
    s = d["e5_sqrt_tdd"].values.astype(float)
    ss = d["e5_sqrt_tdd_soil"].values.astype(float)
    c = d["cci_alt"].values.astype(float)
    st = k["E"] * s
    st_soil = k["E_soil"] * ss
    cci_cal = k["cci_a"] + k["cci_b"] * c
    ku = k["c_ku"] * d["p4_ku"].values.astype(float)
    ku = np.where(np.isfinite(ku), ku, st)                   # Ku 비동토 판정 셀은 Stefan 대체
    table = {
        "stefan": st, "stefan_med": k["E_med"] * s, "stefan_k2": k["st_a"] + k["st_E"] * s, "cci_cal": cci_cal, "cci_raw": c,
        "stefan_cci": 0.5 * (st + c), "stefan_cci_cal": 0.5 * (st + cci_cal),
        "stefan_soil": st_soil, "stefan_soil_cci": 0.5 * (st_soil + c),
        "ku_cal": ku, "stefan_ku": 0.5 * (st + ku), "stefan_ku_cci": (st + ku + c) / 3.0,
    }
    if kind in table:
        return table[kind]
    if kind in ("emap_ridge", "emap_cb"):
        return emap_predict(k["emap"], d, kind) * s
    raise ValueError(kind)


def pseudo_label(kind: str, d: pd.DataFrame, k: dict, rng: np.random.RandomState, pool_stefan_mean: float | None = None):
    s = d["e5_sqrt_tdd"].values.astype(float)
    st = k["E"] * s
    if kind == "stefan":
        return st
    if kind == "stefan_k2":
        return k["st_a"] + k["st_E"] * s
    if kind == "stefan_soil":
        return k["E_soil"] * d["e5_sqrt_tdd_soil"].values.astype(float)
    if kind == "ku_cal":
        ku = k["c_ku"] * d["p4_ku"].values.astype(float)
        return np.where(np.isfinite(ku), ku, st)
    if kind == "ku_k2":
        ku = k["ku_a"] + k["ku_b"] * d["p4_ku"].values.astype(float)
        return np.where(np.isfinite(ku), ku, st)
    if kind == "edaphic_k2":
        return k["ed_a"] + k["ed_b"] * d["p2_edaphic"].values.astype(float)
    if kind == "tddlin":
        return k["tdd_a"] + k["tdd_b"] * d["e5_tdd"].values.astype(float)
    if kind == "const":                                      # 알래스카(학습) 평균 상수
        return np.full(len(d), k["ymean"])
    if kind == "const_t":                                    # 대상 유사라벨 풀의 Stefan 평균 상수(수준만 이식, 대안 설명 배제용)
        return np.full(len(d), float(np.nanmean(st)) if pool_stefan_mean is None else pool_stefan_mean)
    if kind == "shuffle":
        return rng.permutation(st)
    if kind == "cci":
        return d["cci_alt"].values.astype(float)
    raise ValueError(kind)


# ---------------------------------------------------------------- 모델
def fit_model(name: str, Xtr, ytr, Xte, seed: int, epochs: int = 100, sample_weight=None):
    """반환 dict(pred=..., [q05, q95, pred_mean]). sample_weight는 중요도 가중(T3) 재표집으로 모델 무관 적용."""
    if sample_weight is not None:
        w = np.asarray(sample_weight, float); w = w / w.sum()
        rs = np.random.RandomState(seed).choice(len(Xtr), len(Xtr), replace=True, p=w)
        Xtr, ytr = Xtr[rs], ytr[rs]
    if name == "ridge":
        from sklearn.linear_model import Ridge
        m = Ridge(alpha=10.0).fit(np.nan_to_num(Xtr), ytr)
        return dict(pred=m.predict(np.nan_to_num(Xte)))
    if name == "catboost_lo":
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=3, l2_leaf_reg=3.0,
                              random_seed=seed, verbose=0, allow_writing_files=False, thread_count=6)
        m.fit(Xtr, ytr)
        return dict(pred=np.asarray(m.predict(Xte)))
    if name == "catboost":
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(iterations=600, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
                              random_seed=seed, verbose=0, allow_writing_files=False, thread_count=6)
        m.fit(Xtr, ytr)
        return dict(pred=np.asarray(m.predict(Xte)))
    from polar.tab_models import fit_predict
    out = fit_predict(name, Xtr, ytr, Xte, seed=seed, epochs=epochs)
    res = dict(pred=np.asarray(out["pred"]))
    if "quantiles" in out:
        res["q05"], res["q95"] = np.asarray(out["quantiles"][0]), np.asarray(out["quantiles"][2])
    if "pred_mean" in out:
        res["pred_mean"] = np.asarray(out["pred_mean"])
    return res


def importance_weights(Xsrc, Xtgt, seed: int = 0, clip: float = 10.0) -> np.ndarray:
    """T3 중요도 가중: 학습 대 대상 공변량 로지스틱 분류기 → 밀도비 w(x)=p_t(x)/p_s(x) (클립)."""
    from sklearn.linear_model import LogisticRegression
    X = np.vstack([Xsrc, Xtgt]); yy = np.r_[np.zeros(len(Xsrc)), np.ones(len(Xtgt))]
    med = np.nanmedian(X, 0); med = np.where(np.isfinite(med), med, 0.0); X = np.where(np.isnan(X), med, X)
    mu, sd = X.mean(0), X.std(0) + 1e-6; X = (X - mu) / sd
    clf = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=seed).fit(X, yy)
    p = clf.predict_proba(X[:len(Xsrc)])[:, 1]
    w = p / np.clip(1 - p, 1e-6, None)
    return np.clip(w / np.mean(w), 1.0 / clip, clip)
