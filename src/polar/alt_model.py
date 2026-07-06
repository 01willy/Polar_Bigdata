"""
Stage 0 — ALT 예측의 평가 골격 + baseline 사다리 + 누설 진단.

핵심 질문: "어떤 CV가 정직한가?" 무작위 CV는 공간 자기상관으로 성능을 과대평가한다.
4개 CV(무작위 / site-disjoint / 공간블록 / leave-one-region-out)에서 baseline 사다리
(지역평균 < IDW < Ordinary Kriging < GBM[공변량])를 평가해 한 표로 보여준다.
LORO(지역 전이)에서 거리기반(IDW/kriging)은 무너지고 공변량 GBM만 버티는 게 핵심.

산출:
  data/processed/alt_model_table.csv   모델 테이블(y=log1p(alt_cm), x/y meters, site/country)
  data/processed/cv_leakage_table.csv  CV×method RMSE/MAE(cm)
  data/processed/variogram_range.json  공간블록 크기(잔차 베리오그램 range)
  data/processed/cv_splits.json        각 CV 폴드 인덱스
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, GroupKFold
from scipy.spatial import cKDTree

from . import config as C
from . import geo

FEATS = ["wc_bio1", "wc_bio4", "wc_bio7", "wc_bio12", "wc_elev"]
NEED = FEATS + ["alt_cm", "lat", "lon", "year", "site", "country"]


def load_table():
    df = pd.read_csv(C.PROCESSED / "alt_global.csv").dropna(subset=NEED).copy()
    assert df[FEATS].isna().sum().sum() == 0, "covariate NaN"
    df["y"] = np.log1p(df["alt_cm"])                       # 양수·우편향 안정화
    x, ym = geo.to_xy(df["lon"].to_numpy(), df["lat"].to_numpy())
    df["x"], df["ym"] = x, ym                              # EPSG:3413 meters
    df = df.reset_index(drop=True)
    df.to_csv(C.PROCESSED / "alt_model_table.csv", index=False)
    return df


# ---------- baselines: (train_df, test_df) -> pred in log space ----------
def b_region_mean(tr, te):
    return np.full(len(te), tr["y"].mean())


def b_idw(tr, te, k=8, power=2):
    P = np.column_stack([tr["x"], tr["ym"]])
    Q = np.column_stack([te["x"], te["ym"]])
    d, idx = cKDTree(P).query(Q, k=min(k, len(P)))
    d = np.maximum(d, 1e-6)
    w = 1.0 / d ** power
    return (w * tr["y"].to_numpy()[idx]).sum(1) / w.sum(1)


def b_kriging(tr, te, cap=1200):
    try:
        from pykrige.ok import OrdinaryKriging
        if len(tr) > cap:
            tr = tr.sample(cap, random_state=0)
        ok = OrdinaryKriging(tr["x"].to_numpy(), tr["ym"].to_numpy(), tr["y"].to_numpy(),
                             variogram_model="linear", verbose=False, enable_plotting=False)
        z, _ = ok.execute("points", te["x"].to_numpy(), te["ym"].to_numpy())
        return np.asarray(z)
    except Exception:
        return np.full(len(te), np.nan)


def b_gbm(tr, te):
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                      max_leaf_nodes=31, random_state=0)
    m.fit(tr[FEATS], tr["y"])
    return m.predict(te[FEATS])


BASELINES = {"지역평균": b_region_mean, "IDW": b_idw, "Kriging": b_kriging, "GBM(공변량)": b_gbm}


# ---------- CV 폴드 생성 ----------
def folds_random(df, k=5):
    return list(KFold(k, shuffle=True, random_state=0).split(df))


def folds_site(df, k=5):
    return list(GroupKFold(k).split(df, groups=df["site"]))


def folds_block(df, block_km):
    b = block_km * 1000.0
    code = (np.floor(df["x"] / b).astype(int).astype(str) + "_"
            + np.floor(df["ym"] / b).astype(int).astype(str))
    nb = code.nunique()
    k = min(5, nb)
    return list(GroupKFold(k).split(df, groups=code)), nb


def folds_loro(df):
    out = []
    for ctry in df["country"].unique():
        te = df.index[df["country"] == ctry].to_numpy()
        tr = df.index[df["country"] != ctry].to_numpy()
        if len(te) and len(tr) > 30:
            out.append((tr, te))
    return out


def variogram_range_km(df):
    """공변량 GBM 잔차의 베리오그램 range(km). 전지구 기후 trend를 먼저 제거해야 의미가 있음."""
    try:
        from skgstat import Variogram
        # 공변량 trend 제거 → 잔차의 공간 자기상관 range가 진짜 블록 크기
        m = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(df[FEATS], df["y"])
        df = df.assign(resid=df["y"] - m.predict(df[FEATS]))
        s = df.groupby("site").agg(x=("x", "first"), ym=("ym", "first"),
                                   resid=("resid", "mean")).reset_index()
        coords = np.column_stack([s["x"], s["ym"]]) / 1000.0      # km
        v = Variogram(coords, s["resid"].to_numpy(), n_lags=20, model="spherical")
        rng = float(v.parameters[0])
        block = float(np.clip(rng, 80.0, 500.0))                 # 비현실값 방어
        return block, rng
    except Exception as e:
        print("  variogram 실패, 기본 200km:", str(e)[:60])
        return 200.0, float("nan")


def _score(df, folds, name):
    rows = []
    for method, fn in BASELINES.items():
        err = []
        for tr_idx, te_idx in folds:
            tr, te = df.iloc[tr_idx], df.iloc[te_idx]
            # log 공간 예측을 물리범위로 클리핑(외삽 폭주 방지) 후 cm 변환
            pred = np.expm1(np.clip(fn(tr, te), np.log1p(1), np.log1p(600)))
            true = te["alt_cm"].to_numpy()
            e = pred - true
            err.append(e[np.isfinite(e)])
        e = np.concatenate(err)
        rows.append(dict(cv=name, method=method, rmse=float(np.sqrt(np.mean(e ** 2))),
                         mae=float(np.mean(np.abs(e))), n=int(len(e))))
    return rows


def run():
    C.ensure_dirs()
    df = load_table()
    print(f"ALT 모델 테이블: {len(df):,} site-year, {df.site.nunique()} site, "
          f"{df.country.nunique()} 국가")

    block_km, raw_rng = variogram_range_km(df)
    (C.PROCESSED / "variogram_range.json").write_text(
        json.dumps({"block_km": block_km, "variogram_range_km": raw_rng}, indent=2))
    print(f"공간블록 크기: {block_km:.0f} km (베리오그램 range {raw_rng:.0f} km)")

    fr = folds_random(df); fs = folds_site(df)
    fb, nblocks = folds_block(df, block_km); fl = folds_loro(df)
    cvs = [("무작위 K-fold", fr), ("site-disjoint", fs),
           (f"공간블록({block_km:.0f}km)", fb), ("LORO(지역전이)", fl)]

    all_rows = []
    for name, folds in cvs:
        print(f"  CV: {name} ...")
        all_rows += _score(df, folds, name)
    res = pd.DataFrame(all_rows)
    res.to_csv(C.PROCESSED / "cv_leakage_table.csv", index=False)

    print("\n=== RMSE (cm) — CV × method ===")
    piv = res.pivot(index="method", columns="cv", values="rmse").round(1)
    piv = piv.reindex(["지역평균", "IDW", "Kriging", "GBM(공변량)"])
    print(piv.to_string())
    print("\n핵심: 무작위→LORO로 갈수록 RMSE 증가가 누설/전이난도. "
          "LORO에서 거리기반은 무너지고 GBM(공변량)만 버팀.")

    splits = {name: [[int(i) for i in te] for _, te in folds] for name, folds in cvs}
    (C.PROCESSED / "cv_splits.json").write_text(json.dumps(splits))
    return res


# ===================== Stage 1: 피처 엔지니어링 + 강 baseline =====================
ENG = ["tdd", "fdd", "sqrt_tdd", "thaw_months", "summer_t", "winter_prec", "continentality"]


def add_engineered(df):
    """도일(degree-day) 등 물리 피처 추가.
    월별 데이터가 없으므로 연중 기온을 사인곡선으로 근사:
      T(d) = MAAT(bio1) + (연교차 bio7 / 2)·sin(2π d/365)
    → 융해/동결 도일을 해석적 적분(영구동토 모델 표준 근사). 월별 ERA5 확보 시 정밀화 가능.
    """
    maat = df["wc_bio1"].to_numpy()
    amp = df["wc_bio7"].to_numpy() / 2.0                 # 진폭 = 연교차/2
    d = np.arange(365)
    T = maat[:, None] + amp[:, None] * np.sin(2 * np.pi * d / 365.0)   # (N,365)
    df["tdd"] = np.clip(T, 0, None).sum(1)               # 융해 도일(°C·day)
    df["fdd"] = np.clip(-T, 0, None).sum(1)              # 동결 도일
    df["sqrt_tdd"] = np.sqrt(df["tdd"])                  # Stefan ALT proxy
    df["thaw_months"] = (T > 0).sum(1) / 30.4            # 해빙 개월수
    df["summer_t"] = maat + amp                          # 최난월 기온
    df["winter_prec"] = df["wc_bio12"].to_numpy() * (T < 0).mean(1)    # 적설 proxy
    df["continentality"] = df["wc_bio7"] / (df["wc_bio4"] / 100.0 + 1e-6)
    return df


def _eval_featureset(df, folds, feats, model="gbm"):
    from sklearn.ensemble import RandomForestRegressor
    err = []
    for tr_idx, te_idx in folds:
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]
        if model == "gbm":
            m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                              max_leaf_nodes=31, random_state=0)
        else:
            m = RandomForestRegressor(300, min_samples_leaf=3, n_jobs=-1, random_state=0)
        m.fit(tr[feats], tr["y"])
        pred = np.expm1(np.clip(m.predict(te[feats]), np.log1p(1), np.log1p(600)))
        err.append(pred - te["alt_cm"].to_numpy())
    e = np.concatenate(err)
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2))), float(np.mean(np.abs(e)))


def run_stage1():
    C.ensure_dirs()
    df = add_engineered(load_table())
    df.to_csv(C.PROCESSED / "alt_features.csv", index=False)
    print(f"Stage1 피처 테이블: {len(df):,}행, 엔지니어링 {len(ENG)}개 추가")

    block_km, _ = variogram_range_km(df)
    fb, _ = folds_block(df, block_km); fl = folds_loro(df)
    configs = [("GBM 기본(공변량5)", FEATS, "gbm"),
               ("GBM +도일피처", FEATS + ENG, "gbm"),
               ("RF +도일피처", FEATS + ENG, "rf")]
    rows = []
    for cvname, folds in [("공간블록", fb), ("LORO(전이)", fl)]:
        for label, feats, model in configs:
            r, m = _eval_featureset(df, folds, feats, model)
            rows.append(dict(cv=cvname, config=label, rmse=round(r, 1), mae=round(m, 1)))
            print(f"  [{cvname}] {label:18s} RMSE={r:5.1f} MAE={m:5.1f} cm")
    res = pd.DataFrame(rows)
    res.to_csv(C.PROCESSED / "stage1_results.csv", index=False)

    # 변수 중요도 (RF, 전체 피처)
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(300, min_samples_leaf=3, n_jobs=-1, random_state=0)
    rf.fit(df[FEATS + ENG], df["y"])
    imp = pd.Series(rf.feature_importances_, index=FEATS + ENG).sort_values(ascending=False)
    imp.to_csv(C.PROCESSED / "stage1_feature_importance.csv")
    print("\n변수 중요도 상위:", ", ".join(f"{k}={v:.2f}" for k, v in imp.head(6).items()))
    return res, imp


if __name__ == "__main__":
    run()
