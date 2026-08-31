"""S14: 연별 ALT 예측 재설계 — 입력 집합과 물리 결합을 공정하게 비교한다.

배경
    기존 연별 실험(timelapse_alaska.py)은 세 모형(Stefan 절편포함·MLP·HistGBM)에
    "그해 기후 9종"만 입력으로 주었다. 그 조건에서는 물리식이 최저였다(14.97 대 16.27·19.19).
    그러나 4.1절의 지역 내 비교는 입력 34종을 쓰며 거기서는 학습 모형이 물리식을 앞선다.
    입력 집합이 다른 두 결과를 나란히 두면 "연별 지도에서만 물리식이 낫다"로 읽히는데,
    이는 모형의 성질이 아니라 입력 조건의 차이일 수 있다. 이 스크립트는 그 교란을 없앤다.

설계
    같은 (위치, 연도) 패널·같은 연도 홀드아웃에서 입력 집합만 바꿔 비교한다.
      A. year9    : 그해 기후 9종                       (기존 조건)
      B. year+stat: 그해 기후 9종 + 지형 6 + 토양 9 = 24종 (4.1절과 대등한 조건)
    모형은 다음 다섯이다.
      stefan   : ALT = a + E·sqrt(TDD_그해).  fold 안에서 절편 포함/미포함 중 train RMSE 낮은 쪽
      mlp / histgbm / catboost : 해당 입력 집합으로 log1p(ALT) 직접 회귀
      resid    : Stefan 앵커 + lam·저용량 잔차(능형). 4.5절과 같은 형태를 시간축에 적용
    잔차 가중 lam 은 {0.25, 0.5, 0.75, 1.0} 을 모두 보고한다(사후 선택 금지, 전량 기록).

평가 2축 (기존 실험과 동일 정의)
    temporal  : leave-one-year-out. 미관측 연도를 다른 해로 예측.
    anomaly   : 같은 지점의 해마다의 변동만 떼어 채점(지점 평균 제거). 시간 신호의 실제 예측력.

주의
    연도 홀드아웃은 같은 위치가 다른 해에 학습에 들어간다. 위치 평균을 외울 수 있으므로
    temporal 축 수치는 "공간 구조를 아는 상태에서의 연도 외삽"으로 읽어야 한다. 공간
    일반화는 4.1절 공간블록 축이 담당한다. 두 축을 섞지 않는다.

산출
    data/processed/s14_annual_results.csv   (입력집합 x 모형 x 축 metrics)
    data/processed/s14_annual_oof.csv       (채택 구성의 셀별 OOF, 지도 재생성용)
    data/processed/s14_annual_meta.json
실행
    GPU=9 PYTHONPATH=src python scripts/3_deep_learning/s14_annual_alt_redesign.py
"""
import os

GPU = os.environ.get("GPU", "9")
assert GPU in {"6", "7", "8", "9"}, f"GPU는 6,7,8,9만 허용(요청: {GPU})"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar import config as C                                   # noqa: E402
from polar.tab_models import available_models, fit_predict, NAN_NATIVE, set_device  # noqa: E402
from polar.preprocessing import fold_prep as prep               # noqa: E402
from polar.eval_metrics import all_metrics                      # noqa: E402

import torch                                                     # noqa: E402
set_device("cuda:0" if torch.cuda.is_available() else "cpu")

PROC = C.PROCESSED
SEEDS = [0, 1, 2]
LAMS = [0.25, 0.5, 0.75, 1.0]

YEAR9 = ["e5t_maat", "e5t_tdd", "e5t_fdd", "e5t_sqrt_tdd", "e5t_twarm",
         "e5t_tcold", "e5t_stl1", "e5t_swe", "e5t_swe_prevwinter"]
TERR = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough"]
SOIL = ["sg_clay_5_15", "sg_sand_5_15", "sg_silt_5_15", "sg_bdod_5_15", "sg_cfvo_5_15",
        "sg_phh2o_5_15", "sg_soc_0_5", "sg_soc_5_15", "sg_soc_15_30"]
STATIC = TERR + SOIL


# ---------------------------------------------------------------- 패널
def build_panel():
    """(위치, 연도) 패널 + 그해 forcing + 최근접 셀의 정적 공변량."""
    pt = pd.read_csv(PROC / "alt_above_pointlevel.csv")
    pt = pt.dropna(subset=["lat", "lon", "year", "alt_cm"]).copy()
    pt["year"] = pt["year"].astype(int)
    pt["key"] = pt["lat"].round(5).astype(str) + "_" + pt["lon"].round(5).astype(str)
    panel = (pt.groupby(["key", "year"])
             .agg(alt_cm=("alt_cm", "mean"), lat=("lat", "mean"), lon=("lon", "mean"))
             .reset_index())

    era = pd.read_csv(PROC / "alt_era5_temporal.csv")
    era["year"] = era["year"].astype(int)
    era["key"] = era["lat"].round(5).astype(str) + "_" + era["lon"].round(5).astype(str)
    era = era.drop_duplicates(["key", "year"])[["key", "year"] + YEAR9]
    df = panel.merge(era, on=["key", "year"], how="inner")

    # 정적 공변량은 최근접 셀에서 가져온다(패널은 점 단위, 공변량은 셀 단위).
    fb = pd.read_csv(PROC / "fidelity_base.csv", low_memory=False)
    fb = fb.dropna(subset=["lat", "lon"])
    from sklearn.neighbors import NearestNeighbors
    R = 6371.0
    def xyz(la, lo):
        la, lo = np.radians(la), np.radians(lo)
        return np.c_[R * np.cos(la) * np.cos(lo), R * np.cos(la) * np.sin(lo), R * np.sin(la)]
    nn = NearestNeighbors(n_neighbors=1).fit(xyz(fb.lat.values, fb.lon.values))
    dist, idx = nn.kneighbors(xyz(df.lat.values, df.lon.values))
    for c in STATIC:
        df[c] = fb[c].values[idx[:, 0]]
    df["static_dist_km"] = dist[:, 0]

    for c in YEAR9 + STATIC + ["alt_cm"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["e5t_sqrt_tdd", "alt_cm"]).reset_index(drop=True)
    return df


def fit_stefan(y, s):
    """절편 포함/미포함 중 train RMSE 낮은 쪽(기존 실험과 동일 규칙)."""
    m = np.isfinite(y) & np.isfinite(s)
    y, s = y[m], s[m]
    E0 = float((s * y).sum() / (s * s).sum())
    r0 = float(np.sqrt(np.mean((y - E0 * s) ** 2)))
    A = np.c_[np.ones_like(s), s]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a1, E1 = float(coef[0]), float(coef[1])
    r1 = float(np.sqrt(np.mean((y - (a1 + E1 * s)) ** 2)))
    return (a1, E1) if r1 < r0 else (0.0, E0)


def anomaly_metrics(df, pred):
    """지점 평균을 제거한 뒤의 metrics — 시간 신호의 실제 예측력."""
    d = df[["key"]].copy()
    d["y"], d["p"] = df["alt_cm"].values, pred
    d = d[np.isfinite(d.p)]
    cnt = d.groupby("key")["y"].transform("size")
    d = d[cnt > 1]
    if len(d) < 50:
        return None
    ya = d.y - d.groupby("key")["y"].transform("mean")
    pa = d.p - d.groupby("key")["p"].transform("mean")
    m = all_metrics(ya.values, pa.values)
    m["anom_corr"] = float(np.corrcoef(ya, pa)[0, 1]) if pa.std() > 0 else np.nan
    m["n_multi"] = int(len(d))
    return m


def main():
    df = build_panel()
    years = sorted(df["year"].unique())
    y = df["alt_cm"].to_numpy(float)
    ylog = np.log1p(y)
    s_all = df["e5t_sqrt_tdd"].to_numpy(float)
    folds = [(np.where(df.year != yy)[0], np.where(df.year == yy)[0]) for yy in years]
    print(f"[패널] {len(df):,}행 · {df.key.nunique():,}위치 · {years[0]}-{years[-1]} "
          f"· 정적 공변량 최근접 거리 중앙 {df.static_dist_km.median():.2f} km")

    # year+soil 은 격자에서 전부 채울 수 있는 집합이다(지형은 알래스카 전역 DEM 미확보).
    SETS = {"year9": YEAR9, "year+soil": YEAR9 + SOIL, "year+stat": YEAR9 + STATIC}
    WANT = ["mlp", "histgbm", "catboost"]
    models = [m for m in available_models() if m in WANT]
    print(f"[모형] 물리식 + 잔차결합 + {models}")

    rows, store = [], {}

    # ---- 물리식 단독(입력 집합과 무관, 1회) ----
    pred = np.full(len(df), np.nan)
    for tr, te in folds:
        a, E = fit_stefan(y[tr], s_all[tr])
        pred[te] = np.clip(a + E * s_all[te], 1.0, 600.0)
    store[("-", "stefan")] = pred
    rows.append(dict(featset="-", model="stefan", seed=-1, axis="temporal", **all_metrics(y, pred)))
    am = anomaly_metrics(df, pred)
    if am: rows.append(dict(featset="-", model="stefan", seed=-1, axis="anomaly", **am))
    print(f"  stefan            temporal {rows[0]['rmse_cm']:.3f}")

    for sname, feats in SETS.items():
        X = df[feats].to_numpy("float32")

        # ---- 학습 모형 ----
        for mdl in models:
            preds = []
            for sd in SEEDS:
                p = np.full(len(df), np.nan)
                for tr, te in folds:
                    Xtr, Xte = prep(X[tr], X[te], mdl in NAN_NATIVE)
                    p[te] = np.expm1(fit_predict(mdl, Xtr, ylog[tr], Xte, seed=sd)["pred"])
                preds.append(np.clip(p, 1.0, 600.0))
                rows.append(dict(featset=sname, model=mdl, seed=sd, axis="temporal",
                                 **all_metrics(y, preds[-1])))
            ens = np.mean(preds, axis=0)
            store[(sname, mdl)] = ens
            rows.append(dict(featset=sname, model=mdl, seed=-1, axis="temporal", **all_metrics(y, ens)))
            am = anomaly_metrics(df, ens)
            if am: rows.append(dict(featset=sname, model=mdl, seed=-1, axis="anomaly", **am))
            print(f"  {mdl:12s} {sname:10s} temporal {all_metrics(y, ens)['rmse_cm']:.3f}")

        # ---- 물리 잔차 결합: Stefan 앵커 + lam·능형 잔차 ----
        from sklearn.linear_model import Ridge
        base = np.full(len(df), np.nan)
        gfit = np.full(len(df), np.nan)
        for tr, te in folds:
            a, E = fit_stefan(y[tr], s_all[tr])
            anchor_tr, anchor_te = a + E * s_all[tr], a + E * s_all[te]
            Xtr, Xte = prep(X[tr], X[te], False)
            r = Ridge(alpha=10.0).fit(Xtr, y[tr] - anchor_tr)
            base[te], gfit[te] = anchor_te, r.predict(Xte)
        for lam in LAMS:
            p = np.clip(base + lam * gfit, 1.0, 600.0)
            store[(sname, f"resid_lam{lam:g}")] = p
            rows.append(dict(featset=sname, model=f"resid_lam{lam:g}", seed=-1,
                             axis="temporal", **all_metrics(y, p)))
            am = anomaly_metrics(df, p)
            if am: rows.append(dict(featset=sname, model=f"resid_lam{lam:g}", seed=-1,
                                    axis="anomaly", **am))
            print(f"  resid λ={lam:<4g} {sname:10s} temporal {all_metrics(y, p)['rmse_cm']:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(PROC / "s14_annual_results.csv", index=False)

    t = res[(res.axis == "temporal") & (res.seed == -1)].sort_values("rmse_cm")
    print("\n[연도 홀드아웃 순위]")
    print(t[["featset", "model", "rmse_cm", "r2", "skill_over_mean"]].head(12).to_string(index=False))
    a = res[(res.axis == "anomaly")].sort_values("rmse_cm")
    print("\n[within-site anomaly 축]")
    print(a[["featset", "model", "rmse_cm", "r2", "anom_corr"]].head(8).to_string(index=False))

    best = t.iloc[0]
    key = (best.featset, best.model)
    oof = df[["key", "lat", "lon", "year", "alt_cm"]].copy()
    oof["pred"] = store[key]
    oof["pred_stefan"] = store[("-", "stefan")]
    oof.to_csv(PROC / "s14_annual_oof.csv", index=False)

    json.dump({
        "purpose": "연별 ALT 예측 재설계 — 입력 집합·물리 결합을 공정 비교",
        "panel": {"rows": int(len(df)), "locs": int(df.key.nunique()),
                  "years": [int(v) for v in years],
                  "static_join_median_km": float(df.static_dist_km.median())},
        "featsets": {k: len(v) for k, v in SETS.items()},
        "protocol": "leave-one-year-out(temporal) + 지점평균 제거(anomaly). 3-seed 앙상블 예측.",
        "caveat": "연도 홀드아웃은 같은 위치가 다른 해로 학습에 들어간다. 공간 일반화 근거가 아니다.",
        "best": {"featset": str(best.featset), "model": str(best.model),
                 "rmse_cm": float(best.rmse_cm), "r2": float(best.r2)},
        "stefan_only_cm": float(t[t.model == "stefan"].rmse_cm.iloc[0]),
    }, open(PROC / "s14_annual_meta.json", "w"), ensure_ascii=False, indent=2)
    print("\n저장: s14_annual_{results.csv,oof.csv,meta.json}")


if __name__ == "__main__":
    main()
