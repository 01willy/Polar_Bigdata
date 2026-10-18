"""v5 시험지(재설정 문서 docs/RESET_2026-10-10_KOPRI_FEEDBACK.md 4.4–4.6, 단계 B).

자료: 셀-연도 라벨(labels_cell_year_v5) + 연별 강제력(forcing_year) + 기후값(forcing_clim, 2011–2024 연별 평균) + 정적 공변량(static_v5)
      + 제품(products_cell_year_v5, 선택). 지역은 Alaska·Lena·Canada.
강제력 판: 'year' = 셀 관측 연도의 값, 'clim' = 2011–2024 평균값(같은 정의). with_forcing() 이 e5_* 열을 고른 판으로 채운다.
분할: random(V-R) · block(V-B, 0.5° 블록 k겹) · year(V-Y, 연도 홀드아웃) · block_year(V-BY, 블록 겹 × 연도) · region(V-L, 100 km 완충).
방법: P0(원천 계수 Stefan, 다른 두 지역 전량 적합) · P1(학습 겹 재보정 Stefan) · P1a(아핀) · Ku(Kudryavtsev 척도 보정) · D0(직접 ML)
      · F1(물리 출력 입력) · D1(물리 유사라벨 증강) · R1(앵커 + λ 잔차 ML) · S(P1·Ku·CCI 비음수 적층).
지표: RMSE·편향·MAE·r·R²·블록 등가중 RMSE, 연간 편차(셀 평균 제거) RMSE·r·skill, 블록 부트스트랩 짝지은 ΔRMSE CI.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from polar.fidelity import TERRAIN, SOIL, INSAR, POLSAR, CCI, FLAGS

REGIONS = ("Alaska", "Lena", "Canada")
F8 = ["e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
F_EXTRA = ["e5_tdd_soil", "e5_sqrt_tdd_soil", "e5_swe_max"]
FORCING_COLS = F8 + F_EXTRA
MODIS = ["ndvi_summer", "evi_summer", "snow_off_doy", "snow_days"]
LAG = ["e5_tdd_prev", "e5_fdd_prev", "e5_swe_prev"]                       # 전년 강제력(지연 효과)
HIRES = ["ncscd_soc", "cavm_class", "wc_tree", "wc_shrub", "wc_grass", "wc_wet", "wc_water"]   # 고해상 입력(있을 때만)
FEATSETS = {
    "F8": F8,
    "F8T": F8 + TERRAIN,
    "F8TS": F8 + TERRAIN + SOIL,
    "F8TS_M": F8 + TERRAIN + SOIL + MODIS,
    "F8TS_L": F8 + TERRAIN + SOIL + LAG,
    "F8TS_X": F8 + TERRAIN + SOIL + HIRES,
    "F8TS_MS": F8 + TERRAIN + SOIL + MODIS + INSAR + POLSAR + FLAGS,
    "F8TS_ALL": F8 + TERRAIN + SOIL + MODIS + LAG + HIRES,
    "F8TS_SAR": F8 + TERRAIN + SOIL + INSAR + POLSAR + FLAGS,
}
PRODUCT_YEARLY = ("cci4y", "cci5y", "wei", "liu", "yk")
METHODS = ("P0", "P1", "P1a", "Ku", "Ed", "PE", "D0", "F1", "D1", "R1", "R1cv", "S")
PHYS_METHODS = ("P0", "P1", "P1a", "Ku", "Ed", "PE")
LAM_CV = (0.25, 0.5, 0.75, 1.0)
NAN_NATIVE = {"catboost", "catboost_lo", "histgbm", "lightgbm", "xgboost"}


# ================================================================ 자료
def load_v5(proc, regions=REGIONS, products=True):
    proc = Path(proc)
    lab = pd.read_csv(proc / "labels_cell_year_v5.csv")
    lab = lab[(lab.in_forcing == 1) & lab.region.isin(regions)].copy()
    fy = pd.read_csv(proc / "forcing_year.csv")
    fc = pd.read_csv(proc / "forcing_clim.csv")
    st = pd.read_csv(proc / "static_v5.csv", low_memory=False)
    ycols = {c: f"y_{c}" for c in FORCING_COLS}
    ccols = {f"clim1124_{c[3:]}": f"c_{c}" for c in FORCING_COLS}
    cum = [c for c in fy.columns if c.startswith("tdd_cum_")]
    df = lab.merge(fy[["loc_id", "year", "winter_partial", "e5_grid_lat", "e5_grid_lon", "e5_fallback_deg"] + FORCING_COLS + cum].rename(columns=ycols),
                   on=["loc_id", "year"], how="left")
    df = df.merge(fc[["loc_id"] + list(ccols)].rename(columns=ccols), on="loc_id", how="left")
    lag = fy[["loc_id", "year", "e5_tdd", "e5_fdd", "e5_swe"]].copy(); lag["year"] = lag.year + 1          # 전년 값을 해당 연도 행에 붙인다
    lag = lag.rename(columns={"e5_tdd": "y_e5_tdd_prev", "e5_fdd": "y_e5_fdd_prev", "e5_swe": "y_e5_swe_prev"})
    df = df.merge(lag, on=["loc_id", "year"], how="left")
    for c in ("e5_tdd_prev", "e5_fdd_prev", "e5_swe_prev"):
        df[f"c_{c}"] = df[f"c_{c[:-5]}"]                                                            # 기후값 판에서는 당해 기후값과 같다
    df["obs_month_last"] = df.obs_months.astype(str).map(lambda v: 9 if v == "eos" else max(int(x) for x in v.split(";") if x.strip().isdigit()))
    hp = proc / "hires_cell_v5.csv"
    if hp.exists():
        hi = pd.read_csv(hp)
        df = df.merge(hi[["loc_id"] + [c for c in HIRES if c in hi]], on="loc_id", how="left")
    stcols = ["loc_id"] + TERRAIN + SOIL + INSAR + POLSAR + CCI + FLAGS
    df = df.merge(st[stcols], on="loc_id", how="left")
    if products:
        pr = pd.read_csv(proc / "products_cell_year_v5.csv")
        wide = product_wide(pr, df[["loc_id", "year"]])
        df = df.merge(wide, on=["loc_id", "year"], how="left")
    mp = proc / "modis_cell_year_v5.csv"
    if mp.exists():
        mo = pd.read_csv(mp)
        ycols = {c: f"y_{c}" for c in MODIS}
        df = df.merge(mo[["loc_id", "year"] + MODIS].rename(columns=ycols), on=["loc_id", "year"], how="left")
        mcols = {f"{c}_mean": f"c_{c}" for c in MODIS}
        df = df.merge(mo.drop_duplicates("loc_id")[["loc_id"] + list(mcols)].rename(columns=mcols), on="loc_id", how="left")
        for c in MODIS:                                   # 연별 값이 없는 해(2015–2020 밖)는 다년 평균으로 채우고 표지를 둔다
            df[f"{c}_is_mean"] = df[f"y_{c}"].isna().astype(int)
            df[f"y_{c}"] = df[f"y_{c}"].fillna(df[f"c_{c}"])
    df["cell_year_id"] = np.arange(len(df))
    return df.reset_index(drop=True)


def product_wide(pr, keys):
    """제품 긴 표 → (loc_id, year) 별 열. 연별 제품은 같은 해 값(없으면 NaN), 대표값 제품(year = -1)은 그대로. 열 이름 prod_<p>."""
    out = keys.drop_duplicates().copy()
    for p, s in pr.groupby("product"):
        if (s.year == -1).all():
            m = s.set_index("loc_id").value_cm
            out[f"prod_{p}"] = out.loc_id.map(m).values
        else:
            m = s.set_index(["loc_id", "year"]).value_cm
            out[f"prod_{p}"] = pd.MultiIndex.from_frame(out[["loc_id", "year"]]).map(m).values
            allm = s.groupby("loc_id").value_cm.mean()
            out[f"prod_{p}_mean"] = out.loc_id.map(allm).values
    return out


def with_forcing(df, version):
    """e5_* 열을 'year' 또는 'clim' 판으로 채운 복사본. 물리 열(p2_edaphic, p4_ku)도 다시 계산한다."""
    d = df.copy()
    pre = {"year": "y_", "clim": "c_", "year_obs": "y_"}[version]
    for c in FORCING_COLS + LAG:
        if f"{pre}{c}" in d:
            d[c] = d[f"{pre}{c}"].values
    if version == "year_obs":                              # 관측일까지의 누적 TDD(월 누적값을 관측일로 선형 보간, 날짜 없는 CALM 은 9월 1일)
        import calendar
        cum = np.stack([d[f"tdd_cum_{k}"].values for k in range(1, 13)], axis=1)
        doy = d["obs_doy"].values if "obs_doy" in d else np.full(len(d), np.nan)
        doy = np.where(np.isfinite(doy), doy, 244.0)
        yr = d.year.values.astype(int)
        starts = np.array([[sum(calendar.monthrange(int(y), k)[1] for k in range(1, m)) for m in range(1, 13)] for y in (2012, 2013)])  # 윤년/평년
        leap = np.array([calendar.isleap(int(y)) for y in yr])
        st = np.where(leap[:, None], starts[0][None, :], starts[1][None, :])                       # 각 달의 시작 전 누적 일수
        mon = np.clip(np.searchsorted(st[0] if False else np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]), doy, side="right") - 1, 0, 11)
        ndays = np.array([calendar.monthrange(int(y), int(m) + 1)[1] for y, m in zip(yr, mon)])
        frac = np.clip((doy - st[np.arange(len(d)), mon]) / ndays, 0, 1)
        prev = np.where(mon > 0, cum[np.arange(len(d)), np.clip(mon - 1, 0, 11)], 0.0)
        tdd = prev + frac * (cum[np.arange(len(d)), mon] - prev)
        d["e5_tdd"] = tdd; d["e5_sqrt_tdd"] = np.sqrt(np.clip(tdd, 0, None)); d["tdd_obs_frac"] = tdd / np.where(cum[:, 11] > 0, cum[:, 11], np.nan)
    d.loc[d.e5_tdd_soil <= 0, ["e5_tdd_soil", "e5_sqrt_tdd_soil"]] = np.nan
    for c in MODIS:
        if f"{pre}{c}" in d:
            d[c] = d[f"{pre}{c}"].values
    if version == "clim":                                 # 기후값 판에서는 제품도 다년 평균을 쓴다
        for p in PRODUCT_YEARLY:
            if f"prod_{p}_mean" in d:
                d[f"prod_{p}"] = d[f"prod_{p}_mean"].values
    attach_physics(d)
    d["forcing"] = version
    return d


def attach_physics(d):
    from polar.physics import physics_ensemble
    need = ["e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_maat", "e5_twarm", "e5_tcold", "e5_swe", "sg_bdod_5_15", "sg_sand_5_15", "sg_cfvo_5_15",
            "sg_soc_5_15"]
    sub = d[need].copy()
    ok = sub.notna().all(axis=1).values
    d["p2_edaphic"] = np.nan; d["p4_ku"] = np.nan
    if ok.any():
        P = physics_ensemble(sub[ok], E=1.0)
        d.loc[ok, "p2_edaphic"] = P["p2_edaphic"]; d.loc[ok, "p4_ku"] = P["p4_kudryavtsev"]
    return d


def usable_mask(d, feats):
    """라벨·√TDD 유효 행. 특징 결측은 학습기가 처리하거나 대치한다."""
    return np.isfinite(d.alt_cm.values) & np.isfinite(d.e5_sqrt_tdd.values) & (d.e5_sqrt_tdd.values > 0)


# ================================================================ 분할
@dataclass
class Fold:
    scheme: str
    name: str
    tr: np.ndarray
    te: np.ndarray


def _balanced_group_folds(groups, k, seed):
    """그룹(블록)을 행 수 균형으로 k 겹에 배정. 같은 그룹은 같은 겹."""
    g = np.asarray(groups)
    u, cnt = np.unique(g, return_counts=True)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(u))
    u, cnt = u[order], cnt[order]
    order = np.argsort(-cnt, kind="stable")
    fold_of = {}
    load = np.zeros(k, int)
    for j in order:
        f = int(np.argmin(load)); fold_of[u[j]] = f; load[f] += cnt[j]
    return np.array([fold_of[v] for v in g])


def folds_random(d, k=5, seed=0):
    rng = np.random.RandomState(seed); f = rng.permutation(len(d)) % k
    return [Fold("random", f"r{i}", np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]


def folds_block(d, k=5, seed=0):
    f = _balanced_group_folds(d.block.values, k, seed)
    return [Fold("block", f"b{i}", np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]


def folds_year(d, min_test=20):
    y = d.year.values
    out = []
    for yy in sorted(np.unique(y)):
        te = np.where(y == yy)[0]
        if len(te) < min_test:
            continue
        out.append(Fold("year", f"y{yy}", np.where(y != yy)[0], te))
    return out


def folds_block_year(d, k=5, seed=0, min_test=10):
    f = _balanced_group_folds(d.block.values, k, seed); y = d.year.values
    out = []
    for i in range(k):
        for yy in sorted(np.unique(y[f == i])):
            te = np.where((f == i) & (y == yy))[0]
            if len(te) < min_test:
                continue
            out.append(Fold("block_year", f"b{i}y{yy}", np.where((f != i) & (y != yy))[0], te))
    return out


def folds_region(d, buffer_km=100.0):
    """지역 홀드아웃. 학습 집합에서 시험 지역 셀로부터 buffer_km 안의 셀을 뺀다."""
    from sklearn.neighbors import BallTree
    out = []
    reg = d.region.values
    for r in sorted(np.unique(reg)):
        te = np.where(reg == r)[0]; tr = np.where(reg != r)[0]
        if len(te) == 0 or len(tr) == 0:
            continue
        tree = BallTree(np.radians(d[["lat", "lon"]].values[te]), metric="haversine")
        dist = tree.query(np.radians(d[["lat", "lon"]].values[tr]), k=1)[0][:, 0] * 6371.0
        out.append(Fold("region", f"L_{r}", tr[dist > buffer_km], te))
    return out


def make_folds(d, scheme, k=5, seed=0):
    return {"random": lambda: folds_random(d, k, seed), "block": lambda: folds_block(d, k, seed), "year": lambda: folds_year(d),
            "block_year": lambda: folds_block_year(d, k, seed), "region": lambda: folds_region(d)}[scheme]()


# ================================================================ 물리식·방법
def ls_scale(p, y):
    m = np.isfinite(y) & np.isfinite(p) & (p > 0)
    return float((p[m] @ y[m]) / (p[m] @ p[m])) if m.sum() >= 3 else np.nan


def ls_affine(p, y):
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 3:
        return np.nan, np.nan
    b, a = np.polyfit(p[m], y[m], 1)
    return float(a), float(b)


def fit_phys(tr):
    y = tr.alt_cm.values.astype(float); s = tr.e5_sqrt_tdd.values.astype(float); ku = tr.p4_ku.values.astype(float)
    ed = tr.p2_edaphic.values.astype(float)
    k = dict(E=ls_scale(s, y), c_ku=ls_scale(ku, y), c_ed=ls_scale(ed, y), n=int(np.isfinite(y).sum()))
    k["a"], k["b"] = ls_affine(s, y)
    from scipy.optimize import nnls
    A = np.c_[k["E"] * s, np.where(np.isfinite(ku), k["c_ku"] * ku, k["E"] * s), np.where(np.isfinite(ed), k["c_ed"] * ed, k["E"] * s)]
    m = np.isfinite(A).all(1) & np.isfinite(y)
    w, _ = nnls(A[m], y[m]) if m.sum() >= 3 else (np.array([1.0, 0.0, 0.0]), 0.0)
    k["w_pe"] = (w / w.sum()).tolist() if w.sum() > 0 else [1.0, 0.0, 0.0]
    return k


def phys_pred(kind, d, k, E0=None):
    s = d.e5_sqrt_tdd.values.astype(float)
    st = k["E"] * s
    if kind == "P0":
        return E0 * s
    if kind == "P1":
        return st
    if kind == "P1a":
        return k["a"] + k["b"] * s
    if kind == "Ku":
        ku = k["c_ku"] * d.p4_ku.values.astype(float)
        return np.where(np.isfinite(ku), ku, st)
    if kind == "Ed":
        ed = k["c_ed"] * d.p2_edaphic.values.astype(float)
        return np.where(np.isfinite(ed), ed, st)
    if kind == "PE":                                       # 물리 앙상블(Stefan·Kudryavtsev·토양형 Stefan 의 비음수 가중)
        w = k["w_pe"]
        return w[0] * st + w[1] * phys_pred("Ku", d, k) + w[2] * phys_pred("Ed", d, k)
    raise ValueError(kind)


def _impute_scale(Xtr, Xte):
    """학습 겹 중앙값 대치 + 표준화. 학습 겹에서 거의 상수인 특징은 나누지 않고(척도 1) 표준화 값을 ±10 으로 자른다(신경망 폭주 방지)."""
    med = np.nanmedian(Xtr, axis=0); med = np.where(np.isfinite(med), med, 0.0)
    Xtr = np.where(np.isfinite(Xtr), Xtr, med); Xte = np.where(np.isfinite(Xte), Xte, med)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return np.clip((Xtr - mu) / sd, -10, 10), np.clip((Xte - mu) / sd, -10, 10)


def fit_learner(name, Xtr, ytr, Xte, seed=0, threads=8, epochs=100, sample_weight=None):
    Xtr = np.asarray(Xtr, float); Xte = np.asarray(Xte, float); ytr = np.asarray(ytr, float)
    if name in ("catboost", "catboost_lo"):
        from catboost import CatBoostRegressor
        kw = dict(iterations=600, learning_rate=0.03, depth=6) if name == "catboost" else dict(iterations=200, learning_rate=0.05, depth=3)
        m = CatBoostRegressor(l2_leaf_reg=3.0, random_seed=seed, verbose=0, allow_writing_files=False, thread_count=threads, **kw)
        m.fit(Xtr, ytr, sample_weight=sample_weight)
        return np.asarray(m.predict(Xte))
    if name == "rf":
        from sklearn.ensemble import RandomForestRegressor
        a, b = _impute_scale(Xtr, Xte)
        m = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5, n_jobs=threads, random_state=seed).fit(a, ytr, sample_weight=sample_weight)
        return m.predict(b)
    if sample_weight is not None:                           # 신경망·ridge: 가중 재표집(m1_core 규약)
        w = np.asarray(sample_weight, float); w = w / w.sum()
        rs = np.random.RandomState(seed).choice(len(Xtr), len(Xtr), replace=True, p=w)
        Xtr, ytr = Xtr[rs], ytr[rs]
    if name == "ridge":
        from sklearn.linear_model import Ridge
        a, b = _impute_scale(Xtr, Xte)
        return Ridge(alpha=10.0).fit(a, ytr).predict(b)
    from polar.tab_models import fit_predict
    a, b = _impute_scale(Xtr, Xte)
    return np.asarray(fit_predict(name, a.astype(np.float32), ytr, b.astype(np.float32), seed=seed, epochs=epochs)["pred"])


def year_weights(tr):
    """연도별 행 수의 역수 가중(연도 합이 같아지도록). 2014 집중 관측의 지배를 줄인다."""
    cnt = tr.year.map(tr.year.value_counts()).values.astype(float)
    w = 1.0 / cnt
    return w * len(tr) / w.sum()


def run_method(method, tr, te, feats, learner="catboost", lam=0.5, seed=0, threads=8, E0=None, r_pseudo=1.0, epochs=100, year_weight=False):
    """학습 겹 tr, 시험 겹 te(DataFrame, with_forcing 뒤)에서 방법의 시험 예측(cm)을 돌려준다."""
    k = fit_phys(tr)
    if method in PHYS_METHODS:
        return phys_pred(method, te, k, E0=E0), k
    Xtr, Xte = tr[feats].values.astype(float), te[feats].values.astype(float)
    ytr = tr.alt_cm.values.astype(float)
    sw = year_weights(tr) if year_weight else None
    if method == "D0":
        return fit_learner(learner, Xtr, ytr, Xte, seed, threads, epochs, sample_weight=sw), k
    if method == "R1cv":                                   # λ 를 학습 겹 안 블록 3겹 CV 로 고른다
        atr = phys_pred("P1", tr, k); ate = phys_pred("P1", te, k)
        inner = folds_block(tr, 3, seed)
        res_oof = np.full(len(tr), np.nan)
        for f in inner:
            ki = fit_phys(tr.iloc[f.tr]); a_i = phys_pred("P1", tr.iloc[f.tr], ki); a_o = phys_pred("P1", tr.iloc[f.te], ki)
            res_oof[f.te] = a_o + fit_learner(learner, Xtr[f.tr], ytr[f.tr] - a_i, Xtr[f.te], seed, threads, epochs) - a_o
        best, best_r = 0.0, rmse(ytr, atr)
        for lv in LAM_CV:
            r = rmse(ytr, atr + lv * res_oof)
            if r < best_r:
                best, best_r = lv, r
        k["lam_cv"] = best
        res = fit_learner(learner, Xtr, ytr - atr, Xte, seed, threads, epochs, sample_weight=sw)
        return ate + best * res, k
    if method == "F1":
        ptr = np.c_[phys_pred("P1", tr, k), phys_pred("Ku", tr, k)]; pte = np.c_[phys_pred("P1", te, k), phys_pred("Ku", te, k)]
        return fit_learner(learner, np.c_[Xtr, ptr], ytr, np.c_[Xte, pte], seed, threads, epochs), k
    if method == "D1":
        rng = np.random.RandomState(seed)
        npl = int(min(len(te), round(r_pseudo * len(tr))))
        j = rng.choice(len(te), npl, replace=False) if npl < len(te) else np.arange(len(te))
        Xp = Xte[j]; yp = phys_pred("P1", te, k)[j]
        return fit_learner(learner, np.r_[Xtr, Xp], np.r_[ytr, yp], Xte, seed, threads, epochs), k
    if method == "R1":
        atr, ate = phys_pred("P1", tr, k), phys_pred("P1", te, k)
        res = fit_learner(learner, Xtr, ytr - atr, Xte, seed, threads, epochs, sample_weight=sw)
        return ate + lam * res, k
    if method == "S":
        from scipy.optimize import nnls
        def cols(d):
            c = [phys_pred("P1", d, k), phys_pred("Ku", d, k)]
            for p in ("prod_cci4y", "prod_cci5y"):
                if p in d and d[p].notna().any():
                    v = d[p].values.astype(float); c.append(np.where(np.isfinite(v), v, c[0]))
            return np.c_[tuple(c)]
        A, B = cols(tr), cols(te)
        w, _ = nnls(A, ytr)
        if w.sum() <= 0:
            w = np.array([1.0] + [0.0] * (A.shape[1] - 1))
        k["w_stack"] = w.tolist()
        return B @ w, k
    raise ValueError(method)


# ================================================================ 지표
def rmse(y, p):
    m = np.isfinite(y) & np.isfinite(p); return float(np.sqrt(np.mean((p[m] - y[m]) ** 2))) if m.any() else np.nan


def block_eq_rmse(y, p, blocks):
    m = np.isfinite(y) & np.isfinite(p)
    s = pd.DataFrame({"e2": (p[m] - y[m]) ** 2, "b": np.asarray(blocks)[m]}).groupby("b").e2.mean()
    return float(np.sqrt(s.mean())) if len(s) else np.nan


def metrics(y, p, blocks=None):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 3:
        return dict(n=int(m.sum()), rmse=np.nan, bias=np.nan, mae=np.nan, r=np.nan, r2=np.nan, rmse_beq=np.nan)
    e = p[m] - y[m]
    r = float(np.corrcoef(y[m], p[m])[0, 1]) if np.std(p[m]) > 0 else np.nan
    out = dict(n=int(m.sum()), rmse=float(np.sqrt(np.mean(e ** 2))), bias=float(e.mean()), mae=float(np.abs(e).mean()), r=r,
               r2=float(1 - np.sum(e ** 2) / np.sum((y[m] - y[m].mean()) ** 2)))
    out["rmse_beq"] = block_eq_rmse(y, p, blocks) if blocks is not None else np.nan
    return out


def anomaly_metrics(y, p, loc):
    """셀 평균을 뺀 연간 편차. 셀에 2개 이상 연도가 있는 행만."""
    d = pd.DataFrame({"y": y, "p": p, "loc": loc}).dropna()
    n = d.groupby("loc").y.transform("size")
    d = d[n >= 2]
    if len(d) < 5:
        return dict(n_anom=int(len(d)), cells_anom=int(d["loc"].nunique()), rmse_anom=np.nan, r_anom=np.nan, skill_anom=np.nan, sd_anom=np.nan)
    ya = d.y - d.groupby("loc").y.transform("mean"); pa = d.p - d.groupby("loc").p.transform("mean")
    mse = float(np.mean((pa - ya) ** 2)); var = float(np.mean(ya ** 2))
    r = float(np.corrcoef(ya, pa)[0, 1]) if pa.std() > 0 else np.nan
    return dict(n_anom=int(len(d)), cells_anom=int(d["loc"].nunique()), rmse_anom=float(np.sqrt(mse)), r_anom=r,
                skill_anom=float(1 - mse / var) if var > 0 else np.nan, sd_anom=float(np.sqrt(var)))


def boot_delta(y, pA, pB, blocks, nboot=2000, seed=0):
    """블록 부트스트랩 짝지은 ΔRMSE = RMSE(A) − RMSE(B)(음수 = A 가 낫다). 행 가중판과 블록 등가중판."""
    y, pA, pB = (np.asarray(v, float) for v in (y, pA, pB))
    m = np.isfinite(y) & np.isfinite(pA) & np.isfinite(pB)
    y, pA, pB, b = y[m], pA[m], pB[m], np.asarray(blocks)[m]
    ub, inv = np.unique(b, return_inverse=True)
    nb = len(ub)
    eA, eB = (pA - y) ** 2, (pB - y) ** 2
    sA = np.bincount(inv, eA, nb); sB = np.bincount(inv, eB, nb); cnt = np.bincount(inv, minlength=nb).astype(float)
    rng = np.random.RandomState(seed)
    W = rng.multinomial(nb, np.full(nb, 1.0 / nb), size=nboot).astype(float)
    dA = np.sqrt((W @ sA) / (W @ cnt)) - np.sqrt((W @ sB) / (W @ cnt))
    mA, mB = sA / cnt, sB / cnt
    dBq = np.sqrt((W @ mA) / nb) - np.sqrt((W @ mB) / nb)
    d0 = float(np.sqrt(sA.sum() / cnt.sum()) - np.sqrt(sB.sum() / cnt.sum()))
    d0q = float(np.sqrt(mA.mean()) - np.sqrt(mB.mean()))
    return dict(delta=d0, lo=float(np.percentile(dA, 2.5)), hi=float(np.percentile(dA, 97.5)),
                delta_beq=d0q, lo_beq=float(np.percentile(dBq, 2.5)), hi_beq=float(np.percentile(dBq, 97.5)), n_blocks=int(nb), n=int(len(y)))
