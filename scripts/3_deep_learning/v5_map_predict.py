"""v5 연도 명시 ALT 지도(재설정 문서 단계 G). 지역(알래스카·레나) 1 km 격자에 연도별 강제력을 넣어 P1·R1 예측을 만든다.

학습: 그 지역의 모든 셀-연도 라벨(연도 정합 강제력, F8TS)로 P1 계수(E)와 R1(CatBoost λ 잔차)을 적합한다(지도는 학습 밖 추정이 아니다. 오차는
     같은 설정의 블록 교차검증(core_cb)에서 보고한다).
입력: 격자 정적 x25(map_<region>/<region>_grid_x25_v1.csv.gz: 지형 6·토양 9 등), 격자 연별 강제력(v5/grid/<region>_forcing_year.csv), 마스크(gray=0 만 그린다).
출력: data/processed/v5/maps/<region>/pred_<year>.csv.gz (cell_id, lat, lon, p1, r1, e5_tdd), mean_2011_2024.csv.gz, interval.json(블록 CV 잔차 분위수),
      meta.json. 연도 범위 2011–2024(2010 은 겨울 변수 부분).
구간: core_cb 의 R1(블록 CV, 연도 정합) OOF 잔차의 지역별 분위수(5·95 %)를 conformal 반폭으로 쓴다(전역 폭). AOA: 표준화 F8TS 공간에서 학습 행까지의
      최근접 거리가, 학습 행마다 '다른 블록' 학습 행까지의 최근접 거리(블록 CV 구조)의 95 분위를 넘으면 밖(셀 단위, 2011–2024 평균 강제력).
실행: python3 scripts/3_deep_learning/v5_map_predict.py --region Alaska [--lam 0.5] [--threads 16] [--years 2011-2024]
"""
from __future__ import annotations
import os, argparse, json, sys, time
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--region", required=True, choices=["Alaska", "Lena", "Canada"])
ap.add_argument("--lam", type=float, default=0.5)
ap.add_argument("--learner", default="catboost")
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--years", default="2011-2024")
ap.add_argument("--run", default="data/processed/v5/runs/core_cb")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--forcing", default="year", help="학습 강제력 판: year | year_obs")
ap.add_argument("--ref-doy", type=int, default=0, help="지도 격자의 TDD 를 이 날짜(1월 1일 = 1)까지 누적한 값으로 바꾼다(0 = 그 해 전체 TDD). 2026-10-20: 244 = 9월 1일, 학습 입력(관측일 누적)과 같은 범위")
ap.add_argument("--out-root", default="data/processed/v5/maps")
a = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.nice(10)
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import polar.testbed_v5 as T  # noqa: E402

t0 = time.time()
reg = a.region; key = reg.lower()
MAP = ROOT / "data" / "processed" / f"map_{key}"
OUT = ROOT / a.out_root / reg; OUT.mkdir(parents=True, exist_ok=True)
y0, y1 = (int(v) for v in a.years.split("-")); YEARS = list(range(y0, y1 + 1))
feats = T.FEATSETS["F8TS"]

# ---------------- 학습 자료(그 지역 모든 셀-연도)
df = T.load_v5(ROOT / "data" / "processed" / "v5", regions=(reg,), products=False)
d = T.with_forcing(df, a.forcing); d = d[T.usable_mask(d, feats)].reset_index(drop=True)
k = T.fit_phys(d)
Xtr = d[feats].values.astype(float); ytr = d.alt_cm.values.astype(float)
atr = T.phys_pred("P1", d, k)
seeds = [int(s) for s in a.seeds.split(",")]
models = []
from catboost import CatBoostRegressor
for s in seeds:
    kw = dict(iterations=600, learning_rate=0.03, depth=6) if a.learner == "catboost" else dict(iterations=200, learning_rate=0.05, depth=3)
    m = CatBoostRegressor(l2_leaf_reg=3.0, random_seed=s, verbose=0, allow_writing_files=False, thread_count=a.threads, **kw)
    m.fit(Xtr, ytr - atr); models.append(m)
print(f"[train] {reg} rows {len(d):,} E {k['E']:.3f} seeds {len(models)} {time.time()-t0:.0f}s", flush=True)

# ---------------- 구간(블록 CV OOF 잔차 분위수)과 AOA 기준
interval = {}
fs = sorted((ROOT / a.run / "preds").glob(f"{reg}__block__{a.forcing}__F8TS__R1__{a.learner}__lam{a.lam:g}.csv.gz"))
if fs:
    p = pd.read_csv(fs[0]).groupby("cell_year_id").agg(y=("y", "first"), pred=("pred", "mean"))
    res = (p.y - p.pred).values
    interval = dict(source=fs[0].name, n=int(len(res)), q05=float(np.percentile(res, 5)), q95=float(np.percentile(res, 95)),
                    rmse_oof=float(np.sqrt(np.mean(res ** 2))), half_width_90=float((np.percentile(res, 95) - np.percentile(res, 5)) / 2))
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
Ztr = np.nan_to_num((Xtr - mu) / sd)
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=1).fit(Ztr)
# AOA 기준: 블록 교차검증 구조대로, 학습 행마다 '다른 블록'의 학습 행까지의 최근접 거리(Meyer & Pebesma 2021 의 CV 기반 DI 에 해당)
blk = d.block.values; dtr = np.full(len(Ztr), np.nan)
for b in np.unique(blk):
    m = blk == b
    if (~m).sum() == 0:
        continue
    dtr[m] = NearestNeighbors(n_neighbors=1).fit(Ztr[~m]).kneighbors(Ztr[m])[0][:, 0]
aoa_thr = float(np.nanpercentile(dtr, 95))
print(f"[interval] {interval} aoa_thr {aoa_thr:.3f} {time.time()-t0:.0f}s", flush=True)

# ---------------- 격자
x25 = pd.read_csv(MAP / f"{key}_grid_x25_v1.csv.gz", low_memory=False)
mask = pd.read_csv(MAP / f"{key}_grid_mask_v1.csv", usecols=["cell_id", "gray", "land"])
x25 = x25.merge(mask, on="cell_id", how="left", suffixes=("", "_m"))
keep = x25[(x25.gray == 0)].copy() if "gray" in x25 else x25.copy()
static_cols = [c for c in feats if not c.startswith("e5_")]
print(f"[grid] cells {len(x25):,} drawn {len(keep):,} {time.time()-t0:.0f}s", flush=True)
fy = pd.read_csv(ROOT / "data" / "processed" / "v5" / "grid" / f"{key}_forcing_year.csv")
fy = fy[fy.cell_id.isin(keep.cell_id)].copy()
fcols = [c for c in feats if c.startswith("e5_")]
ref_info = {}
if a.ref_doy > 0:                                         # TDD accumulated to the reference day (linear between month-end cumulative sums, as testbed_v5.with_forcing year_obs)
    import calendar
    cum_cols = [f"tdd_cum_{k}" for k in range(1, 13)]
    assert all(c in fy.columns for c in cum_cols), "grid forcing table has no tdd_cum columns; rebuild it with build_v5_forcing_year.py"
    cum = fy[cum_cols].values.astype(float); yrs = fy.year.values.astype(int)
    st0 = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    mon = int(np.clip(np.searchsorted(st0, a.ref_doy, side="right") - 1, 0, 11))
    leap = np.array([calendar.isleap(int(y)) for y in yrs])
    st = st0[mon] + (1 if (leap.any() and mon >= 2) else 0) * leap.astype(int)
    ndays = np.array([calendar.monthrange(int(y), mon + 1)[1] for y in yrs])
    frac = np.clip((a.ref_doy - st) / ndays, 0, 1)
    prev = cum[:, mon - 1] if mon > 0 else np.zeros(len(fy))
    tdd_ref = prev + frac * (cum[:, mon] - prev)
    ref_info = dict(ref_doy=a.ref_doy, month_index=mon + 1, mean_fraction_of_full_season=float(np.nanmean(tdd_ref / np.where(cum[:, 11] > 0, cum[:, 11], np.nan))),
                    note="e5_tdd and e5_sqrt_tdd of the grid replaced by the TDD accumulated to the reference day; other climate inputs unchanged")
    fy["e5_tdd"] = tdd_ref; fy["e5_sqrt_tdd"] = np.sqrt(np.clip(tdd_ref, 0, None))
    print(f"[ref-doy] {ref_info}", flush=True)
acc = None; n_acc = 0
for y in YEARS:
    g = keep[["cell_id", "lat", "lon"] + static_cols].merge(fy[fy.year == y][["cell_id"] + fcols], on="cell_id", how="inner")
    g = g[np.isfinite(g.e5_sqrt_tdd.values) & (g.e5_sqrt_tdd.values > 0)]
    X = g[feats].values.astype(float)
    p1 = k["E"] * g.e5_sqrt_tdd.values
    r = np.mean([m.predict(X) for m in models], axis=0)
    r1 = p1 + a.lam * r
    out = pd.DataFrame({"cell_id": g.cell_id.values, "lat": g.lat.values, "lon": g.lon.values, "year": y, "e5_tdd": g.e5_tdd.values,
                        "p1": p1, "r1": r1})
    out.to_csv(OUT / f"pred_{y}.csv.gz", index=False)
    s = out.set_index("cell_id")[["p1", "r1"]]
    acc = s if acc is None else acc.add(s, fill_value=0); n_acc += 1
    print(f"[{y}] cells {len(out):,} p1 {np.nanmean(p1):.1f} r1 {np.nanmean(r1):.1f} {time.time()-t0:.0f}s", flush=True)
mean = (acc / n_acc).reset_index().merge(keep[["cell_id", "lat", "lon"]], on="cell_id")
# AOA: 2011–2024 평균 지도 입력(ref-doy 가 있으면 그 날짜까지의 TDD) + 정적
gm = keep[["cell_id"] + static_cols].merge(fy[fy.year.between(y0, y1)].groupby("cell_id")[fcols].mean().reset_index(), on="cell_id", how="inner")
Zg = np.nan_to_num((gm[feats].values.astype(float) - mu) / sd)
dg = np.concatenate([nn.kneighbors(Zg[i:i + 200000], n_neighbors=1)[0][:, 0] for i in range(0, len(Zg), 200000)])
mean = mean.merge(pd.DataFrame({"cell_id": gm.cell_id.values, "aoa_di": dg, "aoa_in": (dg <= aoa_thr).astype(int)}), on="cell_id", how="left")
mean.to_csv(OUT / f"mean_{y0}_{y1}.csv.gz", index=False)
json.dump(dict(region=reg, years=YEARS, lam=a.lam, learner=a.learner, seeds=seeds, E=k["E"], n_train=int(len(d)), feats=feats, interval=interval,
               forcing=a.forcing, run=a.run, ref_doy=a.ref_doy, ref_doy_info=ref_info,
               aoa=dict(threshold=aoa_thr, frac_inside=float(np.mean(dg <= aoa_thr))), n_cells_drawn=int(len(keep)), elapsed_s=round(time.time() - t0, 1),
               created=time.strftime("%Y-%m-%d %H:%M:%S")), open(OUT / "meta.json", "w"), ensure_ascii=False, indent=1)
print(f"[done] {time.time()-t0:.0f}s")
