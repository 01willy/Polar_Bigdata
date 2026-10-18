# -*- coding: utf-8 -*-
"""v9_extra.py: 사용자 10-15 질문·수정 항목에 따른 추가 실험(Rescale 전용, 관측일 기후 입력 year_obs, 특징 F8TS, CatBoost).

PA   제품 기반 잔차 결합(질문 3·4: 지역 라벨을 쓰는 이점이 물리 기준에서 오는가, 지역 라벨 자체에서 오는가).
     지역별 공간 블록 5겹(주 실험과 같은 겹). 각 겹에서 제품 값이 있는 행만 학습·시험한다(제품 p ∈ {wei, liu}).
     방법 Praw(제품 값 그대로) · Pcal(학습 겹에서 a + b·제품 재보정) · P1(학습 겹 재보정 Stefan) · R1(P1 + 0.5·잔차 CatBoost)
          · RP(Pcal + 0.5·잔차) · RP1(Pcal + 1.0·잔차, RP 와 같은 학습기) · R1x(R1 + 제품 값을 특징으로 추가).
EXT  외부 지역 무라벨 시험(질문 5). 세 연구 지역 전량으로 학습하고 CALM 외부 지점(data/processed/v9_ext)을 예측한다.
     방법 P0(세 지역 E) · D0 · F1 · R1(0.5) · P1loc(같은 외부 소지역의 다른 셀로 E 재보정, 시험 셀 제외) · R1loc(P1loc + 0.5·원천 잔차).
SHAP 지역별 잔차 학습기(전량, y − P1)와 직접 학습기의 SHAP 값(CatBoost ShapValues). 특징별 평균 |SHAP|와 행별 값.
출력: <out>/<tag>/preds_PA_<region>_<product>.csv.gz, preds_EXT.csv.gz, shap_<kind>_<region>.csv.gz, shap_summary.csv, run_*.json
실행(Rescale): python3 scripts/3_deep_learning/v9_extra.py --tag extra_v9 --exp PA --regions Alaska --threads 24
로컬 점검(학습 없음): python3 scripts/3_deep_learning/v9_extra.py --tag dry --exp PA,EXT,SHAP --dry
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--proc", default="data/processed/v5"); ap.add_argument("--ext", default="data/processed/v9_ext"); ap.add_argument("--out", default="data/processed/v5/runs")
ap.add_argument("--tag", required=True); ap.add_argument("--exp", default="PA,EXT,SHAP"); ap.add_argument("--regions", default="Alaska,Lena,Canada")
ap.add_argument("--products", default="wei,liu"); ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--k", type=int, default=5)
ap.add_argument("--lam", type=float, default=0.5); ap.add_argument("--threads", type=int, default=16); ap.add_argument("--nice", type=int, default=10)
ap.add_argument("--learner", default="catboost"); ap.add_argument("--dry", action="store_true", help="학습기 대신 0 을 돌려 자료 흐름만 확인")
a = ap.parse_args()
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, str(a.threads))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.nice(a.nice)
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import polar.testbed_v5 as T  # noqa: E402

OUT = ROOT / a.out / a.tag; OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time(); FORC = "year_obs"; feats = T.FEATSETS["F8TS"]; seeds = [int(s) for s in a.seeds.split(",")]
df = T.load_v5(ROOT / a.proc, products=True)
d_all = T.with_forcing(df, FORC); d_all = d_all[T.usable_mask(d_all, None)].reset_index(drop=True)
print(f"[load] usable {len(d_all):,} {time.time()-t0:.0f}s", flush=True)
json.dump(dict(args=vars(a), started=time.strftime("%Y-%m-%d %H:%M:%S"), rows=int(len(d_all)), feats=feats), open(OUT / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json", "w"), indent=1)


def fit(Xtr, ytr, Xte, seed):
    if a.dry:
        return np.zeros(len(Xte))
    return T.fit_learner(a.learner, Xtr, ytr, Xte, seed, a.threads)


def rows_of(te, pred, **tags):
    o = te[["cell_year_id", "loc_id", "year", "block", "region"]].copy(); o["y"] = te.alt_cm.values; o["pred"] = np.asarray(pred, float)
    for k, v in tags.items():
        o[k] = v
    return o


def X(d, extra=None):
    m = d[feats].values.astype(float)
    return m if extra is None else np.c_[m, d[extra].values.astype(float)]


# ---------------------------------------------------------------- PA
def run_pa(region, prod):
    d = d_all[d_all.region == region].reset_index(drop=True)
    folds = T.folds_block(d, a.k, 0)                       # 주 실험(core_v6)과 같은 겹: 같은 행 순서·같은 블록
    col = f"prod_{prod}"; out = []; t1 = time.time()
    for f in folds:
        tr, te = d.iloc[f.tr], d.iloc[f.te]
        tr = tr[np.isfinite(tr[col].values)].reset_index(drop=True); te = te[np.isfinite(te[col].values)].reset_index(drop=True)
        if len(tr) < 30 or len(te) < 5:
            continue
        k = T.fit_phys(tr); p_tr, p_te = T.phys_pred("P1", tr, k), T.phys_pred("P1", te, k)
        aa, bb = T.ls_affine(tr[col].values.astype(float), tr.alt_cm.values.astype(float))
        c_tr, c_te = aa + bb * tr[col].values, aa + bb * te[col].values
        tags = dict(exp="PA", product=prod, fold=f.name, n_train=len(tr), a_cal=aa, b_cal=bb, E=k["E"])
        out += [rows_of(te, te[col].values, method="Praw", seed=0, **tags), rows_of(te, c_te, method="Pcal", seed=0, **tags),
                rows_of(te, p_te, method="P1", seed=0, **tags)]
        for s in seeds:
            y = tr.alt_cm.values
            out.append(rows_of(te, p_te + a.lam * fit(X(tr), y - p_tr, X(te), s), method="R1", seed=s, **tags))
            res = fit(X(tr), y - c_tr, X(te), s)
            out.append(rows_of(te, c_te + a.lam * res, method="RP", seed=s, **tags)); out.append(rows_of(te, c_te + 1.0 * res, method="RP1", seed=s, **tags))
            out.append(rows_of(te, p_te + a.lam * fit(X(tr, [col]), y - p_tr, X(te, [col]), s), method="R1x", seed=s, **tags))
        print(f"[PA {region} {prod} {f.name}] tr {len(tr):,} te {len(te):,} {time.time()-t1:.0f}s", flush=True)
    o = pd.concat(out, ignore_index=True); o.to_csv(OUT / f"preds_PA_{region}_{prod}.csv.gz", index=False)
    print(f"[PA {region} {prod}] rows {len(o):,} {time.time()-t1:.0f}s", flush=True)


# ---------------------------------------------------------------- EXT
def run_ext():
    ex_raw = pd.read_csv(ROOT / a.ext / "labels_cell_year_v5.csv")
    regs = sorted(ex_raw.region.unique())
    ex = T.load_v5(ROOT / a.ext, regions=regs, products=True)
    ex = T.with_forcing(ex, FORC); ex = ex[T.usable_mask(ex, None)].reset_index(drop=True)
    src = d_all; k_src = T.fit_phys(src); E0 = k_src["E"]
    a_src, a_ex = E0 * src.e5_sqrt_tdd.values, E0 * ex.e5_sqrt_tdd.values
    out = []; t1 = time.time()
    # 소지역 안 다른 셀로 E 재보정(시험 셀의 모든 연도 제외). 행이 10개 미만이면 원천 E
    Eloc = np.full(len(ex), E0)
    for i, (r, c) in enumerate(zip(ex.region.values, ex.loc_id.values)):
        m = (ex.region.values == r) & (ex.loc_id.values != c)
        if m.sum() >= 10:
            Eloc[i] = T.ls_scale(ex.e5_sqrt_tdd.values[m], ex.alt_cm.values[m])
    p_loc = Eloc * ex.e5_sqrt_tdd.values
    tags = dict(exp="EXT", E_src=E0)
    out += [rows_of(ex, a_ex, method="P0", seed=0, **tags), rows_of(ex, p_loc, method="P1loc", seed=0, **tags)]
    ku_s = np.c_[T.phys_pred("P1", src, k_src), T.phys_pred("Ku", src, k_src)]; ku_e = np.c_[T.phys_pred("P1", ex, k_src), T.phys_pred("Ku", ex, k_src)]
    for s in seeds:
        out.append(rows_of(ex, fit(X(src), src.alt_cm.values, X(ex), s), method="D0", seed=s, **tags))
        out.append(rows_of(ex, fit(np.c_[X(src), ku_s], src.alt_cm.values, np.c_[X(ex), ku_e], s), method="F1", seed=s, **tags))
        res = fit(X(src), src.alt_cm.values - a_src, X(ex), s)
        out.append(rows_of(ex, a_ex + a.lam * res, method="R1", seed=s, **tags)); out.append(rows_of(ex, p_loc + a.lam * res, method="R1loc", seed=s, **tags))
        print(f"[EXT seed {s}] {time.time()-t1:.0f}s", flush=True)
    o = pd.concat(out, ignore_index=True)
    pcols = [c for c in ex.columns if c.startswith("prod_") and not c.endswith("_mean")]
    o = o.merge(ex[["cell_year_id"] + pcols], on="cell_year_id", how="left")
    o.to_csv(OUT / "preds_EXT.csv.gz", index=False); print(f"[EXT] rows {len(o):,} cells {ex.loc_id.nunique()} E_src {E0:.3f} {time.time()-t1:.0f}s", flush=True)


# ---------------------------------------------------------------- SHAP
def run_shap():
    rows = []; t1 = time.time()
    for region in a.regions.split(","):
        d = d_all[d_all.region == region].reset_index(drop=True); k = T.fit_phys(d); anc = T.phys_pred("P1", d, k)
        for kind, y in (("residual", d.alt_cm.values - anc), ("direct", d.alt_cm.values)):
            if a.dry:
                sv = np.zeros((len(d), len(feats) + 1))
            else:
                from catboost import CatBoostRegressor, Pool
                m = CatBoostRegressor(iterations=600, learning_rate=0.03, depth=6, l2_leaf_reg=3.0, random_seed=0, verbose=0, allow_writing_files=False, thread_count=a.threads)
                m.fit(X(d), y); sv = m.get_feature_importance(Pool(X(d)), type="ShapValues")
            sh = pd.DataFrame(sv[:, :-1], columns=feats); sh.insert(0, "loc_id", d.loc_id.values); sh.insert(1, "year", d.year.values); sh.insert(2, "block", d.block.values)
            sh["expected"] = sv[:, -1]; sh.to_csv(OUT / f"shap_{kind}_{region}.csv.gz", index=False)
            for j, fname in enumerate(feats):
                rows.append(dict(region=region, kind=kind, feature=fname, mean_abs=float(np.mean(np.abs(sv[:, j]))), E=k["E"], n=len(d)))
            print(f"[SHAP {region} {kind}] {time.time()-t1:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "shap_summary.csv", index=False)


exps = a.exp.split(",")
if "PA" in exps:
    for region in a.regions.split(","):
        for prod in a.products.split(","):
            run_pa(region, prod)
if "EXT" in exps:
    run_ext()
if "SHAP" in exps:
    run_shap()
print(f"[done] {time.time()-t0:.0f}s", flush=True)
