"""지도 과제(MAP) · 알래스카 1 km 지도 예측. 지역 내 최종 방법(라벨 전량의 재보정 Stefan + 잔차 ML, λ 블록 교차검증)을 격자에 적용한다.

지시(2026-10-05)
  라벨  data/processed/fidelity_base_v3.csv 의 알래스카 F4_direct 셀 전량(h40.Data 의 대상 'Alaska', 13,606셀, 0.5° 블록 74개).
  방법  WF6 의 R1(교차검증 λ)과 같은 정의를 h54 의 함수로 계산한다(h54 는 importlib 로 읽고 고치지 않는다).
        P1 = E_n·s, s = e5_sqrt_tdd, E_n = (n·E_ls + κ·E0)/(n + κ), κ = 10, E_ls = Σ s·y / Σ s², E0 = LG 원천(모드 x, 100 km 버퍼) 최소제곱 계수
        (h54.build_rctx 와 같은 계산). R1 = E_n·s + λ·ḡ, g = catboost_lo(h40.cb_fit: 200회, 깊이 3, 학습률 0.05, l2 3)를 잔차 y − E_n·s 에
        x25 로 적합한 것, ḡ = seed 0·1 평균(h54 SEEDS). λ ∈ {0.25, 0.5, 1.0} 은 RUnit.lam_cv(h42.cv_folds_of 블록 5묶음, seed 0 적합,
        묶음마다 E_n 재계산, 동률은 작은 λ)로 고른다. 지도에는 분할이 없으므로 cv_folds_of 의 씨앗 키는 (Alaska, r, split 0, n −1, draw 0)이다.
        RUnit 의 __init__ 만 바꾼 얇은 하위 클래스(MapUnit)를 쓴다. 이유: RUnit.__init__ 은 채점 저장소(BlockStore)를 B 행의 관측값으로 만드는데
        격자 셀에는 관측값이 없다. coefs, folds, lam_cv, fit_resid, anchor_A 는 h54 의 메서드 그대로다.
  구간  90 % 구간은 상수 폭 분할 등각(split-conformal)이다. 점수 = |y − ŷ_oof|, ŷ_oof 는 0.5° 블록 5겹 교차검증(위와 같은 묶음)의 블록 밖 R1 예측이고
        각 겹에서 E_n 과 λ 를 학습 겹 안에서 다시 구한다(안쪽 λ 교차검증의 씨앗 키 draw = 겹 번호 + 1). q = 점수의 ⌈(n + 1)·0.9⌉ 번째 작은 값.
        격자 구간 = [max(R1 − q, 0), R1 + q]. 폭은 하한 절단 셀을 빼면 2q 로 일정하다.
  보정량 R1 − P1 = λ·ḡ(양수 = 잔차 ML 이 더 깊게 예측).
  외삽  h49.extrap_flags 를 그대로 쓴다. 학습 행(알래스카 라벨 전량)의 x25 연속 24열(cci_valid 제외) 0.5–99.5 백분위 밖 열 수 n_extrap_out,
        결측 열 수 n_x25_missing, 범주 extrap_cat = min(합, 3). extrap_flag = n_extrap_out ≥ 1(지시: 어느 공변량이든 학습 범위 밖).
입력  data/processed/map_alaska/alaska_grid_x25_v1.csv.gz(scripts/1_data_prep/build_map_grid_alaska_v1.py 산출)
산출  data/processed/map_alaska/alaska_pred_v1.csv.gz, alaska_pred_v1_meta.json, alaska_cv_oof_v1.csv(라벨 셀의 블록 밖 예측)
실행(ROOT, 공유 서버 규칙: CatBoost CPU 스레드 4, nice 10)
  nice -n 10 python3 scripts/2_evaluation/map_alaska_v1.py --threads 4
  --labels-only: 격자 없이 라벨 단계(λ, 교차검증, q)만 계산해 메타 초안을 쓴다.
"""
from __future__ import annotations

import os
import sys


def _peek_threads(default="4"):
    av = sys.argv
    for i, v in enumerate(av):
        if v == "--threads" and i + 1 < len(av):
            return av[i + 1]
    return default


for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    if __name__ == "__main__":
        os.environ[_v] = str(min(int(_peek_threads()), 4))
    else:
        os.environ.setdefault(_v, "4")

import argparse                                                                                         # noqa: E402
import hashlib                                                                                          # noqa: E402
import importlib.util                                                                                   # noqa: E402
import json                                                                                             # noqa: E402
import math                                                                                             # noqa: E402
import resource                                                                                         # noqa: E402
import time                                                                                             # noqa: E402
from pathlib import Path                                                                                # noqa: E402

import numpy as np                                                                                      # noqa: E402
import pandas as pd                                                                                     # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
for _p in (str(ROOT / "src"), str(ROOT / "scripts" / "3_deep_learning"), str(ROOT / "scripts" / "2_evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TARGET = "Alaska"
SPLIT_MAP = 0                      # 지도: 분할 없음(cv_folds_of 씨앗 키)
N_ALL = -1                         # 라벨 전량
D_OUTER = 0
Q_LEVEL = 0.9


def log(*a):
    print(*a, flush=True)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def file_info(p) -> dict:
    p = Path(p)
    rel = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
    return dict(path=rel, bytes=int(p.stat().st_size), sha256=sha256_file(p)) if p.exists() else dict(path=rel, exists=False)


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_h54(threads: int):
    """h54(그 안에서 h40, h42)를 읽고 WF6 형식 인자를 만든다(seed 0·1, 스레드 상한 4)."""
    H54 = _load("h54_workflow", "scripts/3_deep_learning/h54_workflow.py")
    a = H54.parse_args(["--exps", "wf6", "--threads", str(int(threads))])
    return H54, a


def unit_class(H54):
    class MapUnit(H54.RUnit):
        """RUnit 의 __init__ 만 바꾼다(채점 저장소 없음). 나머지 메서드는 h54 그대로다."""

        def __init__(self, a, c):
            self.a, self.c, self.exp, self.variant, self.dry = a, c, "wf6", "map", False
            self.F = H54.WFFitter(H54.h40_args(a), False)
            self.F.nrow_fn = lambda ax, lr, info: int(info.get("nrow", 1))
            self.name = f"{c.target}|{c.mode}"
            self.seeds = list(a.SEEDS)
            self.nA, self.nB = len(c.yA), len(c.yB)
            self.notes, self._km, self._std, self._di = {}, {}, None, None
            self.rows, self.n_rows, self.n_bad = [], 0, 0
            self.xstores = []
    return MapUnit


def region_context(H54, a, target: str):
    """(df, 대상 라벨 색인, parent, E0, 원천 구성). E0 는 h54.build_rctx 와 같은 계산(LG 원천, 모드 x, 버퍼 100 km)."""
    H = H54.H
    _, D = H54.get_data(a)
    df = D.df
    t_idx = D.target_idx(target)
    _, parent, src_idx, comp = D.source_idx(target, "x")
    ok = np.isfinite(df.y.values[src_idx]) & np.isfinite(df.s.values[src_idx])
    E0 = H.ls_E(df.y.values[src_idx[ok]], df.s.values[src_idx[ok]])
    return df, t_idx, parent, float(E0), comp


def make_unit(H54, a, target, parent, df, lab_idx, XB, sB, latB, lonB, E0, split=SPLIT_MAP):
    """A = 라벨 행(lab_idx), B = 예측 행(XB). B 의 관측값은 NaN, 블록은 0."""
    F = list(H54.FEATS)
    nb = len(XB)
    c = H54.RCtx(target, split, parent, df[F].values[lab_idx].astype(np.float32), df.y.values[lab_idx], df.s.values[lab_idx],
                 df.block.values[lab_idx], df.lat.values[lab_idx], df.lon.values[lab_idx], np.asarray(XB, np.float32), np.full(nb, np.nan),
                 np.asarray(sB, float), np.zeros(nb, int), np.asarray(latB, float), np.asarray(lonB, float), E0,
                 meta=dict(note="map: A = 라벨 전량, B = 예측 행"))
    return unit_class(H54)(a, c)


def fit_r1(H54, u, sel, n_key, d_key, preds):
    """WF6 R1(교차검증 λ): E_n = coefs(sel), λ = lam_cv('R1', catboost_lo, x25), g = seed 0·1 의 fit_resid 평균."""
    E1, E_ls = u.coefs(sel)
    lam, K, flag = u.lam_cv("R1", H54.LO, "x25", sel, n_key, d_key)
    gs = []
    for seed in u.seeds:
        (g,) = u.fit_resid(H54.LO, "R1", sel, E1 * u.c.sA[sel], seed, n_key, d_key, preds=preds)
        gs.append(np.asarray(g, float))
    return dict(E1=float(E1), E_ls=float(E_ls), lam=float(lam), K=int(K), flag=str(flag or ""), g=np.mean(gs, axis=0),
                g_seed_sd=float(np.mean(np.std(gs, axis=0))) if len(gs) > 1 else 0.0)


def nested_oof(H54, u, sel, n_key=N_ALL, d_key=D_OUTER):
    """블록 5겹(cv_folds_of, 씨앗 키 d_key)의 블록 밖 예측. 겹마다 E_n 과 λ(안쪽 블록 교차검증, 씨앗 키 d_key + 1 + j)를 학습 겹에서 구한다."""
    fid, K, flag = u.folds(sel, n_key, d_key)
    r1 = np.full(len(sel), np.nan); p1 = np.full(len(sel), np.nan)
    per = []
    for j in range(int(K)):
        m = fid == j
        tr, te = sel[~m], sel[m]
        r = fit_r1(H54, u, tr, n_key, d_key + 1 + j, preds=[u.c.XA[te]])
        p1[m] = r["E1"] * u.c.sA[te]
        r1[m] = r["E1"] * u.c.sA[te] + r["lam"] * r["g"]
        per.append(dict(fold=j, n_train=int(len(tr)), n_test=int(len(te)), nb_test=int(len(np.unique(u.c.blkA[te]))), E1=r["E1"], lam=r["lam"],
                        inner_K=r["K"], inner_flag=r["flag"]))
    return dict(fid=fid, K=int(K), flag=str(flag or ""), oof_r1=r1, oof_p1=p1, folds=per)


def conformal_q(scores, level=Q_LEVEL) -> float:
    s = np.sort(np.asarray(scores, float)[np.isfinite(scores)])
    k = int(math.ceil((len(s) + 1) * level))
    return float(s[min(k, len(s)) - 1])


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="알래스카 1 km 지도 예측(지역 내 R1, 교차검증 λ)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--grid", default="data/processed/map_alaska/alaska_grid_x25_v1.csv.gz")
    ap.add_argument("--out-dir", default="data/processed/map_alaska")
    ap.add_argument("--labels-only", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t_start = time.time()
    timing = {}
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    H54, a = load_h54(args.threads)
    h49 = _load("h49_transfer_map", "scripts/2_evaluation/h49_transfer_map.py")
    F = list(H54.FEATS)
    t0 = time.time()
    df, t_idx, parent, E0, comp = region_context(H54, a, TARGET)
    assert np.isfinite(df.y.values[t_idx]).all() and np.isfinite(df.s.values[t_idx]).all(), "라벨 또는 s 결측"
    timing["data"] = round(time.time() - t0, 1)
    log(f"[data] {TARGET} 라벨 {len(t_idx):,}셀 · 블록 {len(np.unique(df.block.values[t_idx]))} · E0 {E0:.4f} · 원천 {comp}")

    grid = None
    if not args.labels_only:
        t0 = time.time()
        grid = pd.read_csv(ROOT / args.grid, dtype={"cell_id": str}, low_memory=False)
        timing["read_grid"] = round(time.time() - t0, 1)
        XB, sB = grid[F].values.astype(np.float32), grid.e5_sqrt_tdd.values.astype(float)
        latB, lonB = grid.lat.values, grid.lon.values
        log(f"[grid] {len(grid):,}셀 · 표시 {int((grid.gray == 0).sum()):,} · 읽기 {timing['read_grid']}s")
    else:
        XB, sB, latB, lonB = np.zeros((1, len(F)), np.float32), np.ones(1), np.zeros(1), np.zeros(1)
    u = make_unit(H54, a, TARGET, parent, df, t_idx, XB, sB, latB, lonB, E0)
    sel = np.arange(u.nA)

    # 1) 블록 교차검증의 블록 밖 예측(구간 보정과 기록용)
    t0 = time.time()
    cv = nested_oof(H54, u, sel)
    timing["cv"] = round(time.time() - t0, 1)
    y = u.c.yA
    res_r1, res_p1 = cv["oof_r1"] - y, cv["oof_p1"] - y
    q = conformal_q(np.abs(res_r1))
    cv_stats = dict(K=cv["K"], flag=cv["flag"], folds=cv["folds"], n=int(len(y)),
                    rmse_r1=float(np.sqrt(np.mean(res_r1 ** 2))), rmse_p1=float(np.sqrt(np.mean(res_p1 ** 2))),
                    bias_r1=float(np.mean(res_r1)), bias_p1=float(np.mean(res_p1)), q90=q,
                    coverage_in_sample_check=float(np.mean(np.abs(res_r1) <= q)))
    log(f"[cv] 블록 {cv['K']}겹 · RMSE R1 {cv_stats['rmse_r1']:.2f} · P1 {cv_stats['rmse_p1']:.2f} cm · q90 {q:.2f} cm · λ 겹별 "
        f"{[f['lam'] for f in cv['folds']]} · {timing['cv']}s")
    oof = pd.DataFrame(dict(loc_id=df.loc_id.values[t_idx], lat=u.c.latA, lon=u.c.lonA, block=u.c.blkA, fold=cv["fid"], y=y, s=u.c.sA,
                            oof_p1=cv["oof_p1"], oof_r1=cv["oof_r1"]))
    oof.to_csv(out / "alaska_cv_oof_v1.csv", index=False, float_format="%.6g")

    # 2) 라벨 전량 적합(최종)
    t0 = time.time()
    fin = fit_r1(H54, u, sel, N_ALL, D_OUTER, preds=[u.c.XB])
    timing["final_fit"] = round(time.time() - t0, 1)
    log(f"[fit] E_ls {fin['E_ls']:.4f} · E_n {fin['E1']:.4f} · λ {fin['lam']} (K {fin['K']}) · {timing['final_fit']}s")

    meta = dict(created=time.strftime("%Y-%m-%d %H:%M %Z"), script="scripts/2_evaluation/map_alaska_v1.py", script_sha256=sha256_file(Path(__file__)),
                reused=dict(h54=file_info(ROOT / "scripts/3_deep_learning/h54_workflow.py"), h40=file_info(ROOT / "scripts/3_deep_learning/h40_label_grid.py"),
                            h42=file_info(ROOT / "scripts/3_deep_learning/h42_label_grid_ext.py"), h49=file_info(ROOT / "scripts/2_evaluation/h49_transfer_map.py")),
                labels=dict(target=TARGET, n=int(len(t_idx)), n_blocks=int(len(np.unique(df.block.values[t_idx]))),
                            regions=pd.Series(df.region.values[t_idx]).value_counts().to_dict(),
                            source="data/processed/fidelity_base_v3.csv F4_direct(polar.m1_core.load_base)", data_sha=file_info(PROC / "fidelity_base_v3.csv")),
                model=dict(E0=E0, E0_source=comp, kappa=float(H54.KAPPA), E_ls=fin["E_ls"], E_n=fin["E1"], lam=fin["lam"], lam_grid=list(H54.LAMS),
                           lam_cv_folds=fin["K"], lam_cv_flag=fin["flag"], seeds=list(u.seeds), learner="catboost_lo(h40.cb_fit: 200, depth 3, lr 0.05, l2 3)",
                           cv_seed_key=dict(target=TARGET, mode="r", split=SPLIT_MAP, n=N_ALL, draw=D_OUTER), features=F, threads=int(a.threads),
                           g_seed_sd_mean=fin["g_seed_sd"]),
                interval=dict(kind="constant-width split-conformal on 0.5° block 5-fold CV residuals(|y − ŷ_oof|, R1, 겹 안 λ·E_n 재추정)",
                              level=Q_LEVEL, q=q, width=2 * q, rule="[max(R1 − q, 0), R1 + q]"),
                cv=cv_stats, timing_s=timing)
    if args.labels_only:
        (out / "alaska_pred_v1_meta_labels_only.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
        log(f"[done] labels-only · {time.time() - t_start:.0f}s")
        return

    # 3) 격자 예측, 구간, 보정량, 외삽
    p1 = fin["E1"] * sB
    corr = fin["lam"] * fin["g"]
    r1 = p1 + corr
    lo, hi = np.maximum(r1 - q, 0.0), r1 + q
    shown = (grid.gray.values == 0) & np.isfinite(r1)
    cont = [c_ for c_ in F if c_ != "cci_valid"]
    Xs = df[cont].values[t_idx].astype(float)
    Xg = grid[cont].values.astype(float)
    ex, ex_info = h49.extrap_flags(Xs, Xg, cont, shown)
    pred = pd.DataFrame(dict(cell_id=grid.cell_id.values, lat=grid.lat.values, lon=grid.lon.values, ky=grid.ky.values, kx=grid.kx.values,
                             gray=grid.gray.values, gray_reason=grid.gray_reason.fillna("").values, land=grid.land.values,
                             pred_r1=r1, pred_p1=p1, corr_r1_minus_p1=corr, lo90=lo, hi90=hi, width90=hi - lo))
    pred = pd.concat([pred, ex.reset_index(drop=True)], axis=1)
    pred["extrap_flag"] = (pred.n_extrap_out >= 1).astype(int)
    for c_ in ("pred_r1", "pred_p1", "corr_r1_minus_p1", "lo90", "hi90", "width90"):
        pred.loc[~np.isfinite(pred[c_].values), c_] = np.nan
    ppath = out / "alaska_pred_v1.csv.gz"
    pred.to_csv(ppath, index=False, compression=dict(method="gzip", mtime=0), float_format="%.6g")

    def stats(v):
        v = np.asarray(v, float)[shown]
        v = v[np.isfinite(v)]
        return dict(n=int(len(v)), min=float(v.min()), p01=float(np.percentile(v, 1)), p05=float(np.percentile(v, 5)), p25=float(np.percentile(v, 25)),
                    p50=float(np.median(v)), p75=float(np.percentile(v, 75)), p95=float(np.percentile(v, 95)), p99=float(np.percentile(v, 99)),
                    max=float(v.max()), mean=float(v.mean())) if len(v) else dict(n=0)
    meta.update(
        grid=dict(file=file_info(ROOT / args.grid), meta=file_info((ROOT / args.grid).with_name("alaska_grid_x25_v1_meta.json"))),
        counts=dict(n_cells=int(len(pred)), n_shown=int(shown.sum()), n_gray=int((grid.gray == 1).sum()), n_pred_finite=int(np.isfinite(r1).sum()),
                    n_lo_clipped=int((np.isfinite(r1) & (r1 - q < 0)).sum()), n_extrap_flag_shown=int(pred.extrap_flag.values[shown].sum()),
                    frac_extrap_flag_shown=float(pred.extrap_flag.values[shown].mean()) if shown.any() else None,
                    extrap_cat_shown={str(k): int(v) for k, v in pd.Series(pred.extrap_cat.values[shown]).value_counts().sort_index().items()}),
        shown_stats=dict(pred_r1=stats(r1), pred_p1=stats(p1), corr=stats(corr), width90=stats(hi - lo)),
        extrap=ex_info,
        effective_resolution=("1 km 는 표시 해상도이다. 기후 8열과 s 는 ERA5-Land 0.1°, 토양 9열은 SoilGrids 약 5 km 창, CCI ALT 는 약 1 km, 지형 6열만 30 m "
                              "DEM 창이다. WF9 에서 알래스카 지역 내 R1 의 이득은 ERA5 격자 사이 성분이었다(C8)"),
        outputs=dict(pred=file_info(ppath), oof=file_info(out / "alaska_cv_oof_v1.csv")),
        timing_s=timing, elapsed_s=round(time.time() - t_start, 1), max_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        threads=dict(OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS"), catboost=int(a.threads), nice=os.nice(0)),
        fits=dict(n=dict(u.F.n), sec={k: round(v, 1) for k, v in u.F.sec.items()}, fail=dict(u.F.fail), errors=list(u.F.errors)))
    (out / "alaska_pred_v1_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    log(f"[pred] {ppath} · 셀 {len(pred):,} · 표시 {int(shown.sum()):,} · 외삽 표시 비율 {meta['counts']['frac_extrap_flag_shown']:.3f} · "
        f"R1 중앙값 {meta['shown_stats']['pred_r1'].get('p50', float('nan')):.1f} cm")
    log(f"[done] {timing} · 합계 {time.time() - t_start:.0f}s · 최대 RSS {meta['max_rss_mb']:.0f} MB")


if __name__ == "__main__":
    main()
