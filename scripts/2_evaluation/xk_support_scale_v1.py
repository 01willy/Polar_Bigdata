"""XK_support_scale(격자 크기별 오차). 등록 docs/EXPERIMENT_PLAN_FINAL_BATCH_ADDENDUM_XK_XL_2026-10-05.md(판정어 없는 서술 분석).

예측(지역 내, 알래스카·레나·캐나다, x25)
  0.5° 블록 5겹 교차검증(h42.cv_folds_of, 씨앗 키 (지역, r, split 0, n −1, draw 0))의 블록 밖 셀 예측. 방법은 지도 스크립트(map_alaska_v1.py)의
  nested_oof 와 같다. P1 = 학습 겹 라벨로 다시 구한 E_n·s(κ 10 수축, E0 = LG 원천 모드 x). R1 = WF6 R1(교차검증 λ): 학습 겹 안 블록 교차검증
  (RUnit.lam_cv, 씨앗 키 draw = 겹 번호 + 1)으로 λ ∈ {0.25, 0.5, 1.0} 를 고르고 catboost_lo seed 0·1 평균 잔차를 더한다.
  제품 원값: CCI v4(x25 의 cci_alt, 1997–2021 평균), CCI v5 1997–2023 평균, Wei 2026 2000–2024 평균, 알래스카는 Yi·Kimball 2001–2015 평균.
  제품 값은 XG 추출 표(xg_product_values_v1.csv 의 value_static_cm: cci5y, wei, yk)를 쓴다. 이 표가 모든 라벨 셀을 덮으므로 새로 표집하지 않는다.
격자 크기(지지 면적)
  1 km(라벨 셀 그대로), 0.05°, 0.1°, 0.25°. 격자 경계는 ERA5-Land 격자 경계(격자점 ± 0.05°)에 맞춘다: 색인 = floor((좌표 + 0.05) / g).
  g = 0.1 의 격자가 ERA5-Land 셀과 같고 0.05° 격자는 그 안에 들어간다. 같은 격자 안 라벨 평균과 예측 평균을 비교한다. 라벨 셀 3개 미만 격자는 뺀다.
  격자의 블록 = 그 격자 라벨 셀이 가장 많이 속한 0.5° 블록.
지표: RMSE(cm), 편향(예측 − 라벨, cm), 격자 수, 블록 재표집 95 % CI(셀 가중 = 격자 단위 평균, 블록 등가중 = 블록별 평균의 평균, 1,000회,
  seed_of('xk', 지역, 셀 집합, 마스크, 지지)). 같은 재표집에서 방법 − R1 의 RMSE 차와 CI 도 낸다. 판정어는 쓰지 않는다.
셀 집합: 'models' = P1·R1·CCI v4 가 유한한 셀(라벨 셀 거의 전부), 'common' = 지역의 모든 방법(알래스카는 Yi·Kimball 제외) 값이 유한한 셀,
  'with_yk'(알래스카) = Yi·Kimball 까지 유한한 셀. 레나는 Wei 값이 3,037셀 가운데 1,342셀에만 있어 'common' 이 작다(구현 결정, 결과 열람 뒤 추가한 집합).
마스크: 'none' 과 'wei5'(XG 셀 단위 5 km 규칙: Wei 학습 지점에서 5 km 안 셀, xg_leak_cells_v1.csv 의 target '<지역>|r', product 'wei').
  마스크는 모든 방법에 같이 적용한다(같은 셀 비교).
산출 data/processed/xbatch/XK_support_scale/{xk_oof_cells_v1.csv, xk_support_scale_v1.csv, xk_meta.json}
실행 nice -n 10 python3 scripts/2_evaluation/xk_support_scale_v1.py --threads 4
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "4" if __name__ == "__main__" else os.environ.get(_v, "4")

import argparse                                                                                         # noqa: E402
import json                                                                                             # noqa: E402
import resource                                                                                         # noqa: E402
import time                                                                                             # noqa: E402
from pathlib import Path                                                                                # noqa: E402

import numpy as np                                                                                      # noqa: E402
import pandas as pd                                                                                     # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "2_evaluation"))
sys.path.insert(0, str(ROOT / "src"))
import map_alaska_v1 as MA                                                                              # noqa: E402
from polar.h4_common import seed_of                                                                     # noqa: E402

OUT = ROOT / "data" / "processed" / "xbatch" / "XK_support_scale"
XG = ROOT / "data" / "processed" / "xbatch" / "XG_product_comparison"
REGIONS = ("Alaska", "Lena", "Canada")
SUPPORTS = (("1km", None), ("0.05deg", 0.05), ("0.1deg", 0.1), ("0.25deg", 0.25))
MIN_CELLS = 3
NBOOT = 1000
PROD = dict(cci5=("cci5y", "value_static_cm"), wei=("wei", "value_static_cm"), yk=("yk", "value_static_cm"))
METHODS = ("p1", "r1", "cci4", "cci5", "wei", "yk")
LABEL = dict(p1="Recalibrated Stefan", r1="Anchor + residual ML", cci4="CCI v4", cci5="CCI v5", wei="Wei 2026", yk="Yi-Kimball")


def oof_region(H54, a, region):
    df, t_idx, parent, E0, comp = MA.region_context(H54, a, region)
    nF = len(H54.FEATS)
    u = MA.make_unit(H54, a, region, parent, df, t_idx, np.zeros((1, nF), np.float32), np.ones(1), np.zeros(1), np.zeros(1), E0)
    t0 = time.time()
    cv = MA.nested_oof(H54, u, np.arange(u.nA))
    sec = round(time.time() - t0, 1)
    o = pd.DataFrame(dict(region=region, loc_id=df.loc_id.values[t_idx].astype(np.int64), lat=df.lat.values[t_idx], lon=df.lon.values[t_idx],
                          block=df.block.values[t_idx].astype(str), fold=cv["fid"], y=u.c.yA, s=u.c.sA, p1=cv["oof_p1"], r1=cv["oof_r1"],
                          cci4=df.cci_alt.values[t_idx].astype(float)))
    info = dict(n=int(len(o)), n_blocks=int(o.block.nunique()), E0=E0, source=comp, K=cv["K"], flag=cv["flag"], folds=cv["folds"], seconds=sec,
                fits=dict(n=dict(u.F.n), sec={k: round(v, 1) for k, v in u.F.sec.items()}, fail=dict(u.F.fail)))
    return o, info


def attach_products(o):
    v = pd.read_csv(XG / "xg_product_values_v1.csv")
    for k, (p, col) in PROD.items():
        s = v[v["product"] == p].drop_duplicates("loc_id").set_index("loc_id")[col]
        o[k] = s.reindex(o.loc_id.values).values.astype(float)
    m = pd.read_csv(XG / "xg_leak_cells_v1.csv")
    m = m[m["product"] == "wei"]
    within = {}
    for r in REGIONS:
        mm = m[m.target == f"{r}|r"].drop_duplicates("loc_id").set_index("loc_id").within_cell_d
        within.update({(r, int(k)): int(v) for k, v in mm.items()})
    o["wei5_within"] = [within.get((r, int(i)), -1) for r, i in zip(o.region, o.loc_id)]
    return o


def units_for(d, g):
    """격자 크기 g 의 격자 단위. 반환 (단위별 라벨 평균, 방법별 예측 평균 dict, 단위 블록, 단위 셀 수)."""
    if g is None:
        key = np.arange(len(d))
    else:
        iy = np.floor((d.lat.values + 0.05) / g).astype(np.int64)
        ix = np.floor((d.lon.values + 0.05) / g).astype(np.int64)
        key = pd.factorize(pd.Series(iy).astype(str) + "_" + pd.Series(ix).astype(str))[0]
    tab = pd.DataFrame(dict(u=key, y=d.y.values, block=d.block.values))
    cnt = tab.groupby("u").size()
    keep = cnt.index[cnt >= (1 if g is None else MIN_CELLS)]
    sel = tab.u.isin(keep).values
    y = tab[sel].groupby("u").y.mean()
    blk = tab[sel].groupby("u").block.agg(lambda s: s.value_counts().index[0])
    preds = {m: pd.Series(d[m].values[sel]).groupby(tab.u.values[sel]).mean().reindex(y.index).values for m in METHODS if m in d}
    return y.values, preds, blk.reindex(y.index).values, cnt.reindex(y.index).values


def boot_stats(err: dict, blk, seed, nboot=NBOOT):
    """err = 방법별 단위 오차(예측 − 라벨). 블록 재표집(복원 추출, 블록 수만큼) 1,000회의 셀 가중·블록 등가중 RMSE·편향과 R1 대비 RMSE 차."""
    ub, bi = np.unique(blk, return_inverse=True)
    nb = len(ub)
    C = np.bincount(bi, minlength=nb).astype(float)
    S2 = {m: np.bincount(bi, weights=e ** 2, minlength=nb) for m, e in err.items()}
    S1 = {m: np.bincount(bi, weights=e, minlength=nb) for m, e in err.items()}
    rng = np.random.RandomState(seed)
    W = np.stack([np.bincount(rng.randint(0, nb, nb), minlength=nb) for _ in range(nboot)]).astype(float)
    out = {}
    for m in err:
        cw_mse = (W @ S2[m]) / (W @ C)
        be_mse = (W @ (S2[m] / C)) / W.sum(1)
        cw_b = (W @ S1[m]) / (W @ C)
        be_b = (W @ (S1[m] / C)) / W.sum(1)
        out[m] = dict(cw_rmse=np.sqrt(cw_mse), be_rmse=np.sqrt(be_mse), cw_bias=cw_b, be_bias=be_b)
    return out, nb


def summarize(o):
    rows = []
    for region in REGIONS:
        d0 = o[o.region == region]
        sets = [("models", ["p1", "r1", "cci4"]), ("common", [m for m in METHODS if m != "yk"])]
        if region == "Alaska":
            sets.append(("with_yk", list(METHODS)))
        for cs, ms in sets:
            ok = np.all(np.isfinite(d0[ms].values.astype(float)), axis=1) & np.isfinite(d0.y.values)
            for mask in ("none", "wei5"):
                sel = ok & ((d0.wei5_within.values == 0) if mask == "wei5" else True)
                d = d0[sel]
                for sup, g in SUPPORTS:
                    y, preds, blk, ncell = units_for(d, g)
                    if len(y) == 0:
                        continue
                    err = {m: preds[m] - y for m in ms}
                    bs, nb = boot_stats(err, blk, seed_of("xk", region, cs, mask, sup))
                    base = dict(region=region, cell_set=cs, mask=mask, support=sup, support_deg=g if g is not None else 0.009, n_units=int(len(y)),
                                n_cells=int(ncell.sum()), n_blocks=int(nb), n_cells_region=int(len(d0)), n_cells_excluded_nonfinite=int((~ok).sum()),
                                n_cells_masked=int((ok & ~sel).sum()))
                    ub, bi = np.unique(blk, return_inverse=True)
                    for m in ms:
                        e = err[m]
                        pe_cw = float(np.sqrt(np.mean(e ** 2)))
                        pb = np.array([np.mean(e[bi == k] ** 2) for k in range(len(ub))])
                        pe_be = float(np.sqrt(np.mean(pb)))
                        b = bs[m]
                        r = dict(base, method=m, method_label=LABEL[m], rmse_cw=pe_cw, rmse_be=pe_be, bias_cw=float(np.mean(e)),
                                 bias_be=float(np.mean([np.mean(e[bi == k]) for k in range(len(ub))])),
                                 rmse_cw_lo=float(np.percentile(b["cw_rmse"], 2.5)), rmse_cw_hi=float(np.percentile(b["cw_rmse"], 97.5)),
                                 rmse_be_lo=float(np.percentile(b["be_rmse"], 2.5)), rmse_be_hi=float(np.percentile(b["be_rmse"], 97.5)),
                                 bias_cw_lo=float(np.percentile(b["cw_bias"], 2.5)), bias_cw_hi=float(np.percentile(b["cw_bias"], 97.5)),
                                 bias_be_lo=float(np.percentile(b["be_bias"], 2.5)), bias_be_hi=float(np.percentile(b["be_bias"], 97.5)))
                        if m != "r1":
                            dcw = b["cw_rmse"] - bs["r1"]["cw_rmse"]; dbe = b["be_rmse"] - bs["r1"]["be_rmse"]
                            r.update(d_rmse_vs_r1_cw=pe_cw - float(np.sqrt(np.mean(err["r1"] ** 2))),
                                     d_rmse_vs_r1_cw_lo=float(np.percentile(dcw, 2.5)), d_rmse_vs_r1_cw_hi=float(np.percentile(dcw, 97.5)),
                                     d_rmse_vs_r1_be=pe_be - float(np.sqrt(np.mean([np.mean(err["r1"][bi == k] ** 2) for k in range(len(ub))]))),
                                     d_rmse_vs_r1_be_lo=float(np.percentile(dbe, 2.5)), d_rmse_vs_r1_be_hi=float(np.percentile(dbe, 97.5)))
                        rows.append(r)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--reuse-oof", action="store_true", help="xk_oof_cells_v1.csv 가 있으면 적합하지 않고 다시 집계만 한다")
    args = ap.parse_args(argv)
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    infos = {}
    oof_path = OUT / "xk_oof_cells_v1.csv"
    if args.reuse_oof and oof_path.exists():
        o = pd.read_csv(oof_path, dtype={"block": str})
        old = json.loads((OUT / "xk_meta.json").read_text())
        infos = old.get("regions", {})
    else:
        H54, a = MA.load_h54(args.threads)
        parts = []
        for r in REGIONS:
            d, info = oof_region(H54, a, r)
            parts.append(d); infos[r] = info
            print(f"[xk] {r}: 셀 {info['n']:,} · 블록 {info['n_blocks']} · λ 겹별 {[f['lam'] for f in info['folds']]} · {info['seconds']}s", flush=True)
        o = attach_products(pd.concat(parts, ignore_index=True))
        o.to_csv(oof_path, index=False, float_format="%.6g")
    t0 = time.time()
    tab = summarize(o)
    tab.to_csv(OUT / "xk_support_scale_v1.csv", index=False, float_format="%.6g")
    meta = dict(created=time.strftime("%Y-%m-%d %H:%M %Z"), script="scripts/2_evaluation/xk_support_scale_v1.py", script_sha256=MA.sha256_file(Path(__file__)),
                registration="docs/EXPERIMENT_PLAN_FINAL_BATCH_ADDENDUM_XK_XL_2026-10-05.md(서술, 판정어 없음)",
                design=dict(supports=[s for s, _ in SUPPORTS], grid_rule="floor((좌표 + 0.05)/g), ERA5-Land 격자 경계에 맞춤", min_cells=MIN_CELLS,
                            nboot=NBOOT, weights=["셀 가중(격자 단위 평균)", "블록 등가중(블록별 평균의 평균)"], unit_block="격자 라벨 셀의 최빈 0.5° 블록",
                            methods=LABEL, products={k: dict(xg_product=p, column=c) for k, (p, c) in PROD.items()} | {"cci4": "x25 cci_alt"},
                            cell_sets=dict(models="P1·R1·CCI v4 가 유한한 셀(결과 열람 뒤 추가: 레나 Wei 결측 1,695셀)", common="지역의 모든 방법(알래스카 Yi·Kimball 제외) 값이 유한한 셀",
                                           with_yk="알래스카, Yi·Kimball 까지 유한한 셀"),
                            masks=dict(none="마스크 없음", wei5="XG 셀 단위 5 km 규칙(Wei 학습 지점 5 km 안 셀 제외, 모든 방법 공통)"),
                            oof="map_alaska_v1.nested_oof(블록 5겹, 겹 안 λ 교차검증과 E_n 재추정)"),
                regions=infos, inputs=dict(xg_values=MA.file_info(XG / "xg_product_values_v1.csv"), xg_cells=MA.file_info(XG / "xg_leak_cells_v1.csv"),
                                           data=MA.file_info(ROOT / "data/processed/fidelity_base_v3.csv")),
                outputs=dict(oof=MA.file_info(oof_path), table=MA.file_info(OUT / "xk_support_scale_v1.csv")), summarize_s=round(time.time() - t0, 1),
                elapsed_s=round(time.time() - t_start, 1), max_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                threads=int(args.threads))
    (OUT / "xk_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    show = tab[(tab.cell_set == "common") & (tab["mask"] == "none") & tab.method.isin(["p1", "r1"])]
    print(show[["region", "support", "n_units", "method", "rmse_cw", "rmse_cw_lo", "rmse_cw_hi"]].to_string(index=False), flush=True)
    print(f"[done] {meta['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
