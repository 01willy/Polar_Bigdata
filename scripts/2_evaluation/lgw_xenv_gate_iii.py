"""교차 환경 점검 (iii). GPU 학습기(FT-Transformer) 조각의 로컬 재적합과 Rescale 조각의 대조.

근거: docs/EXECUTION_PLAN_REMAINING_2026-09-30.md 6.1(LG 개정 15 초안의 (d)). 등록은 LG 개정 15 의 커밋으로 효력이 생긴다.
이 스크립트는 대조만 한다(학습 없음, GPU 없음). 재적합 조각은 scripts/local/run_lgx_gpu_local.sh 가
별도 출력 폴더(data/processed/lgw/xenv/lgx_gpu_1, lgx_gpu_2)에 만든다.

대상 단위(기본값, Rescale 에서 끝난 FT-T 조각 가운데 작은 것):
  lgxnn__gpu__Russia_W__x__s1__ftt, lgxr0__gpu__Russia_W__x__s1__ftt, lgxg__gpu__AL-2__x__s4__ftt, lgxg__gpu__AL-2__x__s5__ftt
짝: (로컬 1, Rescale), (로컬 2, Rescale), (로컬 1, 로컬 2).
대조: 두 조각의 blocksse.npz 를 h4_common.load_stores 로 읽고, 공통 (저장소, 분할)의 공통 키마다 셀 가중 RMSE 의 절대 차를 낸다.
키 분류: 물리식 = 학습기 none 이고 방법 P0–P3(허용 차 1e-9 cm), 신경망 = 그 밖.
출력 제한: 키별 절대 차와 그 분포(중앙값, 최댓값, 한계 초과 키 수)만 낸다. RMSE 값, 방법 사이의 Δ, 판정 문구는 계산하거나 출력하지 않는다.

계속 규칙(LG 개정 15 초안):
  물리식 키의 |로컬 1 − Rescale| 가 모두 1e-9 cm 이하이고(아니면 '멈춤: 물리식 불일치', 자료 경로나 구현 차이),
  신경망 키에서 (가) |로컬 1 − Rescale| 가 --eq-cm(0.5 cm)를 넘는 키가 0 이거나
  (나) |로컬 1 − Rescale| 의 중앙값 ≤ --ratio-max(3) × |로컬 1 − 로컬 2| 의 중앙값(로컬 반복 차가 --det-floor-cm 1e-6 cm 를 넘을 때만)
  이면 '계속'.
  (다) 로컬 반복 차의 중앙값이 --det-floor-cm 이하(로컬 재적합이 사실상 결정적)이면 (나) 대신 |로컬 1 − Rescale| 의 중앙값
  ≤ --med-cm(0.1 cm) 이고 --eq-cm 를 넘는 키의 비율 ≤ --frac-over-max(5 %) 이면 '계속'.
  (가)–(다) 모두 아니면 '멈춤: 사용자 보고'. 로컬 2 가 없으면 (나)·(다)를 볼 수 없으므로 (가)만으로 정하고 '잠정'을 붙인다.
  (다)는 2026-09-30 심사 지적(로컬 반복 차가 0 이면 (나)가 3e-6 cm 기준으로 퇴화한다)에 따라 결과 전에 더했다(LG 개정 15 (f)).

보충 대조(--supplement): 이어 실행 폴더에서 로컬로 다시 적합한 Rescale 완료 단위(기본 레나 x g45 분할 5 FT-T)를 Rescale 조각과
대조한다. 계속 규칙에는 쓰지 않고, 이어 돌릴 대상과 같은 규모의 플랫폼 차이를 보고하는 데만 쓴다(--gate-csv, --summary 를 따로 준다).

산출
  data/processed/lgw/lgw_xenv_gate_iii.csv            키별 절대 차(단위, 짝, 저장소, 분할, 키, 분류, 절대 차). RMSE 값은 싣지 않는다
  data/processed/lgw/lgw_xenv_gate_iii_summary.json   짝·분류별 분포, 단위별 설정 해시 대조, 판 정보, 계속 규칙 결과

실행(ROOT): nice -n 10 python3 scripts/2_evaluation/lgw_xenv_gate_iii.py
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                                                                   # noqa: E402
import pandas as pd                                                                                  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from polar.h4_common import load_stores                                                              # noqa: E402

PHYS_METHODS = ("P0", "P1", "P2", "P3")
TOL_PHYS = 1e-9
DEFAULT_UNITS = ("lgxnn__gpu__Russia_W__x__s1__ftt", "lgxr0__gpu__Russia_W__x__s1__ftt",
                 "lgxg__gpu__AL-2__x__s4__ftt", "lgxg__gpu__AL-2__x__s5__ftt")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="교차 환경 점검 (iii): FT-T 조각의 로컬 재적합과 Rescale 조각 대조(대조만)")
    ap.add_argument("--ref-dir", default="data/processed/lgx/shards", help="Rescale 조각(읽기 전용)")
    ap.add_argument("--run1-dir", default="data/processed/lgw/xenv/lgx_gpu_1/shards", help="로컬 재적합 1회차 조각")
    ap.add_argument("--run2-dir", default="data/processed/lgw/xenv/lgx_gpu_2/shards", help="로컬 재적합 2회차 조각(없으면 (나)를 보지 않는다)")
    ap.add_argument("--units", default=",".join(DEFAULT_UNITS), help="조각 이름(접미사 _unit.json 등을 뺀 것)의 쉼표 목록")
    ap.add_argument("--eq-cm", type=float, default=0.5, help="계속 규칙 (가)의 한계(동등성 한계와 같은 0.5 cm)")
    ap.add_argument("--ratio-max", type=float, default=3.0, help="계속 규칙 (나)의 배율")
    ap.add_argument("--det-floor-cm", type=float, default=1e-6, help="로컬 반복 차 중앙값이 이 값 이하이면 (나) 대신 (다)를 쓴다")
    ap.add_argument("--med-cm", type=float, default=0.1, help="계속 규칙 (다)의 |로컬 1 − Rescale| 중앙값 상한")
    ap.add_argument("--frac-over-max", type=float, default=0.05, help="계속 규칙 (다)의 --eq-cm 초과 키 비율 상한")
    ap.add_argument("--supplement", action="store_true", help="보충 대조: 짝 (로컬 1, Rescale)만, 계속 규칙 없음")
    ap.add_argument("--gate-csv", default="data/processed/lgw/lgw_xenv_gate_iii.csv")
    ap.add_argument("--summary", default="data/processed/lgw/lgw_xenv_gate_iii_summary.json")
    return ap.parse_args(argv)


def _abs(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def key_class(k):
    learner = str(k[1]) if len(k) > 1 else ""
    return "phys" if (learner in ("none", "") and str(k[0]) in PHYS_METHODS) else "ml"


def unit_files(d: Path, base: str):
    return dict(unit=d / f"{base}_unit.json", npz=d / f"{base}_blocksse.npz")


def unit_meta(d: Path, base: str):
    p = unit_files(d, base)["unit"]
    if not p.exists():
        return None
    try:
        u = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return {k: u.get(k) for k in ("cfg_hash", "cfg_common", "code_sha", "code_sha_h40", "status", "device", "threads", "elapsed_s")}


def compare(da: Path, db: Path, base: str, pair: str):
    """두 조각의 공통 키 절대 차. 반환 (행 목록, 단위 요약). 조각이 없으면 (None, 사유)."""
    fa, fb = unit_files(da, base), unit_files(db, base)
    for f in (fa["npz"], fb["npz"]):
        if not f.exists():
            return None, dict(unit=base, pair=pair, missing=str(f.relative_to(ROOT) if f.is_relative_to(ROOT) else f))
    sa, sb = load_stores(fa["npz"]), load_stores(fb["npz"])
    rows, n_nan, blocks_ok = [], 0, True
    common_stores = sorted(set(sa) & set(sb), key=lambda t: (str(t[0]), int(t[1])))
    for sk in common_stores:
        A, B = sa[sk], sb[sk]
        if not (np.array_equal(A.blocks, B.blocks) and np.array_equal(A.ncell, B.ncell)):
            blocks_ok = False
        kb = set(B.keys)
        for k in A.keys:
            if k not in kb:
                continue
            ra, rb = float(A.rmse(k)), float(B.rmse(k))
            if not (np.isfinite(ra) and np.isfinite(rb)):
                n_nan += 1
                continue
            rows.append(dict(unit=base, pair=pair, store=str(sk[0]), split=int(sk[1]), method=str(k[0]),
                             learner=str(k[1]) if len(k) > 1 else "", key=json.dumps([str(x) for x in k]),
                             key_class=key_class(k), abs_diff_cm=abs(ra - rb)))
    ma, mb = unit_meta(da, base), unit_meta(db, base)
    summ = dict(unit=base, pair=pair, n_stores_common=len(common_stores), n_stores=[len(sa), len(sb)], n_keys_common=len(rows),
                n_nonfinite_skipped=n_nan, blocks_equal=blocks_ok,
                cfg_common_equal=bool(ma and mb and ma.get("cfg_common") == mb.get("cfg_common")),
                meta=[ma, mb])
    return rows, summ


def dist(df):
    if not len(df):
        return dict(n_keys=0, median_cm=None, max_cm=None)
    x = df.abs_diff_cm.to_numpy(float)
    return dict(n_keys=int(len(x)), median_cm=float(np.median(x)), max_cm=float(np.max(x)))


def main(argv=None):
    a = parse_args(argv)
    ref, r1, r2 = _abs(a.ref_dir), _abs(a.run1_dir), _abs(a.run2_dir)
    units = [u for u in a.units.split(",") if u]
    pairs = [("local1_vs_rescale", r1, ref)] if a.supplement else \
        [("local1_vs_rescale", r1, ref), ("local2_vs_rescale", r2, ref), ("local1_vs_local2", r1, r2)]
    allrows, summs, missing = [], [], []
    for base in units:
        for pname, da, db in pairs:
            rows, s = compare(da, db, base, pname)
            if rows is None:
                missing.append(s)
                continue
            allrows += rows
            summs.append(s)
    df = pd.DataFrame(allrows, columns=["unit", "pair", "store", "split", "method", "learner", "key", "key_class", "abs_diff_cm"])
    out = {}
    for pname, _, _ in pairs:
        q = df[df.pair == pname]
        out[pname] = dict(phys=dist(q[q.key_class == "phys"]), ml=dist(q[q.key_class == "ml"]),
                          ml_n_over_eq=int((q[q.key_class == "ml"].abs_diff_cm > a.eq_cm).sum()),
                          phys_n_over_tol=int((q[q.key_class == "phys"].abs_diff_cm > TOL_PHYS).sum()))
    l1r = out["local1_vs_rescale"]
    l12 = out.get("local1_vs_local2", dict(ml=dict(n_keys=0, median_cm=None, max_cm=None)))
    units_l1r = sorted({s["unit"] for s in summs if s["pair"] == "local1_vs_rescale"})
    n_ml = int(l1r["ml"]["n_keys"])
    l1r["ml_frac_over_eq"] = (l1r["ml_n_over_eq"] / n_ml) if n_ml else None
    if a.supplement:
        rule = "보충 대조(계속 규칙 없음)" if units_l1r else "점검 불가(보충 단위의 로컬·Rescale 공통 조각 없음)"
    elif not units_l1r:
        rule = "점검 불가(로컬 1회차와 Rescale 의 공통 조각 없음)"
    elif l1r["phys"]["n_keys"] == 0 or l1r["phys_n_over_tol"] > 0 or not all(s["blocks_equal"] for s in summs if s["pair"] == "local1_vs_rescale"):
        rule = "멈춤: 물리식 불일치 또는 채점 블록 불일치(자료 경로나 구현 차이, 원인 확인)"
    else:
        cond_a = l1r["ml_n_over_eq"] == 0
        have2 = l12["ml"]["n_keys"] > 0
        med1 = l1r["ml"]["median_cm"]
        det = bool(have2 and float(l12["ml"]["median_cm"]) <= a.det_floor_cm)      # 로컬 재적합이 사실상 결정적
        cond_b = bool(have2 and not det and med1 is not None and med1 <= a.ratio_max * float(l12["ml"]["median_cm"]))
        cond_c = bool(have2 and det and med1 is not None and med1 <= a.med_cm
                      and l1r["ml_frac_over_eq"] is not None and l1r["ml_frac_over_eq"] <= a.frac_over_max)
        tag = f" [(가) {cond_a}, (나) {cond_b if (have2 and not det) else 'NA'}, (다) {cond_c if (have2 and det) else 'NA'}]"
        if cond_a or cond_b or cond_c:
            rule = "계속" + ("" if have2 else "(잠정: 로컬 2회차 없음, (가)만 확인)") + tag
        else:
            rule = "멈춤: 사용자 보고" + ("" if have2 else "(잠정: 로컬 2회차 없음)") + tag
    gp, sp = _abs(a.gate_csv), _abs(a.summary)
    gp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(gp, index=False)
    summary = dict(check="xenv_iii", plan="EXECUTION_PLAN_REMAINING_2026-09-30 6.1, LG 개정 15 (d)", created=time.strftime("%Y-%m-%d %H:%M"),
                   script_version=2, local=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, host=platform.node()),
                   dirs=dict(ref=str(a.ref_dir), run1=str(a.run1_dir), run2=str(a.run2_dir)), units=units,
                   eq_cm=a.eq_cm, ratio_max=a.ratio_max, det_floor_cm=a.det_floor_cm, med_cm=a.med_cm, frac_over_max=a.frac_over_max,
                   supplement=bool(a.supplement), tol_phys_cm=TOL_PHYS, by_pair=out, unit_checks=summs, missing=missing,
                   rule=rule, output_limit="키별 절대 차와 분포만. RMSE 값, Δ, 판정 문구 없음")
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"[xenv iii] 단위 {len(units)} · 대조 {len(summs)} · 누락 {len(missing)} · 규칙: {rule}")
    print(f"[xenv iii] 신경망 키 |로컬1−Rescale| 중앙값 {l1r['ml']['median_cm']} · 최댓값 {l1r['ml']['max_cm']} · {a.eq_cm} cm 초과 {l1r['ml_n_over_eq']} · "
          f"|로컬1−로컬2| 중앙값 {l12['ml']['median_cm']}")
    print(f"[xenv iii] 산출 {gp.relative_to(ROOT) if gp.is_relative_to(ROOT) else gp}, {sp.relative_to(ROOT) if sp.is_relative_to(ROOT) else sp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
