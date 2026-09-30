"""AB10 입력 표(LGU-A1 구간 점수 대비의 두 가중 p). LGU 개정 4 초안(docs/EXECUTION_PLAN_REMAINING_2026-09-30.md 6.3)의 보조 스크립트.

배경: h39 summarize 의 lgu_ab10(h39 1287–1303행)은 --lgu-dir 안의 '*tests*.csv' 파일에서 test_id 열이 'LGU-A1' 인 행을 찾고
p_cell·p_beq 두 열을 읽는다. h44 의 lgu_a_tests.csv 는 열 이름이 test 이고 p 는 셀 가중 p_boot 한 열뿐이라 AB10 이 '행 없음'(p = 1)이
된다. 이 스크립트는 h44 가 저장한 2단 재표집 분포(lgua_boot__<대상>__x.npz, dist[n, 방법, 재표집, 지표, 가중])로 h44 pool_contrast 와 같은
방식(지역별 A − B, 같은 재표집 번호끼리 지역 등가중 평균)의 풀 분포를 만들고 두 가중의 p 를 계산한다.

정의(등록 초안과 같다)
  대비: LGU-A1 의 단 (iii) − B4, 지표 is10(α 0.1 구간 점수), n = 10, 풀 = lgu_a_tests.csv 의 해당 행 regions 열(기본 레나·캐나다·알래스카)
  p_cell = boot_p(풀 분포[..., 0]), p_beq = boot_p(풀 분포[..., 1]). boot_p 는 h44 가 쓰는 LC.boot_p 와 같다(양측, 최소 1/R)
  p_eq   = max(eq_p(풀 분포[..., 0], 한계_셀), eq_p(풀 분포[..., 1], 한계_블록)). 한계는 해당 행 margin 열(기준 점수의 5 %, 가중별)
           eq_p(d, δ) = max(P(d ≤ −δ), P(d ≥ δ), 1/R)(h42 eq_p 와 같다)
  verdict4 = 해당 행 verdict 를 그대로 옮긴다. '판정 불가(…)' 는 '판정 불가' 로 적는다(h39 NA_V 와 맞춘다)
  delta  = 해당 행 delta(셀 가중 점 추정)
대조: p_cell 이 해당 행 p_boot 와 1e-12 안에서 같은지 적는다(같은 분포에서 나온 값이어야 한다). 값은 화면에 출력하지 않는다.

산출: <lgu-dir>/lgu_a_ab10_tests.csv(열 test_id, scope, contrast, n, metric, pool, delta, p_cell, p_beq, p_eq, verdict4, nboot, source),
      <lgu-dir>/lgu_a_ab10_meta.json(대조 결과, 입력 파일 해시, 재표집 횟수, 작성 시각).
실행 조건: LGU 개정 4 커밋 뒤(scripts/local/run_post_results.sh lgu-ab10 이 확인한다). 스레드 1, GPU 없음.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                                                                   # noqa: E402
import pandas as pd                                                                                  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TEST, CONTRAST, METRIC, N_MAIN = "LGU-A1", "iii − B4", "is10", 10
METHOD_A, METHOD_B = "iii", "B4"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="AB10 입력 표(LGU-A1 두 가중 p)")
    ap.add_argument("--lgu-dir", default="data/processed/lgu")
    ap.add_argument("--tests", default="lgu_a_tests.csv")
    ap.add_argument("--boot-tag", default="lgua", help="h44 --tag(2단 분포 파일 접두사)")
    ap.add_argument("--out", default="lgu_a_ab10_tests.csv", help="h39 가 읽도록 이름에 'tests' 가 들어가야 한다")
    ap.add_argument("--meta", default="lgu_a_ab10_meta.json")
    return ap.parse_args(argv)


def _abs(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def boot_p(d) -> float:
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    if not len(d):
        return float("nan")
    p = 2.0 * min(np.mean(d <= 0), np.mean(d >= 0))
    return float(min(1.0, max(p, 1.0 / len(d))))


def eq_p(d, delta) -> float:
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    if not len(d) or not np.isfinite(delta):
        return float("nan")
    return float(max(np.mean(d <= -float(delta)), np.mean(d >= float(delta)), 1.0 / len(d)))


def level(z, method, n, metric):
    """h44 boot_level 과 같은 선택: dist[n, 방법, :, 지표] → (셀 가중 (R,), 블록 등가중 (R,))."""
    n_list = [int(v) for v in z["n_list"]]
    methods = [str(v) for v in z["methods"]]
    metrics = [str(v) for v in z["metrics"]]
    if method not in methods or int(n) not in n_list or metric not in metrics:
        return None
    d = np.asarray(z["dist"])[n_list.index(int(n)), methods.index(method), :, metrics.index(metric)]
    return d[:, 0], d[:, 1]


def strat_mean(ds):
    """같은 재표집 번호끼리 지역 등가중 평균(길이가 다르면 짧은 쪽). h44 LC.strat_mean 과 같다."""
    ds = [np.asarray(d, float) for d in ds if d is not None]
    if not ds:
        return np.zeros(0)
    L = min(len(d) for d in ds)
    return np.stack([d[:L] for d in ds]).mean(axis=0)


def main(argv=None):
    a = parse_args(argv)
    lgu = _abs(a.lgu_dir)
    tp = lgu / a.tests
    if not tp.exists():
        raise SystemExit(f"[중단] {tp} 가 없다(LGU 실험 A 집계 전)")
    t = pd.read_csv(tp)
    q = t[(t["test"].astype(str) == TEST) & (t["metric"].astype(str) == METRIC)
          & (pd.to_numeric(t["n"], errors="coerce") == N_MAIN) & (t["contrast"].astype(str) == CONTRAST)]
    if len(q) != 1:
        raise SystemExit(f"[중단] {tp.name} 에서 {TEST}·{CONTRAST}·n {N_MAIN}·{METRIC} 행이 {len(q)}개다(1개여야 한다)")
    r = q.iloc[0]
    regions = [s for s in str(r.get("regions", "")).split(",") if s and s != "nan"]
    verdict = str(r.get("verdict", "판정 불가"))
    v4 = "판정 불가" if verdict.startswith("판정 불가") else verdict
    try:
        mg = json.loads(str(r.get("margin", ""))) if str(r.get("margin", "")) not in ("", "nan") else [np.nan, np.nan]
    except ValueError:
        mg = [np.nan, np.nan]
    per_c, per_b, files, why = [], [], {}, {}
    for reg in regions:
        f = lgu / f"{a.boot_tag}_boot__{reg}__x.npz"
        if not f.exists():
            why[reg] = "2단 분포 파일 없음"
            continue
        with np.load(f, allow_pickle=False) as z:
            la, lb = level(z, METHOD_A, N_MAIN, METRIC), level(z, METHOD_B, N_MAIN, METRIC)
            meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        if la is None or lb is None:
            why[reg] = "방법·n·지표 없음"
            continue
        per_c.append(la[0] - lb[0]); per_b.append(la[1] - lb[1])
        files[reg] = dict(file=f.name, sha16=sha16(f), nboot=meta.get("nboot"), cfg_hash=meta.get("cfg_hash"))
    has2 = bool(regions) and not why
    if has2:
        dc, db = strat_mean(per_c), strat_mean(per_b)
        p_cell, p_beq = boot_p(dc), boot_p(db)
        p_eq = float(np.nanmax([eq_p(dc, float(mg[0])), eq_p(db, float(mg[1]))])) if np.isfinite(mg).all() else float("nan")
        R = int(min(len(dc), len(db)))
    else:
        p_cell = p_beq = p_eq = float("nan"); R = 0
        v4 = "판정 불가"
    p_boot = float(r.get("p_boot", np.nan))
    same = bool(np.isfinite(p_boot) and np.isfinite(p_cell) and abs(p_boot - p_cell) <= 1e-12)
    row = dict(test_id=TEST, scope="MEAN", contrast=f"{CONTRAST} | n{N_MAIN} | {METRIC}", n=N_MAIN, metric=METRIC, pool=",".join(regions),
               delta=float(r.get("delta", np.nan)), p_cell=p_cell, p_beq=p_beq, p_eq=p_eq, verdict4=v4, nboot=R,
               source=f"{tp.name}(verdict, delta, margin) + {a.boot_tag}_boot__*__x.npz(dist)")
    out, mp = lgu / a.out, lgu / a.meta
    pd.DataFrame([row]).to_csv(out, index=False)
    mp.write_text(json.dumps(dict(created=time.strftime("%Y-%m-%d %H:%M"), plan="EXECUTION_PLAN_REMAINING_2026-09-30 6.3(LGU 개정 4 초안)",
                                  tests_file=tp.name, tests_sha16=sha16(tp), boot_files=files, excluded=why, has_two_stage=has2,
                                  p_cell_equals_p_boot=same, nboot=R, margin=mg, verdict_normalized=bool(verdict != v4),
                                  output_limit="값은 화면에 출력하지 않는다"), ensure_ascii=False, indent=1, default=str))
    print(f"[ab10] 작성 {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out} · 풀 지역 {len(regions) - len(why)}/{len(regions)} · "
          f"2단 분포 {'있음' if has2 else '없음'} · p_cell 과 p_boot 대조 {'일치' if same else '불일치(원인 확인)'}")
    return 0 if (has2 and same) else 1


if __name__ == "__main__":
    sys.exit(main())
