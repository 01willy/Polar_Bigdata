#!/usr/bin/env python3
"""LGX 재현 점검의 조기 확인. 계획 docs/EXPERIMENT_PLAN_LG_2026-09-29.md §6A.6.

확장의 기준 축 조각(tag lgxb)과 기준 조각(본 실행 lg, 사전 점검 lg_precheck, 스모크 lg_smoke)을 h42 의 gate_table 로 대조한다.
집계(--summarize-only)를 기다리지 않고 조각이 생기는 대로 부를 수 있다. 학습을 하지 않는다(조각 npz 를 읽어 키별 RMSE 의 차를 구한다).
허용 차는 h42 의 값(물리식 1e-9 cm, CatBoost 방법 0.02 cm)을 그대로 쓴다. 판정 규칙을 새로 두지 않는다.

용도
  CPU 부분을 본 실행(iolite-4, AMD EPYC 7V12)과 다른 노드 종류에서 돌릴 때 CatBoost 값이 허용 차 안에서 재현되는지를 작업 초반에 본다.
  재현되지 않아도 확장의 대비는 확장 조각 안에서 닫히므로(계획서 §6A.6) 실행을 멈추지 않는다. 결과는 로그와 표에 남긴다.

종료 코드
  0  대조한 단위가 모두 허용 차 안이다
  3  허용 차를 넘은 단위가 있다
  4  대조할 단위가 없다(확장의 기준 축 조각이 없거나 기준 조각이 없다)
  1  오류

실행(Rescale 작업 안): python3 scripts/rescale/lgx_gate_check.py --gate-tag lg_precheck --out-csv lgx_gate_early.csv
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_h42():
    name = "h42_label_grid_ext"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "3_deep_learning" / "h42_label_grid_ext.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="LGX 재현 점검의 조기 확인")
    ap.add_argument("--out-dir", default="data/processed/lgx", help="확장 산출 디렉터리(조각은 <out-dir>/shards)")
    ap.add_argument("--lg-dir", default="data/processed/lg", help="기준 조각의 상위 디렉터리(조각은 <lg-dir>/shards)")
    ap.add_argument("--gate-tag", default="", help="기준 조각의 tag(lg, lg_precheck, lg_smoke). 기본은 h42 의 기본값(lg)")
    ap.add_argument("--tag", default="lgx")
    ap.add_argument("--smoke", action="store_true", help="확장의 스모크 조각(lgxb_smoke)을 대조한다")
    ap.add_argument("--precheck", action="store_true", help="확장의 사전 점검 조각(lgxb_precheck)을 대조한다")
    ap.add_argument("--out-csv", default="", help="대조 표를 쓸 경로(기본: 쓰지 않는다)")
    return ap.parse_args(argv)


def main(argv=None):
    q = parse_args(argv)
    X = load_h42()
    av = ["--part", "cpu", "--out-dir", q.out_dir, "--lg-dir", q.lg_dir, "--tag", q.tag, "--threads", "1"]
    if q.gate_tag:
        av += ["--gate-tag", q.gate_tag]
    if q.smoke:
        av += ["--smoke"]
    if q.precheck:
        av += ["--precheck"]
    a = X.parse_args(av)
    base_tag = X.axis_tag(a, "base")
    sh = X.find_shards_x(a.SHARDS, base_tag)
    if not sh:
        print(f"[gate] 확장의 기준 축 조각이 없다: {a.SHARDS}/{base_tag}__*", flush=True)
        return 4
    g = X.gate_table(a, sh, a.GATE_DIR, a.GATE_TAG)
    if q.out_csv:
        Path(q.out_csv).parent.mkdir(parents=True, exist_ok=True)
        g.to_csv(q.out_csv, index=False)
    ref = g[g.status != "기준 조각 없음"]
    print(f"[gate] 확장 기준 축 조각 {len(sh)}개(tag {base_tag}) · 기준 tag {a.GATE_TAG} · 기준 조각이 있는 단위 {len(ref)}개 · "
          f"허용 차 물리식 {X.GATE_TOL_PHYS:g} cm, CatBoost {X.GATE_TOL_ML:g} cm", flush=True)
    if not len(ref):
        return 4
    cols = ["target", "mode", "split", "status", "n_keys", "n_keys_phys", "n_keys_ml", "max_diff_phys", "max_diff_ml", "blocks_equal",
            "code_sha_h40_ref", "code_sha_h40_now", "worst_key"]
    print(ref[cols].to_string(index=False), flush=True)
    ok = ref.passed.astype(str) == "True"
    print(f"[gate] 통과 {int(ok.sum())}/{len(ref)} 단위 · 물리식 최대 차 {ref.max_diff_phys.max():.3g} cm · CatBoost 최대 차 {ref.max_diff_ml.max():.3g} cm",
          flush=True)
    return 0 if bool(ok.all()) else 3


if __name__ == "__main__":
    try:
        rc = main()
    except SystemExit:
        raise
    except Exception as e:                                                # noqa: BLE001
        print(f"[gate] 오류: {e!r}", flush=True)
        rc = 1
    sys.exit(rc)
