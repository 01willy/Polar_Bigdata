"""WF9 지역 내 조각의 재현 점검(계획서 docs/EXPERIMENT_PLAN_WF_2026-10-01.md §7, 개정 이력 2026-10-02 00:20 의 2).

WF9 의 총 저장소(<대상>|r)와 WF6 조각(results/rescale_wf2)의 같은 (대상, 분할) 저장소에서 공통 키의 블록별 SSE·셀 수를 대조한다.
WF9 는 WF6 와 같은 문맥·추출·seed·함수를 쓰므로 공통 키(P1, Pk, Pc, R1, Re, D0(catboost))는 같아야 한다(허용 1e-6 cm²).
값(RMSE)은 출력하지 않고 차이와 키 수만 쓴다. 산출: data/processed/wf/wf9_repro_check.csv.
실행: python3 scripts/2_evaluation/wf9_repro_check.py [--wf9-dir results/rescale_wf3/data/processed/wf/shards] [--wf6-dir results/rescale_wf2/data/processed/wf/shards]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polar.h4_common import load_stores  # noqa: E402

TOL = 1e-6


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--wf9-dir", default="results/rescale_wf3/data/processed/wf/shards")
    ap.add_argument("--wf6-dir", default="results/rescale_wf2/data/processed/wf/shards")
    ap.add_argument("--out", default="data/processed/wf/wf9_repro_check.csv")
    a = ap.parse_args(argv)
    d9, d6 = ROOT / a.wf9_dir, ROOT / a.wf6_dir
    rows = []
    for p9 in sorted(d9.glob("wf9__cpu__*__r__s*_blocksse.npz")):
        parts = p9.name[: -len("_blocksse.npz")].split("__")
        t, sp = parts[2], int(parts[4][1:])
        p6 = d6 / f"wf6__cpu__{t}__r__s{sp}_blocksse.npz"
        r = dict(target=t, split=sp, wf6_shard=p6.exists())
        if not p6.exists():
            rows.append(r)
            continue
        s9 = load_stores(p9).get((f"{t}|r", sp))
        s6 = load_stores(p6).get((f"{t}|r", sp))
        if s9 is None or s6 is None:
            r.update(note="저장소 없음")
            rows.append(r)
            continue
        same_blocks = bool(np.array_equal(s9.blocks, s6.blocks) and np.array_equal(s9.ncell, s6.ncell))
        common = [k for k in s9.keys if k in s6]
        only9 = sorted({k[0] for k in s9.keys if k not in s6})
        dmax, cnt_bad, by = 0.0, 0, {}
        for k in common:
            a9, c9 = s9.get(k)
            a6, c6 = s6.get(k)
            d = float(np.max(np.abs(a9 - a6))) if len(a9) else 0.0
            dmax = max(dmax, d)
            cnt_bad += int(not np.array_equal(c9, c6))
            by[k[0]] = max(by.get(k[0], 0.0), d)
        r.update(same_blocks=same_blocks, n_keys_wf9=len(s9.keys), n_common=len(common), only_wf9_methods=";".join(only9), max_abs_dsse=dmax,
                 n_count_mismatch=cnt_bad, ok=bool(same_blocks and dmax <= TOL and cnt_bad == 0),
                 max_abs_dsse_by_method=";".join(f"{m}:{v:.3g}" for m, v in sorted(by.items())))
        rows.append(r)
    df = pd.DataFrame(rows)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    if not len(df):
        print("[wf9-repro] WF9 조각이 없다")
        return 1
    ok = int(df.get("ok", pd.Series(dtype=bool)).fillna(False).sum())
    print(f"[wf9-repro] 조각 {len(df)} · WF6 짝 {int(df.wf6_shard.sum())} · 일치 {ok} · 공통 키 {int(df.get('n_common', pd.Series([0])).fillna(0).sum()):,} · "
          f"최대 |ΔSSE| {float(df.get('max_abs_dsse', pd.Series([np.nan])).max()):.3g} · 셀 수 불일치 {int(df.get('n_count_mismatch', pd.Series([0])).fillna(0).sum())} → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
