"""H49 보조 · 지도 규칙(부록 6.7)에 필요하지만 lgx_lg_aux.csv, lgw_bundle.csv, lgx_tests.csv 에 없는 대비의 10,000회 재표집 판정.
계획 docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md 6.7(커밋 316714c). 산출 lgw_map_aux.csv 는 h49_transfer_map.py 의 규칙 입력이다.

왜 별도 파일인가
  6.7 은 R0 − P0(n = 0)처럼 lgx_lg_aux·lgw_bundle 에 없는 대비를 h39 summarize 가 lgw_aux4.csv 에 함께 계산한다고 정했다. 그러나 h39 의
  aux4_table 에는 지역 행(L1, L3, L6, L7)만 있고 R0 − P0|n0 의 층화 평균 행과 레나 행이 없다(L6 은 n ∈ {10, 40, 160} 만 본다).
  목록 안으로 대체한 P1* 에 대한 R1 − P1* 도 어느 표에도 없다(h42 L29 는 자기가 고른 P1* 에 대해서만 이 행을 만든다). 그래서 이 파일이
  h39 의 대비 엔진(contrast_pool: h40.contrast 의 주 분포, h42.boot_delta_common 의 보조 분포, 층화 평균과 풀 규칙)을 importlib 로 불러 같은
  규약으로 계산한다. h39 는 다른 작업이 관리하므로 고치지 않았다. h39 가 같은 대비를 lgw_aux4.csv 에 더하면 h49 는 두 표를 모두 읽고,
  두 표의 판정이 다르면 '재표집 의존'으로 보고 우세로 세지 않는다.

대비(주 4지역 x 의 지역 행과 층화 평균 행. 라벨은 h42·h39 형식)
  MAP-a   R0-P0|n0                     LG 조각(h40). 6.7 (a)
  MAP-bc  R1-<b>|n10, R1-<b>|n-1        LGX 조각(h42, 태그 lgxb·lgx1·lgx2·lgx9 병합). b ∈ {P1@soil, P1@ku, P1@ed, P1@cci, B:ens}(격자에서 계산할
                                       수 있는 재보정 기준선 목록, 6.7). 6.7 (b)(c)의 R1 − P1*(목록 안 대체 포함)
  재표집: 10,000회(--allow-local 필요. 없으면 스레드 1, 1,000회 이하로 낮추고 nboot_capped 를 적는다. h49 본 실행은 그 표를 거부한다).
  seed = seed_of('lgw', 대상, 'MAP-a|R0-P0|n0' 형식의 라벨)(h39.region_stat 규약).

결과 열람 규칙
  LG·LGX 결과 회수 뒤, 부록 커밋 뒤에만 실행한다. --allow-run 이 없으면 조각을 열기 전에 거부한다. 표준 출력에는 행 수만 쓰고 Δ 와 판정은
  쓰지 않는다.
쓰기 보호: h39.check_out_dir 와 같다(data/processed/ 의 lg, lgx, lgt, lgd, lgu, lgf 와 results/ 안이면 거부).

실행(ROOT, nice 10, GPU 없음)
  nice -n 10 python3 scripts/2_evaluation/h49_map_contrasts.py --allow-run --allow-local --threads 2 --out-dir data/processed/wrapup \
      --lg-shards data/processed/lg/shards --lgx-shards data/processed/lgx/shards
"""
from __future__ import annotations

import os
import sys

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
MAX_THREADS = 4


def _peek_threads(av):
    """numpy 를 부르기 전에 스레드 수를 정한다. --allow-local 이 없으면 1, 있으면 --threads(최대 4)."""
    if "--allow-local" not in av:
        return 1
    v = "1"
    for i, x in enumerate(av):
        if x == "--threads" and i + 1 < len(av):
            v = av[i + 1]
        elif x.startswith("--threads="):
            v = x.split("=", 1)[1]
    try:
        return max(1, min(int(v), MAX_THREADS))
    except ValueError:
        return 1


for _v in THREAD_VARS:
    if __name__ == "__main__":
        os.environ[_v] = str(_peek_threads(sys.argv))
    else:
        os.environ.setdefault(_v, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse                                                                                         # noqa: E402
import hashlib                                                                                          # noqa: E402
import importlib.util                                                                                   # noqa: E402
import json                                                                                             # noqa: E402
import resource                                                                                         # noqa: E402
import time                                                                                             # noqa: E402
from pathlib import Path                                                                                # noqa: E402

import numpy as np                                                                                      # noqa: E402
import pandas as pd                                                                                     # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
NBOOT = 10000
LOCAL_NBOOT_MAX = 1000
MAP_B = ("P1@soil", "P1@ku", "P1@ed", "P1@cci", "B:ens")
OUT_NAME = "lgw_map_aux.csv"


def h39():
    """h39_scenarios 를 파일 경로에서 읽는다(고치지 않는다)."""
    if "h39_scenarios" in sys.modules:
        return sys.modules["h39_scenarios"]
    spec = importlib.util.spec_from_file_location("h39_scenarios", ROOT / "scripts" / "2_evaluation" / "h39_scenarios.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["h39_scenarios"] = mod
    spec.loader.exec_module(mod)
    return mod


def map_contrasts():
    """(test_id, 라벨, 저장소 출처, (A 방법, n), (B 방법, n))."""
    out = [("MAP-a", "R0-P0|n0", "lg", ("R0", 0), ("P0", 0))]
    for b in MAP_B:
        for n, lab in ((10, "n10"), (-1, "n-1")):
            out.append(("MAP-bc", f"R1-{b}|{lab}", "lgx", ("R1", n), (b, n)))
    return out


def map_aux_table(tms_lg: dict, tms_lgx: dict, nboot: int) -> pd.DataFrame:
    """주 4지역 x 의 지역 행과 층화 평균 행. 저장소에 키가 없으면 h39.missing_row('행 없음')."""
    S = h39()
    names = [f"{t}|x" for t in S.MAIN4]
    rows = []
    for test, lab, src, a_, b_ in map_contrasts():
        tms = tms_lg if src == "lg" else tms_lgx
        gA, gB = S._g(src, *a_), S._g(src, *b_)
        pr = S.contrast_pool(tms or {}, names, gA, gB, f"{test}|{lab}", nboot, dict(test_id=test, contrast=lab, source=src, n=int(a_[1])))
        if not pr:
            pr = [S.missing_row(test_id=test, contrast=lab, source=src, target="MEAN[]", n=int(a_[1]))]
        rows += pr
    return S.clean(rows)


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="H49 보조: 지도 규칙(6.7)의 누락 대비(lgw_map_aux.csv)")
    ap.add_argument("--out-dir", default="data/processed/wrapup")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--subregion-map", default="lg_subregion_map_v1.csv")
    ap.add_argument("--lg-shards", default="data/processed/lg/shards")
    ap.add_argument("--lg-tag", default="lg")
    ap.add_argument("--lgx-shards", default="data/processed/lgx/shards")
    ap.add_argument("--lgx-tag", default="lgx")
    ap.add_argument("--allow-run", action="store_true", help="LG·LGX 결과 회수 뒤의 실행을 허용한다(없으면 조각을 열지 않고 거부)")
    ap.add_argument("--allow-local", action="store_true", help="등록한 재표집 횟수(10,000)와 --threads 를 허용한다")
    ap.add_argument("--nboot", type=int, default=NBOOT)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--allow-mixed-cfg", action="store_true")
    a = ap.parse_args(argv)
    S = h39()
    a.OUT = S.check_out_dir(a.out_dir)
    a.PROC = S._abs(a.data_dir)
    a.nboot_asked = int(a.nboot)
    if not a.allow_local:
        a.threads = 1
        a.nboot = min(int(a.nboot), LOCAL_NBOOT_MAX)
    if a.threads > MAX_THREADS:
        raise SystemExit(f"[거부] 스레드 {a.threads} > {MAX_THREADS}")
    a.CAPPED = int(a.nboot) < NBOOT
    return a


def main(argv=None, D=None):
    """D 를 주면 그 자료 객체를 쓴다(시험의 합성 자료)."""
    a = parse_args(argv)
    try:
        if os.nice(0) < 10:
            os.nice(10 - os.nice(0))
    except OSError:
        pass
    if not a.allow_run:
        raise SystemExit("[거부] LG·LGX 결과 회수 뒤 --allow-run 으로만 실행한다(조각을 열지 않았다)")
    t0 = time.time()
    S = h39()
    D = D if D is not None else S.make_data(a, 5)
    tms_lg, sh_lg = S.load_tms(S._abs(a.lg_shards), [a.lg_tag], D, a.nboot, allow_mixed=a.allow_mixed_cfg)
    lgx_tags = [f"{a.lgx_tag}{s}" for s in ("b", "1", "2", "9")]
    tms_lgx, sh_lgx = S.load_tms(S._abs(a.lgx_shards), lgx_tags, D, a.nboot, allow_mixed=a.allow_mixed_cfg)
    if not tms_lg:
        raise SystemExit(f"[거부] LG 조각 없음: {a.lg_shards}")
    if not tms_lgx:
        raise SystemExit(f"[거부] LGX 조각 없음: {a.lgx_shards}(태그 {lgx_tags})")
    for nm, tm in tms_lg.items():                                         # h39.run_summarize 와 같다
        tm.n_blocks_target = int(len(np.unique(D.df.block.values[D.target_idx(nm.split("|")[0])])))
    df = map_aux_table(tms_lg, tms_lgx, a.nboot)
    df["nboot"] = int(a.nboot)
    df["nboot_capped"] = bool(a.CAPPED)
    a.OUT.mkdir(parents=True, exist_ok=True)
    out = a.OUT / OUT_NAME
    df.to_csv(out, index=False)
    m = df[df.scope.astype(str) == "MEAN"]
    counts = dict(n_rows=int(len(df)), n_mean=int(len(m)), n_missing=int((df.verdict4.astype(str) == "행 없음").sum()),
                  pool={str(r["contrast"]): str(r.get("pool", "")) for r in m.to_dict("records")})
    meta = dict(created=time.strftime("%Y-%m-%d %H:%M:%S %Z"), script="scripts/2_evaluation/h49_map_contrasts.py", script_sha256=sha256_file(__file__),
                h39_sha256=sha256_file(ROOT / "scripts" / "2_evaluation" / "h39_scenarios.py"), argv=list(sys.argv[1:] if argv is None else argv),
                nboot=int(a.nboot), nboot_asked=int(a.nboot_asked), nboot_capped=bool(a.CAPPED), threads=int(a.threads),
                allow_local=bool(a.allow_local), contrasts=[dict(test_id=t, contrast=lab, source=s) for t, lab, s, _, _ in map_contrasts()],
                lg_shards=[str(s_["npz"].name) for s_ in sh_lg], lgx_shards=[str(s_["npz"].name) for s_ in sh_lgx], counts=counts,
                output=dict(path=str(out), bytes=int(out.stat().st_size), sha256=sha256_file(out)),
                max_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1), elapsed_s=round(time.time() - t0, 1))
    (a.OUT / "lgw_map_aux_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    print(f"[map-aux] {out} · 행 {counts['n_rows']} · 층화 평균 행 {counts['n_mean']} · 행 없음 {counts['n_missing']} · nboot {a.nboot}"
          f"{' (제한, 판정에 쓰지 않음)' if a.CAPPED else ''}", flush=True)
    return df


if __name__ == "__main__":
    main()
