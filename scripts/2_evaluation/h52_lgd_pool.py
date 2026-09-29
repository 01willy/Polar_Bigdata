"""H52 · LGD 확장 풀 집계기. 계획 docs/EXPERIMENT_PLAN_LG_2026-09-29.md 6B.5·6B.6 6단계(개정 10, 13, 14)와
docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md 7.2–7.5(커밋 316714c, 개정 1)의 집계기 규칙을 구현한다. 학습은 하지 않는다.

입력(읽기 전용)
  LG 본 실행 조각(--lg-dir/shards, tag lg, cpu 부분): 주 4지역 P4(레나, 캐나다, 러시아 W, 러시아 E, 모드 x)
  LGD 조각(--lgd-dir/<spec>/shards, tag lg d): h51 이 만든 표별 조각. 실행 표가 주 설정과 같은 변형은 manifest 의 same_as 로 주 설정 조각을 쓴다
  적격 표 data/processed/lgd_eligibility_v1.csv(개정 13 등록), h51 manifest(lgd_run_manifest.json), 재현 점검 lgd_repro_gate.csv,
  교차 환경 점검 (i) 요약(data/processed/lgw/lgw_xenv_gate_i_summary.json), LGX-N4 분할 분포(lgx_splitdist.csv, L40 의 (a)5)

h40 MAIN_POINT 처리(WRAPUP 7.4 (b)1)
  h40 을 import 한 뒤 h40.MAIN_POINT 를 적격 표의 부적격 새 지역 목록(+ 점 추정 전용 변형 저장소 이름)으로 바꾸고, 그 뒤에 TMx 를 만든다.
  h40 파일은 고치지 않는다. 주 설정의 대상 이름은 v3 와 같다(Russia_C|x 등). 변형 저장소는 '대상~표지|x' 로 부른다.

풀과 가설(6B.5)
  P4 = 주 4지역(LG 조각). PE1 = P4 + 적격 얕은 레짐 새 지역. PE2 = PE1 + 적격 심부 레짐 새 지역.
  L1e, L4e, L8e: h40 의 L1, L4, L8 과 같은 규칙을 풀만 바꿔 계산한다(재표집 1,000회, h40.build_curve·strat_mean, 두 가중 CI).
    L1e 기각 기준 = 유의 개선 지역 수 > ⌊N/4⌋(N = 4 이면 h40 L1 과 같다). PE2 는 cm 단위와 상대 단위(지역 Δ / 그 지역 P0 RMSE 점 추정)를
    함께 내고 두 단위의 판정이 다르면 '척도 의존'을 붙인다.
  L38–L42: 4분 판정(재표집 10,000회, h42.region_stats·stats_row·pool_rows, δ 0.5·1.0 cm).

집계기 규칙(WRAPUP 7.2, 7.4 (b)1)
  (a)1 P4 와 PE1·PE2 의 비교: L1e 는 지지·기각(유의 개선 지역 비율 k/N 병기), L4e 는 'n ≤ 10 에서 성립 여부'(최소 n 병기), L8e 는 지지·기각.
       판정이 다르면 풀 대비의 4분 판정에 L28 분류(강건, 약화, 의존)를 적용하고 새 지역만의 층화 평균으로 반전·희석을 나눈다.
       6B.5 의 병기 문구 뒤에 '반전(새 지역 평균 …)' 또는 '희석(새 지역 평균 …)'을 덧붙인다.
  (a)2 지역 수준 추론 보조 열(h42.region_inference): D0 − P0(n 0, 10), R1 − P1(n 3, 10), R1 − P0(전량). 판정에 쓰지 않는다.
  (a)3 L38 은 6B.5 원문 판정. 보조 열: 점 추정 부호와 가설 방향의 대칭 이름. P4 4지역 참조 행.
  (a)4 공통 재표집 보조 CI(h42.boot_delta_common, 열 ci_lo_c·ci_hi_c·ci_lo_beq_c·ci_hi_beq_c). 주 CI 와 판정이 다르면 '분할 독립 가정 의존'
       (ci_dependence). 붙이는 곳: L1e 지역 행, L4e·L8e 의 지역 행과 MEAN 행(재표집 1,000회, 판정어는 improve·ns·worse 의 sig_common),
       L4e-PE2 의 (a)10 병기 행, (a)1 풀 대비의 지역 행과 MEAN 행, L38–L42 의 행(10,000회, verdict4_common). PE2 상대 단위 행에는 붙이지 않는다.
       분할별 채점 블록 최솟값 < 5 인 표(적격 표의 min_nb_eval_used)의 L38 행과 L41·L42 변형 행에 '소수 블록'.
  (a)5 L40 의 '분할 변동 범위' 표지(LGX-N4 의 5분할 평균 분포, seed_of("lgw-l40", 대상, 대비), 2,000회, 10–90 백분위).
  (a)10 티베트: P0 기반 대비에 '예측된 결과(비맹검)'. L4e-PE2 에 R1 − P2, R1 − P3, P2 − P1 의 PE2 층화 평균과 지역 행(L4e 의 n 격자마다)을,
        L38 (c)에 티베트의 같은 세 대비(n = 10)를 병기한다(판정에 쓰지 않는다. L41·L42 의 L28 분류에도 넣지 않고 aux_l28 열에 따로 적는다).
        L38 (c)의 티베트 P2 − P1(n = 10)이 우세이면 PE2 의 R1 − P1(L4e)과 L38 (c) R1 − P1 에 '수축 사전분포 오지정 의존'.
        공변량 지지 밖 비율은 h51 manifest 에서 읽는다.
  6B.7 비맹검 표시: 티베트의 P0 기반 대비(L1e·L8e 지역 행, L38)와 PE2 의 L8e 에 '비맹검(라벨 통계 열람)', L39 의 모든 행에
        '비맹검(라벨 통계 열람)', L40 의 모든 행에 '재현(비맹검)'.
  L41·L42 의 난수(개정 14): 변형 저장소의 부트스트랩 seed 를 주 설정 저장소 이름으로 둔다(공통 난수). 실행 표가 주 설정과 같은 변형(same_as)은
        주 설정의 TMx 를 그대로 쓰므로 모든 행이 주 설정과 같다. '변형 불가'는 적격 표의 eligible 열에서 읽는다.
  (a)11 L8e 행에 지역별 실제 라벨 수와 전량 R1 − P1. 전량 라벨 160개 이상·40개 이하 지역의 평균을 나눠 적는다.
  플랫폼(WRAPUP 8.6 (3)): P4 는 Rescale, 새 지역은 로컬이다. 모든 대비는 지역 안에서 닫혀 있다. 플랫폼이 섞인 풀 행에 '교차 환경'과
  교차 환경 점검 (i)의 상태('(i) 통과', '(i) CatBoost 불통과' 등)를 붙인다. 요약의 점검 범위가 등록 범위(네 단위)보다 작으면 '부분(k/K 단위,
  잠정)'을 덧붙인다. L40 은 LG 기준값을 쓰면 '교차 환경(보조)'다.
  재현 점검(WRAPUP 7.5 (c)2): lgd_repro_gate.csv 의 주 판정 범위(scope = base, 6A.6 기준 방법) 행으로 정한다. 물리식 불통과면 집계를 멈춘다.
  CatBoost 만 불통과면 L40 의 기준값을 로컬 v3(v3local 표)로 바꾼다. 병기 범위(scope = all)의 결과는 메타에 적는다.
  시험 전용 인자 --allow-no-repro-gate 는 환경 변수 LGD_TEST=1 일 때만 받는다(개정 14).

산출(--out-dir, 기본 data/processed/lgd): lgd_curve.csv, lgd_minn.csv, lgd_tests.csv, lgd_region_inference.csv, lgd_pool.csv, lgd_meta.json

실행(ROOT, LG 본 실행 회수와 LGD 본 실행 뒤):
  nice -n 10 python3 scripts/2_evaluation/h52_lgd_pool.py --lg-dir results/rescale_lg/data/processed/lg --allow-local --threads 4
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path


def _peek(flag, default=None):
    av = sys.argv
    for i, v in enumerate(av):
        if v == flag and i + 1 < len(av):
            return av[i + 1]
        if v.startswith(flag + "="):
            return v.split("=", 1)[1]
    return default


for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, str(_peek("--threads", "4")) if "--allow-local" in sys.argv else "1")

import numpy as np                                                                                   # noqa: E402
import pandas as pd                                                                                  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DL = ROOT / "scripts" / "3_deep_learning"
for _p in (str(DL), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


H = _load("h40_label_grid", DL / "h40_label_grid.py")
X = _load("h42_label_grid_ext", DL / "h42_label_grid_ext.py")
R = _load("h51_lgd_run", DL / "h51_lgd_run.py")

from polar.h4_common import load_stores, seed_of                                                     # noqa: E402
from polar import m1_stats as MS                                                                     # noqa: E402

MAIN_POINT_ORIG = list(H.MAIN_POINT)
P4 = [f"{t}|x" for t in H.MAIN4]
N_ALL = -1
L1_NS = (0, 3, 10, 40)
FEW_BLOCKS = 5
BASE = H.BASE_LEARNER
NA = ("행 없음", "판정 불가", "")
L40_SPECS = {"Russia_W_expanded": "Russia_W|x", "Russia_E_expanded": "Russia_E|x", "Canada_expanded": "Canada|x",
             "Russia_E_expanded_noKytalyk": "Russia_E|x"}
CORE = [("D0-P0|n0", "D0", "P0", 0), ("P1-P0|n10", "P1", "P0", 10), ("R1-P1|n10", "R1", "P1", 10), ("R1-P1|all", "R1", "P1", N_ALL),
        ("R1-P0|all", "R1", "P0", N_ALL)]                                                         # L28 의 핵심 대비(L39)
N4_LABEL = {("D0", "P0", 0): "D0-P0|n0", ("P1", "P0", 10): "P1-P0|n10", ("R1", "P0", N_ALL): "R1-P0|all"}


# ================================================================ 인자
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="H52 LGD 확장 풀 집계기(계획서 6B.5, WRAPUP 7절)")
    ap.add_argument("--lg-dir", default="results/rescale_lg/data/processed/lg", help="LG 본 실행 산출(읽기 전용). 조각은 <lg-dir>/shards")
    ap.add_argument("--lg-tag", default="lg")
    ap.add_argument("--lg-platform", default="rescale")
    ap.add_argument("--lgd-dir", default="data/processed/lgd")
    ap.add_argument("--out-dir", default="", help="기본 = --lgd-dir")
    ap.add_argument("--eligibility", default="data/processed/lgd_eligibility_v1.csv")
    ap.add_argument("--splitdist", default="data/processed/lgx/lgx_splitdist.csv", help="LGX-N4 분할 분포(WRAPUP 7.2 (a)5)")
    ap.add_argument("--xenv-gate", default="data/processed/lgw/lgw_xenv_gate_i_summary.json", help="교차 환경 점검 (i) 요약(WRAPUP 8.4)")
    ap.add_argument("--xenv-i", default="auto", choices=["auto", "pass", "catboost_fail", "phys_fail", "unknown"])
    ap.add_argument("--repro-gate", default="", help="기본 = <lgd-dir>/lgd_repro_gate.csv")
    ap.add_argument("--allow-no-repro-gate", action="store_true", help="시험 전용(LGD_TEST=1): 재현 점검 기록 없이 집계한다(PE 행에 '재현 점검 미실시')")
    ap.add_argument("--nboot-h40", type=int, default=1000, help="L1e, L4e, L8e 의 재표집 횟수(h40 집계와 같다)")
    ap.add_argument("--nboot", type=int, default=10000, help="4분 판정(L38–L42, 풀 대비, 지역 수준 추론)의 재표집 횟수(6A.2)")
    ap.add_argument("--delta-eq", type=float, default=0.5)
    ap.add_argument("--delta-eq-aux", type=float, default=1.0)
    ap.add_argument("--l40-draws", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--allow-local", action="store_true", help="스레드를 --threads 로 쓴다(없으면 1)")
    a = ap.parse_args(argv)
    a.ARGV = list(sys.argv[1:] if argv is None else argv)
    if a.allow_no_repro_gate and os.environ.get("LGD_TEST", "") != "1":
        raise SystemExit("[거부] --allow-no-repro-gate 는 시험 전용이다(환경 변수 LGD_TEST=1 일 때만 받는다. 개정 14)")

    def ab(p):
        return Path(p) if os.path.isabs(str(p)) else ROOT / str(p)
    a.LG = ab(a.lg_dir); a.LGD = ab(a.lgd_dir); a.OUT = ab(a.out_dir) if a.out_dir else a.LGD
    a.ELIG = ab(a.eligibility); a.SPLITDIST = ab(a.splitdist); a.XENV = ab(a.xenv_gate)
    a.REPRO = ab(a.repro_gate) if a.repro_gate else a.LGD / "lgd_repro_gate.csv"
    return a


def sha256(path: Path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()
    except OSError:
        return ""


# ================================================================ 풀 정의와 MAIN_POINT(WRAPUP 7.4 (b)1)
def pools_from_eligibility(el):
    """적격 표 → 풀. 새 지역 = kind 가 new_region 으로 시작하는 행. 적격이고 얕은 레짐 → PE1·PE2, 적격이고 심부 → PE2, 부적격 → 점 추정."""
    nr = el[el.kind.astype(str).str.startswith("new_region")]
    shallow = [f"{t}|x" for t, e, g in zip(nr.target, nr.eligible, nr.regime) if bool(e) and g == "shallow"]
    deep = [f"{t}|x" for t, e, g in zip(nr.target, nr.eligible, nr.regime) if bool(e) and g == "deep"]
    inel = [str(t) for t, e in zip(nr.target, nr.eligible) if not bool(e)]
    regime = {f"{t}|x": str(g) for t, g in zip(nr.target, nr.regime)}
    regime.update({nm: "shallow" for nm in P4})
    few = {f"{t}|x": float(v) for t, v in zip(nr.target, nr.min_nb_eval_used)}
    return dict(P4=list(P4), PE1=list(P4) + shallow, PE2=list(P4) + shallow + deep, NEW1=shallow, NEW2=shallow + deep, ineligible=inel,
                regime=regime, min_nb_eval=few)


def set_main_point(ineligible, point_only_names=()):
    """h40.MAIN_POINT 를 적격 표의 부적격 대상(+ 점 추정 전용 변형 저장소의 대상 이름)으로 바꾼다. TMx 를 만들기 전에 부른다."""
    H.MAIN_POINT = sorted(set(str(t) for t in ineligible) | set(str(t) for t in point_only_names))
    return list(H.MAIN_POINT)


# ================================================================ 조각 읽기
def find_cpu_shards(shards_dir: Path, tag: str):
    out = []
    if not shards_dir.exists():
        return out
    for p in sorted(shards_dir.glob(f"{tag}__cpu__*_unit.json")):
        parts = p.name[:-len("_unit.json")].split("__")
        if len(parts) != 5 or parts[0] != tag:
            continue
        b = str(p)[:-len("_unit.json")]
        if Path(b + "_runs.csv").exists() and Path(b + "_blocksse.npz").exists():
            out.append(dict(unit=p, runs=Path(b + "_runs.csv"), npz=Path(b + "_blocksse.npz"), target=parts[2], mode=parts[3],
                            split=int(parts[4][1:])))
    return out


def read_runs(paths):
    frames = []
    for p in paths:
        if p.stat().st_size <= 1:
            continue
        f = pd.read_csv(p, dtype=dict(alpha=str, alpha_sel=str, fit_flag=str), keep_default_na=False, na_values=["", "nan", "NaN"])
        frames.append(f)
    if not frames:
        return pd.DataFrame()
    runs = pd.concat(frames, ignore_index=True)
    for c, v in (("alpha_sel", ""), ("fit_flag", ""), ("n_nonfinite", 0)):
        if c not in runs:
            runs[c] = v
    runs["alpha_sel"] = runs.alpha_sel.fillna("").astype(str)
    runs["n_nonfinite"] = runs.n_nonfinite.fillna(0).astype(int); runs["fit_flag"] = runs.fit_flag.fillna("").astype(str)
    return runs


def load_group(shards_dir: Path, tag: str, target: str, name: str, platform: str):
    """한 디렉터리의 한 대상(모드 x) cpu 조각을 읽는다. 반환 dict(by_split, info, runs, units, platform) 또는 None."""
    sh = [s for s in find_cpu_shards(shards_dir, tag) if s["target"] == target and s["mode"] == "x"]
    if not sh:
        return None
    stores = load_stores([s["npz"] for s in sh])
    units = [json.loads(s["unit"].read_text()) for s in sh]
    by_split = {sp: st for (nm, sp), st in stores.items() if nm == f"{target}|x"}
    info = {int(u["split"]): dict(dup_of=int(u.get("dup_of", -1)), valid=bool(u.get("valid", True))) for u in units}
    runs = read_runs([s["runs"] for s in sh])
    if len(runs):
        runs = runs[(runs["mode"] == "x") & (runs.target == target)].drop_duplicates(subset=["target", "mode", "split"] + H.KEY_COLS, keep="first")
        runs = runs.copy(); runs["target"] = name.split("|")[0]
    return dict(by_split=by_split, info=info, runs=runs, units=units, platform=platform, dir=str(shards_dir),
                code_sha=sorted({str(u.get("code_sha", "")) for u in units}), cfg_common=sorted({str(u.get("cfg_common", "")) for u in units}),
                status=sorted({str(u.get("status", "")) for u in units}))


def method_view(tm):
    """방법 축 키(학습기 none·catboost_lo, α 1, 셀 무작위)만 남긴 TMx 사본."""
    t2 = copy.copy(tm)
    t2.idx = {sp: {g: ks for g, ks in gd.items() if g[1] in ("none", BASE) and str(g[2]) == "1" and g[3] == "cell"} for sp, gd in tm.idx.items()}
    return t2


def make_tm(name, L, nboot):
    return H.TMx(name, L["by_split"], L["info"], nboot)


def variant_base(spec, spec_rows=None):
    """L41·L42 변형의 주 설정 표 이름(Tibet, NAtlantic, Russia_C). 그 밖의 표는 None."""
    for sp in R.NEW_MAIN:
        if spec.startswith(f"{sp}_L41") or spec == f"{sp}_L42":
            return sp
    return None


def crn_seed(tm, main_name):
    """L41·L42 변형의 공통 난수(개정 14). h40.TMx 는 seed_of('lgboot', name)을, h42.boot_delta_common 은 seed_of('lgxboot', tm.name)을
    쓰므로 두 값을 주 설정 저장소 이름으로 바꾼다. tm.target(곡선·행의 이름)은 변형 이름 그대로다."""
    tm.seed = seed_of("lgboot", main_name)
    tm.name = main_name
    return tm


# ================================================================ 대비 도우미
def G(method, n):
    return X.G(method, n) if method != "P0" else H.P0_GRP


def rstats(tm, ma, mb, n):
    """지역 대비 통계(h42.region_stats). mb 가 P0 이면 기준은 P0(n 무관)."""
    if tm is None:
        return None
    return X.region_stats(tm, G(ma, n), H.P0_GRP if mb == "P0" else G(mb, n))


def row_of(s_, a, target, extra=None):
    """4분 판정 행. 주 CI 와 공통 재표집 CI 의 판정이 다르면 '분할 독립 가정 의존'(WRAPUP 7.2 (a)4)."""
    if s_ is None:
        r = dict(target=target, verdict4="행 없음", verdict4_common="행 없음", ci_dependence="")
        r.update(extra or {})
        return r
    r = X.stats_row(s_, a, target, extra)
    r["ci_dependence"] = "분할 독립 가정 의존" if (r["verdict4"] not in NA and r["verdict4_common"] not in NA
                                                  and r["verdict4"] != r["verdict4_common"]) else ""
    return r


def l28_shift(v0, v1):
    """L28 분류(6A.5a): 같으면 강건, 동등과 미결정 사이도 강건, 우세·열세와 동등·미결정 사이는 약화, 우세와 열세가 바뀌면 의존."""
    if v0 in NA or v1 in NA:
        return "판정 불가"
    if v0 == v1:
        return "강건"
    if {v0, v1} == {"우세", "열세"}:
        return "의존"
    if (v0 in ("우세", "열세")) != (v1 in ("우세", "열세")):
        return "약화"
    return "강건"


def worst_shift(vals):
    vals = list(vals)
    if "의존" in vals:
        return "의존"
    if "약화" in vals:
        return "약화"
    ok = [v for v in vals if not str(v).startswith("판정 불가")]
    if not ok:
        return f"판정 불가(판정한 대비 0/{len(vals)})"
    return "강건" + (f"(판정한 대비 {len(ok)}/{len(vals)})" if len(ok) < len(vals) else "")


def sym_dir(verdict4, delta, hyp_sign):
    """WRAPUP 7.2 (a)3 의 대칭 이름. hyp_sign = +1(가설 방향 Δ ≥ 0) 또는 −1(Δ < 0). 판정과 원고 문장에 쓰지 않는다."""
    if verdict4 in NA or delta is None or not np.isfinite(delta):
        return ""
    if hyp_sign > 0:
        if verdict4 == "열세":
            return "같은 방향(우세)"
        if verdict4 == "우세":
            return "반대(열세)"
        return "같은 부호(비유의)" if delta >= 0 else "반대 부호(비유의)"
    if verdict4 == "우세":
        return "같은 방향(우세)"
    if verdict4 == "열세":
        return "반대(열세)"
    return "같은 부호(비유의)" if delta < 0 else "반대 부호(비유의)"


# ================================================================ 교차 환경·재현 점검 상태
def xenv_status(a):
    """교차 환경 점검 (i)의 상태 문자열(WRAPUP 8.4, 8.6 (3)). auto 이면 요약 JSON 에서 정한다. 물리식 최대 차(max_phys) > 1e-9 cm 또는 채점 블록
    불일치(blocks_equal 거짓)이면 '물리식 불통과', ridge·CatBoost 최대 차(max_ml) > 0.02 cm 이면 'CatBoost 불통과', 둘 다 안이면 '통과'.
    요약의 점검 범위(scope.units_done / units_expected)가 등록 범위보다 작으면 '; 부분(k/K 단위, 잠정)', 범위 기록이 없으면
    '; 범위 미기록(부분 점검일 수 있다)' 을 붙인다."""
    if a.xenv_i != "auto":
        return {"pass": "(i) 통과", "catboost_fail": "(i) CatBoost 불통과", "phys_fail": "(i) 물리식 불통과", "unknown": "(i) 미확인"}[a.xenv_i], {}
    if not a.XENV.exists():
        return "(i) 미확인", {}
    d = json.loads(a.XENV.read_text())
    mp, mm = float(d.get("max_phys", np.nan) if d.get("max_phys") is not None else np.nan), float(
        d.get("max_ml", np.nan) if d.get("max_ml") is not None else np.nan)
    if not np.isfinite(mp):
        return "(i) 미확인", d
    if mp > R.GATE_TOL_PHYS or d.get("blocks_equal") is False:
        st = "(i) 물리식 불통과"
    else:
        st = "(i) 통과" if np.isfinite(mm) and mm <= R.GATE_TOL_ML else "(i) CatBoost 불통과"
    sc = d.get("scope") if isinstance(d.get("scope"), dict) else None
    if sc is None or sc.get("units_done") is None or sc.get("units_expected") is None:
        st += "; 범위 미기록(부분 점검일 수 있다)"
    elif int(sc["units_done"]) < int(sc["units_expected"]):
        st += f"; 부분({int(sc['units_done'])}/{int(sc['units_expected'])} 단위, 잠정)"
    return st, d


def repro_all_scope(a):
    """재현 점검의 병기 범위(scope = all, 방법 축 전체) 행 요약(메타 기록용, 상태를 정하지 않는다)."""
    if not a.REPRO.exists():
        return {}
    g = pd.read_csv(a.REPRO)
    if "scope" not in g:
        return {}
    g = g[g.scope.astype(str) == "all"]
    cols = [c for c in ("status", "n_keys", "max_diff_phys", "max_diff_ridge", "max_diff_catboost", "n_ml_over_tol", "methods_ml_over_tol") if c in g]
    return g[cols].to_dict("records")


def repro_status(a):
    """(c)2 재현 점검. 반환 (상태, 물리식 통과, CatBoost 통과). 물리식 불통과면 호출자가 집계를 멈춘다."""
    if not a.REPRO.exists():
        return "재현 점검 미실시", None, None
    g = pd.read_csv(a.REPRO)
    if "scope" in g:                                              # 개정 14: 주 판정 범위(6A.6 기준 방법)의 행으로 정한다
        g = g[g.scope.astype(str) == "base"]
    if not len(g) or "pass_phys" not in g or g.pass_phys.isna().all():
        return "재현 점검 기록 불완전", None, None
    pp = bool(g.pass_phys.fillna(False).astype(bool).all() and g.get("blocks_equal", pd.Series([True])).fillna(False).astype(bool).all())
    pm = bool(g.pass_ml.fillna(False).astype(bool).all())
    return ("재현 점검 통과" if pp and pm else ("재현 점검 물리식 불통과" if not pp else "재현 점검 CatBoost 불통과")), pp, pm


# ================================================================ L1e, L4e, L8e(재표집 1,000회, h40 규칙)
def curve_row(cur, name, method, n, lam=None):
    t = name.split("|")[0]
    g = H._g(method, n) if lam is None else H._g(method, n, lam=lam)
    m = cur[(cur.target == t) & (cur["mode"] == "x")]
    for col, v in zip(H.GRP_COLS, g):
        m = m[m[col] == v]
    return m.iloc[0] if len(m) else None


def common_ci(tm, gA, gB):
    """WRAPUP 7.2 (a)4: 공통 재표집 보조 CI(h42.region_stats → boot_delta_common, tm 의 재표집 횟수). CI 풀 조건은 주 CI 와 같다.
    반환 dict(ci_lo_c, ci_hi_c, ci_lo_beq_c, ci_hi_beq_c, sig_common, cdist, cdist_beq)."""
    out = dict(ci_lo_c=np.nan, ci_hi_c=np.nan, ci_lo_beq_c=np.nan, ci_hi_beq_c=np.nan, sig_common="", cdist=None, cdist_beq=None)
    if tm is None:
        return out
    s_ = X.region_stats(tm, gA, gB)
    if s_ is None or s_.get("cdist") is None:
        return out
    lo, hi = X._ci(s_["cdist"]); lob, hib = X._ci(s_["cdist_beq"])
    out.update(ci_lo_c=lo, ci_hi_c=hi, ci_lo_beq_c=lob, ci_hi_beq_c=hib, sig_common=H._sig(lo, hi, lob, hib), cdist=s_["cdist"],
               cdist_beq=s_["cdist_beq"])
    return out


def dep_flag(sig, sig_c):
    """주 CI 와 공통 재표집 CI 의 판정(sig)이 다르면 '분할 독립 가정 의존'."""
    return "분할 독립 가정 의존" if (sig and sig_c and str(sig) != str(sig_c)) else ""


def attach_common(rows, tms, gA_fn, gB_fn):
    """h40.strat_mean 의 지역 행과 MEAN 행에 공통 재표집 보조 CI 를 붙인다(행을 고친다). MEAN 의 공통 분포는 MEAN 이름의 CI 풀 지역 가운데
    공통 분포가 있는 지역의 층화 결합(MS.strat)이다. 반환 공통 CI 와 판정이 다른 행 수."""
    per, n_dep = {}, 0
    for r in rows:
        t = str(r.get("target", ""))
        if t.startswith("MEAN["):
            continue
        tm = tms.get(t)
        c = common_ci(tm, gA_fn(tm), gB_fn(tm)) if tm is not None else common_ci(None, None, None)
        per[t] = c
        r.update({k: v for k, v in c.items() if k not in ("cdist", "cdist_beq")})
        r["ci_dependence"] = dep_flag(r.get("sig", ""), c["sig_common"])
        n_dep += bool(r["ci_dependence"])
    for r in rows:
        t = str(r.get("target", ""))
        if not t.startswith("MEAN["):
            continue
        pool = [nm for nm in t[5:-1].split(",") if nm in per and per[nm]["cdist"] is not None]
        if len(pool) >= H.MIN_POOL_REGIONS:
            lo, hi = MS.ci(MS.strat([per[nm]["cdist"] for nm in pool])); lob, hib = MS.ci(MS.strat([per[nm]["cdist_beq"] for nm in pool]))
            sc = H._sig(lo, hi, lob, hib)
        else:
            lo = hi = lob = hib = np.nan; sc = ""
        r.update(ci_lo_c=lo, ci_hi_c=hi, ci_lo_beq_c=lob, ci_hi_beq_c=hib, sig_common=sc, n_common_regions=len(pool))
        r["ci_dependence"] = dep_flag("" if H.mean_state(r) == "undetermined" else r.get("sig", ""), sc)
        n_dep += bool(r["ci_dependence"])
    return n_dep


def l1e(cur, names, pool, tms=None):
    """L1e: 유의 개선(두 가중 CI 상한 < 0) 지역이 ⌊N/4⌋ 이하이면 지지. 전량 행은 실제 라벨 수 ≤ 40 인 지역만(h40 L1).
    tms 가 있으면 지역 행마다 공통 재표집 보조 CI 를 붙인다(WRAPUP 7.2 (a)4. 판정에 쓰지 않는다)."""
    N, thr = len(names), len(names) // 4
    rows, n_beat, n_tested, det, n_dep = [], 0, 0, [], 0
    ns_all = sorted({int(v) for v in cur.n.unique()}) if len(cur) else []
    for nm in names:
        beat, tested = [], []
        for n in [n for n in ns_all if 0 <= n <= 40] + ([-1] if -1 in ns_all else []):
            r = curve_row(cur, nm, "D0", n)
            if r is None:
                continue
            if n == -1 and not (np.isfinite(r.n_lab) and r.n_lab <= 40):
                continue
            tested.append(n)
            row = dict(test_id="L1e", pool=pool, item="D0-P0", scope="region", target=nm, n=n, n_lab=r.n_lab, delta=r.d_p0, ci_lo=r.d_p0_lo,
                       ci_hi=r.d_p0_hi, delta_blockeq=r.d_p0_beq, ci_lo_beq=r.d_p0_beq_lo, ci_hi_beq=r.d_p0_beq_hi, sig=r.sig_p0,
                       ci_flag=r.ci_flag, note="전량(라벨 수 ≤ 40)" if n == -1 else "")
            if tms is not None:
                c = common_ci(tms.get(nm), H._g("D0", n), H.P0_GRP)
                row.update({k: v for k, v in c.items() if k not in ("cdist", "cdist_beq")})
                row["ci_dependence"] = dep_flag(r.sig_p0, c["sig_common"])
                n_dep += bool(row["ci_dependence"])
            rows.append(row)
            if r.sig_p0 == "improve":
                beat.append(n)
        n_tested += bool(tested); n_beat += bool(beat)
        det.append(f"{nm.split('|')[0]}: 유의 {beat}, 검사 {tested}")
    if n_beat > thr:
        txt = "기각"
    elif n_tested == N and N > 0:
        txt = "지지"
    elif n_tested == 0:
        txt = "판정 불가(행 없음)"
    else:
        txt = f"부분(지역 {n_tested}/{N}): 검사한 지역에서는 지지 조건 충족"
    binary = None if n_tested == 0 else ("지지" if n_beat <= thr else "기각")
    stat = f"두 가중 CI 상한 < 0 인 지역 {n_beat}/{n_tested}(기각 기준 > ⌊N/4⌋ = {thr}); 유의 개선 지역 비율 k/N = {n_beat}/{N} ({'; '.join(det)})"
    rows.append(dict(test_id="L1e", pool=pool, item="verdict", scope="verdict", verdict=txt, stat=stat, binary=binary, n_regions=N,
                     k_improve=n_beat, n_tested=n_tested, n_ci_dependence=n_dep if tms is not None else np.nan))
    return rows, binary


def mean_state_row(rows):
    mr = [r for r in rows if str(r.get("target", "")).startswith("MEAN[")]
    if not mr:
        return None, "undetermined", 0
    r = mr[0]
    return r, H.mean_state(r), int(r.get("n_ci_regions", 0))


def rel_strat(tms, names, gA, gB):
    """상대 단위 층화 평균(6B.5): 지역 Δ 와 분포를 그 지역 P0 RMSE 점 추정(셀 가중)으로 나눈다. 반환 (점, 셀 CI, 블록 CI, 상태, 풀 지역 수)."""
    per = {}
    for nm in names:
        tm = tms.get(nm)
        if tm is None:
            continue
        r = H.contrast(tm, gA, gB, return_dist=True)
        if r is None or not np.isfinite(r["delta"]):
            continue
        p0 = X.grp_rmse(tm, H.P0_GRP)
        if not np.isfinite(p0) or p0 <= 0:
            continue
        ok = tm.has_ci and tm.nb_union >= MS.MIN_BLOCKS_CI and "dist" in r
        per[nm] = dict(d=r["delta"] / p0, db=r["delta_beq"] / p0, dist=(r["dist"] / p0) if ok else None, distb=(r["dist_beq"] / p0) if ok else None)
    pool = [nm for nm in per if per[nm]["dist"] is not None]
    if len(pool) < H.MIN_POOL_REGIONS:
        return dict(delta=np.nan, ci_lo=np.nan, ci_hi=np.nan, ci_lo_beq=np.nan, ci_hi_beq=np.nan, state="undetermined", k=len(pool))
    d = float(np.mean([per[nm]["d"] for nm in pool]))
    sd, sdb = MS.strat([per[nm]["dist"] for nm in pool]), MS.strat([per[nm]["distb"] for nm in pool])
    lo, hi = MS.ci(sd); lob, hib = MS.ci(sdb)
    st = H._sig(lo, hi, lob, hib) or "undetermined"
    if np.isfinite(lo) and (d < lo - 1e-9 or d > hi + 1e-9):
        st = "undetermined"
    return dict(delta=d, ci_lo=lo, ci_hi=hi, ci_lo_beq=lob, ci_hi_beq=hib, state=st, k=len(pool))


AUX10 = (("R1", "P2"), ("R1", "P3"), ("P2", "P1"))                                            # WRAPUP 7.2 (a)10 의 병기 대비


def l4e(tms, cur, names, pool, rel=False, flags="", aux10=False):
    """L4e: R1 − P1 층화 평균 CI 상한 < 0 인 최소 n(h40 L4 와 같은 규칙, 풀 지역 N). n 마다 k/N. (a)1 의 이진 비교 = n ≤ 10 에서 성립.
    지역 행과 MEAN 행에 공통 재표집 보조 CI((a)4). aux10 이면 n 마다 R1 − P2, R1 − P3, P2 − P1 의 층화 평균과 지역 행을 병기한다((a)10,
    판정에 쓰지 않는다)."""
    N = len(names)
    rows, res, n_dep = [], [], 0
    tg = {nm.split("|")[0] for nm in names}
    ns = sorted({int(v) for v in cur[cur.target.isin(tg) & (cur.method == "R1")].n.unique() if v > 0}) if len(cur) else []
    ns += [-1] if len(cur) and (-1 in set(cur[cur.target.isin(tg)].n.unique())) else []
    for n in ns:
        out = H.strat_mean(f"L4e|{pool}|R1-P1|n{n}", tms, names, lambda tm, n=n: H._g("R1", n), lambda tm, n=n: H._g("P1", n))
        n_dep += attach_common(out, tms, lambda tm, n=n: H._g("R1", n), lambda tm, n=n: H._g("P1", n))
        for r in out:
            rows.append(dict(test_id="L4e", pool=pool, item="R1-P1", scope="MEAN" if str(r.get("target", "")).startswith("MEAN[") else "region",
                             n=n, lam=H.LAM_BASE, flags=flags, **{k: v for k, v in r.items() if k not in ("test",)}))
        if aux10:
            for ma, mb in AUX10:
                ga = (lambda tm, n=n, ma=ma: H._g(ma, n)); gb = (lambda tm, n=n, mb=mb: H._g(mb, n))
                ox = H.strat_mean(f"L4e|{pool}|{ma}-{mb}|n{n}", tms, names, ga, gb)
                attach_common(ox, tms, ga, gb)
                for r in ox:
                    rows.append(dict(test_id="L4e", pool=pool, item=f"{ma}-{mb}", scope="MEAN" if str(r.get("target", "")).startswith("MEAN[")
                                     else "region", n=n, role="병기((a)10, 판정에 쓰지 않는다)", flags=";".join(v for v in (flags, "(a)10 병기") if v),
                                     aux_a10=True, **{k: v for k, v in r.items() if k not in ("test",)}))
        mr, stt, k = mean_state_row(out)
        e = [n, stt, k]
        if rel:
            rr = rel_strat(tms, names, H._g("R1", n), H._g("P1", n))
            rows.append(dict(test_id="L4e", pool=pool, item="R1-P1(상대 단위)", scope="MEAN_rel", n=n, delta=rr["delta"], ci_lo=rr["ci_lo"],
                             ci_hi=rr["ci_hi"], ci_lo_beq=rr["ci_lo_beq"], ci_hi_beq=rr["ci_hi_beq"], sig=rr["state"], n_ci_regions=rr["k"],
                             scale_dependence="척도 의존" if rr["state"] != stt else ""))
            e.append(rr["state"])
        res.append(e)
    fin = [e for e in res if e[0] > 0 and e[1] != "undetermined"]
    fN = [e[0] for e in fin if e[1] == "improve" and e[2] == N]
    fp = [e[0] for e in fin if e[1] == "improve" and e[2] < N]
    nN = min(fN) if fN else None; np_ = min(fp) if fp else None
    kp = {e[0]: e[2] for e in fin}
    if not fin:
        txt = "판정 불가"
    elif nN is not None and nN <= 40:
        txt = "희소 라벨에서도 ML 순가치 있음"
    elif np_ is not None and np_ <= 40:
        txt = f"부분(지역 {kp[np_]}/{N}): 희소 라벨에서 순가치 있음"
    elif nN is not None or np_ is not None:
        nn = min(v for v in (nN, np_) if v is not None)
        txt = "순가치는 n > 40 에서만" + ("" if kp[nn] == N else f"(부분, 지역 {kp[nn]}/{N})")
    else:
        txt = "유한 n 격자에서 순가치 미확인"
    le10 = [e for e in res if 0 < e[0] <= 10]
    if not le10 or all(e[1] == "undetermined" for e in le10):
        binary = None
    else:
        binary = "성립" if any(e[1] == "improve" for e in le10) else "불성립"
    scale = ""
    if rel:
        dif = [e[0] for e in res if len(e) > 3 and e[1] != e[3]]
        scale = f"척도 의존(n = {dif})" if dif else ""
    stat = (f"두 가중 CI 상한 < 0 인 최소 n: 풀 전체 {nN}, 일부 지역 {np_}; n ≤ 10 성립 {binary}; "
            + "; ".join(f"n={e[0]}: {e[1]}(지역 {e[2]}/{N})" for e in res))
    rows.append(dict(test_id="L4e", pool=pool, item="verdict", scope="verdict", verdict=txt, stat=stat, binary=binary, n_star=nN, n_star_partial=np_,
                     scale_dependence=scale, flags=flags, n_regions=N, n_ci_dependence=n_dep))
    return rows, binary, res


def l8e(tms, cur, names, pool, rel=False):
    """L8e: 전량 R1 − P0 의 층화 평균 CI 상한 < 0(두 가중)이면 지지. (a)11: 지역별 실제 라벨 수와 전량 R1 − P1, 라벨 수 묶음 평균."""
    N = len(names)
    out = H.strat_mean(f"L8e|{pool}|R1-P0|all", tms, names, lambda tm: H._g("R1", -1), lambda tm: H.P0_GRP)
    n_dep = attach_common(out, tms, lambda tm: H._g("R1", -1), lambda tm: H.P0_GRP)
    rows = []
    for r in out:
        is_m = str(r.get("target", "")).startswith("MEAN[")
        d = dict(test_id="L8e", pool=pool, item="R1-P0", scope="MEAN" if is_m else "region", n=-1, lam=H.LAM_BASE,
                 **{k: v for k, v in r.items() if k not in ("test",)})
        if not is_m:
            cr = curve_row(cur, r["target"], "R1", -1)
            if cr is not None:
                d.update(n_lab=cr.n_lab, r1_p1_all=cr.d_p1, r1_p1_all_lo=cr.d_p1_lo, r1_p1_all_hi=cr.d_p1_hi, r1_p1_all_beq_lo=cr.d_p1_beq_lo,
                         r1_p1_all_beq_hi=cr.d_p1_beq_hi, r1_p1_all_sig=cr.sig_p1)
        rows.append(d)
    mr, stt, k = mean_state_row(out)
    if mr is None or stt == "undetermined":
        txt = f"판정 불가(지역 {k}/{N})"; binary = None
    elif k == N:
        txt = "지지" if stt == "improve" else "기각"; binary = txt
    else:
        txt = f"부분(지역 {k}/{N}): " + ("지지 조건 충족" if stt == "improve" else "기각"); binary = "지지" if stt == "improve" else "기각"
    reg = [r for r in rows if r["scope"] == "region" and np.isfinite(r.get("n_lab", np.nan))]
    hi_ = [r for r in reg if r["n_lab"] >= 160]; lo_ = [r for r in reg if r["n_lab"] <= 40]
    grp = "; ".join(f"{lab}(지역 {len(g)}: {', '.join(x['target'].split('|')[0] for x in g)}) R1 − P0 평균 "
                    f"{np.mean([x['delta'] for x in g]):.2f}, R1 − P1 평균 {np.nanmean([x.get('r1_p1_all', np.nan) for x in g]):.2f} cm"
                    for lab, g in (("전량 라벨 ≥ 160", hi_), ("전량 라벨 ≤ 40", lo_)) if g)
    scale = ""
    if rel:
        rr = rel_strat(tms, names, H._g("R1", -1), H.P0_GRP)
        rows.append(dict(test_id="L8e", pool=pool, item="R1-P0(상대 단위)", scope="MEAN_rel", n=-1, delta=rr["delta"], ci_lo=rr["ci_lo"],
                         ci_hi=rr["ci_hi"], ci_lo_beq=rr["ci_lo_beq"], ci_hi_beq=rr["ci_hi_beq"], sig=rr["state"], n_ci_regions=rr["k"]))
        scale = "척도 의존" if rr["state"] != stt else ""
    stat = "; ".join(f"Δ {r.get('delta', np.nan):.2f} [{r.get('ci_lo', np.nan):.2f}, {r.get('ci_hi', np.nan):.2f}], 블록 등가중 "
                     f"[{r.get('ci_lo_beq', np.nan):.2f}, {r.get('ci_hi_beq', np.nan):.2f}], 지역 {r.get('n_ci_regions', 0)}/{N}"
                     for r in rows if r["scope"] == "MEAN")
    rows.append(dict(test_id="L8e", pool=pool, item="verdict", scope="verdict", verdict=txt, stat=stat, binary=binary, scale_dependence=scale,
                     label_groups=grp, n_regions=N, n_ci_dependence=n_dep,
                     note="ML 순가치 문장은 L4e 에서만 쓴다. LGD 는 n ∈ {0, 3, 10} 과 전량 행에만 지역을 더한다(새 얕은 지역의 |A| 는 약 19–29)"))
    return rows, binary


# ================================================================ (a)1 비교(4분 판정 풀 대비와 반전·희석)
def pool4(tms10, names, ma, mb, n, a, label):
    """풀 대비의 4분 판정(10,000회, h42.pool_rows). 반환 (MEAN 행 또는 None, 지역 행)."""
    per = {}
    for nm in names:
        s_ = rstats(tms10.get(nm), ma, mb, n)
        if s_ is not None:
            per[nm] = s_
    rows = X.pool_rows(per, names, a, label) if per else []
    for r in rows:                                                # (a)4: 지역 행과 MEAN 행
        r["ci_dependence"] = "분할 독립 가정 의존" if (r["verdict4"] not in NA and r["verdict4_common"] not in NA
                                                      and r["verdict4"] != r["verdict4_common"]) else ""
    mr = next((r for r in rows if r.get("scope") == "MEAN"), None)
    return mr, rows


def compare_contrasts(hyp):
    if hyp == "L1e":
        return [("D0", "P0", n) for n in L1_NS]
    if hyp == "L4e":
        return [("R1", "P1", 3), ("R1", "P1", 10), ("R1", "P1", N_ALL)]
    return [("R1", "P0", N_ALL)]


def phrase_6b5(b4, b1, b2, n_new, npe2):
    """6B.5 의 병기 문구."""
    if n_new == 0:
        return "확장 풀을 구성하지 못했다"
    if b4 is None or b1 is None or b2 is None:
        return "판정 불가(§4 의 판정 세부 규칙: 지역 k/N 표기)"
    if b4 == b1 == b2:
        return f"4지역의 판정이 확장 풀(지역 {npe2}개)에서 유지된다"
    if b4 == b1 and b1 != b2:
        return "얕은 레짐의 확장 풀에서 유지된다. 심부 레짐(티베트)을 더하면 달라진다"
    return "4지역의 판정은 확장 풀에서 유지되지 않는다"


def compare_rows(hyp, bins, pools, tms10, a, cross_txt):
    """(a)1: P4 대 PE1·PE2 의 이진 비교, 병기 문구, 판정이 다르면 L28 분류와 반전·희석."""
    rows = []
    b4, b1, b2 = bins.get("P4"), bins.get("PE1"), bins.get("PE2")
    txt = phrase_6b5(b4, b1, b2, len(pools["NEW2"]), len(pools["PE2"]))
    extra = []
    for pe, new_names, bx in (("PE1", pools["NEW1"], b1), ("PE2", pools["NEW2"], b2)):
        if bx is None or b4 is None or bx == b4 or not new_names:
            continue
        shifts, notes = [], []
        for ma, mb, n in compare_contrasts(hyp):
            lab = f"{ma}-{mb}|n{'all' if n == N_ALL else n}"
            r4, _ = pool4(tms10, pools["P4"], ma, mb, n, a, f"{hyp}|P4|{lab}")
            rx, _ = pool4(tms10, pools[pe], ma, mb, n, a, f"{hyp}|{pe}|{lab}")
            rn, rn_reg = pool4(tms10, new_names, ma, mb, n, a, f"{hyp}|{pe}-new|{lab}")
            if len(new_names) == 1:                                   # 새 지역이 하나면 그 지역 행을 새 지역 평균으로 쓴다(풀 평균은 판정 불가)
                rn = next((r for r in rn_reg if r.get("scope") == "region"), None)
            v4 = r4["verdict4"] if r4 else "행 없음"; vx = rx["verdict4"] if rx else "행 없음"
            sh = l28_shift(v4, vx)
            shifts.append(sh)
            kind = ""
            if rn is not None and r4 is not None and rn.get("verdict4") in ("우세", "열세") and np.isfinite(r4.get("delta", np.nan)):
                opp = (rn["verdict4"] == "우세" and r4["delta"] > 0) or (rn["verdict4"] == "열세" and r4["delta"] < 0)
                kind = "반전" if opp else "희석"
            elif rn is not None:
                kind = "희석"
            nt = (f"{kind}(새 지역 평균 {rn.get('delta', np.nan):.2f} [{rn.get('ci_lo', np.nan):.2f}, {rn.get('ci_hi', np.nan):.2f}], "
                  f"{rn.get('verdict4', '')})") if rn is not None else "새 지역 평균 없음"
            notes.append(f"{lab}: P4 {v4} → {pe} {vx}, L28 {sh}; {nt}")
            for tag_, r_ in (("P4", r4), (pe, rx), (f"{pe}-new", rn)):
                if r_ is not None:
                    rows.append(dict(test_id=hyp, pool=tag_, item=f"(a)1 풀 대비 {lab}", scope="compare_pool", n=n,
                                     **{k: v for k, v in r_.items() if k in ("target", "delta", "ci_lo", "ci_hi", "delta_blockeq", "ci_lo_beq",
                                                                             "ci_hi_beq", "ci_lo_c", "ci_hi_c", "ci_lo_beq_c", "ci_hi_beq_c",
                                                                             "verdict4", "verdict4_d10", "verdict4_common", "ci_dependence",
                                                                             "n_ci_regions", "pool_regions", "p_boot", "p_eq")},
                                     l28=sh if tag_ == pe else "", cross_env=cross_txt if tag_ != "P4" else ""))
        extra.append(f"{pe}: L28 {worst_shift(shifts)}. " + " | ".join(notes))
    rows.append(dict(test_id=hyp, pool="P4·PE1·PE2", item="(a)1 비교", scope="compare", verdict=txt + (". " + " / ".join(extra) if extra else ""),
                     stat=f"P4 {b4}, PE1 {b1}, PE2 {b2}", cross_env=cross_txt))
    return rows


# ================================================================ (a)2 지역 수준 추론
def region_inference_rows(tms10, pools):
    core = [("D0-P0|n0", "D0", "P0", 0), ("D0-P0|n10", "D0", "P0", 10), ("R1-P1|n3", "R1", "P1", 3), ("R1-P1|n10", "R1", "P1", 10),
            ("R1-P0|all", "R1", "P0", N_ALL)]
    rows = []
    for lab, ma, mb, n in core:
        per = {}
        for nm in sorted(set(pools["PE2"]) | set(pools["P4"])):
            s_ = rstats(tms10.get(nm), ma, mb, n)
            if s_ is not None:
                d_ = s_["dist"]
                per[nm] = (s_["delta"], float(np.nanvar(d_, ddof=1)) if d_ is not None and np.isfinite(d_).sum() > 2 else np.nan)
        for pname in ("P4", "PE1", "PE2"):
            used = [nm for nm in pools[pname] if nm in per]
            if len(used) < 2:
                rows.append(dict(contrast=lab, pool=pname, regions=",".join(used), k=len(used), note="지역 2개 미만"))
                continue
            th = np.array([per[nm][0] for nm in used], float); vv = np.array([per[nm][1] for nm in used], float)
            ri = X.region_inference(th, vv)
            hk_excl = bool(np.isfinite(ri.get("hk_lo", np.nan)) and (ri["hk_lo"] > 0 or ri["hk_hi"] < 0))
            rows.append(dict(contrast=lab, pool=pname, regions=",".join(used), neg_ratio=f"{ri['n_neg']}/{ri['k']}", **ri,
                             wording="지역 일반" if hk_excl else f"{len(used)}개 지역에 조건부인 평균", role="보조(판정에 쓰지 않는다)"))
            for j, nm in enumerate(used):
                rows.append(dict(contrast=lab, pool=f"{pname} − {nm}", regions=",".join(u for u in used if u != nm), k=len(th) - 1,
                                 mean=float(np.delete(th, j).mean()), note="지역 하나 제외 평균"))
    return pd.DataFrame(rows)


# ================================================================ L38 과 그 변형(L41, L42)
def l1_ns_of(tm10, nlab_all):
    ns = sorted({int(g[4]) for gd in tm10.idx.values() for g in gd if g[0] == "D0" and 0 <= int(g[4]) <= 40}) if tm10 else []
    if nlab_all is not None and np.isfinite(nlab_all) and nlab_all <= 40:
        ns.append(-1)
    return ns


UNBLIND_STAT = "비맹검(라벨 통계 열람)"                                                         # 6B.7
UNBLIND_PRED = "예측된 결과(비맹검)"                                                           # WRAPUP 7.2 (a)10
UNBLIND_REPRO = "재현(비맹검)"                                                                 # 6B.7(L40)


def add_flag(r, *flags):
    r["flags"] = ";".join(v for v in dict.fromkeys([x for x in str(r.get("flags", "") or "").split(";") if x] + [f for f in flags if f]))
    return r


def l38_region(tm10, nm, a, nlab_all, tibet=False, flags=""):
    """L38 (a)–(c) 의 지역 행. 반환 (행 목록, dict(a_pass, b_pass, 판정 문자열들)).
    core = L38 (a)–(c)(D0 − P0, R1 − P0, P1 − P0, R1 − P1)의 4분 판정(L41·L42 의 L28 분류에 쓴다). aux = 티베트의 (a)10 병기 대비
    (R1 − P2, R1 − P3, P2 − P1)의 4분 판정(분류에 넣지 않는다. 개정 14)."""
    rows = []
    va = []
    for n in l1_ns_of(tm10, nlab_all):
        r = row_of(rstats(tm10, "D0", "P0", n), a, nm, dict(item="(a) D0-P0", n=n))
        r["direction"] = sym_dir(r["verdict4"], r.get("delta"), +1)
        va.append(r["verdict4"]); rows.append(r)
    rb = row_of(rstats(tm10, "R1", "P0", N_ALL), a, nm, dict(item="(b) R1-P0", n=N_ALL))
    rb["direction"] = sym_dir(rb["verdict4"], rb.get("delta"), -1)
    rows.append(rb)
    rc = [row_of(rstats(tm10, "P1", "P0", 10), a, nm, dict(item="(c) P1-P0", n=10)),
          row_of(rstats(tm10, "R1", "P1", 10), a, nm, dict(item="(c) R1-P1", n=10))]
    rows += rc
    if tibet:                                                     # WRAPUP 7.2 (a)10
        for ma, mb in AUX10:
            rows.append(row_of(rstats(tm10, ma, mb, 10), a, nm, dict(item=f"(c) {ma}-{mb}", n=10, note="(a)10 병기", aux_a10=True)))
    for r in rows:
        r["flags"] = ";".join(v for v in (flags, r.get("ci_dependence", ""), r.pop("note", "") if "note" in r else "") if v)
        if tibet and ("P0" in r["item"].split("-")[-1]):
            add_flag(r, UNBLIND_PRED, UNBLIND_STAT)
    a_known = bool(va) and all(v not in NA for v in va)
    a_pass = bool(a_known and not any(v == "우세" for v in va))
    b_pass = rb["verdict4"] == "우세"
    shrink = bool(tibet and any(r["item"] == "(c) P2-P1" and r["verdict4"] == "우세" for r in rows))
    return rows, dict(a_pass=a_pass, a_known=a_known, b_pass=b_pass, b_verdict=rb["verdict4"], c=[r["verdict4"] for r in rc], shrink=shrink,
                      core={r["item"] + f"|n{r['n']}": r["verdict4"] for r in rows if not r.get("aux_a10")},
                      aux={r["item"] + f"|n{r['n']}": r["verdict4"] for r in rows if r.get("aux_a10")})


def l38(tms10, pools, a, nlab, cross_txt, tibet_support):
    rows, st = [], {}
    names = pools["NEW2"]
    for nm in names:
        few = pools["min_nb_eval"].get(nm, np.nan)
        flags = "소수 블록" if np.isfinite(few) and few < FEW_BLOCKS else ""
        tib = nm.startswith("Tibet")
        rr, s = l38_region(tms10.get(nm), nm, a, nlab.get(nm), tibet=tib, flags=flags)
        for r in rr:
            r.update(test_id="L38", pool="새 지역", scope="region", regime=pools["regime"].get(nm, ""), cross_env="", few_blocks=flags)
        rows += rr; st[nm] = s
    M = len(names)
    ka = sum(s["a_pass"] for s in st.values()); kb = sum(s["b_pass"] for s in st.values())
    off = [f"{nm.split('|')[0]}({pools['regime'].get(nm, '')}; (a) {'충족' if s['a_pass'] else ('판정 불가' if not s['a_known'] else '불충족')}, "
           f"(b) {s['b_verdict']})" for nm, s in st.items() if not (s["a_pass"] and s["b_pass"])]
    if M == 0:
        txt = "확장 풀을 구성하지 못했다(새 적격 지역 없음)"
    elif ka == M and kb == M:
        txt = "새 지역에서 재현"
    else:
        txt = "어긋난 지역: " + "; ".join(off)
    nb = [f"새 지역에서 재현을 확인하지 못했다({nm.split('|')[0]}, {s['b_verdict']})" for nm, s in st.items() if not s["b_pass"]]
    # 대칭 참조: P4 4지역에 같은 기준
    ref = {}
    for nm in pools["P4"]:
        rr, s = l38_region(tms10.get(nm), nm, a, nlab.get(nm))
        for r in rr:
            r.update(test_id="L38", pool="P4 참조", scope="reference", regime="shallow")
        rows += rr; ref[nm] = s
    ra = sum(s["a_pass"] for s in ref.values()); rb = sum(s["b_pass"] for s in ref.values())
    shrink = any(s["shrink"] for s in st.values())
    sup = ""
    if tibet_support:
        sup = (f"티베트 채점 셀의 원천 지지 밖 비율: 고도 {tibet_support.get('dem_elev', {}).get('frac_outside', np.nan):.3f}, "
               f"√TDD {tibet_support.get('e5_sqrt_tdd', {}).get('frac_outside', np.nan):.3f}, 둘 중 하나 {tibet_support.get('frac_outside_either', np.nan):.3f}"
               f"(원천 0.5–99.5 백분위)")
    rows.append(dict(test_id="L38", pool="새 지역", item="verdict", scope="verdict", role="보조", verdict=txt,
                     stat=f"(a) 충족 {ka}/{M}, (b) 우세 {kb}/{M}. 대칭 참조(P4): (a) {ra}/{len(ref)}, (b) {rb}/{len(ref)}",
                     note="; ".join(nb + ([sup] if sup else []) + (["티베트 결과는 NOVELTY-N7 문장의 근거로 쓰지 않는다"] if any(n.startswith("Tibet") for n in names) else [])),
                     shrink_prior=("수축 사전분포 오지정 의존" if shrink else ""), cross_env=cross_txt))
    return rows, st, shrink


def variant_compare(test_id, base_state, tm_var, nm_var, a, nlab_var, tibet, label, point_only=False, same_as=None, few=np.nan):
    """L41·L42: 변형 설정에서 L38 (a)–(c)를 다시 계산하고 주 설정과 대비별 L28 분류(core 만. (a)10 병기 대비는 aux_l28 열에 따로 적는다).
    few = 적격 표의 사용 분할 최소 채점 블록(5 미만이면 행에 '소수 블록', WRAPUP 7.2 (a)4)."""
    rows = []
    fb = "소수 블록" if np.isfinite(few) and few < FEW_BLOCKS else ""
    if tm_var is None:
        return [dict(test_id=test_id, item="verdict", scope="verdict", role="보조", target=nm_var, variant=label, verdict="행 없음(조각 없음)",
                     few_blocks=fb)]
    rr, s = l38_region(tm_var, nm_var, a, nlab_var, tibet=tibet, flags=fb)
    for r in rr:
        r.update(test_id=test_id, scope="region", variant=label, point_only=bool(point_only), few_blocks=fb,
                 same_as=same_as or "")
    rows += rr
    if point_only:
        rows.append(dict(test_id=test_id, item="verdict", scope="verdict", role="보조", target=nm_var, variant=label,
                         verdict="점 추정만(WRAPUP 7.3 (a)9)", note="4분 판정을 쓰지 않는다"))
        return rows
    shifts = {}
    for k, v in s["core"].items():
        v0 = base_state["core"].get(k, "행 없음")
        shifts[k] = l28_shift(v0, v)
    w = worst_shift(shifts.values())
    aux = {k: l28_shift(base_state.get("aux", {}).get(k, "행 없음"), v) for k, v in s.get("aux", {}).items()}
    rows.append(dict(test_id=test_id, item="verdict", scope="verdict", role="보조", target=nm_var, variant=label, verdict=w,
                     stat="; ".join(f"{k}: {base_state['core'].get(k, '행 없음')} → {v}({shifts[k]})" for k, v in s["core"].items()),
                     aux_l28=(worst_shift(aux.values()) if aux else ""),
                     aux_stat="; ".join(f"{k}: {base_state.get('aux', {}).get(k, '행 없음')} → {v}({aux[k]})" for k, v in s.get("aux", {}).items()),
                     few_blocks=fb, same_as=same_as or "",
                     note=("주 설정과 같은 실행 표(same_as): 주 설정의 TMx 를 그대로 써서 모든 행이 주 설정과 같다" if same_as else
                           "부트스트랩 seed 는 주 설정 저장소 이름(공통 난수, 개정 14)")))
    return rows


def l39(tm_main, tm_temp, a, point_only):
    """L39: 티베트 직접 라벨 대 지온 유도 라벨. 핵심 대비(L28)의 4분 판정과 분류."""
    rows, sh = [], {}
    for lab, ma, mb, n in CORE:
        r0 = row_of(rstats(tm_main, ma, mb, n), a, "Tibet_LGD|x", dict(item=lab, n=n, label_def="직접(qtp_du_gpr)"))
        r1 = row_of(rstats(tm_temp, ma, mb, n), a, "Tibet_LGD~L39|x", dict(item=lab, n=n, label_def="지온 유도(qtp_fu_temp)"))
        for r in (r0, r1):
            r.update(test_id="L39", scope="region", flags=";".join(v for v in (r.get("ci_dependence", ""), UNBLIND_STAT,
                                                                               UNBLIND_PRED if mb == "P0" else "") if v))
        rows += [r0, r1]
        sh[lab] = l28_shift(r0["verdict4"], r1["verdict4"])
    if point_only:
        txt = "점 추정만(지온 유도 셀이 적격 조건을 채우지 못함)"
    else:
        txt = worst_shift(sh.values())
    rows.append(dict(test_id="L39", item="verdict", scope="verdict", role="보조", verdict=txt, stat="; ".join(f"{k}: {v}" for k, v in sh.items()),
                     flags=UNBLIND_STAT, note="두 라벨 집합은 위치가 달라 위치 효과가 섞인다"))
    return rows


# ================================================================ L40(확충판)과 (a)5 분할 변동 범위
def splitdist_table(a):
    if not a.SPLITDIST.exists():
        return None
    sd = pd.read_csv(a.SPLITDIST)
    if "primary_source" in sd:
        sd = sd[sd.primary_source.astype(bool)]
    return sd


def split_range_flag(sd, ref_name, ma, mb, n, value, n_draws, subst_note):
    """WRAPUP 7.2 (a)5: LGX-N4 의 분할별 Δ(delta_splits)에서 분할 5개를 무작위로 n_draws 회 뽑아 평균한 분포의 10–90 백분위 안이면
    '분할 변동 범위'. LGX-N4 에 없는 대비(D0 − P0 의 n > 0)는 D0 − P0(n = 0) 분포로 대신한다. seed = seed_of("lgw-l40", 대상, 대비)."""
    lab = f"{ma}-{mb}|n{'all' if n == N_ALL else n}"
    if sd is None:
        return dict(split_range="분포 없음(lgx_splitdist.csv 없음)", split_p10=np.nan, split_p90=np.nan, split_source="")
    key = N4_LABEL.get((ma, mb, n))
    sub_note = ""
    if key is None and ma == "D0" and mb == "P0":
        key, sub_note = "D0-P0|n0", subst_note
    q = sd[(sd.target == ref_name) & (sd.contrast == key)] if key else sd.iloc[:0]
    if not len(q):
        return dict(split_range="분포 없음", split_p10=np.nan, split_p90=np.nan, split_source=str(key or ""))
    v = np.array(list(json.loads(q.iloc[0].delta_splits).values()), float)
    if len(v) < 5 or not np.isfinite(value):
        return dict(split_range="분포 없음(분할 < 5)", split_p10=np.nan, split_p90=np.nan, split_source=key)
    rng = np.random.RandomState(seed_of("lgw-l40", ref_name.split("|")[0], lab))
    m5 = np.array([v[rng.choice(len(v), 5, replace=False)].mean() for _ in range(int(n_draws))])
    p10, p90 = float(np.percentile(m5, 10)), float(np.percentile(m5, 90))
    inside = bool(p10 <= value <= p90)
    return dict(split_range=("분할 변동 범위" if inside else "") + (f"; {sub_note}" if sub_note else ""), split_p10=p10, split_p90=p90,
                split_source=f"{key}({q.iloc[0].get('source', '')}, 분할 {len(v)})")


def l40(tms10, exp_tms, ref_tms, a, nlab, sd, ref_label, cross_label):
    """L40: 확충판(로컬)과 v3 기준값(LG 본 실행 또는 로컬 v3)의 지역 단위 판정 비교. 분류는 L28."""
    rows = []
    for spec, ref_name in L40_SPECS.items():
        ex = exp_tms.get(spec)
        rf = ref_tms.get(ref_name)
        nm_ex = R.tm_name(dict(target=ref_name.split("|")[0], suffix="expnokyt" if spec.endswith("noKytalyk") else "exp"))
        if ex is None or rf is None:
            rows.append(dict(test_id="L40", item="verdict", scope="verdict", role="보조", target=nm_ex, spec=spec, flags=UNBLIND_REPRO,
                             verdict="행 없음(" + ("확충판 조각 없음" if ex is None else "기준값 조각 없음") + ")", reference=ref_label))
            continue
        cmp, sh, l1c, d0 = [], {}, {}, {}
        # L1 의 지역 조건: n ≤ 40 에서 D0 − P0 가 우세인 n 이 있는가(전량 행은 그 쪽의 실제 라벨 수 ≤ 40 일 때만)
        for which, tm, nm in (("확충판", ex, nm_ex), ("기준", rf, ref_name)):
            vs = []
            for n in l1_ns_of(tm, nlab.get(nm)):
                r = row_of(rstats(tm, "D0", "P0", n), a, nm, dict(item="D0-P0(L1 조건)", n=n, side=which))
                if which == "확충판":
                    r.update(split_range_flag(sd, ref_name, "D0", "P0", n, r.get("delta", np.nan), a.l40_draws, "D0 − P0(n = 0) 분포로 대신"))
                d0[(which, n)] = r
                vs.append(r["verdict4"]); cmp.append(r)
            l1c[which] = None if (not vs or any(v in NA for v in vs)) else any(v == "우세" for v in vs)
        for n in sorted({n for w, n in d0 if w == "확충판"} & {n for w, n in d0 if w == "기준"}):
            lab = f"D0-P0|n{'all' if n == N_ALL else n}"
            sh[lab] = l28_shift(d0[("기준", n)]["verdict4"], d0[("확충판", n)]["verdict4"])
            d0[("확충판", n)]["l28"] = sh[lab]
        for ma, mb, n in (("R1", "P0", N_ALL), ("P1", "P0", 10)):
            r_e = row_of(rstats(ex, ma, mb, n), a, nm_ex, dict(item=f"{ma}-{mb}", n=n, side="확충판"))
            r_r = row_of(rstats(rf, ma, mb, n), a, ref_name, dict(item=f"{ma}-{mb}", n=n, side="기준"))
            r_e.update(split_range_flag(sd, ref_name, ma, mb, n, r_e.get("delta", np.nan), a.l40_draws, ""))
            lab = f"{ma}-{mb}|n{'all' if n == N_ALL else n}"
            sh[lab] = l28_shift(r_r["verdict4"], r_e["verdict4"])
            r_e["l28"] = sh[lab]
            cmp += [r_r, r_e]
        for r in cmp:
            r.update(test_id="L40", scope="region", spec=spec, reference=ref_label, cross_env=cross_label)
            add_flag(r, r.get("ci_dependence", ""), UNBLIND_REPRO)
        rows += cmp
        l1s = "판정 불가" if l1c["확충판"] is None or l1c["기준"] is None else ("같음" if l1c["확충판"] == l1c["기준"] else "다름")
        rows.append(dict(test_id="L40", item="verdict", scope="verdict", role="보조", target=nm_ex, spec=spec, flags=UNBLIND_REPRO,
                         verdict=worst_shift(sh.values()), stat=f"L1 지역 조건 {l1s}(기준 {l1c['기준']}, 확충판 {l1c['확충판']}); "
                         + "; ".join(f"{k}: {v}" for k, v in sh.items()),
                         reference=ref_label, cross_env=cross_label,
                         note="분할 변동 범위 표지가 붙은 대비는 새 셀 효과로 쓰지 않는다(WRAPUP 7.2 (a)5)"
                         + ("; Kytalyk 제외 변형(병기)" if spec.endswith("noKytalyk") else "")))
    return rows


# ================================================================ 주 흐름
def main(argv=None):
    a = parse_args(argv)
    t0 = time.time()
    try:
        cur_n = os.nice(0)
        if cur_n < 10:
            os.nice(10 - cur_n)
    except OSError:
        pass
    rs, rp, rm = repro_status(a)
    if rp is False:
        raise SystemExit("[중단] 재현 점검(WRAPUP 7.5 (c)2)의 물리식 차이가 허용 차를 넘었다. 원인을 고칠 때까지 LGD 집계를 멈춘다")
    if rp is None and not a.allow_no_repro_gate:
        raise SystemExit(f"[거부] 재현 점검 기록이 없다({a.REPRO}). h51 --repro-check 를 먼저 한다(LG 본 실행 회수 뒤). "
                         "시험에서는 --allow-no-repro-gate")
    xs, xd = xenv_status(a)
    el = pd.read_csv(a.ELIG)
    pools = pools_from_eligibility(el)
    man_path = a.LGD / "lgd_run_manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {"specs": {}}
    specs_all = R.GROUPS
    spec_rows = man.get("specs", {})

    # 점 추정 전용 저장소(부적격 새 지역, (h) 변형, 적격 표에서 부적격인 변형)의 이름. 적격 표에서 정하고 manifest 의 기록을 더한다
    el_by = el.set_index("spec")
    po_names = []
    for sp_, r_ in el_by.iterrows():
        k_ = str(r_.get("kind", ""))
        if k_.startswith("variant_L41") and (k_ == "variant_L41h" or not bool(r_.get("eligible", False))):
            po_names.append(R.tm_name(dict(target=str(r_["target"]), suffix=k_.replace("variant_", ""))).split("|")[0])
    for sp_, info_ in spec_rows.items():
        if info_.get("point_only") and info_.get("tm_name"):
            po_names.append(info_["tm_name"].split("|")[0])
    mp_new = set_main_point(pools["ineligible"], sorted(set(po_names)))

    # ---------------- 조각 읽기
    loaded, plat = {}, {}
    for nm in P4:
        L = load_group(a.LG / "shards", a.lg_tag, nm.split("|")[0], nm, a.lg_platform)
        if L is not None:
            loaded[nm] = L; plat[nm] = a.lg_platform

    def lgd_group(spec):
        info_ = spec_rows.get(spec, {})
        src_spec = info_.get("same_as") or spec
        target = info_.get("target") or (R.NEW_MAIN.get(spec) if spec in R.NEW_MAIN else None)
        if target is None:
            return None, None
        name = info_.get("tm_name") or R.tm_name(dict(target=target, suffix=""))
        L = load_group(a.LGD / src_spec / "shards", R.TAG, target, name, "local")
        if L is not None:
            L["same_as"] = info_.get("same_as")
        return name, L

    lgd_loaded = {}
    for g in ("main", "l40", "l39", "l41", "l42", "v3local"):
        for spec in specs_all[g]:
            nm, L = lgd_group(spec)
            if L is not None:
                lgd_loaded[spec] = (nm, L)
    for spec in specs_all["main"]:
        if spec in lgd_loaded:
            nm, L = lgd_loaded[spec]
            loaded[nm] = L; plat[nm] = "local"

    # ---------------- TMx(MAIN_POINT 덮어쓴 뒤)
    def tms_for(nb):
        return {nm: make_tm(nm, L, nb) for nm, L in loaded.items()}
    tms1 = {nm: method_view(tm) for nm, tm in tms_for(a.nboot_h40).items()}
    tms10 = {nm: method_view(tm) for nm, tm in tms_for(a.nboot).items()}
    spec_tm1, spec_tm10, crn = {}, {}, {}
    for spec, (nm, L) in lgd_loaded.items():
        spec_tm1[spec] = method_view(make_tm(nm, L, a.nboot_h40))
        base_ = variant_base(spec, spec_rows)
        main_nm = f"{R.NEW_MAIN[base_]}|x" if base_ in R.NEW_MAIN else None
        if main_nm and L.get("same_as") and main_nm in tms10:
            spec_tm10[spec] = tms10[main_nm]                          # same_as: 주 설정의 TMx 를 그대로 쓴다(개정 14)
            crn[spec] = f"same_as → {main_nm} 의 TMx"
            continue
        tm_ = make_tm(nm, L, a.nboot)
        if main_nm and tm_.has_ci:
            crn_seed(tm_, main_nm); crn[spec] = f"seed ← {main_nm}"
        spec_tm10[spec] = method_view(tm_)

    # ---------------- 곡선(1,000회, h40.build_curve)과 최소 n
    curs = []
    for nm, tm in tms1.items():
        c_ = H.build_curve({nm: tm}, loaded[nm]["runs"]) if len(loaded[nm]["runs"]) else pd.DataFrame()
        if len(c_):
            c_["pool_member"] = nm; c_["platform"] = plat[nm]; c_["spec"] = "LG" if plat[nm] != "local" else next(
                (s for s, (n2, _) in lgd_loaded.items() if n2 == nm), "")
            curs.append(c_)
    for spec, tm in spec_tm1.items():
        nm, L = lgd_loaded[spec]
        if nm in tms1:
            continue
        c_ = H.build_curve({nm: tm}, L["runs"]) if len(L["runs"]) else pd.DataFrame()
        if len(c_):
            c_["pool_member"] = ""; c_["platform"] = "local"; c_["spec"] = spec
            curs.append(c_)
    cur = pd.concat(curs, ignore_index=True) if curs else pd.DataFrame(columns=["target", "mode", "n", "method"] + H.GRP_COLS)
    mn = H.build_minn(cur) if len(cur) else pd.DataFrame()
    nlab = {}
    if len(cur):
        for r in cur[(cur.method == "R1") & (cur.n == -1) & (cur.learner == BASE) & (cur.lam == H.LAM_BASE)].itertuples():
            nlab[f"{r.target}|x"] = float(r.n_lab)

    # ---------------- 풀 판정
    cross_new = len({plat.get(nm, "") for nm in pools["PE2"] if nm in plat}) > 1
    cross_txt = f"교차 환경; {xs}" if cross_new else ""
    tests, bins = [], {}
    for hyp in ("L1e", "L4e", "L8e"):
        bins[hyp] = {}
    for pname in ("P4", "PE1", "PE2"):
        names = pools[pname]
        r1, b1 = l1e(cur, names, pname, tms=tms1); tests += r1; bins["L1e"][pname] = b1
        r4, b4, _ = l4e(tms1, cur, names, pname, rel=(pname == "PE2"), aux10=(pname == "PE2" and any(nm.startswith("Tibet") for nm in names)))
        tests += r4; bins["L4e"][pname] = b4
        r8, b8 = l8e(tms1, cur, names, pname, rel=(pname == "PE2")); tests += r8; bins["L8e"][pname] = b8
    for r in tests:
        if r.get("pool") in ("PE1", "PE2"):
            r["cross_env"] = cross_txt
            if rs != "재현 점검 통과":
                r["repro"] = rs
    # (a)10 티베트 P2 − P1: PE2 의 R1 − P1(L4e) 표지
    l38_rows, l38_state, shrink = l38(tms10, pools, a, nlab, cross_txt, (spec_rows.get("Tibet", {}) or {}).get("tibet_support"))
    if shrink:
        for r in tests:
            if r.get("test_id") == "L4e" and r.get("pool") == "PE2" and r.get("item") in ("R1-P1", "verdict"):
                r["flags"] = ";".join(v for v in (str(r.get("flags", "") or ""), "수축 사전분포 오지정 의존") if v)
        for r in l38_rows:
            if r.get("item") == "(c) R1-P1" and str(r.get("target", "")).startswith("Tibet"):
                r["flags"] = ";".join(v for v in (str(r.get("flags", "") or ""), "수축 사전분포 오지정 의존") if v)
    for r in tests:                                              # 티베트 지역 행: 예측된 결과(비맹검)((a)10), 비맹검(라벨 통계 열람)(6B.7)
        if str(r.get("target", "")).startswith("Tibet") and r.get("scope") == "region" and r.get("test_id") in ("L1e", "L8e"):
            add_flag(r, UNBLIND_PRED, UNBLIND_STAT)
        if r.get("test_id") == "L8e" and r.get("pool") == "PE2" and r.get("scope") in ("MEAN", "MEAN_rel", "verdict"):
            add_flag(r, UNBLIND_STAT + ": 티베트 포함")
    for hyp in ("L1e", "L4e", "L8e"):
        tests += compare_rows(hyp, bins[hyp], pools, tms10, a, cross_txt)
    tests += l38_rows

    # ---------------- L39
    main_tib = tms10.get("Tibet_LGD|x")
    t39 = spec_tm10.get("Tibet_L39_temp")
    po39 = bool(spec_rows.get("Tibet_L39_temp", {}).get("point_only", False))
    tests += l39(main_tib, t39, a, po39)

    # ---------------- L40
    repro_ml_fail = rm is False
    sd = splitdist_table(a)
    exp_tms = {spec: spec_tm10[spec] for spec in L40_SPECS if spec in spec_tm10}
    if repro_ml_fail:
        ref_tms = {f"{t}|x": spec_tm10.get(f"v3local_{t}") for t in R.V3LOCAL}
        ref_tms = {k: v for k, v in ref_tms.items() if v is not None}
        ref_label, cross_l40 = "로컬 v3(v3local, (c)2 CatBoost 불통과)", ""
        if not ref_tms:
            ref_label = "판정 보류(로컬 v3 기준값 없음, (c)2 CatBoost 불통과)"
    else:
        ref_tms = {nm: tms10.get(nm) for nm in ("Russia_W|x", "Russia_E|x", "Canada|x") if tms10.get(nm) is not None}
        ref_label = "LG 본 실행(Rescale)"
        cross_l40 = "교차 환경(보조)" + (f"; {xs}" if xs else "")
    tests += l40(tms10, exp_tms, ref_tms, a, nlab, sd, ref_label, cross_l40)

    # ---------------- L41, L42
    def el_few(spec_):
        return float(el_by.loc[spec_, "min_nb_eval_used"]) if spec_ in el_by.index and pd.notna(el_by.loc[spec_, "min_nb_eval_used"]) else np.nan

    for sp_main, tgt in R.NEW_MAIN.items():
        nm_main = f"{tgt}|x"
        base_state = l38_state.get(nm_main)
        if base_state is None:
            continue
        for v in R.L41_VARIANTS:
            spec = f"{sp_main}_L41{v}"
            info_ = spec_rows.get(spec, {})
            nm_v = info_.get("tm_name") or R.tm_name(dict(target=tgt, suffix=f"L41{v}"))
            inel = bool(spec in el_by.index and not bool(el_by.loc[spec, "eligible"]))       # 적격 표에서 직접 읽는다(개정 14)
            if inel or not info_ or spec not in spec_tm10:
                tests.append(dict(test_id="L41", item="verdict", scope="verdict", role="보조", target=nm_v, variant=f"({v})",
                                  verdict="변형 불가(적격 표에서 부적격)" if inel else "행 없음(실행 기록 없음)"))
                continue
            tests += variant_compare("L41", base_state, spec_tm10.get(spec), nm_v, a, nlab.get(nm_v), tgt.startswith("Tibet"), f"({v})",
                                     point_only=bool(info_.get("point_only")) or v == "h", same_as=info_.get("same_as"), few=el_few(spec))
        spec = f"{sp_main}_L42"
        nm_v = (spec_rows.get(spec, {}) or {}).get("tm_name") or R.tm_name(dict(target=tgt, suffix="L42"))
        tests += variant_compare("L42", base_state, spec_tm10.get(spec), nm_v, a, nlab.get(nm_v), tgt.startswith("Tibet"), "원천 + 다른 새 지역",
                                 few=el_few(sp_main))

    # ---------------- 지역 수준 추론, 풀 표
    ri = region_inference_rows(tms10, pools)
    pool_rows_ = []
    for pname in ("P4", "PE1", "PE2", "NEW1", "NEW2"):
        names = pools[pname]
        pool_rows_.append(dict(pool=pname, n_regions=len(names), regions=",".join(names), loaded=",".join(nm for nm in names if nm in loaded),
                               regimes=",".join(f"{nm.split('|')[0]}:{pools['regime'].get(nm, '')}" for nm in names),
                               platforms=",".join(sorted({plat.get(nm, "없음") for nm in names})),
                               cross_env=(f"교차 환경; {xs}" if len({plat.get(nm, '') for nm in names if nm in plat}) > 1 else ""),
                               few_blocks=",".join(nm.split("|")[0] for nm in names if pools["min_nb_eval"].get(nm, 99) < FEW_BLOCKS)))
    pool_df = pd.DataFrame(pool_rows_)

    # ---------------- 쓰기
    O = a.OUT; O.mkdir(parents=True, exist_ok=True)
    td = pd.DataFrame(tests)
    front = ["test_id", "pool", "item", "scope", "role", "target", "spec", "variant", "n", "delta", "ci_lo", "ci_hi", "delta_blockeq", "ci_lo_beq",
             "ci_hi_beq", "sig", "verdict4", "verdict4_common", "ci_dependence", "direction", "l28", "flags", "cross_env", "verdict", "stat"]
    td = td[[c for c in front if c in td] + [c for c in td.columns if c not in front]] if len(td) else td
    cur.to_csv(O / "lgd_curve.csv", index=False)
    mn.to_csv(O / "lgd_minn.csv", index=False)
    td.to_csv(O / "lgd_tests.csv", index=False)
    ri.to_csv(O / "lgd_region_inference.csv", index=False)
    pool_df.to_csv(O / "lgd_pool.csv", index=False)
    meta = dict(stage="LGD 확장 풀 집계(h52)", plan="docs/EXPERIMENT_PLAN_LG_2026-09-29.md 6B.5·6B.6(개정 14), WRAPUP 7.2–7.5(커밋 316714c)",
                script="scripts/2_evaluation/h52_lgd_pool.py", script_sha256=sha256(Path(__file__)), h40_code_sha=H.code_sha(),
                h42_code_sha=X.code_sha_x() if hasattr(X, "code_sha_x") else "", created=time.strftime("%Y-%m-%d %H:%M"),
                args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items() if k.islower()},
                inputs=dict(eligibility=sha256(a.ELIG), manifest=sha256(man_path), splitdist=sha256(a.SPLITDIST), xenv=sha256(a.XENV),
                            repro=sha256(a.REPRO)),
                main_point=dict(original=MAIN_POINT_ORIG, overridden=mp_new, rule="WRAPUP 7.4 (b)1: 적격 표의 부적격 새 지역 + 점 추정 전용 변형 저장소"),
                pools={k: pools[k] for k in ("P4", "PE1", "PE2", "NEW1", "NEW2", "ineligible")}, regime=pools["regime"],
                platform=plat, cross_env=cross_txt, xenv_i=xs, xenv_summary={k: v for k, v in (xd or {}).items() if k not in ("over_tol_keys", "by_method")},
                repro_gate=rs, repro_gate_all_scope=repro_all_scope(a), crn=crn, argv=a.ARGV,
                l40_reference=ref_label, l40_cross_env=cross_l40,
                loaded=dict(lg={nm: dict(dir=L["dir"], code_sha=L["code_sha"], cfg_common=L["cfg_common"], status=L["status"],
                                         splits=sorted(L["by_split"])) for nm, L in loaded.items() if plat.get(nm) != "local"},
                            lgd={spec: dict(name=nm, dir=L["dir"], same_as=L.get("same_as"), code_sha=L["code_sha"], cfg_common=L["cfg_common"],
                                            status=L["status"], splits=sorted(L["by_split"])) for spec, (nm, L) in lgd_loaded.items()}),
                nboot=dict(L1e_L4e_L8e=a.nboot_h40, four_way=a.nboot, l40_split_draws=a.l40_draws), delta_eq=[a.delta_eq, a.delta_eq_aux],
                rules=dict(L1e="유의 개선 지역 수 > ⌊N/4⌋ 이면 기각. 전량 행은 실제 라벨 수 ≤ 40 인 지역만",
                           compare="(a)1: L1e 지지·기각, L4e n ≤ 10 성립, L8e 지지·기각. 다르면 풀 4분 판정의 L28 분류와 반전·희석",
                           relative="PE2 상대 단위 = 지역 Δ / 지역 P0 셀 가중 RMSE 점 추정(분포도 같은 값으로 나눔)",
                           l38_direction="(a) 가설 방향 D0 − P0 ≥ 0, (b) R1 − P0 < 0. 보조 열이며 판정에 쓰지 않는다",
                           same_as="실행 표가 주 설정과 같은 변형은 주 설정 조각과 주 설정의 TMx 를 쓴다(모든 행이 주 설정과 같다)",
                           crn="L41·L42 변형의 부트스트랩 seed 는 주 설정 저장소 이름이다(h40 TMx.seed 와 h42 boot_delta_common 의 tm.name)",
                           common_ci="(a)4: L1e 지역, L4e·L8e 지역과 MEAN, L4e-PE2 병기, (a)1 풀 대비, L38–L42 에 공통 재표집 보조 CI",
                           platform="모든 대비는 한 지역의 한 플랫폼 조각 안에서 닫힌다(WRAPUP 8.6 (1))"),
                n_rows=dict(curve=int(len(cur)), minn=int(len(mn)), tests=int(len(td)), region_inference=int(len(ri))),
                elapsed_s=round(time.time() - t0, 1))
    (O / "lgd_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    print(f"[h52] 곡선 {len(cur):,} · 판정 표 {len(td):,} · 지역 추론 {len(ri)} · 풀 P4 {len(pools['P4'])}, PE1 {len(pools['PE1'])}, "
          f"PE2 {len(pools['PE2'])} · {rs} · {xs} · {time.time() - t0:.0f}s → {O}/lgd_*", flush=True)
    return dict(curve=cur, minn=mn, tests=td, region_inference=ri, pool=pool_df, meta=meta)


if __name__ == "__main__":
    main()
