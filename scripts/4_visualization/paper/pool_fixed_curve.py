"""Fig 2e·2f 와 Fig 3d 의 구성 고정 풀 평균 곡선(서술 산출, 판정에 쓰지 않는다).

근거
  figures/figure_spec.json 의 derived_modules.pool_fixed_curve, conditional_rules.fig2_pool_editions,
  docs/DISPLAY_ITEMS_2026-09-30.md 6절 1번(커밋 903dce9). LG 결과를 열람하기 전에 작성했다.

판(구성 고정)
  E1_P4_n_le_10        주 4지역 P4(Lena, Canada, Russia_W, Russia_E, 모드 x), n ∈ {0, 3, 10}
  E2_LenaCanada_all_n  레나·캐나다(모드 x). 두 지역 모두의 사용 분할 저장소에 키가 있는 격자 n 전부와 전량(n = -1).
                       한 지역에만 있는 n(예: 1,000)은 판에서 빼고 메타의 excluded_n 에 적는다.
  한 판 안의 모든 n 에서 풀 지역 집합은 같다. 판의 지역 가운데 하나라도 대비가 없거나(키 없음) CI 풀 조건(h40.strat_mean 과 같은 조건:
  유효 분할 1개 이상, 채점 블록 합집합 8개 이상)을 채우지 못하면 그 n 의 층화 평균 행은 '판정 불가(구성 불완전)'이고 평균 값 열은
  비운다(NaN). 남은 지역만으로 평균을 내지 않는다. 지역 행은 그대로 남긴다.

대비
  방법 {P1, P2, P3, V1, R0, R1, R2, D0} × n. 기준선은 P0(Δ = RMSE(방법) − RMSE(P0))와 P1(같은 n 의 P1).
  방법 키와 P1 키는 h40._g 로 만든다(학습기 catboost_lo 또는 none, α 1, 셀 무작위, 방법별 기준 λ: P·V1 0.0, D0 1.0, 잔차 방법 0.25).
  n = 0 에서 P1 키는 E0·s 이므로 기준선 P1 의 값은 기준선 P0 의 값과 같다(h40 규약). P1 대 P1 은 계산하지 않는다.

계산(새 통계 코드 없음, 재사용 함수)
  지역 대비   h42.region_stats: h40.contrast → h4_common.boot_delta_blocks(분할 안 채점 블록 재표집, 방법·추출·seed 공통 인덱스,
              셀 가중·블록 등가중 두 분포)와 h42.boot_delta_common(지역 블록 공통 재표집, 보조 CI)
  층화 평균   h42.pool_rows(m1_stats.strat, h42.stats_row, h42.verdict4, h42.eq_p). 주 CI 는 h40.strat_mean 과 같은 값이다
              (tests/test_pool_fixed_curve.py 가 대조한다)
  4분 판정    h42.verdict4, 동등 한계 δ 0.5 cm(verdict4), 보조 δ 1.0 cm(verdict4_d10). 서술 열이다. 그림의 판정 표지는
              lg_tests·lgx_lg_aux·lgw_bundle 에서만 가져온다(스펙 status_label)
  재표집      10,000회(h42 lg_aux_table 과 같다). seed 는 하네스 규칙 그대로다. 지역 저장소 이름 nm 에 대해
              h40.TMx.seed = seed_of('lgboot', nm), 분할 s 의 재표집 seed = seed_of(TMx.seed, s)(h4_common.boot_delta_blocks),
              공통 재표집 seed = seed_of('lgxboot', nm)(h42.boot_delta_common). 그래서 같은 대비의 lgx_lg_aux 행(L4 R1 − P1 등)과 값이 같다
  분할 구조   h40.Data.split_structure(자료 적재) 대신 조각 unit.json 에 실행 때 기록된 dup_of·valid·n_eval·n_A 를 쓴다(ShardSplitInfo,
              얇은 대체. h52 의 load_group 과 같은 방식). h42.make_tm 이 그 값으로 TMx 를 만든다. 기대 분할 수는 unit.json 의
              n_valid_splits(유효 분할이 없으면 n_unique_splits)이고, 조각이 모자라거나 그 n 의 키가 일부 분할에만 있으면
              splits_short 열과 status 에 '부분(분할 k/K)'을 적는다(값은 남긴다)
  조각        기본은 cpu 부분 조각만 읽는다(방법 축 키는 모두 cpu 조각에 있다. h52 와 같다). 조각 사이 설정 해시는 h40.check_shard_cfg 로 대조한다

입력과 산출
  --lg-shards   $LGD/shards(LG 본 실행 조각, 읽기 전용). $LGD 는 docs/EXECUTION_PLAN_REMAINING_2026-09-30.md 418행의 조각 폴더
  --lgs         $LGS(LG 표 폴더, 선택). 주면 <tag>_meta.json 의 sha256 만 메타에 적는다(출처 대조용)
  --out         기본 data/processed/paper. pool_fixed_curve.csv 와 pool_fixed_curve_meta.json(입력 조각 sha256, 코드 해시, nboot, seed 규칙)
  --synthetic   합성 조각(make_synthetic_shards)을 <out>/_synthetic_shards 에 만들고 계산한다. 산출 이름에 _synthetic 을 붙인다
  화면에는 파일 이름과 행 수만 낸다(결과 값을 출력하지 않는다).

실행(ROOT, CPU 전용, nice 10, 스레드 1–2. 결과 열람 규칙에 따라 C1·J1 뒤에만 실제 조각으로 돌린다)
  CUDA_VISIBLE_DEVICES= nice -n 10 python3 scripts/4_visualization/paper/pool_fixed_curve.py --lg-shards "$LGD/shards" --lgs "$LGS" --threads 2
  시험용 합성: CUDA_VISIBLE_DEVICES= nice -n 10 python3 scripts/4_visualization/paper/pool_fixed_curve.py --synthetic --out /tmp/x --nboot 200
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

THREADS_MAX = 2
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _peek_threads(argv=None, default=1):
    """numpy 를 부르기 전에 --threads 를 읽어 1–2 로 자른다."""
    av = list(sys.argv if argv is None else argv)
    val = default
    for i, v in enumerate(av):
        if v == "--threads" and i + 1 < len(av):
            val = av[i + 1]
        elif v.startswith("--threads="):
            val = v.split("=", 1)[1]
    try:
        return max(1, min(int(val), THREADS_MAX))
    except (TypeError, ValueError):
        return int(default)


if __name__ == "__main__":                                       # 스크립트 실행: 스레드 상한과 CPU 전용을 강제한다(대입)
    for _v in THREAD_VARS:
        os.environ[_v] = str(_peek_threads())
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:                                                            # import(시험): 호출한 쪽의 설정을 따른다
    for _v in THREAD_VARS:
        os.environ.setdefault(_v, "1")

import numpy as np                                                                                   # noqa: E402
import pandas as pd                                                                                  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DL = ROOT / "scripts" / "3_deep_learning"
for _p in (str(DL), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load(name, path):
    """하네스를 파일 경로에서 읽어 sys.modules 에 등록한다(h42._load_h40, h52._load 와 같은 방식). 이미 있으면 그 모듈을 쓴다."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


X = _load("h42_label_grid_ext", DL / "h42_label_grid_ext.py")      # h42 는 h40 을 'h40_label_grid' 로 등록해 X.H 로 둔다
H = X.H
import polar.h4_common as H4                                                                          # noqa: E402
from polar.h4_common import BlockStore, save_stores, load_stores, seed_of                           # noqa: E402

# ================================================================ 고정 설계값(스펙 derived_modules.pool_fixed_curve)
NBOOT = 10000
DELTA_EQ = 0.5                                                   # h42 --delta-eq 기본값
DELTA_EQ_AUX = 1.0                                               # h42 --delta-eq-aux 기본값
N_ALL = -1
P4 = [f"{t}|x" for t in H.MAIN4]
EDITIONS = {
    "E1_P4_n_le_10": dict(regions=list(P4), n_rule="fixed", n_list=[0, 3, 10]),
    "E2_LenaCanada_all_n": dict(regions=["Lena|x", "Canada|x"], n_rule="common", n_list=None),
}
METHODS = ["P1", "P2", "P3", "V1", "R0", "R1", "R2", "D0"]
BASELINES = ["P0", "P1"]
STATUS_LABEL = "서술(판정에 쓰지 않는다)"
BLANK_ON_INCOMPLETE = ("delta", "ci_lo", "ci_hi", "delta_blockeq", "ci_lo_beq", "ci_hi_beq", "p_boot", "p_eq", "rmse_A", "rmse_B",
                       "ci_lo_c", "ci_hi_c", "ci_lo_beq_c", "ci_hi_beq_c", "delta_allregions")
NA_VERDICT = "판정 불가"
FRONT = ["edition", "edition_regions", "baseline", "method", "learner", "lam", "n", "n_label", "contrast", "scope", "target", "status",
         "composition_complete", "delta", "ci_lo", "ci_hi", "delta_blockeq", "ci_lo_beq", "ci_hi_beq", "p_boot", "p_eq", "verdict4",
         "verdict4_d10", "verdict4_common", "ci_lo_c", "ci_hi_c", "ci_lo_beq_c", "ci_hi_beq_c", "rmse_A", "rmse_B", "n_ci_regions",
         "n_regions_target", "pool_regions", "missing_regions", "no_ci_regions", "splits_min", "splits_expected", "splits_short", "n_splits",
         "n_splits_expected", "n_blocks_split_min", "nboot", "delta_eq", "delta_eq_aux", "role"]


def n_label(n):
    return "all" if int(n) == N_ALL else str(int(n))


def n_order(n):
    """정렬 키. 전량(-1)은 가장 뒤."""
    return (1, 0) if int(n) == N_ALL else (0, int(n))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ================================================================ 분할 구조(h40.Data.split_structure 의 얇은 대체)
class ShardSplitInfo:
    """조각 unit.json 에 실행 때 기록된 분할 구조를 h40.Data.split_structure 형식으로 돌려준다. h42.make_tm 의 D 인자로 쓴다.
    unit.json 의 값은 실행 때 h40.Data.split_structure 가 준 값(h40.build_ctx 의 meta)이다. 자료(load_base)를 읽지 않는다.
    같은 (대상, 분할)의 조각(모드 i·x, cpu·gpu)이 서로 다른 값을 적었으면 ValueError."""

    KEYS = ("dup_of", "mirror_of", "valid", "n_eval", "n_A", "n_valid_splits", "n_unique_splits")

    def __init__(self, units):
        self.info: dict = {}
        for u in units:
            t, sp = str(u["target"]), int(u["split"])
            d = {k: u[k] for k in self.KEYS if k in u}
            d.setdefault("dup_of", -1); d.setdefault("valid", True)
            old = self.info.setdefault(t, {}).get(sp)
            if old is not None and any(old.get(k) != d.get(k) for k in ("dup_of", "valid", "n_eval", "n_A")):
                raise ValueError(f"분할 구조 불일치: {t} 분할 {sp}")
            self.info[t][sp] = d

    def split_structure(self, t):
        return {sp: dict(v) for sp, v in self.info.get(str(t), {}).items()}

    def expected_count(self, t):
        """실행하기로 한 분할 수(unit.json 의 n_valid_splits, 유효 분할이 없으면 n_unique_splits). 기록이 없으면 None."""
        vals = self.info.get(str(t), {}).values()
        nv = [int(v["n_valid_splits"]) for v in vals if "n_valid_splits" in v]
        nu = [int(v["n_unique_splits"]) for v in vals if "n_unique_splits" in v]
        if nv and max(nv) > 0:
            return max(nv)
        return max(nu) if nu else None


def build_tms(stores, info: ShardSplitInfo, names, nboot):
    """{저장소 이름: TMx}. h42.make_tm 으로 만들고, 조각이 모자라면 기대 분할 목록을 기대 수까지 음수 자리표로 채운다
    (h42.n_expected 가 기대 분할 수를 돌려주게 한다). 반환 (tms, missing_shards{이름: (있는 분할 수, 기대 분할 수)})."""
    tms, missing = {}, {}
    for nm in names:
        by = {sp: st for (n_, sp), st in stores.items() if n_ == nm}
        if not by:
            continue
        tm = X.make_tm(nm, by, info, nboot)
        K = info.expected_count(nm.split("|")[0].split("~")[0])
        have = list(getattr(tm, "expected_splits", sorted(tm.used)))
        if K is not None and len(have) < K:
            tm.expected_splits = have + [-(i + 1) for i in range(K - len(have))]
            missing[nm] = (len(have), int(K))
        tms[nm] = tm
    return tms, missing


# ================================================================ 조각 읽기
def find_shards(shards_dir, tag, parts=("cpu",), names=None):
    """h42.find_shards_x(h40 조각 이름 규칙) 가운데 부분과 대상·모드를 고른다."""
    out = []
    for s in X.find_shards_x(shards_dir, tag):
        if s["part"] not in parts:
            continue
        if names is not None and f"{s['target']}|{s['mode']}" not in names:
            continue
        out.append(s)
    return out


def load_inputs(shards_dir, tag, names, nboot, parts=("cpu",), allow_mixed_cfg=False):
    sh = find_shards(shards_dir, tag, parts, set(names))
    units = [json.loads(Path(s["unit"]).read_text()) for s in sh]
    cfg_info = H.check_shard_cfg(SimpleNamespace(allow_mixed_cfg=allow_mixed_cfg), units) if units else {}
    info = ShardSplitInfo(units)
    stores = load_stores([s["npz"] for s in sh]) if sh else {}
    tms, missing = build_tms(stores, info, names, nboot)
    return dict(shards=sh, units=units, cfg_info=cfg_info, tms=tms, missing_shards=missing)


# ================================================================ 판의 n 격자
def n_presence(tm):
    """{n: 그 n 의 P1 키(방법 축 해석식, 셀 무작위)가 있는 사용 분할 수}. P1 은 방법 축의 모든 (n, 추출) 칸에 저장된다(h40.run_ctx)."""
    out: dict = {}
    for sp, gd in tm.idx.items():
        for g in gd:
            if g[0] == "P1" and g[1] == "none" and str(g[2]) == "1" and g[3] == "cell":
                out[int(g[4])] = out.get(int(g[4]), 0) + 1
    return out


def edition_n_list(ed, tms):
    """판의 n 목록과 뺀 n. fixed = 등록 목록. common = 판의 지역(저장소가 있는 지역) 모두에 키가 있는 n."""
    spec = EDITIONS[ed]
    if spec["n_rule"] == "fixed":
        return list(spec["n_list"]), {}
    pres = {nm: n_presence(tms[nm]) for nm in spec["regions"] if nm in tms}
    if not pres:
        return [], {}
    allns = set().union(*[set(p) for p in pres.values()])
    common = set.intersection(*[set(p) for p in pres.values()])
    excluded = {n_label(n): sorted(nm for nm in spec["regions"] if n not in pres.get(nm, {})) for n in sorted(allns - common, key=n_order)}
    return sorted(common, key=n_order), excluded


# ================================================================ 대비
def groups_for(method, baseline, n):
    """(gA, gB). 방법 키는 h40._g, 기준선 P0 는 h40.P0_GRP, 기준선 P1 은 같은 n 의 h40._g('P1', n)(h40 L4 와 같은 키)."""
    gA = H._g(method, n)
    gB = H.P0_GRP if baseline == "P0" else H._g("P1", n)
    return gA, gB


def fixed_pool_rows(tms, regions, gA, gB, a, label):
    """지역 행과 구성 고정 층화 평균 행. 계산은 h42.region_stats·pool_rows 이고, 판의 지역 가운데 대비가 없거나 CI 풀 밖인 지역이 있으면
    평균 행을 판정 불가로 두고 평균 값 열을 비운다(남은 지역만의 평균을 쓰지 않는다)."""
    per = {}
    for nm in regions:
        tm = tms.get(nm)
        if tm is None:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s_ = X.region_stats(tm, gA, gB)
        if s_ is not None:
            per[nm] = s_
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = X.pool_rows(per, regions, a, label) if per else []
    reg = [r for r in rows if r.get("scope") == "region"]
    mr = next((r for r in rows if r.get("scope") == "MEAN"), None)
    missing = [nm for nm in regions if nm not in per]
    no_ci = [nm for nm in regions if nm in per and per[nm].get("dist") is None]
    for r in reg:
        k_, K_ = int(r.get("n_splits", 0)), int(r.get("n_splits_expected", 0))
        r["status"] = "CI 없음(CI 풀 밖)" if r["target"] in no_ci else (f"부분(분할 {k_}/{K_})" if k_ < K_ else "ok")
        r["composition_complete"] = np.nan
    if mr is None:
        mr = dict(scope="MEAN", n_ci_regions=0, n_regions_target=len(regions), pool_regions="", undetermined=True)
    mr["target"] = f"MEAN[{','.join(regions)}]"
    mr["missing_regions"] = ",".join(missing)
    mr["no_ci_regions"] = ",".join(no_ci)
    complete = not missing and not no_ci
    mr["composition_complete"] = bool(complete)
    if not complete:
        parts = []
        if missing:
            parts.append(f"대비 없는 지역 {','.join(missing)}")
        if no_ci:
            parts.append(f"CI 풀 밖 지역 {','.join(no_ci)}")
        for c in BLANK_ON_INCOMPLETE:
            mr[c] = np.nan
        mr.update(verdict4=NA_VERDICT, verdict4_d10=NA_VERDICT, verdict4_common=NA_VERDICT, undetermined=True,
                  status=f"판정 불가(구성 불완전: {'; '.join(parts)}; 지역 {len(regions) - len(missing) - len(no_ci)}/{len(regions)})")
    elif mr.get("undetermined"):
        mr["status"] = f"판정 불가(계산 실패: {str(mr.get('ci_flag', '')).split(';')[0]})"
    elif mr.get("splits_short"):
        mr["status"] = f"부분(분할 {mr['splits_short']})"
    else:
        mr["status"] = "ok"
    return [mr] + reg


def compute(tms, editions=None, methods=None, baselines=None, nboot=NBOOT):
    """판 × 기준선 × 방법 × n 의 행. 반환 (DataFrame, 판 정보)."""
    a = SimpleNamespace(delta_eq=DELTA_EQ, delta_eq_aux=DELTA_EQ_AUX, nboot=int(nboot))
    editions = list(EDITIONS) if editions is None else list(editions)
    methods = list(METHODS) if methods is None else list(methods)
    baselines = list(BASELINES) if baselines is None else list(baselines)
    out, ed_info = [], {}
    for ed in editions:
        regions = EDITIONS[ed]["regions"]
        ns, excluded = edition_n_list(ed, tms)
        ed_info[ed] = dict(regions=regions, n_rule=EDITIONS[ed]["n_rule"], n_list=[n_label(n) for n in ns], excluded_n=excluded,
                           regions_without_store=[nm for nm in regions if nm not in tms])
        for bl in baselines:
            for m in methods:
                if m == bl:
                    continue
                for n in ns:
                    gA, gB = groups_for(m, bl, n)
                    label = f"{m}-{bl}|n{n_label(n)}"
                    for r in fixed_pool_rows(tms, regions, gA, gB, a, label):
                        r.update(edition=ed, edition_regions=",".join(regions), baseline=bl, method=m, learner=gA[1], lam=float(gA[5]), n=int(n),
                                 n_label=n_label(n), contrast=label, nboot=int(nboot), delta_eq=DELTA_EQ, delta_eq_aux=DELTA_EQ_AUX, role=STATUS_LABEL)
                        out.append(r)
    if hasattr(H4.boot_weights, "cache_clear"):
        H4.boot_weights.cache_clear()                                  # h40 이 씌운 재표집 행렬 캐시(10,000회 행렬)를 비운다
    df = pd.DataFrame(out)
    if not len(df):
        return df, ed_info
    for c in FRONT:
        if c not in df:
            df[c] = np.nan
    df["_e"] = df.edition.map({e: i for i, e in enumerate(editions)})
    df["_b"] = df.baseline.map({b: i for i, b in enumerate(baselines)})
    df["_m"] = df.method.map({m: i for i, m in enumerate(methods)})
    df["_n"] = [n_order(n) for n in df.n]
    df["_s"] = [0 if s == "MEAN" else 1 for s in df.scope]
    df["_r"] = [(EDITIONS[e]["regions"].index(t) if t in EDITIONS[e]["regions"] else -1) for e, t in zip(df.edition, df.target)]
    df = df.sort_values(["_e", "_b", "_m", "_n", "_s", "_r"], kind="mergesort").drop(columns=["_e", "_b", "_m", "_n", "_s", "_r"])
    drop = [c for c in ("undetermined", "ci_flag") if c in df]
    rest = [c for c in df.columns if c not in FRONT and c not in drop]
    return df[FRONT + drop + rest].reset_index(drop=True), ed_info


# ================================================================ 합성 조각(시험·--synthetic)
SYN_REGIONS = {
    "Lena": dict(nA=400, grid=(0, 3, 10, 40, 160, 320, 1000, N_ALL), blocks=16),
    "Canada": dict(nA=300, grid=(0, 3, 10, 40, 160, 320, N_ALL), blocks=18),
    "Russia_W": dict(nA=16, grid=(0, 3, 10, N_ALL), blocks=14),
    "Russia_E": dict(nA=15, grid=(0, 3, 10, N_ALL), blocks=14),
}
SYN_BASE_RMSE = dict(P0=30.0, P1=29.0, P2=29.5, P3=29.2, V1=30.5, D0=33.0, R0=29.8, R1=28.5, R2=28.4)


def _syn_rmse(method, n):
    """합성 방법·n 의 기준 RMSE(cm). 라벨이 늘수록 학습 방법이 조금씩 나아진다. 결과 자료가 아니다."""
    if method == "P0":
        return SYN_BASE_RMSE["P0"]
    k = 0.0 if n == 0 else (np.log10(max(n, 1) + 1.0) if n > 0 else 3.2)
    slope = dict(P1=0.3, P2=0.2, P3=0.3, V1=0.4, D0=1.2, R0=0.6, R1=0.9, R2=0.9)[method]
    return SYN_BASE_RMSE[method] - slope * k


def make_synthetic_shards(shards_dir, tag="lg", regions=None, splits=(1, 2, 3), draws=2, seeds=(0, 1), lams=(0.25, 0.5, 1.0),
                          methods=("P0", "P1", "P2", "P3", "V1", "D0", "R0", "R1", "R2"), drop=None, seed=0, mode="x", parents=None,
                          point_only_valid=True):
    """h40 형식의 합성 cpu 조각을 쓴다(blocksse.npz, runs.csv, unit.json). 결과 자료가 아니다.

    regions: {대상: dict(nA, grid, blocks[, eval_blocks])}(기본 SYN_REGIONS). drop(대상, 모드, 분할, 키) → True 이면 그 키를 저장하지 않는다.
    분할마다 채점 블록은 대상의 블록 풀에서 eval_blocks 개(기본 풀의 절반)를 뽑는다. 키의 블록 SSE = (기준 RMSE × 블록 잡음)² × 셀 수.
    runs.csv 는 h40.build_curve 가 읽는 열(rmse_beq_cm, bias_cm, n_lab, E_used, n_nonfinite, fit_flag, axis, parent, alpha_sel)을 둔다."""
    shards_dir = Path(shards_dir); shards_dir.mkdir(parents=True, exist_ok=True)
    regions = dict(SYN_REGIONS if regions is None else regions)
    parents = parents or {}
    paths = []
    for t, spec in regions.items():
        nb_pool = int(spec["blocks"]); nb_eval = int(spec.get("eval_blocks", max(2, nb_pool // 2)))
        pool = [f"{t}_b{j:02d}" for j in range(nb_pool)]
        for sp in splits:
            rng = np.random.RandomState(seed_of("syn-blk", t, sp, seed))              # 분할은 대상마다 하나(모드 i·x 공통, h40 과 같다)
            blk = sorted(rng.choice(pool, nb_eval, replace=False).tolist())
            ncell = rng.randint(3, 30, size=len(blk))
            cell_blocks = np.repeat(np.array(blk), ncell)
            st = BlockStore(f"{t}|{mode}", sp, cell_blocks, meta=dict(target=t, mode=mode, part="cpu"))
            blk_eff = rng.normal(0.0, 0.10, size=st.nb)                      # 블록 난이도(모든 방법 공통)
            rows = []

            def put(key, method, n, n_lab):
                if drop is not None and drop(t, mode, sp, key):
                    return
                ck, cm = key, method                                              # n = 0 에서 P1–P3 = P0, R1 = R0(h40 규약)을 합성에도 둔다
                if n == 0 and method in ("P1", "P2", "P3"):
                    ck, cm = H.P0_KEY, "P0"
                elif n == 0 and method == "R1":
                    ck, cm = ("R0",) + tuple(key[1:]), "R0"
                r_ = np.random.RandomState(seed_of("syn-key", t, mode, sp, *ck, seed))
                rm = _syn_rmse(cm, n) * (1.0 + blk_eff + r_.normal(0.0, 0.03, size=st.nb))
                sse = (rm ** 2) * st.ncell
                st.add_sse(key, sse, st.ncell)
                rows.append(dict(target=t, mode=mode, parent=parents.get(t, t), split=sp, part="cpu", axis="method", method=key[0],
                                 learner=key[1], alpha=key[2], placement=key[3], n=key[4], n_lab=n_lab, draw=key[5], seed=key[6], lam=key[7],
                                 rmse_cm=float(np.sqrt(sse.sum() / st.ncell.sum())), rmse_beq_cm=float(np.mean(np.sqrt(sse / st.ncell))),
                                 bias_cm=0.0, E_used=np.nan, alpha_sel="", n_blocks_lab=0, n_nonfinite=0, fit_flag=""))

            nA = int(spec["nA"])
            put(H.P0_KEY, "P0", 0, 0)
            for n in spec["grid"]:
                if n > 0 and n >= nA:
                    continue
                n_lab = nA if n == N_ALL else n
                for d in ([0] if n in (0, N_ALL) else range(draws)):
                    for m in ("P1", "P2", "P3", "V1"):
                        if m in methods:
                            put((m, "none", "1", "cell", n, d, -1, 0.0), m, n, n_lab)
                    for s_ in seeds:
                        if "D0" in methods:
                            put(("D0", H.BASE_LEARNER, "1", "cell", n, d, s_, 1.0), "D0", n, n_lab)
                        for m in ("R0", "R1", "R2"):
                            if m in methods:
                                for lam in lams:
                                    put((m, H.BASE_LEARNER, "1", "cell", n, d, s_, float(lam)), m, n, n_lab)
            base = shards_dir / f"{tag}__cpu__{t}__{mode}__s{sp}"
            save_stores([st], Path(str(base) + "_blocksse.npz"))
            pd.DataFrame(rows).to_csv(Path(str(base) + "_runs.csv"), index=False)
            valid = bool(point_only_valid or t not in H.MAIN_POINT)
            unit = dict(target=t, mode=mode, parent=parents.get(t, t), split=int(sp), part="cpu", learner="", tag=tag, status="ok",
                        dup_of=-1, mirror_of=-1, valid=valid, n_eval=int(st.ncell.sum()), nb_eval=int(st.nb), n_A=nA,
                        n_valid_splits=len(splits) if valid else 0, n_unique_splits=len(splits), code_sha="synthetic", cfg_hash="synthetic",
                        cfg_common="synthetic")
            Path(str(base) + "_unit.json").write_text(json.dumps(unit, ensure_ascii=False, indent=1))
            paths.append(base)
    return paths


# ================================================================ 기록
def code_hashes():
    files = dict(module=Path(__file__).resolve(), h40=DL / "h40_label_grid.py", h42=DL / "h42_label_grid_ext.py",
                 h4_common=ROOT / "src" / "polar" / "h4_common.py", m1_stats=ROOT / "src" / "polar" / "m1_stats.py")
    return {k: sha256(p) for k, p in files.items() if Path(p).exists()}


def git_commit():
    return H.git_commit()


def run(shards_dir, out_dir, tag="lg", nboot=NBOOT, editions=None, methods=None, baselines=None, parts=("cpu",), lgs=None,
        synthetic=False, allow_mixed_cfg=False, argv=None, threads=1):
    """계산하고 CSV 와 메타를 쓴다. 반환 dict(csv, meta, n_rows, n_shards)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    editions = list(EDITIONS) if editions is None else list(editions)
    names = sorted({nm for e in editions for nm in EDITIONS[e]["regions"]})
    L = load_inputs(shards_dir, tag, names, nboot, parts, allow_mixed_cfg)
    df, ed_info = compute(L["tms"], editions, methods, baselines, nboot)
    stem = "pool_fixed_curve" + ("_synthetic" if synthetic else "")
    csv_path, meta_path = out_dir / f"{stem}.csv", out_dir / f"{stem}_meta.json"
    tmp = csv_path.with_name(csv_path.name + f".tmp{os.getpid()}")
    df.to_csv(tmp, index=False); os.replace(tmp, csv_path)
    inputs = []
    for s in L["shards"]:
        for k in ("npz", "unit"):
            p = Path(s[k]); inputs.append(dict(path=str(p), sha256=sha256(p)))
    lgs_meta = {}
    if lgs:
        p = Path(lgs) / f"{tag}_meta.json"
        lgs_meta = dict(path=str(p), sha256=sha256(p)) if p.exists() else dict(path=str(p), sha256="", note="파일 없음")
    tms = L["tms"]
    meta = dict(
        module="scripts/4_visualization/paper/pool_fixed_curve.py", spec="figures/figure_spec.json derived_modules.pool_fixed_curve",
        decision="docs/DISPLAY_ITEMS_2026-09-30.md 6절 1번(커밋 903dce9)", status_label=STATUS_LABEL, synthetic=bool(synthetic),
        created=_dt.datetime.now().isoformat(timespec="seconds"), git_commit=git_commit(), code_sha256=code_hashes(),
        argv=list(argv or []), threads=int(threads), niceness=_niceness(), shards_dir=str(shards_dir), tag=tag, parts=list(parts),
        n_shards=len(L["shards"]), inputs=inputs, lgs_meta=lgs_meta, shard_cfg=L["cfg_info"], missing_shards={k: list(v) for k, v in L["missing_shards"].items()},
        unit_status=sorted({str(u.get("status", "")) for u in L["units"]}),
        nboot=int(nboot), delta_eq=DELTA_EQ, delta_eq_aux=DELTA_EQ_AUX, editions=ed_info, methods=list(methods or METHODS),
        baselines=list(baselines or BASELINES),
        seeds={nm: dict(lgboot=int(tm.seed), lgxboot=int(seed_of("lgxboot", tm.name)), splits_used=sorted(int(s) for s in tm.used),
                        split_seeds={str(int(s)): int(seed_of(tm.seed, s)) for s in sorted(tm.used)}, has_ci=bool(tm.has_ci),
                        nb_union=int(tm.nb_union)) for nm, tm in tms.items()},
        rules=dict(
            composition="판 안의 모든 n 에서 풀 지역 집합이 같다. 판의 지역 가운데 대비가 없거나 CI 풀 밖인 지역이 있으면 평균 행은 판정 불가, 평균 값 열 NaN",
            e2_n="E2 의 n = 레나·캐나다 두 저장소 모두에 P1 키(방법 축 칸)가 있는 n. 한 지역에만 있는 n 은 excluded_n",
            ci="h42.region_stats(h40.contrast → h4_common.boot_delta_blocks) + h42.pool_rows(m1_stats.strat). 두 가중(셀 가중, 블록 등가중)",
            verdict4=f"h42.verdict4: 우세 = 두 가중 CI 상한 < 0, 열세 = 두 가중 CI 하한 > 0, 동등 = 네 끝값 절댓값 ≤ {DELTA_EQ} cm, 그 밖 미결정, "
                     f"끝값 비유한 = 판정 불가. verdict4_d10 은 δ {DELTA_EQ_AUX} cm",
            seed="하네스 규칙: TMx.seed = seed_of('lgboot', 이름), 분할 seed = seed_of(TMx.seed, 분할), 공통 재표집 seed_of('lgxboot', 이름)",
            split_info="unit.json 의 dup_of·valid·n_eval·n_A(실행 때 h40.Data.split_structure 값). 기대 분할 수 = n_valid_splits(없으면 n_unique_splits)",
            keys="h40._g(방법, n). 기준선 P0 = h40.P0_GRP, P1 = h40._g('P1', n)"),
        n_rows=int(len(df)), n_mean_rows=int((df.scope == "MEAN").sum()) if len(df) else 0)
    tmpm = meta_path.with_name(meta_path.name + f".tmp{os.getpid()}")
    tmpm.write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=_py)); os.replace(tmpm, meta_path)
    return dict(csv=csv_path, meta=meta_path, n_rows=int(len(df)), n_shards=len(L["shards"]), df=df)


def _py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return str(v)


def _niceness():
    try:
        return int(os.nice(0))
    except OSError:
        return -1


def _raise_nice(target=10):
    try:
        cur = os.nice(0)
        if cur < target:
            os.nice(target - cur)
    except OSError:
        pass


# ================================================================ 명령행
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Fig 2e·2f·3d 구성 고정 풀 평균 곡선(서술)")
    ap.add_argument("--lg-shards", default="", help="LG 본 실행 조각 폴더($LGD/shards, 읽기 전용). --synthetic 이 아니면 필수")
    ap.add_argument("--lg-tag", default="lg")
    ap.add_argument("--lgs", default="", help="LG 표 폴더($LGS, 선택). <tag>_meta.json 의 sha256 만 기록한다")
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "paper"))
    ap.add_argument("--nboot", type=int, default=NBOOT)
    ap.add_argument("--editions", default=",".join(EDITIONS))
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--baselines", default=",".join(BASELINES))
    ap.add_argument("--parts", default="cpu", help="읽을 조각 부분(기본 cpu. 방법 축 키는 모두 cpu 조각에 있다)")
    ap.add_argument("--threads", type=int, default=1, help="BLAS 스레드(1–2)")
    ap.add_argument("--allow-mixed-cfg", action="store_true", help="조각 사이 설정 해시가 달라도 진행한다(h40.check_shard_cfg)")
    ap.add_argument("--synthetic", action="store_true", help="합성 조각으로 계산한다(<out>/_synthetic_shards, 산출 이름에 _synthetic)")
    a = ap.parse_args(argv)
    a.threads = max(1, min(int(a.threads), THREADS_MAX))
    for lst, allowed, what in ((a.editions, EDITIONS, "판"), (a.methods, METHODS, "방법"), (a.baselines, BASELINES, "기준선")):
        bad = [v for v in lst.split(",") if v and v not in allowed]
        if bad:
            raise SystemExit(f"알 수 없는 {what}: {bad}")
    if not a.synthetic and not a.lg_shards:
        raise SystemExit("--lg-shards 가 필요하다($LGD/shards). 시험은 --synthetic")
    return a


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    a = parse_args(argv)
    _raise_nice(10)
    out = Path(a.out)
    shards = Path(a.lg_shards) if a.lg_shards else None
    if a.synthetic:
        shards = out / "_synthetic_shards"
        make_synthetic_shards(shards, tag=a.lg_tag)
    res = run(shards, out, tag=a.lg_tag, nboot=a.nboot, editions=[v for v in a.editions.split(",") if v],
              methods=[v for v in a.methods.split(",") if v], baselines=[v for v in a.baselines.split(",") if v],
              parts=tuple(v for v in a.parts.split(",") if v), lgs=a.lgs or None, synthetic=a.synthetic, allow_mixed_cfg=a.allow_mixed_cfg,
              argv=argv, threads=a.threads)
    print(f"[pool_fixed_curve] 조각 {res['n_shards']}개 읽음 · 행 {res['n_rows']} → {res['csv']}", flush=True)
    print(f"[pool_fixed_curve] 메타 → {res['meta']}", flush=True)
    return res


if __name__ == "__main__":
    main()
