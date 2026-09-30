"""NOVELTY N1 대상별 요약(Fig 4d, SI ST_n1_target_summary). 서술 산출이며 판정에 쓰지 않는다.

근거
  figures/figure_spec.json 의 derived_modules.n1_target_summary 와 Fig 4 패널 d, docs/DISPLAY_ITEMS_2026-09-30.md 6절 2번(커밋 903dce9),
  docs/NOVELTY_POSITIONING_2026-09-29.md 4절 N1('각 n 에서 평균과 중앙값, 오차가 커진 대상 수, 최대 증가'),
  docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md 1.3('대상 N개 가운데 우세 k, 열세 m, 동등 e, 미결정 j'). LG 결과를 열람하기 전에 작성했다.

입력
  주 표     $LGS/<tag>_curve.csv(h40.build_curve 형식, 곡선 CI = 분할 안 블록 부트스트랩, 두 가중). --curve 로 경로를 직접 줄 수 있다
  보조 표   --aux-curve(여러 번): 같은 형식의 lgt_curve.csv(h43), lgf_curve.csv(h47), lgfn_curve.csv(h48). 표마다 따로 센다(플랫폼 사이의
            값을 합치지 않는다. LGF 2.2)
  재표집 횟수는 새로 정하지 않는다. 곡선 표 옆의 <tag>_meta.json 에서 읽어 nboot 열에 적는다(h40 은 args.nboot, h43·h47·h48 은 nboot).
  메타가 없으면 --curve-nboot(기본 1,000)를 적고 nboot_source 에 'default' 를 적는다.

행 선택(스펙)
  방법 {P1, R0, R1, D0}, α '1', 배치 cell, 방법별 기준 λ(h40.base_lam: P 0.0, D0 1.0, 잔차 0.25).
  주 표의 학습기는 {catboost_lo, none}(스펙 Fig 2·4 의 필터). 보조 표는 표에 있는 학습기를 모두 따로 센다(예: lgfn 의 mlp, mlp_tuned).
  대비 종류 d_p0(기준선 P0), d_p1(기준선 같은 n 의 P1. n = 0 은 h40 규약상 P0). P1 의 d_p1 은 0 이므로 빼고 셈한다.
  point_only 행(러시아 C, 그린란드, 유효 분할이 없는 대상)은 빼고 그 수를 n_point_only 에 적는다.

대상 묶음(구성 고정. 묶음의 대상 목록은 n 과 무관하다)
  indep  독립 지역: P4 + Alaska(x)(스펙 derived_modules 의 묶음)
  P4     주 4지역(모드 x). NOVELTY N1 문장의 '지역 전체를 제외한 4지역'과 WRAPUP 1.2('Alaska(x)는 주 4지역 평균과 합치지 않는다')에 맞춘 보조 묶음
  sub_i  하위 지역 10개(AL-1–AL-6, CA-2, CA-3, LE-1, LE-2), 모드 i
  sub_x  같은 하위 지역, 모드 x
  묶음의 대상 가운데 그 (방법, 학습기, n)의 행이 없는 대상은 n_missing 과 missing_targets 에 적고 composition_complete 를 거짓으로 둔다.
  평균·중앙값은 행이 있고 Δ 가 유한한 대상만으로 계산하므로 composition_complete 가 거짓인 n 의 평균은 대상 구성이 다르다(열로 구분한다).

센 값(새 재표집 없음)
  평균·중앙값      대상별 점 추정 Δ(셀 가중 d, 블록 등가중 d_beq)의 평균과 중앙값. CI 평균이 아니다(하위 지역은 WRAPUP 1.3 대로 개수로 쓴다)
  4분 판정 수      대상별로 h42.verdict4(d_lo, d_hi, d_beq_lo, d_beq_hi, δ)를 적용해 우세·동등·미결정·열세·판정 불가를 센다.
                   δ = 0.5 cm(주), 1.0 cm(보조, _d10 열). 우세 = 두 가중 CI 상한 < 0, 열세 = 두 가중 CI 하한 > 0,
                   동등 = 그 밖에서 네 끝값의 절댓값이 모두 δ 이하, 미결정 = 그 외, 끝값이 비유한 = 판정 불가
  악화 대상 수     n_inferior(열세 대상 수, 두 가중 CI 기준). 점 추정 Δ > 0 인 대상 수는 n_increase_point 로 따로 적는다
  최대 증가        max_increase = 대상별 점 추정 Δ(셀 가중)의 최댓값과 그 대상(동률이면 묶음 목록 순서의 앞 대상). 값이 음수일 수 있다

산출
  --out(기본 data/processed/paper)/n1_target_summary.csv 와 n1_target_summary_meta.json(입력 sha256, 코드 해시, nboot, 규칙).
  --synthetic 은 pool_fixed_curve.make_synthetic_shards 의 합성 조각에서 h40.build_curve 로 합성 곡선 표를 만든 뒤 센다(산출 이름에 _synthetic).
  화면에는 파일 이름과 행 수만 낸다(결과 값을 출력하지 않는다).

실행(ROOT, CPU 전용, nice 10, 스레드 1–2. 결과 열람 규칙에 따라 C1·J1 뒤에만 실제 표로 돌린다)
  CUDA_VISIBLE_DEVICES= nice -n 10 python3 scripts/4_visualization/paper/n1_target_summary.py --lgs "$LGS" \
      --aux-curve data/processed/lgt/lgt_curve.csv --aux-curve data/processed/lgf/lgf_curve.csv --aux-curve data/processed/lgf/lgfn_curve.csv
  시험용 합성: CUDA_VISIBLE_DEVICES= nice -n 10 python3 scripts/4_visualization/paper/n1_target_summary.py --synthetic --out /tmp/x
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

THREADS_MAX = 2
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _peek_threads(argv=None, default=1):
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


if __name__ == "__main__":
    for _v in THREAD_VARS:
        os.environ[_v] = str(_peek_threads())
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    for _v in THREAD_VARS:
        os.environ.setdefault(_v, "1")

import numpy as np                                                                                   # noqa: E402
import pandas as pd                                                                                  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
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


X = _load("h42_label_grid_ext", DL / "h42_label_grid_ext.py")      # verdict4 의 출처. h40 은 X.H
H = X.H

# ================================================================ 고정 설계값(스펙 derived_modules.n1_target_summary)
DELTA_EQ = 0.5
DELTA_EQ_AUX = 1.0
N_ALL = -1
METHODS = ["P1", "R0", "R1", "D0"]
KINDS = {"p0": "P0", "p1": "P1"}
MAIN_LEARNERS = (H.BASE_LEARNER, "none")
SUBS = list(H.SUB_AL) + list(H.SUB_OTHER)
GROUPS = {
    "indep": [f"{t}|x" for t in list(H.MAIN4) + [H.ALASKA]],
    "P4": [f"{t}|x" for t in H.MAIN4],
    "sub_i": [f"{t}|i" for t in SUBS],
    "sub_x": [f"{t}|x" for t in SUBS],
}
GROUP_NOTE = {
    "indep": "독립 지역(P4 + Alaska(x), 스펙 묶음). 평균은 점 추정 평균이다(CI 평균 아님)",
    "P4": "주 4지역(NOVELTY N1 문장의 4지역, 보조 묶음). 평균은 점 추정 평균이다(CI 평균 아님)",
    "sub_i": "하위 지역 모드 i: 본문은 개수로만 쓴다(WRAPUP 1.3). CI 평균 없음",
    "sub_x": "하위 지역 모드 x: 본문은 개수로만 쓴다(WRAPUP 1.3). CI 평균 없음",
}
V4 = ("우세", "동등", "미결정", "열세", "판정 불가")
V4_COL = dict(zip(V4, ("n_superior", "n_equivalent", "n_undecided", "n_inferior", "n_na")))
STATUS_LABEL = "서술(판정에 쓰지 않는다)"
DEFAULT_NBOOT = 1000


def n_label(n):
    return "all" if int(n) == N_ALL else str(int(n))


def n_order(n):
    return (1, 0) if int(n) == N_ALL else (0, int(n))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _bool_col(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


# ================================================================ 표 읽기
def read_curve(path):
    """곡선 표(h40.build_curve 형식). α 와 배치는 문자열로 읽는다."""
    df = pd.read_csv(path, dtype=dict(alpha=str, placement=str, learner=str, method=str, target=str, mode=str), keep_default_na=False,
                     na_values=["", "nan", "NaN"])
    need = ["target", "mode", "method", "learner", "alpha", "placement", "n", "lam"]
    miss = [c for c in need if c not in df]
    if miss:
        raise ValueError(f"{path}: 곡선 표 열이 없다 {miss}")
    return df


def read_curve_nboot(path, fallback=DEFAULT_NBOOT):
    """곡선 표 옆 <tag>_meta.json 의 재표집 횟수. 반환 (nboot, 출처, 메타 경로 또는 '')."""
    p = Path(path)
    mp = p.with_name(p.name[:-len("_curve.csv")] + "_meta.json") if p.name.endswith("_curve.csv") else None
    if mp is not None and mp.exists():
        try:
            m = json.loads(mp.read_text())
            v = m.get("nboot", (m.get("args") or {}).get("nboot"))
            if v is not None:
                return int(v), "meta", str(mp)
        except (OSError, ValueError):
            pass
    return int(fallback), "default", ""


def select_rows(cur, main=True):
    """스펙의 행 선택: 방법 {P1, R0, R1, D0}, α '1', 배치 cell, 방법별 기준 λ, 주 표는 학습기 {catboost_lo, none}."""
    c = cur.copy()
    c["alpha"] = c["alpha"].astype(str); c["placement"] = c["placement"].astype(str); c["learner"] = c["learner"].astype(str)
    c = c[c.method.isin(METHODS)]
    lamb = c.method.map(H.base_lam).astype(float)
    keep = (c.alpha == "1") & (c.placement == "cell") & np.isclose(c.lam.astype(float), lamb)
    if main:
        keep &= c.learner.isin(MAIN_LEARNERS)
    c = c[keep].copy()
    c["n"] = c.n.astype(int)
    c["name"] = c.target.astype(str) + "|" + c["mode"].astype(str)
    c["point_only"] = _bool_col(c["point_only"]) if "point_only" in c else False
    return c


# ================================================================ 세기
def verdicts(sub, kind, delta):
    """대상 행마다 h42.verdict4(d_lo, d_hi, d_beq_lo, d_beq_hi, δ)."""
    p = f"d_{kind}"
    cols = [f"{p}_lo", f"{p}_hi", f"{p}_beq_lo", f"{p}_beq_hi"]
    for c in cols:
        if c not in sub:
            sub[c] = np.nan
    return [X.verdict4(lo, hi, lob, hib, delta) for lo, hi, lob, hib in sub[cols].astype(float).itertuples(index=False, name=None)]


def group_row(sub, kind, names):
    """묶음 하나의 요약 dict. sub = 그 (방법, 학습기, n)에서 묶음 대상의 행(point_only 포함)."""
    order = {nm: i for i, nm in enumerate(names)}
    sub = sub.assign(_o=sub["name"].map(order)).sort_values("_o", kind="mergesort")
    dup = sub["name"].duplicated()
    if dup.any():
        raise ValueError(f"같은 대상의 행이 둘 이상이다: {sorted(set(sub['name'][dup]))}")
    po = sub[sub["point_only"]]
    s = sub[~sub["point_only"]].copy()
    present = set(sub["name"])
    missing = [nm for nm in names if nm not in present]
    p = f"d_{kind}"
    d = s[p].astype(float).values if p in s else np.full(len(s), np.nan)
    db = s[f"{p}_beq"].astype(float).values if f"{p}_beq" in s else np.full(len(s), np.nan)
    v = verdicts(s, kind, DELTA_EQ) if len(s) else []
    v10 = verdicts(s, kind, DELTA_EQ_AUX) if len(s) else []
    fin, finb = np.isfinite(d), np.isfinite(db)
    out = dict(n_group=len(names), n_rows=int(len(s)), n_point_only=int(len(po)), n_missing=len(missing), missing_targets=",".join(missing),
               point_only_targets=",".join(po["name"]), composition_complete=bool(not missing and not len(po)), n_finite=int(fin.sum()),
               mean_d=float(np.mean(d[fin])) if fin.any() else np.nan, median_d=float(np.median(d[fin])) if fin.any() else np.nan,
               mean_d_beq=float(np.mean(db[finb])) if finb.any() else np.nan, median_d_beq=float(np.median(db[finb])) if finb.any() else np.nan,
               n_increase_point=int((d[fin] > 0).sum()))
    for lab, col in V4_COL.items():
        out[col] = int(sum(x == lab for x in v))
        out[f"{col}_d10"] = int(sum(x == lab for x in v10))
    nm = s["name"].values
    out["superior_targets"] = ",".join(nm[i] for i, x in enumerate(v) if x == "우세")
    out["inferior_targets"] = ",".join(nm[i] for i, x in enumerate(v) if x == "열세")
    if fin.any():
        j = int(np.flatnonzero(fin)[np.argmax(d[fin])])                  # 동률이면 묶음 순서의 앞 대상(argmax 의 첫 위치)
        out.update(max_increase=float(d[j]), max_increase_target=str(nm[j]))
    else:
        out.update(max_increase=np.nan, max_increase_target="")
    return out


def summarize(cur, source, source_file="", main=True, nboot=DEFAULT_NBOOT, nboot_source="default"):
    """곡선 표 하나의 요약 행."""
    c = select_rows(cur, main)
    rows = []
    for (m, lr, n), sub in c.groupby(["method", "learner", "n"], sort=False):
        for kind, bl in KINDS.items():
            if m == "P1" and kind == "p1":
                continue
            for g, names in GROUPS.items():
                s = sub[sub["name"].isin(names)]
                if not len(s):
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = group_row(s, kind, names)
                lam = float(H.base_lam(m))
                r.update(source=source, source_file=source_file, group=g, group_targets=",".join(names), kind=f"d_{kind}", baseline=bl, method=m,
                         learner=lr, lam=lam, n=int(n), n_label=n_label(n), contrast=f"{m}[{lr}]-{bl}", nboot=int(nboot), nboot_source=nboot_source,
                         delta_eq=DELTA_EQ, delta_eq_aux=DELTA_EQ_AUX, note=GROUP_NOTE[g], role=STATUS_LABEL)
                rows.append(r)
    return rows


FRONT = ["source", "group", "kind", "baseline", "method", "learner", "lam", "n", "n_label", "contrast", "n_group", "n_rows", "n_point_only",
         "n_missing", "composition_complete", "n_finite", "mean_d", "median_d", "mean_d_beq", "median_d_beq", "n_superior", "n_equivalent",
         "n_undecided", "n_inferior", "n_na", "n_superior_d10", "n_equivalent_d10", "n_undecided_d10", "n_inferior_d10", "n_na_d10",
         "n_increase_point", "max_increase", "max_increase_target", "superior_targets", "inferior_targets", "missing_targets",
         "point_only_targets", "group_targets", "nboot", "nboot_source", "delta_eq", "delta_eq_aux", "note", "role", "source_file"]


def frame(rows, sources):
    df = pd.DataFrame(rows)
    if not len(df):
        return pd.DataFrame(columns=FRONT)
    df["_s"] = df.source.map({s: i for i, s in enumerate(sources)})
    df["_g"] = df.group.map({g: i for i, g in enumerate(GROUPS)})
    df["_k"] = df.kind.map({f"d_{k}": i for i, k in enumerate(KINDS)})
    df["_m"] = df.method.map({m: i for i, m in enumerate(METHODS)})
    df["_n"] = [n_order(n) for n in df.n]
    df = df.sort_values(["_s", "_g", "_k", "_m", "learner", "_n"], kind="mergesort").drop(columns=["_s", "_g", "_k", "_m", "_n"])
    return df[FRONT].reset_index(drop=True)


# ================================================================ 합성 입력(시험·--synthetic)
SYN_TARGETS_X = {"Lena": dict(nA=400, grid=(0, 3, 10, 40, 160, 1000, N_ALL), blocks=16),
                 "Canada": dict(nA=300, grid=(0, 3, 10, 40, 160, N_ALL), blocks=18),
                 "Russia_W": dict(nA=16, grid=(0, 3, 10, N_ALL), blocks=14),
                 "Russia_E": dict(nA=15, grid=(0, 3, 10, N_ALL), blocks=14),
                 "Alaska": dict(nA=500, grid=(0, 3, 10, 40, 160, N_ALL), blocks=30),
                 "Russia_C": dict(nA=4, grid=(0, 3, N_ALL), blocks=6)}
SYN_SUBS = {"AL-1": dict(nA=120, grid=(0, 3, 10, 40, N_ALL), blocks=12), "AL-2": dict(nA=90, grid=(0, 3, 10, 40, N_ALL), blocks=12),
            "CA-2": dict(nA=60, grid=(0, 3, 10, 40, N_ALL), blocks=10)}
SYN_PARENT = {"AL-1": "Alaska", "AL-2": "Alaska", "CA-2": "Canada"}


def synthetic_curve(in_dir, nboot=200, tag="lg"):
    """합성 조각(pool_fixed_curve.make_synthetic_shards)에서 h40.build_curve 로 형식이 같은 합성 곡선 표를 만든다. 결과 자료가 아니다.
    반환 (주 곡선 경로, 보조 곡선 경로). 보조 표는 주 곡선의 D0 행의 학습기 이름을 'mlp_tuned' 로 바꾼 lgfn 형식 표다."""
    P = _load("pool_fixed_curve", HERE / "pool_fixed_curve.py")
    in_dir = Path(in_dir); sh = in_dir / "shards"
    P.make_synthetic_shards(sh, tag=tag, regions=SYN_TARGETS_X, mode="x", methods=("P0", "P1", "D0", "R0", "R1"), splits=(1, 2), draws=2,
                            seeds=(0,), lams=(0.25, 1.0))
    for md in ("i", "x"):
        P.make_synthetic_shards(sh, tag=tag, regions=SYN_SUBS, mode=md, methods=("P0", "P1", "D0", "R0", "R1"), splits=(1, 2), draws=2,
                                seeds=(0,), lams=(0.25, 1.0), parents=SYN_PARENT, seed=1)
    found = X.find_shards_x(sh, tag)
    units = [json.loads(Path(s["unit"]).read_text()) for s in found]
    info = P.ShardSplitInfo(units)
    stores = X.load_stores([s["npz"] for s in found])
    names = sorted({k[0] for k in stores})
    tms, _ = P.build_tms(stores, info, names, nboot)
    runs = pd.concat([pd.read_csv(s["runs"], dtype=dict(alpha=str, alpha_sel=str, fit_flag=str), keep_default_na=False, na_values=["", "nan", "NaN"])
                      for s in found], ignore_index=True)
    runs["alpha_sel"] = runs.alpha_sel.fillna(""); runs["fit_flag"] = runs.fit_flag.fillna("").astype(str)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cur = pd.concat([H.build_curve({nm: tm}, runs[(runs.target == tm.target) & (runs["mode"] == tm.mode)]) for nm, tm in tms.items()],
                        ignore_index=True)
    if hasattr(X.H4.boot_weights, "cache_clear"):
        X.H4.boot_weights.cache_clear()
    main = in_dir / f"{tag}_curve.csv"
    cur.to_csv(main, index=False)
    (in_dir / f"{tag}_meta.json").write_text(json.dumps(dict(stage="synthetic", args=dict(nboot=int(nboot))), ensure_ascii=False))
    aux = cur[cur.method == "D0"].copy(); aux["learner"] = "mlp_tuned"
    aux = pd.concat([cur[cur.method == "P1"], aux], ignore_index=True)
    aux_p = in_dir / "lgfn_curve.csv"
    aux.to_csv(aux_p, index=False)
    (in_dir / "lgfn_meta.json").write_text(json.dumps(dict(stage="synthetic", nboot=int(nboot)), ensure_ascii=False))
    return main, aux_p


# ================================================================ 기록
def code_hashes():
    files = dict(module=Path(__file__).resolve(), h40=DL / "h40_label_grid.py", h42=DL / "h42_label_grid_ext.py")
    return {k: sha256(p) for k, p in files.items() if Path(p).exists()}


def _source_name(path):
    n = Path(path).name
    return n[:-len("_curve.csv")] if n.endswith("_curve.csv") else Path(path).stem


def run(curve, out_dir, aux=(), synthetic=False, curve_nboot=DEFAULT_NBOOT, argv=None, threads=1):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tables = [(Path(curve), True)] + [(Path(p), False) for p in aux]
    rows, sources, inputs = [], [], []
    for p, is_main in tables:
        cur = read_curve(p)
        nb, nsrc, mp = read_curve_nboot(p, curve_nboot)
        src = _source_name(p)
        if src in sources:
            src = f"{src}#{len(sources)}"
        sources.append(src)
        rows += summarize(cur, src, str(p), main=is_main, nboot=nb, nboot_source=nsrc)
        inputs.append(dict(path=str(p), sha256=sha256(p), main=bool(is_main), source=src, nboot=nb, nboot_source=nsrc,
                           meta_path=mp, meta_sha256=sha256(mp) if mp else ""))
    df = frame(rows, sources)
    stem = "n1_target_summary" + ("_synthetic" if synthetic else "")
    csv_path, meta_path = out_dir / f"{stem}.csv", out_dir / f"{stem}_meta.json"
    tmp = csv_path.with_name(csv_path.name + f".tmp{os.getpid()}")
    df.to_csv(tmp, index=False); os.replace(tmp, csv_path)
    meta = dict(
        module="scripts/4_visualization/paper/n1_target_summary.py", spec="figures/figure_spec.json derived_modules.n1_target_summary",
        decision="docs/DISPLAY_ITEMS_2026-09-30.md 6절 2번(커밋 903dce9)", status_label=STATUS_LABEL, synthetic=bool(synthetic),
        created=_dt.datetime.now().isoformat(timespec="seconds"), git_commit=H.git_commit(), code_sha256=code_hashes(), argv=list(argv or []),
        threads=int(threads), inputs=inputs, groups=GROUPS, methods=METHODS, kinds={f"d_{k}": v for k, v in KINDS.items()},
        main_learners=list(MAIN_LEARNERS), delta_eq=DELTA_EQ, delta_eq_aux=DELTA_EQ_AUX,
        rules=dict(
            select="방법 {P1, R0, R1, D0}, α '1', 배치 cell, h40.base_lam 기준 λ. 주 표 학습기 {catboost_lo, none}, 보조 표는 학습기별. "
                   "P1 의 d_p1 제외. point_only 행 제외(n_point_only)",
            verdict4=f"h42.verdict4(d_lo, d_hi, d_beq_lo, d_beq_hi, δ). δ {DELTA_EQ} cm(주), {DELTA_EQ_AUX} cm(_d10). 새 재표집 없음(nboot 열 = 곡선 CI 의 횟수)",
            mean_median="행이 있고 Δ 가 유한한 대상의 점 추정 Δ 평균·중앙값(셀 가중 d, 블록 등가중 d_beq). CI 평균이 아니다",
            worse="n_inferior = 열세 대상 수. n_increase_point = 점 추정 Δ > 0 인 대상 수",
            max_increase="대상별 점 추정 Δ(셀 가중)의 최댓값과 대상. 동률이면 묶음 목록 순서의 앞 대상",
            composition="묶음 대상 목록은 n 과 무관하게 고정. 행이 없는 대상 = n_missing, composition_complete = 거짓"),
        n_rows=int(len(df)), sources=sources)
    tmpm = meta_path.with_name(meta_path.name + f".tmp{os.getpid()}")
    tmpm.write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str)); os.replace(tmpm, meta_path)
    return dict(csv=csv_path, meta=meta_path, n_rows=int(len(df)), n_tables=len(tables), df=df)


def _raise_nice(target=10):
    try:
        cur = os.nice(0)
        if cur < target:
            os.nice(target - cur)
    except OSError:
        pass


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="NOVELTY N1 대상별 요약(서술)")
    ap.add_argument("--lgs", default="", help="LG 표 폴더($LGS). <lg-tag>_curve.csv 를 읽는다")
    ap.add_argument("--lg-tag", default="lg")
    ap.add_argument("--curve", default="", help="주 곡선 표 경로(주면 --lgs 대신 쓴다)")
    ap.add_argument("--aux-curve", action="append", default=[], help="같은 형식의 보조 곡선 표(lgt_curve, lgf_curve, lgfn_curve). 여러 번")
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "paper"))
    ap.add_argument("--curve-nboot", type=int, default=DEFAULT_NBOOT, help="곡선 표 메타에 재표집 횟수가 없을 때 nboot 열에 적는 값")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--synthetic", action="store_true", help="합성 조각 → h40.build_curve 합성 곡선 표로 센다(산출 이름에 _synthetic)")
    ap.add_argument("--synthetic-nboot", type=int, default=200, help="합성 곡선 표를 만들 때의 재표집 횟수")
    a = ap.parse_args(argv)
    a.threads = max(1, min(int(a.threads), THREADS_MAX))
    if not a.synthetic and not (a.curve or a.lgs):
        raise SystemExit("--lgs 또는 --curve 가 필요하다. 시험은 --synthetic")
    return a


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    a = parse_args(argv)
    _raise_nice(10)
    out = Path(a.out)
    aux = list(a.aux_curve)
    if a.synthetic:
        curve, aux_p = synthetic_curve(out / "_synthetic_inputs", nboot=a.synthetic_nboot, tag=a.lg_tag)
        aux = aux + [str(aux_p)]
    else:
        curve = Path(a.curve) if a.curve else Path(a.lgs) / f"{a.lg_tag}_curve.csv"
    res = run(curve, out, aux=aux, synthetic=a.synthetic, curve_nboot=a.curve_nboot, argv=argv, threads=a.threads)
    print(f"[n1_target_summary] 곡선 표 {res['n_tables']}개 · 행 {res['n_rows']} → {res['csv']}", flush=True)
    print(f"[n1_target_summary] 메타 → {res['meta']}", flush=True)
    return res


if __name__ == "__main__":
    main()
