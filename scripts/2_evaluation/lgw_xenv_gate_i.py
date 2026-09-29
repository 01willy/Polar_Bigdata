"""교차 환경 점검 (i). 계획 docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md 8.4 (i)(커밋 316714c)와 LG 계획서 개정 14.

LG 사전 점검과 스모크의 CPU 단위를 로컬에서 h40 으로 다시 적합해 Rescale 조각과 키별 셀 가중 RMSE 로 대조한다.
  대상(8.4): results/rescale_lg_smoke/data/processed/lg/shards/ 의
    lg_precheck__cpu__Canada__x__s1   (qOkSo 사전 점검, n {0, 40}, 추출 0)
    lg_smoke__cpu__Russia_W__x__s1, lg_smoke__cpu__AL-3__i__s1, lg_smoke__cpu__Canada__x__s1   (스모크, n {0, 3, 40, 전량}, 추출 0)
  재적합: h40 을 고치지 않고 importlib 로 불러 h40.main 을 호출한다. 사전 점검은 --precheck, 스모크는 --smoke 로 조각 설정을 재현한다.
    --part cpu --tag lgxe --out-dir data/processed/lgw/xenv/lg --threads 4 --workers 0 --allow-local --no-summarize(조각 tag 는 h40 규칙대로
    lgxe_precheck, lgxe_smoke 가 된다). 자료는 기본 경로(data/processed 의 v3 표와 하위 지역 대응표)다. GPU 학습기는 넣지 않는다.
  대조: 두 조각의 공통 키 전부(방법 축, α 축, 중첩 선택, 배치, ridge 학습기). 조각 설정의 공통 해시(cfg_common)와 h40 코드 해시를 함께 대조한다.
  키 분류(LG 개정 14, h51.key_class 와 같다): 물리식 = P0–P3(학습기 none, 허용 차 1e-9 cm), ridge = V1(sklearn Ridge 앵커, 학습기 none)과
    ridge 학습기(0.02 cm), CatBoost = catboost_lo(0.02 cm).
  판정 범위: 8.4 가 정한 범위(공통 키 전부)가 주 판정이다. 6A.6 의 기준 방법(P0, P1, P2, D0, D1, R0, R1, R2)의 α 1·셀 무작위 키만의 결과를 병기한다.
  출력 제한(8.4): 키별 RMSE 차이만 낸다. RMSE 값, Δ, 판정은 출력하지 않는다. 재적합 조각은 이 대조 외에는 열지 않는다.
  선택: --local-ref DIR:TAG 가 있으면 같은 단위의 다른 로컬 재적합(예: 스레드 2 판)과 이번 재적합을 같은 방식으로 대조해 요약에 병기한다(판정에 쓰지 않는다).

산출
  data/processed/lgw/xenv/lg/shards/lgxe_*    재적합 조각(게이트 계산 외에는 열지 않는다)
  data/processed/lgw/lgw_xenv_gate_i.csv       키별 대조(단위, 키, 분류, 허용 차, 절대 차, 초과 여부). RMSE 값은 싣지 않는다
  data/processed/lgw/lgw_xenv_gate_i_summary.json  요약(점검 범위: 단위 수, 스레드, tag. 분류별 최댓값·중앙값·초과 키 수, 판정)

실행(ROOT): OMP_NUM_THREADS=4 nice -n 10 python3 scripts/2_evaluation/lgw_xenv_gate_i.py --allow-local --threads 4
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path


def _peek(flag, default=None):
    for i, v in enumerate(sys.argv):
        if v == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
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
from polar.h4_common import load_stores                                                              # noqa: E402

TOL_PHYS = 1e-9
TOL_ML = 0.02
PHYS_METHODS = ("P0", "P1", "P2", "P3")
BASE_METHODS = ("P0", "P1", "P2", "D0", "D1", "R0", "R1", "R2")                                     # 6A.6 의 기준 방법(h42.BASE_METHODS)
TAG = "lgxe"
UNITS = [("precheck", "Canada", "x", 1), ("smoke", "Russia_W", "x", 1), ("smoke", "AL-3", "i", 1), ("smoke", "Canada", "x", 1)]


def key_class(k):
    """키 분류(LG 개정 14). 물리식 = P0–P3(학습기 none), ridge = V1(학습기 none, sklearn Ridge 앵커)과 ridge 학습기, CatBoost = 그 밖."""
    if k[1] == "none":
        return "phys" if k[0] in PHYS_METHODS else "ridge"
    return "ridge" if k[1] == "ridge" else "catboost"


def key_tol(cls):
    return TOL_PHYS if cls == "phys" else TOL_ML


def in_base(k):
    return k[0] in BASE_METHODS and k[1] in ("none", H.BASE_LEARNER) and str(k[2]) == "1" and k[3] == "cell"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="WRAPUP 8.4 (i) 교차 환경 점검")
    ap.add_argument("--ref-dir", default="results/rescale_lg_smoke/data/processed/lg/shards", help="Rescale 조각(읽기 전용)")
    ap.add_argument("--out-dir", default="data/processed/lgw/xenv/lg", help="재적합 조각 폴더(<out-dir>/shards)")
    ap.add_argument("--gate-csv", default="data/processed/lgw/lgw_xenv_gate_i.csv")
    ap.add_argument("--summary", default="data/processed/lgw/lgw_xenv_gate_i_summary.json")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--allow-local", action="store_true", help="재적합(학습)을 허용한다")
    ap.add_argument("--compare-only", action="store_true", help="재적합 없이 이미 있는 조각으로 대조만 한다")
    ap.add_argument("--resume", action="store_true", help="h40 --resume(설정 해시가 같은 완료 조각은 다시 적합하지 않는다)")
    ap.add_argument("--local-ref", default="", help="병기용 다른 로컬 재적합 'DIR:TAG'(예: 스레드 2 판). 판정에 쓰지 않는다")
    ap.add_argument("--load-start-max", type=float, default=64.0)
    a = ap.parse_args(argv)

    def ab(p):
        return Path(p) if os.path.isabs(str(p)) else ROOT / str(p)
    a.REF = ab(a.ref_dir); a.OUT = ab(a.out_dir); a.CSV = ab(a.gate_csv); a.SUM = ab(a.summary)
    return a


def sha256(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def versions():
    out = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, host=platform.node())
    for m in ("sklearn", "catboost", "scipy"):
        try:
            out[m] = __import__(m).__version__
        except Exception:                                                                            # noqa: BLE001
            out[m] = "NA"
    return out


def refit(a):
    """h40 으로 사전 점검 단위와 스모크 세 단위를 다시 적합한다(프로세스 1개, 스레드 --threads)."""
    common = ["--part", "cpu", "--tag", TAG, "--out-dir", str(a.OUT), "--threads", str(a.threads), "--workers", "0", "--allow-local",
              "--no-summarize"] + (["--resume"] if a.resume else [])
    t0 = time.time()
    res = {}
    for kind in ("precheck", "smoke"):
        av = [f"--{kind}"] + common
        print(f"[refit] h40 {' '.join(av)}", flush=True)
        r = H.main(av)
        res[kind] = dict(n_fail=int(r.get("n_fail", 0) or 0), executed=len(r.get("executed", [])), resumed=len(r.get("resumed", [])),
                         argv=av)
    return res, round(time.time() - t0, 1)


def shard_base(d: Path, tag: str, kind: str, target: str, mode: str, split: int):
    return d / f"{tag}_{kind}__cpu__{target}__{mode}__s{split}"


def compare_unit(pa: Path, pb: Path, target, mode, split, label):
    """두 조각의 공통 키 대조. 반환 (키별 행, 단위 요약)."""
    ua, ub = json.loads(Path(str(pa) + "_unit.json").read_text()), json.loads(Path(str(pb) + "_unit.json").read_text())
    nm = f"{target}|{mode}"
    sa = load_stores(Path(str(pa) + "_blocksse.npz"))[(nm, split)]
    sb = load_stores(Path(str(pb) + "_blocksse.npz"))[(nm, split)]
    keys = [k for k in sa.keys if k in sb]
    rows = []
    for k in keys:
        c = key_class(k)
        d_ = abs(float(sa.rmse(k)) - float(sb.rmse(k)))
        rows.append(dict(unit=label, target=target, mode=mode, split=split, method=k[0], learner=k[1], alpha=str(k[2]), placement=k[3],
                         n=int(k[4]), draw=int(k[5]), seed=int(k[6]), lam=float(k[7]), key=json.dumps([str(x) for x in k]), key_class=c,
                         in_base_scope=bool(in_base(k)), tol_cm=key_tol(c), abs_diff_cm=d_, over_tol=bool(d_ > key_tol(c))))
    summ = dict(unit=label, n_keys_local=len(sa.keys), n_keys_ref=len(sb.keys), n_common=len(keys),
                blocks_equal=bool(np.array_equal(sa.blocks, sb.blocks) and np.array_equal(sa.ncell, sb.ncell)),
                cfg_common_equal=bool(ua.get("cfg_common") == ub.get("cfg_common")), cfg_common=[ua.get("cfg_common"), ub.get("cfg_common")],
                code_sha=[ua.get("code_sha"), ub.get("code_sha")], threads=[ua.get("threads"), ub.get("threads")],
                status=[ua.get("status"), ub.get("status")], elapsed_s=[ua.get("elapsed_s"), ub.get("elapsed_s")])
    return rows, summ


def class_stats(df):
    out = {}
    for c in ("phys", "ridge", "catboost"):
        q = df[df.key_class == c]
        out[c] = dict(n_keys=int(len(q)), max_diff_cm=float(q.abs_diff_cm.max()) if len(q) else None,
                      median_diff_cm=float(q.abs_diff_cm.median()) if len(q) else None, n_over_tol=int(q.over_tol.sum()),
                      methods_over_tol=sorted(set(q[q.over_tol].method)))
    ml = df[df.key_class != "phys"]
    out["ml"] = dict(n_keys=int(len(ml)), max_diff_cm=float(ml.abs_diff_cm.max()) if len(ml) else None,
                     median_diff_cm=float(ml.abs_diff_cm.median()) if len(ml) else None, n_over_tol=int(ml.over_tol.sum()))
    return out


def verdict_of(st, blocks_ok, n_units, n_expected):
    phys_ok = bool(st["phys"]["n_keys"] > 0 and st["phys"]["n_over_tol"] == 0 and blocks_ok)
    ml_ok = bool(st["ml"]["n_over_tol"] == 0)
    v = "물리식 불통과" if not phys_ok else ("통과" if ml_ok else "물리식 통과, CatBoost·ridge 불통과")
    if n_units < n_expected:
        v += f"(부분 점검 {n_units}/{n_expected} 단위, 잠정)"
    return v, phys_ok, ml_ok


def main(argv=None):
    a = parse_args(argv)
    t0 = time.time()
    try:
        cur = os.nice(0)
        if cur < 10:
            os.nice(10 - cur)
    except OSError:
        pass
    refit_info, refit_s = {}, 0.0
    if a.compare_only and a.SUM.exists():                          # 대조만 다시 할 때는 앞선 재적합 기록을 보존한다
        old = json.loads(a.SUM.read_text())
        refit_info, refit_s = old.get("refit", {}), old.get("refit_elapsed_s", 0.0)
    if not a.compare_only:
        if not a.allow_local:
            raise SystemExit("[거부] 재적합은 --allow-local 이 있어야 한다(WRAPUP 8.2, 8.4)")
        la = os.getloadavg()[0]
        if la > a.load_start_max:
            raise SystemExit(f"[거부] 1분 load average {la:.1f} > {a.load_start_max}. 시작하지 않는다(WRAPUP 8.5)")
        refit_info, refit_s = refit(a)
    rows, units = [], []
    for kind, t, m, sp in UNITS:
        label = f"{kind}:{t}|{m}|s{sp}"
        pa = shard_base(a.OUT / "shards", TAG, kind, t, m, sp)
        pb = shard_base(a.REF, "lg", kind, t, m, sp)
        if not Path(str(pa) + "_unit.json").exists() or not Path(str(pb) + "_unit.json").exists():
            units.append(dict(unit=label, status="조각 없음", local=str(pa), ref=str(pb)))
            continue
        r_, s_ = compare_unit(pa, pb, t, m, sp, label)
        rows += r_; units.append(s_)
    df = pd.DataFrame(rows)
    done = [u for u in units if "n_common" in u]
    blocks_ok = all(u["blocks_equal"] for u in done) and bool(done)
    a.CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.CSV, index=False)
    summ = dict(check="xenv_i", plan="WRAPUP 8.4 (i), LG 개정 14", script="scripts/2_evaluation/lgw_xenv_gate_i.py", script_sha256=sha256(Path(__file__)),
                created=time.strftime("%Y-%m-%d %H:%M"), local=versions(), local_threads=a.threads, tag=TAG,
                ref="Rescale(qOkSo 사전 점검과 스모크 조각) python 3.11.10, numpy 2.4.6, scikit-learn 1.9.1, catboost 1.2.10, 스레드 4(WRAPUP 8.3)",
                scope=dict(units_expected=len(UNITS), units_done=len(done), units=[u["unit"] for u in done], threads=a.threads,
                           registered="8.4 (i): 네 단위, --threads 4, tag lgxe, 출력 data/processed/lgw/xenv/lg/"),
                tolerance=dict(phys_cm=TOL_PHYS, ridge_catboost_cm=TOL_ML),
                key_class_rule="물리식 = P0–P3(학습기 none), ridge = V1(학습기 none)과 ridge 학습기, CatBoost = catboost_lo",
                units=units, refit=refit_info, refit_elapsed_s=refit_s)
    if len(df):
        st_all = class_stats(df)
        st_base = class_stats(df[df.in_base_scope])
        v_all, p_all, m_all = verdict_of(st_all, blocks_ok, len(done), len(UNITS))
        v_base, p_base, m_base = verdict_of(st_base, blocks_ok, len(done), len(UNITS))
        v1 = df[df.method == "V1"]
        summ.update(n_common=int(len(df)), max_phys=st_all["phys"]["max_diff_cm"], max_ml=st_all["ml"]["max_diff_cm"],
                    median_ml=st_all["ml"]["median_diff_cm"], n_ml_over_002=st_all["ml"]["n_over_tol"],
                    by_class=st_all, by_class_base=st_base, blocks_equal=blocks_ok,
                    verdict=v_all, pass_phys=p_all, pass_ml=m_all, verdict_base=v_base, pass_phys_base=p_base, pass_ml_base=m_base,
                    verdict_rule="주 판정 = 8.4 의 등록 범위(공통 키 전부). 기준 방법 범위는 병기(판정에 쓰지 않는다)",
                    v1_keys=dict(n=int(len(v1)), max_diff_cm=float(v1.abs_diff_cm.max()) if len(v1) else None,
                                 max_by_unit={u: float(q.abs_diff_cm.max()) for u, q in v1.groupby("unit")}),
                    catboost_excl_v1r=dict(n=int(((df.key_class == "catboost") & (df.method != "V1r")).sum()),
                                           max_diff_cm=float(df[(df.key_class == "catboost") & (df.method != "V1r")].abs_diff_cm.max())
                                           if ((df.key_class == "catboost") & (df.method != "V1r")).any() else None),
                    by_method={f"{c}|{mth}": dict(n=int(len(q)), max_diff_cm=float(q.abs_diff_cm.max()), median_diff_cm=float(q.abs_diff_cm.median()),
                                                  n_over_tol=int(q.over_tol.sum())) for (c, mth), q in df.groupby(["key_class", "method"])},
                    over_tol_keys=df[df.over_tol][["unit", "key", "key_class", "abs_diff_cm"]].to_dict("records"),
                    by_unit={u: dict(n_keys=int(len(q)), max_phys=float(q[q.key_class == "phys"].abs_diff_cm.max()) if (q.key_class == "phys").any() else None,
                                     max_ml=float(q[q.key_class != "phys"].abs_diff_cm.max()) if (q.key_class != "phys").any() else None,
                                     n_over_tol=int(q.over_tol.sum())) for u, q in df.groupby("unit")})
    if a.local_ref:
        d, tg = a.local_ref.rsplit(":", 1)
        dd = Path(d) if os.path.isabs(d) else ROOT / d
        extra = []
        for kind, t, m, sp in UNITS:
            if not tg.endswith("_" + kind):                        # 같은 종류(사전 점검, 스모크)의 단위끼리만 대조한다
                continue
            pa = shard_base(a.OUT / "shards", TAG, kind, t, m, sp)
            pb = dd / f"{tg}__cpu__{t}__{m}__s{sp}"
            if Path(str(pa) + "_unit.json").exists() and Path(str(pb) + "_unit.json").exists():
                r_, s_ = compare_unit(pa, pb, t, m, sp, f"local-ref:{kind}:{t}|{m}|s{sp}")
                q = pd.DataFrame(r_)
                extra.append(dict(unit=s_["unit"], threads=s_["threads"], n_common=s_["n_common"], by_class=class_stats(q)))
        summ["local_ref"] = dict(source=a.local_ref, role="병기(판정에 쓰지 않는다): 같은 로컬 환경에서 스레드 수만 다른 재적합과의 차", units=extra)
    a.SUM.write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=str))
    print(f"[xenv-i] 단위 {len(done)}/{len(UNITS)} · 공통 키 {len(df)} · 판정 {summ.get('verdict', '대조 없음')} · 기준 방법 범위 "
          f"{summ.get('verdict_base', '')} · {time.time() - t0:.0f}s → {a.CSV.name}, {a.SUM.name}", flush=True)
    return summ


if __name__ == "__main__":
    main()
