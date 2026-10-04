"""SI 학습기·새 지역 근거 수치 추출. 원천 집계 표를 읽기만 하고 고치지 않는다.

실행: OMP_NUM_THREADS=1 python3 paper/claims/SI_learners_new_regions/extract_evidence.py [--copies]
  --copies 를 주면 원천 경로 대신 같은 폴더 tables/ 의 사본을 읽는다(sha256 은 tables/MANIFEST.csv).
출력: paper/claims/SI_learners_new_regions/evidence_values.csv (README 2절 표의 수치 원본)
가벼운 pandas·json 읽기만 한다(학습·적합·재표집 없음, 수 초).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "evidence_values.csv"

# 약호: (원천 경로, 사본 이름)
SRC = {
    "T": ("data/processed/lgt/lgt_tests.csv", "lgt_tests.csv"),
    "F": ("data/processed/lgf/lgf_tests.csv", "lgf_tests.csv"),
    "N": ("data/processed/lgf/lgfn_tests.csv", "lgfn_tests.csv"),
    "D": ("data/processed/lgd/lgd_tests_lic.csv", "lgd_tests_lic.csv"),
    "Dfull": ("data/processed/lgd/lgd_tests.csv", "lgd_tests.csv"),
    "Dpool": ("data/processed/lgd/lgd_pool_lic.csv", "lgd_pool_lic.csv"),
    "Delig": ("data/processed/lgd/lgd_eligibility_lic.csv", "lgd_eligibility_lic.csv"),
    "Drepro": ("data/processed/lgd/lgd_repro_gate.csv", "lgd_repro_gate.csv"),
    "Tgate": ("data/processed/lgt/lgt_gate.csv", "lgt_gate.csv"),
    "Fgate": ("data/processed/lgf/lgf_gate.csv", "lgf_gate.csv"),
    "Ncross": ("data/processed/lgf/lgfn_cross.csv", "lgfn_cross.csv"),
    "Ngate": ("data/processed/lgf/lgfn_cross_gate.csv", "lgfn_cross_gate.csv"),
    "WF9": ("data/processed/wf/wf9_repro_check.csv", "wf9_repro_check.csv"),
    "XC14": ("data/processed/lgw/lgw_xenv_c14_summary.json", "lgw_xenv_c14_summary.json"),
    "XI": ("data/processed/lgw/lgw_xenv_gate_i_summary.json", "lgw_xenv_gate_i_summary.json"),
    "XIII": ("data/processed/lgw/lgw_xenv_gate_iii_summary.json", "lgw_xenv_gate_iii_summary.json"),
    "Fig6a": ("outputs/figures/paper/source_data/v2/Fig6_a.csv", "Fig6_a.csv"),
    "Tab1": ("outputs/figures/paper/source_data/v2/Table1_rows.csv", "Table1_rows.csv"),
}

VSEC = {
    "T": "LG 7.4 (J4); 가설 6C.6",
    "F": "LGF 10.2; 가설 3.6",
    "N": "LGF 10.3; 가설 4.5",
    "D": "LG 7.3 (J5); 가설 6B.5",
    "Dfull": "LG 7.3 전체 판 열(SI); 개정 15 (m)2",
}

rows: list[dict] = []
_cache: dict = {}


def path_of(key: str, copies: bool) -> Path:
    orig, copy = SRC[key]
    return (HERE / "tables" / copy) if copies else (ROOT / orig)


def load(key: str, copies: bool):
    if key not in _cache:
        p = path_of(key, copies)
        _cache[key] = json.loads(p.read_text()) if p.suffix == ".json" else pd.read_csv(p)
    return _cache[key]


def r2(x):
    try:
        return f"{float(x):.2f}"
    except (TypeError, ValueError):
        return str(x)


def one(df: pd.DataFrame, **flt) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for k, v in flt.items():
        if isinstance(v, float) and pd.isna(v):
            m &= df[k].isna()
        else:
            m &= (df[k] == v)
    sub = df[m]
    if len(sub) != 1:
        raise SystemExit(f"행이 1개가 아니다({len(sub)}): {flt}")
    return sub.iloc[0]


def fstr(flt: dict) -> str:
    return "; ".join(f"{k}={v}" for k, v in flt.items())


def add(eid, src, filt, col, value, raw=None, verdict="", vsec="", note=""):
    rows.append(dict(id=eid, source=src, source_path=SRC[src][0], row_filter=filt, column=col,
                     value=value, value_raw=("" if raw is None else raw), verdict=verdict,
                     verdict_section=vsec or VSEC.get(src, ""), note=note))


def contrast_row(eid, src, df, vcol="verdict4", note="", **flt):
    """Δ 와 두 가중 CI, 4분 판정을 한 행으로 기록한다."""
    r = one(df, **flt)
    val = (f"{r2(r['delta'])} [{r2(r['ci_lo'])}, {r2(r['ci_hi'])}]; 블록 등가중 "
           f"{r2(r['delta_blockeq'])} [{r2(r['ci_lo_beq'])}, {r2(r['ci_hi_beq'])}]")
    extra = []
    if "holm_p" in r.index and pd.notna(r.get("holm_p")):
        extra.append(f"holm_p {r2(r['holm_p'])}")
    if "small_note" in r.index and isinstance(r.get("small_note"), str):
        extra.append(r["small_note"])
    if "flags" in r.index and isinstance(r["flags"], str):
        extra.append(f"flags: {r['flags']}")
    if "pool_regions" in r.index and isinstance(r.get("pool_regions"), str):
        extra.append(f"pool_regions: {r['pool_regions']}")
    nt = "; ".join([x for x in [note] + extra if x])
    add(eid, src, fstr(flt), "delta, ci_lo, ci_hi, delta_blockeq, ci_lo_beq, ci_hi_beq", val,
        raw=f"{r['delta']!r}", verdict=str(r.get(vcol, "")), note=nt)
    return r


def text_row(eid, src, df, col="verdict", note="", **flt):
    r = one(df, **flt)
    add(eid, src, fstr(flt), col, str(r[col]), verdict=str(r[col])[:60], note=note)
    return r


def main(copies: bool) -> None:
    # ------------------------------------------------------------------ LGT (TabPFN v2)
    T = load("T", copies)
    for n in ["n0", "n10", "n40", "n160", "nall"]:
        contrast_row(f"T-L32-{n}", "T", T, test_id="L32", scope="MEAN", contrast=f"R1[T]-R1[C]|{n}|lam0.25")
    for n in ["n0", "n10", "n40", "n160", "nall"]:
        contrast_row(f"T-L32-lam1-{n}", "T", T, test_id="L32", scope="MEAN", contrast=f"R1[T]-R1[C]|{n}|lam1.0")
    for c in ["D0[T]-D0[C]|n10", "D0[T]-D0[C]|nall"]:
        contrast_row(f"T-L32-aux-{c}", "T", T, test_id="L32", scope="MEAN", contrast=c)
    text_row("T-L32-V", "T", T, test_id="L32", scope="verdict_aux")
    for n in ["n3", "n10", "n40", "n160", "n320", "nall"]:
        contrast_row(f"T-L33-{n}", "T", T, test_id="L33", scope="MEAN", contrast=f"R1[T]-P1|{n}")
    for n in ["n3", "n10", "n40", "nall"]:
        contrast_row(f"T-L33C-{n}", "T", T, test_id="L33", scope="MEAN", contrast=f"R1[C]-P1|{n}")
    text_row("T-L33-V", "T", T, test_id="L33", scope="verdict_aux")
    contrast_row("T-L34a", "T", T, test_id="L34", scope="MEAN", contrast="D0[T]-P0|n0")
    contrast_row("T-L34a-C", "T", T, test_id="L34", scope="MEAN", contrast="D0[C]-P0|n0")
    contrast_row("T-L34b", "T", T, test_id="L34", scope="MEAN", contrast="R0[T]-P0|n0")
    contrast_row("T-L34b-C", "T", T, test_id="L34", scope="MEAN", contrast="R0[C]-P0|n0")
    text_row("T-L34a-V", "T", T, test_id="L34", scope="verdict_aux", clause="(a) 직접 예측")
    text_row("T-L34b-V", "T", T, test_id="L34", scope="verdict_aux", clause="(b) 잔차(방향 중립)")
    contrast_row("T-L35", "T", T, test_id="L35", scope="MEAN", contrast="R1[T]-P0|all")
    contrast_row("T-L35-C", "T", T, test_id="L35", scope="MEAN", contrast="R1[C]-P0|all")
    contrast_row("T-L35-D0", "T", T, test_id="L35", scope="MEAN", contrast="D0[T]-P0|all")
    for n in ["n0", "n10", "n40", "n160", "nall"]:
        contrast_row(f"T-L36-{n}", "T", T, test_id="L36", scope="MEAN", contrast=f"D0[T]-R1[T]|{n}")
    for v in ["d3", "d10", "rid", "tgt"]:
        text_row(f"T-L37-V-{v}", "T", T, test_id="L37", scope="verdict_aux", clause="안정성", variant=v)
    contrast_row("T-M3-L34a", "T", T, test_id="L34", scope="MEAN3", contrast="D0[T]-P0|n0", note="3지역 보조 열(레나·캐나다·Alaska x)")
    contrast_row("T-M3-L35", "T", T, test_id="L35", scope="MEAN3", contrast="R1[T]-P0|all", note="3지역 보조 열")
    contrast_row("T-M3-L33-n10", "T", T, test_id="L33", scope="MEAN3", contrast="R1[T]-P1|n10", note="3지역 보조 열")

    # ------------------------------------------------------------------ LGF-F (TabICL v2)
    F = load("F", copies)
    for c in ["D0[I]-P0|n0", "D0@full[I]-P0|n0", "D0[C]-P0|n0", "D0@full[C]-P0|n0"]:
        contrast_row(f"F-F1-{c}", "F", F, test_id="LGF-F1", scope="MEAN", contrast=c)
    text_row("F-F1-V", "F", F, test_id="LGF-F1", scope="verdict")
    for n in ["n3", "n10", "n40"]:
        contrast_row(f"F-F1a-{n}", "F", F, test_id="LGF-F1", scope="MEAN", contrast=f"D0[I]-P0|{n}")
    for n in ["n0", "n10", "n40"]:
        contrast_row(f"F-F1-M3-{n}", "F", F, test_id="LGF-F1", scope="MEAN3", contrast=f"D0[I]-P0|{n}", note="3지역 보조 열(레나·캐나다·Alaska x)")
    text_row("F-F1a-V", "F", F, test_id="LGF-F1", scope="verdict_aux", clause="(a) n ∈ {3, 10, 40}")
    text_row("F-F1b-V", "F", F, test_id="LGF-F1", scope="verdict_aux", clause="(b) L1 형식 지역 수", col="verdict")
    rw = contrast_row("F-F1-RW-n10", "F", F, test_id="LGF-F1", scope="region", contrast="D0[I]-P0|n10", target="Russia_W|x",
                      note="지역 행")
    for tg in ["Lena", "Canada", "Russia_E"]:
        contrast_row(f"F-F1-{tg}-n10", "F", F, test_id="LGF-F1", scope="region", contrast="D0[I]-P0|n10",
                     target=f"{tg}|x", note="지역 행")
    contrast_row("F-F1-AK-n0", "F", F, test_id="LGF-F1", scope="region", contrast="D0[I]-P0|n0", target="Alaska|x", note="지역 행")
    for n in ["n0", "n10", "n40", "n160", "nall"]:
        contrast_row(f"F-F2-{n}", "F", F, test_id="LGF-F2", scope="MEAN", contrast=f"R1[I]-R1[C]|{n}|lam0.25", vcol="verdict4",
                     note="")
        r = one(F, test_id="LGF-F2", scope="MEAN", contrast=f"R1[I]-R1[C]|{n}|lam0.25")
        rows[-1]["note"] = (rows[-1]["note"] + f"; verdict4_aux {r['verdict4_aux']}").strip("; ")
    for c in ["D0[I]-D0[C]|n10", "D0[I]-D0[C]|nall", "R0[I]-R0[C]|n10", "R0[I]-R0[C]|nall"]:
        contrast_row(f"F-F2-aux-{c}", "F", F, test_id="LGF-F2", scope="MEAN", contrast=c)
    text_row("F-F2-V", "F", F, test_id="LGF-F2", scope="verdict_aux")
    contrast_row("F-F3-D0-n0", "F", F, test_id="LGF-F3", scope="MEAN", contrast="D0[I]-D0[T]|n0")
    for n in ["n0", "n10", "nall"]:
        contrast_row(f"F-F3-R1-{n}", "F", F, test_id="LGF-F3", scope="MEAN", contrast=f"R1[I]-R1[T]|{n}|lam0.25")
    contrast_row("F-F3-R1-lam1-n0", "F", F, test_id="LGF-F3", scope="MEAN", contrast="R1[I]-R1[T]|n0|lam1.0")
    text_row("F-F3-V", "F", F, test_id="LGF-F3", scope="verdict_aux")
    for n in ["n3", "n10", "n40", "n160", "n320", "nall"]:
        contrast_row(f"F-F4-{n}", "F", F, test_id="LGF-F4", scope="MEAN", contrast=f"R1[I]-P1|{n}")
    contrast_row("F-F4-P0-all", "F", F, test_id="LGF-F4", scope="MEAN", contrast="R1[I]-P0|all")
    for n in ["n10", "n40"]:
        contrast_row(f"F-F4-M3-{n}", "F", F, test_id="LGF-F4", scope="MEAN3", contrast=f"R1[I]-P1|{n}", note="3지역 보조 열")
    contrast_row("F-F4-M3-P0-all", "F", F, test_id="LGF-F4", scope="MEAN3", contrast="R1[I]-P0|all", note="3지역 보조 열")
    text_row("F-F4-V1", "F", F, test_id="LGF-F4", scope="verdict_aux", clause="순가치")
    text_row("F-F4-V2", "F", F, test_id="LGF-F4", scope="verdict_aux", clause="전량 R1[I] − P0(L35 형식)")
    contrast_row("F-F5", "F", F, test_id="LGF-F5", scope="MEAN", contrast="R0[I]-P0|n0")
    contrast_row("F-F5-C", "F", F, test_id="LGF-F5", scope="MEAN", contrast="R0[C]-P0|n0")
    contrast_row("F-F5-M3", "F", F, test_id="LGF-F5", scope="MEAN3", contrast="R0[I]-P0|n0", note="3지역 보조 열")
    text_row("F-F5-V", "F", F, test_id="LGF-F5", scope="verdict_aux")
    for n in ["n0", "n10", "nall"]:
        contrast_row(f"F-F6-R1-{n}", "F", F, test_id="LGF-F6", scope="MEAN", contrast=f"R1@full[I]-R1[I]|{n}|lam0.25")
    contrast_row("F-F6-D0-nall", "F", F, test_id="LGF-F6", scope="MEAN", contrast="D0@full[I]-D0[I]|nall")
    text_row("F-F6-V", "F", F, test_id="LGF-F6", scope="verdict_aux")
    text_row("F-fam-1", "F", F, test_id="LGF-family", scope="verdict_aux", clause="라벨 0 직접 예측")
    text_row("F-fam-2", "F", F, test_id="LGF-family", scope="verdict_aux", clause="학습기 동등성")
    # 이번 점검(사후, 판정 아님): 러시아 W n = 10 의 TabICL 직접 RMSE 와 P1 RMSE(같은 LGF 조각)
    p1 = one(F, test_id="LGF-F4", scope="region", contrast="R1[I]-P1|n10", target="Russia_W|x")
    add("F-posthoc-RW-D0-P1", "F",
        "A = rmse_A of (test_id=LGF-F1; scope=region; contrast=D0[I]-P0|n10; target=Russia_W|x); "
        "B = rmse_B of (test_id=LGF-F4; scope=region; contrast=R1[I]-P1|n10; target=Russia_W|x)",
        "rmse_A, rmse_B (A − B)",
        f"D0[I] {r2(rw['rmse_A'])}, P1 {r2(p1['rmse_B'])}, 차 {r2(rw['rmse_A'] - p1['rmse_B'])}",
        raw=f"{rw['rmse_A'] - p1['rmse_B']!r}", verdict="(판정 아님)", vsec="이번 점검(사후, 2026-10-04)",
        note="점 추정 차이만. CI 와 판정이 없다")

    # ------------------------------------------------------------------ LGF-N (조정 신경망)
    N = load("N", copies)
    for l in ["mlp", "tabm", "ftt", "realmlp"]:
        contrast_row(f"N-N1-{l}", "N", N, test_id="LGF-N1", scope="MEAN", contrast=f"D0[{l}*]-P0|n0")
    for l in ["mlp", "tabm", "ftt", "realmlp", "catboost_lo"]:
        contrast_row(f"N-N1aux-{l}", "N", N, test_id="LGF-N1", scope="MEAN", contrast=f"D0[{l}]-P0|n0",
                     note="로컬 기본판(조정 전)" if l != "catboost_lo" else "로컬 catboost_lo 병기")
    text_row("N-N1-V", "N", N, test_id="LGF-N1", scope="verdict")
    for tid in ["LGF-N2", "LGF-N2s", "LGF-N3"]:
        for l in ["mlp", "tabm", "ftt", "realmlp"]:
            text_row(f"N-{tid[4:]}-V-{l}", "N", N, test_id=tid, scope="verdict_aux", learner=l)
    for tid in ["LGF-N2", "LGF-N2s"]:
        text_row(f"N-{tid[4:]}-V-all", "N", N, test_id=tid, scope="verdict_aux", clause="전체")
    r = one(N, test_id="LGF-N4", contrast="원천 CV 이득(seed 1·2) 대 대상 Δ(l* − l, n = 0)의 순위 상관")
    add("N-N4-rho", "N", "test_id=LGF-N4; contrast=원천 CV 이득(seed 1·2) 대 대상 Δ(l* − l, n = 0)의 순위 상관",
        "spearman, n_points", f"{r2(r['spearman'])} (점 {int(r['n_points'])}개)", raw=f"{r['spearman']!r}",
        verdict="서술(검정 없음)")

    # ------------------------------------------------------------------ LGD (약관 확인분 판, 주 판정)
    D = load("D", copies)
    for pool in ["P4", "PE1", "PE2"]:
        r = one(D, test_id="L1e", pool=pool, scope="verdict")
        add(f"D-L1e-V-{pool}", "D", f"test_id=L1e; pool={pool}; scope=verdict", "verdict, stat",
            f"{r['verdict']} | {r['stat']}", verdict=r["verdict"])
    for tid in ["L1e", "L4e", "L8e"]:
        r = one(D, test_id=tid, pool="P4·PE1·PE2", scope="compare")
        add(f"D-{tid}-compare", "D", f"test_id={tid}; pool=P4·PE1·PE2; scope=compare", "verdict, stat",
            f"{r['verdict']} | {r['stat']}", verdict=str(r["verdict"])[:60])
    for pool in ["P4", "PE1", "PE2"]:
        r = one(D, test_id="L4e", pool=pool, scope="verdict")
        add(f"D-L4e-V-{pool}", "D", f"test_id=L4e; pool={pool}; scope=verdict", "verdict, stat",
            f"{r['verdict']} | {r['stat']}", verdict=r["verdict"], note=(r["flags"] if isinstance(r["flags"], str) else ""))
    contrast_row("D-L4e-PE1-n10", "D", D, vcol="sig", test_id="L4e", pool="PE1", item="R1-P1", scope="MEAN", n=10.0)
    contrast_row("D-L4e-PE2-n10", "D", D, vcol="sig", test_id="L4e", pool="PE2", item="R1-P1", scope="MEAN", n=10.0)
    contrast_row("D-L4e-PE2-R1P2-n10", "D", D, vcol="sig", test_id="L4e", pool="PE2", item="R1-P2", scope="MEAN", n=10.0)
    for pool, tgt_n in [("P4", 10.0), ("PE1", 10.0), ("PE1-new", 10.0), ("PE1-new", -1.0)]:
        it = "(a)1 풀 대비 R1-P1|n10" if tgt_n == 10.0 else "(a)1 풀 대비 R1-P1|nall"
        contrast_row(f"D-L4e-pool-{pool}-{int(tgt_n)}", "D", D, test_id="L4e", pool=pool, item=it, scope="compare_pool")
    for pool in ["P4", "PE1", "PE2"]:
        contrast_row(f"D-L8e-{pool}", "D", D, vcol="sig", test_id="L8e", pool=pool, item="R1-P0", scope="MEAN")
    r = one(D, test_id="L8e", pool="PE2", scope="MEAN_rel")
    add("D-L8e-PE2-rel", "D", "test_id=L8e; pool=PE2; scope=MEAN_rel", "delta, ci_lo, ci_hi",
        f"{r2(r['delta'])} [{r2(r['ci_lo'])}, {r2(r['ci_hi'])}] (상대 단위)", raw=f"{r['delta']!r}", verdict=str(r["sig"]))
    # L38 새 지역
    tib = {"(a) D0-P0": [0.0, 3.0, 10.0, 40.0], "(b) R1-P0": [-1.0], "(c) P1-P0": [10.0], "(c) R1-P1": [10.0],
           "(c) R1-P2": [10.0], "(c) R1-P3": [10.0], "(c) P2-P1": [10.0]}
    for it, ns in tib.items():
        for n in ns:
            r = contrast_row(f"D-L38-Tibet-{it.split()[1]}-n{int(n)}", "D", D, test_id="L38", pool="새 지역",
                             item=it, target="Tibet_LGD|x", n=n)
            rows[-1]["note"] = (rows[-1]["note"] + f"; RMSE A {r2(r['rmse_A'])}, B {r2(r['rmse_B'])}").strip("; ")
    rc = {"(a) D0-P0": [0.0, 3.0, 10.0, -1.0], "(b) R1-P0": [-1.0], "(c) P1-P0": [10.0], "(c) R1-P1": [10.0]}
    for it, ns in rc.items():
        for n in ns:
            r = contrast_row(f"D-L38-RussiaC-{it.split()[1]}-n{int(n)}", "D", D, test_id="L38", pool="새 지역",
                             item=it, target="Russia_C|x", n=n)
            rows[-1]["note"] = (rows[-1]["note"] + f"; RMSE A {r2(r['rmse_A'])}, B {r2(r['rmse_B'])}").strip("; ")
    for it, n in [("(b) R1-P0", -1.0), ("(c) P1-P0", 10.0), ("(c) R1-P1", 10.0)]:
        contrast_row(f"D-L38-RussiaW-ref-{it.split()[1]}-n{int(n)}", "D", D, test_id="L38", pool="P4 참조",
                     item=it, target="Russia_W|x", n=n, note="P4 참조 행(LG 본 실행 CatBoost)")
    text_row("D-L38-V", "D", D, test_id="L38", pool="새 지역", scope="verdict")
    r = one(D, test_id="L38", pool="새 지역", scope="verdict")
    add("D-L38-V-note", "D", "test_id=L38; pool=새 지역; scope=verdict", "stat; note", f"{r['stat']} | {r['note']}")
    text_row("D-L39-V", "D", D, test_id="L39", scope="verdict")
    contrast_row("D-L39-tempderived-R1P1-n10", "D", D, test_id="L39", item="R1-P1|n10", target="Tibet_LGD~L39|x",
                 note="지온 유도 라벨 집합(위치가 다르다)")
    def verdict_stat(eid, tid, tgt):
        r = one(D, test_id=tid, scope="verdict", target=tgt)
        st = r["stat"] if isinstance(r["stat"], str) else ""
        add(eid, "D", f"test_id={tid}; scope=verdict; target={tgt}", "verdict, stat",
            f"{r['verdict']} | {st}".rstrip(" |"), verdict=str(r["verdict"])[:60])
    for tgt in ["Russia_W~exp~lic|x", "Russia_E~exp|x", "Canada~exp~lic|x", "Russia_E~expnokyt|x"]:
        verdict_stat(f"D-L40-V-{tgt.split('~')[0]}{'-nokyt' if 'nokyt' in tgt else ''}", "L40", tgt)
    for tgt in ["Tibet_LGD~L41b|x", "Tibet_LGD~L41a|x", "Russia_C~L41b|x", "Russia_C~L41e|x", "Russia_C~L41c|x",
                "Tibet_LGD~L42~lic|x", "Russia_C~L42~lic|x", "NAtlantic~L42~lic|x"]:
        tid = "L42" if "L42" in tgt else "L41"
        verdict_stat(f"D-{tid}-V-{tgt.split('|')[0]}", tid, tgt)
    P = load("Dpool", copies)
    for pool in ["PE1", "PE2", "point_only"]:
        r = one(P, pool=pool)
        add(f"D-pool-{pool}", "Dpool", f"pool={pool}", "n_regions, regions, cross_env, version_role",
            f"{r['n_regions']} | {r['regions']} | {r['cross_env']} | {r['version_role']}", vsec="LG 7.3; 개정 15 (r)")
    E = load("Delig", copies)
    for spec in ["NAtlantic_lic", "Tibet_L42_lic", "Russia_C_L42_lic"]:
        r = one(E, spec=spec)
        add(f"D-elig-{spec}", "Delig", f"spec={spec}", "n_cells, nb_union, min_nb_eval_used, eligible, regime",
            f"{r['n_cells']} | {r['nb_union']} | {r['min_nb_eval_used']} | {r['eligible']} | {r['regime']}",
            vsec="LG 6B.4 적격 규칙; 개정 15 (r)")
    # 전체 판(SI): NAtlantic
    DF = load("Dfull", copies)
    for it, n in [("(a) D0-P0", 0.0), ("(a) D0-P0", 10.0), ("(b) R1-P0", -1.0), ("(c) P1-P0", 10.0), ("(c) R1-P1", 10.0)]:
        r = contrast_row(f"Dfull-L38-NAtl-{it.split()[1]}-n{int(n)}", "Dfull", DF, test_id="L38", pool="새 지역",
                         item=it, target="NAtlantic|x", n=n, note="전체 판(약관 확인 전 자료 포함), SI 전용")
        rows[-1]["note"] = (rows[-1]["note"] + f"; RMSE A {r2(r['rmse_A'])}, B {r2(r['rmse_B'])}").strip("; ")
    r = one(DF, test_id="L38", pool="새 지역", scope="verdict")
    add("Dfull-L38-V", "Dfull", "test_id=L38; pool=새 지역; scope=verdict", "verdict", r["verdict"], verdict=r["verdict"][:60])
    for tid in ["L1e", "L8e"]:
        r = one(DF, test_id=tid, pool="PE1", scope="verdict")
        add(f"Dfull-{tid}-V-PE1", "Dfull", f"test_id={tid}; pool=PE1; scope=verdict", "verdict, stat",
            f"{r['verdict']} | {r['stat']}", verdict=r["verdict"])
    r = one(DF, test_id="L4e", pool="P4·PE1·PE2", scope="compare")
    add("Dfull-L4e-compare", "Dfull", "test_id=L4e; pool=P4·PE1·PE2; scope=compare", "verdict, stat",
        f"{r['verdict']} | {r['stat']}")

    # ------------------------------------------------------------------ 계산 환경 간 재현성
    x = load("XC14", copies)
    p = x["by_pair"]["local1_vs_rescale"]
    add("R-C14", "XC14", "by_pair.local1_vs_rescale", "ml.n_keys, ml.max_cm, phys.n_keys, phys.max_cm, ml_n_over_eq",
        f"학습 키 {p['ml']['n_keys']}개 최대 {p['ml']['max_cm']:.1e} cm; 물리식 키 {p['phys']['n_keys']}개 최대 "
        f"{p['phys']['max_cm']:.1e} cm; 0.5 cm 초과 {p['ml_n_over_eq']}; 단위 {len(x['units'])}개",
        raw=repr(p["ml"]["max_cm"]), vsec="LG 개정 15 (t); EXECUTION_PLAN 10 (12:57 C14)")
    x = load("XI", copies)
    bc = x["by_class"]
    add("R-gate-i", "XI", "verdict; by_class; catboost_excl_v1r; by_method['catboost|V1r']", "max_diff_cm",
        f"판정 '{x['verdict']}'; CatBoost {bc['catboost']['n_keys']}키 최대 {bc['catboost']['max_diff_cm']:.2f} cm"
        f"(허용 차 초과 {bc['catboost']['n_over_tol']}키, 모두 V1r); V1r 제외 CatBoost "
        f"{x['catboost_excl_v1r']['n']}키 최대 {x['catboost_excl_v1r']['max_diff_cm']:.1e} cm; ridge "
        f"{bc['ridge']['n_keys']}키 최대 {bc['ridge']['max_diff_cm']:.4f} cm; 물리식 {bc['phys']['n_keys']}키 최대 "
        f"{bc['phys']['max_diff_cm']:.1e} cm", raw=repr(bc["catboost"]["max_diff_cm"]), verdict=x["verdict"],
        vsec="LG 개정 14 (점검 (i)); WRAPUP 8.4")
    x = load("XIII", copies)
    p = x["by_pair"]
    add("R-gate-iii", "XIII", "by_pair.local1_vs_rescale; by_pair.local1_vs_local2; rule", "ml.median_cm, ml.max_cm, ml_n_over_eq",
        f"FT-T {p['local1_vs_rescale']['ml']['n_keys']}키: 로컬 − Rescale 중앙값 {p['local1_vs_rescale']['ml']['median_cm']:.3f} cm, "
        f"최대 {p['local1_vs_rescale']['ml']['max_cm']:.2f} cm, 0.5 cm 초과 {p['local1_vs_rescale']['ml_n_over_eq']}키"
        f"({100 * p['local1_vs_rescale']['ml_frac_over_eq']:.1f} %); 로컬 반복 차 최대 {p['local1_vs_local2']['ml']['max_cm']}; "
        f"물리식 {p['local1_vs_rescale']['phys']['n_keys']}키 최대 {p['local1_vs_rescale']['phys']['max_cm']:.1e} cm; 규칙 '{x['rule']}'",
        raw=repr(p["local1_vs_rescale"]["ml"]["max_cm"]), verdict=x["rule"], vsec="LG 개정 15 (u)")
    C = load("Ncross", copies).copy()
    C["learner"] = C["contrast"].str.extract(r"\[(\w+)\]", expand=False)
    med = C.groupby("learner")["delta"].apply(lambda s: s.abs().median())
    i = C["delta"].abs().idxmax()
    add("R-lgfn-cross", "Ncross", "전 행(scope region 128 + MEAN 40); learner = contrast 의 첫 대괄호",
        "|delta| 중앙값(학습기별), |delta| 최댓값",
        "; ".join(f"{k} {med[k]:.2f}" for k in ["ftt", "mlp", "tabm", "realmlp"])
        + f"; 최대 {abs(C.loc[i, 'delta']):.2f} cm({C.loc[i, 'contrast']}, {C.loc[i, 'target']}, scope {C.loc[i, 'scope']})",
        raw=repr(abs(C.loc[i, "delta"])), verdict="교차 환경(보조), 판정에 쓰지 않음", vsec="LGF 10.3 교차 환경 행")
    W = load("WF9", copies)
    add("R-wf9", "WF9", "전 행", "ok, max_abs_dsse, n_common, n_count_mismatch, same_blocks",
        f"조각 {len(W)}개 중 ok {int(W['ok'].sum())}; 최대 |ΔSSE| {W['max_abs_dsse'].max()}; 공통 키 {int(W['n_common'].sum())}; "
        f"셀 수 불일치 {int(W['n_count_mismatch'].sum())}; 채점 블록 같음 {bool(W['same_blocks'].all())}",
        raw=repr(W["max_abs_dsse"].max()), vsec="WF 5.3 재현 점검(열람 전)")
    G = load("Tgate", copies)
    add("R-lgt-gate", "Tgate", "전 행", "passed, max_diff, n_missing_ref",
        f"단위 {len(G)}개 중 통과 {int(G['passed'].sum())}; P0·P1 최대 차 {G['max_diff'].max():.1e} cm; n_missing_ref 합 {int(G['n_missing_ref'].sum())}",
        raw=repr(G["max_diff"].max()), vsec="LG 6C.8; 7.4")
    G = load("Fgate", copies)
    bad = G[~G["passed"]]
    add("R-lgf-gate", "Fgate", "전 행; 통과 여부(passed)", "passed, status, n_sha_mismatch/n_sha_keys, p01_max_diff, cb_max_diff",
        f"단위 {len(G)}개 중 통과 {int(G['passed'].sum())}; 불통과 대상 {sorted(bad['target'].unique().tolist())}; "
        f"불통과 단위의 ctx_sha 불일치 비율 {(bad['n_sha_mismatch'] / bad['n_sha_keys']).min():.3f}–{(bad['n_sha_mismatch'] / bad['n_sha_keys']).max():.3f}; "
        f"P0·P1 최대 차 {G['p01_max_diff'].max():.1e} cm; 통과 단위 catboost_ctx 최대 차 {G.loc[G['passed'], 'cb_max_diff'].max()}",
        raw=repr(int(G["passed"].sum())), vsec="LGF 10.1")
    G = load("Ngate", copies)
    add("R-lgfn-gate", "Ngate", "전 행", "passed, max_diff",
        f"단위 {len(G)}개 중 통과 {int(G['passed'].sum())}; 물리식 키 최대 차 {G['max_diff'].max():.1e} cm",
        raw=repr(G["max_diff"].max()), vsec="LGF 10.1; DISPLAY_ITEMS 6절 10번")
    G = load("Drepro", copies)
    for sc in ["base", "all"]:
        r = one(G, scope=sc)
        add(f"R-lgd-repro-{sc}", "Drepro", f"scope={sc}", "status, n_keys_catboost, max_diff_phys, max_diff_catboost, n_ml_over_tol, methods_ml_over_tol",
            f"{r['status']} | CatBoost {r['n_keys_catboost']}키 최대 "
            f"{(format(r['max_diff_catboost'], '.2f') if r['max_diff_catboost'] >= 0.01 else format(r['max_diff_catboost'], '.2e'))} cm | "
            f"물리식 최대 {r['max_diff_phys']:.1e} cm | "
            f"허용 차 초과 {r['n_ml_over_tol']}키 ({r['methods_ml_over_tol']})",
            raw=repr(r["max_diff_catboost"]), vsec="LG 7.3 원천; 개정 14 (c)2")

    # ------------------------------------------------------------------ 그림 원천(대조용)
    A = load("Fig6a", copies)
    for blk in ["lgt", "lgf_f"]:
        sub = A[A["block"] == blk]
        add(f"Fig6a-{blk}", "Fig6a", f"block={blk}", "key, variant, contrast, delta, verdict4",
            "; ".join(f"{k}/{v}: {c} {r2(d)} {vv}" for k, v, c, d, vv in
                      zip(sub["key"], sub["variant"], sub["contrast"], sub["delta"], sub["verdict4"])),
            vsec="Fig 6a v2 원천 자료")
    sub = A[A["block"].astype(str).str.startswith("lgf_n")]
    add("Fig6a-lgf_n", "Fig6a", "block startswith lgf_n", "key, variant, contrast, delta, verdict4",
        "; ".join(f"{k}/{v}: {c} {r2(d)} {vv}" for k, v, c, d, vv in
                  zip(sub["key"], sub["variant"], sub["contrast"], sub["delta"], sub["verdict4"])),
        vsec="Fig 6a v2 원천 자료")
    B = load("Tab1", copies)
    sub = B[B["group"] == "New region (LGD)"]
    add("Tab1-LGD", "Tab1", "group=New region (LGD)", "target, role, label_rows, blocks, licence, edition",
        "; ".join(f"{t}: {ro}, 행 {lr}, 블록 {bl}, {li}, {ed}" for t, ro, lr, bl, li, ed in
                  zip(sub["target"], sub["role"], sub["label_rows"], sub["blocks"], sub["licence"], sub["edition"])),
        vsec="Table 1 v2 원천 자료")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"{len(out)} 행 → {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", action="store_true", help="tables/ 사본을 읽는다")
    main(ap.parse_args().copies)
