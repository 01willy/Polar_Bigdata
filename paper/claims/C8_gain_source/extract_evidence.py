"""C8 근거 수치 추출(읽기 전용).

tables/ 의 사본에서 C8 README 의 근거 표 값을 읽어 evidence_c8.csv 로 쓰고,
tables/MANIFEST.csv(원본 경로, 사본 경로, sha256, 행 수, 바이트)를 만든다.
원본 파일은 읽기만 한다. 실행: OMP_NUM_THREADS=1 python paper/claims/C8_gain_source/extract_evidence.py
"""
import csv
import hashlib
import os

import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HERE = os.path.relpath(os.path.dirname(os.path.abspath(__file__)), ROOT)
T = os.path.join(HERE, "tables")

COPIES = [  # (원본, 사본)
    ("results/rescale_wf3/data/processed/wf/wf3b_tests.csv", "rescale_wf3__wf3b_tests.csv"),
    ("results/rescale_wf3/data/processed/wf/wf3b_decomp.csv", "rescale_wf3__wf3b_decomp.csv"),
    ("results/rescale_wf3/data/processed/wf/wf3b_targets.csv", "rescale_wf3__wf3b_targets.csv"),
    ("results/rescale_wf3/data/processed/wf/wf3b_meta.json", "rescale_wf3__wf3b_meta.json"),
    ("data/processed/wf_soil/wf3b_tests.csv", "wf_soil__wf3b_tests.csv"),
    ("data/processed/wf_soil/wf3b_decomp.csv", "wf_soil__wf3b_decomp.csv"),
    ("data/processed/wf_soil/wf3b_meta.json", "wf_soil__wf3b_meta.json"),
    ("data/processed/wf/wf9_repro_check.csv", "wf__wf9_repro_check.csv"),
    ("data/processed/wf/wf0_misspec.csv", "wf__wf0_misspec.csv"),
]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def manifest():
    rows = []
    for orig, cp in COPIES:
        o, c = os.path.join(ROOT, orig), os.path.join(ROOT, T, cp)
        so, sc = sha(o), sha(c)
        assert so == sc, f"사본이 원본과 다르다: {cp}"
        nrow = len(pd.read_csv(c)) if c.endswith(".csv") else "NA"
        rows.append(dict(orig_path=orig, copy_path=os.path.join(T, cp), sha256=sc, rows=nrow, bytes=os.path.getsize(c)))
    with open(os.path.join(ROOT, T, "MANIFEST.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def r2(x):
    return "NA" if pd.isna(x) else f"{x:.2f}"


def main():
    manifest()
    tA = pd.read_csv(os.path.join(ROOT, T, "rescale_wf3__wf3b_tests.csv"))
    tS = pd.read_csv(os.path.join(ROOT, T, "wf_soil__wf3b_tests.csv"))
    dA = pd.read_csv(os.path.join(ROOT, T, "rescale_wf3__wf3b_decomp.csv"))
    dS = pd.read_csv(os.path.join(ROOT, T, "wf_soil__wf3b_decomp.csv"))
    tg = pd.read_csv(os.path.join(ROOT, T, "rescale_wf3__wf3b_targets.csv"))
    rc = pd.read_csv(os.path.join(ROOT, T, "wf__wf9_repro_check.csv"))
    ms = pd.read_csv(os.path.join(ROOT, T, "wf__wf0_misspec.csv"))
    src = {"A": ("rescale_wf3__wf3b_tests.csv", tA), "S": ("wf_soil__wf3b_tests.csv", tS)}
    out = []

    def test(eid, grp, which, test_id, contrast, target, role=None, doc=""):
        name, t = src[which]
        q = t[(t.test_id == test_id) & (t.contrast == contrast) & (t.target == target)]
        if role:
            q = q[q.role == role]
        assert len(q) == 1, (eid, len(q))
        r = q.iloc[0]
        val = (f"{r2(r.delta)} [{r2(r.ci_lo)}, {r2(r.ci_hi)}] / "
               f"{r2(r.delta_blockeq)} [{r2(r.ci_lo_beq)}, {r2(r.ci_hi_beq)}]")
        filt = f"test_id={test_id}; contrast={contrast}; target={target}; n={r.n:g}; lam={r2(r.lam)}" + (f"; role={role}" if role else "")
        out.append(dict(id=eid, group=grp, source=f"tables/{name}", filter=filt,
                        columns="delta [ci_lo, ci_hi] / delta_blockeq [ci_lo_beq, ci_hi_beq]",
                        value=val, verdict4=r.verdict4, verdict_doc=doc))

    def verdict(eid, grp, which, test_id, doc):
        name, t = src[which]
        q = t[(t.test_id == test_id) & (t.scope == "verdict")]
        assert len(q) == 1
        out.append(dict(id=eid, group=grp, source=f"tables/{name}", filter=f"test_id={test_id}; scope=verdict",
                        columns="verdict", value=q.iloc[0].verdict, verdict4="", verdict_doc=doc))

    def dec(eid, grp, d, name, exp, base, method, lam, col, n=-1, learner=None, doc=""):
        q = d[(d.exp == exp) & (d.base == base) & (d.method == method) & (d.n == n) & (d.lam == lam)]
        if learner:
            q = q[q.learner == learner]
        assert len(q) == 1, (eid, len(q))
        v = q.iloc[0][col]
        out.append(dict(id=eid, group=grp, source=f"tables/{name}",
                        filter=f"exp={exp}; base={base}; method={method}; learner={q.iloc[0].learner}; n={n}; lam={lam}",
                        columns=col, value=f"{r2(v)} ({v * 100:.1f} %)", verdict4="서술", verdict_doc=doc))

    def dec_max(eid, grp, d, name, exp, doc):
        ml = d[(d.exp == exp) & d.method.isin(["D0", "D1", "R1", "R2", "Re"])]
        r = ml.loc[ml.expl_within.idxmax()]
        out.append(dict(id=eid, group=grp, source=f"tables/{name}",
                        filter=f"exp={exp}; method in D0,D1,R1,R2,Re; 모든 n·lam 의 최댓값 → base={r.base}, method={r.method}, n={r.n}, lam={r.lam}",
                        columns="expl_within (max)", value=f"{r2(r.expl_within)} ({r.expl_within * 100:.1f} %)",
                        verdict4="서술", verdict_doc=doc))

    S53 = "WF 계획서 5.3"
    H1132 = "WF 계획서 개정 이력 2026-10-02 11:32"
    NEW = "판정 문서 없음(이 폴더에서 처음 인용, 서술)"
    # A. 알래스카 지역 내 라벨 전량, R1(교차검증 λ) − P1 의 총·격자 사이·격자 안
    test("A1", "A 분해(기온)", "A", "WF9-b", "R1(λ cv)-P1[총]|n전량", "Alaska|r", doc=S53 + " 해석 (2)(사후)")
    test("A2", "A 분해(기온)", "A", "WF9-b", "R1(λ cv)-P1[격자 사이]|n전량", "Alaska~b|r", doc=S53 + " WF9-b 행, 해석 (2)(사후)")
    test("A3", "A 분해(기온)", "A", "WF9-a", "R1(λ cv)-P1[격자 안]|n전량", "Alaska~w|r", role="주", doc=S53 + " WF9-a 행")
    test("A4", "A 분해(토양)", "S", "WF9-b", "R1(λ cv)-P1[총]|n전량", "Alaska|r", doc=H1132)
    test("A5", "A 분해(토양)", "S", "WF9-b", "R1(λ cv)-P1[격자 사이]|n전량", "Alaska~b|r", doc=H1132 + " 대상 행")
    test("A6", "A 분해(토양)", "S", "WF9-a", "R1(λ cv)-P1[격자 안]|n전량", "Alaska~w|r", role="주", doc=H1132 + " 대상 행")
    # B. SSE 비율(서술)
    nA, nS = "rescale_wf3__wf3b_decomp.csv", "wf_soil__wf3b_decomp.csv"
    for i, b in enumerate(["Alaska|r", "Lena|r", "Canada|r"]):
        dec(f"B{i + 1}", "B SSE 비율", dA, nA, "wf9", b, "P1", 0.0, "share_within_of_sse", doc=S53 + " 설명 비율(서술)")
    dec("B4", "B SSE 비율", dA, nA, "wf9", "Alaska|r", "R1", -1.0, "share_gain_within", doc=NEW)
    dec("B5", "B SSE 비율", dA, nA, "wf9", "Alaska|r", "R1", 0.25, "share_gain_within", doc=NEW)
    dec("B6", "B SSE 비율", dA, nA, "wf9", "Lena|r", "R1", 0.25, "share_gain_within", doc=NEW)
    dec("B7", "B SSE 비율", dA, nA, "wf9", "Canada|r", "R1", 0.25, "share_gain_within", doc=NEW)
    dec("B8", "B SSE 비율", dA, nA, "wf9x", "Lena|x", "R1", 0.25, "share_gain_within", doc=NEW)
    dec("B9", "B SSE 비율", dA, nA, "wf9x", "Alaska|x", "R1", 0.25, "share_gain_within", doc=NEW)
    dec("B10", "B SSE 비율(토양)", dS, nS, "wf9", "Alaska|r", "R1", -1.0, "share_gain_within", doc=NEW)
    dec("B11", "B SSE 비율(토양)", dS, nS, "wf9", "Lena|r", "R1", 0.25, "share_gain_within", doc=NEW)
    def dec_raw(eid, grp, method, lam, base, cols, doc):
        q = dA[(dA.exp == "wf9") & (dA.base == base) & (dA.method == method) & (dA.n == -1) & (dA.lam == lam)]
        assert len(q) == 1, (eid, len(q))
        r = q.iloc[0]
        out.append(dict(id=eid, group=grp, source=f"tables/{nA}",
                        filter=f"exp=wf9; base={base}; method={method}; learner={r.learner}; n=-1; lam={lam}",
                        columns="; ".join(cols), value="; ".join(f"{r[c]:,.2f}" for c in cols), verdict4="서술", verdict_doc=doc))
    dec_raw("B12", "B 집계 점검", "P1", 0.0, "Alaska|r", ["rmse_tot", "rmse_b", "rmse_w"], NEW)
    dec_raw("B13", "B 집계 점검", "R1", -1.0, "Alaska|r", ["rmse_tot", "rmse_b", "rmse_w"], NEW)
    dec_raw("B14", "B 집계 점검", "R1", -1.0, "Canada|r", ["gain_total_sse", "share_gain_within"], NEW)
    # C. 격자 안 설명 비율 1 − MSE_w(M)/MSE_w(P1)
    E = "expl_within"
    dec("C1", "C 설명 비율", dA, nA, "wf9", "Alaska|r", "R1", -1.0, E, doc=S53 + " 설명 비율(서술)")
    dec("C2", "C 설명 비율", dA, nA, "wf9", "Alaska|r", "R1", 0.25, E, doc=S53 + " 설명 비율(서술)")
    dec("C3", "C 설명 비율", dA, nA, "wf9", "Alaska|r", "D0", 1.0, E, learner="catboost", doc=S53 + " 설명 비율(서술)")
    dec("C4", "C 설명 비율", dA, nA, "wf9", "Lena|r", "R1", 0.25, E, doc=S53 + " 설명 비율(서술)")
    dec("C5", "C 설명 비율", dA, nA, "wf9", "Lena|r", "D0", 1.0, E, learner="catboost", doc=S53 + " 설명 비율(서술)")
    dec("C6", "C 설명 비율", dA, nA, "wf9", "Canada|r", "R1", -1.0, E, doc=S53 + " 설명 비율(서술)")
    dec("C7", "C 설명 비율", dA, nA, "wf9", "Canada|r", "D0", 1.0, E, learner="catboost", doc=S53 + " 설명 비율(서술)")
    dec("C8", "C 설명 비율", dA, nA, "wf9x", "Lena|x", "R1", 0.25, E, doc=S53 + " 설명 비율(서술)")
    dec("C9", "C 설명 비율", dA, nA, "wf9x", "Alaska|x", "R1", 0.25, E, doc=S53 + " 설명 비율(서술)")
    dec("C10", "C 설명 비율", dA, nA, "wf9x", "Canada|x", "R1", 0.25, E, doc=S53 + " 설명 비율(서술)")
    dec("C11", "C 설명 비율", dA, nA, "wf9x", "Russia_E|x", "R1", 0.25, E, doc=S53 + " 설명 비율(서술)")
    dec_max("C12", "C 설명 비율(최댓값)", dA, nA, "wf9", NEW)
    dec_max("C13", "C 설명 비율(최댓값)", dS, nS, "wf9", NEW)
    dec_max("C14", "C 설명 비율(최댓값)", dA, nA, "wf9x", NEW)
    # D–G. 등록 가설 WF9-a, WF9-c (두 묶음 정의)
    for which, g, doc in (("A", "D WF9-a(기온)", S53 + " WF9-a 행"), ("S", "E WF9-a(토양)", H1132 + " WF9-a")):
        k = "D" if which == "A" else "E"
        test(f"{k}1", g, which, "WF9-a", "R1(λ cv)-P1[격자 안]|n500", "MEAN[Alaska~w|r,Lena~w|r]", role="주", doc=doc)
        test(f"{k}2", g, which, "WF9-a", "R1(λ cv)-P1[격자 안]|n1000", "MEAN[Alaska~w|r,Lena~w|r]", role="주", doc=doc)
        test(f"{k}3", g, which, "WF9-a", "R1(λ cv)-P1[격자 안]|n전량", "MEAN[Alaska~w|r,Lena~w|r,Canada~w|r]", role="주", doc=doc)
        test(f"{k}4", g, which, "WF9-a", "R1(λ cv)-P1[격자 안]|n전량", "Lena~w|r", role="주", doc=doc)
        test(f"{k}5", g, which, "WF9-a", "R1(λ cv)-P1[격자 안]|n전량", "Canada~w|r", role="주", doc=doc)
        verdict(f"{k}6", g, which, "WF9-a", doc)
    for which, g, doc in (("A", "F WF9-c(기온)", S53 + " WF9-c 행"), ("S", "G WF9-c(토양)", H1132 + " WF9-c")):
        k = "F" if which == "A" else "G"
        test(f"{k}1", g, which, "WF9-c", "R1(λ 0.25)-P1[격자 안]|n0", "MEAN[Lena~w|x,Canada~w|x]", role="주", doc=doc)
        test(f"{k}2", g, which, "WF9-c", "R1(λ 0.25)-P1[격자 안]|n10", "MEAN[Lena~w|x,Canada~w|x]", role="주", doc=doc)
        test(f"{k}3", g, which, "WF9-c", "R1(λ 0.25)-P1[격자 안]|n전량", "MEAN[Lena~w|x,Canada~w|x]", role="주", doc=doc)
        test(f"{k}4", g, which, "WF9-c", "R1(λ 0.25)-P1[격자 안]|n전량", "Lena~w|x", role="주", doc=doc)
        test(f"{k}5", g, which, "WF9-c", "R1(λ 0.25)-P1[격자 안]|n전량", "Russia_E~w|x", role="주", doc=doc)
        verdict(f"{k}6", g, which, "WF9-c", doc)
    # H. 고정 λ 0.25 의 격자 안·격자 사이(서술, 등록 가설 밖)
    test("H1", "H λ 0.25(기온)", "A", "WF9-b", "R1(λ 0.25)-P1[격자 안]|n전량", "MEAN[Alaska~w|r,Lena~w|r,Canada~w|r]", doc=NEW)
    test("H2", "H λ 0.25(기온)", "A", "WF9-b", "R1(λ 0.25)-P1[격자 안]|n전량", "Lena~w|r", role="서술", doc=S53 + " WF9-b 행")
    test("H3", "H λ 0.25(토양)", "S", "WF9-b", "R1(λ 0.25)-P1[격자 안]|n전량", "MEAN[Alaska~w|r,Lena~w|r,Canada~w|r]", doc=NEW)
    test("H4", "H λ 0.25(토양)", "S", "WF9-b", "R1(λ 0.25)-P1[격자 안]|n전량", "Lena~w|r", role="서술", doc=NEW)
    test("H5", "H λ 0.25(기온)", "A", "WF9-b", "R1(λ 0.25)-P1[격자 사이]|n전량", "Alaska~b|r", doc=S53 + " WF9-b 행")
    test("H6", "H λ 0.25(기온)", "A", "WF9-b", "R1(λ 0.25)-P1[격자 사이]|n전량", "Lena~b|r", doc=S53 + " WF9-b 행")
    test("H7", "H λ 0.25(기온)", "A", "WF9-b", "R1(λ 0.25)-P1[격자 사이]|n전량", "Canada~b|r", doc=S53 + " WF9-b 행")
    test("H8", "H 직접 ML·위성 앵커(기온)", "A", "WF9-b", "D0(catboost)-P1[격자 안]|n전량", "MEAN[Alaska~w|r,Lena~w|r,Canada~w|r]", doc=NEW)
    test("H9", "H 직접 ML·위성 앵커(기온)", "A", "WF9-b", "Re(λ cv)-P1[격자 안]|n전량", "MEAN[Alaska~w|r,Lena~w|r,Canada~w|r]", doc=NEW)
    # I. WF10
    W10 = S53 + " WF10 행"
    for k, tid in (("I1", "WF10-a"), ("I2", "WF10-b"), ("I3", "WF10-c")):
        verdict(k, "I WF10", "A", tid, W10)
    test("I4", "I WF10(알래스카 서술)", "A", "WF10-a", "EP[D0(catboost)-P1]|warm|n100", "Alaska~warm|r", role="주", doc=W10)
    test("I5", "I WF10(알래스카 서술)", "A", "WF10-a", "EP[D0(catboost)-P1]|warm|n500", "Alaska~warm|r", role="서술", doc=NEW)
    test("I6", "I WF10(알래스카 서술)", "A", "WF10-a", "EP[D0(catboost)-P1]|warm|n전량", "Alaska~warm|r", role="주", doc=W10)
    test("I7", "I WF10(알래스카 서술)", "A", "WF10-b", "EP[R1(λ cv)-D0(catboost)]|warm|n100", "Alaska~warm|r", role="주", doc=W10)
    test("I8", "I WF10(알래스카 서술)", "A", "WF10-b", "EP[R1(λ cv)-D0(catboost)]|warm|n전량", "Alaska~warm|r", role="주", doc=W10)
    test("I9", "I WF10(알래스카 서술)", "A", "WF10-c", "R1(λ cv)-P1[W]|warm|n100", "Alaska~warmW|r", role="주", doc=W10)
    test("I10", "I WF10(알래스카 서술)", "A", "WF10-c", "R1(λ cv)-P1[W]|warm|n전량", "Alaska~warmW|r", role="주", doc=W10)
    for k, tgt, var in (("I11", "Canada", "warm"), ("I12", "Alaska", "warm")):
        q = tg[(tg.exp == "wf10") & (tg.target == tgt) & (tg.wf10_variant == var)]
        out.append(dict(id=k, group="I WF10(설계)", source="tables/rescale_wf3__wf3b_targets.csv",
                        filter=f"exp=wf10; target={tgt}; wf10_variant={var}; 분할 {len(q)}개",
                        columns="nb_eval_W (min–max)", value=f"{q.nb_eval_W.min():g}–{q.nb_eval_W.max():g}",
                        verdict4="", verdict_doc="WF 계획서 개정 이력 2026-10-02 00:20 항목 4, " + S53))
    q = tg[(tg.exp == "wf10") & (tg.status != "ok")]
    out.append(dict(id="I13", group="I WF10(설계)", source="tables/rescale_wf3__wf3b_targets.csv",
                    filter="exp=wf10; status!=ok", columns="target, wf10_variant, split, status",
                    value="; ".join(f"{r.target} {r.wf10_variant} 분할 {r.split} {r.status}" for r in q.itertuples()),
                    verdict4="", verdict_doc=S53 + " 원천"))
    # J. 재현 점검(WF9 지역 내 총 저장소 = WF6)
    out.append(dict(id="J1", group="J 재현", source="tables/wf__wf9_repro_check.csv", filter="전체 행",
                    columns="행 수; ok 모두 True; n_common 합; max_abs_dsse 최대; n_count_mismatch 합",
                    value=f"{len(rc)}; {bool(rc.ok.all())}; {int(rc.n_common.sum())}; {rc.max_abs_dsse.max():g}; {int(rc.n_count_mismatch.sum())}",
                    verdict4="", verdict_doc=S53 + " 재현 점검"))
    # K. C2 좁힘(재보정 몫과 재보정을 넘어선 ML 몫)
    r_ml = spearmanr(ms.recal_gain, -ms.ml_vs_p1)
    r_p0 = spearmanr(ms.recal_gain, -ms.ml_best_delta)
    out.append(dict(id="K1", group="K C2 좁힘", source="tables/wf__wf0_misspec.csv",
                    filter=f"전체 {len(ms)}행", columns="Spearman(recal_gain, -ml_vs_p1) (p)",
                    value=f"{r_ml.correlation:.2f} (p {r_ml.pvalue:.2f})", verdict4="사후",
                    verdict_doc="QA_FINAL_REVIEW Q1 C2 점검"))
    out.append(dict(id="K2", group="K C2 좁힘", source="tables/wf__wf0_misspec.csv",
                    filter=f"전체 {len(ms)}행", columns="Spearman(recal_gain, -ml_best_delta) (p)",
                    value=f"{r_p0.correlation:.2f} (p {r_p0.pvalue:.4f})", verdict4="사후",
                    verdict_doc="RESEARCH_CLAIMS C2(WF0 0.66, 사후 서술)"))
    ev = pd.DataFrame(out)
    ev.to_csv(os.path.join(ROOT, HERE, "evidence_c8.csv"), index=False)
    for r in ev.itertuples():
        print(f"| {r.id} | `{r.source}` | {r.filter} | {r.columns} | {r.value} | {r.verdict4} | {r.verdict_doc} |")


if __name__ == "__main__":
    main()
