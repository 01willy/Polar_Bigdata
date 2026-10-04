"""C6 근거 표(README 2절)를 tables/ 사본에서 다시 만든다. 가벼운 pandas 읽기만 한다(OMP_NUM_THREADS=1 권장).

실행: python extract_evidence.py [출력.md]   (인자가 없으면 표준 출력으로 낸다)
사본의 원천 경로와 sha256 은 tables/MANIFEST.csv 에 있다. 수치는 소수 둘째 자리로 반올림한다.
"""
import os
import sys

import pandas as pd

T = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")
wf = pd.read_csv(os.path.join(T, "wf_tests.csv"))        # 원천 results/rescale_wf/data/processed/wf/wf_tests.csv
wf2b = pd.read_csv(os.path.join(T, "wf2b_tests.csv"))    # 원천 results/rescale_wf2/data/processed/wf/wf2b_tests.csv
l43 = pd.read_csv(os.path.join(T, "lgw_l43.csv"))        # 원천 data/processed/lgw/lgw_l43.csv
lgwt = pd.read_csv(os.path.join(T, "lgw_tests.csv"))     # 원천 data/processed/lgw/lgw_tests.csv

HDR = ("| 번호 | 원천 | 행 필터 | `delta` [`ci_lo`, `ci_hi`] (셀 가중) | `delta_blockeq` [`ci_lo_beq`, `ci_hi_beq`] (블록 등가중) "
       "| `rmse_A` / `rmse_B` | `verdict4` | 판정 문서 |")
SEP = "|---|---|---|---|---|---|---|---|"
D51 = "WF 계획서 5.1(WF2-a 서술 대상)"
D51M = "WF 계획서 5.1(WF2-a 주)"
D52 = "WF 계획서 5.2"
DJ7 = "WRAPUP 결과 판정 기록(J7) L43"


def f(x):
    return "NaN" if pd.isna(x) else f"{x:.2f}"


def esc(s):
    return str(s).replace("|", "\\|")


def row_fmt(no, src, filt, r, doc, extra=True):
    v = f"{f(r.delta)} [{f(r.ci_lo)}, {f(r.ci_hi)}]"
    b = f"{f(r.delta_blockeq)} [{f(r.ci_lo_beq)}, {f(r.ci_hi_beq)}]"
    ex = f"{f(r.rmse_A)} / {f(r.rmse_B)}" if extra else ""
    return f"| {no} | {src} | {esc(filt)} | {v} | {b} | {ex} | {r.verdict4} | {doc} |"


def wf_row(tab, src, exp, tid, target, contrast, doc, no):
    m = tab[(tab.exp == exp) & (tab.test_id == tid) & (tab.target == target) & (tab.contrast == contrast)]
    assert len(m) == 1, (exp, tid, target, contrast, len(m))
    r = m.iloc[0]
    filt = f"exp={exp}, test_id={tid}, role={r.role}, target={target}, contrast=`{contrast}`, n={int(r.n)}, lam={f(r.lam)}"
    return row_fmt(no, src, filt, r, doc)


def doc_of(tgt):
    return D51M if tgt == "Alaska|r" else D51


# A. 지역 내 S2·S4
secA, k = [], 0
for tgt, s, ns in [("Canada|r", "S2", (20, 50, 100, 200)), ("Canada|r", "S4", (20, 50, 100, 200)),
                   ("AL-1|r", "S4", (50, 100, 200)), ("AL-2|r", "S2", (50, 100, 200)), ("AL-2|r", "S4", (50,)),
                   ("Lena|r", "S2", (50, 100, 200, 500)), ("Lena|r", "S4", (50, 100, 200)),
                   ("Alaska|r", "S2", (50, 100, 200)), ("Alaska|r", "S3", (50, 100, 200)), ("Alaska|r", "S4", (50, 100, 200))]:
    for n in ns:
        k += 1
        secA.append(wf_row(wf, "W", "wf2", "WF2-a", tgt, f"{s}-S1|R1(0.25)|n{n}", doc_of(tgt), f"A{k}"))

# B. S6 순차 능동 선택
secB, k = [], 0
for tgt in ("Alaska|r", "Lena|r", "AL-1|r", "AL-2|r", "Canada|r"):
    for n in (50, 100, 200):
        k += 1
        secB.append(wf_row(wf, "W", "wf2", "WF2-a", tgt, f"S6-S1|R1(0.25)|n{n}", doc_of(tgt), f"B{k}"))

# F. S5·S7(보조)
secF, k = [], 0
for tgt, s, ns in [("Alaska|r", "S5", (100, 200)), ("AL-2|r", "S5", (50, 200)), ("Lena|r", "S5", (100,)), ("Canada|r", "S5", (100,)),
                   ("AL-1|r", "S5", (100,)), ("Lena|r", "S7", (50, 100, 200)), ("Alaska|r", "S7", (100,)), ("Canada|r", "S7", (100,)),
                   ("AL-1|r", "S7", (100,)), ("AL-2|r", "S7", (100,))]:
    for n in ns:
        k += 1
        secF.append(wf_row(wf, "W", "wf2", "WF2-a", tgt, f"{s}-S1|R1(0.25)|n{n}", doc_of(tgt), f"F{k}"))

# C. WF2-b
secC = [wf_row(wf, "W", "wf2", "WF2-b", tgt, "S3-S1|R1(0.25)|n100", "WF 계획서 5.1(WF2-b 주)", f"C{i + 1}")
        for i, tgt in enumerate(("Lena|r", "Canada|r"))]

# D. WF8 전이
secD, k = [], 0
for tgt in ("Canada|x", "Lena|x"):
    k += 1
    secD.append(wf_row(wf2b, "W2", "wf8", "WF8-a", tgt, "S4-S1|R1(0.25)|n40", D52 + "(WF8-a 주)", f"D{k}"))
for tgt in ("Lena|x", "Canada|x", "Russia_W|x", "Russia_E|x", "Alaska|x"):
    for s in ("S2", "S4"):
        k += 1
        secD.append(wf_row(wf2b, "W2", "wf8", "WF8-b", tgt, f"{s}-S1|R1(0.25)|n10", D52 + "(WF8-b 주)", f"D{k}"))
for tgt, s in (("Canada|x", "S2"), ("Lena|x", "S2"), ("Alaska|x", "S2"), ("Alaska|x", "S4")):
    k += 1
    secD.append(wf_row(wf2b, "W2", "wf8", "WF8", tgt, f"{s}-S1|R1(0.25)|n40", D52 + "(WF8 서술 행)", f"D{k}"))

# E. L43(2단 재표집 행)
secE, k = [], 0
for method, n in (("P1", 10), ("P1", 40), ("R1", 40), ("R1", 160), ("P1", 160)):
    m = l43[(l43.test_id == "L43") & (l43.scope == "MEAN") & (l43.method == method) & (l43.n == n) & (l43.ci_kind == "2단 재표집")]
    assert len(m) == 1, (method, n, len(m))
    r = m.iloc[0]
    k += 1
    secE.append(row_fmt(f"E{k}", "L", f"test_id=L43, scope=MEAN, role={r.role}, target=`{r.target}`, method={method}, n={n}, "
                                      f"ci_kind=2단 재표집", r, DJ7, extra=False))
for tgt, method, n in (("Lena|x", "P1", 10), ("Canada|x", "P1", 10), ("Russia_W|x", "P1", 10), ("Russia_E|x", "P1", 10),
                       ("Lena|x", "P1", 40), ("Canada|x", "P1", 40), ("Canada|x", "R1", 40), ("Alaska|x", "R1", 40)):
    m = l43[(l43.test_id == "L43") & (l43.scope == "region") & (l43.target == tgt) & (l43.method == method) & (l43.n == n)
            & (l43.ci_kind == "2단 재표집")]
    assert len(m) == 1, (tgt, method, n, len(m))
    r = m.iloc[0]
    k += 1
    secE.append(row_fmt(f"E{k}", "L", f"test_id=L43, scope=region, role={r.role}, target={tgt}, method={method}, n={n}, "
                                      f"ci_kind=2단 재표집", r, DJ7 + "(지역 행, 서술)", extra=False))

# NB. 선택 라벨의 평균 블록 수
nb = l43[(l43.scope == "n_blocks_lab") & (l43.method == "P1")].sort_values(["n", "placement"])
nb_lines = [f"| scope=n_blocks_lab, method=P1, n={int(r.n)}, placement={r.placement} | `n_blocks_lab_mean` | {f(r.n_blocks_lab_mean)} |"
            for _, r in nb.iterrows()]

# VERDICT. 판정 행 원문
verd = []
for tab, tag, exp, tid in ((wf, "W", "wf2", "WF2-a"), (wf, "W", "wf2", "WF2-b"), (wf2b, "W2", "wf8", "WF8-a"), (wf2b, "W2", "wf8", "WF8-b")):
    for _, r in tab[(tab.exp == exp) & (tab.test_id == tid) & tab.verdict.notna()].iterrows():
        t = esc(r.target) if isinstance(r.target, str) else "(대상 묶음)"
        verd.append(f"| {tag} | exp={exp}, test_id={tid}, role={r.role}, target={t}, `verdict` 비어 있지 않은 행 | {esc(r.verdict)} |")
for _, r in lgwt[lgwt.test_id == "L43"].iterrows():
    verd.append(f"| LT | test_id=L43, scope={r.scope}, role={r.role} | {r.verdict} ({r.stat}; blind 열: {r['blind']}) |")

wf2b_stat = wf[(wf.exp == "wf2") & (wf.test_id == "WF2-b") & wf.verdict.notna()].iloc[0].stat

# XC. WF8 S2 − S1 과 L43 블록 분산 − 셀 무작위(같은 라벨 집합)
xc = []
w8 = wf2b[(wf2b.exp == "wf8") & (wf2b.test_id == "WF8") & (wf2b.strategy == "S2")].copy()
w8["M"] = w8.contrast.str.split("|").str[1].str.replace("(0.25)", "", regex=False)
lr = l43[(l43.scope == "region") & (l43.ci_kind == "2단 재표집")]
for _, r in w8.sort_values(["target", "M", "n"]).iterrows():
    q = lr[(lr.target == r.target) & (lr.method == r.M) & (lr.n == r.n)]
    if len(q) == 1:
        q = q.iloc[0]
        xc.append(f"| {esc(r.target)} | {r.M} | {int(r.n)} | {r.delta:.4f} [{f(r.ci_lo)}, {f(r.ci_hi)}] {r.verdict4} | "
                  f"{q.delta:.4f} [{f(q.ci_lo)}, {f(q.ci_hi)}] {q.verdict4} |")

buf = []
for title, sec in (("A", secA), ("B", secB), ("F", secF), ("C", secC), ("D", secD), ("E", secE)):
    buf.append(f"\n### {title}\n\n{HDR}\n{SEP}\n" + "\n".join(sec) + "\n")
buf.append("\n### NB\n\n| 행 필터 | 열 | 값 |\n|---|---|---|\n" + "\n".join(nb_lines) + "\n")
buf.append("\n### VERDICT\n\n| 원천 | 행 필터 | `verdict` |\n|---|---|---|\n" + "\n".join(verd) + "\n")
buf.append("\n### WF2B_STAT\n\n" + esc(wf2b_stat) + "\n")
buf.append("\n### XC\n\n| target | method | n | WF8 `delta` [CI] `verdict4` | L43 2단 `delta` [CI] `verdict4` |\n|---|---|---|---|---|\n"
           + "\n".join(xc) + "\n")
text = "".join(buf)
if len(sys.argv) > 1:
    with open(sys.argv[1], "w") as fh:
        fh.write(text)
else:
    sys.stdout.write(text)
