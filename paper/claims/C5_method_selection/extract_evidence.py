"""C5(방법 선택) 근거 수치 추출.

tables/ 의 사본만 읽고 README 의 수치(머리말 예외 (2) 6절 비용, (3) 3절 실행 정보 제외)를 evidence_values.csv 한 표로 낸다.
학습이나 재표집은 하지 않는다. N3 의 ρ −0.08 은 tables/wf0_misspec.csv 로 재계산한다(7절).
실행: OMP_NUM_THREADS=1 python3 paper/claims/C5_method_selection/extract_evidence.py
열: item, source(사본 경로), row_filter, column, value(파일 값 그대로), value_r2(소수 2자리), note.
'kind' 가 'recheck' 인 행은 이번 점검(2026-10-04, 사후)에서 파일 값으로 새로 계산한 것이다. 판정이 아니다.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
T = HERE / "tables"
OUT = HERE / "evidence_values.csv"
MEAN4 = "MEAN[Lena|x,Canada|x,Russia_W|x,Russia_E|x]"
MEAN2 = "MEAN[Lena|x,Canada|x]"
rows: list[dict] = []


def add(item, source, flt, col, val, kind="file", note=""):
    v2 = round(float(val), 2) if isinstance(val, (int, float)) and not isinstance(val, bool) and pd.notna(val) else val
    rows.append(dict(item=item, kind=kind, source=f"tables/{source}", row_filter=flt, column=col, value=val, value_r2=v2, note=note))


def one(df, **kw):
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        m &= df[k] == v
    s = df[m]
    if len(s) != 1:
        raise SystemExit(f"행 {len(s)}개: {kw}")
    return s.iloc[0]


# 1. WF4-a: 규칙 W − 고정 R1(λ 0.25)
wt = pd.read_csv(T / "wf_tests.csv")
cols_a = ["delta", "ci_lo", "ci_hi", "delta_blockeq", "ci_lo_beq", "ci_hi_beq", "verdict4", "verdict4_common", "ci_lo_c", "ci_hi_c",
          "ci_lo_beq_c", "ci_hi_beq_c", "ci_dependence", "holm_p", "p_ni", "ni", "holm_p_ni", "pool", "pool_regions", "rmse_A", "rmse_B",
          "delta_pct_p0", "design_note"]
for nkey, n, tgt in [("n10", 10.0, MEAN4), ("n40", 40.0, MEAN2), ("n160", 160.0, MEAN2), ("n전량", -1.0, MEAN4)]:
    r = one(wt, test_id="WF4-a", scope="MEAN", n=n)
    assert r["target"] == tgt, (r["target"], tgt)
    flt = f"test_id=WF4-a; scope=MEAN; contrast=W-R1(0.25)|{nkey}"
    for c in cols_a:
        add("WF4-a", "wf_tests.csv", flt, c, r[c] if pd.notna(r[c]) else "")
for reg in ["Lena|x", "Canada|x", "Russia_W|x", "Russia_E|x"]:
    for nkey, n in [("n10", 10.0), ("n40", 40.0), ("n160", 160.0), ("n전량", -1.0)]:
        s = wt[(wt.test_id == "WF4-a") & (wt.scope == "region") & (wt.target == reg) & (wt.n == n)]
        if len(s) == 0:
            continue
        r = s.iloc[0]
        flt = f"test_id=WF4-a; scope=region; target={reg}; contrast=W-R1(0.25)|{nkey}"
        for c in ["delta", "ci_lo", "ci_hi", "ci_lo_beq", "ci_hi_beq", "verdict4", "verdict4_common", "ci_dependence"]:
            add("WF4-a region", "wf_tests.csv", flt, c, r[c] if pd.notna(r[c]) else "")
# 1b. 공통 재표집 CI 로 본 비열등(이번 점검, 사후): 등록 기준(두 가중 CI 상한 < ni_margin)을 공통 CI 에 그대로 적용한다.
#     파일의 ci_dependence 는 4분 판정만 비교하므로 비열등의 CI 의존은 표시하지 않는다.
ni_margin = json.loads((T / "wf_meta.json").read_text())["ni_margin"]
for nkey, n in [("n10", 10.0), ("n40", 40.0), ("n160", 160.0), ("n전량", -1.0)]:
    r = one(wt, test_id="WF4-a", scope="MEAN", n=n)
    flt = f"test_id=WF4-a; scope=MEAN; contrast=W-R1(0.25)|{nkey}"
    ni_c = bool(r["ci_hi_c"] < ni_margin and r["ci_hi_beq_c"] < ni_margin)
    add("WF4-a NI common", "wf_tests.csv", flt, "공통 CI 비열등(ci_hi_c, ci_hi_beq_c < ni_margin)", ni_c, kind="recheck",
        note=f"ci_hi_c {r['ci_hi_c']:.4f}, ci_hi_beq_c {r['ci_hi_beq_c']:.4f}, 주 CI 비열등(ni) {r['ni']}, 한계 {ni_margin} cm, pool {r['pool']}")
r = one(wt, test_id="WF4-a", scope="verdict")
add("WF4-a", "wf_tests.csv", "test_id=WF4-a; scope=verdict", "verdict", r["verdict"])
add("WF4-a", "wf_tests.csv", "test_id=WF4-a; scope=verdict", "stat", r["stat"])

# 2. WF4-b: P1 대비 우세 대상 수
r = one(wt, test_id="WF4-b", scope="verdict")
add("WF4-b", "wf_tests.csv", "test_id=WF4-b; scope=verdict", "verdict", r["verdict"])
add("WF4-b", "wf_tests.csv", "test_id=WF4-b; scope=verdict", "stat", r["stat"])
b = wt[(wt.test_id == "WF4-b") & (wt.scope == "region")].copy()
b["nk"] = b.contrast.str.split("|").str[1]
for nk in ["n160", "n전량"]:
    p = b[b.nk == nk].pivot(index="target", columns="method", values="verdict4")
    ok = p.dropna()
    ok = ok[~ok.isin(["판정 불가", "행 없음"]).any(axis=1)]
    add("WF4-b recount", "wf_tests.csv", f"test_id=WF4-b; scope=region; contrast∈{{W-P1|{nk}, R1(0.25)-P1|{nk}}}", "판정 가능 대상 수",
        len(ok), kind="recheck", note=f"전체 {len(p)}")
    for m in ["W", "R1"]:
        for v in ["우세", "열세"]:
            tg = ok.index[ok[m] == v].tolist()
            add("WF4-b recount", "wf_tests.csv", f"test_id=WF4-b; scope=region; method={m}; {nk}", f"verdict4=={v} 대상 수", len(tg),
                kind="recheck", note=";".join(tg))

# 3. wf_curve: 주 4지역의 W, R1(λ 0.25, 1.0) 대 P0·P1
cv = pd.read_csv(T / "wf_curve.csv")
cv = cv[(cv.exp == "wf4") & (cv["mode"] == "x")]
for tgt in ["Lena", "Canada", "Russia_W", "Russia_E", "Alaska"]:
    for n in [10, 40, 160, -1]:
        for meth, lam in [("W", 0.0), ("R1", 0.25), ("R1", 1.0), ("P1", 0.0), ("P2", 0.0)]:
            s = cv[(cv.target == tgt) & (cv.n == n) & (cv.method == meth) & (cv.lam == lam)]
            if len(s) == 0:
                continue
            r = s.iloc[0]
            flt = f"exp=wf4; mode=x; target={tgt}; method={meth}; lam={lam}; n={n}"
            for c in ["rmse", "d_p0", "d_p0_lo", "d_p0_hi", "d_p0_beq", "d_p0_beq_lo", "d_p0_beq_hi", "verdict4_p0",
                      "d_p1", "d_p1_lo", "d_p1_hi", "d_p1_beq", "d_p1_beq_lo", "d_p1_beq_hi", "verdict4_p1", "rmse_p0", "rmse_p1"]:
                add("WF4 curve", "wf_curve.csv", flt, c, r[c])
    for n in [10, 40, 160, -1]:
        w_ = cv[(cv.target == tgt) & (cv.n == n) & (cv.method == "W")]
        for lam in [0.25, 1.0]:
            r1 = cv[(cv.target == tgt) & (cv.n == n) & (cv.method == "R1") & (cv.lam == lam)]
            if len(w_) and len(r1):
                add(f"W - R1({lam}) 점 추정", "wf_curve.csv", f"exp=wf4; mode=x; target={tgt}; n={n}; rmse(W) - rmse(R1, lam {lam})", "rmse 차",
                    float(w_.rmse.iloc[0] - r1.rmse.iloc[0]), kind="recheck", note="셀 가중 RMSE 점 추정의 차, CI 없음")

# 4. wf_meta: 규칙 W 의 선택 횟수(분할 × 추출)
meta = json.loads((T / "wf_meta.json").read_text())
ch = meta["choices"]["wf4|W"]
for tgt in ["Lena", "Canada", "Russia_W", "Russia_E"]:
    for n in ["10", "40", "160", "-1"]:
        k = f"{tgt}|{n}"
        if k in ch:
            add("W 선택 횟수", "wf_meta.json", f"choices['wf4|W']['{k}']", "선택 횟수", json.dumps(ch[k], ensure_ascii=False))
add("W 후보", "wf_meta.json", "grids['wf4']", "n 격자", json.dumps(meta["grids"]["wf4"]))
add("W 후보", "wf_meta.json", "ni_margin", "비열등 한계(cm)", meta["ni_margin"])
add("W 후보", "wf_meta.json", "implementation['cv']", "교차검증 규약", meta["implementation"]["cv"])

# 5. LGF-N2: 조정판 − 기본판(원천 지역 하나 제외 교차검증 조정)
nt = pd.read_csv(T / "lgfn_tests.csv")
n2 = nt[(nt.test_id == "LGF-N2") & (nt.scope == "MEAN") & (nt.role.isin(["주", "보조(λ 1.0)"]))]
for _, r in n2.iterrows():
    flt = f"test_id=LGF-N2; scope=MEAN; role={r['role']}; contrast={r['contrast']}"
    for c in ["delta", "ci_lo", "ci_hi", "ci_lo_beq", "ci_hi_beq", "verdict4", "verdict4_common", "ci_dependence", "holm_p", "precision_ok",
              "small_note", "holm_note"]:
        add("LGF-N2", "lgfn_tests.csv", flt, c, r[c] if pd.notna(r[c]) else "")
for lr, g in n2.groupby("learner"):
    add("LGF-N2 recheck", "lgfn_tests.csv", f"test_id=LGF-N2; scope=MEAN; role∈{{주, 보조(λ 1.0)}}; learner={lr}", "precision_ok==False 대비 수",
        int((g["precision_ok"] == False).sum()), kind="recheck", note=f"대비 {len(g)}개 중")  # noqa: E712
for _, r in nt[(nt.test_id == "LGF-N2") & (nt.scope == "verdict_aux")].iterrows():
    who = r["learner"] if pd.notna(r["learner"]) else "전체"
    add("LGF-N2 verdict", "lgfn_tests.csv", f"test_id=LGF-N2; scope=verdict_aux; learner={who}", "verdict", r["verdict"])
    if pd.notna(r.get("k_default")):
        add("LGF-N2 verdict", "lgfn_tests.csv", f"test_id=LGF-N2; scope=verdict_aux; learner={who}", "k_default", r["k_default"])

# 6. LGF-N4: 원천 CV 이득 대 대상 Δ 의 순위 상관(서술, 검정 없음)
r = one(nt, test_id="LGF-N4", role="서술(검정 없음)")
flt = "test_id=LGF-N4; role=서술(검정 없음)"
add("LGF-N4", "lgfn_tests.csv", flt, "contrast", r["contrast"])
add("LGF-N4", "lgfn_tests.csv", flt, "spearman", r["spearman"])
add("LGF-N4", "lgfn_tests.csv", flt, "n_points", r["n_points"])
pts = json.loads(r["points"])
g = [p["gain"] for p in pts]
dl = [p["delta"] for p in pts]
rho, p = spearmanr(g, dl)
add("LGF-N4 recheck", "lgfn_tests.csv", flt + "; points 열 재계산", "spearman(전체)", float(rho), kind="recheck", note=f"p {p:.4f}(등록상 검정 없음)")
tie = [q for q in pts if q["gain"] == 0 and q["delta"] == 0]
add("LGF-N4 recheck", "lgfn_tests.csv", flt + "; points 열", "gain=0·delta=0(시도 0 선택) 점 수", len(tie), kind="recheck")
nz = [q for q in pts if q["gain"] != 0]
rho2, p2 = spearmanr([q["gain"] for q in nz], [q["delta"] for q in nz])
add("LGF-N4 recheck", "lgfn_tests.csv", flt + "; points 열 중 gain≠0", "spearman(시도 0 선택 제외)", float(rho2), kind="recheck",
    note=f"점 {len(nz)}개, p {p2:.4f}")
add("LGF-N4 recheck", "lgfn_tests.csv", flt + "; points 열 중 gain≠0", "delta>0 인 점 수", sum(q["delta"] > 0 for q in nz), kind="recheck",
    note=f"점 {len(nz)}개 중")

# 7. N3 의 ρ −0.08(QA 문서 Q1 C2 점검) 재계산: 재보정 이득(P0 − P1) 대 재보정을 넘어선 ML 몫(P1 − 최선 ML), 라벨 전량 30대상
w0 = pd.read_csv(T / "wf0_misspec.csv")
rho3, p3 = spearmanr(w0["recal_gain"], -w0["ml_vs_p1"])
add("N3 recheck", "wf0_misspec.csv", "전체 30행; x = recal_gain(P0 − P1), y = −ml_vs_p1(P1 − 최선 ML)", "spearman", float(rho3), kind="recheck",
    note=f"p {p3:.4f}, 대상 {len(w0)}개(QA_FINAL_REVIEW Q1 C2 점검 −0.08 재현, 사후 서술)")

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print(f"{OUT.relative_to(HERE.parents[2])}: {len(out)}행")
