#!/usr/bin/env python3
"""build_reclass_summary.py : summary of the reclassified past results for Supplementary Table S17 b (round 10, 5 October 2026).

Read-only pandas over data/processed/lgw/lgw_rescore.csv (WRAPUP 1.4 (b), execution step C7, 1 October 2026), the source named by
the frame of docs/MANUSCRIPT_DRAFT_SUPPORT_2026-09-30.md M4.4 ('Values come from lgw_rescore.csv'). No value is recomputed:
for each past ID the script counts the stored rows by CI type (column ci_kind) and by the stored four-way verdict (column verdict4).
Rows of kind 're-scored (unblinded)' (stored predictions of H18-H22 re-scored with block resampling) are counted separately.
Descriptive tables (fmt desc) are listed with their stored format note. Output: build/reclass_summary.json.
Usage: python3 tools/build_reclass_summary.py   (run from paper/manuscript/en)
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC = os.path.join(ROOT, "data", "processed", "lgw", "lgw_rescore.csv")
OUT = os.path.join(HERE, "build", "reclass_summary.json")

PAST = {  # past ID of the frame -> hyp keys of lgw_rescore.csv
    "M1 H-series": ["M1 H 계열"], "FP-F4": ["F4(b1)"], "FP-F5, FP-F6": ["F5(b2)", "F6(b2)"], "FP-F7": ["F7(b3)"],
    "FP-F8": ["F8(b4)"], "FP-F9": ["F9(c1)"], "H18–H22": ["H18–H22"], "H23, H24": ["H23", "H24"], "H25": ["H25"],
    "H26": ["H26"], "H27": ["H27"], "H28": ["H28"], "H29": ["H29"], "H30": ["H30"],
}
CI = {"두 가중": "two weightings", "CI 없음": "no CI", "셀 가중만": "cell-weighted only", "행 재표집 CI": "row resampling",
      "비율·상관·개수 형식": "ratio, correlation or count format", "없음": "none"}
VERD = [("우세", "lower"), ("열세", "higher"), ("동등", "equivalent"), ("미결정", "undetermined"),
        ("판정 불가(블록 < 8)", "fewer than 8 blocks"), ("판정 불가(CI 없음)", "no CI"),
        ("4분 판정 불가(CI 규약 다름)", "one weighting only"), ("규약 다름, 서술", "descriptive"), ("원자료 없음", "source missing")]
NOTE = {"악화 대상 수·오라클 분기 격차(개수 형식)": "number of targets with higher error and oracle-branch gap",
        "F5 요약(넓은 표)": "F5 summary (wide table)", "Spearman 상관": "Spearman correlation", "E 추정 분산 비": "variance ratio of the E estimate",
        "손익분기 n": "break-even n", "LOO R²·Spearman": "LOO R² and Spearman", "Spearman·Kendall·MW": "Spearman, Kendall and Mann–Whitney",
        "Spearman·Kendall·MW(09-26 감사 정정판)": "Spearman, Kendall and Mann–Whitney (audit-corrected edition)",
        "해로운 블록 비율·상관": "share of harmful blocks and correlation",
        "해로운 블록 비율·상관(고유 블록 단위, 09-26 감사)": "share of harmful blocks and correlation (unique blocks, audit edition)"}


def counts(q):
    vc = q["verdict4"].value_counts()
    return ", ".join(f"{en} {int(vc[ko]):,}" for ko, en in VERD if ko in vc)


def main():
    d = pd.read_csv(SRC, low_memory=False)
    out = {}
    for pid, keys in PAST.items():
        q = d[d["hyp"].isin(keys) & (d["kind"] == "재분류(비맹검)")]
        ci = q[q["fmt"] != "desc"].groupby("ci_kind", dropna=False).size()
        ci_txt = "; ".join(f"{CI.get(k, str(k))} ({int(v):,} rows)" for k, v in ci.items())
        desc = q[q["fmt"] == "desc"]
        desc_txt = "; ".join(f"{r.source.split('/')[-1]}: {NOTE.get(r.note, r.note)}" for r in desc.itertuples())
        tab = q[q["fmt"] != "desc"]
        out[pid] = {
            "rows": int(len(q)),
            "ci_type": ci_txt,
            "verdict_counts": counts(tab) if len(tab) else "",
            "descriptive_tables": desc_txt,
        }
    r = d[d["kind"] == "재채점(비맹검)"]
    out["H18–H22"]["rescored"] = {"rows": int(len(r)), "tests": int(r["test_id"].nunique()), "verdict_counts": counts(r)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for k, v in out.items():
        print(k, "|", v["ci_type"], "|", v["verdict_counts"], "|", v["descriptive_tables"][:80], "|", v.get("rescored", ""))


if __name__ == "__main__":
    main()
