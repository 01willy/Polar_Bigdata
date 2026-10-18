#!/usr/bin/env python3
"""Word counts of the rendered manuscript (MANUSCRIPT_SPEC 10: pdftotext, cut at section
boundaries, count whitespace-separated tokens as `wc -w` does).

Counts are reported twice: as rendered, and without the bracketed placeholders
[DECISION: ...], [미확인: ...], [MISSING: ...] and [X?: ...]. Superscript citation
numbers are rendered as separate tokens by pdftotext; they are removed before counting
(a token of digits, commas and en dashes that directly follows a word in the source
cannot be separated reliably, so the count may differ from a source count by a few words).
Usage: python3 tools/count_words.py [main.pdf]  -> prints a JSON-like summary.
"""
import json
import re
import subprocess
import sys
import unicodedata

PDF = sys.argv[1] if len(sys.argv) > 1 else "main.pdf"
PH = re.compile(r"\[(?:DECISION|미확인|MISSING|PENDING|X[A-J]):[^\]]*\]", re.S)


def text_of(pdf):
    t = subprocess.run(["pdftotext", "-enc", "UTF-8", pdf, "-"], capture_output=True,
                       text=True, check=True).stdout
    return unicodedata.normalize("NFKC", t.replace("\f", "\n"))  # page breaks, ligatures


def clean(block):
    lines = []
    for ln in block.split("\n"):
        s = ln.strip().replace("\f", "")
        if re.fullmatch(r"\d{1,3}", s):  # page numbers
            continue
        lines.append(s)
    t = " ".join(lines)
    # superscript citation tokens: pdftotext writes "permafrost 1,2 ." (digits after a word, then
    # a space before punctuation); plain numbers in the text are followed by a word, not by " ."
    t = re.sub(r"(?<=[A-Za-z)]) (\d+(?:[,–-]\d+)*) (?=[.,;:)])", "", t)
    return t


def wc(t):
    return len(t.split())


def counts(t):
    return {"rendered": wc(t), "without_placeholders": wc(PH.sub(" ", t))}


def main():
    full = text_of(PDF)
    lines = full.split("\n")

    def find(pattern, start=0):
        for i in range(start, len(lines)):
            if re.fullmatch(pattern, lines[i].strip().replace("\f", "")):
                return i
        raise SystemExit(f"marker not found: {pattern}")

    i_abs = find(r"Abstract")
    i_kw = next(i for i in range(i_abs, len(lines)) if lines[i].startswith("Keywords:"))
    i_intro = find(r"Introduction", i_kw)
    i_res = find(r"Results", i_intro)
    i_disc = find(r"Discussion", i_res)
    i_meth = find(r"Methods", i_disc)
    i_data = find(r"Data availability", i_meth)
    i_refs = find(r"References", i_data)
    i_leg = find(r"Figure legends", i_refs)
    i_tab = find(r"Table caption", i_leg)
    i_tabfloat = next(i for i in range(i_tab + 1, len(lines)) if re.match(r"Table 1 Target", lines[i]))

    seg = lambda a, b: clean("\n".join(lines[a:b]))
    out = {
        "abstract": counts(seg(i_abs + 1, i_kw)),
        "introduction": counts(seg(i_intro + 1, i_res)),
        "results": counts(seg(i_res + 1, i_disc)),
        "discussion": counts(seg(i_disc + 1, i_meth)),
        "methods": counts(seg(i_meth + 1, i_data)),
        "data_availability": counts(seg(i_data + 1, i_refs)),
    }
    out["main_text_intro_results_discussion"] = {
        k: out["introduction"][k] + out["results"][k] + out["discussion"][k] for k in ("rendered", "without_placeholders")}

    # results subsections: headings are the subsection titles of results_a/b.tex
    subs = ["Evaluation design and validation ladder", "Transfer without target labels",
            "Coefficient recalibration with three to ten labels", "Method selection with tens to hundreds of labels",
            "Gains within label-rich regions", "Placement of new labels", "Independent regions", "Prediction intervals"]
    idx = []
    for s in subs:
        idx.append(next(i for i in range(i_res, i_disc) if lines[i].strip() == s))
    idx.append(i_disc)
    out["results_subsections"] = {s: counts(seg(idx[k] + 1, idx[k + 1])) for k, s in enumerate(subs)}

    leg = "\n".join(lines[i_leg + 1:i_tab])
    parts = re.split(r"(?m)^(?=Figure [1-7]\. )", leg)
    legends = {}
    for p in parts:
        m = re.match(r"Figure ([1-7])\. ", p)
        if m:
            legends[f"Fig. {m.group(1)}"] = counts(clean(p))
    out["figure_legends"] = legends
    out["table1_caption"] = counts(clean("\n".join(lines[i_tab + 1:i_tabfloat])))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
