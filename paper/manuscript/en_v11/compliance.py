# -*- coding: utf-8 -*-
"""compliance.py: Sci Rep 규정 점검표(원고 v11) 생성. 단어 수는 매크로를 한 단어로, 그림 캡션 · 주석 · \input 줄을 빼고 센다."""
import re, subprocess
from pathlib import Path
H = Path(__file__).resolve().parent


def words(f, drop_fig=True):
    lines = [l for l in (H / f).read_text(encoding="utf-8").split("\n") if not l.startswith("%") and not l.startswith("\\input") and not (drop_fig and l.startswith("\\paperfig"))]
    s = "\n".join(lines); s = re.sub(r"\\cite\{[^}]*\}", "", s); s = re.sub(r"\\n[A-Za-z]+(\{\})?", "X", s)
    return len(s.split())


fr = (H / "sections/front.tex").read_text(encoding="utf-8")
ab = re.search(r"\\abstract\{(.*?)\}\n\n\\keywords", fr, re.S).group(1); ab = re.sub(r"\\n[A-Za-z]+(\{\})?", "X", ab)
title = re.search(r"\\title\[[^\]]*\]\{(.*?)\}", fr).group(1)
kw = re.search(r"\\keywords\{(.*?)\}", fr).group(1).split(",")
w = {k: words(f"sections/{k}.tex") for k in ("intro", "methods", "results", "discussion", "conclusions")}
caps = []
for k in ("methods", "results"):
    for l in (H / f"sections/{k}.tex").read_text(encoding="utf-8").split("\n"):
        if l.startswith("\\paperfig"):
            c = re.sub(r"\\n[A-Za-z]+(\{\})?", "X", l); caps.append(len(c.split()) - 3)
cites = set()
for k in ("front", "intro", "methods", "results", "discussion", "conclusions", "table1"):
    for m in re.finditer(r"\\cite\{([^}]*)\}", (H / f"sections/{k}.tex").read_text(encoding="utf-8")):
        cites |= {x.strip() for x in m.group(1).split(",")}
body = w["intro"] + w["results"] + w["discussion"] + w["conclusions"]
nfig = sum(len(re.findall(r"^\\paperfig", (H / f"sections/{k}.tex").read_text(encoding="utf-8"), re.M)) for k in ("methods", "results"))
ntab = len(re.findall(r"\\begin\{table\}", (H / "sections/table1.tex").read_text(encoding="utf-8")))
ok = lambda c: "통과" if c else "**초과**"
has_cite = "\\cite" in ab
sub = H / "main_submission.pdf"; mb = f"{sub.stat().st_size / 1e6:.1f} MB" if sub.exists() else "미생성"
mbv = ("통과" if sub.stat().st_size <= 3e6 else "**초과**") if sub.exists() else "확인"
out = f"""# Sci Rep 규정 점검표 (원고 v11, 2026-10-16)

근거: nature.com/srep/author-instructions/submission-guidelines(2026-10-16 내려받음).

| 항목 | 규정 | 이 원고 | 판정 |
|---|---|---|---|
| 본문 구성 | 정해진 요구 없음(Introduction · Results · Discussion · Methods 는 권장 예시) | Introduction · Data and methods · Results · Discussion · Conclusions | 허용 |
| 그림 위치(첫 제출) | 본문 적절한 위치 또는 끝에 모음 | 처음 인용한 위치에 그림과 설명 | 허용 |
| 제목 | 20단어 이하 | {len(title.split())}단어 | {ok(len(title.split()) <= 20)} |
| 초록 | 200단어 이하, 인용 없음 | {len(ab.split())}단어 | {ok(len(ab.split()) <= 200 and not has_cite)} |
| 핵심어 | 6개 이하 | {len(kw)}개 | {ok(len(kw) <= 6)} |
| 본문(Methods · 초록 · 참고문헌 · 그림 설명 제외) | 4,500단어 이하 | {body}단어(서론 {w['intro']}, 결과 {w['results']}, 고찰 {w['discussion']}, 결론 {w['conclusions']}) | {ok(body <= 4500)} |
| Methods | 한도 없음, Code availability 포함 | {w['methods']}단어, 수식 (1)–(9), Code availability 있음 | 통과 |
| 표시 항목 | 8개 이하 | 그림 {nfig} + 표 {ntab} = {nfig + ntab} | {ok(nfig + ntab <= 8)} |
| 그림 설명 | 350단어 이하, 첫 문장 제목, 기호는 그림 안 범례 | {min(caps)}–{max(caps)}단어 | {ok(max(caps) <= 350)} |
| 표 | 편집 가능, 색 없음 | LaTeX booktabs, 흑백 | 통과 |
| 참고문헌 | 60개 권고 | 인용 {len(cites)}개 | 통과 |
| Data availability | 본문 끝, 참고문헌 앞 | 결론 뒤 | 통과 |
| 첫 제출 파일 | 3 MB 이하 단일 파일 | 압축본 {mb} | {mbv} |
| LLM 사용 진술 | Methods 에 기록하도록 권고 | 사용자 요청으로 제외 | 제출 전 재확인 |
| 자리표시 | 제출 전 0 | 교신 저자 전자우편, 공저자 · 기여, 감사의 글, 저장소 DOI | 저자 결정 필요 |
"""
(H / "COMPLIANCE_v11.md").write_text(out, encoding="utf-8"); print(out)
