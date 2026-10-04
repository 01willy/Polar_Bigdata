# Scientific Reports 투고 양식, 기존 원고 초안 점검, 영·한 원고 작성 명세 (2026-10-04)

**성격**: 영문·국문 원고 두 편(Scientific Reports 양식)을 쓰기 전의 작성 명세다. 세 부분으로 이루어진다. (1) 투고 규정의 원문 확인, (2) 기존 초안의 목록·분량·자리표시·충돌 점검, (3) 원고 구성·단어 예산·표시 항목·주장 문장 규칙·새 실험(XA–XJ) 자리·빌드 절차.
**근거**: 투고 규정은 2026-10-04 에 nature.com 과 springernature.com 에서 내려받은 원문이다(URL 은 각 표에 적었다). 초안 수치는 아래 경로의 파일에서 Python 으로 셌다(단일 스레드, 1분 미만). 연구 수치는 `docs/RESEARCH_CLAIMS_WORKFLOW_2026-10-01.md`, `docs/QA_FINAL_REVIEW_2026-10-02.md`, `docs/RESEARCH_OVERVIEW_2026-10-02.md`, `docs/MANUSCRIPT_DRAFT_RESULTS_ABSTRACT_2026-09-30.md` 에서 옮겼고 다시 계산하지 않았다.
**계산한 것**: 초안 단어 수와 자리표시 태그 수(Python 정규식), Springer Nature LaTeX 템플릿의 시험 컴파일(스크래치 폴더, 저장소 밖). 모형 학습이나 무거운 계산은 하지 않았다. 저장소의 기존 파일은 고치지 않았다.
**표기**: 확인하지 못한 내용은 [미확인]으로 적는다. 단어 예산은 제안값이고 사실이 아니다.

---

## 1. 요약

- Scientific Reports 의 Article 은 본문 4,500단어 이하(초록, Methods, 참고문헌, 그림 설명 제외), 초록 200단어 이하(비구조형, 인용 없음), 제목 20단어 이하, 표시 항목 8개 이하, 그림 설명 350단어 이하, 참고문헌 60개(엄격 적용은 아님)다. 본문·초록·제목 한도는 '강하게 권고'하는 지침(Format of articles)으로 적혀 있고, 표시 항목은 'limited to 8' 로 적혀 있다. 제출 점검표(PDF)에는 단어 한도 항목이 없고 구조 요건만 있다.
- Methods 에는 단어 한도가 없다. 'Code availability' 제목의 진술은 Methods 안에, Data availability 는 본문 끝과 참고문헌 사이에, 저자 기여와 그림 설명은 참고문헌 뒤에 둔다. 대형 언어 모델 사용은 Methods 에 적어야 한다.
- 기존 초안은 있다(사용자 질문 4). 영문 마크다운 초안 두 편이 핵심이다. 방법·서론 초안은 서론 797단어, Methods 6,998단어이고, 결과·초록 초안은 초록 198단어, Results 7,500단어다. 둘 다 09-30 LG 판정 순서로 쓰였고 WF 실험(10-01–10-02)은 들어 있지 않다.
- 본문 한도(4,500) 대비 현재 서론 + Results 는 약 8,300단어다. Results 를 약 2,900단어로 줄이고 Discussion 을 약 850단어로 새로 써야 한다.
- 자리표시는 방법 초안에 [verify] 19, [UPDATE] 12, [RESULT] 6개, 결과 초안에 [MISSING] 30개(목록 21건), [DECISION] 12개(목록 7건)가 있다.
- 기준 문서(10-01 주장 문서, 10-02 질의응답)와 초안 사이 충돌 17건을 찾았다. 큰 것은 틀(가설 순서 대 라벨 수별 능력과 워크플로), C1 의 '안전' 표현, C2 의 범위, 관측 위치·알래스카 지역 내 문장의 '쓰지 않을 문장' 목록 충돌, 라벨 0 의 CCI 앵커 권고와 AB2 의 충돌이다.
- 참고문헌은 67행이다. 추가할 5편(Ploton 2020, Meyer & Pebesma 2021·2022, Wadoux 2021, Linnenbrink 2024)의 서지를 Crossref 와 출판사 메타데이터로 확인했다. 추가하면 72편으로 권고치 60을 넘는다.
- 빌드는 가능하다. Springer Nature 템플릿(v3.1, 2024-12)을 내려받아 시험했다. `sn-jnl[pdflatex,sn-nature]` 를 xelatex 로 컴파일하면 xeCJK 한글 포함 오류 0건이다. 다만 TinyTeX 에 패키지 5종(threeparttable, ncctools, jknapltx, rsfs, appendix)이 없고, xelatex 의 EPS 변환은 GLIBC 2.27 때문에 실패하므로 그림은 PDF 만 쓴다.

---

## 2. Scientific Reports 투고 규정

### 2.1 확인 경로

| 자료 | URL | 확인 방법 |
|---|---|---|
| 투고 지침 | https://www.nature.com/srep/author-instructions/submission-guidelines | WebFetch 는 idp.nature.com 쿠키 우회로 요약만 반환했다. 원문은 curl(쿠키 저장)로 받아 HTML 에서 문장을 그대로 뽑았다 |
| 제출 준비 쪽 | https://www.nature.com/srep/author-instructions/ready-to-submit | curl |
| 첫 제출 점검표 | https://www.nature.com/documents/srep-checklist-for-initial-submissions.pdf | curl, pdftotext |
| 편집 정책 | https://www.nature.com/srep/journal-policies/editorial-policies | WebFetch(요약)와 curl(원문) |
| LaTeX 지원 쪽 | https://www.springernature.com/gp/authors/campaigns/latex-author-support | WebFetch |
| 템플릿 압축 파일 | https://cms-resources.apps.public.k8s.springernature.io/springer-cms/rest/v1/content/18782940/data/v12 | curl. 파일명 'Download the journal article template package (December 2024 version).zip', sha256 812e76dc…a3ecae, 901,814 바이트 |
| Overleaf 템플릿 | https://www.overleaf.com/latex/templates/springer-nature-latex-template/gsvvftmrppwq | LaTeX 지원 쪽에 적힌 주소. 열지 않았다 |

### 2.2 규정 원문과 원고 적용

| 항목 | 원문(그대로 인용) | 출처 | 원고 적용 |
|---|---|---|---|
| 논문 유형 | "Scientific Reports publishes original research in two formats: Article and Registered Report." | 투고 지침, Format of articles | Article 로 낸다. 우리 사전 등록은 저널의 Registered Report(1단계 원칙 승인)가 아니므로 'internally pre-registered' 로 쓴다 |
| 분량 성격 | "In most cases, we do not impose strict limits on word count or page number. However, we strongly recommend that you write concisely and stick to the following guidelines: Articles should ideally be no more than 11 typeset pages" | 같은 곳 | 11쪽은 권고다. 첫 컴파일 뒤 쪽수를 잰다 |
| 본문 한도 | "The main text should be no more than 4,500 words (not including Abstract, Methods, References and figure legends)" | 같은 곳 | 서론 + Results + Discussion ≤ 4,500 |
| 제목 | "The title should be no more than 20 words, should describe the main message of the article using a single scientifically accurate sentence, and should not contain puns or idioms" | 같은 곳 | 4.2절 |
| 초록 한도 | "The abstract should be no more than 200 words" | 같은 곳 | 4.3절 |
| 의무 한도의 정본 | "For a definitive list of which limits are mandatory please visit the submission checklist page." | 같은 곳 | 점검표 PDF 에는 단어 한도 항목이 없다(2.3절). 한도는 권고로 다루되 지킨다 |
| 초록 형식 | "Please do not include any references or figures in your Abstract. Make sure it serves both as a general introduction to the topic and as a brief, non-technical summary of the main results and their implications. Abstract should be unstructured, i.e. should not contain sections or subheadings." | 투고 지침, Abstract | 인용·기호 없이 쓴다 |
| 핵심어 | "We allow the use of up to 6 keywords/key phrases that can be used for indexing purposes." | 투고 지침, Keywords | 6개 이하 |
| 본문 구성 | "For the main body of the text, there are no specific requirements. … the following structure will be suitable in many cases: Introduction / Results (with subheadings) / Discussion (without subheadings) / Methods" | 투고 지침, Manuscript | 이 순서를 따른다 |
| 본문 뒤 순서 | "References (limited to 60 references, though not strictly enforced) / Acknowledgements (optional) / Author contributions / Data availability statement (mandatory) / Additional Information (including a Competing Interests Statement) / Figure legends (these are limited to 350 words per figure) / Tables (maximum size of one page)" | 같은 곳 | Data availability 위치는 아래 행과 다르다 |
| Data availability 위치 | "You must include a Data Availability Statement in all submitted manuscripts (at the end of the main text, before the References section)" | 투고 지침, Data availability | 점검표 PDF 도 "at the end of the main text, before the 'References' section" 이다. 위 목록 순서와 어긋나므로 점검표를 따른다 |
| 각주 | "Please note, footnotes should not be used." | 투고 지침, Manuscript | Table 1 v2 의 '각주 e' 는 표 설명문으로 옮긴다 |
| 쪽·행 번호 | "Please consider including those in your manuscript" | 같은 곳 | 템플릿 `lineno` 선택지를 쓴다 |
| 표시 항목 | "Display items are limited to 8 (figures and/or tables)." | 같은 곳 | 현행 Fig 1–7 + Table 1 = 8 |
| LLM 사용 | "Use of an LLM should be properly documented in the Methods section (and if a Methods section is not available, in a suitable alternative part) of the manuscript." | 같은 곳 | Methods 끝에 사용 진술을 둔다 |
| 첫 제출 파일 | "you may incorporate the manuscript text and figures into a single file up to 3 MB in size. Whilst Microsoft Word is preferred we also accept LaTeX, or PDF format." / "Supplementary information should be combined and supplied as a single separate file, preferably in PDF format." | 같은 곳 | v2 그림 PDF 8개 합계 1,382,166 바이트(`outputs/figures/paper/v2/*.pdf`)라 3 MB 안이다 |
| LaTeX | "For LaTeX submissions we encourage authors to use the Springer Nature LaTeX template" / "Please ensure before submission that the complete .tex file compiles successfully with no errors or warnings." | 투고 지침, Manuscript 와 TeX/LaTeX files | 경고 0건을 빌드 점검 항목에 넣는다 |
| 수정본 | "we do not accept PDF files for the article text of revised manuscripts" / "Format the manuscript file as single-column text without justification." / "Use the default Computer Modern fonts for your text" | 투고 지침, Revised manuscripts | 영문판은 Computer Modern 계열(Latin Modern) 유지 |
| Methods | "We don't impose word limits on the description of methods. Make sure it includes adequate experimental and characterisation data for others to be able to reproduce your work." | 투고 지침, Methods | 4.5절 |
| 코드 공개 | "The Methods section must include a statement with the heading ‘Code availability’ that describes how readers can access the code or algorithm, including any restrictions to access." | 편집 정책, Availability of computer code and algorithm | Code availability 를 Methods 마지막 소절로 둔다 |
| 자료 공개 | "Data availability statements should include information on where data supporting the results reported in the article can be found including, where applicable, hyperlinks to publicly archived datasets analysed or generated during the study." / "Where figure source data are provided, statements confirming this should be included in data availability statements." | 편집 정책, Availability of materials and data | source_data 기탁을 진술에 넣는다 |
| 저자 기여 | "You must supply an Author Contribution Statement" / 점검표: "This should be included after the references, and every author’s contribution must be listed" | 투고 지침, 점검표 | 참고문헌 뒤 |
| 이해상충 | "If there is no conflict of interest, you should include a statement declaring this." 예시: "The author(s) declare no competing interests." / 점검표: "Competing interests statement is provided in the Manuscript File under the heading “Additional Information”" | 투고 지침, 점검표 | Additional Information 제목 아래 |
| 감사의 글 | "Please keep any acknowledgements brief, and don’t include thanks to anonymous referees and editors, or any effusive comments." | 투고 지침 | 짧게 |
| 그림 설명 | "Please begin your figure legends with a brief title sentence for the whole figure and continue with a short description of what is shown in each panel. … Keep each legend total to no more than 350 words." / 점검표: "These should be placed at the end of the manuscript, after the references" | 투고 지침, 점검표 | v2 설명문 230–319단어(4.6절) |
| 표 | "Please submit any tables in your main article document in an editable format (Word or TeX/LaTeX, as appropriate), and not as images. Tables that include statistical analysis of data should describe their standards of error analysis and ranges in a table legend." | 투고 지침, Tables | `outputs/figures/paper/v2/Table1_data.tex` 를 쓴다 |
| 수식 | "Include any equations and mathematical expressions in the main text of the paper. Identify equations that are referred to in the text by parenthetical numbers, such as (1), and refer to them in the manuscript as "equation (1)" etc." | 투고 지침, Equations | 본문 수식은 Stefan 식 등 최소로 |
| 참고문헌 형식 | "we use the standard Nature referencing style … Run sequentially (and are always numerical). Sit within square brackets." / "Include all authors unless there are six or more, in which case only the first author should be given, followed by 'et al.'." / "Only include papers or datasets that have been published or accepted by a named publication, recognised preprint server or data repository" | 투고 지침, References | `sn-nature.bst` 사용. 자료 인용도 참고문헌에 든다 |
| 보충 자료 | "We do not support inclusion of any additional information or appendices in the main body of the paper" / "Number Supplementary Tables and Figures as, for example, "Supplementary Table S1"." / "Be sure to include the word "Supplementary" each time one is mentioned. Please do not refer to individual panels of supplementary figures." / "We do not edit, typeset or proof Supplementary Information" / "a maximum size of 50 MB" / "Please avoid including any "data not shown" statements" | 투고 지침, Supplementary Information | 4.12절 |
| 그림 형식 | "supply all line art, graphs, charts and schematics in vector format, such as EPS or AI" / "All digitized images submitted with the final revision of the manuscript should be 300 DPI if possible." / "Use a clear, sans-serif typeface (for example, Helvetica) for figure lettering." / "Figures divided into parts should be labelled with a lower-case, bold letter" / "Units should have a single space between the number and the unit" / "Commas should be used to separate thousands … this applies to five or more digits." / "Include a description of the statistical treatment of error analysis in the figure legend." | 투고 지침, Figure guidelines·Figures for publication | 그림 QA 항목으로 쓴다. 네 자리 수(예: 1,000, 3,037)의 쉼표는 이 규칙상 빼는 것이 맞다 [미확인: 본문에도 같은 규칙을 적용하는지] |
| 통계 | "it should state the name of the statistical test, the n value for each statistical analysis, the comparisons of interest, … the alpha level for all tests, whether the tests were one-tailed or two-tailed, and the actual P value for each test (not merely "significant" or "P < 0.05")" / 다중 비교: "you should explain how you adjusted the alpha level to avoid an inflated Type I error rate" / 작은 표본: "when the sample size is small (less than about 10), you should use tests appropriate to small samples or justify the use of large-sample tests." | 투고 지침, Statistical guidelines | 보정 전 p 를 모든 대비에 제공해야 한다. 결과 초안 [MISSING] 19(AB1–AB10 의 보정 전 p)를 채워야 한다. 지역 4–5개 추론은 작은 표본 항목에 해당한다 |
| 표지 편지 | "The affiliation and contact information of your corresponding author / A brief explanation of why the work is appropriate for Scientific Reports / The names and contact information of any reviewers you consider suitable / The names of any referees you would like excluded from reviewing" | 투고 지침, Cover letter | 대회 보고서 공개 사실을 적는다 |
| 관련 선행 공개 | "If similar or related work has been published or submitted elsewhere, then you must provide a copy with the submitted manuscript." | 편집 정책, Submission policies | 대회 보고서(`submission/` 동결본) 사본을 함께 낸다 |
| 프리프린트 | "Scientific Reports allows and encourages prior publication on recognized community preprint servers for review by other scientists in the field before formal submission to a journal." | 편집 정책 | 선택 사항 |
| Reporting Summary | 투고 지침 쪽과 점검표 PDF 에서 'Reporting Summary' 문구를 찾지 못했다 | 투고 지침, 점검표 | `docs/PAPER_SCIREP_FRONTMATTER_DRAFT.md` 5절의 '환경과학이면 Reporting Summary 요구' 서술은 근거를 확인하지 못했다 [미확인]. 통계 지침 항목으로 대신한다 |

### 2.3 점검표 PDF 의 의무 구조 요건(요약)

`srep-checklist-for-initial-submissions.pdf` 의 항목이다. 원고 파일은 하나(.doc, .docx, .tex, .pdf), 표는 원고 파일 안에 편집 가능한 형식, 보충 자료는 별도 파일이다. 초록은 인용과 소제목이 없어야 한다. 본문에 각주가 없어야 한다. 저자 기여는 참고문헌 뒤, 이해상충은 'Additional Information' 제목 아래, 본문 그림·표의 설명은 참고문헌 뒤, Data Availability 는 참고문헌 앞이다. 단어 한도는 점검표에 없다.

### 2.4 Springer Nature LaTeX 템플릿

| 항목 | 내용 | 근거 |
|---|---|---|
| 판 | 템플릿 첫 줄 "%Version 3.1 December 2024" | 압축 파일 안 `sn-article.tex` |
| 파일 | `sn-jnl.cls`, `sn-article.tex`, `sn-bibliography.bib`, `bst/sn-nature.bst` 외 8종, `user-manual.pdf` | 압축 파일 목록 |
| Nature 계열 선택지 | "\documentclass[pdflatex,sn-nature]{sn-jnl}% Style for submissions to Nature Portfolio journals" (주석 처리된 예시 줄) | `sn-article.tex` 31행(줄 끝을 LF 로 바꾼 사본 기준) |
| 단일 파일 제출 | "Please do not use \input{...} to include other tex files." / "Submit your LaTeX manuscript as one .tex document." | `sn-article.tex` 6–7행 머리 주석 |
| 적용 범위 | "can be used to prepare your journal article submission to any Springer Nature journal inclusive of Springer, Nature Portfolio, and BMC." | LaTeX 지원 쪽(WebFetch 요약 인용) |
| 참고문헌 제출 | .bbl 또는 .bib 를 'Manuscript' 항목으로 올리고 .bst 도 함께 낸다. 또는 참고문헌을 본문 .tex 에 붙여 넣는다 | LaTeX 지원 쪽(WebFetch 요약, 원문 문장 대조 안 함) [미확인: 원문 문장] |
| 선언 절 예시 | 템플릿 `\backmatter` 아래 'Declarations' 목록에 Conflict of interest/Competing interests, Data availability, Materials availability, Code availability, Author contribution 이 있다 | `sn-article.tex` 1101–1125행 |

Scientific Reports 의 절 순서(2.2절)가 템플릿의 일반 'Declarations' 목록보다 우선한다.

---

## 3. 기존 원고 초안 현황 (사용자 질문 4의 답)

### 3.1 파일 목록

| 파일 | 언어·형식 | 작성 | 담긴 것 | 상태 |
|---|---|---|---|---|
| `docs/MANUSCRIPT_DRAFT_METHODS_INTRO_2026-09-30.md` | 영문 본문, 국문 머리말, 마크다운 | 2026-09-30 | Introduction, Methods(13소절), Data availability, Code availability, References 표(67행, 확인 상태 열) | 결과 수치 없음. WF 없음 |
| `docs/MANUSCRIPT_DRAFT_RESULTS_ABSTRACT_2026-09-30.md` | 영문 본문, 국문 머리말·충돌 절, 마크다운 | 2026-09-30, 10-02 LGF 자리 채움 | Abstract(198단어), Results(10소절), Discussion 골격(D1–D6), 주장 점검표, 자리표시 목록, 원천 충돌 11건 | LG 가설 순서. WF 없음(머리말 4행 "새 연구 주제(워크플로)와 WF 결과는 아직 이 초안에 반영하지 않았다") |
| `docs/MANUSCRIPT_DRAFT_SUPPORT_2026-09-30.md` | 영문, 마크다운 | 2026-09-30 | M3 Stefan 계수 E 의 Methods·Discussion 문단, M4 SI 표 틀(가설 등록표 88행, 과거 음성 결과, 선택·철회 이력), M5 동등성 한계·S-a·S-b·SC3w 문단 | 표 상태 열이 [RESULT] 로 비어 있다 |
| `docs/PAPER_SCIREP_FRONTMATTER_DRAFT.md` | 국문 설명 + 영문 진술, 마크다운 | 2026-09-21 | Data/Code availability, Declarations, 통계 명세(H1–H17), Reporting Summary 초안, Gautam 2025 대비 문단, 용어 정정 목록 | M1 시기 설계 기준이라 상당 부분이 낡았다(3.6절) |
| `docs/MANUSCRIPT_RESTRUCTURE_PLAN_2026-10-02.md` | 국문, 마크다운 | 2026-10-02 | 새 틀에 맞춘 결과 절 R1–R8, 그림 배치안(Fig 4·5·7 교체), 제목 후보 3, 사용자 결정 3건 | 결정 대기 |
| `outputs/report/main.tex` | 국문 LaTeX(xelatex + xeCJK, 2단) | 대회 예선(2026-07) | 대회 보고서 전문, 그림 12, 표 8, 식 9 | 대회 시기 주장(예: 초록 101행 "24.11에서 22.92 cm", "커버리지 93.4%")이라 본문 재사용은 제한적이다. 전처리 블록(6–33행)은 국문판에 재사용한다 |
| `outputs/figures/paper/CAPTIONS.md` | 영문 그림 설명 | 09-26(구판)과 09-30–10-02(v2) | 구판 Fig 1–7, Table 1, SI 그림·표 설명과 v2 Fig 1–7, Table 1 설명 | 원고에는 v2 설명만 쓴다 |
| `figures/figure_spec.json` | JSON | 판 "2026-09-30 F1(표시 항목 확정)" | 본문 표시 항목 8개 명세, SI 항목 40개, 조건부 규칙 10개 | 상태 문자열 "결과 열람 전. … 수치와 판정 문장은 없다" 가 v2 산출 뒤에도 남아 있다 |

재구성안 비교 폴더(`docs/QA_FINAL_REVIEW_2026-10-02.md` Q10 의 `paper/figures/` '현행 v2 와 재구성안')는 2026-10-04 13:00 점검 때 없었다. `paper/` 에는 다른 작업이 만든 `claims/` 만 있었다.

### 3.2 절별 단어 수

방법: 마크다운을 제목(##, ###) 단위로 나누고 공백 기준 토큰을 셌다. '전체'는 표 행 포함, '본문'은 표 행(`|` 로 시작)과 HTML 주석을 뺀 값이다. 국문 LaTeX 는 명령과 표·그림 환경을 지운 뒤 어절을 셌다.

| 파일 | 절 | 전체 | 본문 |
|---|---|---|---|
| 방법·서론 초안 | Introduction | 797 | 797 |
| 방법·서론 초안 | Methods(소절 합) | 6,998 | 6,208 |
| 방법·서론 초안 | 그중 Scoring and statistical inference | 1,562 | 1,313 |
| 방법·서론 초안 | 그중 Observational data | 794 | 794 |
| 방법·서론 초안 | 그중 Computing environments | 712 | 623 |
| 방법·서론 초안 | 그중 Additional independent regions (LGD) | 698 | 698 |
| 방법·서론 초안 | 그중 Learning methods | 648 | 403 |
| 방법·서론 초안 | 그중 Predictive uncertainty (LGU) | 490 | 490 |
| 방법·서론 초안 | 그중 LGT·LGF | 380 | 380 |
| 방법·서론 초안 | 그중 Pre-registration | 346 | 346 |
| 방법·서론 초안 | 그중 Study design / Extension contrasts / Targets / Physics baselines / 머리 | 336 / 326 / 306 / 230 / 117 | 129 / 326 / 306 / 230 / 117 |
| 방법·서론 초안 | Data availability | 634 | 634 |
| 방법·서론 초안 | Code availability | 162 | 162 |
| 결과·초록 초안 | Abstract 문단 | 198 | 198 |
| 결과·초록 초안 | Results(소절 합) | 7,500 | 7,211 |
| 결과·초록 초안 | 그중 Learners and validation design [SI] | 1,475 | 1,475 |
| 결과·초록 초안 | 그중 Zero-label transfer and prediction intervals | 1,178 | 1,178 |
| 결과·초록 초안 | 그중 Use of physics information | 1,119 | 1,119 |
| 결과·초록 초안 | 그중 Deployment scenarios and abstract contrasts | 800 | 511 |
| 결과·초록 초안 | 그중 Error as a function of the label count | 740 | 740 |
| 결과·초록 초안 | 그중 Additional independent regions | 726 | 726 |
| 결과·초록 초안 | 그중 Evaluation overview / Minimum label count / Sensitivity / Placement | 423 / 395 / 385 / 182 | 같음 |
| 결과·초록 초안 | Discussion 골격(D1–D6) | 2,012 | 1,076 |
| 지원 초안 | M3.2 Discussion 문단 D1–D8 | 1,148 | 1,148 |
| 지원 초안 | M3.1 E0·E_n Methods 문단 | 897 | 722 |
| 지원 초안 | M5.4–M5.6(S-a, S-b, SC3w) | 593 / 352 / 601 | 같음 |
| 대회 보고서 main.tex | 초록 / 서론 / 데이터 / 방법 / 결과 / 논의 / 결론(어절) | 334 / 381 / 614 / 781 / 2,838 / 383 / 325 | 표·그림 환경 제외 |

사용자에게 말한 '방법·결과 초안이 각각 약 7,000단어'는 위 Methods 6,998단어, Results 7,500단어(본문 7,211)를 가리킨다.

### 3.3 Scientific Reports 한도와의 비교

| 항목 | 한도 | 현재 | 차이 |
|---|---|---|---|
| 본문(서론 + Results + Discussion) | 4,500 | 서론 797 + Results 7,500 = 8,297(Discussion 은 골격이라 제외) | 약 3,800단어 초과 + Discussion 신규 |
| 초록 | 200 | 198 | 안. 다만 WF 와 새 틀이 없다 |
| Methods | 없음 | 6,998 | 한도는 없으나 11쪽 권고 때문에 4,000단어 안팎으로 줄이고 나머지는 Supplementary Methods 로 옮긴다(4.5절) |
| 표시 항목 | 8 | v2 Fig 1–7 + Table 1 = 8(`figures/figure_spec.json` spec_meta.main_display_items) | 꽉 찼다. 새 그림은 기존 그림의 패널로만 넣는다 |
| 그림 설명 | 350 | v2 230(Fig 1), 295(Fig 2), 301(Fig 3), 306(Fig 4), 230(Fig 5), 319(Fig 6), 263(Fig 7), 84(Table 1)(`outputs/figures/paper/CAPTIONS.md` 의 words 주석) | 안 |
| 참고문헌 | 60(엄격 적용 아님) | 67 + 추가 5 = 72 | 12 초과(4.10절) |

### 3.4 자리표시 태그

| 파일 | 태그별 개수 |
|---|---|
| 방법·서론 초안 | [verify] 19, [UPDATE] 12, [RESULT] 6, [old result, recheck] 3, [TABLE] 1 |
| 결과·초록 초안 | [MISSING] 30(본문 22, 목록 절 포함), [RESULT] 20(대부분 10-02 에 채운 기록), [DECISION] 12 |
| 지원 초안 | [RESULT] 117(SI 표 상태 열), [VERIFY] 11, [CITE] 5 |
| 앞부분 초안 | [확인 필요] 14, [자리표시자] 4 |
| 재구성안, main.tex, CAPTIONS.md | 없음 |

**방법·서론 초안의 열린 [RESULT] 3곳**: 23행(서론 끝 결과 문단), 153행(Table 3 결과 열. 결과 초안의 Fig 7d 표로 채울 수 있다), 201행(교차 환경 점검 (i)–(iii) 결과. 결과 초안 74행은 점검 (i)이 CatBoost 에서 통과하지 못했다고 적고, 106행은 점검 (iii)에서 키의 14.9 % 가 0.5 cm 넘게 달랐다고 적는다).

**방법·서론 초안의 [UPDATE] 12개**: 약관 회신과 날짜(71, 213행), 공개 저장소 첫 push 날짜(181행), 사후 개정 4건의 수용 결정(183행), S-dev 표(183행), LG 집계 환경(187행), RealMLP 조각 완료 상태(187행), KPDC 자료 포함 여부(215행), Zenodo DOI(217, 221행), 공개 시점(221행), 그리고 머리말 정의.

**[verify] 19개 가운데 본문 쪽(11개, 머리말 정의 1개 포함)**: 서론의 Pilyugina 분류(15행), CALM·ALLena 범위 규칙(49행), Stefan 1891 서지(55행), 융해관 규칙 정합(65행, 새 지역에서는 직접 측정으로 넣고 v3 CALM 에서는 뺐다), Kudryavtsev 1974 서지(161행), LGF 설정 수(177행), 티베트 현장 자료 약관(211행), SoilGrids 판 태그·CCI 범위 자료 DOI·Pekel 2016(215행). 나머지는 참고문헌 표(4.10절).

**결과 초안의 [MISSING] 21건**(313–347행 목록): R3 − P1*(n 40, 160), P1* 의 n 3·320·전량 여부, R1 − P* 지역 행, L31 의 두 지역 이름, LGX-N2 오차 하한·LGX-N4 지역 추론·SD/SE 비, P2·P3 곡선, n* = 3 행의 라벨 순가치, 대상별 4분 판정 수, PE2 상대 단위, 새 지역 평균의 구성, Δ_L43 구간, L23 (a) 구간, D0 − P* 와 R0 − P*(n 0), 정규화기별 커버리지·폭, L26 행의 n·λ, FT-T 열세 2행과 cfm 우세 1행, LGT 의 P1*·P* 대비, Δ_SC3 와 멈춘 단계, AB1–AB10 의 블록 등가중 구간과 보정 전 p, δ_rel 표지, L28 변형별 행.

**결과 초안의 [DECISION] 7건**(351–357행): 초록의 AB2 포함, 초록의 알래스카 3지역 한정어 표현, L25 문장과 LGU 10절 충돌, 배치 문장(L43 미결정), 토의의 L29 와 연도 불일치 항 연결, NOVELTY 4절 옛 수치의 SI 이관, 티베트 해석.

### 3.5 재사용 가능한 문단

| 출처(파일:행) | 내용 | 재사용 | 필요한 수정 |
|---|---|---|---|
| 방법·서론 초안 13행 | 서론 1: ALT 정의, 관측 희소, 라벨 셀 수 3–13,606 | 수정 후 사용 | 'mapping project' 를 새 지역 라벨 0–수백 개 틀로 바꾼다 |
| 같은 파일 15행 | 서론 2: Stefan 식과 ML·결합 선례, 기법 신규성 부정 | 수정 후 사용 | Pilyugina 분류 [verify] 해결 |
| 같은 파일 17행 | 서론 3: Gautam 2025, 거리 블록 검증, 라벨 수 함수 평가의 부재 | 수정 후 사용 | Ploton 2020, Meyer & Pebesma 2022, Wadoux 2021 문장 추가(검증 사다리 근거) |
| 같은 파일 19행 | 서론 4: 지역 제외 + 100 km, 두 기준선 | 사용 | 지역 내 블록 홀드아웃(WF) 설계 한 문장 추가 |
| 같은 파일 21, 23행 | 방법 범위, 사전 등록 문단 | 축약 | WF 의 '결과 열람 뒤 설계' 표지와 XA–XJ 등록 문장 추가 |
| 같은 파일 49–61행 | Labels, Regions, Label unit, Covariates, Scoring mask | 사용 | SoilGrids 추출 해상도 정정(QA Q8: "약 5 km 로 요청해 추출. 문서의 250 m 표기는 정정 필요"). `scripts/1_data_prep/enrich_soilgrids_wcs.py` 55·100–103행은 지역 창을 res_m 5000(약 5 km)으로 요청하고, `enrich_soilgrids_cell.py` 는 그 창(`data/raw/soilgrids_multi`)에서 최근접 표본을 뽑는다. Methods 에 '250 m 제품을 약 5 km 로 재표본해 추출'로 적는다 |
| 같은 파일 75–83행 | E0, E_n, κ, P0–P3 정의와 식 (2)(3) | 사용 | P*, P1* 정의 추가 |
| 같은 파일 87–104행 | 결합 구조 표(D0, D1, R0–R3, V1) | 축약 | 본문 기호 5개 규칙(4.7절)에 맞춰 이름 위주로 |
| 같은 파일 110–112행 | 대상, 분할, 라벨 추출 | 사용 | |
| 같은 파일 116–136행 | 채점식, 블록 부트스트랩, 4분 판정, n*, 초록 묶음 Holm 규칙 | 사용(핵심) | 보정 전 p 제공 문장 추가(통계 지침) |
| 같은 파일 165–177행 | LGU, LGT·LGF | Supplementary Methods 로 | |
| 같은 파일 181–203행 | 사전 등록 시각, 교차 환경 규칙, 표 4 | 축약 후 일부 SI | Rescale CPU 노드(elm, hematite)와 재현 수치(3.6e-14 cm, 5.6 cm; `docs/RESEARCH_OVERVIEW_2026-10-02.md` 4절) 추가 |
| 같은 파일 209–221행 | Data availability, Code availability | 수정 후 사용 | SAR 자료(WF1-b) 추가, h54 등 WF 하네스 추가, Code availability 를 Methods 안으로 |
| 지원 초안 46행 | E0·E_n 붙여 넣기 문장 | 사용 | |
| 지원 초안 50–64행 | Discussion D1–D8(E 의 물리적 의미, 지역 간 이전성) | 축약해 Discussion 2문단으로 | [CITE] 5건 서지 확인 |
| 지원 초안 244–252행 | 동등성 한계 문장과 공개 문장 | 사용 | |
| 지원 초안 281–312행 | S-a, S-b, SC3w Methods·SI 문단 | SI | |
| 지원 초안 93–185행, 211–236행 | SI 가설 등록표, 선택·철회 이력 | SI | WF0–WF10, XA–XJ 행 추가, 상태 열 채우기 |
| 결과 초안 15행 | 초록(AB1–AB10 나열) | 대체 | 4.3절 설계로 다시 쓴다 |
| 결과 초안 50, 52, 60, 92행 | 위약 대조, 잔차 대 물리 입력, 재보정, 라벨 0 학습기 문단 | 축약해 R2·R3 에 | 기호 줄이기 |
| 결과 초안 74–82행 | 새 독립 지역 | 축약해 R7 에 | |
| 결과 초안 96–102행 | 예측 구간, 공간 진단 | 축약해 R8 에, 나머지 SI | L25 문장 결정 |
| 결과 초안 106–114행 | 학습기 비교, 검증 설계(L19–L21) | SI, L19·L20 은 R1(검증 사다리)에 | |
| 결과 초안 118–126행 | 배포 시나리오 SC1w–SC3w, S-a·S-b | R3·R6 에 일부, 나머지 SI | |
| 결과 초안 193–205행 | 한계 13항 | Discussion 한계 문단의 원천 | WF 한계 추가 |
| 결과 초안 209–234행 | 쓰지 않을 문장 표 | 내부 점검표로 유지 | 4.8절과 합친다 |
| 앞부분 초안 119, 125, 127행 | Acknowledgements, Competing interests, Ethics | 수정 후 사용 | KPDC 자료를 안 쓰면 KPDC 감사 문구 조정 |
| 앞부분 초안 311행 | 표지 편지의 선행 공개 문장 | 수정 후 사용 | 헤드라인 수치 교체 |
| 앞부분 초안 290행 | Gautam 2025 대비 영문 문단 | 다시 쓴다 | "prediction intervals are calibrated by conformalized quantile regression with verified coverage" 는 LGU 판정(A1·B2 미결정)과 맞지 않는다 |
| main.tex 6–33행 | xelatex·xeCJK·글꼴 전처리 | 국문판 전처리로 사용 | 2단을 1단으로 |
| main.tex 123행 | 연구 배경 문단(탄소, 기반시설, 온난화) | 국문 서론 첫 문단 재료 | 인용 번호 체계 교체, 수치 출처 재확인 |
| CAPTIONS.md v2 절(226–276행) | v2 Fig 1–7, Table 1 설명 | 사용 | AB·L 표지와 기호를 본문 규칙에 맞춘다 |

### 3.6 기준 문서와의 충돌

기준 문서는 `docs/RESEARCH_CLAIMS_WORKFLOW_2026-10-01.md`(이하 '주장 문서')와 `docs/QA_FINAL_REVIEW_2026-10-02.md`(이하 'QA')다. 처리 규칙은 4.8절에 모았다.

| 번호 | 초안 위치 | 초안 내용 | 충돌 내용 | 처리 |
|---|---|---|---|---|
| 1 | 결과 초안 전체, 방법 초안 23행 | 기여 = 평가 설계와 등록 결과. 가설 순서(곡선, 물리 정보, 최소 라벨, 배치, 라벨 0, 학습기, 배포) | 주장 문서 1절: 주제 = 라벨 수별 능력과 새 지역 워크플로. 재구성안 1절 | 결과 순서를 R1–R8(4.4절)로 바꾼다 |
| 2 | 결과·방법 초안 전체 | WF0–WF10 없음(WF 언급 0회) | 주장 문서 C1, C2, C4–C6, C8 의 근거 다수가 WF | WF Methods 소절과 결과 절 신설 |
| 3 | 결과 초안 214행(쓰지 않을 문장) | 'Recalibration with 3–10 labels is safe', SC1w '안전성 미확인(4/10)' | 주장 문서 31행 C1 "ML 을 안전하게 쓰는 방법이다". QA 결정: 등록된 비열등 없이는 'safe' 금지 | '손해를 줄였다'(위험표 수치)로 쓰고 'safe' 는 쓰지 않는다 |
| 4 | 주장 문서 45–47행 | C2 "ML 의 이득은 물리 계수가 틀린 정도에 비례" | QA 43행: 재보정을 넘어선 ML 몫과 계수 오차의 순위상관 −0.08(p 0.67, 30대상, 사후 점검) | C2 를 '라벨을 쓰는 보정(재보정 + 잔차)의 이득'으로 좁힌다. XA 로 확인 |
| 5 | 결과 초안 222행 | 'Where to measure', 'observation priority' 는 쓰지 않는다(WRAPUP 4, 11.2) | 주장 문서 C6, 워크플로 표 '다음 관측' 열 | WF2·WF8 은 결과 열람 뒤 설계 표지를 달고 방법 수준 규칙으로만 쓴다. L43(등록, 미결정)을 함께 적는다 |
| 6 | 결과 초안 221행 | 'Exceeds within Alaska' 는 쓰지 않는다(WRAPUP 5, 11.2) | 주장 문서 C4: 알래스카 지역 내 WF6-a 지지(−0.39, −0.54 cm) | WRAPUP 규칙은 WF6 이전의 L6·AK1w 문맥이다. WF6 은 별도 등록 시험이므로 '지역 내 블록 홀드아웃에서 재보정 물리식보다 오차가 작았다'로 쓰고 표지를 단다 |
| 7 | 결과 초안 215행, 지원 초안 216행(N8 폐기) | 검증 설계 결과를 순위·부호 '역전'으로 쓰지 않는다(L19) | 주장 문서 C7 "평가 설계가 결론을 바꾼다", QA Q15 "무작위 분할에서 1등처럼 보인 CatBoost 가 지도 상황에서는 Stefan 보다 나쁘다" | 수치(검증 사다리)만 보고하고 '역전' 단어를 쓰지 않는다. XH 로 CI 를 붙인다 |
| 8 | 결과 초안 94행, `docs/PAPER_PLAN_SCIREP.md` 8절 K2 | 라벨 0 에서 제품·앙상블 기준선은 P0 대비 미결정 또는 열세. AB2 −2.73 미결정 | 주장 문서 96행 워크플로 표: 라벨 0 에 "위성 ALT 제품이 있으면 Stefan·CCI 평균 앵커"(WF7 서술) | 라벨 0 권고에서 CCI 앵커는 빼거나 '서술, 등록 대비 AB2 미결정'으로 병기한다. WF7 의 '약 −3 cm' 는 라벨 10개(−2.82)와 전량(−3.01)의 Re − P0 값이고, 라벨 0 의 값은 Pe − P1(= Pe − P0) −1.17, Re − R1 −1.19 로 둘 다 미결정(서술)이다(`docs/EXPERIMENT_PLAN_WF_2026-10-01.md` 139–140행) |
| 9 | 결과 초안 165–174행 | 배포 표 T0–T40(P0, P1, R1) | 주장 문서 4절 워크플로 표(0, 1–10, 10–160, 160–1,000, 1,000 이상; 규칙 W, P*, 진단) | 워크플로 표로 바꾸고 SC1w·SC1w-P 판정을 같이 적는다. 라벨 단위 '1 km 위치 평균'(S-a) 열 추가 |
| 10 | 결과 초안 234행 | 제목에 'how many' 를 쓰지 않는다(RESEARCH_FRAME A.1) | 재구성안 2절 제목 후보 1 "How many labels does …" | 후보 1 은 쓰지 않는다 |
| 11 | 앞부분 초안 290행, 6.4절 표 | CQR 커버리지 93.4 %, AOA, 주 6지역·심부 4지역 LORO | LGU-A1·B2 미결정, 생성 구간 과소 커버리지(결과 초안 96–98행). 현행 설계는 주 4지역 | 문단을 새로 쓴다 |
| 12 | 앞부분 초안 104행 | TabPFN 은 라이선스 토큰 부재로 제외 | LGT 가 TabPFN v2 를 실행했다(방법 초안 175행) | 낡은 서술. 쓰지 않는다 |
| 13 | 방법 초안 209–215행 Data availability | x25 자료만 적는다 | WF1-b 가 알래스카 지역 내에서 SAR 9열(x34)을 썼다(`docs/EXPERIMENT_PLAN_WF_2026-10-01.md` 1절) | ReSALT(ORNL DAAC 10.3334/ORNLDAAC/2004)와 PolSAR ALT 자료 인용 추가. PolSAR DOI 는 앞부분 초안 42행에서 [확인 필요] |
| 14 | 방법 초안 189–197행 표 4 | 환경 둘(로컬 RTX 3090, Rescale iolite-4 T4) | WF 는 Rescale CPU 노드에서 실행(QA Q13, WF 계획 1절) | 표에 CPU 노드 추가 |
| 15 | `figures/figure_spec.json` Fig 5·Fig 7 메시지 | Fig 5 "방법 수준의 배치 지침만", Fig 7 "배포 절차와 초록 대비 10개" | 재구성안 5절: Fig 5 를 WF2·WF8 로, Fig 7 을 충분 라벨·C8·워크플로로 교체 | 사용자 결정(그림 배치안 A 또는 B) 뒤 명세 갱신 |
| 16 | `docs/GLOSSARY.md` | '활성층 두께' | 원고·대회 제목은 '활동층 두께' | 국문판은 '활동층 두께'로 통일 |
| 17 | 결과 초안 100행과 지원 초안 234행 | L25 등록 문장 '라벨이 구간을 좁힌다' 대 LGU 10절 금지 | 미결 [DECISION] | 서술 문장(커버리지·폭 비교)만 쓴다 |

---

## 4. 원고 작성 명세

### 4.1 틀과 주장 체계

**한 줄 틀**: 라벨이 희소한 조건에서 물리 증강과 물리 잔차는 ML 의 오차를 경험 물리식(Stefan) 수준 이하로 묶어 두고, 라벨이 생기면 그 라벨로 계수를 먼저 고친 뒤 ML 이 작은 격자 단위 보정을 더한다. 이를 라벨 0–수백 개의 새 지역에서 쓸 워크플로로 정리한다. 무작위 분할의 과대평가는 평가 설계의 근거로만 쓴다.

**주장과 절의 대응**(주장 문서 3절의 C1–C8 을 좁힌 형태)

| 주장 | 원고 문장의 범위 | 결과 절 | 근거 문서 |
|---|---|---|---|
| C1 | 라벨 0 에서 직접 ML 은 시험한 학습기 10종 모두 P0 보다 오차가 컸고, 물리 증강과 물리 잔차는 큰 손해를 줄였다 | R2 | 결과 초안 92행, 주장 문서 31–43행 |
| C2(좁힘) | 라벨을 쓰는 보정(재보정 + 잔차)의 이득은 원천 계수 오차의 크기를 따르고, 라벨 10개로 그 크기를 진단할 수 있다. 재보정을 넘어선 ML 의 추가 이득은 계수 오차 크기와 관계가 없었다 | R3 | QA 43행, 주장 문서 47행 |
| C3 | 결합 구조의 우열은 라벨 수에 따라 바뀐다(10개 이하 잔차, 40–160개 물리 입력) | R3, R4 | 주장 문서 58–62행 |
| C4 | 라벨이 많은 지역 안에서 이득은 작고 일정하다(알래스카 0.4–0.5 cm) | R5 | 주장 문서 64–69행 |
| C5 | 대상 라벨 안 교차검증으로 방법을 고르는 규칙이 고정 레시피보다 오차가 작았다(라벨 40·160개, 레나·캐나다) | R4 | 주장 문서 71–73행 |
| C6 | 라벨을 블록·공변량 공간에 퍼뜨리는 추출은 전이 조건에서 무작위보다 나빠지지 않았고, 이질적 지역에서 이득이 있었다. 예측 분산 기반 능동 선택은 이득이 없었다 | R6 | 주장 문서 75–80행 |
| C7(보조) | 검증 설계에 따라 오차 추정이 달라진다 | R1 | QA Q15 표, 결과 초안 114행 |
| C8 | 물리식과 총 오차가 비슷한 지역에서 이득은 격자 단위 편향 보정에서 오고, 격자 안 상세도는 더하지 못했다 | R5 | 주장 문서 84–90행 |

### 4.2 제목

| 항목 | 영문 | 국문 |
|---|---|---|
| 임시 제목 | Permafrost active-layer thickness prediction under sparse observations using physics-based pseudo-label augmentation and machine learning | 희소 관측 조건에서 물리경험식 유사라벨 증강과 기계학습을 이용한 영구동토 활동층 두께 예측 |
| 단어 수 | 14(하이픈 단어를 한 단어로 셈). 한도 20 안 | 해당 없음 |
| 출처 | 대회 제목의 앞부분(`outputs/report/main.tex` 84–85행)의 영역 | 같은 곳 |

임시 제목의 문제: (1) 규정은 "the main message of the article using a single scientifically accurate sentence" 를 권하는데 임시 제목은 명사구다. (2) 결과에서 증강은 라벨 0 전용으로 판정되었다(L2, AB6 동등). 제목이 증강을 앞세우면 본문 비중과 어긋난다. 제목 확정은 사용자 결정 사항으로 남긴다. 금지어(RESEARCH_FRAME A.1: 'beat the Stefan equation', 'how many', 'value-of-information')는 쓰지 않는다.

### 4.3 초록 설계(200단어 이하, 비구조형, 인용·기호 없음)

**판정어 규칙**: 판정어(lower, higher, equivalent within 0.5 cm)는 AB1–AB10 에만 쓴다(`docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md` 1.1, 규칙 (a)–(e)). WF 결과와 XA–XJ 결과에 판정어를 쓰려면 결과 산출 전에 1.1 을 개정해야 한다(사용자 결정).

| 순서 | 문장 역할 | 근거 | 단어 예산 |
|---|---|---|---|
| 1 | 문제: 새 지역의 ALT 예측, 라벨 0–수백 개 | 서론 | 25 |
| 2 | 설계: 지역 제외 + 100 km, 라벨 수 0–전량, 두 물리 기준선, 지역 내 블록 홀드아웃, 사전 등록 | 방법 | 30 |
| 3 | 라벨 0: 직접 ML 오차가 컸다(AB1 +2.25 cm [1.10, 3.42], 학습기 10종), 물리 유사라벨이 섞은 유사라벨보다 오차가 작았다(AB3 −1.63) | 결과 초안 132–134행 | 35 |
| 4 | 라벨 10개: 재보정이 이득의 대부분(AB4 −2.45, 4지역 중 1지역), 잔차 ML 추가 이득은 보정 뒤 확인되지 않음(AB5), 잔차 구조가 물리 입력 구조보다 오차가 작았다(AB7 −3.74) | 결과 초안 135–138행 | 40 |
| 5 | 라벨 전량: R1 이 P0 보다 오차가 작았으나 한 지역에서 왔다(AB8 −2.64), P1 대비 0.5 cm 미만(AB9 −0.40) | 결과 초안 139–140행 | 30 |
| 6 | 지역 내 충분 라벨과 워크플로: WF 수치(판정어 없이) | 주장 문서 65, 72, 80행 | 30 |
| 7 | 함의 한 문장 | | 10 |

합계 약 200단어다. WF 를 수치로 넣으면(사용자 선택지 '수치만') 문장 3–5 를 줄여야 한다. 초안 두 판(WF 제외판, WF 수치판)을 함께 만들어 사용자가 고르게 한다. 사용자가 10-04 에 초록의 WF 결과를 다시 설명해 달라고 했으므로 이 결정은 열려 있다.

### 4.4 본문 구성과 단어 예산(4,500단어 이하)

| 절 | 소제목(영문 명사구) | 예산 | 주장 | 근거 실험 | 표시 항목 | 새 실험 자리 |
|---|---|---|---|---|---|---|
| Introduction | 없음 | 700 | 배경, 기존 평가의 한계, 질문, 설계·기여 | NOVELTY 1·3절 | | XH 문헌(Ploton, Meyer & Pebesma, Wadoux) |
| Results R1 | Evaluation design, data and validation ladder | 300 | C7(보조) | LG §1–3, Table 1, L19·L20, QA Q15 표 | Fig 1, Table 1 | XH, XF(Table 1 행) |
| Results R2 | Zero labels: physics information limits machine-learning losses | 400 | C1 | AB1, L30, L34(a), LGF-F1·N1, AB3, L15, WF0 위험표(사후 서술) | Fig 6a, Fig 3a | XG(라벨 0 제품 비교 행) |
| Results R3 | One to ten labels: coefficient recalibration first | 450 | C2(좁힘), C3 | AB4, AB5, AB7, L10, WF0 ρ 0.66, WF4-c ρ 0.38, SC1w | Fig 2, Fig 3b, Fig 4 | XA, XB(n 10) |
| Results R4 | Tens to hundreds of labels: method selection within target labels | 350 | C5, C3 | WF4-a, L10 보조 행, LGF-F4 | Fig 4 또는 Fig 7(재구성안) | XB, XC |
| Results R5 | Label-rich regions: small gains at the climate-grid scale | 450 | C4, C8 | WF1, WF6, WF9, 오차 하한 | Fig 7(재구성안) | XE, XI |
| Results R6 | Where to place the next labels | 400 | C6 | WF2, WF8, L43, SC3w | Fig 5(재구성안) | XD, XC |
| Results R7 | Independent regions | 250 | C1·C4 일반화 | LGD(L1e, L4e, L8e, L38) | Table 1, Fig 1a | XF, XJ |
| Results R8 | Prediction intervals | 200 | 보조 | LGU-A1·B1·B2, L25 | Fig 6b–d | |
| Discussion | 소제목 없음, 6문단 | 850 | 요약, E 의 의미, 이득의 조건과 규모, 워크플로, 선행 연구, 한계와 후속(XD 학습형 정책, XE 고해상 입력) | 지원 초안 M3.2, 결과 초안 D5 | | |
| 합계 | | 4,350 | | | | 여유 150 |

국문판은 같은 구성과 순서를 따르고 영문판을 1:1 로 옮긴다. 단어 한도는 영문판으로만 판정하고 국문판은 어절 수를 참고로 기록한다.

### 4.5 Methods 구성과 분량(한도 없음, 목표 약 4,100단어)

| 소절 | 목표 | 현재 원천 | 비고 |
|---|---|---|---|
| Data: labels, regions, label unit, covariates | 700 | 방법 초안 49–61행(약 900) | SoilGrids 해상도 정정 |
| Physics baselines (E0, E_n, P*, P1*) | 300 | 같은 파일 75–83행, 지원 초안 46행 | |
| Learning methods and combination structures | 450 | 같은 파일 87–106행 | 학습기 축 세부는 SI |
| Transfer design (targets, splits, label draws) | 300 | 같은 파일 110–112행 | |
| Within-region design (WF1, WF6, WF9 분해) | 300 | 신규. WF 계획 2절 | 결과 열람 뒤 설계 표지 |
| Label placement strategies (WF2, WF8, XD) | 250 | 신규 | 알고리즘을 의사코드 수준으로 |
| Method-selection rule and bias diagnosis (WF4, XC) | 250 | 신규 | |
| Scoring, intervals, four-way verdict, multiplicity, abstract bundle | 600 | 같은 파일 116–155행(약 1,560) | 보정 전 p 제공 |
| Registration and post-hoc labels | 250 | 같은 파일 181–183행 | WF·XA–XJ 등록 시각 |
| Additional analyses XA–XJ | 400 | 신규 | 4.9절 |
| Computing environments and reproducibility | 200 | 같은 파일 187–203행 | Rescale CPU 노드, 재현 수치 |
| Use of large language models | 40 | 신규 | 규정 요구 |
| Code availability | 120 | 같은 파일 221행 | Methods 안 |

Supplementary Methods 로 옮길 것: LGD 상세(698), LGX 대비 목록(326), LGU(490), LGT·LGF(380), 교차 환경 점검 규칙, 등록 시각 표(지원 초안 M4.2), S-a·S-b·SC3w.

### 4.6 표시 항목 계획(8개)

| 항목 | 현행 v2(`outputs/figures/paper/v2/`) | 재구성안 B(재구성안 5절) | XA–XJ 반영 | 설명 단어(v2) |
|---|---|---|---|---|
| Fig 1 | 전이 설계, 지역별 E, 자료 범위 | 그대로 + 지역 내 블록 홀드아웃 설계 한 줄 | XH 검증 사다리를 패널 e 로 넣는 안 | 230 |
| Fig 2 | 라벨 수 곡선(전이) | 그대로 | XB 적층 곡선 선 추가(조건부) | 295 |
| Fig 3 | 물리 정보의 사용 | 그대로 | 없음 | 301 |
| Fig 4 | 최소 라벨 수 | 'ML 이 언제 이득인가': n* + 계수 오차 대 이득 + 규칙 W | XA 분해 패널 | 306 |
| Fig 5 | 배치 L43, L23, L24 | WF2 전략 7종 × 예산, WF8, 능동 선택 | XD 정밀 알고리즘·혼합 k-center 행. 학습형 정책 시험판은 SI | 230 |
| Fig 6 | 라벨 0 학습기, 예측 구간 | 그대로 | XG 제품 행(누설 점검 통과분만) | 319 |
| Fig 7 | 배포 단계, 초록 대비 10개 | 지역 내 라벨 수 곡선 + WF9 분해 + 워크플로 표 | XC, XE, XI | 263 |
| Table 1 | 대상, 라벨, 자료원 | 그대로 | XF 새 지역 행 | 84 |

규칙: 표시 항목 수는 8로 고정한다. 새 결과는 기존 그림의 패널이나 SI 로만 넣는다. AB 묶음 표(현행 Fig 7d)는 재구성안 B 에서 Supplementary Table 로 옮긴다. 그림 문자는 sans-serif, 패널 표지는 소문자 굵게, 숫자와 단위 사이 한 칸이다(2.2절). 재구성안 B 의 그림은 아직 없다. 두 안을 나란히 볼 수 있게 비교 폴더(Q10 의 `paper/figures/`)를 만든 뒤 사용자가 고른다.

### 4.7 기호와 용어

- 본문 기호는 다섯 개로 제한한다: n(대상 라벨 수), E(Stefan 계수), P0(원천 계수 Stefan), P1(같은 라벨로 재보정한 Stefan), R1(재보정 앵커 + 잔차 ML). 나머지 방법은 이름으로 쓴다(direct ML, physics pseudo-label augmentation, physics-input ML, stacked residual).
- 가설 번호(L1–L43, AB1–AB10, WF0–WF10, LGF-F·N, XA–XJ)는 본문에 쓰지 않고 SI 표와 Methods 의 등록 소절에만 둔다. v2 그림의 AB·L 표지는 그림 설명과 SI 대응표로 옮겨야 하므로 그림 재생성이 필요할 수 있다 [미확인: 그림 안 표지 개수].
- 약어는 첫 사용 때 풀어 쓴다(규정 "Keep abbreviations to a minimum").
- 국문 용어: 활동층 두께(ALT), 융해 도일(TDD), 원천 계수, 재보정, 물리 잔차, 물리 증강, 유사라벨, 블록 홀드아웃, 4분 판정(우세, 열세, 동등, 미결정). 문체는 보고서체(~이다), 과장 표현과 em-dash 연결을 쓰지 않는다.

### 4.8 주장 문장 규칙

| 대상 | 쓰는 문장 | 쓰지 않는 문장 | 근거 |
|---|---|---|---|
| 판정어 | 우세·열세는 두 가중 CI 가 같은 쪽일 때, 동등은 ±0.5 cm 안일 때 | 한 가중만으로 '유의', CI 가 0 을 포함한다는 이유로 '같다' | WRAPUP 1.2, 1.5 |
| 초록 | AB1–AB10 만 판정어. WF·XA–XJ 는 수치만 또는 제외 | WF 결과에 판정어 | WRAPUP 1.1 |
| '안전' | 위험표 수치('2 cm 넘게 나빠진 대상 18/30 대 2/30, 사후 서술') | 'safe', 'did not increase error'(등록된 비열등 없이) | 결과 초안 214행, SC1w(4/10 비열등) |
| C2 | 'label-based correction (recalibration plus residual)' 의 이득과 계수 오차 | 'ML gain is proportional to coefficient error' | QA 43행 |
| WF 결과 | 'designed after the label-grid results were viewed' 표지를 Methods 와 해당 결과 문단에 단다 | 사전 등록 확인 실험과 같은 무게의 표현 | WF 계획 머리말 |
| 관측 위치 | 방법 수준 규칙(블록·공변량 분산, 능동 선택 무익)과 L43 미결정 병기 | 특정 지점의 관측 우선순위, 'where to measure' 일반 처방 | 결과 초안 222행, WRAPUP 4 |
| 검증 설계 | 검증 사다리 수치와 문헌의 양쪽 입장(Ploton, Meyer & Pebesma 대 Wadoux) | '역전', '가장 정직한 추정' | L19, 지원 초안 216행 |
| 공간 상세도 | 'gains came from between-grid bias correction' | 'ML adds spatial detail' | 개요 문서 10절 |
| 기후 외삽 | '판정할 수 없었다'(WF10) | '기후 변화에서 물리 잔차가 더 안전하다' | 개요 문서 10절 |
| 학습기 | '시험한 학습기 10종에서' | 'tabular foundation models do not surpass physics'(계열 일반화), 'networks were sufficiently tuned' | LGF 9.2 |
| 지역 일반화 | 지역 4–5개라는 조건과 Hartung-Knapp 구간 | 지역 일반 문장(구간 없이) | WRAPUP 11.2 |
| 신규성 | 평가 설계, 그 결과(음성 포함), 워크플로 | 'first' 주장, 기법 신규성 | NOVELTY 5 |

### 4.9 새 실험 XA–XJ 의 원고 자리

자리표시 형식: `[XA: 절 | 표지 | 결과별 문장 규칙]`. 모든 XA–XJ 는 LG·WF 결과를 연 뒤 설계했으므로 '결과 열람 뒤 설계' 표지를 단다. 실행 전 등록 여부는 Methods 등록 소절에 커밋 시각과 함께 적는다.

| ID | 원고 절 | 표시 항목 | 등록 표지 | 결과별 허용 문장 | 초록 |
|---|---|---|---|---|---|
| XA_c2_gain_decomposition | R3, Discussion | Fig 4(B) 패널 또는 SI | 사후 분석(새 적합 없음) | 재보정 몫(P0 − P1)과 재보정을 넘어선 ML 몫(P1 − R1, P1 − 규칙 W)의 계수 오차 순위상관과 CI. 판정어 없음 | 수치만 |
| XB_multisource_stacking | R3, R4 | Fig 2 선 또는 SI | 결과 열람 뒤 설계, 실행 전 등록 | 4분 판정 문장(적층 − P1, 적층 − R1, 적층 + 잔차 − R1). 우세면 지역 수와 함께, 동등이면 한계 명시, 미결정이면 '차이를 확인하지 못했다' | 판정어 없음 |
| XC_workflow_end_to_end | R4, R6, Discussion | Fig 7(B) 패널 | 확인적으로 실행 전 등록. 구성 요소는 결과 열람 뒤 설계 | 워크플로 대 무작위 배치 + 고정 레시피의 4분 판정, 독립 대상만 CI | WRAPUP 1.1 개정 시에만 판정어(사용자 결정) |
| XD_placement_policy | R6, Methods | Fig 5(B) 행, 학습형 정책은 SI | 결과 열람 뒤 설계. 학습형 정책은 탐색 | 알고리즘은 의사코드로, 효과는 4분 판정. 학습형 정책은 '과제 하나 제외 검증의 시험판'으로만 | 없음 |
| XE_hires_covariates | R5, Discussion | Fig 7(B) 분해 패널 또는 SI | 결과 열람 뒤 설계 | 격자 안 설명 비율(%)과 WF9 분해. 늘지 않으면 '공개 고해상 입력으로도 격자 안 변동을 설명하지 못했다' | 없음 |
| XF_new_regions | R7 | Table 1 행, Fig 1a | 새 지역은 독립 지역 확인 표지 | 핵심 대비 재실행의 4분 판정, 지역별 행 | 없음 |
| XG_product_comparison | R2 또는 SI | Fig 6a 행 또는 Supplementary Table | 결과 열람 뒤 설계 | 학습 자료에 채점 셀이 없음을 확인한 제품만. 확인하지 못하면 '누설 점검 불가로 제외' | 없음 |
| XH_validation_ladder | R1 | Fig 1 패널 e 또는 SI | L19·L20 은 등록, 나머지는 결과 열람 뒤 | 무작위·지점·블록·kNNDM·지역 홀드아웃 오차와 CI. '역전' 금지 | 없음 |
| XI_climate_extrapolation_retest | R5 또는 SI | SI | 결과 열람 뒤 설계. 캐나다 확충판 외삽 영역 블록 ≥ 8 일 때만 | 조건 미달이면 다시 '판정할 수 없었다' | 없음 |
| XJ_tempderived_aux_labels | R7 또는 SI | SI | 결과 열람 뒤 설계 | 보조 라벨 가중별 4분 판정. 라벨 정의 차이(L39) 단서 필수 | 없음 |

### 4.10 참고문헌 상태와 추가

**현황**(`docs/MANUSCRIPT_DRAFT_METHODS_INTRO_2026-09-30.md` 225–297행): 67행. 상태 열은 web 41, meta 20, [verify] 6 이다. 파일 전체의 [verify] 태그는 19개이고 그중 참고문헌 행은 7개다.

| 키 | 확인할 것 |
|---|---|
| Du et al., 2026 | 논문 DOI |
| Kudryavtsev et al., 1974 | 서지 전체 |
| Pekel et al., 2016 | 서지 확인 |
| Pilyugina et al., 2025 | 분류(열방정식 정규화 대 Kudryavtsev 출력 입력) |
| Qu et al., 2026 | 학회(ICML) 확인 |
| Stefan, 1891 | 서지 확인 |
| Streletskiy et al., 2025 | 저자 목록(웹 5명 대 파일 머리말 6명) |

지원 초안 M3.2 의 [CITE] 5건(Riseborough 2008, Klene 2001, Nelson 1997, Shiklomanov & Nelson 2002, Romanovsky & Osterkamp 1997)은 확인하지 않았다 [미확인]. Nelson 1997 은 참고문헌 표에 web 상태로 이미 있다.

**추가 5편(2026-10-04 확인)**

| 문헌 | 서지 | 확인 근거 |
|---|---|---|
| Ploton et al., 2020 | Ploton, P. et al. Spatial validation reveals poor predictive performance of large-scale ecological mapping models. Nat. Commun. 11, 4540 (2020). https://doi.org/10.1038/s41467-020-18321-y | nature.com 쪽 메타데이터(citation_volume 11, citation_firstpage 4540, 저자 13명, 온라인 2020-09-11) |
| Meyer & Pebesma, 2021 | Meyer, H. & Pebesma, E. Predicting into unknown space? Estimating the area of applicability of spatial prediction models. Methods Ecol. Evol. 12, 1620–1633 (2021). https://doi.org/10.1111/2041-210X.13650 | https://api.crossref.org/works/10.1111/2041-210X.13650 |
| Meyer & Pebesma, 2022 | Meyer, H. & Pebesma, E. Machine learning-based global maps of ecological variables and the challenge of assessing them. Nat. Commun. 13, 2208 (2022). https://doi.org/10.1038/s41467-022-29838-9 | nature.com 쪽 메타데이터(온라인 2022-04-22) |
| Wadoux et al., 2021 | Wadoux, A. M. J.-C., Heuvelink, G. B. M., de Bruin, S. & Brus, D. J. Spatial cross-validation is not the right way to evaluate map accuracy. Ecol. Model. 457, 109692 (2021). https://doi.org/10.1016/j.ecolmodel.2021.109692 | Crossref, https://research.wur.nl/en/publications/spatial-cross-validation-is-not-the-right-way-to-evaluate-map-acc/ |
| Linnenbrink et al., 2024 | Linnenbrink, J., Milà, C., Ludwig, M. & Meyer, H. kNNDM CV: k-fold nearest-neighbour distance matching cross-validation for map accuracy estimation. Geosci. Model Dev. 17, 5897–5912 (2024). https://doi.org/10.5194/gmd-17-5897-2024 | https://gmd.copernicus.org/articles/17/5897/2024/, Crossref |

QA Q15 가 쓴 제목 'kNNDM' 의 정식 제목은 'kNNDM CV: …' 다.

**후보(서지 미확인)**: Tama 2025, Wei 2026(ESSD Discussions), Ahajjam 2025, Liu Z. 2024 는 개요 문서·QA 에 이름이 나오지만 참고문헌 표에 없다 [미확인]. XE·XG·XF 가 새 자료를 쓰면 자료 인용이 늘어난다.

**분량 조정안**: 67 + 5 = 72편이다. 규정은 60편이 "not strictly enforced" 라고 한다. 선택지는 (a) 그대로 내고 표지 편지에 자료 인용이 많다고 적기, (b) 새 지역(LGD) 자료 인용(상태 'meta' 20행과 Du 2026 의 대부분)을 Supplementary Information 의 참고문헌으로 옮기기다. (b)가 허용되는지는 규정 쪽에서 확인하지 못했다 [미확인].

### 4.11 말미 요소

| 요소 | 위치 | 원천 | 할 일 |
|---|---|---|---|
| Code availability | Methods 마지막 소절 | 방법 초안 221행 | h54(WF), XA–XJ 하네스, `tests/test_h54_workflow.py`, Rescale 설정(`configs/rescale/`) 추가. Zenodo DOI |
| Use of large language models | Methods | 신규 | 코드 작성 보조와 문장 편집에 LLM 을 썼고 저자가 검증했다는 진술(앞부분 초안 131행 문안 재사용) |
| Data availability | 본문 끝, 참고문헌 앞 | 방법 초안 209–217행 | SAR(WF1-b), XE·XF·XG 자료, 그림 source data 기탁 진술 |
| References | Data availability 뒤 | 4.10절 | |
| Acknowledgements | 참고문헌 뒤 | 앞부분 초안 119행 | 짧게. Copernicus DEM 귀속 문구 유지 |
| Author contributions | 참고문헌 뒤 | 앞부분 초안 121행 | 저자 정보 [자리표시자] |
| Additional Information | 그 뒤 | 앞부분 초안 125행 | 'Competing interests' 진술 |
| Figure legends | 그 뒤 | CAPTIONS.md v2 | 350단어 이하 |
| Tables | 끝 | `v2/Table1_data.tex` | 한 쪽 안, 각주를 설명문으로 |
| 표지 편지 | 별도 | 앞부분 초안 311행 | 대회 보고서 공개 사실, 사본 첨부, 적합성 설명, 추천·제외 심사자 |

### 4.12 Supplementary Information 구성

하나의 PDF 로 묶는다. 번호는 'Supplementary Fig. S1', 'Supplementary Table S1' 형식이고 본문 번호와 따로 센다. 본문에서 SI 그림의 패널을 가리키지 않는다. 구성: Supplementary Methods(4.5절의 이관분), Supplementary Notes(E 의 물리적 의미 세부, 라벨 단위·연도 S-a·S-b), Supplementary Tables(가설 등록표, AB 묶음 전체, 학습기 비교, LGD 세부, LGU 세부, WF 세부, XA–XJ 세부, 선택·철회 이력, 소프트웨어·실행 표), Supplementary Figures(figure_spec.json si_items 40개 가운데 남는 항목과 재구성안에서 밀려난 L23·L24·레나 지도). 큰 표는 .csv 별도 파일로 낸다(규정 허용).

---

## 5. 빌드 파이프라인

### 5.1 로컬 도구 점검(2026-10-04)

| 항목 | 결과 | 근거 |
|---|---|---|
| xelatex | `/home/willy010313/.local/bin/xelatex`, XeTeX 3.141592653-2.6-0.999998 (TeX Live 2026) | `which`, `xelatex --version` |
| pdflatex, lualatex, latexmk, bibtex | 있음(`~/.local/bin`). latexmk 4.88, BibTeX 0.99e | `which`, `--version` |
| biber | 없음 | `which biber` 출력 없음. natbib + bibtex 로 간다 |
| texcount | `~/.TinyTeX/bin/x86_64-linux/texcount` 에 있으나 PATH 밖 | `ls` |
| latexpand | 없음. tlmgr 패키지 `latexpand` 로 설치 가능 | `tlmgr search --global --file /latexpand` |
| pandoc | 1.19.2.4(2017년 판) | `pandoc --version`. 마크다운 변환용으로는 낡았다 |
| xeCJK, kotex, natbib | 있음 | `kpsewhich` |
| 한글 글꼴 | Noto Serif CJK KR, Noto Sans CJK KR(`/usr/share/fonts/opentype/noto/`), Nanum Myeongjo·Gothic·Barun Gothic·Square·Square Round·Gothic Coding(`/usr/share/fonts/truetype/nanum/`), Pretendard 9종(`~/.fonts/`) | `fc-list :lang=ko` |
| Springer 템플릿 | 저장소와 TeX 트리에 없다(`kpsewhich sn-jnl.cls` 출력 없음). 내려받기 가능(2.1절) | |
| 템플릿에 없는 패키지 | threeparttable(threeparttable), manyfoot(ncctools), mathrsfs(jknapltx, 글꼴 rsfs), appendix(appendix). pdflatex 선택지를 빼면 breakurl(breakurl)도 필요 | `kpsewhich`, `tlmgr search --global --file` |
| GLIBC | 2.27 | `ldd --version` |

### 5.2 시험 컴파일(스크래치 폴더, 저장소 밖)

빠진 패키지는 TeX Live 저장소 압축본(`https://tlnet.yihui.org/archive/<패키지>.tar.xz`)을 스크래치 폴더에 풀고 `TEXMFHOME`·`TEXINPUTS` 로만 지정했다. 사용자 TeX 설치는 바꾸지 않았다.

| 시험 | 설정 | 결과 |
|---|---|---|
| 1 | pdflatex, `[pdflatex,sn-nature]`, bibtex `sn-nature.bst` | 오류 0, 경고 10, 12쪽 |
| 2 | xelatex, `[sn-nature]`(pdflatex 선택지 없음), 예시 EPS 그림 | xdvipdfmx 가 EPS 를 변환하지 못해 PDF 를 쓰지 못했다. 원인: "texlua: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.28' not found (required by texlua)" |
| 3 | 시험 2 + 그림을 v2 PDF 로 교체 + xeCJK 한글 | 오류 101건. 클래스가 pdflatex 가 아니면 breakurl 을 불러 PostScript 머리말(`\headerps@out`)을 넣는데 xelatex 와 맞지 않는다 |
| 4 | xelatex, `[pdflatex,sn-nature]` + xeCJK(`Noto Serif CJK KR`) + PDF 그림 | 오류 0, 경고 15, 13쪽. 한글 출력됨 |
| 5 | 시험 4 에서 `\xeCJKsetup{CJKspace=true}` 없음 | 한글 어절 사이 공백이 사라진다("영구동토활동층두께"). 넣으면 정상 |

결론: 영문판은 `sn-jnl` 에 `pdflatex` 선택지를 그대로 두고 xelatex 로 컴파일한다(선택지 이름과 엔진이 다르지만 breakurl 을 피하는 설정이다). 제출 전에는 pdflatex 로도 오류·경고 0을 확인한다(투고 시스템의 컴파일 엔진 [미확인]). 그림은 PDF 만 쓰고 EPS 는 쓰지 않는다. 내장 글꼴 이름은 TTC 묶음 때문에 'NotoSerifCJKjp' 로 표시되었다. 한글 자형은 같으나 한자 자형은 다를 수 있다 [미확인].

### 5.3 폴더 구조(제안)

```
paper/manuscript/
  shared/
    refs.bib                 # 단일 참고문헌(영·한 공용), sn-nature.bst
    numbers.tex              # 결과 수치 매크로(스크립트로 생성, 매크로마다 원천 CSV 경로 주석)
    build_numbers.py         # 원천 표 → numbers.tex, 값 대조 점검 포함
    glossary_en_ko.csv       # 용어 대응
  en/
    main.tex                 # \documentclass[pdflatex,sn-nature,lineno]{sn-jnl}
    sections/*.tex           # 작성용 분할 파일
    figures/ -> ../../../outputs/figures/paper/v2 (PDF만)
    tables/Table1_data.tex
    build/main.pdf, submission/main_flat.tex(latexpand 로 단일 파일)
    si/si.tex → build/si.pdf
  ko/
    main.tex                 # article, 1단, main.tex 6–33행 전처리 + CJKspace=true
    sections/*.tex
    build/main.pdf
  texmf/                     # 빠진 패키지를 저장소에 두는 경우(tlmgr 설치를 안 할 때)
  Makefile                   # make en, make ko, make si, make check
```

### 5.4 영문 빌드

1. `latexmk -xelatex -outdir=build main.tex` 로 PDF 를 만든다. 참고문헌은 bibtex + `sn-nature.bst`.
2. `latexpand main.tex > submission/main_flat.tex` 로 단일 파일을 만든다(템플릿 요구). 같은 파일을 pdflatex 로 한 번 더 컴파일해 오류·경고 0을 확인한다.
3. 결과 수치는 손으로 옮기지 않고 `numbers.tex` 매크로로 넣는다. `build_numbers.py` 가 원천 표(예: `data/processed/lgw/lgw_bundle.csv`, `results/rescale_wf2/data/processed/wf/wf2b_tests.csv`)에서 값을 읽고, 매크로 값과 원천 값이 다르면 실패한다.
4. 그림은 `outputs/figures/paper/v2/*.pdf`(재구성안 확정 뒤 그 판)를 쓴다. 첫 제출은 그림을 원고 PDF 에 넣고 3 MB 이하를 확인한다.

### 5.5 국문 빌드

1. `outputs/report/main.tex` 6–33행(geometry, fontspec, xeCJK, 글꼴, 섹션·캡션 서식)을 가져와 1단으로 바꾼다. `\xeCJKsetup{CJKspace=true}` 는 반드시 둔다(5.2절 시험 5).
2. 본문 글꼴 Noto Serif CJK KR, 제목 Pretendard(대회 보고서와 같다).
3. 그림은 영문판과 같은 PDF 를 쓰고 그림 설명만 국문으로 쓴다. 참고문헌은 영문판과 같은 `refs.bib`.
4. `latexmk -xelatex` 로 `paper/manuscript/ko/build/main.pdf` 를 만든다.

### 5.6 점검 항목(`make check`)

| 점검 | 기준 | 도구 |
|---|---|---|
| 본문 단어 수 | 서론 + Results + Discussion ≤ 4,500 | texcount(절별) |
| 초록 | ≤ 200단어, 인용·기호 없음 | texcount, 정규식 |
| 제목 | ≤ 20단어 | 스크립트 |
| 그림 설명 | 각 ≤ 350단어 | texcount |
| 표시 항목 | ≤ 8 | figure·table 환경 수 |
| 참고문헌 | 개수 보고, 미인용·중복 키 0 | bibtex 로그 |
| 수치 | numbers.tex 매크로와 원천 표 일치 | build_numbers.py |
| 컴파일 | xelatex·pdflatex 오류 0, 경고 0 | 로그 검사 |
| 문체 | em-dash(U+2014) 0개, 연결어로 쓴 en-dash(U+2013) 0개(숫자 범위 제외), 각주 0개 | grep |
| 금지 문장 | 4.8절 표의 금지 문구 0개 | grep |
| 그림 | PDF 벡터, sans-serif, 패널 소문자 굵게 | 사람 점검(시각 QA) |

### 5.7 실험과의 병렬 순서

1. 지금 쓸 수 있는 것: Methods(데이터, 기준선, 전이 설계, 채점), 서론, R1–R3·R7·R8 의 기존 결과 문단, Data/Code availability, SI 의 LG 계열 부분.
2. 자리표시로 둘 것: XA–XJ 문단과 패널, WF 판정어 사용 여부, 초록 두 판.
3. 실험 판정 뒤 채울 것: R4–R6 의 XB–XD 문단, R5 의 XE·XI, R7 의 XF·XJ, Fig 4·5·7(재구성안), Table 1 의 새 지역 행.
4. 국문판은 영문판 절이 확정될 때마다 옮긴다.

---

## 6. 사용자 결정 사항과 남은 문제

1. 제목: 임시 제목(14단어, 명사구, 증강 강조)을 쓸지, 주장을 담은 한 문장 제목으로 바꿀지.
2. 그림 배치: 현행 v2(안 A) 대 재구성안 B(Fig 4·5·7 교체). 비교 폴더가 아직 없다.
3. 초록의 WF 결과: 제외 / 수치만 / 판정어 포함(WRAPUP 1.1 개정 필요).
4. XC 를 확인적 실험으로 등록할 때 초록 판정어 대상에 넣을지(결과 전에 정해야 한다).
5. 결과 초안의 [DECISION] 7건(3.4절)과 사후 개정 4건의 수용 여부(방법 초안 183행).
6. 참고문헌 72편 처리(그대로 / 새 지역 자료 인용을 SI 로).
7. TeX 패키지 5종을 사용자 TinyTeX 에 설치할지(`tlmgr install threeparttable ncctools jknapltx rsfs appendix latexpand`), 저장소 `paper/manuscript/texmf/` 에 둘지.
8. KPDC 콘슬 자료를 SI 에 남길지(남기면 KPDC 식별자 3종과 이용 조건을 Data availability 에 적는다).
9. 확인하지 못한 것: Sci Rep 투고 시스템의 LaTeX 컴파일 엔진, SI 안 참고문헌 허용 여부, LaTeX 지원 쪽의 참고문헌 제출 문장 원문, SoilGrids 원 tif 창이 모든 지역에 같은 해상도로 쓰였는지, v2 그림 안 AB·L 표지 수, 참고문헌 [verify] 7행과 [CITE] 5건.
