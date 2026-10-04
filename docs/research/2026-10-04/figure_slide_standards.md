# 학술지 그림·연구 발표 슬라이드 표준 조사 (2026-10-04)

Sci Rep 원고 그림과 보고 덱의 규칙을 외부 근거로 고정하기 위한 조사 문서이다. 학술지 지침은 출판사 원문 문장을 그대로 인용하고, 지침 사이의 불일치와 채택안을 함께 적는다. 슬라이드 원칙은 실증 연구(assertion-evidence 계열)와 PLOS Comput Biol 'Ten simple rules' 계열을 근거로 한다.

표기 규칙
- 인용문은 영어 원문 그대로 둔다. 쪽 표시는 PDF 쪽 번호이다.
- **[실측]** 이 문서 작성 중 직접 잰 값이다. **[판단]** 근거 문헌이 직접 수치를 주지 않아 이 문서가 정한 값이다. **[미확인]** 원문 대조를 마치지 못한 항목이다.
- 웹 지침은 2026-10-04 에 curl 로 원문 HTML 을 받아 문장을 추출했다(WebFetch 는 nature.com 에서 로그인 경유 주소로 돌려보내 요약만 반환했다).
- 관련 문서: 원고 양식 전반은 `docs/research/2026-10-04/scirep_format_and_drafts.md`, 기존 산출물 감사는 `design/baseline_audit_2026-10-04.md` 에 있다. 이 문서는 그림·슬라이드 규칙만 다룬다.

---

## 1. 핵심 규칙 요약

| 구분 | 규칙 | 근거 |
|---|---|---|
| 그림 폭 | 전폭 그림은 170 mm 로 만든다. 반폭 그림은 85 mm 로 만든다 [판단] | Sci Rep 게재본 49개 그림의 배치 폭 71–170 mm, 최대 170 mm [실측, 2.4절]. Nature 계열 표준 폭 89·183 mm |
| 그림 높이 | 190 mm 이하. 범례를 같은 쪽에 두려면 170 mm 이하 [판단] | Sci Rep 게재본 최대 194 mm [실측]. Nature 최대 170 mm |
| 서체 | sans-serif 한 종류(Arial 또는 Helvetica 계열), 모든 그림에서 같은 서체·같은 크기 | Sci Rep 투고 지침, Nature 그림 지침 |
| 글자 크기 | 최종 크기에서 5–7 pt. 패널 문자는 굵은 직립 소문자 a, b, c | Nature 그림 지침. Sci Rep 은 패널 문자를 "same type size as used elsewhere" 로 규정한다(2.3절) |
| 선 굵기 | 최종 크기에서 가장 얇은 선 1 pt 이상(Sci Rep 문구). Nature 범위는 0.25–1 pt | Sci Rep 투고 지침, Nature 최종 제출 지침(2.3절 충돌 정리) |
| 해상도·형식 | 선 그림은 벡터(PDF/EPS), 글꼴 내장(Type 42), 윤곽선 변환 금지. 래스터 영상은 300 dpi 이상(Nature 450 dpi 권장) | Sci Rep, Nature 그림 지침 |
| 금지 요소 | 과도한 상자, 불필요한 색, 3차원 장식, 배경 격자선, 그림자, 패턴, 색 글씨, 장식 아이콘, 복잡한 배경 위 글자, 겹친 글자, 히스토그램 세로축 절단 | Sci Rep, Nature 그림 지침 |
| 색 | 지각 균일·색각 이상 대응 색지도, rainbow·jet·적녹 조합 금지, 색 막대 항상 표시, 글자 대비 4.5:1 이상 | Nature 그림 지침, Crameri 2020, Stoelzle 2021 |
| 그림 안 글자 | 설명은 범례(캡션)로 옮기고 그림에는 축 이름, 단위, 기호 열쇠만 둔다 | Nature 서식 지침, Rougier 2014 규칙 4·8 |
| 범례 | 그림 전체를 요약하는 짧은 제목 문장으로 시작, 패널별 설명, 350단어 이하, 색 이름 대신 기호 | Sci Rep 투고 지침 |
| AI | 그림에 생성형 AI 사용 불가. 원고 작성의 LLM 사용은 Methods 에 기록 | Nature 그림 지침(Image Integrity), Sci Rep 투고 지침 |
| 슬라이드 | 결론을 문장으로 쓴 헤드라인(1–2줄, 좌측 정렬) + 시각 근거, 글머리표 목록 지양 | Alley & Neeley 2005, Alley 2006, Garner & Alley 2013 |
| 슬라이드 글자 | 헤드라인 28 pt, 본문 18–24 pt, sans-serif, 대문자 전용 금지, 글 덩어리 2줄 이하, 목록 2–4항목 | Alley & Neeley 2005 Table 1 |
| 슬라이드 수 | 한 장에 생각 하나, 한 장당 약 1분(발표 분 수 ≈ 장 수) | Naegle 2021, Alley & Neeley 2005, Bourne 2007 |
| 논문 그림 재사용 | 논문 그림을 그대로 붙이지 않는다. 다패널 그림은 패널 하나씩, 선·글자를 키우고 세로 글자를 피한다 | Rougier 2014 규칙 3, Lortie 2017 규칙 4, Naegle 2021 규칙 6 |

---

## 2. 학술지 그림 규격

### 2.1 Scientific Reports 투고 지침

출처: https://www.nature.com/srep/author-instructions/submission-guidelines (Figure guidelines, Figures for publication, Figure legends, Statistical guidelines 절)

서체·크기·배경
- "Use a clear, sans-serif typeface (for example, Helvetica) for figure lettering. Use the same typeface in the same font size for all figures in your paper. For Greek letters, use a 'symbols' font."
- "Put all display items on a white background, and avoid excessive boxing, unnecessary colour, spurious decorative effects (such as three-dimensional 'skyscraper' histograms) and highly pixelated computer drawings."
- "Never truncate the vertical axis of histograms to exaggerate small differences."
- "Ensure any labelling is of sufficient size and contrast to be legible, even after appropriate reduction. The thinnest lines in the final figure should be no smaller than one point wide."

글자 표기
- "Figures divided into parts should be labelled with a lower-case, bold letter (a, b, c and so on) in the same type size as used elsewhere in the figure."
- "Lettering in figures should be in lower-case type, with only the first letter of each label capitalised."
- "Units should have a single space between the number and the unit, and follow SI nomenclature (for example, ms rather than msec) or the nomenclature common to a particular field."
- "Commas should be used to separate thousands and the decimal point to separate decimals. As per our style guidelines, this applies to five or more digits."
- "Unusual units or abbreviations should be spelled out in full or defined in the legend."
- "Scale bars should be used rather than magnification factors, with the length of the bar defined on the bar itself rather than in the legend."

그림 수와 구성
- "Avoid unnecessary figures: data presented in small tables or histograms, for instance, can generally be stated briefly in the text instead. Figures should not contain more than one panel unless the parts are logically connected; each panel of a multipart figure should be sized so that the whole figure can be reduced by the same amount and reproduced at the smallest size at which essential details are visible."
- "Display items are limited to 8 (figures and/or tables)." "Please note that schemes should not be used and should be presented as figures instead."
- "Scientific Reports does not support graphical abstracts."

파일
- "For optimal results, you should supply all line art, graphs, charts and schematics in vector format, such as EPS or AI."
- "Multi-part/panel figures must be prepared and arranged as a single image file (including all sub-parts; a, b, c, etc.). Please do not upload each panel individually. All digitized images submitted with the final revision of the manuscript should be 300 DPI if possible."
- "Please do not supply Word or Powerpoint files with placed images. Images can be supplied as RGB or CMYK (note: we will not convert image colour modes)."
- 저작권: "We cannot publish images downloaded from the internet without appropriate permission." 지도: "Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations."

범례
- "Please begin your figure legends with a brief title sentence for the whole figure and continue with a short description of what is shown in each panel. Use any symbols in sequence and minimise the methodological details as much as possible. In legends, please use visual cues rather than verbal explanations such as "open red triangles". Keep each legend total to no more than 350 words."
- "Include error bars when appropriate. Include a description of the statistical treatment of error analysis in the figure legend."

통계 표현
- "Graphs should include clearly labelled error bars. You must state whether a number that follows the ± sign is a standard error (s.e.m.) or a standard deviation (s.d.)."
- "Ranges are more appropriate than standard deviations or standard errors for small data sets."
- "Use of the word "significant" should always be accompanied by a P value; otherwise, use "substantial," "considerable," etc."

AI
- "Use of an LLM should be properly documented in the Methods section (and if a Methods section is not available, in a suitable alternative part) of the manuscript."

### 2.2 Nature Portfolio 그림 지침

출처 세 곳이다. (1) Nature research figure guide, https://research-figure-guide.nature.com/figures/ (Preparing figures, Building and exporting figure panels, Image Integrity, Top 10 ways to delay your paper). (2) Nature 최종 제출 지침, https://www.nature.com/nature/for-authors/final-submission. (3) Nature 서식 지침, https://www.nature.com/nature/for-authors/formatting-guide. Sci Rep 은 Nature Portfolio 학술지이므로 (1)을 상위 참고 기준으로 쓴다. (2)와 (3)은 Nature 본지 규정이다.

그래프 필수 요소 (1, Preparing figures)
- "Axis lines and tick marks to be included"
- "All axes to be labelled with units in parentheses, e.g. Data (unit)"
- "An accessible colour palette to be used" (예시로 Wong, B. Points of view: Colour blindness. Nature Methods 8, 441 (2011) 을 든다)
- "Legible text (a minimum of 5 pt in size)"
- "Standard fonts (e.g. Arial or Helvetica) to be used"

피할 요소 (1, Preparing figures, "We avoid")
- "Background gridlines"
- "Superfluous icons and other decorative elements"
- "Drop shadows"
- "Patterns"
- "Text placed on top of busy images and hard-to-read backgrounds"
- "Overlapping text"
- "Coloured text; keylines, keys, etc. should be used instead"

글자 (1)
- "Maximum text size: 7 pt" / "Minimum text size: 5 pt"
- "Separate panels in multi-panelled figures should be labelled with 8-pt bold, upright (not italic) and lowercase a, b, c, etc."
- "Do not outline text" / "Embed fonts (True Type 2 or 42)" / "If you are using Python please use the following setting: Matplotlib.rcParams['pdf.fonttype']=42"

크기와 배치 (1, Building and exporting figure panels)
- "The widths of printed figures are: 89 mm (single column) 183 mm (double column)"
- "The maximum height for a Nature figure is 170 mm, to allow space for the figure legend to fit underneath."
- "Authors are encouraged to submit figures at the smallest appropriate size, ensuring all fonts are between 5pt and 7pt."
- "Figures should be laid out in a neat and space-efficient manner, minimizing white space and with panels in an alphabetical order wherever possible."

접근성 (1)
- "Avoiding red/green combinations and rainbow scales helps readers with colour blindness todistinguish datasets and interpret data correctly. Keys or keylines should be used in the figure wherever possible, rather than having colour descriptions in the figure caption"
- "High contrast text (>4.5 contrast ratio) should be used to make text easy to read for all."
- "Text should be used instead of decorative icons wherever possible. Icons can be open to interpretation and confuse the meaning of figures."
- 색각 이상 대응 예시 팔레트(같은 쪽 표): Black #000000, Orange #e69f00, Sky blue #56b4e9, Bluish green #009e73, Yellow #f0e442, Blue #0072b2, Vermillion #d55e00, Redish purple #cc79a7.

형식 (1)
- 주 그림: "For main figures we require vector files with editable layers" 선호 형식 .ai, .eps, .pdf. ".jpeg, .tiff, .png" 등은 받지 않는다고 적는다(Nature 본지 기준).
- 영상: "All photographic images must be supplied at a minimum of 300 dpi. The maximum dpi of online proofs is 450 dpi"

최종 제출 지침 (2)
- "Nature's standard figure sizes are 89 mm wide (single column) and 183 mm wide (double column). The full depth of a Nature page is 247 mm. Figures can also be a column-and-a-half where necessary (120–136 mm)."
- "Line weights and strokes should be set between 0.25 and 1 pt at the final size (lines thinner than 0.25 pt may vanish in print)."
- "Where practical, avoid placing lettering directly over images or shaded areas."
- "Please do not rasterize line art or text in submitted figures"

서식 지침 (3)
- "Figures should be as small and simple as is compatible with clarity." "Avoid unnecessary complexity, colouring and excessive detail."
- "Nature's standard figure sizes are 90 mm (single column) and 180 mm (double column) and the full depth of the page is 170 mm."
- "Layering type directly over shaded or textured areas and using reversed type (white lettering on a coloured background) should be avoided where possible."
- "Where possible, text, including keys to symbols, should be provided in the legend rather than on the figure itself."
- "Each figure legend should begin with a brief title for the whole figure and continue with a short description of each panel and the symbols used. If the paper contains a Methods section, legends should not contain any details of methods."

### 2.3 지침 간 불일치와 채택안

| 항목 | Sci Rep 투고 지침 | Nature 그림 지침 (1) | Nature 최종 제출 (2) | Nature 서식 (3) | 채택안 |
|---|---|---|---|---|---|
| 폭 | 규정 없음 | 89 / 183 mm | 89 / 183 mm, 1.5단 120–136 mm | 90 / 180 mm | 170 / 85 mm [판단, 2.4절 실측 근거] |
| 최대 높이 | 규정 없음 | 170 mm | 쪽 깊이 247 mm | 170 mm | 190 mm 이하, 범례를 같은 쪽에 두려면 170 mm 이하 [판단] |
| 본문 글자 | 모든 그림 같은 서체·같은 크기 | 5–7 pt | 5–7 pt | 규정 없음 | 6–7 pt, 한 원고 안에서 크기 2종(본문, 패널 문자) [판단] |
| 패널 문자 | 굵은 소문자, "same type size as used elsewhere" | 8 pt 굵은 직립 소문자 | 8 pt 굵은 직립 | 규정 없음 | 8 pt 굵은 직립 소문자. Sci Rep 문구와 충돌하므로 투고 전 Sci Rep 게재본 표본으로 재확인 [미확인] |
| 최소 선폭 | "no smaller than one point wide" | 규정 없음 | 0.25–1 pt | 규정 없음 | 데이터 선 1.0–1.5 pt, 축·눈금 1.0 pt 기본값 [판단]. 0.5 pt 미만 선은 쓰지 않는다 |
| 영상 해상도 | 300 DPI | 최소 300, 450 dpi 권장 | 300–600 dpi | 규정 없음 | 래스터 요소 450 dpi, 미리보기 PNG 600 dpi |
| 형식 | 벡터 EPS·AI, 비트맵 TIFF·JPG·PSD | PDF·EPS·AI 선호 | AI·EPS·PDF | 규정 없음 | PDF(글꼴 Type 42 내장) 정본 + EPS 변환본 |
| 그림 안 설명 글 | 규정 없음 | 색 설명 대신 열쇠·keyline 을 그림 안에 | 규정 없음 | 열쇠 포함 글자는 가능하면 범례로 | 기호 열쇠(작은 범례)만 그림 안, 해석·방법 설명은 범례 |

선폭 충돌에 대한 설명. Sci Rep 문구는 가장 얇은 선이 1 pt 이상이어야 한다고 적고, Nature 최종 제출 지침은 0.25–1 pt 범위를 준다. 목표 학술지가 Sci Rep 이므로 Sci Rep 문구를 우선한다. 다만 실제 Sci Rep 게재본이 이 규칙을 지키는지는 게재 PDF 가 그림을 래스터로 넣기 때문에 확인하지 못했다 [미확인].

### 2.4 Sci Rep 게재본 실측 [실측]

방법. `references/14_scirep_exemplars/` 8편과 `references/01_benchmark/gautam2025_alaska_alt.pdf` 1편(모두 Sci Rep, 2022–2026)에 `pdfimages -list` 를 적용했다. 가로 800 px·세로 400 px 를 넘는 이미지 49개를 그림으로 보고, 배치 해상도(x-ppi)로 쪽 위 배치 폭을 환산했다.

결과
- 쪽 크기는 210 × 276 mm 이다(1편은 210 × 279 mm).
- 모든 그림이 약 301 ppi 래스터로 들어가 있다. 벡터로 제출해도 게재 PDF 에서는 300 ppi 영상이 된다.
- 배치 폭은 71–170 mm, 중앙값 150 mm 이다. 90 mm 이하 9개, 90–135 mm 8개, 135–160 mm 16개, 160 mm 초과 16개이다. 최대 폭은 170 mm 이다.
- 배치 높이는 최대 194 mm 이다.

| 논문 (Sci Rep) | DOI | 그림 수 | 폭 (mm) | 최대 높이 (mm) |
|---|---|---|---|---|
| Feeney 2022 | 10.1038/s41598-022-05476-5 | 4 | 78–170 | 194 |
| Hatami 2022 | 10.1038/s41598-022-06320-6 | 6 | 100–170 | 129 |
| Isaksen 2022 | 10.1038/s41598-022-13568-5 | 3 | 127–159 | 150 |
| Fisher 2023 | 10.1038/s41598-023-34921-2 | 6 | 71–169 | 155 |
| Khanal 2023 | 10.1038/s41598-023-34247-z | 7 | 82–150 | 152 |
| Jones 2024 | 10.1038/s41598-024-58998-5 | 6 | 81–167 | 125 |
| Gautam 2025 | 10.1038/s41598-025-26586-w | 5 | 131–162 | 154 |
| Gay 2026 | 10.1038/s41598-026-61719-9 | 6 | 168–169 | 188 |
| Hall 2026 | 10.1038/s41598-026-70171-8 | 6 | 127–168 | 118 |

시사점
- Sci Rep 의 전폭은 약 170 mm 이다. 183 mm 나 180 mm 로 만든 그림은 0.93–0.94 배로 줄어든다. 180 mm 캔버스의 5 pt 글자는 약 4.7 pt 가 되어 Nature 하한 5 pt 를 밑돈다.
- 따라서 전폭 그림은 처음부터 170 mm 로 그리거나, 180 mm 로 그릴 경우 최소 글자를 5.5 pt 이상으로 둔다 [판단].
- 90 mm 이하 그림 9개의 실측 폭은 71–82 mm 이다. 반폭 캔버스 85 mm 는 0.84–0.96 배로 줄 수 있으므로 반폭 그림의 최소 글자는 6 pt 이상으로 둔다 [판단].

### 2.5 다른 지구과학 학술지 지침 (비교용)

Copernicus (The Cryosphere 등), https://www.the-cryosphere.net/submission.html, "Figures & tables" 절
- "The width should not be less than 8 cm." 해상도 300 dpi.
- "A legend should clarify all symbols used and should appear in the figure itself, rather than verbal explanations in the captions (e.g. "dashed line" or "open green circles")."
- "Please use only one font family in your figures (e.g. Arial or Helvetica) and consider using sans-serif fonts."
- 패널 문자: "Labels of panels must be included with brackets around letters being lower case (e.g. (a), (b), etc.)" (Sci Rep 의 굵은 a, b 와 다르다).
- 좌표: "Coordinates need a degree sign and a space when naming the direction (e.g. 30° N, 25° E)."
- 색각 이상: "colour vision deficiencies (CVD; this affects ~5% of the global population)" 를 언급하고 Coblis 시뮬레이터 점검과 Scientific colour maps(Crameri) 사용을 권한다.

Elsevier 공통 지침, https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-sizing
- 폭: 최소 30 mm, 단 90 mm, 1.5단 140 mm, 2단 190 mm.
- "the lettering on the artwork should have a finished, printed size of 7 pt for normal text and no smaller than 6 pt for subscript and superscript characters."
- "300 DPI for halftone images; 500 DPI for combination art; 1000 DPI for line art."

AGU 지침 쪽은 자동 접근이 차단되어(HTTP 403) 확인하지 못했다 [미확인].

정리. 출판사마다 폭은 85–90 mm(단)와 170–190 mm(전폭) 사이이고, 글자는 최종 크기 5–7 pt(Elsevier 는 7 pt)이며, 서체는 sans-serif 한 종류이다. Sci Rep 원고는 2.3절 채택안으로 충분하다.

### 2.6 허용 요소와 금지 요소

허용·권장
- 축선과 눈금, 단위가 괄호로 붙은 축 이름 "Variable (unit)" (Nature 1).
- 색 막대와 그 이름·단위. Crameri 2020: "It is imperative to ensure the colour bar is included on all figures where a colour scale is used ... Excluding a colour bar would be equivalent to not including the axis labels and tick marks on the x- or y-axis of a plot." (p. 6)
- 기호 열쇠, keyline(대상과 검은 글자를 잇는 가는 선) (Nature 1).
- 축척 막대(길이를 막대 위에 표기) (Sci Rep).
- 강조할 요소 하나만 색을 쓰고 나머지는 회색·검정으로 두는 방식. Rougier 2014 규칙 6: "to highlight some element of a figure, you can use color for this element while keeping other elements gray or black ... 'Is there any reason this plot is blue and not black?' If you don't know the answer, just keep it black." (p. 2–3)
- 패널 분할(small multiples)로 색 수를 줄이는 방식(Rougier 2014 Fig. 7, Stoelzle & Stein 2021 Fig. 6h, Midway 2020 원칙 6).
- 소표본 자료의 개별 점 표시(Weissgerber 2015).

금지 (근거와 사용자 과거 지적의 대응)

| 금지 요소 | 외부 근거 | 사용자 과거 지적(`design/baseline_audit_2026-10-04.md` 2.2절) |
|---|---|---|
| 글씨를 담은 상자, 카드 상자, 외곽 프레임, 제목 밴드, 균일 상자 격자 | Sci Rep "avoid excessive boxing"; Rougier 2014 규칙 8 (chartjunk) | 외곽 프레임과 제목 밴드, 카드 상자, 균일 상자 격자, 글씨 상자형 설명 도식 |
| 색 배경 구역, 연회색 배경 패널 | Sci Rep "Put all display items on a white background"; Rougier 2014 규칙 8 "gratuitously colored backgrounds" | 색 구역(틴트) 패널, 연회색 배경 패널 |
| 장식 아이콘 | Nature 1 "Superfluous icons and other decorative elements", "Text should be used instead of decorative icons" | 장식 아이콘 |
| 색 글씨, 색 라벨 | Nature 1 "Coloured text; keylines, keys, etc. should be used instead" | 주황 라벨, 판정 태그 상자 |
| 그림 안의 긴 설명문, 사후 강조 라벨, 큰 숫자 카드 | Nature 3 "text, including keys to symbols, should be provided in the legend rather than on the figure itself"; Rougier 2014 규칙 4 | 그래프에 사후로 붙인 강조 라벨, "선+주황 라벨+큰 숫자" 통계 카드 |
| 배경 격자선 | Nature 1 "Background gridlines"; Rougier 2014 규칙 8 "useless grid lines" | (해당 없음) |
| 그림자, 패턴, 3차원 장식 | Nature 1 "Drop shadows", "Patterns"; Sci Rep "three-dimensional 'skyscraper' histograms" | (해당 없음) |
| 복잡한 영상 위 글자, 흰 글자 반전, 겹친 글자 | Nature 1, Nature 2, Nature 3 | (해당 없음) |
| 히스토그램·막대 세로축 절단 | Sci Rep "Never truncate the vertical axis"; Rougier 2014 규칙 7 | (해당 없음) |
| rainbow·jet, 적녹 조합 | Nature 1, Crameri 2020, Stoelzle & Stein 2021 | 냉색 규약(`design/brand_tokens.json`) |
| 곡선·사선 화살표, 번호 틱바 | 직접 근거 문헌 없음. 사용자 규칙 "Keep arrow styles to two kinds or fewer"(`~/.claude/rules/visual-artifacts.md`) | 곡선·사선 화살표, 번호 틱바 |
| 그림에 생성형 AI 사용 | Nature 1 Image Integrity: "The use of any sort of generative AI in figures is not permitted." | (해당 없음) |

### 2.7 색 사용 규칙

색지도 선택
- 순서가 있는 비음수 양(ALT, 오차 폭, 밀도)은 순차형, 0 같은 기준값을 갖는 양(차이, 잔차, 온도 편차)은 발산형을 쓴다. Thyng et al. 2016: "Sequential data should be represented with a monotonically increasing range of lightness values"; 발산형은 "combining two sequential colormaps, but with mirrored lightness values" (p. 9–10). Crameri 2020 Fig. 6 의 결정 흐름도도 같은 분류를 쓴다.
- 지각 균일성이 핵심 조건이다. Kovesi 2015: "The most important factor in designing a colour map is to ensure that the magnitude of the incremental change in perceptual lightness of the colours is uniform." (초록) 같은 글은 공급사 색지도에 "perceptual flat spots that can hide a feature as large as one tenth of the total data range" 가 흔하다고 적는다.
- rainbow 계열의 왜곡 크기: Crameri 2020 은 "visual error can be >7% of the displayed data variation" (Fig. 3 설명), Crameri 2018(GMD) 은 "up to 7.5 %" 로 적는다. Google Turbo 같은 '개선형' rainbow 도 "the perceptual uniformity requirement of a science-ready colour map is not met" 로 판정한다(Crameri 2020, p. 5).
- 수문학 문헌 조사: Stoelzle & Stein 2021 은 약 1,000편 중 "16 %–24 % of the publications have a rainbow color map" 이고 "18 %–29 %" 가 적녹 요소를 쓴다고 보고한다(초록).

점검 4항목 (Crameri 2020, p. 7–8, 그대로 옮김)
1. "The colour bar should be perceptually uniform to prevent data distortion and visual error."
2. "The colour map should not contain red and green at a similar luminosity"
3. "The common rainbow colour map should not be used in data visualisation."
4. "The most secure test is to discard any colour map that is not described as scientifically derived (for example, is not listed in Box 2)"

Box 2 의 과학적 색지도 출처: Colorbrewer, Matplotlib(viridis 등), cividis(Nuñez et al. 2018), cmocean(Thyng et al. 2016), CET(Kovesi 2015), Scientific colour maps(Crameri, Zenodo doi:10.5281/zenodo.1243862). 프로젝트가 쓰는 oslo, davos, broc 은 Crameri 2018(GMD 11, 2541–2562)에서 처음 공개된 Scientific colour maps 이다.

색지도 변형 주의
- Crameri 2020: "parts of a scientifically derived colour map cannot be subsequently deformed by being partly squeezed or elongated ... altering any available scientifically derived colour map is not recommended." (p. 6–7)
- 프로젝트 토큰은 oslo_r 0.12–0.90, acton 0.05–0.95 처럼 양 끝을 자른다. 선형으로 구간만 자르면 남은 구간의 균일성은 유지되지만 명도 범위가 줄어든다. 자를 경우 범례(캡션)나 Methods 에 적는다 [판단].

범주 색
- 범주 4개 이하에서는 Nature 1 표의 Wong 팔레트에서 고른다. 프로젝트 냉색 규약을 따르면 Blue #0072b2, Sky blue #56b4e9, Bluish green #009e73, Black #000000 을 기본으로 하고 Orange #e69f00 을 강조 1색으로 둔다 [판단].
- 색만으로 범주를 구분하지 않는다. 표지 모양, 선 종류, 직접 라벨을 함께 쓴다(Stoelzle & Stein 2021 Table 1 "Add second data encoding (e.g., point shapes) to resolve color ambiguity").
- 글자는 검정 또는 흰색으로 쓰고 배경 대비 4.5:1 이상을 지킨다(Nature 1, WCAG 2.1 AA).

점검 절차
- 색각 이상 모사(deuteranopia, protanopia, tritanopia)와 회색조 변환으로 확인한다(Crameri 2020 Fig. 2, Stoelzle & Stein 2021 Table 1 "run a CVD emulation", Copernicus 지침의 Coblis). 프로젝트에는 `src/polar/cvd.py` 가 있다.

### 2.8 자료 표현과 통계

- 연속 자료를 막대+오차막대로만 그리지 않는다. Weissgerber et al. 2015: "many different data distributions can lead to the same bar or line graph. The full data may suggest different conclusions from the summary statistics." 소표본은 개별 점(univariate scatterplot)을 보인다.
- 오차막대의 정의(s.d., s.e.m., 구간)와 n 을 범례에 적는다. 소표본에서는 범위를 쓴다(Sci Rep 통계 지침).
- 지역 4–5개 같은 소표본 비교는 "less than about 10" 에 해당하므로 소표본 검정 또는 그 정당화가 필요하다(Sci Rep 통계 지침).
- 원 면적과 반지름 혼동, 3차원·원그래프로 양 비교를 피한다(Rougier 2014 규칙 7, Fig. 6).
- 불확실성을 표시한다(Midway 2020 원칙 5 "Include Uncertainty").
- 자료와 모형 출력을 시각적으로 구분한다(Midway 2020 원칙 7 "Data and Models Are Different Things").

### 2.9 이미지 무결성과 AI 사용

- Nature research figure guide (Image Integrity): "The final image must correctly represent the original data and conform to community standards." "Images gathered at different times or from different samples should not be combined into a single image." "The use of any sort of generative AI in figures is not permitted. This includes content-aware editing tools in Photoshop."
- Sci Rep 편집 정책(https://www.nature.com/srep/journal-policies/editorial-policies, Digital image integrity and standards): "Processing (such as changing brightness and contrast) is appropriate only when it is applied equally across the entire image and is applied equally to controls. Contrast should not be adjusted so that data disappear."
- Nature Portfolio 편집 정책(https://www.nature.com/nature-portfolio/editorial-policies/image-integrity) 현미경 항목: "The display lookup table (LUT) and the quantitative map between the LUT and the bitmap should be provided, especially when rainbow pseudocolor is used." 위성·재분석 지도에도 같은 원칙(색 대응표와 범위 명시)을 적용한다 [판단].
- Nature Portfolio AI 정책(https://www.nature.com/nature-portfolio/editorial-policies/ai)은 "Fabricating data, citations or results" 와 "Creating photorealistic images (deepfakes)" 를 'Not permitted' 로 분류한다.
- 프로젝트 적용: 그림은 모두 코드로 자료에서 생성하고, 개념도도 사람이 편집 가능한 벡터로 만든다. 원고 작성에 쓴 LLM 사용 사실은 Methods 끝에 기록한다(2.1절).

### 2.10 범례(캡션) 작성 규칙

- 첫 문장은 그림 전체를 요약하는 짧은 제목 문장이다(Sci Rep, Nature 3). 그림 자체에는 제목을 넣지 않는다. 결론 문장은 범례 첫 문장과 본문에서 말한다 [판단: Nature 3 의 "text ... in the legend rather than on the figure itself" 와 Rougier 2014 Fig. 3 설명("no title")에 근거].
- 패널별로 무엇을 보이는지 적고, 방법 세부는 최소화한다(Sci Rep "minimise the methodological details", Nature 3 "legends should not contain any details of methods").
- 색 이름 서술("open red triangles") 대신 기호를 쓰고, 그림 안에 열쇠를 둔다(Sci Rep, Nature 1, Copernicus).
- 오차막대 정의, n, 통계 처리 방법을 적는다(Sci Rep).
- 350단어 이하(Sci Rep).
- 그림만 보고 이해할 수 있게 캡션을 쓴다. Rougier 2014 규칙 4: "The caption explains how to read the figure and provides additional precision for what cannot be graphically represented." Midway 2020 원칙 8: "Simple Visuals, Detailed Captions".

---

## 3. 연구 발표 슬라이드 원칙

### 3.1 assertion-evidence 구조

Alley & Neeley 2005 (Technical Communication 52(4), 417–426) Table 1, p. 420 의 지침을 그대로 옮긴다.

양식
- "For every slide, but the title slide, use a sentence headline that states the slide's main assertion; left justify the headline in the slide's upper left corner."
- "In the body of each slide, present supporting evidence in a visual way—with images, graphs, or visual arrangements of text (such as a table or text blocks connected by arrows)."
- "Avoid bulleted lists because such lists do not show the connections among the listed items."
- "Limit the number of slides so that at least 1 minute can be spent on each slide (preferably more time in a longer presentation such as an hour seminar)."

글자
- "Use a sans serif typeface such as Arial (in rooms that seat more than 20 people, boldface that text)."
- "On a typical slide, use 28 point type for the headline and 18–24 point type for the body text (larger type is appropriate for the title on the title slide)."
- "Avoid setting text in all capital letters."

배치
- "Keep blocks of text, including headlines, to one or two lines."
- "Keep lists to two, three, or four items."
- "Be generous with white space, but give preference to internal white space between text blocks and graphic elements within the slide, as opposed to border white space on the slide's edges"

구성
- "End with the conclusion slide because that slide is the most important slide of the presentation."

사용자 지시와의 관계. 사용자 문체 규칙은 그림·표 제목을 명사형으로 쓰게 한다(`~/.claude/rules/writing-tone.md`). assertion-evidence 의 헤드라인은 주장 문장이다. 두 규칙을 함께 지키려면 슬라이드 헤드라인은 과장 없는 평서문(~이다/~한다) 한 문장으로 쓰고, 표·그림 안의 소제목과 축 이름은 명사형으로 둔다 [판단]. 예: "물리 기준선 위 잔차 학습은 라벨 160개 이상에서만 이득이 있다" (헤드라인), "라벨 수 (개)" (축 이름). 이 예시 문장의 수치는 형식 예시이며 결과 수치가 아니다.

### 3.2 실증 근거와 한계

| 연구 | 설계 | 결과 |
|---|---|---|
| Alley et al. 2006, Technical Communication 53(2), 225–234 | 대형 지질학 강의 4개 반, 같은 강사·같은 내용, 구 헤드라인 대 문장 헤드라인 슬라이드 | 헤드라인 내용 회상 문항 15개 평균 정답률 69% 대 79%, "statistically significant at the .001 significance level" (p. 229) |
| Garner & Alley 2013, Int. J. Eng. Educ. 29(6), 1564–1579 | 공학 학생 110명, 다매체 학습 원칙 6개를 지킨 assertion-evidence 슬라이드 대 PowerPoint 기본형 | "superior comprehension and fewer misconceptions for the assertion–evidence group as well as lower perceived cognitive load", 지연 사후검사 회상도 높음 (초록) |
| Garner & Alley 2016, Int. J. Eng. Educ. 32(1A), 39–54 | 학생 120명이 MRI 원리 슬라이드 작성, assertion-evidence(n = 59) 대 자유 양식(n = 61) | 작성자 본인의 이해도에서 "statistically significant advantage (p < 0.05)" (초록) |
| Garner et al. 2009, Technical Communication 56(4), 331–345 | PowerPoint 일반 관행 분석 | 일반 관행은 기본값의 영향을 크게 받고 다매체 학습 원칙을 따르지 않는다 (요약) |
| Kosslyn et al. 2012, Front. Psychol. 3:230 | 슬라이드쇼 140개 분석(Study 1), 설문(Study 2), 위반 식별 실험(Study 3) | 심리학 원칙 8개(Relevance, Appropriate Knowledge, Salience, Discriminability, Perceptual Organization, Compatibility, Informative Change, Limited Capacity) 위반이 분야와 무관하게 흔하다 |

Garner & Alley 2013 이 든 다매체 학습 원칙 6개는 multimedia(그림+말), contiguity(관련 요소를 공간·시간상 가깝게), redundancy(화면 글과 말을 똑같이 반복하지 않음), modality(설명은 말로), coherence(비핵심 정보 제거), signaling(구조 단서 제공)이다(p. 1565).

Kosslyn et al. 2012 의 수치 근거: 작업기억은 "about four such units" 를 유지한다(Cowan 2001 인용). 같은 글은 글머리 항목당 두 줄 제한의 근거로 "in general two lines of text convey about four concepts" 를 든다. 직접 라벨은 열쇠 탐색 부담을 줄인다("Eliminating the need to search for labels – for instance by directly labeling items in a display rather than using a key – reduces processing load").

한계. 실증 연구 대부분이 교육 환경이고 Alley 연구진이 수행했다. 학회 발표나 연구 보고 환경에서 같은 효과 크기가 나온다는 근거는 이 조사에서 찾지 못했다 [미확인]. 따라서 효과 크기를 인용하지 않고 설계 원칙으로만 쓴다.

### 3.3 Ten simple rules 계열의 핵심

Bourne 2007 (PLoS Comput Biol 3(4): e77)
- 규칙 2 "Less is More", 규칙 4 "Make the Take-Home Message Persistent": 일주일 뒤 청중이 핵심 세 가지를 기억해야 한다.
- 규칙 8 "Use Visuals Sparingly but Effectively": "if you have more than one visual for each minute you are talking, you have too many" "Avoid reading the visual" "do not overload the visual. Make the points few and clear."

Naegle 2021 (PLoS Comput Biol 17(12): e1009554)
- 규칙 1 "Include only one idea per slide", 규칙 2 "Spend only 1 minute per slide": "a 20-minute presentation should have somewhere around 20 slides."
- 규칙 3 "Make use of your heading": "Instead of titling the slide "Results," try "CTNND1 is central to metastasis""
- 규칙 6 "Use graphics effectively": "you should almost never have slides that only contain text" "A multipanel figure that you might include in a manuscript should often be broken into 1 panel per slide"
- 규칙 7 "Design to avoid cognitive overload": 화면의 완전한 문장은 말과 같은 처리 경로를 써서 과부하를 만든다. 요소 수 6개 이하를 권한다(근거는 TEDx 강연 인용이라 약하다). sans-serif, 큰 글자(그림 범례 포함), 기울임·밑줄·전부 대문자 지양, 강조는 굵게.
- 규칙 8 "Design the slide so that a distracted person gets the main takeaway"
- 규칙 10: 애니메이션 지양, PDF 사본 준비, 동영상은 정지 화면 예비 슬라이드.

Lortie 2017 (PLoS Comput Biol 13(3): e1005373), 짧은 발표용
- 규칙 2 "Provide only one major point per slide", 규칙 3 "Limit use of text"
- 규칙 4 "Use simple visuals": "Do not cut and paste figures prepared for written papers"
- 규칙 5 "Develop a consistent theme": "Do not develop your brand using canned templates."

### 3.4 논문 그림을 슬라이드로 옮길 때

- Rougier 2014 규칙 3: 발표용 그림은 "figure elements must consequently be made thicker (lines) or bigger (points, text), colors should have strong contrast, and vertical text should be avoided" 이고 "you should abandon the practice of extracting a figure from your article to be put, as is, in your oral presentation." Fig. 3 의 발표판은 궤적 수, 눈금 수, 제목을 줄이고 선을 굵게 했다.
- 다패널 논문 그림은 패널 하나 또는 두 개씩 나눈다(Naegle 2021 규칙 6).
- 논문 그림과 같은 자료·같은 색지도·같은 범위를 쓰되 슬라이드 크기용으로 다시 그린다. 그림 생성 코드에 매체 인자(paper/slide)를 두는 방식을 권한다 [판단].

### 3.5 슬라이드 수치 규칙

| 항목 | 값 | 근거 |
|---|---|---|
| 헤드라인 | 문장 1개, 1–2줄, 좌상단 좌측 정렬, 28 pt | Alley & Neeley 2005 |
| 본문 글자 | 18–24 pt, 20명 넘는 방에서는 굵게 | Alley & Neeley 2005 |
| 그림 안 글자 | 본문 하한 18 pt 를 원칙으로, 눈금 숫자는 14 pt 까지 허용 [판단] | Alley & Neeley 2005(본문), Naegle 2021 규칙 7("large font sizes (including figure legends)") |
| 글 덩어리 | 2줄 이하 | Alley & Neeley 2005, Kosslyn et al. 2012 |
| 목록 | 2–4항목, 글머리표 목록 자체를 지양 | Alley & Neeley 2005, Kosslyn et al. 2012 |
| 한 장의 요소 | 6개 이하 | Naegle 2021 (근거 약함) |
| 장 수 | 발표 1분당 1장 안팎 | Naegle 2021, Bourne 2007, Alley & Neeley 2005 |
| 서체 | sans-serif 한 종류, 전부 대문자·밑줄·기울임 지양 | Alley & Neeley 2005, Naegle 2021 |
| 애니메이션 | 쓰지 않는다. 단계적 공개가 필요하면 별도 슬라이드로 나눈다 | Naegle 2021 규칙 1·10 |
| 내보내기 | PDF 사본, 동영상 예비 정지 화면 | Naegle 2021 규칙 10 |

---

## 4. 프로젝트 현행 설정과의 대조

대상: `src/polar/paperstyle.py`(논문 그림), `design/brand_tokens.json`·`design/layout_rules.md`(발표·대회용). 이 문서는 파일을 고치지 않는다. 아래는 변경 권고이다.

| 항목 | 현행 값 | 기준 | 판정 | 권고 |
|---|---|---|---|---|
| 전폭 캔버스 `W2_MM` | 180 mm | Sci Rep 실측 최대 170 mm | 0.944 배 축소됨 | 170 mm 로 바꾸거나 최소 글자 5.5 pt 이상 유지 |
| 반폭 캔버스 `W1_MM` | 88 mm | Nature 89 mm, Sci Rep 실측 71–82 mm | 0.80–0.93 배 축소 가능 | 85 mm, 최소 글자 6 pt 이상 |
| 최대 높이 `HMAX_MM` | 200 mm | Nature 170 mm, Sci Rep 실측 최대 194 mm | 실측 최대보다 6 mm 큼 | 190 mm 로 낮춤 |
| 본문 글자 `FS["base"]`, `FS["label"]` | 7.5 pt | 5–7 pt | 180 mm 기준으로는 최대값 초과, 170 mm 축소 후 7.1 pt | 170 mm 캔버스로 바꿀 경우 7.0 pt |
| 눈금·범례 `FS["tick"]`, `FS["legend"]` | 7.0, 6.5 pt | 5–7 pt, "same font size for all figures" | 범위 안 | 크기 종류를 2–3개로 고정 |
| 패널 문자 `FS["panel"]` | 9 pt | Nature 8 pt, Sci Rep 본문과 같은 크기 | 초과 | 8 pt |
| 축·눈금 선 `LW["axis"]`, `LW["tick"]` | 0.6, 0.5 pt | Sci Rep 최소 1 pt | Sci Rep 문구 미달 | 1.0 pt 검토(2.3절) |
| 주 선 `LW["main"]` | 1.0 pt | Sci Rep 최소 1 pt | 경계값 | 1.0–1.5 pt |
| 부 눈금·해칭 | 0.4 pt | Sci Rep 최소 1 pt, Nature 하한 0.25 pt | Sci Rep 문구 미달 | 부 눈금 제거, 해칭은 0.5 pt 이상 [판단] |
| 글꼴 내장 | `pdf.fonttype` 42 | Type 42 | 적합 | 유지 |
| 서체 | Arial → Liberation Sans 대체 | Arial 또는 Helvetica | 메트릭 호환 대체 | 투고본은 Arial 설치 후 재생성 권장 |
| 래스터 저장 | 600 dpi PNG | 300–450 dpi 이상 | 적합 | 유지 |
| 격자선 | `brand_tokens` grid_alpha 0.25 | Nature 1 "Background gridlines" 지양 | 논문 그림에는 부적합 | 논문 그림에서 격자 제거 |
| 그림 안 결론형 제목 | `layout_rules.md` 1절 "제목은 결론형" | 논문 그림은 제목을 범례 첫 문장으로 | 논문에는 부적합, 슬라이드에는 적합 | 논문: 그림 제목 없음. 슬라이드: 문장 헤드라인 |
| 슬라이드 글자 | `brand_tokens` 제목 15 pt, 축 11 pt, 눈금 9 pt (12.8 × 7.2 in 캔버스) | 헤드라인 28 pt, 본문 18–24 pt | 작다. Alley 지침은 높이 7.5 in 슬라이드(당시 4:3 기본값 10 × 7.5 in) 기준으로 보이며, 높이 7.2 in 캔버스로 환산해도 헤드라인 약 27 pt, 본문 약 17 pt 이다 [판단] | 덱 사양에서 헤드라인 28 pt, 본문 18 pt 이상 |
| 색지도 | oslo_r·acton·davos_r 끝 자름 | Crameri 2020 변형 비권장 | 조건부 | 자를 경우 범례나 Methods 에 기록 |

---

## 5. 제출 전 점검표

그림 (논문)
- [ ] 폭 170 mm 또는 85 mm 캔버스, 높이 190 mm 이하
- [ ] 최종 크기에서 글자 5.5–7 pt, 패널 문자 8 pt 굵은 직립 소문자, 서체 1종
- [ ] 그림 안 제목 없음, 설명문 없음, 상자·배경 구역·격자선·그림자·패턴·아이콘 없음
- [ ] 모든 축에 "Variable (unit)" 과 눈금, 모든 색 막대에 이름과 단위
- [ ] 숫자와 단위 사이 한 칸, 다섯 자리 이상 숫자에 천 단위 쉼표, 좌표는 "65° N"
- [ ] 가장 얇은 선 1 pt 이상(Sci Rep 문구), 데이터 선 1.0–1.5 pt
- [ ] 지각 균일 색지도, rainbow·jet·적녹 없음, 색각 이상 모사와 회색조 점검 통과
- [ ] 색 글씨 없음, 범주는 색과 표지 모양을 함께 사용, 글자 대비 4.5:1 이상
- [ ] 소표본은 개별 점 표시, 오차막대 정의와 n 을 범례에 기록
- [ ] 지도: 축척 막대(길이를 막대 위에), 경위도 표시, 색 범위 고정
- [ ] PDF(Type 42 글꼴 내장, 윤곽선 변환 없음) + 600 dpi PNG
- [ ] 범례: 제목 문장으로 시작, 패널별 설명, 방법 세부 최소, 350단어 이하
- [ ] 생성형 AI 사용 없음

슬라이드
- [ ] 모든 본문 슬라이드에 문장 헤드라인 1개(1–2줄, 좌측 정렬, 28 pt), 과장 없는 평서문
- [ ] 본문은 그림·표·도식 근거, 글머리표 목록 없음, 글 덩어리 2줄 이하
- [ ] 본문 글자 18 pt 이상, 그림 눈금 14 pt 이상, sans-serif 1종
- [ ] 슬라이드 1장 = 주장 1개, 장 수 ≈ 발표 분 수
- [ ] 논문 그림을 그대로 붙이지 않고 슬라이드용으로 다시 그림(선 굵게, 글자 크게, 세로 글자 없음)
- [ ] 애니메이션 없음, PDF 사본 준비

---

## 6. 내려받은 문헌 (`references/13_figure_and_slide_design/`)

모두 합법적 공개본이다. 출판사 공개 쪽, arXiv, 저자 소속 기관(Penn State 공학대학 글쓰기 사이트 writing.engr.psu.edu)에서 받았다. 파일마다 `file`·`pdfinfo` 로 PDF 여부와 쪽 수를 확인했다. 기존 `references/` 아래에 같은 문헌은 없었다.

| 파일 | 서지 | DOI 또는 출처 | 공개 형태 | 용도 |
|---|---|---|---|---|
| rougier2014_ten_simple_rules_better_figures.pdf | Rougier, N. P., Droettboom, M. & Bourne, P. E. Ten simple rules for better figures. PLoS Comput. Biol. 10, e1003833 (2014) | 10.1371/journal.pcbi.1003833 | 출판사 OA (CC0) | 그림 설계 원칙 |
| crameri2020_misuse_of_colour.pdf | Crameri, F., Shephard, G. E. & Heron, P. J. The misuse of colour in science communication. Nat. Commun. 11, 5444 (2020) | 10.1038/s41467-020-19160-7 | 출판사 OA (CC BY) | 색지도 원칙·점검 |
| crameri2018_staglab_scientific_visualisation.pdf | Crameri, F. Geodynamic diagnostics, scientific visualisation and StagLab 3.0. Geosci. Model Dev. 11, 2541–2562 (2018) | 10.5194/gmd-11-2541-2018 | Copernicus OA (CC BY) | oslo·davos·broc 원 출처 |
| stoelzle2021_rainbow_colormap_hydrology.pdf | Stoelzle, M. & Stein, L. Rainbow color map distorts and misleads research in hydrology – guidance for better visualizations and science communication. Hydrol. Earth Syst. Sci. 25, 4549–4565 (2021) | 10.5194/hess-25-4549-2021 | Copernicus OA (CC BY) | 지구과학 색 점검표 |
| thyng2016_true_colors_oceanography_colormaps.pdf | Thyng, K. M., Greene, C. A., Hetland, R. D., Zimmerle, H. M. & DiMarco, S. F. True colors of oceanography: guidelines for effective and accurate colormap selection. Oceanography 29(3), 9–13 (2016) | 10.5670/oceanog.2016.66 | 출판사 무료 공개(교육·연구 복사 허용, 재게시 제한) | cmocean, 색지도 분류 |
| nunez2018_cividis_cvd_colormaps.pdf | Nuñez, J. R., Anderton, C. R. & Renslow, R. S. Optimizing colormaps with consideration for color vision deficiency to enable accurate interpretation of scientific data. PLoS ONE 13, e0199239 (2018) | 10.1371/journal.pone.0199239 | 출판사 OA (CC0) | cividis, 색각 이상 |
| kovesi2015_good_colour_maps.pdf | Kovesi, P. Good colour maps: how to design them. arXiv:1509.03700 (2015) | arXiv:1509.03700 | arXiv | 지각 균일 설계 |
| weissgerber2015_beyond_bar_line_graphs.pdf | Weissgerber, T. L., Milic, N. M., Winham, S. J. & Garovic, V. D. Beyond bar and line graphs: time for a new data presentation paradigm. PLoS Biol. 13, e1002128 (2015) | 10.1371/journal.pbio.1002128 | 출판사 OA (CC BY) | 소표본 자료 표현 |
| jambor2021_image_based_figures.pdf | Jambor, H. et al. Creating clear and informative image-based figures for scientific publications. PLoS Biol. 19, e3001161 (2021) | 10.1371/journal.pbio.3001161 | 출판사 OA (CC BY) | 영상 그림, 축척·주석 |
| alley2005_sentence_headlines_visual_evidence.pdf | Alley, M. & Neeley, K. A. Rethinking the design of presentation slides: a case for sentence headlines and visual evidence. Tech. Commun. 52(4), 417–426 (2005) | DOI 없음, https://writing.engr.psu.edu/2005_alley_neeley.pdf | 저자 소속 기관 공개 | assertion-evidence 지침 |
| alley2006_headline_design_audience_retention.pdf | Alley, M., Schreiber, M., Ramsdell, K. & Muffo, J. How the design of headlines in presentation slides affects audience retention. Tech. Commun. 53(2), 225–234 (2006) | DOI 없음, https://writing.engr.psu.edu/alley_et_al_2006.pdf | 저자 소속 기관 공개 | 문장 헤드라인 실증 |
| garner2009_powerpoint_vs_assertion_evidence.pdf | Garner, J. K., Alley, M., Gaudelli, A. F. & Zappe, S. E. Common use of PowerPoint versus the assertion–evidence structure: a cognitive psychology perspective. Tech. Commun. 56(4), 331–345 (2009) | DOI 없음, https://writing.engr.psu.edu/Garner_et_al_2009.pdf | 저자 소속 기관 공개 | 인지심리 근거 |
| garner2013_assertion_evidence_comprehension.pdf | Garner, J. K. & Alley, M. P. How the design of presentation slides affects audience comprehension: a case for the assertion–evidence approach. Int. J. Eng. Educ. 29(6), 1564–1579 (2013) | DOI 없음, https://writing.engr.psu.edu/ae_comprehension.pdf | 저자 소속 기관 공개 | 이해도 실증 |
| garner2016_slide_structure_presenter_understanding.pdf | Garner, J. K. & Alley, M. P. Slide structure can influence the presenter's understanding of the presentation's content. Int. J. Eng. Educ. 32(1A), 39–54 (2016) | DOI 없음, https://writing.engr.psu.edu/Garner_et_al_2016.pdf | 저자 소속 기관 공개 | 작성자 이해도 실증 |
| kosslyn2012_powerpoint_flaws_psychological_analysis.pdf | Kosslyn, S. M., Kievit, R. A., Russell, A. G. & Shephard, J. M. PowerPoint presentation flaws and failures: a psychological analysis. Front. Psychol. 3, 230 (2012) | 10.3389/fpsyg.2012.00230 | 출판사 OA | 인지 원칙 8개 |
| bourne2007_ten_simple_rules_oral_presentations.pdf | Bourne, P. E. Ten simple rules for making good oral presentations. PLoS Comput. Biol. 3, e77 (2007) | 10.1371/journal.pcbi.0030077 | 출판사 OA (CC BY) | 발표 일반 |
| naegle2021_ten_simple_rules_presentation_slides.pdf | Naegle, K. M. Ten simple rules for effective presentation slides. PLoS Comput. Biol. 17, e1009554 (2021) | 10.1371/journal.pcbi.1009554 | 출판사 OA (CC BY) | 슬라이드 설계 |
| lortie2017_ten_simple_rules_short_presentations.pdf | Lortie, C. J. Ten simple rules for short and swift presentations. PLoS Comput. Biol. 13, e1005373 (2017) | 10.1371/journal.pcbi.1005373 | 출판사 OA | 짧은 발표 |

링크만 기록한 문헌

| 서지 | DOI·주소 | 상태 |
|---|---|---|
| Midway, S. R. Principles of effective data visualization. Patterns 1, 100141 (2020) | 10.1016/j.patter.2020.100141, PMC7733875 | OA(CC BY-NC-ND)이나 Cell·PMC·Europe PMC PDF 가 자동 다운로드를 차단했다(HTTP 403). 본문 XML 로 내용만 확인했다. 브라우저로 받을 것 |
| Light, A. & Bartlein, P. J. The end of the rainbow? Color schemes for improved data graphics. Eos 85(40), 385–391 (2004) | 10.1029/2004EO400002 | 무료 공개(bronze)이나 Wiley 가 자동 다운로드를 차단했다(HTTP 403) |
| Wong, B. Points of view: Color blindness. Nat. Methods 8, 441 (2011) | 10.1038/nmeth.1618 | 구독 필요. Nature 그림 지침이 팔레트 근거로 인용 |
| Kelleher, C. & Wagener, T. Ten guidelines for effective data visualization in scientific publications. Environ. Model. Softw. 26, 822–827 (2011) | 10.1016/j.envsoft.2010.12.006 | 구독 필요 |
| Crameri, F. Scientific colour maps. Zenodo (2018) | 10.5281/zenodo.1243862 | 소프트웨어 보관본. 색지도 사용 시 인용 대상(Copernicus 지침도 인용을 요구) |

---

## 7. 남은 확인 사항

1. Sci Rep 패널 문자 크기. Sci Rep 은 "same type size as used elsewhere in the figure", Nature 그림 지침은 8 pt 이다. 최근 Sci Rep 게재본은 그림이 래스터로 들어가 글자 크기를 직접 잴 수 없다. 투고 전 Sci Rep 원고 담당에 문의하거나 7 pt 로 통일하는 방안을 함께 둔다 [미확인].
2. Sci Rep 최소 선폭 1 pt 규칙의 실제 적용 여부 [미확인].
3. Sci Rep 그림 폭의 공식 수치. 지침 쪽에는 mm 규정이 없다. 이 문서의 170 mm 는 게재본 9편 실측에서 나온 값이다.
4. AGU 그림 지침(HTTP 403) [미확인].
5. assertion-evidence 효과가 학회·연구 보고 환경에서도 같은지에 대한 독립 연구 [미확인].

---

## 출처 URL (2026-10-04 접속)

- Scientific Reports, Submission guidelines: https://www.nature.com/srep/author-instructions/submission-guidelines
- Scientific Reports, Editorial policies: https://www.nature.com/srep/journal-policies/editorial-policies
- Nature research figure guide, Preparing figures: https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/
- Nature research figure guide, Building and exporting figure panels: https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/
- Nature research figure guide, Image Integrity: https://research-figure-guide.nature.com/figures/image-integrity/
- Nature research figure guide, Top 10 ways to delay your paper: https://research-figure-guide.nature.com/figures/top-10-ways-to-delay-your-paper/
- Nature, Final submission: https://www.nature.com/nature/for-authors/final-submission
- Nature, Formatting guide: https://www.nature.com/nature/for-authors/formatting-guide
- Nature Portfolio, Image integrity and standards: https://www.nature.com/nature-portfolio/editorial-policies/image-integrity
- Nature Portfolio, Artificial Intelligence (AI): https://www.nature.com/nature-portfolio/editorial-policies/ai
- The Cryosphere (Copernicus), Submission: https://www.the-cryosphere.net/submission.html
- Elsevier, Artwork sizing: https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-sizing
- Penn State, assertion-evidence 참고문헌: https://writing.engr.psu.edu/slides_references.html, https://www.assertion-evidence.com/references.html
