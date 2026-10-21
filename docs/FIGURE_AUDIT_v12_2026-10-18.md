# 논문 그림 · 표 감사와 v12 그림 양식 (2026-10-18)

대상: 영문 원고 v11(`paper/manuscript/en_v11/main.pdf`, 22쪽)의 본문 그림 7장 · 표 1개, 보충 그림 10장 · 표 6개(`si.pdf`). 사용자 요청 세 가지(배치 재검토, 글꼴 · 색 · AI 투 점검, 상위 학술지 그림 분석)에 대한 결과와, 그 결과로 정한 v12 그림 양식을 적는다. 수치와 인용은 모두 이번 세션에서 직접 확인한 것이다.

## 1. 요약

1. 배치: 본문 그림 7장과 표 1은 모두 처음 인용한 쪽의 1–2쪽 뒤에 놓이고, 번호 순서와 인용 순서가 같다. 고칠 것은 보충 그림의 파일명과 번호 불일치(파일 `FigS10_label_ladder` 가 Supplementary Fig. S4), 보충 그림 설명문의 제목 문장 부재, 보충 표 라벨 불일치뿐이다.
2. 글꼴: 그림 글자는 Helvetica 또는 Arial 계열이 규정이자 관행이다. 현행 FreeSans 는 Helvetica 복제(Nimbus Sans 계열)이므로 계열은 맞다. 설명문(캡션) 글꼴은 저자가 정하지 않고 저널이 본문 서체로 조판한다. 따라서 'AI 투'의 원인은 글꼴이 아니다.
3. AI 투의 원인: 순흑 대신 짙은 회색 글자(#262626) · 회색 축(#404040), 0.5 pt 축선, 채도를 낮춘 팔레트, 연회색 격자선, 그림 상단 가로 범례, Fig 2 의 행 머리글 + 헤어라인 · 둥근 파스텔 블록 · 열린 셰브런 화살 · 'Schematic' 가짜 곡선 · 요소 14개 과밀, Fig 1c 의 간트 차트형 범례 5종이다. 상위 학술지의 사람이 만든 그림은 순흑 글자 · 0.75–1 pt 선 · 원색 · 격자 없음 · 채운 화살촉 · 요소 8개 이하다.
4. v12 양식: 순흑 글자 · 축, 축 0.8 pt · 자료선 1.1–1.4 pt, Okabe–Ito 원색(제안 방법 #D55E00, Stefan #0072B2), 격자선 없음, 글자 7 pt 단일(패널 문자 8 pt 굵게), Fig 2 를 3행(설계 · 모델 · 검증과 산출 지도)으로 재구성, Fig 1c 단순화, 보충 그림 파일명 정리. 명세는 `design/figure_spec_v12.json`.

## 2. 그림 · 표 배치 점검 (en_v11, A4 22쪽)

`pdftotext` 로 쪽마다 첫 인용과 설명문 위치를 대조했다.

| 항목 | 첫 인용 쪽(절) | 배치 쪽 | 간격 | 판정 |
|---|---|---|---|---|
| Table 1 | 3 (Data and methods, Study regions) | 4 | +1 | 적정 |
| Fig. 1 | 3 (같은 문장에서 Table 1 과 함께) | 5 | +2 | 적정. 표가 먼저 놓여 그림이 한 쪽 밀림 |
| Fig. 2 | 6 (Model formulation) | 7 | +1 | 적정, 전면 그림 |
| Fig. 3 | 9 (Results 첫 절) | 10 | +1 | 적정 |
| Fig. 4 | 11 (Sample efficiency) | 12 | +1 | 적정 |
| Fig. 5 | 13 (Comparison with products) | 14 | +1 | 적정 |
| Fig. 6 | 13 (같은 절 넷째 문단) | 15 | +2 | 적정, Fig. 5 뒤 유지 |
| Fig. 7 | 15 (Year-labelled maps) | 16 | +1 | 적정 |

- 본문 인용 순서는 번호 순서(1 → 7)와 같고, 패널 인용(1b, 3a–d, 4a–c, 5a–b, 7b–c)과 보충 그림 S1 → S10, 보충 표 S1 → S6 도 순서가 맞다. Introduction 과 Discussion 은 그림을 인용하지 않는다.
- 보충 그림: `si.tex` 의 `\sifig` 순서가 번호를 정하므로 파일 `FigS10_label_ladder` 가 S4 가 되고, 파일 S4 → S5, S5 → S6, S6 → S7, S7 → S8, S8 → S9, S9 → S10 으로 한 칸씩 밀린다. 라벨(`fig:s11`, `fig:s5` 등)도 번호와 다르다. 본문 인용은 번호 기준으로 맞지만 유지 보수 위험이 크다. v12 에서 파일명과 라벨을 번호에 맞춘다.
- 보충 그림 설명문에 굵은 제목 문장이 없다. Sci Rep 지침("begin your figure legends with a brief title sentence for the whole figure")을 보충 그림에도 적용한다.
- 보충 표 라벨: S4(실행 목록)가 `tab:s6`, S5 가 `tab:s4`, S6 가 `tab:s5`. 번호에는 영향이 없으나 정리한다.
- 그림 크기: 전부 폭 180 mm(510 pt), 높이 106–160 mm. Fig 2(160 mm)는 한 쪽을 다 차지한다.

## 3. 그림 글꼴 조사

### 3.1 규정(원문)

- Scientific Reports 투고 지침(2026-10-18 열람): "Use a clear, sans-serif typeface (for example, Helvetica) for figure lettering. Use the same typeface in the same font size for all figures in your paper." "The thinnest lines in the final figure should be no smaller than one point wide." "Figures divided into parts should be labelled with a lower-case, bold letter (a, b, c and so on) in the same type size as used elsewhere in the figure." "Lettering in figures should be in lower-case type, with only the first letter of each label capitalised." "Put all display items on a white background, and avoid excessive boxing, unnecessary colour, spurious decorative effects". 표시 항목 8개 이하, 설명문 350단어 이하, 선 그림은 벡터(EPS · AI).
- Nature research figure guide(2026-10-18 열람): "All text should be a sans-serif typeface, preferably Helvetica or Arial." 본문 글자 5–7 pt, 패널 문자 "8-pt bold, upright (not italic) and lowercase a, b, c", 선 0.25–1 pt, 피할 것 "Background gridlines, Drop shadows, Patterns, Coloured text, Text placed on top of busy images", 축은 "Axis lines and tick marks" 와 단위 괄호 표기, PDF · EPS, RGB, 글꼴 내장.

### 3.2 실측(참고 논문 33편의 `pdffonts`)

| 학술지 | 편수 | 그림 글꼴 |
|---|---|---|
| Nature Communications(Hjort 2018, Biskaborn 2019, Nitzbon 2020, Ploton 2020, Tsai 2021) | 5 | Helvetica 4, Arial 1 |
| Copernicus(TC · ESSD · GMD: Talucci 2025, Ran 2022, Linnenbrink 2024, Langer 2024, Burke 2020, Du 2026) | 6 | Nimbus Sans(Helvetica 복제) 3, 래스터 3 |
| Scientific Reports(Gautam 2025, Gay 2026, Hall 2026, Jones 2024, Fisher 2023, Feeney 2022, Hatami 2022, Isaksen 2022, Khanal 2023) | 9 | Arial 2, LaTeX Computer Modern 2, Myriad Pro 1, 래스터 4 |
| 그 밖(ERL Liu 2024, GRL Aalto 2018, MEE Meyer 2021, RS Wang 2025) | 4 | Helvetica · Liberation Sans · DejaVu(Liu), Myriad(Aalto), Helvetica(Meyer), 래스터(Wang) |

- Sci Rep 본문은 Minion Pro, 제목은 Corbel 로 저널이 조판한다. 설명문도 같다.
- 결론: 그림 안 글자는 Helvetica · Arial 계열이 표준이고 FreeSans(Helvetica 복제)는 그대로 둔다. Arial 메트릭이 필요하면 Liberation Sans 로 바꿀 수 있으나 외관 차이는 없다. 사용자 질문("보통 논문에서는 그래프나 그림 설명에 어떤 글씨체를 넣지?")의 답은 '그림 안 글자 = Helvetica 또는 Arial, 그림 설명문 = 저널 본문 서체(저자 선택 아님)'이다.

## 4. v11 그림의 AI 투 진단

참고 논문 그림을 원본 해상도로 열람한 뒤 v11 과 대조했다.

| 요소 | v11 | 참고 논문(사람이 만든 그림) | v12 결정 |
|---|---|---|---|
| 글자 · 축 색 | #262626 · #404040 | 순흑(Ploton, Hjort, Tsai, Langer, Gautam 전부) | #000000 |
| 선 굵기 | 축 0.5 pt, 연결선 0.6 pt, 자료선 0.8–1.0 pt | Ploton 축 약 0.75–1 pt, Sci Rep 규정 1 pt 이상 | 축 0.8 pt, 자료선 1.1 pt, 제안 방법 1.4 pt |
| 색 | 채도 낮춘 팔레트(#D1603D, #2E6DA4, #8E5B9E) | 원색(Okabe–Ito, Ploton 적 · 청) | Okabe–Ito 원색 복원 |
| 격자선 | 연회색 가로 격자 | 없음(Nature 금지 항목) | 제거 |
| 글자 크기 | 7 · 6.5 · 6 pt 세 가지 | 한 가지 | 7 pt 단일, 패널 문자 8 pt 굵게 |
| 범례 | 그림 상단 가로 범례(덱 문법) | 패널 안 또는 패널 옆 | 패널 안 |
| 마커 | 3–3.5 pt, 열린 마커 테두리 0.6 pt | 4–5 pt, 테두리 0.8 pt | 4 pt, 0.8 pt |
| Fig 2 도식 | 행 머리글 + 헤어라인, 둥근 파스텔 블록(테두리 0.5 pt 회색), 열린 셰브런, 'Schematic' 가짜 곡선, 구조 글리프 4개, 요소 14개 | Ploton Fig 6: 회색 머리 상자 + 굵은 제목, 흰 상자, 검정 채운 화살촉 0.8 pt, 실자료 썸네일, 요소 8개 | 3행 재구성(7절) |
| Fig 1c | 간트 막대와 범례 5종, 빗금 띠 | 자료 기간은 표 또는 단순 막대 | 막대 2종(사용 연도 진한 색, 가용 기간 가는 선)으로 단순화 |
| Fig 3a | 두 점 기울기 그래프 | 관행 있음 | 유지 |
| 지도 | 옅은 회색 육지, 청색 순차 색표 | Hjort · Langer 와 같은 문법 | 유지, 해안선 0.6 pt |

## 5. 참고 논문 그림 분석(이번 세션 열람분)

- Ploton et al. 2020, Nat Commun. Fig 3: 실제 지도 위 공간 접기(원색 44개), 확대 연결, 검정 축척. Fig 5: 완충 반경에 따른 R² · RMSPE 곡선, 영 모형(검정) 기준선, 오차 막대, 패널 안 범례, 순흑 축. Fig 6(워크플로): 회색 머리 상자(굵은 제목) + 흰 상자(기울임 설명) + 실자료 지도 썸네일 + 작은 표 + 검정 채운 화살. 우리 Fig 2 의 문법 기준.
- Tsai et al. 2021, Nat Commun. Fig 1: PPT 식 원색 평행사변형 · 흰 굵은 글자 · 굵은 화살. 사람 손 티는 강하나 전문성이 낮다. 색을 원색으로 쓰는 점만 참고한다. Fig 2: matplotlib 기본(파랑 · 빨강 · 초록, 10 pt 글자, 격자 없음).
- Hjort et al. 2018, Nat Commun. Fig 3: 지도 + 모서리 확대도 4개 + 축척, 범례 한 상자.
- Langer et al. 2024, TC. Fig 3: 예측 지도 위 관측 잔차 원(두 색표), 산점도 안에 n · RMSE · r · bias 표기. 우리 Fig 6d · h 의 통계 표기와 같은 방식.
- Gautam et al. 2025, Sci Rep(경쟁 논문): ggplot 기본(회색 격자, 그림 제목, 10 pt), 범주색 ALT 지도. Sci Rep 의 하한이 낮음을 보인다. 피할 것.
- Wang et al. 2025, Remote Sens.: 과밀 흐름도(파스텔 배경 4색, 기울임 세리프 글자, 상자 30개 이상). 반면교사.
- Liu et al. 2024, ERL. Fig 4: 무지개 색표, 패널 문자 20 pt. 반면교사.
- Gay et al. 2026, Sci Rep: 31쪽 중 지도 몽타주 그림, 무지개 색표. 반면교사.
- Wei et al. 2026, ESSD(토론): 상관 행렬, Times 글꼴. 참고 안 함.

공통 규칙(10-04 분석 `docs/research/2026-10-04/alt_figure_exemplars.md` 8.1절과 같음): 개념도는 실자료 좌표 위에, 범례보다 직접 라벨, 기준선 동반, 불확실성은 정의를 밝힌 막대 · 띠, 소형 다중 그림은 행 · 열 머리만.

## 6. v12 그림 양식(명세 `design/figure_spec_v12.json`)

- 글꼴: FreeSans Regular, 모든 글자 7 pt, 패널 문자 8 pt Bold 소문자, 굵게는 패널 문자와 도식 상자 제목에만. 수식 기호 기울임.
- 색: 글자 · 축 · 눈금 #000000. 방법색 Okabe–Ito: 제안 방법(Physics + residual ML) #D55E00, Stefan(재보정) #0072B2, Stefan(원천 계수) #808080, 직접 ML #7B3294 파선, 물리 입력 ML #E69F00, 유사라벨 #009E73. 제품: Wei #2CA25F, Liu #D6336C, ESA CCI v4 #8C6D31, v5 #E7B800, Aalto #56B4E9, Yi–Kimball #6B8E23. 지역: 알래스카 #332288, 레나 델타 #3E92CC, 캐나다 #2A8C7A. 영 기준선 #666666 0.6 pt. 지도 색표 유지(ALT oslo_r, 편차 vik).
- 선: 축 · 눈금 0.8 pt, 눈금 길이 2.5 pt 바깥쪽, 왼쪽 · 아래 축만. 자료선 1.1 pt, 제안 방법 1.4 pt, 보조 0.8 pt, 기준선 0.6 pt, 구간 막대 0.8 pt, 해안선 0.6 pt, 경위선 0.4 pt. 격자선 없음.
- 마커: 4.0 pt(정사각형 3.6, 삼각형 4.4), 열린 마커 테두리 0.8 pt.
- 범례: 패널 안 또는 패널 옆, 테두리 없음. 그림 상단 가로 범례 금지.
- 도식: 상자는 흰 채움 또는 #F2F2F2 채움에 검정 0.8 pt 테두리, 모서리 반경 0.6 mm 이하, 상자 제목 7 pt 굵게. 화살은 검정 0.8 pt 직선, 채운 삼각 촉(길이 1.6 mm). 자료 흐름에는 실자료 썸네일. 행 머리글과 헤어라인 금지. 가짜(schematic) 곡선 금지.
- 크기: 폭 180 mm, 높이는 내용에 맞춤. 출력 PDF(글꼴 내장, Type 42) · SVG · PNG 600 dpi. `pdffonts` 결과 FreeSans 계열만.
- Sci Rep 선 굵기 규정(1 pt 이상)과의 관계: 축 0.8 pt 는 Nature 안내(0.25–1 pt)와 참고 논문 관행에 맞춘 값이다. 제작부가 1 pt 를 요구하면 `LW_AXIS` 상수 하나로 바꾼다.

## 7. 그림별 재설계 계획

| 그림 | 변경 |
|---|---|
| Fig 1 | a 지역 상자 1.0 pt, 글자 검정. b 격자 제거, 막대 지역색. c 막대 2종(사용 연도 = 진한 색, 가용 기간 = 가는 회색 선), 분석 구간 연회색 띠, 정적 자료는 글자로만, 범례 2항목 |
| Fig 2 | 3행 유지, 내용 축소. a 설계: 단면 모식도 + 누적 TDD 곡선(2013, 다른 해 옅게, 평균 파선). 표 제거. b 모델: Stefan 앵커 산점 + 흐름(입력 썸네일 → 상자 → 지도 → ⊕ → ALT 지도). Ploton Fig 6 문법. c 검증과 산출: 검증 설계 지도 5장 + 연도별 편차 지도 2장. 제거: 셀-연도 표, 'Schematic' 곡선, 구조 글리프, 같은 셀 점그림, 행 머리글 · 헤어라인. 높이 160 → 약 125 mm |
| Fig 3 | 양식만. 격자 제거, 범례 a 패널 안, d 셀 숫자 7 pt |
| Fig 4 | 양식만. 범례 a 첫 패널 안 |
| Fig 5 | 양식만. b 범례 패널 안 |
| Fig 6 | 양식만. 해안선 0.6 pt, 통계 글자 검정 |
| Fig 7 | 양식만 |
| S1–S10 | 양식, 파일명을 번호에 맞춤, 설명문 제목 문장 |
| 원고 | `en_v12` · `ko_v12` 복사, 그림 경로 v12, Fig 2 설명문과 보충 설명문 수정. 본문 수치 · 문장은 그대로 |

## 8. 작업 순서와 QA

1. `design/style_tokens_v4.json` 에 `color.paper_v12`, `src/polar/style_v5.py` 에 `POLAR_PALETTE=paper12`.
2. `scripts/4_visualization/paper_v12/`(v11 복사 뒤 수정), 출력 `outputs/figures/paper/v12/`, `v12_si/`.
3. 그림마다 스크립트 QA(글자 크기 · 색 · 겹침 · 캔버스 밖 글자) 통과, `pdffonts` 로 글꼴 확인.
4. 전 그림 접촉 시트 렌더 후 눈으로 검토(요소 수, 선 굵기, 색 대비, 화살촉).
5. 원고 v12 빌드, 쪽 배치 재확인.
6. 남은 위험: 색을 원색으로 올리면 Fig 5a 의 직접 ML(자주)과 Aalto(하늘색)처럼 같은 모양 마커의 구분은 색에 기대므로 모양을 함께 유지한다. 선 굵기 0.8 pt 는 Sci Rep 문구(1 pt)보다 가늘다.

## 9. 구현 결과 (2026-10-18 저녁)

- 양식: `design/style_tokens_v4.json` 에 `color.paper_v12` 추가, `src/polar/style_v5.py` 가 `POLAR_PALETTE=paper12` 를 읽음. 스크립트 `scripts/4_visualization/paper_v12/`(공통 모듈 `c12_common.py` · `fig34_common12.py` · `si12_common.py` · `c12_maps.py`, 격자선 함수는 무효화).
- 그림: `outputs/figures/paper/v12/` Fig1–Fig7, `v12_si/` S1–S10(파일명 = 보충 번호). 스크립트 QA(글자 크기 · 색 · 겹침 · 캔버스 밖) 17장 모두 통과, 글자 크기 7 pt 와 8 pt 두 가지, 글꼴 FreeSans 계열만(`pdffonts`).
- Fig 2: 180 × 127 mm(v11 160 mm). a 설계 + Stefan 기준값, b 모델 흐름(검정 채운 화살촉, 검정 테두리 상자, 굵은 상자 제목), c 검증 설계 5장 + 연도별 편차 지도 2장. 제거: 셀-연도 표, 'Schematic' 곡선, 구조 글리프, 같은 셀 제품 비교.
- Fig 1c: 사용 연도 막대 + 가용 기간 회색 선, 정적 입력은 점선과 'static', Aalto 는 선 + 마름모 + 'one value, 2000–2014', 범례 3항목.
- 원고: `paper/manuscript/en_v12/`(22쪽, 오류 0) · `ko_v12/`(20쪽, 오류 0), 보충 20쪽. 그림 배치는 v11 과 같음(Fig 1 p5, Fig 2 p7, Fig 3 p10, Fig 4 p12, Fig 5 p14, Fig 6 p15, Fig 7 p16, Table 1 p4). Fig 2 설명문 189단어로 교체(영 · 국문), 보충 그림 설명문 10개에 굵은 제목 문장, 보충 라벨 `fig:s1`–`fig:s10` 순서 정리.
- 납품: `deliverables/13_paper_v12_2026-10-18/`(`tools/package_paper_v12.sh`), 압축본 2.5 MB, `COMPLIANCE_v12.md`.
- 남은 항목: Fig 3 · 4 의 그림 상단 범례(패널 안 이동은 미실시), 축선 0.8 pt(Sci Rep 문구 1 pt), 저자 자리표시 · LLM 진술 결정, 국문 원고의 Fig 2 설명문 검토.

## 10. v13 (2026-10-18 밤): 사용자 2차 지적 반영

지적: (1) v12 는 선이 굵고 색이 쨍해 발표용 같고 글자가 진하기만 하고 강약이 없다. (2) 워크플로 입력 썸네일의 색을 통일한 것이 이전(변수별 색)보다 못하고, 검증 설계 5종의 차이가 직관적이지 않으며, a 패널에서 글자와 그래프가 겹친다. (3) 그림 수 한도 안에서 그림 하나가 더 많은 정보를 압축해 담아야 한다(제품 · 라벨 그림).

추가로 본 참고 자료: Meyer & Pebesma 2022 Nat Commun Fig 1(지도 + 거리 분포), Linnenbrink 2024 GMD Fig 7(소형 다중 패널 행 · 열 머리), Betti 2025(공간 추출), Obu 2019 ESR(범북극 지도 문법), Isaksen 2022 Sci Rep(표 색), Chang 2024 · Liu 2025 npj Clim Atmos Sci(산점 + 통계 표기, 숲 그림), Uxa 2026 TC(1:1 산점 + 통계), 웹 검색(2025–2026 Nat Commun · Sci Rep 영구동토 ML 논문은 새 OA 그림을 더 찾지 못함, Wei 2026 ESSD 토론판이 최신 제품 비교).

양식 v13(`color.paper_v13`, POLAR_PALETTE=paper13):
- 글자 위계: 축 라벨 · 패널 문자 · 패널 제목 검정 7 pt(패널 문자 8 pt 굵게), 눈금 #333333 6.5 pt, 보조 주석(범례 설명 · 통계 · n) #595959 6.5 pt.
- 선: 축 0.6 pt, 자료선 0.9 pt(제안 방법 1.2 pt), 보조 0.7 pt, 기준선 0.5 pt, 도식 연결선 0.7 pt + 채운 촉 1.4 mm, 상자 테두리 0.6 pt. 마커 3.4 pt(열린 마커 테두리 0.7 pt).
- 색: Okabe–Ito 색상에 채도 약 25 % 낮춤. 제안 #C9671D, Stefan #1A7AB3, 원천 Stefan #8C8C8C, 직접 ML #7F4F9F, Wei #4E9A6A, Liu #C4547A, CCI v4 #8C6D31, CCI v5 #D4AF37. 겹 색 5종 #2A7BB5 #6BB5DE #D8A24A #C98AAE #3A9E7F.

그림 변경:
- Fig 2(180 × 133 mm): 입력 썸네일 변수별 cmcrameri 색표(덱 v7 문법) + "Each input stretched to its 2nd–98th percentile" 주석. 곡선 패널 범례(2013 · 다른 해 · 평균 · TDD_obs · 전체 계절)를 축 위 2열로, Stefan 패널 범례를 축 위 오른쪽으로 옮겨 글자 · 선 겹침 제거(QA 0건). 검증 설계: 무작위 · 공간 블록 = 겹 색 5종 점(블록은 0.5° 상자 테두리도 겹 색), 연도 홀드아웃 · 블록 × 연도 = 연도 순차 색표(bilbao_r, 채점 안 된 해는 회색), 지역 홀드아웃 = 전부 시험(검정). 겹 번호 열쇠 · 연도 색 눈금 · 설명 두 줄.
- Fig 4(180 × 153 mm): a · b 유지 + c 외부 CALM 소지역 지도(번호 1–8, 학습 지역 라벨 셀), d 외부 합동 곡선(옛 c), e 소지역별 0 → 5 라벨 덤벨 + ESA CCI v5(옛 보충 S5c). 보충 S5 는 그대로 둠.
- Fig 5(180 × 144 mm): a 유지 + b 레나 델타 공통 셀-연도(461) 관측 대 예측 산점 3장(제안 · Wei · Liu, 1:1 선, RMSE · 편향 · r) + c 관측 ALT 구간(< 30, 30–45, 45–60, ≥ 60 cm)별 평균 편향(알래스카 · 레나, 제품 4종 + 제안, 구간별 n) + d 연도별 RMSE(옛 b). 축 밖 값(ESA CCI v5 알래스카 ≥ 60 cm +77.4)은 가장자리 표식 + 숫자.
- Fig 1 · 3 · 6 · 7, 보충 S1–S10: 양식만.

원고 v13: `paper/manuscript/en_v13/`(23쪽) · `ko_v13/`(21쪽). Fig 2 · 4 · 5 설명문 교체(영 · 국문), 결과 절 '제품 비교'에 산점 · 구간 편향 문단 추가(매크로 `shared_v13/numbers_v13.tex`, `scripts/2_evaluation/v13_numbers.py`: r 0.73 / 0.05 / 0.07, 편향 +2.8 / +21.9 / +19.8 cm, 30 cm 미만 구간 과대 추정 레나 +11.8 / +33.2 / +30.9, 알래스카 +8.5 / +19.5 / +25.4 cm). 본문 4,056단어(한도 4,500), 설명문 최대 220단어. 배치: Fig 1 p5, 2 p7, 3 p10, 4 p12, 5 p14, 6 p15, 7 p17, Table 1 p4. QA: 그림 17장 통과, 글꼴 FreeSans 계열만. 납품 `deliverables/14_paper_v13_2026-10-18/`(영문 압축본 2.71 MB, 국문 2.81 MB; 압축본만 래스터 220 dpi).
남은 것: 사용자 검토, 저자 자리표시 · LLM 진술, 커밋.

## 11. v13.1 (2026-10-20): 사용자 3차 지적 6항 반영

지적과 조치:
1. Fig 5: b 의 RMSE 글자가 산점과 겹침 → 통계(RMSE · r, 편향)를 패널 제목 아래 두 줄로 올림. a 와 b · c 사이 간격 → b · c 를 4 mm 내림. b · c · d 가 같은 계열을 쓰므로 범례를 하나로 통합해 c 아래(b–d 사이)에 두고 d 의 범례를 없앰. 180 × 150 mm.
2. Fig 4: e 의 패널 문자 · 범례가 d 에 붙음 → d 의 제품 수평선 라벨을 패널 안(선의 오른쪽 끝)으로 옮기고, e 를 3 mm 오른쪽으로, 문자 e 와 범례(0 · 5 labels · ESA CCI v5)를 e 의 눈금 라벨 위로 옮김.
3. Fig 2: 수식 전부 제거(식 (3)–(5) 는 본문 수식). 상자 안 수식 → 'Stefan model', 'Gradient boosting'(굵게), λ → 'weight 0.5', Stefan 패널 범례의 식 · E 값 → 'Stefan anchor (least squares)', 축 라벨 '√TDD to observation day'. a 행 상단이 빡빡해 범례가 잘림 → 행 a 를 3 mm 내리고 곡선 · 산점 축 높이를 22 mm 로 줄여 범례(축 위 2열)에 여유를 줌. 180 × 132 mm.
4. 보충 그림 범례 10장 점검: S5 의 계열 범례(6항목 2열)가 b · c 와 어긋나 있어 b 의 라벨 열(x 62 mm)에 맞춰 정렬하고 라벨을 축약, 지도 범례는 지도 틀에 맞춤. S1–S4 · S6–S10 은 범례가 해당 패널 안 · 위에 있어 유지.
5. 캐나다 지도가 없는 이유: 연도별 1 km 지도는 공변량 격자(DEM 지형 6 · SoilGrids 9 · ERA5-Land 연별 기후)가 있는 지역에서만 만들 수 있고, 격자는 알래스카(7월, `build_map_grid_alaska_v1.py`, 899,633셀)와 레나 델타(09-30, `build_map_grid_lena_v1.py`, 53,011셀)만 만들었다. 캐나다 라벨 범위(60.4–69.4°N, 113.0–140.8°W)는 알래스카보다 넓어(약 1.5 M 셀) DEM 1° 타일 약 210개(현재 79/290 보유, 약 10 GB 내려받기)와 SoilGrids 250 m 창 100여 개(현재 라벨 주변 2° 창만 보유)가 더 필요하다. ERA5-Land 월자료(`nh_monthly_2010-2024.nc`, 25–84°N 전역)는 있다. 서버 디스크 여유가 49 GB(13 TB 중, 공유)라 이번 세션에서는 내려받지 않았다. 만들려면 약 3–4시간(내려받기 1–2시간, 지형 특징 10분, SoilGrids 창, `v5_map_predict.py --region Canada` 추가)과 디스크 약 12 GB 가 든다.
6. Fig 7 재설계(지도 다각화): a 알래스카 평균 지도 + 50 km 확대창 2(툴릭 · 페어뱅크스, 1 km 셀 그대로), b 레나 델타 평균 지도 + 24 km 확대창(사모일로프 · 쿠룽나흐), c 연도별 지도의 2011–2024 선형 추세(셀별 최소제곱, 10년당 cm, broc 발산 색표, 모델 응답이며 관측 추세 아님), d 알래스카 연도별 편차 14장(7 × 2), e 지역 평균 편차 시계열(세로 배치). 추세 수치는 매크로(`\ntrendAk`, `\ntrendLe`, 양의 추세 셀 비율)로 결과 절에 한 문장 추가. Hjort 2018 Fig 3(본 지도 + 모서리 확대), Langer 2024 Fig 2(지도 + 통계 산점), Biskaborn 2019 Fig 2(두 변수 두 기호)의 문법을 따름. Fig 6 은 유지(제품 지도 비교 + 관측 셀 오차).

글씨 · 색 · 아이콘 점검: 그림 안 글자는 FreeSans(Helvetica 계열) 7 / 6.5 pt 두 단계, 패널 문자 8 pt 굵게, 상자 제목만 굵게. 아이콘 · 픽토그램은 쓰지 않는다. 색은 paper_v13(채도 낮춘 Okabe–Ito), 지도 색표는 oslo_r(ALT) · vik(편차) · broc(추세) · 변수별 cmcrameri(입력). 회색 글자는 보조 주석에만.

결과(10-20 12:15): Fig 7 은 180 × 184 mm(a · b 행 54 mm, c · e 행 45 mm, d 행 2 × 22 mm), QA 통과. 추세(식: 셀별 최소제곱, 10년당 cm): 알래스카 평균 +2.7(5–95 백분위 +0.3 ~ +7.4, 양의 셀 98 %), 레나 델타 +6.4(+4.5 ~ +9.5, 100 %). 원고 결과 절 '연도별 지도' 문단에 이 수치를 매크로(`\ntrendAk` 등, `numbers_v13.tex`)로 한 문장 추가. 원고 v13.1: 영문 23쪽 · 국문 21쪽 · 보충 20쪽, 본문 4,115단어, 설명문 최대 221단어, 배치 Fig 1 p5 · 2 p7 · 3 p10 · 4 p12 · 5 p14 · 6 p15 · 7 p17 · Table 1 p4. 납품 `deliverables/14_paper_v13_2026-10-18/` 갱신(압축본 래스터 200 dpi: 영문 2.89 MB, 국문 3.00 MB).

## 12. v13.2 (2026-10-20 오후): 사용자 4차 지적 8항과 변인 통제 감사 반영

지적과 조치:
1. Fig 4: a–b 와 c–e 를 같은 너비의 3열 격자(열 너비 (180 − 11 − 2 × 4) / 3 mm)로 정렬하고, e 의 패널 문자를 e 의 눈금 라벨에서 6 mm 떼었다. 180 × 158 mm.
2. Fig 2: a 의 범례를 축 위 한 줄(세 항목)로 줄이고 곡선 라벨을 축 안 직접 라벨로 바꿔 TDD 패널로 넘어가지 않게 하였다. 지도 입력(9월 1일까지의 TDD)을 채운 표지로 나타내었다.
3. Fig 5: a 의 '68/118' 같은 분수(축 밖 값 표기)를 없애고 축을 끊어(10–45 · 45–130 cm) 그렸다. b 는 제목과 r 만 남겼다(RMSE · 편향은 본문). b–d 범례는 한 줄(c 와 d 사이).
4. Fig 7: 행 순서를 알파벳대로(a·b 평균 지도, c 캐나다 평균 지도, d 지역 평균 편차, e 추세 지도, f 알래스카 연도별 편차 지도) 바꾸고, a–c 가 한 색 범위와 세로 색 막대(1행 오른쪽 여백)를 쓰게 하였다. 행 간격을 넓혔다(가로 180 × 세로 205.6 mm).
5. 보충 그림: 본문과 같은 모듈(c13_common · c13_maps · si13_common)로 색 · 글꼴을 공유한다. S5 범례 정렬, S8 범례 위치 수정.
6. S9 b(레나 델타) 의 '지도 밖으로 삐져나온 셀': Natural Earth 해안선 다각형이 격자 육지 셀의 22 % 를 비켜 가서(알래스카 0.1 %) 생긴 것. 해안선을 격자의 land 열(가우시안 평활 뒤 0.5 등고선)로 그리도록 바꿨다(`c13_maps.coast`, 레나 델타만; 격자 밖은 Natural Earth 육지로 채워 영역 경계선이 생기지 않게 함).
7. 캐나다 지도 제작: DEM 타일 290개(3.95 GB) 내려받기 → `build_map_grid_canada_v1.py`(격자 1,223,119 셀, 60–70°N · 141–112.5°W, PFR ≥ 50 %) → 연별 강제력(`build_v5_forcing_year.py`, 18.3 M 행) · 음영(EPSG:3978, 250 m) → `v5_map_predict.py --region Canada --ref-doy 244`. 확대창: 올드크로 평원(−139.7, 67.9), 매켄지 델타 이누빅(−133.5, 68.3).
8. 변인 통제 감사(에이전트 3건: `docs/AUDIT_FIGURE_CONTROL_2026-10-20.md`, `AUDIT_TEXT_NUMBERS_2026-10-20.md`, `AUDIT_EXPERIMENT_SETTINGS_2026-10-20.md`)에서 나온 결정:
   - 라벨 단위: loc_id 는 좌표를 0.0001° 로 반올림한 지점(약 10 m)이지 1 km 셀이 아니다. 본문 용어를 '지점-연도(site-year)' 로 바꾸고, 1 km 셀 수(알래스카 321 · 레나 143 · 캐나다 71, 셀-연도 910 · 244 · 144)를 표 1 과 방법 절에 병기했다. 1 km 로 재집계해 전부 다시 돌리는 선택지는 결과가 바뀌는 큰 작업이라 사용자 결정으로 남긴다.
   - 지도 입력: 전체 계절 TDD(지도) 대 관측일 TDD(학습)의 공변량 이동을 없애기 위해 지도를 9월 1일(244일)까지의 TDD 로 다시 만들었다(관측일 중앙값 224 · 234 · 237). 격자 강제력에 tdd_cum 열을 추가해 재생성(레나 3분, 알래스카 30분), `v5_map_predict.py --ref-doy 244`. 옛 지도는 `data/processed/v5/maps_fullseason_2026-10-14/`.
   - Fig 6: 알래스카 2014 · 레나 2016 대신 2013 으로 통일(v14 정정: 세 지역 모두 관측이 있는 해는 12개이고 2013 은 그중 알래스카 · 캐나다 관측 셀이 가장 많은 해; 세 지도 모두 값이 있는 1 km 셀: 알래스카 126, 레나 6, 캐나다 33). 아홉 지도가 한 색 범위(우리 세 지도 합산 1–99 백분위수, 10 cm 단위), 오차 축 ±60 cm 공통. 2013 RMSE(겹 밖 예측 / Wei / Liu): 알래스카 14.9 / 13.4 / 12.9 cm(126 셀), 레나 16.6 / 32.3 / 29.6 cm(6 셀, 편향 +0.2 / +19.8 / +20.7), 캐나다 23.7 / 29.9 / 20.2 cm(33 셀, 편향 −4.0 / +20.5 / +7.8; Liu 는 캐나다 관측의 이전 판으로 학습). 공통 색 범위 40–90 cm, 캔버스 180 × 148.5 mm, QA 통과. 산지 차이(68–70°N, 600 m 이상): Wei +18.7, Liu +52.9 cm(2014: +21.9 / +55.2).
   - Fig 7 · S8 · S9: 세 지역 같은 양(2011–2024 평균 · 추세 · 보정량), S8 은 창 4개(툴릭 · 페어뱅크스 · 사모일로프 · 매켄지 델타) + 적용 가능 영역 3장으로, S9 는 세 지역 평균 보정 지도로 재구성.
   - 캡션 보완(감사 목록 P0-10 · P1-2 · P1-5 · P1-7): S5 b 의 in_range 부분집합과 c 가 Fig 4e 와 같다는 사실, Fig 7 d 의 y 범위 상이, Fig 3c 실행 표 참조, S10 별도 3시드 적합, S1c 축 밖 46/2,161 지점-연도, 방법 절 첫 언급에 'Canada (NW)' 표기.
   - 매크로: Fig 6 수치는 `v13_numbers.py` 가 v13 JSON 에서 `\providecommand + \renewcommand` 로 다시 정의(numbers_v13 이 마지막에 읽히므로 v9 · v11 값을 덮어씀). 지도 매크로(`v10_numbers.py` MAP_REGIONS, `v6_numbers.py`)는 새 지도로 재생성.
   - 캐나다 격자 QA(빌드 로그 `data/processed/map_canada/build_grid_v1.log`): 격자 1,223,119 셀 중 DEM 유효 100 %, ERA5-Land 100 %, CCI 1,184,890, SoilGrids 9층 1,195,599(기존 namerica 창 + 새 창 map_canada_v1_0 9,674 셀). 라벨 셀 742개 대조: 27열 중 21열 통과, DEM 6열은 11 % 셀에서 허용 오차 초과(고도 최대 2.4 m, 경사 3.8°; 알래스카 격자와 같은 종류의 창 정렬 차이, 지도 사용에 영향 없음).
   - 캐나다 격자 merge 단계는 PFR 일치 단언(allclose 기본 허용)에서 1셀(99.27 대 99.81 %p, 타일 경계)로 멈춰, 허용을 1 %p 로 풀고(`build_map_grid_canada_v1.py`, 차이 셀 수 · 최대 차이를 로그에 남김) merge 만 다시 실행했다(x25 1,223,119 × 57, 회색 71,950). 지도 예측은 14:25 시작.
   - 9월 1일 TDD 지도 메타(`data/processed/v5/maps/<Region>/meta.json`, ref_doy 244): 적용 가능 영역 안 셀 비율 알래스카 56.2 %, 레나 62.2 %, 캐나다 42.9 %; 90 % 예측 구간 반폭 25.5 / 24.5 / 41.3 cm(캐나다 라벨 1,196 지점-연도, 블록 CV 잔차 5–95 백분위 −35.1 / +47.5).
   - Fig 7 최종(180 × 205.6 mm, QA 통과): a–c 공통 색 범위 40–85 cm(세로 색 막대, 1행 오른쪽), 지도 셀 858,517 / 38,086 / 1,151,169, 관측 1 km 셀 321 / 143 / 71. e 추세(10년당, 9월 1일 TDD 지도): 알래스카 +2.1 cm(양의 셀 78 %), 레나 +6.7(100 %), 캐나다 +1.5(91 %); 전체 계절 지도의 +2.7 / +6.4 를 대체. d 관측 대 예측 편차 상관: 알래스카 0.13(12해), 레나 0.73(10해), 캐나다 0.64(7해; 2023 · 2024 관측 편차 +27 · +26 cm, 각 80 지점-연도). 연속 연도 라벨은 한 라벨로 합침("n = 80, 80").
   - 첫 제출용 단일 파일(Sci Rep 3 MB 한도, 지침 확인): 벡터 원고(19.6 MB)를 Ghostscript 로 래스터만 줄여도 3.4 MB 에 머물렀다(원인은 지도 패널의 벡터 경로: 해안선 · 경위선 · 레나 윤곽 등 'l' 연산 6–12만 개/그림). 경위선 점 수를 400 → 120 으로 줄이고(`c13_maps.graticule`, 범위 ±12° / ±6° 유지) 작은 패널의 해안선을 단순화(`coast(tol_m)`: Fig 6 600 m, Fig 7 평균 500 · 추세 800 · 연도별 2,000 m)했으나 크기 변화가 작아, 압축본은 그림을 600 dpi PNG 로 넣어 220 dpi 로 낮추는 별도 빌드(`tools/build_submission_png.sh`, main_png.tex)로 만든다: 영문 0.98 MB · 국문 1.09 MB, 24 · 22쪽. 벡터 원고 · 그림 PDF 는 그대로 납품(점검표 통과).

## 13. v14 (2026-10-20 저녁): 사용자 지시 12항(그림 대안 · 지도 비교 강조 · 본문 개정 · 감사) 반영

정본: 스크립트 `scripts/4_visualization/paper_v14/`, 그림 `outputs/figures/paper/v14/` · `v14_si/`, 대안 `v14_proposals/`, 원고 `paper/manuscript/{en,ko}_v14/`, 매크로 `shared_v14/numbers_v14.tex`(`scripts/2_evaluation/v14_numbers.py`), 납품 `deliverables/15_paper_v14_2026-10-20/`(`tools/package_paper_v14.sh`, 압축본 `tools/build_submission_png.sh` VER=v14), 마무리 체인 `tools/finish_paper_v14.sh`.

지시와 조치:
1. 지도의 참값 비교 강조 · 눈금: Fig 6 은 관측 셀을 관측 ALT 색으로 채운 원으로 지도 위에 그리고, 제품 지도를 우리 지도의 셀에 마스크하며(같은 영역), 지역별 색 범위를 적용 가능 영역 안 세 지도 합산 5–95 백분위수로 정하고(산지 극값 제외), 오차 벌떼 대신 관측 대 예측 산점(1:1 선, r · RMSE)을 두었다. Fig 7 a–c 와 확대창의 관측 셀도 관측 평균 ALT 로 채웠다.
2. 그림 대안: `docs/FIGURE_PROPOSALS_v14_2026-10-20.md`. 유형 4종(정확도 비교 · 라벨 사다리 · 제품 비교 · 지도) × 3안, 실제 자료로 렌더한 대안 4장(`v14_proposals/`). v14 에는 Fig 3 A(입력 효과 숲 그림, 설계별 점 + Δ 띠), Fig 6 · 7 A 를 적용.
3. 본문: 고찰을 소제목 없이 6문단으로 재구성(Sci Rep 권장 구성; 선행 연구 비교를 본문 흐름에 통합), 결론을 이 연구 고유의 결과 세 가지와 산출물로 한정, 서론의 공백 · 기여 문장을 수사 없이 정리, 결과의 반복 표현 축약, 해석 문장은 고찰로 이동. 새 지역 적용 절차는 "고정 절차가 아니라 기대할 수 있는 바" 로 바꾸고 캐나다의 재보정 악화 사례와 두 가지 점검(지역 안 블록 평가, 적용 가능 영역)을 명시.
4. 수식 추가(총 13개): 세 대안 구조(식 6), 전이 기준값(식 9), 편차 · 추세(식 12), 예측 구간(식 13). 캡션의 방법 세부는 방법 절로 옮김.
5. 캡션: 본문 7장 70–137단어(이전 86–267). 제목 문장 + 패널 설명 + 구간 정의만.
6. 감사(에이전트, `docs/AUDIT_DATAFLOW_NUMBERS_v14_2026-10-20.md`): P0 2건(2013 "유일한 해" 오류 → "세 지역 모두 관측이 있는 해 가운데 알래스카 · 캐나다 셀이 가장 많은 해"; Fig 2 캡션의 전체 계절 문구), P1 5건(보충 PDF 재빌드, Fig 3 캡션 20/10 문구 제거, S5 in_range 정의(ALT ≤ 300 cm), S6c 를 v14 Fig 6 JSON(2013 · 9월 1일 지도)로, S10 계절 말 선을 9월 1일 TDD 로 재계산(`v5_site_series.py --ref-doy 244`)), P2 9건(B 혼재 문구, IQR 224–234 · 12–37 %, 244일 = 평년 9월 1일, 전이 n=0 씨앗 1, ALLena 라벨 · 관측일 정의, 자료 가용성에 캐나다 지도, Yi–Kimball '미채점', 보충이 numbers_v14 읽음, 문서 문자열). 자료 흐름(연도 · 관측일 정합, 겹 구조, 지도 ref_doy 244, 제품 관측 연도)과 매크로 263개의 원천 일치는 감사가 확인함.
7. 국문 쌍둥이판 동기화(서론 · 결과 · 고찰 · 결론 전면, 방법 수식 · 문장, 캡션, 초록, 자료 가용성, 표 1).

## 14. v14.1 (2026-10-21): Fig 6 지역별 연도 · 학습 지점 거리 구분, 보충 표 S7(셀-연도 재채점), 저자 표기, 납품 폴더 재구성

사용자 7항(10-21): 셀-연도 재채점 보충 표, Fig 6 연도 기준, 평균 지도 관행 확인, 제안 보류 · 기각, 저자 · 소속, 그림 2차 제안, 폴더 정리.

1. **Fig 6**: 지역별로 '세 지도 모두 값이 있는 관측 1 km 셀이 가장 많은 해'를 쓴다(알래스카 2013: 126셀, 레나 델타 2016: 14셀, 캐나다 2013: 33셀; `fig6_product_maps.py` `YEARS`, JSON `why_this_year`). 산점도 d · h · l 은 Wei 학습 지점 25 km 안(채운 원) · 밖(빈 원)을 구분하고 'RMSE: all | >25 km' 두 값을 적는다(레나는 전부 밖이라 한 값). 통계 글은 표지와 겹침이 적은 모서리(왼쪽 위 또는 오른쪽 아래)를 렌더러 좌표로 골라 둔다(JSON `stats_text_corner`, `stats_text_marker_hits`: 알래스카 tl 1, 레나 br 0, 캐나다 tl 3). 범례 2줄. 본문 · 캡션(영 · 국문), 보충 표 S2 행, 매크로 `\nfe{Ak,Le,Ca}Far{Oof,Wei,Liu}`, `\nfe*FarR*`, `\nfe*Near*` 추가(`v14_numbers.py`).
2. **알래스카 2013 상관 차이의 해석**: 전체 126셀에서 r 0.34 대 0.53 · 0.57 이지만 59셀이 Wei 학습 지점 25 km 안이고, 밖 67셀에서는 RMSE 9.1 · 10.9 · 8.5 cm, r 0.30 · 0.27 · 0.44 로 같다. 연도별(2010–2020)로도 알래스카는 세 지도가 동급이다(스크래치 `fig6_yearwise_common_cells.csv`; 본문에는 넣지 않음).
3. **보충 표 S7**(`scripts/2_evaluation/v14_cellyear_rescore.py`): 기존 겹 밖 예측을 1 km 셀-연도 평균으로 다시 채점. (a) 세 설계 × 네 방법의 지점-연도 / 셀-연도 RMSE, (b) 쌍 차이와 95 % 블록 부트스트랩 구간(B 2000). 레나 · 캐나다 블록 CV 이득과 레나 전이 벌점은 셀 단위에서도 유지되고, 캐나다 지역 홀드아웃의 물리 잔차 이득(−3.5)만 셀 단위에서 미결(−0.2 [−2.2, +1.6])이다. 셀 안 관측 산포(합동 SD 알래스카 13.0, 레나 12.7, 캐나다 17.5 cm)를 캡션에 적었다. 방법 절 한 문장과 보충 표 수 S1–S7 갱신. 전면 재실행(라벨 재집계)은 하지 않음(사용자 판단: 채점 단위의 차이).
4. **저자 · 소속**: 백승원(1저자, 서울대 에너지시스템공학부), 김승희 · 정윤택(극지연구소 Center of Remote Sensing and GIS; 국문 원격탐사빙권정보센터), 조용채(교신). 기여 문장 초안. 교신 전자우편 · 감사의 글 · 저장소 DOI 는 자리표시. LLM 진술 없음.
5. **지도 관행 조사**(`docs/research/2026-10-21/map_and_figure_practice.md`): 주 지도는 다년 평균(8편), 추세는 별도 그림, 연도별 편차 지도는 선례 없음 → Fig 7f 를 보충으로 옮기는 안을 2차 제안서에 기록(미적용).
6. **그림 2차 제안**(`docs/FIGURE_PROPOSALS_v14_1_2026-10-21.md`, 렌더 8장 `v14_proposals/P*`): Fig 3 D · E · F, Fig 4 E · F, Fig 5 D · F, Fig 7 d 대안. 1차 판정(사양 곡선 기각, 표적 도식 보류) 기록.
7. **납품 · 폴더**: `deliverables/` 를 종류별(paper, figure_proposals, deck, reports, references, archive)로 재구성(`tools/sync_deliverables.sh` v2, 날짜별 묶음은 archive/). 루트 `00_START_HERE.md` 에 폴더 역할 표(results = Rescale 원시 결과, outputs = 생성 산출물, figures = 명세 JSON 등). 마무리 체인 `tools/finish_paper_v14_1.sh`.
8. **결과**: Fig 6 QA 통과, 영문 25쪽 · 국문 22쪽 · 보충 21쪽 오류 0, 점검표 전 항목 통과(본문 4,442단어, 캡션 70–164단어, 압축본 1.0 MB).
