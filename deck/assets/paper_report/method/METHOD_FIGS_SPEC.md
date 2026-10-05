# 방법 도식 명세 (국문 슬라이드판 v4)

작성 2026-10-05. 대상 덱: `deck/render/permafrost_paper_report.pdf`(결과 중심 56쪽)에 넣을 방법 도식 9종.
생성기: `scripts/4_visualization/method_figs_v4.py`(그림마다 함수 하나). 산출: 이 폴더의 `M#_<name>_slide.png`(300 dpi)와 `.pdf`.
덱 배치는 다른 작업자가 한다. 이 폴더는 그림만 만든다.

## 0. 공통 규격

| 항목 | 값 | 근거 |
|---|---|---|
| 크기 | 전폭 12.0 × 5.2 in 이하, 반폭 5.8 × 5.2 in 이하, 300 dpi | 덱 본문 영역(x 0.667, y 1.3, w 12.0, H 5.2) |
| 글꼴 | 라틴 FreeSans Regular, 한글 Pretendard Regular(글리프 대체). 굵게는 그림 안 열 머리에만 | `design/style_tokens_v4.json` font |
| 글자 크기 | 열 머리 15 pt, 본문 13–14 pt, 보조 12 pt. 12 pt 미만 없음 | 작업 지시(12 pt 이상), 토큰 slide |
| 색 | 방법 색은 토큰 그대로(재보정 Stefan #1F5A99, 원천 계수 Stefan #8A9BB0, 물리 잔차 결합 #D55E00 강조, 직접 ML #A3478A, 물리 유사라벨 #009E73, 물리 입력 ML #E69F00, 위약 #9E9E9E). 지역 색은 tokens.regions | 토큰 color |
| 공정 채움 | 학습·신경망 블록 #F7E9AE, 물리·평가 셰브런 #DCEBDD | 토큰 color.process |
| 열 머리 | 덱 제목 주황 #EA851B, Pretendard Bold. 자료 표현에는 쓰지 않는다 | 대회 덱 v5 방법 쪽(사용자 승인), 토큰 deck |
| 상자 | 실자료 그림(자연 테두리 0.6 pt #BDBDBD), 공정 셰브런, 신경망·학습기 블록만 | 대회 덱 v5 문법 |
| 화살표 | 수평·수직 직선만. 학습 때만 쓰는 경로는 붉은 파선(#C0392B). 화살촉은 흐름에만 | 대회 덱 v5 문법, 토큰 symbols |
| 금지 | 바깥 테두리와 제목 띠, 색 구역 판, 회색 바탕 판, 내용 카드, 판정 상자, 곡선·사선 화살표, 주황 번호 막대, 균일 상자 격자, 긴 파선 우회, 통계 카드, 아이콘, 이모지 | 작업 지시 |
| 수치 | 아래 '자료 출처'의 기록 문서에 있는 값만. 내부 코드(P0, R1, 규칙 W, XC 등)는 보이는 글에 쓰지 않는다 | 작업 지시, `deck/DECK_PLAN_paper_report.md` 2절 |
| 개념 그림 | 숫자 눈금 없는 도식이거나 실자료 | 작업 지시 |

실자료 축소 그림(썸네일)
- 범북극 라벨 지도: `data/processed/paper_figs/fig1_blocks.csv`(0.5° 블록별 1 km 위치 수, Fig 1a 와 같은 자료), Natural Earth 50 m 육지, 북극 평사 투영.
- 레나델타 지역 홀드아웃·블록 분할·라벨 40개: `scripts/4_visualization/paper_v3/fig1.py` 의 `lena_design`(Fig 1c·d 와 같은 분할 1, 추출 1).
- Stefan 산점도: `scripts/4_visualization/paper_v3/fig3.py` 의 `load_concept`(Fig 3a 와 같은 러시아 서부 31셀, 원천 계수 1.59, 라벨 10개 재보정 2.11).
- 알래스카 1 km ALT 지도: `deck/assets/paper_report/slide_panels/Alaska_ALT_map_v3_c_slide.png` 의 지도 부분(컬러바 제외).
- 위약 대비 값: `data/processed/paper_figs/fig3_a.csv`(Fig 3b 원천).

## 1. 그림별 명세

### M1_problem (12.0 × 5.2 in)
- 메시지: 라벨이 많은 원천 지역에서 배운 모형을 라벨이 적은 대상 지역에 쓸 때, 대상 라벨 n개에 따라 어떤 방법을 쓸지가 문제다.
- 내용: 왼쪽 범북극 지도(원천 셀 채운 원, 대상 레나델타 빈 원과 100 km 완충 표시). 오른쪽 라벨 수 축(0, 3, 10, 40, 160, 전량)과 그 위 후보 방법 4계열 행(원천 계수 Stefan, 재보정 Stefan, 물리 잔차 결합, 직접 ML), 질문 '라벨 n개일 때 어떤 방법을 쓰는가'. 성능 곡선 없음.
- 수치: 100 km(methods.tex), 원천 셀의 알래스카 비율 78–94 %(Table 1, D README E35), 라벨 수 격자(methods.tex).

### M2_gap (12.0 × 5.2 in)
- 메시지: 선행 연구의 세 한계를 평가 설계로 고쳐 새 지역 오차, 라벨 투자 판단, ML 순이득을 분리한다.
- 내용: 열 3개(선행 연구의 한계 → 본 연구의 개선 → 기대 효과) × 행 3개(평가 지역, 라벨 수, 물리 기준선), 헤어라인 구분, 상자 없음.
- 출처: 덱 2쪽 표(deck_spec S02 rows), `docs/NOVELTY_POSITIONING_2026-09-29.md` 3절, `docs/RESEARCH_OVERVIEW_2026-10-02.md` 7–8절, `paper/manuscript/en/sections/intro.tex` 2문단.

### M3_data (12.0 × 5.2 in)
- 메시지: 라벨은 7개 지역에 30–13,606행으로 치우쳐 있고, 입력은 1 km 셀에 맞췄으나 정보 해상도는 5 km–0.1° 이다.
- a: 범북극 지도, 지역 색(tokens.regions), 지역별 라벨 행 수(Table 1).
- b: 헤어라인 표(지역, 라벨 출처, 라벨 행, 1 km 위치, 0.5° 블록), 각주 라벨 연도 1990–2024(methods.tex).
- c: 입력 해상도 점 그림(log m 축): 라벨 점 → 1 km, 지형 DEM 30 m, 지형 창 약 1 km, 기후 ERA5-Land 0.1°, 토양 SoilGrids 250 m → 약 5 km, CCI 약 1 km, MODIS 250–500 m(지역 내 추가 시험 2단계). 1 km 모형 셀 세로선, 정보 해상도 띠(약 5 km–0.1°).
- 수치 출처: Table1_data.tex, methods.tex(공변량 25종, 해상도), deck_spec S06 values, EXPERIMENT_PLAN_FINAL_BATCH 2.5(MODIS 군).

### M4_workflow (12.0 × 5.2 in)  [우선 1]
- 메시지: 연구는 자료 구축부터 지도·제품 비교까지 여섯 단계이고, 물리 기준선과 같은 시험지 위에서 ML 결합 방식을 평가한다.
- 내용: 셰브런 6개(1 자료 구축, 2 물리 기준선, 3 ML 결합 방식, 4 평가 설계, 5 라벨 수별 워크플로, 6 1 km 지도와 제품 비교). 노랑 = 학습(3, 5), 초록 = 물리·평가(1, 2, 4, 6). 각 단계 아래 실자료 축소 그림 1개와 2–3줄 명사구.
- 축소 그림: 1 범북극 라벨 지도, 2 Stefan 산점도(러시아 서부), 3 결합 구조 4종의 색 표지 목록(도식), 4 레나델타 홀드아웃·블록 분할, 5 라벨 수 축 위 방법 구간(도식, 숫자 눈금은 라벨 수만), 6 알래스카 1 km ALT 지도.
- 수치: 공변량 25종, 1 km, 100 km, κ = 10, 라벨 수 격자(methods.tex).

### M5_augmentation_design (12.0 × 5.2 in)  [우선 3]
- 메시지: 물리 유사라벨과 위약 4종은 셀·행 수·난수가 같고 값만 다르며, 같은 CatBoost 로 학습해 같은 채점 블록에서 비교한다.
- 내용: 왼쪽 대상 라벨 절반 셀(레나델타 실자료) → 조건 5행(물리 유사라벨 E_n·√TDD, 순서를 섞은 유사라벨, 풀 평균 상수, 원천 평균 상수, 선형 융해 지수) → 학습 집합(원천 실측 + 대상 라벨 n개 + 유사라벨, 원천 행당 10개) → CatBoost 블록 → 채점 블록 RMSE 셰브런 → 오른쪽 위약 대비 Δ(라벨 0개, 10개; 셀 가중·블록 등가중 95 % CI, ±0.5 cm 띠). Δ 행은 위약 행과 같은 높이.
- 출처: methods.tex(유사라벨, 위약 4종 정의), `scripts/3_deep_learning/h40_label_grid.py` 24행·860–890행(r = 10, 라벨 셀은 실측), Fig3_legend.md, fig3_a.csv.

### M6_models (12.0 × 5.2 in)  [우선 2]
- 메시지: 비교한 방법은 같은 입력과 학습기를 쓰고, 물리 정보가 들어가는 위치(없음, 학습 라벨, 입력, 앵커)만 다르다.
- 내용: 왼쪽 입력 X(지형 6, 기후 8, 토양 9, CCI 2 = 25) 막대 → 공용 분배선 → 방법 5행. 행마다 입력 → 학습기 블록(노랑) → 예측 식, 학습 목표는 블록 아래에서 붉은 파선으로 들어간다(학습 때만). 물리 행은 초록 계수 추정 블록과 E0, E_n(κ = 10) 식. 오른쪽 학습기 블록 열(CatBoost 3종, 랜덤 포레스트, TabPFN v2, TabICL v2, MLP, 다중 헤드 MLP, FT-Transformer 축소형, RealMLP).
- 식: a = E·√TDD, E0 = Σ s y / Σ s², E_n = (n E_ls + κ E0)/(n + κ), κ = 10, ŷ = a + λ g(X), λ ∈ {0.25, 0.5, 1.0}(methods.tex 식 1–4). CatBoost 200회, 학습률 0.05, 깊이 3(methods.tex).

### M7_evaluation (12.0 × 5.2 in)  [우선 4]
- 메시지: 대상 지역을 100 km 완충과 함께 빼고, 블록 절반에서 뽑은 라벨 n개로 학습해 나머지 절반에서 채점하며, 판정은 두 가중 CI 와 ±0.5 cm 로 정한다.
- a: 레나델타 지역 홀드아웃과 100 km 완충(실자료). b: 0.5° 블록의 라벨 블록·채점 블록과 라벨 40개(실자료). c: 라벨 수 격자 0, 3, 10, 40, 160, 320, 1000, 전량, 분할 5회·추출 5회, 러시아 서부·동부는 10개까지. d: 판정 규칙 도식(오차 감소, 오차 증가, 동등, 미결정), 셀 가중(굵게)·블록 등가중(가늘게), 블록 재표집 10,000회. 아래 가는 시간선(등록 → 실행 → 재현 관문 → 열람).
- 출처: methods.tex Transfer design, Statistical analysis, Internal pre-registration, Computing environments.

### M8_label_workflow (12.0 × 5.2 in)
- 메시지: 라벨 수가 늘면 원천 계수 Stefan → 재보정과 저가중 잔차 → 대상 라벨 교차검증 선정으로 넘어가고, 출력은 1 km 지도와 90 % 구간, 학습 범위 밖 표시다.
- 내용: 가로 라벨 수 축의 결정점(0, 3–10, 40 이상), 단계별 방법·진단 행, 선정 규칙 5단계(methods.tex), 라벨 배치 행(분산 배치 효과는 지역 의존: 캐나다 이득, 레나델타 손해; 사전 지정 절차는 무작위로 확정), 출력(알래스카 1 km 지도 축소 그림, 90 % 구간 ±20.2 cm, 학습 범위 밖 공변량 셀 73 %).
- 출처: methods.tex Method-selection rule, Label placement procedures, Fig7_legend.md d, Fig5_legend.md, maps/Alaska_ALT_map_v3_legend.md.

### M9_deepsets (12.0 × 4.6 in, 선택)
- 메시지: 학습된 배치 정책은 라벨 집합을 원소 특징 29개의 집합으로 보고 효용을 순위 손실로 배운 뒤 탐욕적으로 셀을 더한다.
- 내용: 원소 특징 29 → φ MLP 128-128 → 합·평균 풀링(256) + 문맥 6 → ρ MLP 128 → 효용 1, RankNet 손실, 탐욕 선택(5개씩, 후보 farthest-first 상위 300 ∪ 무작위 200). 블록은 노랑.
- 출처: `scripts/3_deep_learning/x_placement_deepsets.py` 머리말, EXPERIMENT_PLAN_FINAL_BATCH 2.4.

## 2. 점검

렌더 뒤 PNG 를 100 % 로 보고 겹침, 잘림, 밀집, 금지 요소, 사선 화살표를 고친다. 글자 크기 하한(12 pt)과 그림 크기는 생성기의 점검 함수가 기록한다(`method_figs_qa.json`). 접촉 인화지 `contact_sheet.png`.

## 3. 작업 중 바뀐 점(2026-10-05)

- 주 양식 참조 추가: 사용자의 IMAGE 학회 덱(`image_ref_deck.pdf`, 대회 덱 문법의 원본). 반영한 것은 3쪽 배치(한계 | 본 연구의 작은 흐름과 실자료 그림 | 기대 효과)를 M2 에, 4쪽의 색 구분 모듈과 직선 연결을 M4 에(셰브런 6개 + 라벨 축적 고리), 8쪽의 자료 판과 헤어라인 표를 M3 에. 색 구역 판, 내용 카드, 곡선 화살표는 금지 목록대로 쓰지 않았다.
- M4 ① 설명 줄: '라벨 셀 17,467개'와 '7개 지역'을 나란히 두면 17,467 이 7개 지역 합으로 읽혀 '평가 지역 7곳 · 1 km 셀 라벨 · 공변량 25종'으로 바꿨다(7개 지역 행 합은 v4 새 지역 행을 포함해 17,467 과 다르다).
- M7 d: 블록 재표집 횟수는 methods.tex 대로 '1000 또는 10,000회'(라벨 수 격자 집계 가설 1000회, 그 밖 10,000회).
- M8: 사전 지정 배치 절차가 알래스카 선택 규칙에서 무작위 배치로 확정된 사실(methods.tex Label placement procedures)을 라벨 배치 행 아래에 적었다.
- 라벨 수 구간 도식(M2, M4 ⑤, M8)은 Fig 7d·M8 과 같은 세 구간(0, 3–10, 40 이상)으로 맞췄다.
- 글꼴: matplotlib PDF 가 CFF(OTF) 글꼴을 묻지 못해 Pretendard Regular·Bold 를 TrueType 으로 바꿔(`~/.cache/polar_fonts/`, 생성기가 없으면 만든다) 등록했다. font.family = [FreeSans, Pretendard] 목록으로 글리프 대체를 쓰고, 한 Text 안에 한글과 수식을 섞지 않는다(RT 함수).
- 실자료 축소 그림의 원천: 범북극 지도 `fig1_blocks.csv`, 레나델타 설계 `fig1.lena_design`, Stefan 산점도 `fig3.load_concept`, 알래스카 지도 `slide_panels/Alaska_ALT_map_v3_c_slide.png`·`maps/Alaska_ALT_map_v3_slide.png`(패널 a–d 테두리 자동 검출), 검증 설계별 오차 `Fig1_source_data.csv`(panel e, MEAN3), 위약 대비 `fig3_a.csv`, 자료 표 `table1_rows.csv`.
- 점검 결과: `method_figs_qa.json`(그림마다 글자 수, 최소 글자 크기 12.0 pt, 그림 밖 글자 0, 글자 상자 겹침 0, 화살표 수). 화살표 함수는 사선을 거부한다.
