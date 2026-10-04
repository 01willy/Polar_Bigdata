# v3 재구성 그림 명세(Fig 1–7, Table 1)

작성 2026-10-04. 성격: 본문 표시 항목 8개(그림 7, 표 1)의 v3 설계 명세다. 렌더 전 문서이며 그림 파일은 아직 없다. v2 그림(`outputs/figures/paper/v2/`)과 v2 모듈(`scripts/4_visualization/paper/`)은 고치지 않는다. v3 산출물은 이 폴더(`outputs/figures/paper/v3_restructure/`)에 두어 v2 와 나란히 비교한다(사용자 결정, 2026-10-04).

**구속 문서**: `design/journal_grade_style_guide.md`(이하 '지침', 특히 2절 그림, 3절 표, 6절 항목별 계획), `design/slide_archetypes.md`, `design/user_feedback_digest.md`, `design/baseline_audit_2026-10-04.md`. 결과 절의 번호와 순서는 지침 4.3 표를 따른다(원고 명세 `paper/manuscript/MANUSCRIPT_SPEC.md` 4절과 같다. `docs/MANUSCRIPT_RESTRUCTURE_PLAN_2026-10-02.md` 4절과 달리 C8 은 R5 에, 독립 지역은 R7 에 둔다). X 묶음의 자리는 `docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md` 6절(이하 '등록 문서')을 따른다. 주장 문장의 범위는 `paper/claims/*/README.md` 1.3절과 5.2절, `docs/QA_FINAL_REVIEW_2026-10-02.md`, `docs/QA_FOLLOWUP_2026-10-04.md` 의 좁힘을 따른다.

**표기**
- 내부 약호(P0, P*, P1, P2, R0, R1, R2, D0, D1, F1k, F1n, W, S1–S7, AB, L, WF, X)는 이 문서 안에서만 쓴다. 그림 글자와 설명문 초안에는 쓰지 않는다(지침 H3).
- [미확인]: 원천에서 확인하지 못한 값이나 경로. [판단]: 지침이 값을 주지 않아 이 문서가 정한 값. [결정 필요]: 사용자 결정 사항(14절 목록).
- X 묶음의 결과 자리는 등록 형식 `[Xn: 절 | 표지 | 결과별 문장 규칙]`(`docs/research/2026-10-04/scirep_format_and_drafts.md` 4.9절)으로 둔다. 자리표시 문자열은 원고 명세 11절의 위치별 문자열과 글자 하나까지 같다. 첫 칸은 자리표시가 놓인 위치다(설명문은 'Fig 4d' 같은 패널, 결과 절 본문은 'R3'). 결과 절 R3 의 XA 자리표시는 등록 문서 0.4절의 문구 `[XA: R3 | 사후 분석(재현(비맹검)) | 2.1 해석 조각]` 그대로다.
- 설명문 초안의 수치는 각 초안 아래 '수치 출처'에 적은 원천에서 옮겼다. 원고에서는 `numbers.tex` 매크로로 바꾼다(지침 1.2).

---

## 1. 공통 규격(모든 v3 그림)

지침 2절을 그대로 적용한다. 이 절은 v3 구현에 필요한 값만 모은다.

### 1.1 크기, 글자, 선

| 항목 | 값 | 근거 |
|---|---|---|
| 폭 | 전폭 170.0 mm(이번 7개 모두). 반폭 85.0 mm, 1.5단 120.0 mm 는 쓰지 않는다 | 지침 2.1, R-05 |
| 높이 | 180 mm 이하, 목표 165 mm 이하. 그림별 값은 각 절 | 지침 2.1 |
| 제출 배율 | 1.00. `bbox_inches="tight"` 금지 | 지침 2.1 |
| 글꼴 | Liberation Sans 한 계열, `pdf.fonttype 42`, `svg.fonttype none`, Type 3 0개 | 지침 2.2 |
| 글자 크기 | 모든 글자 7 pt(패널 문자 7 pt 굵은 직립 소문자) | 지침 2.2, R-04, R-06 |
| 첨자 | 그림 안 0개. 기호 대신 이름("Source coefficient") | 지침 2.2, R-26 |
| 글자 색 | 검정. 보조 표지와 방향 표지만 #4d4d4d | 지침 2.2 |
| 숫자 | 네 자리 쉼표 없음(1000), 다섯 자리 이상 쉼표(13,606), 음수 U+2212, 숫자와 단위 사이 한 칸 | 지침 2.2, R-13 |
| 그림 안 수치 | 눈금, 컬러바 눈금, 축척 길이, 크기 열쇠 값만 | 지침 2.2, R-25 |
| 선 | 축선·눈금 1.0 pt(왼쪽·아래만, 바깥 눈금 3 pt, 보조 눈금 없음), 주 자료 선 1.25 pt, 보조 1.0 pt, 0선 1.0 pt #4d4d4d | 지침 2.3, R-07 |
| 포레스트 CI | 셀 가중 2.0 pt 둥근 끝, 블록 등가중 1.0 pt(셀 가중 막대 바로 아래 0.8 mm, 세로 막대이면 오른쪽 0.8 mm) | 지침 2.3 |
| 곡선 CI | 띠 채움 alpha 0.20, 테두리 없음, 패널당 2개 이하 | 지침 2.3, 2.7 |
| 동등 띠 | ±0.5 cm, 채움 #ededed, 테두리 없음 | 지침 2.3 |
| 측정점 | 계열 색 채운 원 3.5 pt. 지역별 개별 점 2.5 pt, alpha 0.5, 해당 계열 색 | 지침 2.3 |
| 격자선·배경 | 배경 격자선 0개, 회색 배경 띠 0개, 상자 0개(허용: 지도 확대 사각형, 썸네일 테두리) | 지침 R-08, R-12, H1 |

### 1.2 판정 기호의 대체(모든 그림 공통)

v2 의 4분 판정 기호(▼ ○ ◇ △ × †), 판정표, AB·L 표지, 기호 범례 줄은 모두 없앤다(지침 R-10, 2.7). 대신 다음을 그린다.

| 요소 | 그림 표현 |
|---|---|
| 점추정 | 계열 색 원(대비이면 검정 원) |
| 셀 가중 95 % CI | 굵은 막대(2.0 pt) |
| 블록 등가중 95 % CI | 가는 막대(1.0 pt) |
| 동등 띠 | ±0.5 cm 회색 띠 #ededed |
| 0선 | 1.0 pt #4d4d4d 실선(기준 방법) |

등록 4분 판정은 이 요소에서 그대로 읽힌다. 두 막대가 모두 0 의 같은 쪽이면 오차 감소 또는 증가, 두 막대가 모두 띠 안이면 동등, 그 밖은 미결정이다. 비열등(한계 0.5 cm)은 두 막대의 위끝이 모두 띠의 오른쪽 끝 아래에 있는지로 읽힌다.

- 굵은·가는 막대와 띠의 뜻은 그림마다 한 번, 패널 안 빈 곳의 **열쇠 한 줄**("Cell-weighted", "Block-equal", "±0.5 cm" 견본 3개)로 보인다. 설명문은 이 열쇠를 이름으로 가리키고 모양을 서술하지 않는다(지침 4.4 3).
- 그림에서 읽을 수 없는 판정(Holm 보정, 공통 재표집 CI 의존, 2단 CI)은 설명문 한 문장과 SI 판정표로 옮긴다(지침 2.7, 3.3).
- 곡선 패널은 셀 가중 CI 띠만 그리고, 블록 등가중 값은 Source Data 와 SI 판정표에 둔다. 두 가중의 방향이 갈리는 것이 주장의 일부인 패널(Fig 5e, Fig 7a)은 곡선 대신 n 별 포레스트형 점·두 막대로 그린다(각 절).

### 1.3 방법 색 코드(지침 2.4, 그림과 슬라이드 공통)

| 방법(원고 이름) | 그림 짧은 이름 | 내부 약호 | 색 | 선 | v3 사용 그림 |
|---|---|---|---|---|---|
| Stefan model, source coefficient | Source Stefan | P0 | #4d4d4d | Δ 패널 0선, 절대값 패널 수평 기준선, 실선. 측정점 없음 | 모든 Δ 축, Fig 3a, Fig 7d |
| Stefan model, year-matched thaw index | Year-matched Stefan | P* | #4d4d4d | 점선 (1.2, 1.4) | Fig 2a·b, Fig 6c |
| Stefan model, recalibrated with target labels | Recalibrated Stefan | P1 | #2b5c8f | 실선 | Fig 2, Fig 3a, Fig 4a·d |
| Recalibrated anchor plus residual ML | Anchor + residual ML | R1 | #9a7bc9 | 실선 | Fig 2, Fig 3a(잔차 선분), Fig 4a·d, Fig 7a·d |
| Source anchor plus residual ML | Source anchor + residual | R0 | #9a7bc9 | 파선 (4, 2) | SI 만 |
| Physics pseudo-label augmentation | Physics pseudo-labels | D1 | #568f72 | 일점쇄선 (5, 1.5, 1.5, 1.5) | Fig 6b(지도 열 머리만, 색표는 broc) |
| Direct ML (no physics) | Direct ML | D0 | #6b7280 | 파선 (2, 1.5) | Fig 2, Fig 7a, Fig 1e |
| Physics-input ML | Physics-input ML | F1 계열 | #ad921a | 긴 파선 (7, 2) | SI 만(Fig 3c 는 대비라 검정) |
| Stefan-CCI mean anchor | Stefan-CCI anchor | Re 앵커 | #84480c | 이점쇄선 | SI 만 |
| Selection by CV within target labels | Target-label CV selection | W | 고유 색 없음 | 대비 포레스트의 검정 점(Fig 4c), 워크플로 경로의 검정 선(Fig 7d) | Fig 4b·c, Fig 7d |
| Error floor (reference) | Error floor | | #b0b0b0 | 띠 또는 선 | Fig 7a |

- 검정은 방법이 아닌 요소 전용이다(관측·라벨 위치, 배치 전략, 대비 점추정, 워크플로 경로). 한 축(axes) 안에서 검정은 한 역할만 맡는다.
- 모양은 지역만 뜻한다: Alaska ●, Canada ■, Lena Delta ◆, W Russia ▲, E Russia ▼, Central Russia ✚, Greenland ★(지침 2.5). 하위 지역은 상위 지역의 모양을 쓴다. 티베트와 북대서양은 지침 목록에 모양이 없다 [결정 필요 D-15].
- 빈 마커는 그림마다 한 뜻만 가지며 그 뜻은 그림 안 열쇠 한 줄로 보인다(각 절에 적음).
- 주황 #EA851B 는 자료 표현에 쓰지 않는다.

### 1.4 연속 색표(지침 2.6)

| 양 | 색표 | 쓰는 그림 |
|---|---|---|
| 오차 차이 ΔRMSE(cm) | `cmc.broc`, `TwoSlopeNorm(vcenter=0)`, ±vmax(공유 패널 \|값\|의 99 백분위를 5 cm 단위로 올림). 파랑 = 음수 = 오차 감소 | Fig 6a·b, Fig 7b |
| 예측 차이(보정량, cm) | `cmc.broc`, 같은 규칙. 컬러바 라벨 "ALT change (cm)" | Fig 7b(1 km 판일 때) |
| 영구동토 바탕 | 연속 #d0d0d0, 불연속 #e6e6e6, 그 밖 육지 #f4f4f4, 바다 흰색 | Fig 1a, Fig 6a·b |
| 설계 확대도 | 육지 #f4f4f4, 라벨 블록 #bdbdbd, 채점 블록 #737373 | Fig 1c·d |
| 후보 위치 점 / 선정 지점 | #bdbdbd 2.0 pt / #000000 3.0 pt | Fig 5a·b·c·d |
| 자료 없음 마스크 | #B8BEC6 + `//` 해칭(밀도 1) | Fig 7b(채점 셀 부족 블록) |

### 1.5 축 규칙

- Δ 축 이름은 기준을 이름으로 넣는다: "Error change vs source Stefan (cm)", "Error change vs recalibrated Stefan (cm)", "Error change vs random cells (cm)"(지침 2.12).
- 라벨 수 축(R-24): n ≥ 3 은 log10 축, 눈금은 시험 격자 값에만(전이 3, 10, 40, 160, 320, 1000; 지역 내 20, 50, 100, 200, 500, 1000). n = 0 은 왼쪽 폭 6 mm 별도 축(간격 2 mm, y 공유), 전량('All')은 풀 평균 패널에서만 오른쪽 폭 6–8 mm 별도 축. 별도 축과 본 축 사이를 선으로 잇지 않는다.
- 폭 38 mm 안팎의 작은 다중 그림은 눈금 라벨을 10, 40, 160, 1000 에만 단다(눈금은 남김).
- 혼합 척도(symlog, "log beyond ±2 cm") 0개. 범위 밖 값은 축 끝 검정 삼각 표지(3 pt) 하나로 표시하고 값은 설명문에 쓴다. 한 축에 표지가 둘 이상이면 축 범위나 패널을 다시 정한다(지침 2.7).
- 방향 표지("Lower error" 등, #4d4d4d 7 pt, 화살표 하나)는 그림당 1개다(R-11).

### 1.6 주석 밀도 상한(지침 2.9, 그림마다 F-04 로 점검)

| 항목 | 상한 |
|---|---|
| 문자 수(공백 제외) | 전폭 600. 사유를 이 문서에 적은 그림만 800(Fig 6) |
| 글자 크기 종류 | 1(7 pt) |
| 동사 있는 문장, '읽는 법' 주석 | 0 |
| 요소 라벨 | 1–3 단어('+' 제외). 범주 축 라벨(포레스트 행, 지도 열 머리, 묶음 머리)만 4 단어 이하 |
| 범례 | 그림당 1개 이하, 6항목 이하. 계열 5개 이하면 직접 라벨 |
| 같은 문구 반복 | 0 |
| 내부 코드·방법 약호 | 0 |

### 1.7 내보내기와 기록

- 파일: `outputs/figures/paper/v3_restructure/FigN_<slug>.{pdf,svg,png}`(PNG 600 dpi 검토용, 그림 안 래스터 요소 450 ppi 이상), 패널 값 `outputs/figures/paper/v3_restructure/source_data/FigN_<panel>.csv`, 설명문 `outputs/figures/paper/v3_restructure/CAPTIONS_v3.md`, Table 1 `outputs/figures/paper/v3_restructure/Table1_data.tex`.
- 지침 2.13 의 경로(`outputs/figures/paper/v3/`, `CAPTIONS.md` 의 `## v3/…` 절, `figures/figure_spec.json` 의 `paper_v3_figN`)는 사용자가 비교 뒤 v3 를 채택할 때 옮긴다. 이번 작업은 기존 문서(`CAPTIONS.md`, `figure_spec.json`)를 고치지 않는다 [결정 필요 D-20].
- 그림 함수는 `medium="paper"` 와 `"slide"` 를 받아 같은 자료·색·범위로 두 매체를 그린다(지침 0.4 3). 슬라이드판 차트 규격은 `design/slide_archetypes.md` 1.3절(축 18 pt, 눈금 16 pt, 직접 라벨 16 pt, 배율 1.00, 300 dpi PNG).
- 렌더와 컴파일 전에 `free -g`, `uptime` 을 확인하고 가용 메모리 30 GB 미만이거나 부하 40 초과면 기다린다(작업 지시, 등록 문서 1절 로컬 자원).
- 사후 래스터 보정(채도, 픽셀 덮기, 크롭) 0건(지침 2.13).

---

## 2. 표시 항목 요약

| 항목 | 결과 절 | 주장(축소판) | 크기(mm) | 패널 | 주 패널(면적 비율) | 결과 지도 | X 자리 | v2 대비 변경 |
|---|---|---|---|---|---|---|---|---|
| Fig 1 | R1 | 자료·설계, C7(보조) | 170 × 155 | a·b·c·d·e | a(36 %) | 없음(자료 지도 a, 설계 확대도 c·d) | XH(e) | 중간 |
| Fig 2 | R2–R3 | C2·C3(라벨 수 곡선) | 170 × 135 | a·b·c·d·e·f | a·b(42 %) | 없음 | 없음(XB 는 SI) | 중간 |
| Fig 3 | R2–R3 | C1(위약), C3(구조) | 170 × 120 | a·b·c·d | b·c(48 %) | 없음 | 없음 | 중간 |
| Fig 4 | R3–R4 | C2(좁힘), C5 | 170 × 150 | a·b·c·d | a·b(40 %) | 없음(SI 지도 2종) | XA(d) | 큼 |
| Fig 5 | R6 | C6 | 170 × 140 | a·b·c·d·e·f | a·b·c·d·e(51 %) | a·b·c·d 전략별 선정 지점 지도 | 없음(XD 는 SI) | 전면 교체 |
| Fig 6 | R2, R8 | C1, 예측 구간(보조) | 170 × 160 | a·b·c·d·e | a·b(46 %) | a·b 대상별 오차 변화 지도 | XG(c 행) | 중간 |
| Fig 7 | R4–R5, 워크플로 | C4, C8, 워크플로 | 170 × 165 | a·b·c·d·e | a·b(41 %) | b 알래스카 보정 지도 | XC(e) | 전면 교체 |
| Table 1 | R1 | 자료 범위 | 한 쪽 이내 | | | | 없음(XF 는 Supplementary Table) | 형식 교체 |

본문 결과 지도는 Fig 5, Fig 6, Fig 7 의 세 그림이다(지침 1.2, H5). 결과 절별 SI 지도는 13절.

---

## 3. Fig 1 연구 지역, 라벨, 평가 설계

### 3.1 역할과 주장

- 결과 절 R1. 새 지역은 라벨이 적고 Stefan 계수 E 가 지역마다 다르며, 모든 방법은 같은 채점 블록에서 평가된다는 전제를 보인다. 설명문 제목은 명사구다(지침 4.4 1).
- 근거: `paper/claims/D_data_and_design/README.md` 1.3절, `paper/claims/C7_evaluation_design/README.md` 1.3절.

### 3.2 배치(170 × 약 155 mm, 위에서부터 y)

| 패널 | x(mm) | y(mm) | 크기 | 비고 |
|---|---|---|---|---|
| a 범북극 지도 | 0–98 | 0–98 | 98 × 98 | 주 패널(36 %, 가장 큼). 열쇠 구석은 왼쪽 아래 |
| b E 분포 | 106–170 | 0–98 | 64 × 98 | 행 이름 열 20 mm 포함 |
| c 지역 홀드아웃 확대도 | 0–47 | 104–151 | 47 × 47 | a 의 레나델타 사각형과 1.0 pt 회색 직선으로 연결 |
| d 블록 홀드아웃 확대도 | 51–98 | 104–151 | 47 × 47 | c 와 같은 범위·축척 |
| e 검증 사다리 | 106–170 | 104–151 | 64 × 47 | |

- 패널 문자 배정: 지침 6.1 은 확대도 2개를 c 하나로, 검증 사다리를 d 로 두었다. 등록 문서 6절은 XH 를 'Fig 1 패널 e'로 고정했다. 확대도를 c 와 d 로 나누면 두 문서가 맞고 패널 수는 5 이다(6 이하) [판단].

### 3.3 패널 명세

| 패널 | 내용 | 자료 원천과 필터 | 부호화 | 그림 안 글자 |
|---|---|---|---|---|
| a | 0.5° 블록별 라벨 위치(v3 + LGD 약관 확인분), 영구동토 구역 바탕, 지역 이름, 티베트 삽도, 레나델타 확대 사각형 | `data/processed/paper_figs/fig1_blocks.csv`(열 `kind, region, block, lat, lon, n_loc_1km`). 원 중심 = 블록 `lat, lon`, 면적 ∝ `n_loc_1km`. `kind == 'lgd_added'` 이고 `region ∈ {Russia_C, Tibet_LGD}` 인 블록(독립 새 지역)만 흰 채움이다. `kind == 'lgd_added'` 의 Canada, Russia_E, Russia_W 블록(주 지역 확충 라벨)과 `kind == 'v3'` 블록은 검정 채움이다. `kind == 'natl_si'`(North Atlantic, SI 전용)와 약관 미확인 셀(`fig1_not_drawn.csv`)은 그리지 않는다. 바탕: `data/processed/cci_pfr_mean_1997_2021.nc`(연속 ≥ 90 %, 불연속 50–90 %) [미확인: 자료 이름과 판, 지침 8절 15] | 원: 검정 테두리 1.0 pt, 채움 #000000 alpha 0.35(관측 위치 = 검정, R-18). 독립 새 지역: 흰 채움. 투영 EPSG:3413 형식(중심 경도 −45°, 진척 위도 70° N). 경위선 #e0e0e0 1.0 pt(위도 60°, 70°, 80° N, 경도 30° 간격). 축척 막대 "1000 km"를 70° N 근처, 1.5 pt. 티베트 삽도: 지역 중심 Lambert 방위 등적, 자체 축척 막대 | 지역 이름 7–8개(지시선 1.0 pt), 위도 라벨 3개, "1000 km", 삽도 축척 길이, 열쇠 2줄: 크기 열쇠(3단계 중첩 원, 값 1, 10, 50 [판단: `n_loc_1km` 분포로 확정]) + 견본 "Continuous", "Discontinuous" |
| b | 지역별 Stefan 계수 E 분포(로그 축)와 지역별 원천 계수 | `fig1_z_points.csv`(열 `row, label, z`; E = exp(z)), `fig1_z_summary.csv`(열 `row, E0, n_cells`). 행 순서는 Table 1 순서 | x: "Stefan coefficient, E (cm per √(°C d))", log10, 눈금 1, 2, 5, 10, 20 [판단: 티베트 E 평균 11.03 을 포함]. 개별 점: 지역 모양 2.5 pt, 검정 alpha 0.5(행당 최대 400개 무작위 표본, seed 기록). 중앙값 원 3.5 pt + 사분위 선 2.0 pt 검정. 원천 계수: 행마다 짧은 세로 눈금 1.0 pt #4d4d4d | 행 이름(지역 이름, 4 단어 이하), x 축 이름, 눈금, 직접 라벨 "Source coefficient" 1개(첫 행). 라벨 수 "(3,037)" 등은 Table 1 로 옮긴다 |
| c | 지역 홀드아웃: 레나델타 대상 셀, 대상 셀에서 100 km 거리 등치선(파선), 범위 안 원천 셀 | `data/processed/fidelity_base_v3.csv`(열 `region == 'Lena'`, `lat, lon`). 100 km 등치선은 대상 셀 위치 합집합의 측지 거리 100 km. 범위 안 원천 셀은 없을 수 있다 [미확인] | 대상 셀 검정 원 2.0 pt, 파선 1.0 pt 검정 (3, 2), 육지 #f4f4f4. 블록은 그리지 않는다. 축척 막대 "100 km" | "100 km" 축척, 직접 라벨 "Buffer" 1개 |
| d | 블록 홀드아웃: 같은 범위에서 라벨 블록·채점 블록(분할 1), 라벨 셀 40개(추출 1) | 블록 구성: `src/polar/m1_core.py` 의 `half_split_blocks`(분할 seed 1, 라벨 값 미사용). 라벨 셀: `scripts/3_deep_learning/h40_label_grid.py` 의 `draw_cells`(seed_of(대상, 모드, 분할, n, 추출) 기록) | 0.5° 블록 채움(라벨 #bdbdbd, 채점 #737373), 블록 경계 1.0 pt #e0e0e0, 라벨 셀 검정 3.0 pt. 블록 경도 폭 = 47 mm × 17.2 km ÷ 약 420 km ≈ 1.9 mm(1.0 mm 조건 충족, 지침 2.11 계산식, 범위는 레나 블록 범위 약 221 km + 양쪽 100 km) [판단: 렌더 때 실제 범위로 다시 계산] | 직접 라벨 "Label blocks", "Scoring blocks" 2개 |
| e | 검증 사다리: 채점 셀에서 가장 가까운 학습 셀까지 거리 중앙값(x)과 RMSE(y), 직접 ML 과 재보정 Stefan | 최종: XH 산출(`data/processed/xbatch/XH_validation_ladder/`, 등록 문서 2.8, 3지역 층화 평균과 지역 행) `[XH: Fig 1e \| 결과 열람 뒤 설계, 서술(판정어 없음) \| 2.8 사전 고정 서술 규칙]`. 비교용 미리보기(제출 불가): `data/processed/m1/cv_scheme_comparison.csv`(알래스카, `scheme ∈ {random_cell, site_0.05, block_0.5_canonical, knndm}`, `model ∈ {catboost_lo, stefan}`, seed 0–2 평균, 열 `nnd_median_km, rmse_cm`) | x: "Distance to nearest training cell (km)", log10. y: "RMSE (cm)". Direct ML #6b7280 파선, Recalibrated Stefan #2b5c8f 실선, 측정점 원 3.5 pt. XH 의 블록 재표집 CI 는 세로 막대(셀 가중 굵게, 블록 등가중 가늘게). 3지역 평균 위에 지역 점 2.5 pt alpha 0.5 | 직접 라벨 "Direct ML", "Recalibrated Stefan". 단 이름("Random", "Region holdout")은 점 옆 직접 라벨, 5개 이하 [판단: 글자 수 상한 안에서 렌더 뒤 조정] |

### 3.4 판정 기호 대체

Fig 1 에는 판정이 없다. e 는 등록 문서 2.8 이 판정어 없는 서술로 정했으므로 동등 띠도 두지 않는다. CI 는 XH 산출에 있을 때만 그린다.

### 3.5 주석 상한

문자 수 600 이하, 범례 0개(지도 안 열쇠 2줄), 지도 요소: 경위선, 위도 라벨, 축척 막대, 위치 삽도(티베트), 확대 연결선, 확대도 축척. 투영 이름과 기준 위도는 설명문에 둔다. 확대도 블록 경도 폭 1.0 mm 이상(F-12).

### 3.6 설명문 초안

```text
Study regions, Stefan coefficients and evaluation design. a, Labelled cells of the licence-verified label tables aggregated to 0.5° blocks; circle area is proportional to the number of 1 km label locations in a block, and white circles mark the new regions added after the label tables were fixed (Central Russia, expanded edition, and the Tibetan Plateau, inset; Table 1). Shading shows the continuous and discontinuous permafrost zones [미확인: 영구동토 구역 자료 이름과 판]. The rectangle marks the area of c and d. b, Stefan coefficient E = ALT/√TDD of each labelled cell by region on a logarithmic axis, where ALT is the active-layer thickness and TDD the ERA5-Land air thawing index (°C d, 2015–2020 mean); up to 400 cells are shown per region, and Source coefficient marks the coefficient fitted on the source pool when that region is held out. c, Region holdout: the labelled cells of the Lena Delta and all cells within 100 km of them are removed from the training data. d, Split of the same region into label blocks and scoring blocks (one of five splits) with 40 label cells drawn from the label blocks; errors are scored only in the scoring blocks. The within-region tests use the same block split without source data and were designed after the transfer results had been inspected. e, RMSE of direct ML and of the recalibrated Stefan model under random-cell, site, 0.5° block, kNNDM and region-holdout validation against the median distance from scoring cells to the nearest training cell; [XH: Fig 1e | 결과 열람 뒤 설계, 서술(판정어 없음) | 2.8 사전 고정 서술 규칙]. Cells without a verified licence (Canada 38, North Atlantic 19, W Russia 1) are not drawn. Polar stereographic projection (central meridian 45° W), scale bar true at 70° N; c and d centred on the Lena Delta; Tibetan Plateau inset in a Lambert azimuthal equal-area projection. Coastlines, Natural Earth 50 m; maps drawn with Cartopy [미확인: 판].
```

수치 출처: 약관 미확인 셀 수(38, 19, 1)는 `outputs/figures/paper/CAPTIONS.md` 의 'v2/Fig1_problem' 절과 `fig1_not_drawn.csv`. 기후 기간은 같은 절과 `paper/claims/D_data_and_design/README.md` 5절 단서 5.

### 3.7 모범 그림

Hjort 2018 Fig 1(확대도 연결과 확대도마다 축척), Karjalainen 2019 Fig 2(자료 종류를 모양·명도로), Biskaborn 2019 Fig 2(영구동토 구역 공유 견본), Talucci 2025 Fig 1(구역 단위 비례 원), Ploton 2020 Fig 3(평가 설계를 실제 지도 확대로), Meyer & Pebesma 2022 Fig 1(거리 분포 짝, e 의 거리 축).

### 3.8 v2 대비 차이

| 항목 | v2(`v2/Fig1_problem`) | v3 |
|---|---|---|
| 캔버스 | 180 mm 폭, 글자 6종, 5 pt 미만 4자, 1008자(baseline 3.2) | 170 mm, 7 pt 한 종, 600자 이하 |
| 범례 | 그림 위 8항목 2줄 | 지도 안 왼쪽 아래 열쇠 2줄(크기 열쇠 + 구역 견본) |
| 관측 원 색 | 파랑(재보정 Stefan 색과 겹침) | 검정 alpha 0.35, 독립 새 지역은 흰 채움 |
| 지도 안 문장 | "North Atlantic: SI only (licence check pending)" | 없음(약관 상태는 Table 1 설명문, Data availability) |
| 티베트 삽도 | 범북극 투영 안 | 별도 투영과 자체 축척 막대 |
| b | 이중 축(E 와 z), 행 이름에 라벨 수 | E 로그 축 하나, 지역 모양, 라벨 수는 Table 1 |
| c | 문장·수식 도식("1 Region holdout, 2 A/B split, 3 Methods and score") | 레나델타 실제 지도 확대도 2개(c 지역 홀드아웃, d 블록 홀드아웃) |
| d | 원천 셀 구성 누적 막대(막대 안 백분율, 해칭) | SI 로 이동. 알래스카 비율은 Table 1 열 |
| e | 없음 | 검증 사다리(XH) |

---

## 4. Fig 2 라벨 수 곡선

### 4.1 역할과 주장

- 결과 절 R2–R3, C2·C3. 라벨이 늘 때 원천 계수 Stefan 대비 오차 변화를 방법별로 보인다. 설명문 제목은 초록 대비 AB4(규칙 (a))와 AB5(규칙 (b))의 문구다(원고 명세 7절과 같은 문장). C2 README 5.2 에 따라 '재보정 몫이 대부분'은 주 4지역 평균(러시아 W 지배)에만 쓰고 지역 일반으로 쓰지 않는다.

### 4.2 배치(170 × 약 135 mm)

| 패널 | x(mm) | y(mm) | 크기 | 비고 |
|---|---|---|---|---|
| a 주 4지역 평균(n ≤ 10) | 0–56 | 0–70 | 56 × 70 | n = 0 별도 축 6 mm 포함. 주 패널 |
| b 레나델타·캐나다 평균(전체 n) | 62–144 | 0–70 | 82 × 70 | n = 0 별도 축, 전량 별도 축. 주 패널(가장 큼) |
| b 직접 라벨 열 | 144–170 | 0–70 | 26 × 70 | 가장 긴 라벨 "Anchor + residual ML" 실측 폭 + 1.5 mm 로 렌더 뒤 확정 [판단] |
| c·d·e·f 지역 4개 | 0–38, 44–82, 88–126, 132–170 | 82–124 | 각 38 × 42 | c Lena Delta, d Canada, e W Russia, f E Russia |

주 패널 a·b 면적 합 약 42 %(지침 2.1 표).

### 4.3 패널 명세

| 패널 | 자료 원천과 필터 | 부호화 |
|---|---|---|
| a | `data/processed/paper_figs/fig2_pool_curves.csv`: `edition == 'E1_P4_n_le_10'`, `method ∈ {P1, R1, D0}`, 열 `n, delta, ci_lo, ci_hi`(셀 가중, 10,000회). P*: `fig2_pstar_pool.csv` `edition == 'E1_P4_n_le_10'`(열 `pstar, pstar_lo, pstar_hi`) | x: n = 0 별도 축 + log10(3, 10). y: "Error change vs source Stefan (cm)". 0선 = Source Stefan. Recalibrated Stefan #2b5c8f 실선, Anchor + residual ML #9a7bc9 실선, Direct ML #6b7280 파선, 측정점 원 3.5 pt. CI 띠 2개(재보정 Stefan, 앵커 + 잔차). Year-matched Stefan: n = 0 별도 축의 #4d4d4d 점선 짧은 수평선과 CI 세로 막대 |
| b | 같은 파일 `edition == 'E2_LenaCanada_all_n'`(n 0, 3, 10, 40, 160, 320, −1). P*: `fig2_pstar_pool.csv` `edition == 'E2_LenaCanada_all_n'`(점추정만) | x: n = 0 별도 축 + log10(3–320) + 'All' 별도 축. 계열과 띠는 a 와 같다. 직접 라벨은 b 의 선 오른쪽 끝에만 한 번(4개) |
| c·d·e·f | `fig2_region_curves.csv`: `target ∈ {Lena, Canada, Russia_W, Russia_E}`, `method ∈ {P1, R1, D0}`, `n ≥ 3`, 열 `d_p0, d_p0_lo, d_p0_hi`(셀 가중, `nboot` 1000) | x: log10(n ≥ 3), 전량 축 없음. c, d, f 는 y 공유 [−6, 6] cm [판단]. e(W Russia)는 재보정 n 10 값이 −11.96 cm, 앵커 + 잔차 −12.52 cm 라 삼각 표지가 2개가 되므로 자체 y 범위 [−14, 4] cm 와 자체 눈금 라벨을 둔다(지침 2.7 '표지가 둘 이상이면 축 범위를 다시 정한다'). 후보 셀 수를 넘는 n(W Russia·E Russia 의 n ≥ 40, Canada n 1000)은 #B8BEC6 + `//` 마스크. 지역 이름은 패널 위 열 머리 |

- 원천 앵커 + 잔차(R0)와 증강 + 앵커 + 잔차(R2)는 본문에서 빼고 SI 짝 그림에 둔다. 지침 6.2 는 b 의 직접 라벨에 "Physics pseudo-labels"를 넣었으나 풀 곡선 표에는 물리 증강 직접 ML(D1)이 없고 R2 만 있다(`pool_fixed_curve.csv` 방법 목록). R2 를 "Physics pseudo-labels"로 부르면 방법 이름이 틀리고, R2 는 라벨 10개에서 R1 과 ±0.5 cm 안이라(AB6, `paper/claims/C3_structure_by_label_count/README.md` 1.4) 선이 겹친다 [결정 필요 D-7].
- 토양 보정 Stefan(P1*)은 SI 로 옮긴다(지침 6.2).
- XB(다원천 적층)는 등록 문서 6절이 SI 로 고정했으므로 b 에 선을 더하지 않는다. 지침 6.2 의 '조건이 서면 b 에 선 하나'와 다르며 등록 문서를 따른다 [결정 필요 D-2].

### 4.4 판정 기호 대체

곡선 패널이므로 셀 가중 CI 띠 2개와 0선으로 그린다. 판정 기호 줄("◇ D0 − P0 (L1)")과 오른쪽 아래 판정표를 없애고, 등록 판정(Holm 보정 포함)은 설명문 한 문장과 SI 판정표로 옮긴다. 이 그림은 판정을 직접 말하지 않으므로 동등 띠를 두지 않는다(F-11 은 판정을 말하는 패널에만 동등 띠를 요구).

### 4.5 주석 상한

문자 수 600 이하, 직접 라벨 4개(Year-matched Stefan, Recalibrated Stefan, Anchor + residual ML, Direct ML), 패널당 CI 띠 2개 이하, 눈금 라벨 간격 1.5 mm 이상, 삼각 표지 패널당 1개 이하, 방향 표지 1개(a 의 y 축 아래, "Lower error").

### 4.6 설명문 초안

```text
With ten labels, Stefan recalibration lowered the four-region mean error; no further difference was established for residual ML. Error change is the RMSE of a method minus the RMSE of the source-coefficient Stefan model (zero line) on the scoring blocks of each held-out region; negative values mean lower error. Recalibrated Stefan refits the coefficient E with the n target labels, shrunk towards the source coefficient; Anchor + residual ML adds a CatBoost residual with weight 0.25 to that anchor; Direct ML is CatBoost without a physics anchor; Year-matched Stefan uses thawing degree days of the label years. a, Mean over the four main regions (Lena Delta, Canada, W Russia and E Russia) with up to ten labels. b, Mean over the Lena Delta and Canada, the two regions with at least 40 candidate labels; All, every available label; Year-matched Stefan is a point estimate. c to f, Single regions; W Russia and E Russia have 14–17 candidate label cells, label counts above the candidate pool are masked, and e has its own vertical range because values reach −12.5 cm. Lines are cell-weighted means over five splits; bands are 95% block-bootstrap confidence intervals for the two recalibration-based methods (a, b, fixed-composition stratified means, 10,000 resamples; c to f, 1000 resamples). With ten labels, recalibration lowered the four-region error by 2.45 cm (95% CI 1.88 to 3.04 cm; Holm-adjusted P = 0.001), with lower error in W Russia only and higher error in Canada (2.26 cm; 95% CI 0.71 to 2.94 cm). Adding the residual changed the four-region error by −0.18 cm (95% CI −0.53 to −0.01 cm), significant before correction only (Holm-adjusted P = 0.136). Verdicts for all plotted contrasts are listed in Supplementary Table [number]. The label-grid analyses were internally pre-registered; all values come from CatBoost and the analytic Stefan model run on one computing platform.
```

수치 출처: `paper/claims/C2_bias_diagnosis/README.md` 2.1 A6(AB4 −2.45 [−3.04, −1.88], Holm p 0.001), A7(AB5 −0.18 [−0.53, −0.01], Holm p 0.136), CA1(캐나다 +2.26 [0.71, 2.94]); W Russia −12.5 cm 는 `fig2_region_curves.csv`(`target == 'Russia_W'`, `method == 'R1'`, `n == 10`, `d_p0` −12.52); 재표집 횟수는 `CAPTIONS.md` 'v2/Fig2_label_curve'; 후보 셀 14–17 은 D README 단서 11.

### 4.7 모범 그림

Ploton 2020 Fig 5(곡선과 영 모형 기준선의 교차), Tsai 2021 Fig 7b(외부 기준선), Biskaborn 2019 Fig 1b(선 끝 직접 라벨), Hjort 2018 Fig 2(공유 축).

### 4.8 v2 대비 차이

| 항목 | v2(`v2/Fig2_label_curve`) | v3 |
|---|---|---|
| 배치 | 같은 크기 6패널 + 판정표(7구획) | 위 큰 풀 평균 2판(a, b) + 아래 지역 작은 다중 그림 4개 |
| n 축 | 범주 등간격 눈금, 축 꺾임 + "all" 범주 | log10 + n = 0·전량 별도 축 |
| y 축 | "±2 cm 안 선형, 밖 로그" 혼합 축 | 선형 한 척도, W Russia 패널만 자체 범위 |
| 계열 | P1, R0, R1, R2, D0 + P*, P1* | P1, R1, D0 + P*(R0, R2, P1* 는 SI) |
| 판정 | 기호 줄, † 표지, AB 표지, 판정표 | 없음. 설명문 한 문장 + SI 판정표 |
| 범례 | 그림 위 2줄 | b 의 직접 라벨 4개 |
| 색 | 연도 정합 Stefan 검정 | #4d4d4d 점선(R-23) |

---

## 5. Fig 3 물리 정보의 사용

### 5.1 역할과 주장

- 결과 절 R2–R3. C1 의 위약 대비(물리 유사라벨의 이득이 물리 정보에서 오는지)와 C3 의 결합 구조(잔차 구조와 물리 입력 구조의 우열이 라벨 수에 따라 바뀜)를 보인다.
- 좁힘: 증강 이득의 정보 출처는 부분적이다(√TDD 선형 변환 위약과의 차이는 라벨 0 에서 0.48 cm, 라벨 10개에서 동등, C1 README 5.1 8). 라벨 40–160개의 역전은 캐나다 한 지역에서 왔다(C3 README 1.3 2).

### 5.2 배치(170 × 약 120 mm)

| 패널 | x(mm) | y(mm) | 크기 |
|---|---|---|---|
| a 개념 좌표도 | 0–55 | 0–55 | 55 × 55 |
| b 위약 포레스트 | 63–170 | 0–55 | 107 × 55(행 이름 32 mm 포함). 주 패널(가장 큼) |
| c 결합 구조 곡선 | 0–80 | 65–115 | 80 × 50. 주 패널 |
| d 재보정 방식 차이 | 90–170 | 65–115 | 80 × 50 |

주 패널 b·c 면적 합 약 48 %.

### 5.3 패널 명세

| 패널 | 내용 | 자료 원천과 필터 | 부호화 | 그림 안 글자 |
|---|---|---|---|---|
| a | 레나델타 실제 라벨 점 위의 Stefan 직선 2개와 잔차 선분 | `data/processed/fidelity_base_v3.csv`(`region == 'Lena'`, 열 `e5_sqrt_tdd, alt_cm`). 원천 계수 E0 = 1.619(`table1_rows.csv` `target == 'Lena'` 의 `E0_x`). 재보정 계수 = (m × E_ls + κ × E0)/(m + κ), κ = 10, 라벨 10개(분할 1, 추출 1, `draw_cells`) | x: "Square root of thaw index (√(°C d))". y: "Active-layer thickness (cm)", 아래로 깊어짐(축 뒤집기). 라벨 점 전체 검정 alpha 0.35 2.0 pt, 뽑힌 10개 검정 3.5 pt. Source Stefan #4d4d4d 실선, Recalibrated Stefan #2b5c8f 실선, 잔차 선분 #9a7bc9 1.25 pt(뽑힌 10개에서 재보정 직선까지 세로) | 직접 라벨 3개: "Source Stefan", "Recalibrated Stefan", "Residual". 수식 0개(기호는 설명문) |
| b | 물리 유사라벨 − 위약 유사라벨(직접 ML), 주 4지역 평균, 라벨 0개와 10개 | `data/processed/paper_figs/fig3_a.csv`(`placebo ∈ {shuffle, const_t, const_src, tddlin}`, `n ∈ {0, 10}`, 열 `delta, ci_lo, ci_hi, delta_beq, ci_lo_beq, ci_hi_beq`) | x: "Error change vs placebo (cm)". 포레스트: 검정 점 3.5 pt, 셀 가중 굵은 막대, 블록 등가중 가는 막대, 0선, ±0.5 cm 띠. 묶음 머리 "No labels", "Ten labels"(여백으로 구분, 배경 띠 없음) | 행 이름: "Shuffled physics labels", "Constant at pool mean", "Constant at source mean", "Linear thaw index". 열쇠 한 줄(1.2절) |
| c | 잔차 구조 − 물리 입력 구조, 라벨 수별 | `fig3_b_L10_alln.csv`(`series ∈ {F1k, F1n}`, 열 `n, delta, ci_lo, ci_hi, delta_beq, ci_lo_beq, ci_hi_beq, registered, pool`). F1a(n 0 한 점)는 SI | x: n = 0 별도 축 + log10(3–160) + 'All' 별도 축. y: "Error change, residual vs physics input (cm)". 대비이므로 검정. 두 계열: physics outputs 입력(F1k) 실선, 재보정 계수 입력(F1n) 파선. 등록 n(0, 10)은 채운 원, 보조 n 은 빈 원(그림 안 열쇠 한 줄 "Registered", "Auxiliary"). CI 는 점마다 세로 막대(셀 가중 굵게, 블록 등가중 가늘게, 오른쪽 0.8 mm). 0선, ±0.5 cm 띠. n 40 과 160 은 레나델타·캐나다 2지역이다 | 직접 라벨 "Physics inputs", "Recalibrated inputs". 방향 표지 1개: y 축 아래 "Residual lower"(#4d4d4d, 그림 전체 1개) |
| d | 현지 최소제곱 재보정 − 수축 재보정(서술) | `data/processed/paper_figs/pool_fixed_curve.csv`(`method == 'P2'`, `baseline == 'P1'`, `learner == 'none'`, `scope ∈ {MEAN, region}`, `edition ∈ {E1_P4_n_le_10, E2_LenaCanada_all_n}`, 열 `n, delta, ci_lo, ci_hi`). `role` 열이 '서술(판정에 쓰지 않는다)'이다 | x: log10(3–320). y: "Error change, least squares vs shrinkage (cm)". 풀 평균 선 #2b5c8f 실선 1.25 pt(재보정 계열 색. 대비이나 같은 방법의 두 변형이므로 방법 색을 쓴다 [판단]) + CI 띠 1개. 지역 점 2.5 pt alpha 0.5 지역 모양. 0선. 동등 띠 없음(서술) | 직접 라벨 없음(계열 1개). 지역 모양 열쇠 한 줄(4개) |

### 5.4 판정 기호 대체

b, c 는 두 가중 CI 와 ±0.5 cm 띠로 4분 판정이 읽힌다. 위약 대비의 AB 표지, 'registered' 회색 열, 판정 미니 표를 없앤다. Holm 보정 값(AB3, AB7)은 설명문에 둔다. d 는 서술 대비라 판정을 말하지 않는다.

### 5.5 주석 상한

상자 0개, 개념 패널 라벨 3개, 포레스트 행 이름 4 단어 이하, 방향 표지 1개, 문자 수 600 이하. 빈 마커의 뜻은 '보조 라벨 수' 하나다.

### 5.6 설명문 초안

```text
Physics pseudo-labels lowered error relative to shuffled pseudo-labels, and the better combination structure depended on the number of labels. a, Concept drawn on labels of the Lena Delta: active-layer thickness against the square root of the thaw index, with the Stefan line using the source coefficient (Source Stefan), the line recalibrated with ten labels (Recalibrated Stefan) and the departures of those ten labels from it that residual ML learns (Residual); thickness increases downwards. b, Direct ML trained with physics pseudo-labels minus the same model trained with placebo pseudo-labels, four-region mean, without labels and with ten labels. Physics pseudo-labels lowered error relative to shuffled pseudo-labels by 1.63 cm without labels (Holm-adjusted P = 0.001); the difference from pseudo-labels of a linear thaw-index model was 0.48 cm without labels and within ±0.5 cm with ten labels. c, Residual structure minus physics-input structure against the number of labels; the key separates the registered label counts (0 and 10) from auxiliary counts. With up to ten labels the residual structure had 2.49–3.74 cm lower four-region error under both weightings; at 40 and 160 labels the physics-input structure had 1.85–2.38 cm lower error in the mean of the Lena Delta and Canada, a difference that came from Canada. d, Local least-squares minus shrinkage recalibration of the Stefan coefficient (descriptive; positive values favour shrinkage), four-region mean up to ten labels and Lena Delta and Canada mean beyond. Intervals are 95% block-bootstrap confidence intervals (10,000 resamples of 0.5° scoring blocks), weighted by cell and by block as indicated in the key; the grey band marks ±0.5 cm. The placebo and structure analyses were internally pre-registered.
```

수치 출처: `paper/claims/C1_label0_safety/README.md` 1.3(AB3 −1.63, Holm p 0.001), `fig3_a.csv`(`tddlin`, n 0 −0.4769, n 10 동등), `paper/claims/C3_structure_by_label_count/README.md` 1.4(−2.49부터 −3.74까지 cm, AB7 Holm p 0.001)와 1.3 2(+1.85부터 +2.38까지, 캐나다), 재표집 횟수는 `CAPTIONS.md` 'v2/Fig3_physics_use'.

### 5.7 모범 그림

Burke 2020 Fig 1(물리 개념을 실제 좌표 위에), Biskaborn 2019 Fig 7(좌표 위 개념도), Langer 2024 Fig 1(보정 대상만 강조색), Talucci 2025 Fig 6c(축 옆 방향 표지), Nitzbon 2020 Fig 3j(소형 다중 결과).

### 5.8 v2 대비 차이

| 항목 | v2(`v2/Fig3_physics_use`) | v3 |
|---|---|---|
| 개념 | 없음 | a 개념 좌표도(실제 레나델타 라벨) |
| 위약 | 행 이름 내부 표기, 판정 기호, AB3 표지 | 평이한 행 이름, 두 CI + 동등 띠 |
| 구조 | 판정 미니 표, 'registered' 회색 열 | 채운·빈 원 열쇠 한 줄, 방향 표지 1개 |
| v2 c(지역별 대비, 혼합 척도, 분할 승률 막대, 오차 하한 눈금) | 본문 | SI(지역별 전량 값은 Fig 2 에 있음, H6) |
| L12·L17 포레스트 | 본문 | SI |
| 재보정 변형 | P1, P2, P3, V1 의 P0 대비 4계열 | 최소제곱 − 수축 하나(나머지 SI) |

---

## 6. Fig 4 ML 이득의 조건

### 6.1 역할과 주장

- 결과 절 R3–R4, C2(좁힘)와 C5. 라벨을 쓰는 보정의 이득이 라벨 10개로 잰 원천 계수 Stefan 편향과 순위상관하고, 그 이득을 재보정 몫과 재보정을 넘어선 ML 몫으로 나누면 어떻게 되는지(XA), 대상 라벨 교차검증 선정이 고정 레시피에 비열등이었음(라벨 40개와 160개, 레나델타·캐나다)을 보인다.
- 문장 규칙: C2 는 'label-based correction (recalibration plus residual)'의 이득으로 쓰고 'ML gain is proportional to coefficient error'로 쓰지 않는다(`scirep_format_and_drafts.md` 4.8, C2 README 5.2). 재보정을 넘어선 ML 몫의 관계는 XA 판정 뒤 등록 해석 조각으로 채운다. C5 는 '비열등(한계 0.5 cm)'만 n 과 풀을 밝혀 쓰고 '안전'을 쓰지 않는다(C5 README 1.2 N2). 모든 b·c·d 결과에 '결과 열람 뒤 설계' 표지를 단다.

### 6.2 배치(170 × 약 150 mm)

| 패널 | x(mm) | y(mm) | 크기 | 비고 |
|---|---|---|---|---|
| a 최소 라벨 수 | 0–75 | 0–70 | 75 × 70 | 행 이름 14 mm 포함. 주 패널(가장 큼) |
| b 편향 대 이득 산점도 | 90–160 | 0–70 | 70 × 70 | 주 패널(정사각) |
| c 선정 규칙 포레스트 | 0–80 | 82–127 | 80 × 45 | 행 이름 30 mm 포함 |
| d 이득 분해(XA) | 90–170 | 82–127 | 80 × 45 | 두 하위 축(각 36 mm, 간격 4 mm, x 공유) |

주 패널 a·b 면적 합 약 40 %(지침 2.1 표). d 는 하위 축 2개로 이루어진 한 패널이다.

### 6.3 패널 명세

| 패널 | 내용 | 자료 원천과 필터 | 부호화 | 그림 안 글자 |
|---|---|---|---|---|
| a | 대상별 n*: 재보정(재보정 Stefan − 원천 계수 Stefan)과 재보정을 넘어선 잔차 ML(앵커 + 잔차 − 재보정 Stefan) | `data/processed/paper_figs/fig4_a.csv`(`contrast ∈ {'P1 − P0', 'R1 − P1'}`, 열 `target, mode, n_star, censored, n_lo, n_max, flag`, 15대상). 행 정렬 값: `results/rescale_wf/data/processed/wf/wf_tests.csv`(`test_id == 'WF4-c'`, `scope == 'region'`, `target` 을 `<대상>\|<모드>` 로 맞춤, 열 `diag_abs`) | x: "Labels, n", log10(3, 10, 40, 160, 320, 1000). 행마다 두 계열(세로 ±0.8 mm 어긋남): Recalibration #2b5c8f, Residual ML #9a7bc9. n* 는 원 3.5 pt, 구간 n_lo < n* ≤ n* 는 1.25 pt 선. 미도달(`censored`)은 n_max 에서 오른쪽 축 끝까지 1.0 pt 선과 축에 닿는 화살표 머리(그림 전체 화살표 1종). 행은 10개 편향 크기 내림차순 | 행 이름(대상 이름, AL-1 등은 Supplementary Table S1 정의 뒤에만). 직접 라벨 "Recalibration", "Residual ML", "Not reached" 각 1회 |
| b | 라벨 10개로 잰 원천 계수 Stefan 편향(x)과 대상 라벨 교차검증 선정의 원천 계수 Stefan 대비 오차 변화(y, 라벨 전량) | x: `wf_tests.csv` WF4-c 지역 행 `diag_abs`(28대상). y: `results/rescale_wf/data/processed/wf/wf_curve.csv`(`exp == 'wf4'`, `method == 'W'`, `n == -1`, 열 `target, mode, d_p0, d_p0_lo, d_p0_hi, d_p0_beq, d_p0_beq_lo, d_p0_beq_hi`; `d_p0` = −`gain_w` 확인, `nboot` 10,000) | x: "Bias at ten labels (cm)", log10, 눈금 3, 10, 30 [판단]. y: "Error change vs source Stefan (cm)". 점: 지역 모양 3.5 pt 검정(대비 점추정), 하위 지역과 점 추정 대상(북대서양)은 빈 모양(그림 안 열쇠 한 줄 "Sub-region"). 대상별 세로 CI(셀 가중 굵게, 블록 등가중 가늘게). 0선, ±0.5 cm 띠. 티베트(x 227.71, y −159.55)는 축 밖이므로 오른쪽 아래 모서리 삼각 표지 1개 | 직접 라벨 없음. 방향 표지 "Lower error" 1개(그림 전체 1개). ρ 와 CI 는 설명문(R-25) |
| c | 대상 라벨 교차검증 선정 − 고정 잔차 레시피(λ 0.25), 라벨 10, 40, 160, 전량 | `wf_tests.csv`(`test_id == 'WF4-a'`, `scope ∈ {MEAN, region}`, 열 `contrast, target, n, delta, ci_lo, ci_hi, delta_blockeq, ci_lo_beq, ci_hi_beq, ni`) | x: "Error change vs fixed recipe (cm)". 풀 평균: 검정 점 3.5 pt + 두 막대. 지역 점: 검정 alpha 0.5 2.5 pt 지역 모양. 0선, ±0.5 cm 띠(오른쪽 끝 +0.5 cm 가 비열등 한계와 같다) | 묶음 머리 "Four main regions"(행 "10 labels", "All labels"), "Lena Delta and Canada"(행 "40 labels", "160 labels"). 열쇠 한 줄(1.2절) |
| d | 라벨 전량 이득의 분해: 왼쪽 하위 축 재보정 몫(재보정 Stefan − 원천 계수 Stefan), 오른쪽 하위 축 재보정을 넘어선 ML 몫(앵커 + 잔차 − 재보정 Stefan 가운데 나은 쪽 대비, XA 의 G_ML2) | 최종: XA 산출(`data/processed/xbatch/XA_c2_gain_decomposition/`, 대상별 G_recal, G_ML2 와 ρ 의 계열 군집 재표집 CI) `[XA: Fig 4d \| 사후 분석(재현(비맹검)) \| 2.1 결과 범주(양, 확인하지 못함, 음)와 CE1, CE2 판]`. 비교용 미리보기(제출 불가, '사후' 표지): `paper/claims/C2_bias_diagnosis/tables/derived_c2_wf4c_split.csv`(열 `diag_abs, recal_share, ml_beyond_R1`; 부호를 Δ 규약으로 바꿈) | 두 하위 축 x 는 b 와 같다(log10, 같은 범위). y: 왼쪽 "Error change, recalibration (cm)", 오른쪽 "Error change beyond recalibration (cm)"(두 축 같은 범위 [판단: 티베트 제외 값으로 정함]). 점: b 와 같은 모양·채움 규칙, 왼쪽 #2b5c8f, 오른쪽 #9a7bc9(방법 색이 더해진 방법을 뜻함). 0선, ±0.5 cm 띠. 대상별 CI 는 그리지 않는다(Source Data). 티베트는 각 하위 축 삼각 표지 1개 | 하위 축 머리 "Recalibration", "Beyond recalibration"(4 단어 이하). ρ 는 설명문 |

- a 의 행 정렬 변수: 지침 6.4 는 |ln(E_own/E0)|("Source coefficient error")로 정렬해 b 와 잇자고 했다. b 의 등록 진단(WF4-c)과 XA 의 주 변수(CE2)가 라벨 10개 편향 |b10| 이므로 a 도 같은 변수로 정렬해야 b·d 와 이어진다. |ln(E_own/E0)|(CE1, XA 의 오라클 정의)는 SI 판에 쓴다 [결정 필요 D-8].
- v2 a 의 R1 − P0 n* 는 재보정 몫과 ML 몫을 섞으므로 본문에서 빼고 SI 에 둔다. 남긴 두 대비가 d 의 분해와 같은 구성이다.
- C2 README 4.3 의 대안(지역별 누적 막대)은 쓰지 않는다. 누적 막대는 저장소마다 정의가 다른 RMSE 차를 더하는 문제가 없지만(같은 채점 셀) 티베트가 척도를 지배하고 대상별 순위 관계를 보이지 못한다.

### 6.4 판정 기호 대체

- b, c 는 두 가중 CI 와 ±0.5 cm 띠로 읽힌다. v2 의 판정표, 판정 수 누적 막대, '정확도 이내' 채움 계급(초판 지침의 회색·채운·빈 점)은 쓰지 않는다(H6).
- c 의 비열등은 두 가중 CI 상한이 +0.5 cm(띠 오른쪽 끝) 아래에 있는지로 그림에서 읽힌다. 공통 재표집 CI 의존(라벨 40개의 우세, 전량의 비열등)은 설명문 한 문장과 SI 판정표로 둔다(C5 README 1.2 N4).
- d 는 사후 분석이며 XA 의 범주('양(벗어남)', '확인하지 못함', '음(벗어남)')는 그림에 쓰지 않고 설명문 자리표시로 둔다.

### 6.5 주석 상한

무늬 0개, 회색 행 띠 0개, 그림 안 통계 0줄, 채움 계급 0개, 문자 수 600 이하, 방향 표지 1개(b), 삼각 표지 축당 1개. 빈 마커의 뜻은 '하위 지역 또는 점 추정 대상' 하나다.

### 6.6 설명문 초안

```text
The error reduction of label-based correction was rank-correlated with the source-coefficient bias measured with ten labels. a, Smallest tested number of labels n* (interval from the previous grid value) at which recalibration lowered error relative to the source-coefficient Stefan model (Recalibration) and residual ML lowered error relative to the recalibrated model (Residual ML), under three registered conditions (Methods); Not reached marks targets without n* up to the largest tested count. Rows follow the ten-label bias of b; sub-regions are not independent. b, Error change of the method chosen by cross-validation within the target labels, relative to the source-coefficient Stefan model with all labels, against the absolute mean bias of that model at the first ten labels (Spearman ρ = 0.38, 95% CI 0.17 to 0.55, 28 targets; signed bias ρ = −0.01). The Tibetan Plateau lies outside the axes (bias 227.71 cm, error change −159.55 cm). c, Cross-validation selection minus a fixed residual recipe. The selection was non-inferior (margin 0.5 cm) at 40 and 160 labels in the mean of the Lena Delta and Canada, with 1.32 cm lower error at 160 labels; the 0.87 cm reduction at 40 labels and non-inferiority with all labels depend on the split-independence assumption, and non-inferiority did not hold at ten labels. The gain came from Canada (2.10 and 2.72 cm); the Lena Delta changed by +0.35 and +0.07 cm. d, Error change from recalibration and from residual ML beyond recalibration with all labels, against the same bias; [XA: Fig 4d | 사후 분석(재현(비맹검)) | 2.1 결과 범주(양, 확인하지 못함, 음)와 CE1, CE2 판]. Intervals in b and c are 95% block-bootstrap confidence intervals (10,000 resamples of 0.5° scoring blocks), weighted by cell and by block as indicated in the key; the grey band marks ±0.5 cm. Analyses in b and c were designed after earlier results had been inspected; d is post hoc.
```

수치 출처: `paper/claims/C2_bias_diagnosis/README.md` 2.1 A3, A4(ρ 0.38 [0.17, 0.55], 28대상, 부호 ρ −0.01), 5.1 4와 2.2 TB9(티베트 진단 227.71, 이득 159.55); `paper/claims/C5_method_selection/README.md` 1.3, 2.1 E1–E4, 2.2 E6, E7, E10, E11; n* 조건과 재표집 횟수는 `CAPTIONS.md` 'v2/Fig4_min_labels'.

### 6.7 모범 그림

Hjort 2018 Fig 2(범주별 점과 범위), Aalto 2018 Fig 3(흑백에서도 읽히는 구분), Talucci 2025 Fig 6c(축 옆 방향 표지).

### 6.8 v2 대비 차이

| 항목 | v2(`v2/Fig4_min_labels`) | v3 |
|---|---|---|
| 메시지 | 최소 라벨 수와 개선·악화 분포 | ML 이 언제 이득인가(편향 진단, 이득 분해, 선정 규칙) |
| a | 대상별 3대비 n*, 교차 회색 행 띠, 묶음 정렬 | 2대비(재보정, 재보정을 넘어선 잔차), 회색 띠 없음, 10개 편향 순 정렬 |
| b | R1 − P1 층화 곡선 + 판정표(Fig 2 b 와 중복) | 편향 대 선정 규칙 이득 산점도와 대상별 두 CI(WF4-c, v2 에 없던 결과) |
| c | L3 α 비교와 오른쪽 보조 판정 | 선정 규칙 − 고정 레시피 포레스트(WF4-a, v2 에 없던 결과) |
| d | n 별 판정 수 누적 막대(점·빗금·격자 무늬) | XA 이득 분해(자리표시). 판정 수는 SI 표 |
| 판정 | 기호, †, AB5·AB9 표지 | 두 CI + 띠, 설명문 |

---

## 7. Fig 5 새 라벨의 위치

### 7.1 역할과 주장

- 결과 절 R6, C6. 라벨을 0.5° 블록과 공변량 공간에 퍼뜨리는 추출이 캐나다에서 셀 무작위보다 오차가 작았고 레나델타에서는 그렇지 않았으며, 예측 분산 기반 능동 선정은 이득이 없거나 작았음을 보인다. 원천은 WF2(지역 내)와 WF8(전이)이다.
- 문장 규칙(C6 README 1.2–1.4, 5.2): '안전', '어디서도 나빠지지 않았다', '이질적 지역에서 이득'(일반 법칙), 처방 규칙('S4 로 고른다'), 관측 우선순위 문장을 쓰지 않는다. 방법 수준 서술만 쓴다(WRAPUP 11.2). 모든 결과에 '결과 열람 뒤 설계(실행 전 등록)' 표지를 단다.

### 7.2 배치(170 × 약 140 mm)

| 패널 | x(mm) | y(mm) | 크기 | 비고 |
|---|---|---|---|---|
| a·b·c·d 전략별 선정 지점 지도 | 0–40, 43.3–83.3, 86.7–126.7, 130–170 | 4–44(열 머리 0–3) | 각 40 × 40 | 같은 범위·축척. 주 패널 |
| e 예산 곡선 | 0–105 | 56–111 | 105 × 55 | 하위 축 3개(Canada, Lena Delta, Alaska, 각 약 31 mm, y 공유). 주 패널(가장 큼) |
| f 전이 포레스트 | 113–170 | 56–111 | 57 × 55 | 행 이름 18 mm 포함 |

주 패널 a·b·c·d·e 면적 합 약 51 %.

### 7.3 패널 명세

| 패널 | 내용 | 자료 원천과 필터 | 부호화 | 그림 안 글자 |
|---|---|---|---|---|
| a·b·c·d | 캐나다 지역 내 시험(모드 r), 분할 1, 라벨 100개, 추출 1 의 전략별 선정 지점: a 셀 무작위(S1), b 블록 층화(S2), c 공변량 최대 최소 거리(S4), d 예측 분산 능동 선정(S6) | 후보: 분할 1 의 A 블록 셀(`half_split_blocks`, 라벨 값 미사용). 선정: a `h40.draw_cells`, b `h40.draw_blocks`, c `scripts/3_deep_learning/h54_workflow.py` 의 `kcenter_order`(x25 표준화), d `h54` 의 S6 순차 선정(라벨 20개 무작위에서 시작, CatBoost virtual ensemble 분산). WF2 조각(`results/rescale_wf/data/processed/wf/shards/wf2__cpu__Canada__r__s1__S*_runs.csv`)에는 선정 셀 ID 가 없다 [확인: 열 목록]. a·b·c 는 라벨 값 없이 seed 로 다시 만들 수 있다. d 는 A 쪽 라벨을 쓰는 CatBoost 적합이 필요하다(로컬 CPU 4스레드 이하, 1분할) [결정 필요 D-12] | 범북극과 같은 평사 투영에서 중심 경도를 캐나다 중앙으로 옮김, `aspect="equal"`, 네 지도 같은 범위. 육지 #f4f4f4, 후보 셀 #bdbdbd 2.0 pt, 선정 지점 #000000 3.0 pt. 지역 경계(A·B 블록 합집합 윤곽) 1.0 pt 검정 [판단]. 0.5° 블록 격자는 그리지 않는다(블록 경도 폭 약 0.26 mm < 1.0 mm). 축척 막대와 위치 삽도는 a 에만 [판단: 네 지도가 같은 축척] | 열 머리(4 단어 이하): "Random", "Block stratified", "Covariate spread", "Variance-based active". a 의 "500 km" 축척 길이 [판단] |
| e | 지역 내 예산별 Δ(전략 − 셀 무작위), 앵커 + 잔차(λ 0.25) | `wf_tests.csv`(`test_id == 'WF2-a'`, `target ∈ {Canada\|r, Lena\|r, Alaska\|r}`, `contrast` 가 `S2-S1\|R1(0.25)\|n*`, `S4-S1\|R1(0.25)\|n*`, `S6-S1\|R1(0.25)\|n*`, 열 `n, delta, ci_lo, ci_hi, delta_blockeq, ci_lo_beq, ci_hi_beq`) | x: "Labels, n", log10(20, 50, 100, 200, 500; 캐나다는 200 까지). y: "Error change vs random cells (cm)", 세 하위 축 공유. n 마다 세 전략을 가로 ±1.2 mm 어긋나게 두고 점 + 두 막대(1.2절). 전략 구분은 지침 6.5: Block stratified 검정 실선, Covariate spread 검정 파선, Variance-based active #4d4d4d 점선(점 사이를 1.0 pt 선으로 잇는다). 점 채움: 모두 채운 원. 0선, ±0.5 cm 띠. S6 은 n 20 에서 정의상 0 이므로 n 50 부터 | 하위 축 머리 "Canada", "Lena Delta", "Alaska". 직접 라벨 3개(첫 하위 축 선 끝, 위아래 순서를 지도 열 순서 b, c, d 와 맞춤). 방향 표지 "Lower error" 1개(그림 전체 1개) |
| f | 전이 조건 Δ(전략 − 셀 무작위), 라벨 10개와 40개 | `results/rescale_wf2/data/processed/wf/wf2b_tests.csv`(`exp == 'wf8'`, `test_id ∈ {WF8-a, WF8-b, WF8}`, `contrast` 가 `S2-S1\|R1(0.25)\|n10/n40`, `S4-S1\|R1(0.25)\|n10/n40`, 열 `target, delta, ci_lo, ci_hi, delta_blockeq, ci_lo_beq, ci_hi_beq`). C6 README 2.5 D1–D16 과 같은 행 | x: "Error change vs random cells (cm)". 묶음 머리 "Ten labels"(행 Lena Delta, Canada, W Russia, E Russia, Alaska), "Forty labels"(행 Lena Delta, Canada, Alaska). 행마다 두 전략을 세로 ±0.8 mm 로 두고 e 와 같은 선 모양의 짧은 막대 [판단: 점 + 두 막대, Block stratified 위, Covariate spread 아래]. 0선, ±0.5 cm 띠 | 행 이름(지역), 묶음 머리 2개. 열쇠 한 줄(1.2절) |

- 전략 7종 가운데 본문은 S2, S4, S6 만 그린다. S3, S5, S7 과 알래스카 하위 지역(AL-1, AL-2), L43·L23·L24, 거리 분포는 SI(plan B, C6 README 4.2).
- XD(배치 알고리즘, 학습 정책)는 등록 문서 6절이 SI 로 고정했으므로 f 에 행을 더하지 않는다. 지침 6.5 f 의 'XD 결과는 행으로 더한다'와 다르며 등록 문서를 따른다 [결정 필요 D-3]. 워크플로 아래 배치 효과(XC-5)는 Fig 7e 에 둔다.

### 7.4 판정 기호 대체

e, f 는 두 가중 CI 와 ±0.5 cm 띠로 그린다. 알래스카(지역 내)의 셀 가중과 블록 등가중 방향 차이와 레나델타 S2 의 셀 가중 CI 가 0 위에 있는 것이 그림에서 보이도록 곡선 띠 대신 점·두 막대를 쓴다. WF2·WF8 의 CI 가 추출 조건부라는 점(C6 README 5.1 5)은 설명문에 적는다.

### 7.5 주석 상한

지도 4개 같은 범위·축척, 전략 구분이 흑백에서 선 모양과 직접 라벨로 읽힘, 문자 수 600 이하, 관측 우선순위 표현 0개, 지도 위 수치 0개(축척 제외).

### 7.6 설명문 초안

```text
Spreading labels across blocks and covariate space lowered error versus random cells in Canada, not in the Lena Delta. a to d, One example draw of 100 labels in Canada (within-region test, split 1) by random cells (Random), one cell per 0.5° block in turn (Block stratified), greedy maximisation of the minimum distance in standardized covariate space (Covariate spread) and sequential selection of cells with the largest predictive variance (Variance-based active); candidate cells lie in the label blocks. The maps illustrate the procedures and do not rank sites. e, Error change relative to random cells for the recalibrated anchor plus residual ML (residual weight 0.25) against the number of labels within Canada, the Lena Delta and Alaska. In Canada both spreading strategies lowered error by 2.54–3.21 cm at 50 and 100 labels and by 1.25–1.44 cm at 200 labels. In the Lena Delta the error change of block stratification was +1.47 to +1.98 cm at 20–500 labels, with higher error under both weightings at 100 labels (+1.65 cm). In Alaska the two weightings disagreed (block-equal −1.08 to −1.73 cm, cell-weighted +0.28 to +0.74 cm). Variance-based active selection had 0.12–2.79 cm higher error in Alaska, the Lena Delta and the Alaskan sub-region AL-1 (Supplementary Table S1) at 50–200 labels, and smaller gains than the spreading strategies in Canada. f, Transfer with ten and forty labels. Covariate spread lowered error in Canada at forty labels by 2.39 cm (95% CI 0.49 to 3.40 cm). At ten labels no region had higher error under both weightings, but the cell-weighted interval of block stratification in the Lena Delta lay above zero (+1.89 cm). Intervals are 95% block-bootstrap confidence intervals (10,000 resamples of 0.5° scoring blocks, conditional on the label draws), weighted by cell and by block as indicated in the key; the grey band marks ±0.5 cm. The placement tests were designed after earlier results had been inspected and registered before they were run. Maps use a polar stereographic projection centred on Canada with a scale bar in a; coastlines, Natural Earth 50 m; Cartopy [미확인: 판].
```

수치 출처: `paper/claims/C6_label_placement/README.md` 1.3(2.54–3.21, 1.25–1.44, +1.47부터 +1.98까지, +1.65, −1.08부터 −1.73까지, +0.28부터 +0.74까지, 0.12–2.79, 캐나다 S6 이득이 S2·S4 보다 작음)과 2.5 D1(−2.39 [−3.40, −0.49]), D3(+1.89 [0.30, 3.08]); 재표집 횟수 10,000 은 같은 README 2.0 과 WF 메타(`nboot`).

### 7.7 모범 그림

Linnenbrink 2024 Fig 3(위 지도와 아래 곡선의 열 정렬. 열마다 곡선을 짝짓지 않고 하나로 모은 점이 다르다), Ploton 2020 Fig 3, Fig 5c, Meyer & Pebesma 2022 Fig 1(거리 분포는 SI).

### 7.8 v2 대비 차이

| 항목 | v2(`v2/Fig5_placement`) | v3 |
|---|---|---|
| 내용 | L43(블록 분산 대 셀 무작위, P1), L23 근접 대비, L24 근접 − 원거리, 거리 층 | WF2 지역 내 전략 × 예산, WF8 전이, 능동 선정. L43·L23·L24 는 SI |
| 지도 | 없음 | a·b·c·d 전략별 선정 지점 지도(결과 지도) |
| 판정 | 기호, †, 2단·추출 조건부 CI 병기 | 두 가중 CI + 띠. 추출 조건부임을 설명문에 |
| 변경 정도 | | 전면 교체(재구성안 5절 '큼') |

---

## 8. Fig 6 라벨 0 전이와 예측 구간

### 8.1 역할과 주장

- 결과 절 R2, R8. 설명문 제목은 주 패널 a·b 의 사후 서술 수(18/30 대 2/30)이고 원고 명세 7절과 같은 문장이다(C1 README 1.3 의 사후 서술 문장, 판정어 없음). 학습기 판정은 패널 c 문장으로 둔다. C1 축소판(`paper/claims/C1_label0_safety/README.md` 1.3): 라벨 0 직접 ML 의 오차가 원천 계수 Stefan 보다 컸다는 판정은 두 CI 판정이 같은 5종(랜덤 포레스트, 원천 교차검증으로 조정한 신경망 4종)에만 단정형으로 쓰고, 나머지 5종(CatBoost 세 설정, TabPFN v2, TabICL v2)에는 '분할 독립 가정에 의존'을 병기한다. '시험한 학습기 10종 모두', '안전', '손해 없는'은 쓰지 않는다. 대상별 지도와 '2 cm 넘게 나빠진 대상 수'는 사후 서술이다.
- 예측 구간은 보조 주장이다(`paper/claims/SI_uncertainty/README.md` 1.3).

### 8.2 배치(170 × 약 160 mm)

| 패널 | x(mm) | y(mm) | 크기 | 비고 |
|---|---|---|---|---|
| a 직접 ML 대상별 지도 | 0–82 | 0–76 | 82 × 76 | 주 패널 |
| b 물리 유사라벨 대상별 지도 | 88–170 | 0–76 | 82 × 76 | 주 패널(a 와 같은 크기) |
| 공유 컬러바 | 30–140 | 80–82.5 | 가로 2.5 mm + 라벨 | 공유 패널 폭의 약 67 % |
| c 학습기 포레스트 | 0–96 | 92–156 | 96 × 64 | 행 이름 40 mm 포함 |
| d 포함률 대 구간 폭 | 104–170 | 92–122 | 66 × 30 | |
| e 라벨 있는 구간 | 104–170 | 126–156 | 66 × 30 | |

주 패널 a·b 면적 합 약 46 %.

### 8.3 패널 명세

| 패널 | 내용 | 자료 원천과 필터 | 부호화 | 그림 안 글자 |
|---|---|---|---|---|
| a, b | 라벨 0 대상별 Δ(방법 − 원천 계수 Stefan): a 직접 ML(`D0_1.0`), b 물리 유사라벨 증강(`D1_1.0`) | 새 집계(판정 없음): `scripts/2_evaluation/wf0_reanalysis.py` 의 `load()`, `deltas()` 를 읽기 전용으로 불러 `n == 0` 행의 `D0_1.0`, `D1_1.0` 을 `source_data/Fig6_ab.csv` 로 낸다(입력 `results/rescale_lg/data/processed/lg/lg_curve.csv`, `data/processed/lgd/lgd_curve_lic.csv`. C1 README 2.8 1 이 같은 계산을 저장 없이 했다). 지도에는 원천이 부모 지역을 뺀 모드 x 대상 17개(독립 7: Alaska, Lena, Canada, Russia_W, Russia_E, Russia_C, Tibet_LGD; 하위 지역 10: AL-1–AL-6, CA-2, CA-3, LE-1, LE-2)를 그리고, 모드 i 10개와 확충판 3개(`~exp`)는 SI 지도에 둔다 [결정 필요 D-9]. 대상 중심 좌표: 대상 라벨 셀(`fidelity_base_v3.csv`, v4 는 `fidelity_base_v4.csv`)의 위·경도 평균. 하위 지역 셀 소속은 LG 하네스의 대상 정의 함수에서 가져온다 [미확인: 함수 이름] | 범북극 평사 투영(Fig 1a 와 같은 범위), 영구동토 바탕, 대상 원 지름 3 mm 고정, 검정 테두리 1.0 pt, 채움 `cmc.broc`, `TwoSlopeNorm(vcenter=0)`, ±vmax 두 지도 공유. vmax 는 티베트를 뺀 34값(17대상 × 2)의 \|Δ\| 99 백분위를 5 cm 단위로 올림. 티베트(삽도)와 범위를 넘는 값은 끝 색, 컬러바 양끝 연장 표지 [판단]. 겹치는 원(알래스카 7개 등)은 지시선 부채꼴(1.0 pt 회색)로 푼다. 티베트는 Lambert 방위 등적 삽도와 자체 축척 | 열 머리 "Direct ML", "Physics pseudo-labels". 컬러바 라벨 "Error change vs source Stefan (cm)", 눈금 3–5개. 위도 라벨 3개, "1000 km" 축척(a 에만 [판단]). 지도 위 수 0개 |
| c | 라벨 0 직접 ML 과 물리 기준선의 원천 계수 Stefan 대비, 주 4지역 평균 | `data/processed/paper_figs/fig6_a.csv`(열 `block, key, variant, contrast, delta, ci_lo, ci_hi, delta_beq, ci_lo_beq, ci_hi_beq`). 본문 행 14개: 기준선 `B:ens`, `P*`; `direct_rescale` 의 `catboost_lo`, `catboost`, `catboost_tuned`, `rf`, `F1k`; `lgt` `tabpfn`; `lgf_f` `tabicl`(solid, dashed); `lgf_n` 의 `mlp`, `tabm`, `ftt`, `realmlp`(solid = 원천 교차검증 조정판). 같은 컨텍스트 CatBoost(`catboost_ctx`)와 신경망 기본판(dashed)은 SI | x: "Error change vs source Stefan (cm)". 포레스트: 검정 점 3.5 pt + 두 막대, 0선, ±0.5 cm 띠. 묶음 머리(여백 구분, 배경 띠 없음): "Physics baselines", "Boosting and random forest", "Tabular foundation models", "Neural networks". XG 결과가 나오면 묶음 "Existing ALT maps" 를 더한다 `[XG: Fig 6c \| 결과 열람 뒤 설계, 등록 이탈(WRAPUP 10) \| 2.7 XG-1 행]`(등록 문서 6절 'Fig 6a 행'은 v3 에서 포레스트가 c 로 옮겨 'Fig 6c 행'이 된다) | 행 이름(4 단어 이하): "Year-matched Stefan", "Anchor ensemble", "CatBoost, main", "CatBoost, default", "CatBoost, tuned", "Random forest", "CatBoost, physics inputs", "TabPFN v2", "TabICL v2", "TabICL v2, full context", "MLP", "Multi-head MLP", "FT-Transformer, reduced", "RealMLP". 계산 환경 괄호와 내부 묶음 이름 0개. 열쇠 한 줄(1.2절) |
| d | 라벨 0 예측 구간(90 %)의 포함률 대 평균 폭, 구간 척도화 방식 5종 | `data/processed/paper_figs/fig6_b.csv`(`method ∈ {const, phys, nflow, nflow#placebo, cbq}`, `kind ∈ {pool, region}`, 열 `cov10, wid10`, 풀 행의 CI 열 `cov10_lo1, cov10_hi1, wid10_lo1, wid10_hi1`). 흐름 정합(`cfm`)과 09-26 계층 conformal 참조점은 SI | x: "Interval width (cm)". y: "Coverage". 명목 0.90 수평선 1.0 pt #4d4d4d. 풀 평균 검정 원 3.5 pt + 두 방향 CI 막대(1.0 pt), 지역 점 검정 alpha 0.5 2.5 pt 지역 모양. 이 축에서 검정은 '구간 방법 추정' 한 역할 [판단: 지침 2.4 의 검정 역할 목록에 없는 쓰임이라 확인 필요, 결정 필요 D-17] | 직접 라벨 5개: "Constant", "Physics prior", "Normalizing flow", "Permuted flow", "CatBoost quantile" |
| e | 라벨 40개와 160개에서 앵커 + 잔차 구간의 포함률 대 라벨 0 대비 폭 비 | `data/processed/paper_figs/fig6_d.csv`(열 `target, n, coverage, cov_lo, cov_hi, width_ratio`) | x: "Width relative to no labels". y: "Coverage". 명목 띠 0.85–0.95(#ededed). 점 #9a7bc9(앵커 + 잔차의 구간이므로 방법 색) 3.5 pt, 지역 모양, n 40 채운 모양·n 160 빈 모양 [판단: 빈 마커 뜻이 그림 안에서 이것 하나여야 하므로 a·b·c·d 에는 빈 마커를 쓰지 않는다]. 포함률 CI 세로 막대 1.0 pt. 폭 비는 CI 없음 | 열쇠 한 줄 "40 labels", "160 labels" |

- 예측 지도 행(ALT 예측, 90 % 구간 폭, 외삽 영역)은 지침 6.6 의 세 조건 가운데 LGU-B2 '전이' 판정이 서지 않았으므로(미결정, SI_uncertainty README 4절) SI 한 장으로 둔다. a, b 가 본문 결과 지도이다.
- 지침 6.6 의 e '관측 대 예측(구간 막대, 1:1 선)'은 셀 단위 구간 파일이 필요하다. `data/processed/lgu/lgu_b_intervals.csv` 는 대상 집계 표라 쓸 수 없다 [미확인: 셀 단위 구간 파일]. 그래서 e 는 v2 d(라벨 있는 구간, L25)를 옮긴다 [결정 필요 D-16].

### 8.4 판정 기호 대체

c 는 두 가중 CI 와 ±0.5 cm 띠로 그린다. v2 의 판정 기호 열, 지지 등급 숫자("1, 2 support class"), 오른쪽 등록 판정 글자 열을 없앤다. '분할 독립 가정 의존'(공통 재표집 CI 와 판정이 다른 행, `fig6_a.csv` 의 `ci_dependence`)과 Holm 값은 설명문 한 문장과 SI 판정표로 둔다. 지도 a, b 는 사후 서술이라 판정을 말하지 않는다. d, e 의 등록 판정(LGU-B2, LGU-A1 미결정)은 설명문과 SI 에 둔다.

### 8.5 주석 상한

포레스트 행 14개(XG 를 더하면 16개)의 행 이름 때문에 문자 수 800 이하로 둔다(지침 2.9 사유 기록). 내부 표기 0개, 상자 0개(v2 의 회색 확대 상자 2개 삭제), 두 지도 같은 정규화 객체, 지도 위 수 0개.

### 8.6 설명문 초안

```text
Without target labels, 2 of 30 targets lost over 2 cm with physics pseudo-labels and 18 with direct ML. a, b, Error change relative to the source-coefficient Stefan model without target labels for direct ML (a) and for direct ML trained with physics pseudo-labels (b), at the centre of each held-out target whose source excludes the parent region (17 targets); both maps share one colour scale. The target counts are a post hoc description over 30 targets (adding sub-regions held out within their region and three expanded-label editions; threshold applied to point estimates). c, Error change of direct ML by learner and of two physics baselines, four-region mean. The random forest and four neural networks configured by leave-one-source-region-out cross-validation had higher error (2.44–5.21 cm). The other five learners, including the main CatBoost learner (2.25 cm; 95% CI 1.10 to 3.42 cm; Holm-adjusted P = 0.023), gave the same verdict under the primary intervals, but this verdict depends on the split-independence assumption. Year-matched Stefan had 1.00 cm lower error than the source-coefficient model, also dependent on that assumption. [XG: Fig 6c | 결과 열람 뒤 설계, 등록 이탈(WRAPUP 10) | 2.7 XG-1 행] d, Coverage of 90% prediction intervals against mean width without labels for five ways of scaling the interval, four-region mean and regional values; the line marks the nominal 0.90. e, Coverage of residual-model intervals with 40 and 160 labels against their width relative to the zero-label width. Intervals in c are 95% block-bootstrap confidence intervals (10,000 resamples of 0.5° scoring blocks), weighted by cell and by block as indicated in the key; the grey band marks ±0.5 cm. Polar stereographic projection (central meridian 45° W), scale bar true at 70° N; Tibetan Plateau inset in a Lambert azimuthal equal-area projection; coastlines, Natural Earth 50 m; permafrost zones [미확인: 자료 이름과 판]; Cartopy [미확인: 판].
```

수치 출처: `paper/claims/C1_label0_safety/README.md` 1.3(+2.44부터 +5.21까지, CatBoost +2.25 [1.10, 3.42] Holm p 0.023, 18/30, 2/30, 사후 서술), 2.5(`tables/wf0_risk.csv` `n == 0` 의 `n_worse_2cm` D0_1.0 18, D1_1.0 2, 30대상 구성), 2.7(P* −1.00 [−1.35, −0.32], 분할 독립 가정 의존); 재표집 횟수는 `CAPTIONS.md` 'v2/Fig6_label0_uncertainty'. 계산 환경이 다른 행을 빼지 않는다는 문장은 Methods M12 로 옮겼다.

### 8.7 모범 그림

Langer 2024 Fig 3(예측 지도 위 0 중심 발산색 관측 원, 구간 막대), Talucci 2025 Fig 1(구역 단위 원), Wei 2026 Fig 9g(구간 폭 계급 면적 막대와 지도 계급색 연결을 순차 색표로, SI 지도판), Karjalainen 2019 Fig 3(SI 소형 다중 지도).

### 8.8 v2 대비 차이

| 항목 | v2(`v2/Fig6_label0_uncertainty`) | v3 |
|---|---|---|
| 지도 | 없음 | a, b 대상별 오차 변화 지도(결과 지도, 사후 서술) |
| 학습기 포레스트 | a, 21행, 계산 환경 괄호("Rescale (LG, LGX shards)", "local GPU (RTX 3090)"), 내부 묶음 이름, "F1k", "B:ens" | c, 14행, 학습기 계열 묶음, 평이한 이름. 계산 환경은 Methods·SI |
| 판정 | 판정 기호 열, 지지 등급 숫자, 등록 판정 글자 열 | 두 CI + 띠, 설명문 |
| 구간 | b 정규화기 6종 + 회색 확대 상자 2개, c 구간 점수 포레스트("(iii) − B4") | d 정규화기 5종 직접 라벨, 상자 없음. 구간 점수 포레스트는 SI |
| 라벨 있는 구간 | d | e(같은 자료, 방법 색) |
| 글자 | 1498자, 크기 19종(baseline 3.2) | 800자 이하, 7 pt 한 종 |

---

## 9. Fig 7 라벨이 많은 지역과 워크플로

### 9.1 역할과 주장

- 결과 절 R4–R5, C4·C8 과 워크플로. 알래스카 지역 내에서 재보정 Stefan 대비 이득이 작고(라벨 1000개 0.54 cm), 그 이득이 기후 격자 단위 편향 보정에서 오며, 라벨 수에 따른 워크플로를 실제 결과 위 경로로 보인다.
- 문장 규칙: C4 는 '알래스카, 재보정 Stefan 대비'로 한정하고 레나델타·캐나다는 '차이를 확인하지 못했다'로 쓴다(C4 README 5.2). C8 은 '알래스카 지역 내'로 한정하고 '고해상 지도', 'ML adds spatial detail'을 쓰지 않는다(C8 README 5.2). WF 와 XC 결과는 '결과 열람 뒤 설계' 표지를 단다. 초록에는 WF·XC 결과를 판정어 없이 수치로만 쓴다(사용자 선택지 B, 원고 명세 2.2절 판 B, 결정 대기).

### 9.2 배치(170 × 약 165 mm)

| 패널 | x(mm) | y(mm) | 크기 | 비고 |
|---|---|---|---|---|
| a 지역 내 라벨 수 | 0–52, 59–111, 118–170 | 0–42 | 각 52 × 42 | Alaska, Lena Delta, Canada. y 공유. 주 패널 |
| b 알래스카 지도 | 0–84 | 54–112 | 84 × 58 | 아래 컬러바 112–120. 주 패널(가장 큼) |
| c 이득 분해 | 94–170 | 54–104 | 76 × 50 | 세 열(Total, Between, Within) |
| d 워크플로 경로 | 0–110 | 124–165 | 110 × 41 | 경로 축 26 mm + 관측·진단 띠 축 15 mm |
| e 워크플로 대비(XC) | 118–170 | 124–165 | 52 × 41 | |

주 패널 a·b 면적 합 약 41 %.

### 9.3 패널 명세

| 패널 | 내용 | 자료 원천과 필터 | 부호화 | 그림 안 글자 |
|---|---|---|---|---|
| a | 지역 내 시험(0.5° 블록 절반 채점, 분할 25회)의 Δ(방법 − 재보정 Stefan): 앵커 + 잔차(교차검증 λ), 직접 ML | `results/rescale_wf2/data/processed/wf/wf2b_tests.csv`(`test_id == 'WF6-a'`, `item == 'WF6-a'`, `target ∈ {Alaska\|r, Lena\|r, Canada\|r}`, `contrast ∈ {'R1(λ cv)-P1\|n*', 'D0(catboost)-P1\|n*'}`, 열 `n, delta, ci_lo, ci_hi, delta_blockeq, ci_lo_beq, ci_hi_beq, rmse_B`). 오차 하한: `data/processed/lgx/lgx_floor.csv`(`scope == 'eval'`, 열 `floor_rmse_cm`) | x: "Labels, n", log10(200, 500, 1000; 캐나다는 200) + 'All' 별도 축. y: "Error change vs recalibrated Stefan (cm)", 세 하위 축 공유 [−6, 4] cm [판단]. n 마다 두 방법을 가로 ±1.0 mm 어긋나게 점 + 두 막대(1.2절): Anchor + residual ML #9a7bc9, Direct ML #6b7280. 0선(= 재보정 Stefan), ±0.5 cm 띠. Error floor: 하한 − 재보정 Stefan RMSE(n 별 `rmse_B` 로 계산)를 #b0b0b0 1.25 pt 선으로. 캐나다 하한선(약 −10.7 cm)은 축 밖이라 삼각 표지 1개 | 하위 축 머리 "Alaska", "Lena Delta", "Canada". 직접 라벨 "Anchor + residual ML", "Direct ML", "Error floor"(첫 하위 축에 한 번). 열쇠 한 줄(1.2절) |
| b | 알래스카, 라벨 전량, 지역 내 | 1안(지침 6.7 b): 앵커 + 잔차 예측 − 재보정 Stefan 예측을 1 km 셀로 표시('1 km display of a climate-grid-scale correction'). 셀 단위 예측 파일이 없다 [미확인, 지침 8절 12]. 만들려면 알래스카 전량 라벨로 R1(교차검증 λ)과 P1 을 적합하고 1 km 알래스카 격자에 예측하는 새 실행이 필요하다 [결정 필요 D-11]. 2안(기본값, 지침 6.7 의 대체안): 채점 블록별 오차 변화. `results/rescale_wf3/data/processed/wf/shards/wf9__cpu__Alaska__r__s*_blocksse.npz` 의 단위 `u0`(총, `Alaska\|r`)에서 키 R1(교차검증 λ, n −1)과 P1(n −1)의 블록별 `sse`, `cnt` 를 분할 25개에 걸쳐 합산해 블록별 RMSE 차를 낸다(키 문자열은 같은 파일 `u0_keys`, 블록 좌표는 블록 ID 해석 또는 `fig1_blocks.csv` 결합 [미확인: 블록 ID 해석 함수]). 합산 채점 셀이 10개 미만인 블록은 자료 없음 마스크 [판단] | 평사 투영(중심 경도 알래스카 중앙), `aspect="equal"`. 2안: 0.5° 블록 다각형 채움 `cmc.broc` ±vmax(이 지도 값의 99 백분위, 5 cm 단위 올림), 블록 경도 폭 = 84 mm × 23.5 km ÷ 약 1115 km ≈ 1.8 mm(1.0 mm 조건 충족). 1안: 1 km 셀 `pcolormesh`(rasterized, 450 ppi) `cmc.broc`. 관측 위치(1 km 라벨 위치 343곳) 검정 점 1.5 pt. 위치 삽도, 축척 막대 "200 km" [판단], 경위선 방향마다 5개 이하 | 컬러바 라벨: 2안 "Error change vs recalibrated Stefan (cm)", 1안 "ALT change (cm)". 지도 위 글자 없음 |
| c | 앵커 + 잔차(교차검증 λ) − 재보정 Stefan 을 총·격자 사이·격자 안 성분으로 나눈 값, 라벨 전량, 지역 내 | `results/rescale_wf3/data/processed/wf/wf3b_tests.csv`(`test_id ∈ {WF9-a, WF9-b}`, `contrast ∈ {'R1(λ cv)-P1[총]\|n전량', 'R1(λ cv)-P1[격자 사이]\|n전량', 'R1(λ cv)-P1[격자 안]\|n전량'}`, `target ∈ {Alaska\|r, Alaska~b\|r, Alaska~w\|r, Lena…, Canada…}` 과 층화 평균 행, 열 `delta, ci_lo, ci_hi, delta_blockeq, ci_lo_beq, ci_hi_beq`). C8 README 2.1 A1–A3 과 같은 행 | 세 열 작은 다중 포레스트(열 머리 "Total", "Between climate cells", "Within climate cells", 각 약 15 mm, x 공유 범위). 행: Alaska, Lena Delta, Canada, Three-region mean. 검정 점 + 두 막대, 0선, ±0.5 cm 띠. RMSE 차는 저장소마다 정의가 달라 쌓지 않는다(C8 README 5.1 3). 누적 막대·무늬 0개 | 행 이름 4개, 열 머리 3개. x 축 이름 "Error change vs recalibrated Stefan (cm)" 한 번(가운데 열 아래) |
| d | 워크플로 경로: 라벨 수 구간마다 권고 방법의 실제 Δ(원천 계수 Stefan 대비), 레나델타·캐나다 전이 풀 | n = 0: Source Stefan(0선 자체, 값 0. 연도 정합 Stefan 은 Fig 2a·b 의 n = 0 축에 있으므로 경로에 다시 그리지 않는다, H6). 3–10: Anchor + residual ML(λ 0.25), `fig2_pool_curves.csv` `edition == 'E2_LenaCanada_all_n'`, `method == 'R1'`, `n ∈ {3, 10}`. 40–160 과 전량: Target-label CV selection, `results/rescale_wf/data/processed/wf/wf_curve.csv`(`exp == 'wf4'`, `method == 'W'`, `target ∈ {Lena, Canada}`, `mode == 'x'`, `n ∈ {40, 160, −1}`, 열 `d_p0`)의 두 지역 단순 평균(새 집계, 판정 없음, `h42.pool_rows` 로 계산할 수 있으면 그 값). 지역 값은 점으로 함께. 경로 값은 서술이며 CI 는 SI 표 | 경로 축: x = n(0 별도 축, log10 3–160, 'All' 별도 축), y = "Error change vs source Stefan (cm)". 0선 = Source Stefan. 구간마다 권고 방법 선분: Source Stefan 은 n 0 축의 0선에 직접 라벨만 단다, Anchor + residual ML #9a7bc9 실선 2.0 pt(3–10), Target-label CV selection 검정 실선 2.0 pt(40–160) + 'All' 축 검정 점. 구간 경계에서 선을 잇지 않는다. 지역 값 2.5 pt alpha 0.5 지역 모양. 구간 경계는 x 축 아래 긴 눈금만. 띠 축(y 눈금 없음): 관측 띠 검정 2.0 pt 선, 진단 띠 검정 2.0 pt 선, 같은 내용이 이어지면 선 하나 | 구간 이름 5개("0", "3–10", "40–160", "320–1000", "All"; 320–1000 은 전이 풀에 W 값이 없어 빈 구간 [판단]), 방법 라벨 3개("Source Stefan", "Anchor + residual ML", "Target-label CV selection"), 띠 머리 2개("Placement", "Diagnosis", #4d4d4d), 관측 라벨 1개("Spread placement", 0–160), 진단 라벨 2개("Extrapolation share" 0, "Ten-label bias" 3–All). 텍스트 객체 13개(15 이하) |
| e | 워크플로 전체(배치 → 라벨 10개 진단 → 라벨 수별 방법 규칙) 대비, 같은 지역을 새로 나눈 분할 201–210 | XC 산출(`data/processed/xbatch/XC_workflow_end_to_end/`, 등록 문서 2.3 주 가설 8개) `[XC: Fig 7e \| 결과 열람 뒤 설계, 재사용 지역의 재검정(비맹검 부분 포함) \| 2.3 XC-1, XC-2 포레스트]` | 포레스트: 검정 점 + 두 막대(2단 CI, 셀 가중·블록 등가중), 0선, ±0.5 cm 띠. 묶음 머리 3개: "vs random, fixed recipe"(행 40, 160 labels), "vs recalibrated Stefan"(행 10, 40, 160 labels), "vs random, CV selection"(행 40, 160 labels) | 묶음 머리 3개(4 단어 이하), 행 이름 "10 labels", "40 labels", "160 labels". XC 가 없으면 d 를 전폭(0–170)으로 넓힌다(지침 6.7 e) |

- a 의 y 축: 지침 6.7 a 는 절대 RMSE(cm)와 오차 하한 회색 띠를 적었다. 이 명세는 y 를 '재보정 Stefan 대비 오차 변화'로 바꾸고 하한을 '하한 − 재보정 Stefan RMSE' 선으로 옮긴다. 이유는 두 가지다. (1) C4 의 판정 대비가 R1 − P1 이고 두 가중 CI 가 이 척도에만 있어, 절대 RMSE 축에서는 정의를 밝힌 불확실성(지침 1.2, 2.7)을 그릴 수 없다. (2) 줄일 수 있는 오차(하한까지의 거리)와 이득(0.54 cm)이 같은 축에서 비교된다. 절대 RMSE 곡선은 SI 짝 그림과 Source Data 에 둔다 [결정 필요 D-21].
- 지침 6.7 c 는 분해 패널에 XE 를 함께 둔다. 등록 문서 6절은 XE 를 SI 로 고정했으므로 c 에는 WF9 만 두고 XE 는 Supplementary Fig 로 둔다 [결정 필요 D-4].
- d 의 검정 역할: 지침 2.4 는 한 패널 안에서 검정이 한 역할만 맡게 한다. 교차검증 선정의 경로(검정 선)와 배치 띠(검정 선)가 한 패널에 있으므로, 띠를 경로 축과 다른 축 객체로 그리고 '축마다 한 역할'로 해석한다 [결정 필요 D-13]. 다른 안은 띠 선을 #4d4d4d 로 바꾸는 것인데, #4d4d4d 실선은 원천 계수 Stefan 의 뜻을 가지므로 쓰지 않는다.
- d 구간 이름은 지침 8절 14 의 시험 격자 값을 쓴다. claims 4절 표('1–10', '10–160', '160–1,000', '1,000 이상')와 맞추는 결정이 남아 있다. 방법은 claims 4절 표를 따른다(지침 6.7 d): 라벨 0 은 원천 계수 Stefan(표의 첫 권고. 연도 정합 Stefan 은 '연도 정합 도일이 있으면'의 조건부 권고라 경로에 넣지 않는다), 3–10 은 재보정과 저가중 잔차, 40–1000 은 대상 라벨 교차검증 선정이다. claims 4절의 '1,000 이상 = 물리 앵커 + 잔차(교차검증 λ)'와 슬라이드 A9 예시의 전량 "물리 잔차 결합"은 지역 내 알래스카(a)의 결과이고, 전이 풀의 전량 값은 규칙 W 이므로 경로의 전량 점은 규칙 W 로 그린다. 원고 명세 5절 D1 과 덱 S26 은 이 정의를 그대로 쓴다 [결정 필요 D-14].
- **내용 위험(사용자 확인 필요)**: 이번 작성 때 `wf_curve.csv` 에서 읽은 전이 풀의 규칙 W − 원천 계수 Stefan 점추정은 라벨 40개에서 레나 +0.20, 캐나다 +0.90 cm(단순 평균 약 +0.55 cm), 160개에서 +0.02, +0.35 cm(약 +0.18 cm, 반올림 전 값의 평균), 전량에서 −0.32, +0.32 cm 다(판정 아님). 같은 풀에서 직접 ML 은 라벨 40개부터 0 아래다(−0.47, −0.62, −1.10, −1.27 cm, `fig2_pool_curves.csv` `E2_LenaCanada_all_n` `D0`). 따라서 경로는 40–160 구간에서 0 위에 그려진다. 규칙 W 의 권고 근거는 고정 레시피 대비 비열등(Fig 4c)이며 원천 계수 Stefan 대비 이득이 아니다. 경로를 이대로 그릴지(기본값, 지침 6.7 d 의 레나델타·캐나다 풀), y 를 '재보정 Stefan 대비' 또는 '고정 레시피 대비'로 바꿀지, 라벨 10개까지는 주 4지역 풀을 쓰는 혼합 풀(덱 초판의 안)로 할지는 사용자 결정이다. 원고 명세 D1 과 덱 S26 은 같은 기본값과 같은 선택지를 적는다 [결정 필요 D-13]. 설명문 d 에는 '40개부터의 선정 권고는 고정 레시피 대비 비열등(Fig 4c)에 근거한다'를 적었다.

### 9.4 판정 기호 대체

a, c, e 는 두 가중 CI 와 ±0.5 cm 띠로 그린다(a 는 두 가중의 방향 차이가 직접 ML 에서 크므로 곡선 띠 대신 점·두 막대). v2 의 상자 친 판정 격자(b), 정지 규칙 곡선(c), 초록 대비 포레스트(d)는 SI 로 옮기고 초록 대비 묶음은 Supplementary Table 로 둔다. 공통 재표집 CI 의존(알래스카 라벨 500개와 전량)은 설명문 한 문장으로 둔다. d 는 서술 경로라 판정을 말하지 않는다.

### 9.5 주석 상한

글씨 상자 0개, 순서도 상자·마름모·셰브런·신호등 색·구간 사이 화살표 0개, d 의 텍스트 객체 15개 이하와 반복 0, 동사 문장 0개, 문자 수 600 이하, 패널 순서가 읽는 순서, 방향 표지 1개(a 의 첫 하위 축 y 아래 "Lower error").

### 9.6 설명문 초안

```text
Within Alaska, residual ML gains over recalibrated Stefan were below 1 cm and came from climate-grid-scale bias correction. a, Within-region tests (half of the 0.5° blocks scored, 25 splits): error change of Anchor + residual ML (cross-validated residual weight) and Direct ML relative to the Stefan model recalibrated on the same labels. Error floor is the covariate-conditional floor (Alaska 11.31, Lena Delta 14.83, Canada 17.90 cm) minus the recalibrated Stefan RMSE; the Canadian line (−10.6 to −10.8 cm) lies below the axis. In Alaska with 1000 labels the residual model had 0.54 cm lower error (14.41 to 13.87 cm; 95% CI 0.32 to 0.69 cm), 19% of the squared error above the floor; the reductions with 500 and all labels (0.39 and 0.52 cm) did not hold under the common-resampling interval. No difference was established in the Lena Delta or Canada. b, Alaska with all labels: error change by scoring block. c, Error change split between and within ERA5-Land climate cells. In Alaska about 99% of the squared-error reduction came from the between-cell component (−1.03 cm; within-cell −0.01 cm, within ±0.5 cm); in the three-region mean the within-cell predictions increased error by 0.09 cm. d, Recommended method for each label interval at its error change relative to the source-coefficient Stefan model in the transfer pool of the Lena Delta and Canada (two-region mean of point estimates; 320 and 1000 labels not tested), with placement and diagnosis steps below; selection from 40 labels is recommended for its non-inferiority to a fixed recipe (Fig. 4c). e, [XC: Fig 7e | 결과 열람 뒤 설계, 재사용 지역의 재검정(비맹검 부분 포함) | 2.3 XC-1, XC-2 포레스트]. Intervals are 95% block-bootstrap confidence intervals (10,000 resamples of 0.5° scoring blocks), weighted by cell and by block as indicated in the key; the grey band marks ±0.5 cm. a to d were designed after the transfer results had been inspected. Polar stereographic projection centred on Alaska; coastlines, Natural Earth 50 m; Cartopy [미확인: 판].
```

b 가 1안(1 km 보정 지도)이면 b 문장을 "b, Alaska with all labels: 1 km display of the climate-grid-scale correction (residual model minus recalibrated Stefan prediction), with label locations."로 바꾼다.

수치 출처: `paper/claims/C4_sufficient_labels/README.md` 1.3(−0.54, 14.41 → 13.87, 19 %, 0.39, 0.52, 공통 CI)과 2.1 E1.R1.n1000(CI [−0.69, −0.32] / [−0.80, −0.40]), 2.6 E8.floor(11.31, 14.83, 17.90); `paper/claims/C8_gain_source/README.md` 1.3(약 99 %, −1.03, −0.01 동등, +0.09). 캐나다 하한선 −10.6 에서 −10.8 cm 는 `data/processed/lgx/lgx_floor.csv`(`region == 'Canada'`, `scope == 'eval'`, 17.90)에서 `results/rescale_wf2/data/processed/wf/wf2b_tests.csv`(`test_id == 'WF6-a'`, `target == 'Canada|r'`, `R1(λ cv)-P1`, 열 `rmse_B` 28.55, 28.66)를 뺀 값이다(검토 때 계산, 판정 아님).

### 9.7 모범 그림

Nitzbon 2024 Fig 3(개념 위·결과 아래, 프리프린트 분석이라 게재본 확인 필요 [미확인]), Ploton 2020 Fig 6(실제 지도 축소판의 흐름), Tsai 2021 Fig 7b(자료량 곡선 위 외부 기준선 = 오차 하한), Langer 2024 Fig 5(차이 지도의 같은 발산 범위).

### 9.8 v2 대비 차이

| 항목 | v2(`v2/Fig7_deployment`) | v3 |
|---|---|---|
| a | 둥근 상자 5개(T0–T160)와 문장 3줄씩의 순서도(H1 위반) | 지역 내 라벨 수 대비(WF6, v2 에 없던 결과)와 오차 하한선 |
| b | 상자 친 판정 격자 | 알래스카 결과 지도(블록 또는 1 km) |
| c | 정지 규칙 곡선 | 격자 사이·격자 안 분해(WF9, v2 에 없던 결과) |
| d | 초록 대비 AB1–AB10 포레스트, AB10 별도 축 | 워크플로 경로(실제 값 위 선분과 띠). AB 묶음은 Supplementary Table |
| e | 없음 | XC 워크플로 대비(자리표시) |
| 글자 | 1744자, 내부 코드(T0, SC1w, AB) | 600자 이하, 내부 코드 0 |
| 변경 정도 | | 전면 교체 |

---

## 10. Table 1 대상, 라벨, 자료

### 10.1 역할과 형식

- 결과 절 R1. 설명문 제목은 명사구 "Target regions, labels and data sources."
- 편집 가능한 LaTeX booktabs 표(`Table1_data.tex`), 이미지 아님, 한 쪽 이내, 각주 0개, 세로선·셀 채움·색 글자 0개, 묶음 행 머리는 기울임(지침 3.1, R-14).
- 원천 표에서 스크립트로 만든다. 손으로 고친 셀 0개.

### 10.2 열

| 열 머리(영문) | 원천과 열 | 형식 |
|---|---|---|
| Region | `table1_rows.csv` `name` → 지역 이름 결정(2.5절, D-14) | 왼쪽 정렬 |
| Label source | `table1_rows.csv` `label_type` 을 짧게: "Points (ALLena)", "Points (ABoVE)", "CALM site means", "1 km cell means" [미확인: 자료 인용명과 일치 여부, 지침 6.8] | 왼쪽 정렬 |
| Labels | `label_rows` | 정수, 오른쪽 정렬, 다섯 자리 이상만 쉼표(13,606) |
| 1 km locations | `loc_1km` | 정수 |
| 0.5° blocks | `blocks` | 정수 |
| Splits | `valid_splits_x` | 정수 |
| $E_0$ (cm per √(°C d)) | `E0_x` | 소수 둘째 자리 |
| Alaska share of source cells | `data/processed/paper_figs/fig1_source.csv` `share_alaska`(주 지역, Alaska). 새 지역 행은 LGD 단위 파일 [미확인: 경로] | 백분율 정수(94 %) |

- 없애는 열: Modes(x, i), Role(P4, PE1, PE2, "Point est."), Licence(v3), A cells, Scored cells(범위는 Supplementary Table).

### 10.3 행

| 묶음(기울임) | 행 |
|---|---|
| Main regions | Lena Delta, Canada, W Russia, E Russia |
| Reference | Alaska |
| New regions | Central Russia (expanded), Tibetan Plateau |

- 하위 지역 10행(AL-1 등), 점 추정 전용 행(Russia C v3 7셀, Greenland 3셀), North Atlantic 은 Supplementary Table S1 로 옮긴다. 지침 6.8 은 하위 지역만 옮기고 나머지 처리는 정하지 않았다 [결정 필요 D-15].
- XF 새 지역 행: 지침 6.8 은 'XF 새 지역 행은 결과가 나오면 더한다'고 했으나 등록 문서 6절은 XF 를 'Supplementary Table'로 고정했다. 등록 문서를 따른다 [결정 필요 D-5].

### 10.4 설명문 초안(150 단어 이하)

```text
Target regions, labels and data sources. Labels counts label rows; 1 km locations groups rows by a 0.009° cell index; 0.5° blocks are the units of the label and scoring split; Splits is the number of valid block splits in the transfer design; E0 is the Stefan coefficient (cm per √(°C d)) fitted on the source pool when the region is held out; Alaska share of source cells is the fraction of source cells located in Alaska. Main regions were used in earlier experiments and are retested here; Alaska is a reference target that is not averaged with them; new regions were added after the label tables had been fixed (licence-verified edition). Sub-regions, targets with point estimates only and the North Atlantic group are listed in Supplementary Table S1. Access conditions are given under Data availability.
```

### 10.5 모범과 v2 대비 차이

- 모범: Sci Rep 표 규정(편집 가능, 한 쪽 이내, 각주 없음), Polar 최종 덱의 북탭스 헤어라인 표(baseline 4절 강점 7).
- v2(`v2/Table1_data`): PDF·PNG 그림 표, 각주 a·b·c·d·e·f, 2575자, 내부 코드 열(Modes, Role, Licence), 하위 지역 10행 포함, 네 자리 쉼표(3037) → v3: LaTeX 표, 각주는 설명문, 내부 코드 0, 7행, 쉼표 규칙(3037, 13,606).

---

## 11. 판정 기호 대체 요약

| 그림 | 판정을 읽는 요소 | 그림 밖으로 옮긴 것 |
|---|---|---|
| Fig 1 | 없음(서술) | 없음 |
| Fig 2 | 곡선 CI 띠 2개, 0선 | 4분 판정, Holm(설명문 1문장, SI 판정표) |
| Fig 3 | b·c 두 CI + 띠 | AB3·AB7 Holm(설명문) |
| Fig 4 | b·c 두 CI + 띠(c 의 띠 오른쪽 끝 = 비열등 한계) | 공통 재표집 CI 의존(설명문), XA 범주(자리표시) |
| Fig 5 | e·f 점 + 두 막대 + 띠 | 추출 조건부 CI 단서(설명문) |
| Fig 6 | c 두 CI + 띠 | 분할 독립 가정 의존, Holm, LGU 판정(설명문, SI) |
| Fig 7 | a·c·e 점 + 두 막대 + 띠 | 공통 재표집 CI 의존(설명문), XC Holm(자리표시) |

---

## 12. X 묶음 자리표시 목록(등록 문서 6절 기준)

| 묶음 | 등록 문서 6절 배치 | v3 의 자리 | 자리표시 |
|---|---|---|---|
| XA | 본문(Fig 4 패널 또는 문장) | Fig 4d, 설명문 d 문장(결과 절 R3 은 `[XA: R3 \| 사후 분석(재현(비맹검)) \| 2.1 해석 조각]`) | `[XA: Fig 4d \| 사후 분석(재현(비맹검)) \| 2.1 결과 범주(양, 확인하지 못함, 음)와 CE1, CE2 판]` |
| XB | SI | 없음(Fig 2 에 선을 더하지 않음) | |
| XC | 본문(Fig 7 패널) | Fig 7e | `[XC: Fig 7e \| 결과 열람 뒤 설계, 재사용 지역의 재검정(비맹검 부분 포함) \| 2.3 XC-1, XC-2 포레스트]` |
| XD | SI(알고리즘 정의는 Methods) | 없음(Fig 5f 에 행을 더하지 않음) | |
| XE | SI | 없음(Fig 7c 는 WF9 만) | |
| XF | Supplementary Table | 없음(Table 1 에 행을 더하지 않음) | |
| XG | 본문(Fig 6a 행) 또는 Supplementary Table | Fig 6c 의 묶음 "Existing ALT maps" | `[XG: Fig 6c \| 결과 열람 뒤 설계, 등록 이탈(WRAPUP 10) \| 2.7 XG-1 행]` |
| XH | 본문(Fig 1 패널 e) | Fig 1e | `[XH: Fig 1e \| 결과 열람 뒤 설계, 서술(판정어 없음) \| 2.8 사전 고정 서술 규칙]` |
| XI, XJ | SI | 없음 | |

- 결과가 나오기 전 비교용 렌더는 자리표시 패널을 빈 축과 축 이름만으로 그리고, 미리보기 자료를 쓴 판에는 파일 이름에 `_preview` 를 붙인다. 미리보기 판은 제출하지 않는다.

---

## 13. 결과 지도 계획(지침 1.2 표 갱신)

| 결과 절 | 본문 결과 지도 | SI 결과 지도(2종 이상 충족 여부) | 자료 상태 |
|---|---|---|---|
| R1 평가 설계 | Fig 1c·d 설계 확대도(자료 지도 a 는 셈에서 제외) | 검증 사다리 분할 지도 | 있음(XH 뒤 사다리 지도) |
| R2 라벨 0 전이 | Fig 6a·b | 모드 i 와 확충판 대상 지도, 저가중 잔차판·앵커 + 잔차판(2종 이상) | 대상별 값은 새 집계 필요(8.3), 대상 중심 좌표 [미확인] |
| R3 라벨 3–10개 | 없음 | 대상별 10개 편향 지도, 라벨 10개 재보정 이득 지도 | `wf_tests.csv` WF4-c `diag_abs`, `wf_curve.csv` P1 `d_p0` |
| R4 라벨 40–160개 | 없음 | 대상별 규칙 선정 결과 지도(고른 방법을 방법 색 원으로), 규칙 − 고정 레시피 대상별 지도 | `wf_meta.json` 선정 횟수 [미확인: 대상별 선정 열] |
| R5 라벨이 많은 지역 | Fig 7b | 1 km 보정 지도 또는 블록 지도 가운데 본문에 쓰지 않은 판, 레나델타·캐나다 블록 지도 | 블록판 있음(WF9 조각), 1 km 판 새 실행 필요 |
| R6 새 라벨의 위치 | Fig 5a·b·c·d | 대상별 전략 − 무작위 지도, 레나델타 선정 지점 지도 | 있음(선정 재현 필요, 7.3) |
| R7 독립 지역 | 없음 | XF 뒤 독립 지역 대상별 지도, 예측 지도 | XF 뒤 |
| R8 예측 구간 | 없음(조건 미충족) | 예측 ALT, 90 % 구간 폭, 외삽 영역 지도(레나델타) | `outputs/maps/transfer_lena/` 있음 |

관측 우선순위(어디를 재야 하는지)를 순위로 보이는 지도는 어느 그림에도 두지 않는다(WRAPUP 11.1, 11.2, C6 README 5.2). Fig 5a·b·c·d 는 절차의 예시 추출이며 설명문에 그 사실을 적는다.

---

## 14. 결정·확인 필요 사항

| 번호 | 항목 | 이 명세의 기본값 | 근거 |
|---|---|---|---|
| D-1 | Fig 1 패널 문자(확대도 c·d 분리, 사다리 e) | 분리 | 지침 6.1 과 등록 문서 6절('Fig 1 패널 e')을 함께 맞춤 |
| D-2 | XB 선을 Fig 2b 에 더할지 | 더하지 않음(SI) | 등록 문서 6절이 결과 전에 SI 로 고정 |
| D-3 | XD 행을 Fig 5f 에 더할지 | 더하지 않음(SI) | 같음 |
| D-4 | XE 를 Fig 7c 에 넣을지 | 넣지 않음(SI) | 같음 |
| D-5 | XF 행을 Table 1 에 더할지 | 더하지 않음(Supplementary Table) | 같음 |
| D-6 | XG 행 위치 | Fig 6c 의 새 묶음 | 포레스트가 v3 에서 a 에서 c 로 이동 |
| D-7 | Fig 2 의 증강 결합(R2) 곡선 | SI 로 이동 | 풀 표에 D1 이 없고 R2 는 R1 과 겹침, 방법 색 표에 R2 없음 |
| D-8 | Fig 4a 행 정렬 변수 | 라벨 10개 편향 \|b10\| | b·d 의 변수와 같게. 지침 6.4 는 \|ln(E_own/E0)\| |
| D-9 | Fig 6a·b 지도의 대상 범위 | 모드 x 17대상(모드 i 와 확충판은 SI) | 같은 위치 원이 겹침. 30대상 수는 설명문 |
| D-10 | Fig 6a·b vmax 규칙 | 티베트를 뺀 99 백분위, 넘는 값은 끝 색 | 티베트 값이 척도를 지배 |
| D-11 | Fig 7b 판 | 2안(채점 블록 지도). 1안(1 km 보정 지도)은 새 적합·예측 실행 승인 뒤 | 셀 단위 예측 파일 없음 |
| D-12 | Fig 5d(능동 선정 지도) | 로컬 1분할 재선정 승인 뒤. 승인 없으면 a·b·c 3열(각 52 mm) | S6 은 라벨을 쓰는 CatBoost 적합 필요 |
| D-13 | Fig 7d 경로의 y 기준, 풀, 검정 역할 | 원천 계수 Stefan 대비, 레나델타·캐나다 전이 풀(지침 6.7 d), 띠는 별도 축. 대안: y 를 재보정 Stefan 또는 고정 레시피 대비로, 혼합 풀(라벨 10개까지 주 4지역) | 3–160 구간 경로가 0 위에 놓임(9.3 내용 위험). 원고 명세 D1 과 덱 S26 이 같은 기본값을 쓴다 |
| D-14 | 지역 이름, 구간 이름, 경로의 구간별 방법 | "Lena Delta", "W Russia", "E Russia", "Central Russia"; 구간 "0", "3–10", "40–160", "320–1000", "All"; 방법 Source Stefan(0), Anchor + residual ML(3–10), Target-label CV selection(40–All) | 지침 2.5, 8절 7, 8절 14(사용자 확인), claims 4절 표 |
| D-15 | Table 1 의 점 추정 행·북대서양, 티베트·북대서양 모양 | Supplementary Table S1 로 이동, 모양은 미정 | 지침 6.8, 2.5 에 정의 없음 |
| D-16 | Fig 6e 내용 | v2 d(라벨 있는 구간) 이동 | 셀 단위 구간 파일 없음 |
| D-17 | Fig 6d 의 검정 쓰임 | 구간 방법 추정을 검정으로 | 지침 2.4 검정 역할 목록 밖 |
| D-18 | 제목 | 임시 제목 'Permafrost active-layer thickness prediction under sparse observations using physics-based pseudo-label augmentation and machine learning' 유지(그림에는 영향 없음) | 사용자 결정 |
| D-19 | 초록의 WF 결과 | 판정어 없이 수치(선택지 B) | 사용자 결정 대기. 그림 설명문은 WF 결과에 '결과 열람 뒤 설계' 표지를 단다 |
| D-20 | v3 채택 뒤 경로 | `outputs/figures/paper/v3/`, `CAPTIONS.md` 의 `## v3/…`, `figure_spec.json` 의 `paper_v3_figN` 으로 옮김 | 지침 2.13. 이번 작업은 기존 문서를 고치지 않음 |
| D-21 | Fig 7a 의 y 축 | 재보정 Stefan 대비 오차 변화와 '하한 − 재보정 Stefan RMSE' 선(절대 RMSE 곡선은 SI) | 지침 6.7 a 는 절대 RMSE 와 하한 띠. 두 가중 CI 가 Δ 척도에만 있음 |

**미확인 목록**: 영구동토 구역 자료 이름과 판(Fig 1a, 6a·b), 대상 중심 좌표 함수(Fig 6a·b), 블록 ID 해석 함수(Fig 7b), 알래스카 1 km 예측 격자(Fig 7b 1안), 레나델타 확대 범위 안 원천 셀 유무(Fig 1c), 라벨 출처 이름과 자료 인용명(Table 1), 새 지역 행의 알래스카 비율 원천(Table 1), 규칙 W 의 대상별 선정 열(SI 지도), Cartopy 판, Nitzbon 2024 게재본 그림.

---

## 15. QA 계획(렌더 뒤, 지침 7.1)

| ID | 점검 | 이 명세의 기준 |
|---|---|---|
| F-01 | 캔버스 | 7개 모두 폭 170.0 mm(±0.5), 높이 각 절 값(165 mm 이하) |
| F-02, F-03 | 글꼴, 글자 크기 | Liberation Sans 한 계열, 모든 문자 7.0 pt(회전 글자는 `char_size()`) |
| F-04 | 문자 수, 라벨 | 600(Fig 6 은 800), 라벨 1–3 단어, 범주 축 4 단어 |
| F-05 | 내부 코드 | 정규식 0건(P0, P1, R1, D0, D1, W, AB, L, WF, X 포함) |
| F-06, F-07 | 제목·문장, 상자·화살표 | 축 제목·suptitle 0, 동사 문장 0, 글씨 상자 0, 화살표 1종(Fig 4a 미도달)·곡선·사선 0 |
| F-08 | 선 굵기 | 1.0 pt 이상 |
| F-09 | 색 | 1.3절 색과 1.4절 색표만. 4종 색각 모사(3형 행렬 추가 필요, 지침 8절 3)와 흑백 렌더에서 선 모양·직접 라벨로 구분 |
| F-10 | 중복 부호화 | y 값과 채움 계급 겹침 0, 그림 안 수치 0(눈금·컬러바·축척·크기 열쇠 제외) |
| F-11 | 기준선 | 모든 Δ 축에 0선, 판정을 말하는 패널에 띠, 방향 표지 그림당 1개 |
| F-12 | 지도 | 축척 막대, 위도 라벨, `aspect="equal"`, 비교 지도 같은 범위·정규화, 확대도 블록 경도 폭 1.0 mm 이상(Fig 1d 약 1.9 mm, Fig 7b 약 1.8 mm) |
| F-13 | 수치 | 그림 값 = `source_data/FigN_*.csv` = 3–10절 원천(스크립트 대조) |
| F-14 | 숫자 표기 | 네 자리 쉼표 0, U+2212, 가운뎃점 숫자 나열 0 |
| F-15 | 설명문 | 350 단어 이하, "This figure" 시작 0, 구간 정의와 n, 색 이름·모양 서술 0, 지도는 투영·기준 위도·자료 출처 |
| F-16 | 비교 검토 | 170 mm 인쇄 크기 PNG 를 3–9절의 모범 그림과 v2 그림 옆에 놓고 검토, 결함 목록을 이 폴더의 QA 기록에 남김 |
| F-17 | 패널 위계 | 2절 표의 주 패널 비율과 렌더 실측 대조 |
| F-18 | 결과 지도 | Fig 5, 6, 7 에 결과 지도, 관측 우선순위 지도 0 |

설명문 초안 단어 수(이 문서 작성 때 공백 단위로 잼, 자리표시 포함과 제외): 16절.

---

## 16. 설명문 초안 단어 수

공백 단위로 센 값이다(2026-10-04 검토 반영 뒤 다시 잼, 이 문서의 `text` 블록). '자리표시 제외'는 `[Xn: …]` 를 뺀 값이다.

| 항목 | 단어 수 | 자리표시 제외 | 제목 문장 단어 수 | 한도 | 자리표시 치환 예산 |
|---|---|---|---|---|---|
| Fig 1 | 323 | 308 | 7(명사구) | 350 | XH 수치 문장 40 단어 이하(치환 뒤 348 이하) |
| Fig 2 | 306 | 306 | 18 | 350 | 없음 |
| Fig 3 | 268 | 268 | 19 | 350 | 없음 |
| Fig 4 | 310 | 295 | 16 | 350 | XA 수치 문장 50 단어 이하(치환 뒤 345 이하) |
| Fig 5 | 343 | 343 | 19 | 350 | 없음 |
| Fig 6 | 306 | 291 | 19 | 350 | XG 수치 문장 50 단어 이하(치환 뒤 341 이하) |
| Fig 7 | 327 | 310 | 18 | 350 | XC 수치 문장 40 단어 이하(치환 뒤 350 이하) |
| Table 1 | 136 | 136 | 6(명사구) | 150 | 없음 |

- 모든 초안이 한도(350, 표 150) 안에 있다. 다만 지침 4.1 의 목표(150–250)는 넘는다. 두 가중 CI 정의, 설계 표지, 지도 기재 사항이 들어가기 때문이다. 원고 조립 때 Methods 로 옮길 수 있는 문장(n* 조건, 방법 정의)부터 줄인다.
- 자리표시 치환 규칙: 설명문에는 X 결과의 수치 문장(점추정, CI, 대상 수, Holm 값)만 넣고, 등록 문서의 해석 조각·해석 문장 전문은 해당 결과 절 첫 문단에 둔다. 치환 뒤 350 단어를 넘으면 그 그림의 방법 정의 문장을 Methods 로 옮긴다.

---

## 17. 검토 반영(2026-10-04, 원고·그림·덱 명세 교차 검토)

원고 명세(`paper/manuscript/MANUSCRIPT_SPEC.md`), 이 문서, 덱 스펙(`deck/deck_spec_paper_report.json`)을 대조하고 이 문서를 고쳤다. 설명문 초안의 수치는 원천 CSV 를 다시 열어 확인했다(원고 명세 14절의 42개 값 목록. 이 문서의 값은 모두 일치했고, 라벨 160개 규칙 W 의 2지역 평균만 반올림 전 값으로 +0.18 cm 로 고쳤다).

1. **결과 절 번호**: 머리말의 '결과 절 순서는 재구성안(R1–R8)'을 '지침 4.3 표(원고 명세 4절과 같음)'로 고쳤다. 이 문서의 R5(C4 + C8), R7(독립 지역) 배정이 재구성안 4절과 달랐다.
2. **자리표시**: 설명문과 패널 표의 X 자리표시를 원고 명세 11절의 위치별 문자열로 바꿨다(`[XH: Fig 1e …]`, `[XA: Fig 4d …]`, `[XG: Fig 6c …]`, `[XC: Fig 7e …]`). 결과 절 R3 의 XA 문자열은 등록 문서 0.4 그대로 12절 표에 병기했다.
3. **설명문 첫 문장**: 원고 명세 7절과 한 문장으로 맞췄다. Fig 2 는 AB4·AB5 초록 규칙 문구, Fig 3 은 'lowered', Fig 4 는 C2 README 1.3 의 좁힌 문구, Fig 6 은 주 패널 a·b 의 사후 서술 수(19단어, 원래 문장은 22단어로 20단어 한도를 넘었다), Fig 7 은 '1 cm 미만'과 '기후 격자 단위 편향 보정'으로 바꿨다. 학습기 판정 문장은 Fig 6 패널 c 문장에 남겼다.
4. **Fig 1 설명문**: 흰 원을 'independent new regions'에서 'new regions added after the label tables were fixed'로 바꿨다(FINAL_BATCH 1절: '독립 지역 확인'은 10-04 조사 뒤 지역에만). 자료 판과 Cartopy 판의 빈칸을 `[미확인: …]` 형식으로 바꿨다.
5. **지도 기재 사항**: Fig 5, Fig 6, Fig 7 설명문에 해안선 자료와 Cartopy 판, Fig 6 에 영구동토 구역 자료를 더했다(지침 4.4 의 7, F-15).
6. **단어 예산**: 5의 지도 기재 사항과 3의 Fig 6 제목 변경 뒤 X 치환 예산(Fig 6 50단어, Fig 7 40단어)을 더하면 350 단어를 넘게 되어 두 설명문을 줄였다. Fig 6 은 계산 환경 문장을 Methods M12 로 옮겼고, Fig 7 은 방법 이름을 직접 라벨 이름으로 줄였다. 캐나다 하한선의 축 밖 값(−10.6 에서 −10.8 cm)을 R-25 에 따라 설명문에 적었다. 16절 표를 다시 쟀다.
7. **Fig 7d 경로**: 라벨 0 구간을 연도 정합 Stefan 에서 원천 계수 Stefan(0선, 직접 라벨)으로 바꿨다. claims 4절 표의 첫 권고, 지침 A9 예시, 원고 명세 D1, 덱 S26 이 모두 원천 계수 Stefan 이었다. 연도 정합 Stefan 은 Fig 2 의 n = 0 축에 있으므로 경로에 다시 그리지 않는다(H6). 방법과 풀의 정의, 내용 위험, 대안(y 기준 변경, 혼합 풀)을 D-13, D-14 에 적었고 원고 명세와 덱 스펙이 같은 기본값을 쓴다. 설명문 d 에 '40개부터의 선정 권고는 고정 레시피 대비 비열등(Fig 4c)에 근거한다'를 더했다.
8. **숫자 표기**: 네 자리 수의 쉼표를 지웠다(Fig 2 설명문의 1000 resamples 포함). v2 그림 글자와 claims 표 인용 두 곳은 원문 그대로 두었다.
9. **용어**: 초록의 WF 표기 '(나)'를 원고 명세와 같은 '선택지 B'로 바꿨다.

**지침과 다르게 둔 것**(사용자 결정 표 14절): Fig 1 패널 e(D-1), Fig 4a 정렬 변수(D-8), Fig 4b 대상 수 28(지침 6.4 는 30 이나 C2 README A3 와 원천 CSV 는 28), Fig 6e 내용(D-16), Fig 7a y 축(D-21), XB·XD·XE·XF 의 SI 배치(D-2–D-5).
