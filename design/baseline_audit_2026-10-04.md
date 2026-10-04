# 기준선 시각 품질 감사 (2026-10-04)

새 Sci Rep 원고 그림과 보고 덱이 넘어야 할 기준선을 정한다. 대상은 최근 프로젝트 4개(Polar_Bigdata, RTM_Imaging_CNF, Digital_Rock, EMP)의 발표덱과 논문 그림이다. 판정 기준은 Nature 그림 지침, Sci Rep 투고 지침, 사용자의 과거 지적 기록이다. 마지막 절에 개선 우선순위 15개를 예시 이미지 경로와 함께 정리한다.

렌더 표본 위치: `/tmp/claude-1025/-home-willy010313-Polar-Bigdata/baseline_audit/` (이하 `$B`). `/tmp` 는 지워질 수 있으므로 원본 경로와 쪽 번호를 함께 적고, 재생성 명령은 부록 A에 둔다.

---

## 1. 범위와 방법

| 프로젝트 | 산출물 | 원본 경로 | 검토 표본 |
|---|---|---|---|
| Polar_Bigdata | 최종 발표덱 (23쪽) | `deck/render/permafrost_final.pdf` | 전 쪽 50 dpi 일람, 11쪽 80–90 dpi |
| Polar_Bigdata | 논문 그림 v2 (Fig1–7, Table1) | `outputs/figures/paper/v2/*.pdf` | 8개, 90 dpi |
| RTM_Imaging_CNF | SEG 발표덱 (22쪽), 워크숍 덱 (17쪽) | `deck/seg_talk/exports/seg_talk.pdf`, `deck/workshop_intro/exports/research_overview.pdf` | 일람 2장, 6쪽 80 dpi |
| RTM_Imaging_CNF | 논문 그림 | `results/paper_figures/{workflow_fm, marmousi_realobs_paper, donghae_final_x3000_*}.pdf` | 4개, 70 dpi |
| Digital_Rock | 논문 그림 | `slice_interp_research/paper/figures/` | 7개, 80 dpi |
| Digital_Rock | 덱 (1쪽 카드) | `slice_interp_research/deck/existing_research_card/render/기존연구_소개.pdf` | 1쪽, 80 dpi |
| EMP | 랩미팅 덱 국문 (31쪽) | `EMP/deliverables/2026-05-12_labmeeting/EMP_LabMeeting_2026-05-12_KR.pdf` | 일람, 4쪽 80 dpi |
| 외부 기준 | Gautam 2025 Sci Rep Fig. 2, Ran 2022 ESSD Fig. 4, Jones 2024 Sci Rep Fig. 3·6 | `references/00_core10/`, `references/14_scirep_exemplars/` | 4쪽, 60 dpi |

정량 점검은 세 가지다.
- 글자 크기·문자 수·서체: pdfminer 로 벡터 PDF 의 문자 단위 크기를 읽고, 폭이 180 mm 를 넘는 그림은 180 mm 로 축소했을 때의 크기로 환산했다.
- 덱의 텍스트 벡터 여부: `pdffonts`, `pdfimages -list`.
- 시각 판정: 렌더 이미지를 직접 열람했다.

---

## 2. 판정 기준

### 2.1 외부 지침 (2026-10-04 웹 확인)

웹 요약 도구로 확인했다. 인용 문장은 투고 직전에 원문과 다시 대조한다.

**Nature 그림 지침** (https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/)
- 서체는 sans-serif 이며 Helvetica 또는 Arial 을 권장한다.
- 글자 크기는 5–7 pt 이고, 패널 문자는 8 pt 굵은 직립 소문자(a, b, c)로 쓴다.
- 피할 요소로 배경 격자선, 불필요한 아이콘과 장식 요소, 그림자, 패턴, 색 글씨, 복잡한 이미지 위의 글자, 겹친 글자를 든다.
- 색각 이상을 고려한 팔레트를 쓴다.

**Sci Rep 투고 지침** (https://www.nature.com/srep/author-instructions/submission-guidelines)
- "Use the same typeface in the same font size for all figures in your paper."
- "avoid excessive boxing, unnecessary colour, spurious decorative effects (such as three-dimensional 'skyscraper' histograms)"
- 패널 문자는 굵은 소문자로 쓰고, 글자는 첫 글자만 대문자로 쓴다.
- 디스플레이 항목(그림과 표)은 8개 이하이고, 그림 범례는 350단어 이하이다. 본문은 4,500단어 이하이다.
- 숫자와 단위 사이는 한 칸 띄우고 SI 표기를 따른다. 이미지는 가능하면 300 dpi 로 낸다.
- 그림 폭(mm, 단·2단)은 위 두 페이지에서 확인되지 않았다 [미확인]. 현재 논문 그림 폭 180 mm 가 맞는지는 원고 양식 조사 결과로 확정한다.

### 2.2 사용자 과거 지적

출처는 `deck/deck_spec_final.json` 의 revision_v5–v9 와 메모 journal-grade-visual-bar 이다.
- **금지 요소**: 외곽 프레임과 제목 밴드, 색 구역(틴트) 패널, 연회색 배경 패널, 내용을 담은 카드 상자, 판정 태그 상자, 곡선·사선 화살표, 번호 틱바, 균일 상자 격자, "선+주황 라벨+큰 숫자" 통계 카드, 그림 안의 쪽 번호 상호참조, 그래프에 사후로 붙인 강조 라벨, 글씨 상자형 설명 도식, 장식 아이콘, 과장 문장.
- **상자 허용 대상**: 실자료 이미지와 연산 블록뿐이다. 나머지 구조는 글자 위계와 헤어라인으로 나타낸다.

### 2.3 외부 예시의 실제 수준

| 예시 | 관찰 | 시사점 |
|---|---|---|
| Gautam et al. 2025, Sci Rep 15:42420, doi:10.1038/s41598-025-26586-w, Fig. 2 (`$B/ex_gautam_fig2-06.png`) | 알래스카 ALT 지도 두 장을 상하로 배치했다. 단일 이산 범례를 쓰고 그림 안 글자는 거의 없다. 팔레트(분홍·주황·청·녹·황·흑)는 지각 순서가 없고 경위도 격자와 축척이 없다. | Sci Rep ALT 논문의 그림 수준은 높지 않다. 우리 지도(덱 p17)는 지도 요소에서 이미 앞선다. 차별은 서체, 밀도, 메시지 위계에서 나와야 한다. |
| Ran et al. 2022, ESSD 14:865–884, doi:10.5194/essd-14-865-2022, Fig. 4 (`$B/ex_ran_fig4-09.png`) | 반구 지도 하나가 쪽의 대부분을 차지한다. 경위도 격자, 축척, 범례가 하나씩 있다. 녹–주황 이산 팔레트를 쓴다. | 결과 그림은 공간 분포 하나를 크게 보이고, 수치는 본문과 표로 넘긴다. |
| Jones et al. 2024, Sci Rep 14:8499, doi:10.1038/s41598-024-58998-5, Fig. 6 (`$B/ex_jones_fig6-08.png`), Fig. 3 (`$B/ex_jones_fig3-04.png`) | 개념도는 단면 그림 자체로 과정을 보이고, 글자는 단계명과 기호 범례뿐이다. 상자 안 문장이 없다. Fig. 3 은 변화 지도와 막대 하나로 구성된다. | 개념도의 기준은 "그림이 설명하고 글자는 이름만 붙인다"이다. |

---

## 3. 산출물별 진단

### 3.1 Polar_Bigdata 최종 발표덱 (`permafrost_final.pdf`)

**강점**
- 모든 결과 쪽 하단에 결론 문장 하나를 두고, 제목은 명사형으로 통일했다. (일람 `$B/sheet_polar_deck.png`)
- 지도 요소를 갖췄다. 경위도 격자, 축척, 단위 있는 컬러바, 위치 인셋이 있다(p17 `$B/hi_polar_deck_p17-17.png`, p10 `$B/hi_polar_deck_p10-10.png`). 외부 예시인 Gautam Fig. 2 보다 충실하다.
- Crameri 계열 색(oslo, broc, vik)을 쓰고 붉은 계열을 배제했다.
- 표는 북탭스 헤어라인으로 정리했다(p10, p19, p21 `$B/hi_polar_deck_p21-21.png`).
- 방법부 도식(p6 `$B/hi_polar_deck_p6-06.png`)은 실자료 썸네일, 헤어라인, 수식으로 구성해 상자를 최소화했다.

**약점**
- **래스터 차트**: 그래프와 도식이 모두 래스터 PNG 이다. `pdfimages` 로 래스터 이미지 27개(표지 로고 3개를 빼면 내용 그림 24개, 2–22쪽 전부)가 검출되었고 유효 해상도는 140–301 ppi 이다. `pdffonts` 에는 Pretendard 만 나오므로 차트 글자는 전부 이미지이다. 그 결과 한 쪽 안에 차트 서체(NanumGothic·DejaVu)와 본문 서체(Pretendard) 두 종이 섞이고, 편집과 확대가 불가능하다.
- **한 쪽 안의 글자 크기 불일치**
  - p11 (`$B/hi_polar_deck_p11-11.png`): 왼쪽 그림의 축 라벨과 눈금이 오른쪽 그림보다 약 1.3–1.5배 크다.
  - p13 (`$B/hi_polar_deck_p13-13.png`): 왼쪽 막대그림 글씨가 오른쪽 선그림보다 작다.
  - p14 (`$B/hi_polar_deck_p14-14.png`): 막대 라벨이 왼쪽 본문보다 크다.
  - 원인은 크기가 다른 figsize 로 저장한 그림을 칸에 맞춰 늘이고 줄인 데 있다.
- **띄어쓰기 오류의 반복**: "개선 , 부정확한", "( 표준편차 )", "0.49 cm 에서", "3 차원", "( 순가치 ) 은" 같은 형태가 p11, p14, p17, p19, p22 하단 문장에 반복된다. 자동 공백 삽입의 흔적으로 보이며 조판 품질을 낮춘다.
- **색 밴드 카드의 잔존**: p4 (`$B/polar_deck-04.png`) 는 "목표" 상자 밴드에 3열 카드(회색·남색·청색 제목 밴드)를 얹은 구성이다. v7 금지 목록인 "외곽 프레임+제목 밴드"와 "색 구역"에 해당한다. 금지 목록이 방법부 그림에만 적용되었다.
- **p19 지중온도 지도** (`$B/hi_polar_deck_p19-19.png`)
  - 이산 컬러바의 눈금 간격(4, 3, 2, 1, 0.5, 0, −0.5, −1, −2, −3, −4)이 불균등한데 칸 크기는 같아서 색 거리가 왜곡된다.
  - 컬러바 라벨이 눈금 숫자와 겹친다.
  - 점 자료를 알래스카 전역의 회색 바탕에 찍어서 패널 면적의 대부분이 빈 영역이다.
- **p18 연별 지도** (`$B/hi_polar_deck_p18-18.png`)
  - ALT 상단이 검정으로 포화되어 깊은 구역의 구조가 보이지 않는다.
  - contourf 계단 경계가 보인다.
  - 패널 제목("2013년")이 슬라이드 본문보다 크다.
- **회색 배경 패널의 혼재**: 그래프 배경의 연회색 패널이 흰 슬라이드 위에 섞인다(p13 오른쪽, p21 왼쪽). Nature 지침이 피하라고 한 배경 격자선도 함께 남아 있다.
- **같은 양의 다중 표기**: 막대 길이, 숫자 라벨, 지역 내 점, 기준 파선을 한 그림에 겹친다(p14). 하단 결론 문장이 같은 숫자를 다시 쓴다.
- **사후 래스터 보정**: 채도 1.22배 보정 사본, 겹침을 가린 백패치, 원본 그림 크롭을 썼다(deck_spec revision_v2–v4, v9). p12 생성 스크립트는 소실되었다(revision_v9). 이 그림들은 원본에서 다시 만들 수 없다.

### 3.2 Polar_Bigdata 논문 그림 v2

| 그림 | 문자 수 | 크기 종류 | 5 pt 미만 문자 | 중앙 크기 | 서체 |
|---|---|---|---|---|---|
| Fig1_problem | 1,008 | 6 | 4 | 6.5 pt | Liberation Sans |
| Fig2_label_curve | 1,009 | 10 | 18 | 6.5 pt | Liberation Sans |
| Fig3_physics_use | 1,163 | 13 | 31 | 6.5 pt | Liberation Sans |
| Fig4_min_labels | 1,012 | 14 | 50 | 6.5 pt | Liberation Sans |
| Fig5_placement | 792 | 10 | 3 | 7.0 pt | Liberation Sans |
| Fig6_label0_uncertainty | 1,498 | 19 | 71 | 6.5 pt | Liberation Sans |
| Fig7_deployment | 1,744 | 10 | 6 | 6.5 pt | Liberation Sans |
| Table1_data | 2,575 | 3 | 4 | 6.5 pt | Liberation Sans |

(폭은 모두 180 mm 이다. 5 pt 미만 문자는 대부분 mathtext 첨자이다.)

**강점**
- 폭 180 mm 고정, 벡터 PDF·SVG, sans-serif 단일 서체(Arial 메트릭 호환), 굵은 소문자 패널 문자를 갖춰 Nature 형식과 맞는다.
- 본문 글자의 중앙값이 6.5–7.0 pt 로 Nature 지침의 5–7 pt 안에 있다.
- 결과를 점추정과 신뢰구간(forest plot)으로 보고해 통계 보고가 정직하다.
- 남색·라벤더·회색의 절제된 팔레트를 그림 전체에 일관되게 썼다.

**약점**
- **내부 코드 노출**: L4, L29, AB1–AB10, P4, LGX-N2, LGF-F1, SC1w, SC2w, C2, B2, "(iii) − B4" 같은 내부 코드가 그림에 그대로 나온다(Fig6 `$B/polarfig_Fig6_label0_uncertainty.png` 오른쪽 열, Fig7 `$B/polarfig_Fig7_deployment.png` d). 캡션 없이는 해독할 수 없고, 캡션이 있어도 독자가 대응표를 거쳐야 한다.
- **판정 기호 범례의 반복**: 판정 기호 체계(▼ ○ ◇ △ × †)를 Fig2–Fig7 상단에 2–3줄 범례 블록으로 매번 반복한다. 그림 높이의 약 10–15% 가 범례이다.
- **삼중 부호화**: 같은 판정을 신뢰구간, 기호, 표로 세 번 표기한다(Fig2 오른쪽 아래 판정표 `$B/polarfig_Fig2_label_curve.png`, Fig4 b 아래 표, Fig7 b 판정 격자).
- **글자 과다와 크기 체계 부재**: Fig7 1,744자, Fig6 1,498자이다. 크기 종류는 최대 19종이고, 5 pt 미만 문자가 최대 71개이다.
- **글씨 상자형 도식**
  - Fig7 a: 둥근 상자 5개(T0–T160)에 문장 3줄씩을 넣고 화살표로 이었다.
  - Fig1 c (`$B/polarfig_Fig1_problem.png`): "1 Region holdout / 2 A/B split / 3 Methods and score" 아래에 문장과 식을 나열했다.
- **지배 메시지 부재**: Fig2 는 패널 6개와 판정표, Fig4 (`$B/polarfig_Fig4_min_labels.png`) 는 블록 4개로 이루어지며 패널 크기가 비슷해 핵심 결과가 드러나지 않는다.
- **공간 시각화 부족**: 7개 그림 중 지도는 Fig1 a 하나뿐이다. 프로젝트 규칙(VISUALIZATION, 결과마다 공간 시각화)과 외부 예시(Ran, Gautam 의 지도 중심 구성)에 비해 약하다. Fig1 a 지도의 범례는 그림 상단에 8항목으로 흩어져 있다.
- **해석 부담이 큰 축**: "log scale beyond ±2 cm" 혼합 축(Fig3 c `$B/polarfig_Fig3_physics_use.png`, Fig7 d), 축 꺾임 기호와 "all" 범주를 같은 축에 둔 구성(Fig2, Fig4 b)이 그렇다.
- **그림으로 만든 표**: Table1 (`$B/polarfig_Table1_data.png`) 은 PDF·PNG 그림으로 만들었고 각주가 6개이다. 편집 가능한 원고 표로 바꿔야 하는지는 Sci Rep 표 규정을 확인한다 [미확인].

### 3.3 RTM_Imaging_CNF 발표덱과 논문 그림

**강점**
- 결과 쪽이 실제 탄성파 단면 중심이고 패널 정렬과 공유 축이 맞다(p10 `$B/hi_rtm_seg_p10-10.png`, p13 `$B/hi_rtm_seg_p13-13.png`).
- 단면은 회색조(분야 관례), 불확실성(CV)은 순차형 색으로 구분했다(덱).
- p6 아키텍처 (`$B/hi_rtm_seg_p6-06.png`) 는 실제 입출력 영상 썸네일을 쓰고 연산 블록에만 상자를 써서, 학습 영역과 추론 영역을 파선으로 나눴다.
- p3 개요 (`$B/hi_rtm_seg_p3-03.png`) 는 3열을 세로 구분선으로 나누고, 상자는 연산 블록(셰브런)과 실제 영상에만 썼다. 다만 연회색 배경 패널과 곡선 파선 화살표가 남아 있다.

**약점**
- **AI 순서도 문법**: p4 (`$B/hi_rtm_seg_p4-04.png`) 와 논문용 `workflow_fm` (`$B/rtmfig_workflow_fm-1.png`) 이 해당한다.
  - 파스텔 색 구역(보라·주황·녹)과 제목 밴드 상자를 썼다.
  - 상태 상자를 신호등 색(녹 Update, 적 Maintain, 회 New state)으로 칠했다.
  - 마름모 판정 노드를 넣었다.
  - 사용자 금지 목록과 그대로 겹친다. 원본 폭은 339 mm 이고, 180 mm 로 줄이면 글자 중앙값이 5.0 pt 가 된다.
- **중복 컬러바와 그림 안 각주**: p10 은 패널 4개에 컬러바 4개를 달았고 그중 3개는 같은 양이다. p13 아래에는 3줄짜리 그림 각주가 있다.
- **제목 중복**: 그림 suptitle 이 슬라이드 제목과 겹친다(p13).
- **텍스트 결론 쪽**: p16 (`$B/hi_rtm_seg_p16-16.png`) 은 불릿 8개, 각 2–4줄로 4불릿 규칙을 넘는다.
- **논문 그림 `donghae_final_x3000_*`** (`$B/rtmfig_donghae_final_x3000_uncertainty-1.png`, `$B/rtmfig_donghae_final_x3000_reconstruction-1.png`)
  - CV 에 jet 계열 무지개 색을 써서 전역 규칙을 위반한다.
  - 제목에 실행 ID "[donghae_final_x3000]" 와 내부 변수명 "FM_mean" 이 그대로 나온다.
  - 제목에 em dash 를 썼다.
  - matplotlib 기본 서체(DejaVu)와 기본 파랑을 썼다.
  - 패널 문자를 "(a)" 형태로 제목 문자열에 넣었다.
- **`marmousi_realobs_paper`** (`$B/rtmfig_marmousi_realobs_paper-1.png`)
  - 원본 폭이 333 mm 라 180 mm 로 줄이면 글자 중앙값이 3.8 pt 이고 5 pt 미만 문자가 422개이다.
  - 패널마다 컬러바를 따로 달았고, CV 는 무지개 색이다.

### 3.4 Digital_Rock 논문 그림과 덱 카드

**강점**
- fig01 (`$B/drfig_fig01_problem_taxonomy.png`) 은 아이콘 없이 기하 도형(슬라이스 평면)만으로 문제를 정의한다. 범례는 아래 한 줄이고 문자는 214자이다. 개념 도식의 좋은 기준이다.
- fig03 (`$B/drfig_fig03_qualitative_panel_v2.png`) 은 그룹 헤더, 행 라벨, 공유 컬러바 1개, 같은 크기 타일로 구성된다. 정성 비교 그림의 모범이다.
- fig_cost_pareto (`$B/drfig_fig_cost_pareto.png`) 는 범례 대신 점 옆에 직접 라벨을 달았고, 개선 방향을 표시했다.

**약점**
- **서체 혼재**: STIX(serif), Liberation Serif, Liberation Mono, DejaVu Sans 가 그림마다 다르다. Sci Rep 의 "same typeface in the same font size"와 sans-serif 권고를 모두 위반한다.
- **fig02** (`$B/drfig_fig02_method_overview.png`)
  - 원본 폭이 503 mm 라 180 mm 로 줄이면 글자 중앙값이 5.4 pt 이다.
  - "The reconstruction preserves" 체크 표시 카드, 손실항 칩 상자 8개, 단계 막대 상자가 카드형 AI-tell 이다.
  - 코드 식별자(tri_mean / tri_weuler)를 고정폭 서체로 노출했고 em dash 를 썼다.
- **fig_aggregation_ablation_bars** (`$B/drfig_fig_aggregation_ablation_bars.png`)
  - 막대마다 붙인 숫자 라벨이 서로 겹치고, 회전된 x 눈금 라벨도 겹친다.
  - 범주색 6종에 의미가 없다.
  - 크기 종류가 18종이다.
- **레이더 차트** (`$B/drfig_fig_phase7p10_morphology_radar.png`): 축마다 기준([B1]/[B2]/[B3])이 다르고, 면적이 축 순서에 따라 달라져 정량 비교에 맞지 않는다.
- **fig_failure_map** (`$B/drfig_fig_failure_map_kz_methods.png`): 범례가 자료점을 덮고, 회귀선 라벨(ρ 값)이 축 경계에서 충돌한다.
- **덱 카드** (`$B/dr_card_hi.png`): 회색 카드에 불릿 3개만 있고 그림이 없다.

### 3.5 EMP 랩미팅 덱 (국문)

**강점**
- 시뮬레이션 단면(지형, 갱도, 쉘터)을 실제 형상으로 보이고, 변수 통제 비교(갱도 20 m 와 40 m)를 좌우에 같은 축으로 배치했다(p16 `$B/hi_emp_p16-16.png`).
- 결과마다 출처 경로를 남겨 추적이 가능하다.

**약점**
- **생성형 인포그래픽**: p7 (`$B/hi_emp_p7-07.png`) 은 생성형 이미지 도구로 만든 것으로 보인다. 번호 원, 체크 아이콘, 일러스트 문, 그라데이션 아이콘이 들어 있고, 그 위에 반투명 설명 상자가 겹쳐 그림 일부를 가린다. 감사 대상 중 가장 강한 AI-tell 이다.
- **p8 PML 표** (`$B/hi_emp_p8-08.png`)
  - 적–녹(RdYlGn) 색 지도를 써서 적록 색각 이상 독자가 읽을 수 없다.
  - 칸 안 숫자와 색으로 같은 값을 이중 부호화했다.
  - "production" 장식 라벨과 타원 강조를 붙였다.
- **p16, p21 (`$B/hi_emp_p21-21.png`)**
  - 주황 색 밴드 상자와 회색 해석 카드를 썼다.
  - 파일 경로(results/03_adit_propagation/...)를 본문에 노출했다.
  - 막대 4색(회·청·녹·남)에 범례가 없다.
  - x 라벨을 45° 회전했다.
  - 그래프 제목에 em dash 를 썼다.
- **조판**: 띄어쓰기 오류(", " 앞 공백, "( 도파 경로 )")가 있고, 제목과 부제에 em dash 를 남용했다(p16 "핵심 [em dash] 갱도 길이는", p8 부제 "적정점 [em dash]").
- **단조로운 구성**: 31쪽 대부분이 "그림 + 오른쪽 텍스트 카드" 2단 고정이다(일람 `$B/sheet_emp_kr.png`).

---

## 4. 유지할 강점

새 산출물에서 퇴보하면 안 되는 항목이다.

1. **지도 요소**: 경위도 격자, 축척, 단위 있는 컬러바, 위치 인셋(Polar 덱 p17).
2. **색 규약**: Crameri 지각균일 색(oslo_r, vik, broc, acton)과 붉은 계열 배제(`design/brand_tokens.json`).
3. **논문 그림 형식**: 180 mm 폭 벡터, sans-serif 단일 서체, 굵은 소문자 패널 문자(Polar v2).
4. **불확실성 보고**: 점추정과 신뢰구간 중심의 보고(Polar v2 forest plot).
5. **개념 도식**: 기하 도형만으로 개념을 전달하는 문법(Digital_Rock fig01).
6. **정성 비교 배치**: 그룹 헤더, 공유 컬러바, 같은 크기 타일(Digital_Rock fig03).
7. **표 양식**: 북탭스 헤어라인 표와 하단 결론 한 문장(Polar 덱).

---

## 5. 개선 우선순위 15

순위는 사용자 지적의 빈도와 강도, Sci Rep 심사에서의 영향, 수정 비용을 함께 고려해 정했다. 각 항목에 문제 예시와 측정 가능한 개선 규칙을 붙인다.

### 1. 글씨 상자형 설명 도식 폐지
- **예시**: `$B/polarfig_Fig7_deployment.png` (a, 둥근 상자 5개와 문장), `$B/rtmfig_workflow_fm-1.png` (색 구역, 신호등 상자, 마름모), `$B/hi_rtm_seg_p4-04.png`, `$B/drfig_fig02_method_overview.png` (체크 카드, 칩 상자), `$B/polar_deck-04.png` (색 밴드 3열 카드), `$B/hi_emp_p7-07.png` (생성형 인포그래픽).
- **규칙**
  - 상자는 실자료 이미지와 연산 블록에만 쓴다.
  - 도식 안에 동사가 있는 문장을 넣지 않는다(0개). 단계명은 3단어 이하로 쓴다.
  - 과정은 지도, 단면, 자료 썸네일 같은 그림 요소로 보이고 설명은 캡션으로 옮긴다.
  - 기준 예시는 Jones 2024 Fig. 6 (`$B/ex_jones_fig6-08.png`) 과 Digital_Rock fig01 이다.

### 2. 내부 코드, 실행 ID, 파일 경로를 그림과 슬라이드에서 제거
- **예시**: `$B/polarfig_Fig6_label0_uncertainty.png` (L29, LGF-F1, LGX-N2), `$B/polarfig_Fig7_deployment.png` (AB1–AB10, SC1w), `$B/rtmfig_donghae_final_x3000_reconstruction-1.png` ([donghae_final_x3000], FM_mean), `$B/hi_emp_p8-08.png` (results/... 경로).
- **규칙**
  - 그림에 나오는 기호는 본문에서 정의한 과학 기호 또는 평이한 이름(예: physics baseline, residual ML)으로 한정한다.
  - 레시피·실험 코드 대응표는 SI 표 하나로 옮긴다.
  - 정규식 점검(`\b(L\d+|AB\d+|LG[A-Z]*-?[A-Z0-9]*|SC\d\w*|H\d{2}\w?)\b`, `results/`, `_x\d{3,}`)에서 0건이어야 한다.

### 3. 그림 하나에 지배 메시지 하나, 주 패널 위계
- **예시**: `$B/polarfig_Fig2_label_curve.png` (패널 6개와 판정표, 크기 위계 없음), `$B/polarfig_Fig4_min_labels.png` (블록 4개).
- **규칙**
  - 주 패널이 그림 면적의 40% 이상을 차지하게 하고, 패널은 4–6개 이하로 둔다.
  - 캡션 첫 문장을 결론 문장(assertion)으로 쓴다.
  - 보조 비교는 SI 로 옮긴다. Sci Rep 디스플레이 항목 상한 8개 안에서 본문 그림 수를 먼저 정한다.

### 4. 같은 정보를 두 번 이상 표기하지 않는다
- **예시**: `$B/polarfig_Fig2_label_curve.png`, `$B/polarfig_Fig7_deployment.png` (신뢰구간, 판정 기호, 판정표의 삼중 표기), `$B/hi_polar_deck_p14-14.png` (막대, 숫자 라벨, 점, 결론 문장).
- **규칙**
  - 양 하나에 부호화는 하나만 쓴다.
  - 판정 기호 체계(▼ ○ ◇ △ ×)는 폐지한다. 동등 구간은 음영 띠로 그려 신뢰구간이 띠를 벗어나는지가 그 자체로 보이게 한다.
  - 막대 위 숫자 라벨은 본문에서 인용하는 값 1–2개에만 붙인다.

### 5. 최종 인쇄 크기 기준 글자 크기 체계
- **예시**: `$B/polarfig_Fig6_label0_uncertainty.png` (크기 19종, 5 pt 미만 71자), `$B/drfig_fig02_method_overview.png` (503 mm 원본, 180 mm 환산 5.4 pt), `$B/rtmfig_marmousi_realobs_paper-1.png` (환산 중앙값 3.8 pt), `$B/hi_polar_deck_p11-11.png` (같은 쪽 좌우 축 글씨 크기 차이).
- **규칙**
  - 그림은 최종 폭(단 또는 2단)으로 직접 만든다. 축소 배치를 금지한다.
  - 글자는 5–7 pt 로 쓴다. 크기는 패널 문자 8 pt 굵게, 축 라벨 7 pt, 눈금·주석 6 pt 의 세 단계 이하로 둔다.
  - 첨자 포함 5 pt 미만 문자는 0개여야 한다.
  - 덱도 같은 원칙으로 차트를 슬라이드 배치 크기로 만들고, 한 쪽 안의 축 글씨 크기를 일치시킨다.

### 6. 서체 단일화와 벡터 텍스트
- **예시**: Polar 덱의 내용 그림 24개가 모두 래스터이다(`$B/hi_polar_deck_p13-13.png`, 차트 서체와 Pretendard 혼재). Digital_Rock 는 STIX, Liberation Serif·Mono, DejaVu 가 그림마다 다르다(`$B/drfig_fig03_qualitative_panel_v2.png`, `$B/drfig_fig02_method_overview.png`).
- **규칙**
  - 논문 그림은 Arial 계열(현 Liberation Sans 유지 가능) 한 종만 쓴다.
  - 덱은 Pretendard 한 종을 차트 내부까지 적용한다. 차트는 SVG 같은 벡터로 넣거나 슬라이드 크기 그대로 렌더한다.
  - `pdffonts` 에서 서체 계열이 그림당 1종이어야 한다.

### 7. 공간 결과를 주 그림으로
- **예시**: `$B/polarfig_Fig1_problem.png` (논문 그림 7개 중 유일한 지도). 대비 예시는 `$B/ex_ran_fig4-09.png`, `$B/ex_gautam_fig2-06.png` (지도 중심)이고, 출발점은 `$B/hi_polar_deck_p17-17.png` 이다.
- **규칙**
  - 본문 그림 중 최소 2개는 지도로 만든다(예측 ALT 와 불확실성·AOA, 지역별 오차나 전이 결과의 공간 분포).
  - 지도에는 경위도 격자, 축척, 단위 있는 공유 컬러바를 넣고 종횡비는 물리비로 맞춘다.

### 8. 색 규칙 엄격화
- **예시**: `$B/rtmfig_donghae_final_x3000_uncertainty-1.png` (jet), `$B/hi_emp_p8-08.png` (RdYlGn), `$B/drfig_fig_aggregation_ablation_bars.png` (의미 없는 6색), `$B/hi_polar_deck_p19-19.png` (불균등 이산 눈금), `$B/hi_polar_deck_p18-18.png` (상단 검정 포화).
- **규칙**
  - cmcrameri 와 `brand_tokens.json` 범주색만 쓴다.
  - 이산 컬러바는 등간격으로 만든다.
  - 순차형 색의 양 끝은 순흑과 순백을 피하도록 잘라 쓴다(현 oslo_r 0.12–0.90 규약을 연별 지도에도 적용).
  - 범주색은 3–4색 이하로 쓰고 각 색에 의미를 하나씩 둔다.
  - 색으로만 정보를 전달하지 않는다(마커나 선 종류 병행).

### 9. 범례와 컬러바 배치
- **예시**: `$B/polarfig_Fig3_physics_use.png` (상단 2–3줄 범례 블록), `$B/hi_rtm_seg_p10-10.png` (같은 양에 컬러바 4개), `$B/hi_polar_deck_p19-19.png` (컬러바 라벨과 눈금 겹침), `$B/drfig_fig_failure_map_kz_methods.png` (범례가 자료를 가림).
- **규칙**
  - 계열이 5개 이하이면 선 끝이나 점 옆에 직접 라벨을 단다.
  - 같은 양을 그린 패널은 컬러바 하나를 공유한다.
  - 범례와 라벨이 자료 영역이나 다른 글자와 겹치는 곳은 0곳이어야 한다(렌더 후 육안 확인).

### 10. 장식 요소 제거
- **예시**: `$B/hi_emp_p7-07.png` (아이콘, 그라데이션, 반투명 덮개 상자), `$B/hi_polar_deck_p13-13.png` (회색 배경 패널과 격자선), `$B/hi_emp_p16-16.png` (주황 색 밴드), `$B/dr_card_hi.png` (텍스트 카드).
- **규칙**
  - 배경 격자선, 회색 배경 패널, 색 제목 밴드, 카드, 아이콘, 그림자, 그라데이션을 쓰지 않는다(Nature 지침, Sci Rep "avoid excessive boxing").
  - 필요한 기준선은 축 위의 얇은 회색 참조선 하나로 한정한다.

### 11. 문장과 조판 품질
- **예시**: `$B/hi_polar_deck_p11-11.png`, `$B/hi_polar_deck_p17-17.png`, `$B/hi_polar_deck_p22-22.png` (하단 문장의 " ,", "( … )" 공백, "0.49 cm 에서"), `$B/hi_emp_p16-16.png` (em dash).
- **규칙**
  - 렌더 전에 문자열 점검을 한다. 검사 대상은 공백+쉼표, 괄호 안쪽 공백, "숫자 단위 조사" 사이 공백, em dash(U+2014)·en dash(U+2013)의 접속사 용도이며 모두 0건이어야 한다.
  - 숫자와 단위 사이는 한 칸(SI)으로 쓴다.
  - 문체는 `~/.claude/rules/writing-tone.md` 를 따른다.

### 12. 슬라이드는 주장 문장과 근거 그림으로
- **예시**: `$B/hi_rtm_seg_p16-16.png` (불릿 8개), `$B/hi_polar_deck_p22-22.png` (불릿 5개와 하위 줄, 지표 열), `$B/hi_polar_deck_p2-02.png` (불릿 4개와 하위 문장 4개).
- **규칙**
  - 쪽마다 주장 문장 하나(제목 아래 또는 하단)와 근거 그림 하나를 둔다.
  - 불릿은 4개 이하, 각 1–2줄로 쓴다.
  - 텍스트만 있는 쪽은 표지, 목차, 참고문헌으로 한정한다.

### 13. 원본에서 다시 만들 수 있는 그림만 쓴다
- **예시**: `$B/hi_polar_deck_p17-17.png` (크롭 사용), `deck/deck_spec_final.json` revision_v2·v3·v9 (채도 1.22배 보정 사본, 백패치, p12 생성 스크립트 소실).
- **규칙**
  - 래스터 사후 보정(채도·밝기 보정, 픽셀 덮기, 크롭)을 금지한다.
  - 그림은 모두 `scripts/4_visualization/` 의 스크립트, 벡터 출력, 슬라이드 순서로 만들고, 그림마다 생성 스크립트와 입력 표 경로를 spec 파일에 기록한다.

### 14. 축 설계 단순화
- **예시**: `$B/polarfig_Fig3_physics_use.png` (c, "log scale beyond ±2 cm"), `$B/polarfig_Fig2_label_curve.png` (축 꺾임과 "all" 범주), `$B/hi_polar_deck_p11-11.png` (축 라벨 안의 "가로축 대칭로그(0.25 이하 선형)" 설명).
- **규칙**
  - 한 축에는 척도 하나만 쓴다.
  - 범위가 큰 값은 패널을 나누거나 로그 축으로 통일하고, 척도 설명은 캡션에 둔다.
  - "all" 같은 범주형 끝점은 오른쪽의 좁은 별도 패널로 뺀다.

### 15. 차트 형식 선택
- **예시**: `$B/drfig_fig_phase7p10_morphology_radar.png` (레이더), `$B/drfig_fig_aggregation_ablation_bars.png` (숫자 라벨 막대와 회전 눈금), `$B/hi_emp_p21-21.png` (45° 회전 라벨), `$B/polarfig_Table1_data.png` (그림으로 만든 표).
- **규칙**
  - 다지표 비교는 점 그림(dot plot)이나 소형 다중 패널(small multiples)로 그린다.
  - 범주 이름이 길면 가로 막대나 점 그림을 써서 회전 라벨을 없앤다.
  - 표는 원고의 편집 가능한 표로 쓴다(Sci Rep 표 규정은 확인 필요 [미확인]).

---

## 6. 새 산출물의 합격 기준 (QA 점검 제안)

아래 값은 Nature·Sci Rep 지침과 이번 측정값을 근거로 한 제안값이다. 확정은 `design/journal_grade_style_guide.md` 작성 때 한다.

| 항목 | 현재 최악값 | 기준 |
|---|---|---|
| 최종 크기 글자 범위 | 3.8 pt 중앙값 (marmousi), 5 pt 미만 71자 (Fig6) | 5–7 pt, 패널 문자 8 pt, 5 pt 미만 0자 |
| 그림당 글자 크기 종류 | 19종 (Fig6) | 4종 이하 |
| 그림당 문자 수 (2단 폭) | 1,744자 (Fig7) | 약 600자 이하 (제안값) |
| 서체 계열 (그림당 / 논문 전체) | fig02 2계열 (Liberation Serif·Mono) / Digital_Rock 전체 3계열 (STIX, Liberation, DejaVu) | 그림당 1계열, 논문 전체 1계열 |
| 내부 코드·경로 정규식 일치 | 다수 (Fig6, Fig7, donghae) | 0건 |
| jet·rainbow·RdYlGn 사용 | 3건 | 0건 |
| 덱 차트의 래스터 비율 | 100% (Polar 덱 내용 그림 24개) | 데이터 차트 0% (사진·렌더 영상 제외) |
| 쪽당 불릿 수 | 8개 (RTM p16) | 4개 이하 |
| " ,", "( ", em dash 검출 | 다수 (Polar 덱, EMP) | 0건 |
| 본문 그림 중 지도 수 | 1개 (Polar v2) | 2개 이상 |

---

## 부록 A. 렌더 재생성 명령

```bash
B=/tmp/claude-1025/-home-willy010313-Polar-Bigdata/baseline_audit; mkdir -p $B; cd $B
pdftoppm -r 50 -png /home/willy010313/Polar_Bigdata/deck/render/permafrost_final.pdf polar_deck
for p in 2 6 11 13 14 17 19 22; do pdftoppm -r 90 -png -f $p -l $p /home/willy010313/Polar_Bigdata/deck/render/permafrost_final.pdf hi_polar_deck_p$p; done
for p in 10 18 21; do pdftoppm -r 80 -png -f $p -l $p /home/willy010313/Polar_Bigdata/deck/render/permafrost_final.pdf hi_polar_deck_p$p; done
for f in /home/willy010313/Polar_Bigdata/outputs/figures/paper/v2/*.pdf; do pdftoppm -r 90 -png -singlefile $f polarfig_$(basename $f .pdf); done
pdftoppm -r 50 -png /home/willy010313/RTM_Imaging_CNF/deck/seg_talk/exports/seg_talk.pdf rtm_seg
for p in 3 4 6 10 13 16; do pdftoppm -r 80 -png -f $p -l $p /home/willy010313/RTM_Imaging_CNF/deck/seg_talk/exports/seg_talk.pdf hi_rtm_seg_p$p; done
for f in workflow_fm marmousi_realobs_paper donghae_final_x3000_reconstruction donghae_final_x3000_uncertainty; do pdftoppm -r 70 -png -f 1 -l 1 /home/willy010313/RTM_Imaging_CNF/results/paper_figures/$f.pdf rtmfig_$f; done
P=/home/willy010313/Digital_Rock/slice_interp_research/paper/figures
for f in fig01_problem_taxonomy fig02_method_overview fig03_qualitative_panel_v2 fig_failure_map_kz_methods fig_cost_pareto fig_phase7p10_morphology_radar fig_aggregation_ablation_bars; do pdftoppm -r 80 -png -singlefile $P/$f.pdf drfig_$f; done
pdftoppm -r 80 -png -singlefile "/home/willy010313/Digital_Rock/slice_interp_research/deck/existing_research_card/render/기존연구_소개.pdf" dr_card_hi
pdftoppm -r 50 -png /home/willy010313/EMP/deliverables/2026-05-12_labmeeting/EMP_LabMeeting_2026-05-12_KR.pdf emp_kr
for p in 7 8 16 21; do pdftoppm -r 80 -png -f $p -l $p /home/willy010313/EMP/deliverables/2026-05-12_labmeeting/EMP_LabMeeting_2026-05-12_KR.pdf hi_emp_p$p; done
pdftoppm -r 60 -png -f 6 -l 6 /home/willy010313/Polar_Bigdata/references/00_core10/gautam2025_alaska_alt.pdf ex_gautam_fig2
pdftoppm -r 60 -png -f 9 -l 9 /home/willy010313/Polar_Bigdata/references/00_core10/ran2022_panarctic.pdf ex_ran_fig4
pdftoppm -r 60 -png -f 4 -l 4 /home/willy010313/Polar_Bigdata/references/14_scirep_exemplars/jones2024_postfire_permafrost_stabilization.pdf ex_jones_fig3
pdftoppm -r 60 -png -f 8 -l 8 /home/willy010313/Polar_Bigdata/references/14_scirep_exemplars/jones2024_postfire_permafrost_stabilization.pdf ex_jones_fig6
for s in polar_deck rtm_seg emp_kr; do montage ${s}-*.png -tile 6x -geometry 320x180+4+4 -background '#888' sheet_$s.png; done
```

글자 크기와 문자 수는 pdfminer(`extract_pages`, LTChar.size)로 측정했다. 폭이 180 mm 를 넘는 그림은 180/폭 비율로 환산했다.

## 부록 B. 이번 감사에서 다루지 않은 것
- RTM_Imaging_CNF 의 BK 포스터, EAGE 초록 그림, 영문 대본 PDF 와 EMP 영문판은 표본에서 뺐다. 국문판과 같은 생성기를 쓴 것으로 보고 대표성은 유지된다고 판단했다.
- 외부 기준 예시는 로컬에 있는 3편(Gautam 2025, Ran 2022, Jones 2024)에 한정했다. Nature 계열·TC·ESSD 최신 ALT 논문의 그림 비교는 참고문헌 수집 결과(`references/14_scirep_exemplars/` 등)가 나온 뒤 보완한다.
- Sci Rep 의 그림 폭(mm)과 표 형식 규정은 확인하지 못했다 [미확인].
