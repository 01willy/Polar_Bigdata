# 국문 원고 점검표 (2026-10-05, 4차: XC 채움)

국문판은 영문 Scientific Reports 원고(`paper/manuscript/en/`, 2026-10-05 4차 수정판, XC 까지 채움)의 쌍둥이판이다. 절 순서, 소절, 수치, 그림 PDF, 표 1 본체, 참고문헌을 영문판과 같게 두고 문장을 국문으로 옮겼다(MANUSCRIPT_SPEC 1.2). 새 주장은 없다.

## 1. 산출물

| 파일 | 내용 |
|---|---|
| `main.tex`, `main.pdf` (29쪽, 3.4 MB) | 국문 원고. article 1단, xelatex + xeCJK(`CJKspace=true`), 본문 Noto Serif CJK KR, 제목 Pretendard(대회 보고서 `outputs/report/main.tex` 6–33행 프리앰블을 1단으로 바꿈) |
| `sections/front.tex` | 제목, 저자 자리, 초록, 핵심어(블록 1), 자료 가용성(블록 3), 감사의 글·저자 기여·추가 정보(블록 4) |
| `sections/intro.tex`, `results_a.tex`, `results_b.tex`, `discussion.tex`, `methods.tex`, `legends.tex` | 영문판 같은 이름 파일의 국문판 |
| `build.sh` | 영문판 `references.bib` 병합 확인 → xelatex → bibtex(`sn-nature.bst`) → xelatex 2회. OMP_NUM_THREADS=2, nice 10. 로그 `build/main.log`, `build/main.blg` |

## 2. 컴파일 상태

| 항목 | 결과 |
|---|---|
| 오류 | 0 |
| 경고 | 3줄. xeCJK·ctex 판이 LaTeX 2026-06-01 을 요구하나 설치 커널은 2025-11-01 이다(영문판과 같은 원인, 동작 영향 없음) |
| Overfull·Underfull | Overfull 0건(코드 경로는 `\path` 로 줄바꿈 허용) |
| 빠진 글자 | 0건 |
| BibTeX | 오류 0, 경고 0, 항목 67(영문판과 같음. 서론의 Biskaborn2019 문장을 영문판과 함께 뺐다) |
| 그림 | 영문판과 같은 `outputs/figures/paper/v3_restructure/Fig1–7.pdf`(Fig. 1 04:15:15, Fig. 4 02:49:59, Fig. 5 04:04:38, Fig. 6 05:00:10, Fig. 7 05:37:28 판). 그림 안 글자는 영문이다 |

## 3. 영문판과의 대응 점검

- **수치**: 절 파일마다 영문판과 국문판의 숫자 목록을 대조하였다(인용 키, URL, 코드 경로, 자리표시 제외). 3차에서는 소수점 수와 백분율을 절별로 다시 대조하였다(초록 15, 서론 8, 결과 A 127, 결과 B 111, 고찰 16, 방법 58, 설명문 134개). 4차 대조에서 남은 차이 11개도 모두 국문의 범위 표기('8.14--83.17' 등)에서 오른쪽 값이 범위로 읽힌 것이고 값은 같다. 차이는 두 종류뿐이다. (1) 영문의 수 낱말(ten, five, four, Sixty-eight, Twenty, forty 등)을 국문에서 숫자로 적은 것. (2) 범위 표기('1.9 to 3.0' → '[1.9, 3.0]', '26.8 and 27.0' → '26.8--27.0'). 바뀐 값은 없다.
- **신뢰구간 표기**: 영문의 '95% CI a to b' 는 '95% CI [a, b]' 로 적었다. 값은 같다.
- **자리표시(4차 뒤)**: X 자리표시는 0건이다(XC 6건을 채웠다). PENDING 6건(초록 XE, R5 XE, R7 XF·XC-F3, 고찰 XF·XE)은 영문판과 문자열이 같고 파란 글씨로 보인다. 초록 XC 결정은 주석 `[DECISION: abstract XC sentence]` 에 후보 문장과 함께 두었다. PENDING 표지는 영문판과 같은 영문 문자열로 두었다. DECISION 20건(영문 21건: 영문은 저자·소속 자리를 `\author` 와 `\affil` 두 곳에 두고, 국문 article 판은 `\author` 한 곳에 합쳤다), 미확인 11건(Fig. 1 설명문 2건과 Fig. 7 의 Cartopy 판 1건은 그림 md 와 맞추며 풀었다), MISSING 3건은 영문판과 같은 문자열이다.
- **그림 표시 이름**: 그림 안 글자가 영문이므로 설명문에서 Source Stefan, Recalibrated Stefan, Anchor + residual ML, Direct ML, Year-matched Stefan, Not reached, Error floor 등은 원문 그대로 적었다.
- **표 1**: 본체는 영문판과 같은 생성 파일(`Table1_data.tex`)의 tabular 를 그대로 넣었고(열 이름 영문), 캡션과 표 설명은 국문이다. 국문 열 이름이 필요하면 표 생성 스크립트에 국문 판을 더해야 한다.
- **보충 자료**: SI 는 영문판(`../en/si_main.pdf`)만 있다. 본문의 지시는 '보충 표 S7', '보충 방법 3', '보충 주석 2', '보충 정보' 로 옮겼다.
- **영문판 2차 수정 반영**: 줄인 본문, XA 계열 수(Methods 와 이탈 목록), Fig. 4·Fig. 6 설명문(렌더판과 맞춘 판)을 반영하였다.
- **영문판 4차 수정 반영**: XC 문장(R4 의 XC-1·XC-2·지역 손해·XC-r, R6 의 배치 무작위와 공통 문장, R7 의 XC-F3 PENDING, 고찰 D6 조건 문장, Methods M7), 같은 자리의 줄임(영문 COMPLIANCE 10.2절), Fig. 6·Fig. 7 설명문 재동기화를 같은 자리에 옮겼다. 표지 대응: in new random splits of the same regions → 같은 지역을 다시 무작위로 나눈 분할에서, re-test in reused regions → 재사용 지역의 재검정, selection family → 선택 계열, pooled mean → 풀 평균.
- **영문판 3차 수정 반영**: X 결과 문장(R1 XH, R2 XG, R3·R4·R5 XB, R5 XI·XE, R6 XD, R7 XJ, Methods M7 Algorithm P, 고찰 D5 지도 문장과 D6), 같은 자리의 줄임(영문 COMPLIANCE 9.2절), Fig. 1 설명문 재동기화, Fig. 6c 의 PENDING 표지를 같은 자리에 옮겼다. 표지 대응: designed after earlier results were viewed → 앞선 결과를 본 뒤 설계, partly unblinded → 비맹검 부분 포함, registration deviation → 등록 이탈, not a true extrapolation test → 실제 외삽 시험이 아니다, spatial proxy with warmer blocks of the same period → 같은 시기의 따뜻한 블록을 쓴 공간 대용, licence basis not recorded → 약관 근거 미기록, product values imputed → 제품 결측 대체.

## 4. 용어

MANUSCRIPT_SPEC 1.4 의 국문 정본을 따랐다. 주요 대응은 다음과 같다.

| 영문 | 국문 |
|---|---|
| active-layer thickness (ALT) | 활동층 두께(ALT) |
| source-coefficient / year-matched / recalibrated / local least-squares / soil-property Stefan model | 원천 계수 / 연도 정합 / 재보정 / 현지 최소제곱 / 토양 보정 Stefan 식 |
| residual ML (recalibrated anchor plus residual ML) | 잔차 ML(재보정 앵커 잔차 결합) |
| low-weight residual | 저가중 잔차 |
| physics pseudo-labels (augmentation) | 물리 유사라벨(증강) |
| direct ML, physics-input ML | 직접 ML, 물리 입력 ML |
| selection rule (cross-validation within the target labels) | 선정 규칙(대상 라벨 안 교차검증) |
| label-based correction (recalibration plus residual) | 라벨을 쓰는 보정(재보정과 잔차) |
| lower error / higher error / equivalent / no difference was established | 오차 감소(오차가 낮았다) / 오차 증가(오차가 높았다) / 동등(±0.5 cm 안) / 차이를 확인하지 못했다 |
| internally pre-registered; designed after earlier results were viewed and registered before running | 내부 사전 등록; 앞선 결과를 본 뒤 설계하고 실행 전에 등록 |
| post hoc analysis / description | 사후 분석 / 사후 서술 |
| depends on the split-independence assumption | 분할 독립 가정에 의존 |
| significant before correction only | 보정 전에만 유의 |
| cell-weighted / block-equal | 셀 가중 / 블록 등가중 |
| thawing degree-days (TDD) | 융해 도일(TDD) |
| block holdout / region holdout | 블록 홀드아웃 / 지역 홀드아웃 |
| kNNDM | $k$겹 최근접 거리 정합(kNNDM) 교차검증 |
| Lena Delta, W Russia, E Russia, Central Russia, Tibetan Plateau, North Atlantic | 레나델타, 러시아 서부, 러시아 동부, 러시아 중부, 티베트 고원, 북대서양 |

- **docs/GLOSSARY.md 와 다른 곳**: GLOSSARY 는 ALT 를 '활성층 두께'로 적는다. 원고 명세 1.3(국문 제목)과 1.4(용어 표)는 '활동층 두께'이므로 명세를 따랐다. 하나로 정할 결정이 남았다.
- 영문 기술 용어는 첫 등장에만 괄호로 원어를 붙였다(활동층 두께(active-layer thickness, ALT), 융해 도일(thawing degree-days, TDD), 기계학습(machine learning, ML), 평균 제곱근 오차(root-mean-square error, RMSE), 신뢰구간(confidence interval, CI), kNNDM, GPR, SAR 등).
- 명세 1.2 가 말한 정본 용어 파일 `paper/manuscript/shared/glossary_en_ko.csv` 는 아직 없다. 위 표를 그 파일의 첫 판으로 쓸 수 있다.

## 5. 문체 점검

| 점검 | 결과 |
|---|---|
| ~습니다/~합니다/~입니다 | 0건 |
| '~이지만' 연결 | 0건 |
| em-dash(—) | 0건 |
| en-dash(–) 연결어 | 0건(범위는 LaTeX `--` 로만 씀: 0.5--0.9, 2010--2024 등) |
| 조사 띄어쓰기 | 라틴 문자·숫자·단위에 붙여 썼다('RMSE가', '13.9~cm였고'). 띄어 쓴 예외는 영문판과 같아야 하는 자리표시 문자열 안에만 있다 |
| 문장 종결 | 사실은 ~이다/~한다, 수행은 ~하였다/~확인하였다. 소제목은 명사구(명세 4절·6절 국문 열) |

## 6. 어절 수(참고, 렌더 PDF 기준)

단어 한도는 영문판으로만 판정한다(명세 1.2). 어절 수는 영문판 한도로만 판정하므로 4차에서는 다시 세지 않았다(3차: 본문 합계 4,059 / 3,824).

## 7. 남은 문제

1. 영문판과 같은 자리표시 40건(DECISION 20, 미확인 11, MISSING 3, PENDING 6)이 남아 있다. 영문판에서 채우면 국문판도 같은 자리를 채운다.
2. 표 1 본체의 열 이름이 영문이다(3절).
3. '활동층'과 '활성층'의 용어 결정(4절).
4. 국문 SI 는 만들지 않았다.
5. Fig. 1, 4, 6, 7 이 다시 바뀌면 영문 설명문을 먼저 맞추고 국문을 따라 고친다.
6. main.pdf 는 3.4 MB 로 줄었다(Fig. 1 재작성판).
