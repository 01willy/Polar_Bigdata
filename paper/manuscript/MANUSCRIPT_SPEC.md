# 원고 명세: Scientific Reports 영문 원고와 국문 원고

작성 2026-10-04. 이 문서는 영문 원고(`paper/manuscript/en/`)와 국문 원고(`paper/manuscript/ko/`)를 쓰기 전의 절별 명세다. 원고 문장, 수치, 그림 패널, 새 실험(XA–XJ)의 자리표시, 단어 예산, 참고문헌 계획을 한곳에 둔다. 원고 파일, 그림, 자료, 기존 문서는 고치지 않았다.

**구속 문서**(충돌하면 위가 우선한다)
1. 사용자 전역 지침과 사용자의 직접 결정(2026-10-04): 임시 제목 유지, v2 그림 보존과 재구성 비교 폴더, 초록의 WF 결과는 판정어 없는 수치(선택지 B, 사용자 확인 대기), 저자·소속과 LLM 사용 기재는 [DECISION] 자리표시
2. `design/journal_grade_style_guide.md`(이하 지침). 원고 규칙은 4절, 그림은 2절과 6절, 원고 QA 는 7.3절
3. `design/user_feedback_digest.md`(이하 digest), `design/slide_archetypes.md`, `design/baseline_audit_2026-10-04.md`
4. 주장별 근거 `paper/claims/*/README.md`(수치의 정본), `docs/QA_FINAL_REVIEW_2026-10-02.md`·`docs/QA_FOLLOWUP_2026-10-04.md`의 축소 규칙, `docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md`(XA–XJ 등록, 6절 배치)
5. `docs/research/2026-10-04/scirep_format_and_drafts.md`(이하 scirep_fmt), `scirep_manuscript_exemplars.md`(이하 scirep_ex), `alt_figure_exemplars.md`, `figure_slide_standards.md`, `docs/MANUSCRIPT_RESTRUCTURE_PLAN_2026-10-02.md`(이하 재구성안)

**수치 규칙**
- 이 문서의 결과 수치는 모두 `paper/claims/*/README.md`의 근거 표에서 옮겼다. 각 행에 README 절과 행 번호(예: C1 README 2.3, AB3)를 적었다. 기억이나 산문 문서에서 옮긴 수치는 없다. README 에 없는 값(중앙값, 러시아 W 를 뺀 평균 등)은 만들지 않고 `[MISSING: …]`으로 둔다.
- 값의 형식은 README 와 같다. `Δ [셀 가중 95% CI] / 블록 등가중 Δ [CI]`, 단위 cm, Δ = RMSE(앞) − RMSE(뒤), 음수면 앞 방법의 오차가 작다.
- 원고의 수치는 손으로 옮기지 않는다. `paper/manuscript/shared/numbers.tex`의 매크로를 `build_numbers.py`가 README 에 적힌 원천 CSV 와 행 필터에서 만든다(지침 1.2절, M-09). 매크로는 지침 4.5절의 자릿수 규칙을 적용한다. |Δ| < 1 cm 는 소수 둘째 자리, 그 밖의 cm 값은 소수 첫째 자리, 상관계수는 둘째 자리, 백분율은 정수, CI 끝값은 점추정과 같은 자릿수다. 아래 영문 문장의 수치는 README 의 소수 둘째 자리 값이므로 원고에서는 매크로 값으로 바뀐다.
- 네 자리 수는 쉼표 없이(1000, 3037), 다섯 자리 이상은 쉼표(13,606)를 쓴다. 음수는 U+2212(−)다. 음수가 든 구간은 영문 "−3.04 to −1.88", 국문 "−3.04에서 −1.88"로 쓴다.

**자리표시 형식**
- 새 실험: `[XA: 절 | 등록 표지 | 해석 조각]`. 등록 형식은 `docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md` 0.4절과 scirep_fmt 4.9절을 따른다. 해석 조각은 그 실험의 '사전 고정 해석 문장(조각)' 표의 위치를 가리킨다. 결과가 나오면 그 표의 해당 갈래 문장으로 바꾸고, 다른 문장을 새로 만들지 않는다.
- 사용자 결정: `[DECISION: …]`. 원천에 없는 수치: `[MISSING: …]`. 확인하지 못한 사실: `[미확인]`.
- 자리표시는 원고 초안에 그대로 두고, 제출 전 점검에서 0건이어야 한다.

---

## 1. 원고 전체 구성

### 1.1 영문 원고 순서(Sci Rep Article)

| 순서 | 요소 | 한도(Sci Rep) | 이 원고의 목표 | 근거 |
|---|---|---|---|---|
| 1 | Title | 20단어 이하 | 14단어(임시 제목) | scirep_fmt 2.2, 4.2 |
| 2 | Authors, affiliations, corresponding author | | [DECISION] | 사용자 결정 |
| 3 | Abstract | 200단어 이하, 비구조형, 인용·그림 없음 | 199단어(판 B), 196단어(판 A) | 2절 |
| 4 | Keywords | 6개 이하 | 6개 | 지침 4.1 |
| 5 | Introduction | 본문에 포함 | 700단어, 4문단 | 3절 |
| 6 | Results | 본문에 포함 | 2800단어, 소절 R1–R8 | 4절 |
| 7 | Discussion | 본문에 포함, 소제목 없음 | 850단어, 6문단 | 5절 |
| 8 | Methods | 한도 없음 | 약 4240단어, 소절 15개 | 6절 |
| 9 | Data availability | 본문 끝, References 앞 | 약 250단어 | 6.16절 |
| 10 | References | 60개 권고(엄격 적용 아님) | 71–72개(9절) | scirep_fmt 4.10 |
| 11 | Acknowledgements | 짧게 | [DECISION] | scirep_fmt 2.2 |
| 12 | Author contributions | 참고문헌 뒤 | [DECISION] | 점검표 |
| 13 | Additional information(Competing interests) | | [DECISION] | 점검표 |
| 14 | Figure legends | 그림마다 350단어 이하 | 150–250단어 | 7절 |
| 15 | Table 1 | 한 쪽 이내, 편집 가능, 각주 없음 | booktabs 표, 설명문 150단어 이하 | 지침 3.1 |

본문(Introduction + Results + Discussion)은 4350단어로 계획하고 150단어를 새 실험 문장의 여유로 남긴다(10절). 조판 11쪽은 권고값이다. 지침 4.1절은 현재 계획을 약 16–17쪽으로 추정했다. 첫 컴파일 뒤 쪽 수를 재고, 넘으면 Methods 세부를 Supplementary Methods 로 옮긴다.

### 1.2 국문 원고(쌍둥이판)

- 같은 절 순서, 같은 소절, 같은 그림 PDF, 같은 `refs.bib`와 `numbers.tex`를 쓴다. 그림 설명문만 국문으로 쓴다(지침 4.7).
- 문체는 사실 서술을 ~이다/~한다로, 수행 서술을 ~하였다/~확인하였다로 쓴다. ~입니다/~합니다, '~이지만' 연결, 연결어 대시는 0건이다.
- 조사는 라틴 문자, 숫자, 단위에 붙여 쓴다("13.87 cm에서", "ML의", "RMSE가").
- 정본 용어는 `paper/manuscript/shared/glossary_en_ko.csv` 하나로 관리한다(아직 없다. 원고 작업에서 만든다). 핵심 대응은 1.4절 표다.
- 단어 한도는 영문판으로만 판정하고 국문판은 어절 수를 참고로 기록한다(scirep_fmt 4.4).
- 국문 소제목은 명사구다(지침 4.3 표의 국문 열).

### 1.3 제목과 핵심어

| 항목 | 영문 | 국문 |
|---|---|---|
| 임시 제목 | Permafrost active-layer thickness prediction under sparse observations using physics-based pseudo-label augmentation and machine learning | 희소 관측 조건에서 물리경험식 유사라벨 증강과 기계학습을 이용한 영구동토 활동층 두께 예측 |
| 단어 수 | 14(하이픈 단어를 한 단어로 셈, `wc -w` 측정) | |
| 규칙 점검 | 콜론, 의문문, 금지어('how many', 'beat', 'novel') 없음 | |

- 사용자가 임시 제목을 유지하기로 했고 나중에 바꾼다. 남은 문제 두 가지는 기록만 한다. (1) Sci Rep 은 한 문장 주장형 제목을 권한다. (2) 증강의 이득은 라벨 0 에서만 확인되었다(라벨 10개 동등 AB6, 라벨 40개 0.34 cm 열세, C3 README 1.3 의 6, E21–E23). 제목의 비중과 본문 비중이 다르다. `[DECISION: 최종 제목]`
- 핵심어(6개 이하, 제안): Active-layer thickness; Permafrost; Stefan model; Physics-guided machine learning; Spatial transfer; Sampling design. 국문: 활동층 두께; 영구동토; Stefan 식; 물리 기반 기계학습; 공간 전이; 라벨 배치. `[DECISION: 핵심어]`

### 1.4 방법 이름과 용어(본문에서 방법 약호를 쓰지 않는다)

지침 H3 은 원고 본문에서 P0, P1, R1, D0, D1, W 같은 방법 약호를 금지한다. scirep_fmt 4.7과 FINAL_BATCH 6절은 본문 기호 다섯 개(n, E, P0, P1, R1)를 허용했으나, 2026-10-04 개정 지침이 더 늦고 구속 문서이므로 지침을 따른다. 본문의 수학 기호는 변수(E, $E_0$, $E_n$, n, $n^{*}$, λ, κ, ρ)뿐이다. 가설 번호(L1–L43, AB1–AB10, WF0–WF10, XA–XJ)는 Methods 의 등록 소절과 SI 표에만 둔다.

| 내부 약호 | 영문 원고 이름(첫 등장) | 이후 줄임 | 그림 짧은 이름 | 국문 정본 |
|---|---|---|---|---|
| P0 | Stefan model with the source coefficient | source-coefficient Stefan model | Source Stefan | 원천 계수 Stefan 식 |
| P* | Stefan model with year-matched thawing degree-days | year-matched Stefan model | Year-matched Stefan | 연도 정합 Stefan 식 |
| P1 | Stefan model recalibrated with the target labels (shrinkage κ = 10) | recalibrated Stefan model | Recalibrated Stefan | 재보정 Stefan 식 |
| P2 | Stefan model fitted to the target labels by least squares | local least-squares Stefan model | (SI) | 현지 최소제곱 Stefan 식 |
| P1* | recalibrated soil-property Stefan model | stronger recalibrated baseline | (SI) | 토양 보정 Stefan 식 |
| R1 | recalibrated anchor plus residual ML | residual ML | Anchor + residual ML | 재보정 물리 잔차 결합 |
| R0 | source anchor plus residual ML | low-weight residual (λ = 0.25) | Source anchor + residual | 원천 물리 잔차 결합 |
| D1 | physics pseudo-label augmentation | physics pseudo-labels | Physics pseudo-labels | 물리 유사라벨 증강 |
| D0 | direct ML (no physics) | direct ML | Direct ML | 직접 ML |
| F1k, F1n | physics-input ML | physics-input structure | Physics-input ML | 물리 입력 ML |
| Re | Stefan-CCI mean anchor plus residual | Stefan-CCI anchor | Stefan-CCI anchor | Stefan·CCI 평균 앵커 |
| W | method selection by cross-validation within the target labels | selection rule | Target-label CV selection | 대상 라벨 교차검증 선정 |
| S2, S4 | block stratification; covariate max-min distance sampling | spreading labels | (Fig 5 열 머리) | 블록 층화, 공변량 최대 최소 거리 추출 |
| S6 | sequential selection by prediction variance | active selection | Active selection | 예측 분산 능동 선정 |

| 용어 | 영문 | 국문 |
|---|---|---|
| 4분 판정 | lower error; higher error; equivalent within ±0.5 cm; undetermined (sentence: "no difference was established") | 오차 감소; 오차 증가; 동등(0.5 cm 안에서 같았다); 미결정(차이를 확인하지 못했다) |
| 등록 | internally pre-registered(LG 계열, WRAPUP), designed after earlier results were viewed and registered before running(WF, XA–XJ) | 내부 사전 등록, 결과 열람 뒤 설계(실행 전 등록) |
| 사후 서술 | post hoc description | 사후 서술 |
| 분할 독립 가정 의존 | depends on the split-independence assumption | 분할 독립 가정 의존 |
| 지역 이름 | Lena Delta, Canada, W Russia, E Russia, Central Russia, Alaska, Tibetan Plateau, North Atlantic | 레나델타, 캐나다, 러시아 서부, 러시아 동부, 러시아 중부, 알래스카, 티베트 고원, 북대서양 |
| 그 밖 | active-layer thickness (ALT), thawing degree-days (TDD), source coefficient, recalibration, block holdout | 활동층 두께(ALT), 융해 도일(TDD), 원천 계수, 재보정, 블록 홀드아웃 |

지역 영문 표기("W Russia" 등)와 레나델타의 범위는 사용자 확인 사항이다(지침 2.5, 8절 7). `[DECISION: 지역 이름 표기]`

### 1.5 서술 방향(모든 절에 적용)

- 중심은 라벨 수에 따른 새 지역 워크플로(학습, 평가, 추가 관측)와 물리 증강·물리 잔차 결합의 라벨 수별 능력이다. '라벨 0 에서 직접 ML 이 물리식보다 나쁘다'와 '무작위 분할은 과대평가한다'는 워크플로의 근거로만 쓴다(지침 1.1, MS-02, MS-03).
- 초록 첫 문장은 워크플로이고, 첫 결과 문장은 평가 설계, 라벨 0 절의 첫 문장은 물리 정보의 출처(위약 대조)다. '직접 ML 의 실패'로 시작하는 문단은 0개다(M-08).
- 'safe', 'did not increase error', '손해를 막는다'는 등록된 비열등 판정 없이 쓰지 않는다. 등록 비열등이 있는 곳은 규칙 W(라벨 40개와 160개, 한계 0.5 cm)와 실행 전인 XB-4, XC-2 뿐이고, 그 경우에도 'non-inferior (margin 0.5 cm)'로 쓴다(QA_FINAL_REVIEW 리뷰어 평가, C1 README 5.2, C5 README N2).
- C2 는 'label-based correction (recalibration plus residual)'의 이득으로 좁힌다. 재보정을 넘어선 ML 몫과 계수 오차의 관계는 XA 결과 전까지 '확인하지 못했다'로 쓰고 자리표시를 단다(C2 README 1.3, FINAL_BATCH 0.4).
- WF 실험과 XA–XJ 는 문단과 Methods 에 'designed after earlier results were viewed' 표지를 단다. WF0 위험표와 C2 README 2.3 의 값은 'post hoc description'이고 판정어를 쓰지 않는다.
- 초록의 판정어는 AB1–AB10 에만 쓴다(WRAPUP 1.1 규칙 (a)–(e)). WF 와 XC 결과는 초록에서 판정어 없는 수치로만 쓴다.
- 단정형은 주 CI 와 공통 재표집 CI 의 판정이 같은 대비에만 쓴다. 다르면 'depends on the split-independence assumption'을 병기한다(C1 README 수치 규칙).

---

## 2. 초록

### 2.1 문장 계획

200단어 이하, 7–8문장, 인용·기호 없음. 결과 수치(Δ 와 CI)는 핵심 2개(AB3, AB4)로 한정한다(지침 4.2). 판정어는 AB 대비 문장(S4)에만 있다. WF 결과(S5–S7)는 판정어 없는 수치다(사용자 선택지 B).

| 문장 | 역할 | 내용과 수치 | 판정어 규칙 | 원천 | 단어(판 B) |
|---|---|---|---|---|---|
| S1 | 워크플로와 의의 | 희소 관측 영구동토 지역의 ALT 지도화를 위한 라벨 수별 워크플로, Stefan 식과 ML | 없음 | 재구성안 1절, 지침 4.2 S1 | 22 |
| S2 | 평가의 빈틈 | 학습 지역 안, 라벨 수 한 조건, 재보정 물리 기준선 없음 | 없음 | NOVELTY(재구성안 3절), scirep_ex 5.3 | 16 |
| S3 | 설계 | 지역 다섯 곳(주 4지역과 알래스카)을 100 km 완충과 함께 홀드아웃, 라벨 0개부터 전량, 원천 계수와 재보정 계수의 두 기준선, 내부 사전 등록 | 없음 | D README 1.3, E24, E25 | 29 |
| S4 | 라벨 0–10개(AB) | AB3: 물리 유사라벨 − 섞은 유사라벨 −1.63 [−1.98, −1.04], Holm *P* 0.001. AB4: 재보정 − 원천 계수(라벨 10개) −2.45 [−3.04, −1.88], Holm *P* 0.001, 4지역 가운데 1지역. AB5: 잔차 ML 의 추가 차이 −0.18, Holm *P* 0.136 | AB3 규칙 (a) 방향, AB4 규칙 (a)와 지역 수(J7), AB5 규칙 (b) 'no difference established' | C1 README 2.3, C2 README 2.1 A6, A7 | 47 |
| S5 | 라벨 수십–수백 개(WF, 수치만) | 규칙 W − 고정 잔차 레시피: 라벨 40개 −0.87, 160개 −1.32 cm(초록 표기 −1.3), 2지역(레나델타, 캐나다), 변화는 캐나다(−2.10, −2.72)에서 발생 | 없음(수치만) | C5 README 2.1 E2, E3, 2.2 E6, E7 | 27 |
| S6 | 라벨이 많은 지역(WF, 수치만) | 알래스카 라벨 1000개: 잔차 ML 13.87, 재보정 Stefan 14.41 cm(초록 표기 13.9, 14.4). 라벨 전량에서 오차 제곱 감소의 약 99%가 기후 격자 사이 성분 | 없음(수치만) | C4 README 2.1 E1.R1.n1000, C8 README 2.2 B4 | 29 |
| S7 | 새 라벨의 위치(WF, 수치만) | 지역 내 라벨 100개 블록 층화 − 무작위: 캐나다 −2.54, 레나델타 +1.65 cm(초록 표기 −2.5, +1.7) | 없음(수치만) | C6 README 2.1 A3, A17 | 22 |
| S8 | 다음에 필요한 자료 | 격자 안 공변량 | 없음 | C8 README 1.3(해석) | 7 |

### 2.2 초안(판 B, 기본값, 199단어)

단어 수는 스크래치 파일에서 `wc -w`로 쟀다(괄호 안 구간 포함). 원고 수치는 매크로가 4.5절 자릿수로 바꾸므로 아래는 이미 그 자릿수로 적었다.

> We provide a label-count workflow for mapping active-layer thickness in sparsely measured permafrost regions with the Stefan model and machine learning (ML). ML models are mostly tested within training regions, at one label count, without a recalibrated baseline. We held out five regions with 100 km buffers and compared errors from zero to all target labels against the Stefan model with source and recalibrated coefficients (internally pre-registered). Without labels, Stefan pseudo-labels gave lower error than shuffled ones (−1.6 cm; 95% CI −2.0 to −1.0); with ten labels, recalibration gave lower four-region mean error than the source coefficient (−2.5 cm; −3.0 to −1.9; lower in one region), with no further difference established for residual ML. With 40 and 160 labels, selecting methods by within-target cross-validation changed RMSE by −0.87 and −1.3 cm against a fixed residual recipe (two regions, change from Canada). Within Alaska, residual ML reached 13.9 against 14.4 cm RMSE for recalibrated Stefan (1000 labels); about 99% of the squared-error reduction with all labels lay between climate grid cells. Spreading 100 labels over blocks changed RMSE against random sampling by −2.5 cm in Canada and +1.7 cm in the Lena Delta. Sub-grid covariates are the data needed next.

국문(판 B):

> 측정이 희소한 영구동토 지역에서 Stefan 식과 기계학습(ML)으로 활동층 두께를 지도화하는 라벨 수별 워크플로를 제시한다. ML 모형은 대부분 학습 지역 안에서, 라벨 수 한 조건으로, 재보정한 물리 기준선 없이 시험되었다. 다섯 지역을 100 km 완충대와 함께 학습 원천에서 빼고, 대상 라벨 0개부터 전량까지 원천 계수와 재보정 계수의 Stefan 식 대비 오차를 비교하였다(내부 사전 등록). 라벨이 없을 때 Stefan 유사라벨은 순서를 섞은 유사라벨보다 오차가 작았고(−1.6 cm, 95% 신뢰구간 −2.0에서 −1.0), 라벨 10개에서는 재보정 계수가 원천 계수보다 4지역 평균 오차가 작았다(−2.5 cm, −3.0에서 −1.9, 오차가 작아진 지역은 1곳). 잔차 ML의 추가 차이는 확인하지 못했다. 라벨 40개와 160개에서 대상 라벨 안 교차검증으로 방법을 선정하면 고정 잔차 레시피 대비 RMSE가 −0.87 cm와 −1.3 cm 달라졌다(2지역, 변화는 캐나다에서 발생). 알래스카 안에서 라벨 1000개일 때 잔차 ML의 RMSE는 13.9 cm, 재보정 Stefan 식은 14.4 cm였고, 라벨 전량에서 오차 제곱 감소의 약 99%는 기후 격자 사이 성분이었다. 라벨 100개를 블록에 퍼뜨리면 무작위 추출 대비 RMSE가 캐나다에서 −2.5 cm, 레나델타에서 +1.7 cm 달라졌다. 다음에 필요한 자료는 기후 격자 안에서 변하는 공변량이다.

### 2.3 비교판(판 A, WF 제외, 196단어)

scirep_fmt 4.3과 QA_FOLLOWUP 5.5 가 두 판을 함께 만들어 고르게 하라고 권했다. 판 A 는 결과 수치가 AB3, AB4 둘뿐이라 지침 4.2 의 '핵심 수치 2개 이하'를 그대로 지킨다.

> We provide a label-count workflow for mapping active-layer thickness (ALT) in sparsely measured permafrost regions with the Stefan thaw model and machine learning (ML). ML models of ALT are usually evaluated within training regions, at one label count and without a recalibrated physical baseline. We held out four Arctic regions with 100 km buffers and compared errors from zero to all target labels against the Stefan model with source and recalibrated coefficients, in an internally pre-registered analysis. Without labels, Stefan pseudo-labels gave lower error than shuffled ones (−1.6 cm; 95% CI −2.0 to −1.0), and no difference was established between a physics-model ensemble and the source-coefficient model. With ten labels, coefficient recalibration gave lower four-region mean error (−2.5 cm; −3.0 to −1.9; one of four regions), no further difference was established for residual ML, and residual learning gave lower error than physics outputs used as ML inputs. With all labels, residual ML gave lower four-region mean error than the source-coefficient model, mainly from one region, and lower error than the recalibrated model by less than 0.5 cm. The workflow therefore uses physics at every label count and assigns ML a small correction once labels allow recalibration.

판 A 의 판정어 근거: AB2 미결정(규칙 (d), C1 README 2.7), AB7 우세(규칙 (a), C3 README E06), AB8 우세(규칙 (a), 러시아 W 지배 병기, C2 README A8, A11), AB9 우세·0.5 cm 미만(C2 README A9).

### 2.4 초록의 열린 결정

- `[DECISION: 초록 판 A 또는 B]`. 사용자는 판 B 를 잠정 선택했다.
- 판 B 는 지침 4.2 의 '결과 수치 2개 이하'(지침 8절 4, KO-04)와 충돌한다. Δ·CI 쌍은 2개로 지켰으나 WF 결과 수치 7개(−0.87, −1.3, 13.9, 14.4, 99%, −2.5, +1.7)가 더해진다. 초록 수치는 지침 4.5 자릿수(|Δ| < 1 cm 둘째 자리, 그 밖 첫째 자리)로 적었다. 판 B 는 8문장이고 지침 4.2 틀은 7문장이다. `[DECISION: 판 B 의 WF 수치 개수 허용 여부]`
- XC 결과는 초록에서 수치만 쓰거나 뺀다(FINAL_BATCH 7.1). 수치를 넣으면 S7 이나 S8 을 대신해 200단어를 지킨다. 자리표시: `[XC: Abstract S8 | 결과 열람 뒤 설계, 재사용 지역의 재검정(비맹검 부분 포함) | 2.3 초록 규칙: 수치만, '개발에 쓴 지역을 다시 나눈 분할', 지역 2곳, 지역별 방향(우세 k, 열세 m)]`
- S8 은 해석이다(C8 README 1.3 의 '격자 안 변동을 줄이려면 격자 안 해상도의 입력이 필요하다(해석)'). XE 결과에 따라 바뀐다: `[XE: Abstract S8 | 결과 열람 뒤 설계 | 2.5 사전 고정 해석 문장(첫째 또는 넷째 갈래)]`

---

## 3. Introduction(700단어, 4문단)

| 문단 | 역할 | 목표 단어 | 담을 내용과 수치(원천) | 인용(키는 9절) |
|---|---|---|---|---|
| I1 | 문제와 수치 맥락 | 170 | ALT 의 정의와 쓰임, 영구동토 온도 상승. 관측 희소: 이 연구의 자료에서 지역별 라벨 행은 30(러시아 동부)에서 13,606(알래스카)이다(D README E10–E14). Stefan 계수 E 는 지역마다 다르다(셀 E 의 기하 평균 레나델타 1.31, 러시아 서부 2.51, 티베트 11.03 cm (°C d)$^{-1/2}$, D README E38, Fig. 1b). 새 지역의 지도는 다른 지역에서 맞춘 모형에 기댄다 | Biskaborn 2019, Streletskiy 2025, Nelson 1997, Stefan 1891 |
| I2 | 평가 조건의 빈틈 | 210 | 선행 연구를 모형 종류가 아니라 평가 조건으로 묶는다. (a) 물리식 출력 입력, 계수의 ML 추정, 규칙 유사라벨, 잔차 학습의 선례가 있어 기법의 새로움은 주장하지 않는다. (b) ALT 평가는 무작위 분할(Gautam 2025: 무작위 70/30 1회, 시험 R² 랜덤 포레스트 0.24, Stefan 0.54, QA_FOLLOWUP 3.3 원문 확인 수준 A) 또는 거리 블록이다. (c) 공간 자료에서 무작위 검증의 낙관(Ploton 2020, Meyer & Pebesma 2022)과 확률 표본이면 무작위 검증이 맞다는 반론(Wadoux 2021)을 함께 적는다. (d) 인접 분야에는 라벨 수 함수와 보정 물리 기준선 평가가 있다. 'to our knowledge'로 한정한 빈틈 문장 하나 | Aalto 2018, Karjalainen 2019, Ran 2022a, Ran 2022b, Zhang 2024, Pilyugina 2025, Wang 2025, Gay 2026, Read 2019, Jia 2021, Willard 2022, Gautam 2025, Roberts 2017, Ploton 2020, Meyer & Pebesma 2022, Wadoux 2021, Pool 2019, Willard 2021, Feng 2023, O'Malley 2026, Portes 2026 |
| I3 | 번호 질문 3개 | 120 | (1) 대상 라벨이 없을 때 물리 정보는 ML 에 무엇을 더하고 그 이득은 어디서 오는가(결과 절 R2). (2) 라벨이 10개에서 수천 개로 늘 때 ML 은 같은 라벨로 재보정한 Stefan 식에 얼마를 더하고, 그 이득은 어느 공간 규모에서 오는가(R3, R5). (3) 라벨 수가 늘 때 방법은 어떻게 선정하고 새 라벨은 어떤 절차로 배치하는가(R4, R6) | 없음 |
| I4 | 설계와 결과 요약 | 200 | 설계 문장(지역 홀드아웃 + 100 km 완충, 0.5° 블록 절반 분할, 라벨 격자 0, 3, 10, 40, 160, 320, 1000, 전량, 두 기준선, 지역 내 블록 홀드아웃, 4분 판정, 내부 사전 등록, 음성 결과 포함). "We find …" 문장에 수치 1–2개(AB3 −1.63, AB4 −2.45, 초록 2.1 S4 와 같은 대비). 기여는 평가 설계, 그 결과(음성 포함), 워크플로라고 쓰고 'to our knowledge'로 한정한다(MS-04) | Read 2019, Jia 2021, Steyerberg 2004 |

- 쓰지 않는 것: 'first', 'novel', 'how many', 질문형 제목 흉내, '무작위 분할은 과대평가한다'를 서론의 중심 주장으로 두는 것.
- 영문 첫 문장 틀(scirep_ex 6.2 #7): "However, because measurements in a new permafrost region are few, the error of ALT maps there depends on models fitted elsewhere."
- I4 끝 문장 틀(scirep_ex #11): "We find that, without target labels, Stefan pseudo-labels lowered error relative to shuffled pseudo-labels by 1.63 cm, and that ten labels used to recalibrate the Stefan coefficient lowered the four-region mean error by 2.45 cm, with most later gains from ML arising at the scale of climate grid cells."
  - 셋째 구절('most later gains … climate grid cells')은 알래스카 지역 내 결과에 한정된다(C8 README 1.2 의 1). 원고에서는 "within Alaska"를 넣는다.
- 국문 I1–I4 는 같은 순서와 수치로 옮긴다. 번호 질문은 "(1) …인가. (2) …인가. (3) …인가."로 쓰고 수사 의문문이 아니라 연구 질문으로 둔다.

---

## 4. Results(2800단어, 소절 R1–R8)

소제목은 콜론 없는 명사구다(지침 4.3, R-17). 각 소절 첫 문장은 주장과 수치다. 틀: "[Method] [reduced/increased] [metric] relative to [baseline] by [Δ] cm (95% CI [a] to [b]; [n] labels; [k] regions) (Fig. [N][panel])." 문단 순서는 주장과 수치, 근거 세부, 조건과 예외, 다음 절 연결이다. 기전 해석은 Discussion 으로 보낸다. 유의하지 않은 결과도 같은 형식으로 쓴다("We did not find evidence that …").

| 절 | 영문 소제목 | 국문 소제목 | 주장 | 단어 | 그림 |
|---|---|---|---|---|---|
| R1 | Evaluation design and validation ladder | 평가 설계와 검증 사다리 | C7(보조), D | 300 | Fig 1, Table 1 |
| R2 | Transfer without target labels | 대상 라벨 없는 전이 | C1 | 400 | Fig 6a–c, Fig 3b, Fig 2 |
| R3 | Coefficient recalibration with three to ten labels | 라벨 3–10개의 계수 재보정 | C2(좁힘), C3 | 450 | Fig 2, Fig 3a,c,d, Fig 4a,b,d |
| R4 | Method selection with tens to hundreds of labels | 라벨 수십–수백 개의 방법 선정 | C5, C3 | 350 | Fig 4c, Fig 3c, Fig 7d,e |
| R5 | Gains within label-rich regions | 라벨이 많은 지역 안의 이득 | C4, C8 | 450 | Fig 7a–c |
| R6 | Placement of new labels | 새 라벨의 위치 | C6 | 400 | Fig 5, Fig 7e |
| R7 | Independent regions | 독립 지역 | C1·C4 일반화 | 250 | Table 1, Fig 1a |
| R8 | Prediction intervals | 예측 구간 | 보조 | 200 | Fig 6d,e, Supplementary Table S12 |

재구성안 4절의 순서(R5 충분 라벨, R6 관측 위치, R7 C8, R8 예측 구간)와 다른 점: 지침 4.3 은 C8 을 R5 에 합치고 R7 을 독립 지역으로 두었다. 이 명세는 지침을 따른다. 라벨 구간 이름은 시험 격자 값으로 쓴다(0, 3–10, 40–160, 320–1000, 전량. 'one to ten'은 쓰지 않는다).

### 4.1 R1 Evaluation design and validation ladder(300단어)

**첫 문장(영문)**: "All methods were compared on one protocol in which each target region and a 100 km buffer were removed from the source data, labels were drawn from half of the target's 0.5° blocks, and errors were scored on the other half, for label counts from zero to all available rows (30 to 13,606 rows per region) (Fig. 1c,d; Table 1)."

**둘째 문장(영문)**: "Averaged over five regions with equal weight, the RMSE of direct ML increased from 22.91 cm under random cell splits to 33.25 cm under region holdout (difference 10.34 cm; 95% CI 7.29 to 12.84; block-equal 6.15 cm, 4.70 to 7.57), whereas the least-squares Stefan model stayed between 26.81 and 26.96 cm (Fig. 1e)."

**국문**: "모든 방법은 대상 지역과 그 주변 100 km를 학습 원천에서 빼고, 대상의 0.5° 블록 절반에서 라벨을 뽑고 나머지 절반에서 채점하는 한 시험지에서 비교하였다. 다섯 지역을 같은 무게로 평균하면 직접 ML의 RMSE는 무작위 셀 분할 22.91 cm에서 지역 홀드아웃 33.25 cm로 커졌고(차 10.34 cm, 95% 신뢰구간 7.29–12.84), 최소제곱 Stefan 식은 26.81–26.96 cm였다."

**근거 수치**

| 항목 | 값 | 판정과 표지 | 원천 |
|---|---|---|---|
| 대상 라벨 행 / 1 km 위치 / 0.5° 블록 | 알래스카 13,606 / 343 / 74, 레나델타 3037 / 201 / 20, 캐나다 750 / 86 / 39, 러시아 서부 31 / 29 / 21, 러시아 동부 30 / 30 / 21 | 서술 | D README E10–E14 |
| 라벨 수 격자, 분할, 추출, 대상 수 | 0, 3, 10, 40, 160, 320, 1000, 전량; 분할 5회, 추출 5회, 대상 27개, 방법 12종 | 사전 등록 | D README E24, E25 |
| 원천 가운데 알래스카 셀 비율 | 0.78–0.94 | 서술 | D README E35 |
| 원천 계수 $E_0$ | 1.50–1.62 cm (°C d)$^{-1/2}$ | 서술 | D README E37 |
| 오차 하한 | 알래스카 11.31, 레나델타 14.83, 캐나다 17.90 cm | 보고 규칙(판정 없음) | D README E44 |
| 직접 ML 열화 V-G − V-R(5지역 층화) | +10.34 [7.29, 12.84] / +6.15 [4.70, 7.57] | 열세(보조 행, 재현(비맹검)) | C7 README 2.2 A1 |
| 같은 대비의 지표 | V-R 22.91, V-G 33.25 cm | 지표 | C7 README A2, B1, B8 |
| 등록 대비 V-C500 − V-R | +8.22 [5.65, 10.38] / +4.49 [3.15, 5.85] | 열세, 단조 열화 L20 지지 | C7 README 2.3 B9, B10 |
| 최소제곱 Stefan 의 사다리 단별 RMSE | 26.81–26.96 cm | 서술 | C7 README 2.5 D1 |
| 알래스카 채점 방식별 RMSE(seed 3개 점추정) | 직접 CatBoost 11.55(무작위 셀)에서 17.91(kNNDM), Stefan 14.24–14.46, 물리 잔차 12.77–13.70 | 서술(CI 없음) | C7 README 2.6 E1–E5 |
| 시험 셀에서 가장 가까운 학습 셀까지 중앙값 | 무작위 셀 0.004 km, kNNDM 53.50 km. 예측 격자에서 가장 가까운 라벨까지 81.57 km | 서술 | C7 README 2.6 E1, E5, E7 |

**문단 구성**: (1) 설계 문장. (2) 두 기준선과 공정성 원칙(ML 이 라벨을 쓰면 물리식도 같은 라벨을 쓴다). (3) 오차 하한. (4) 검증 사다리 수치와 알래스카 방식별 수치. (5) 다음 절 연결.

**조건과 쓰지 않을 문장**: 'reversal', 'ranking reverses', 'most honest estimate', 'pooled data of five regions'(MEAN5 값은 지역 등가중 평균), '학습·채점 거리 순서로 단조 증가'(설계상 분리 단계 순서로만 쓴다), 'our scoring is not too strict' 같은 평가어(C7 README 5.2). kNNDM 은 "mimics the distance between map cells and labels"까지만 쓴다(FINAL_BATCH 0.4). L19 의 확인적 판정은 '지지하지 않음'이고 열화 크기는 보조 행이다.

**새 실험**: `[XH: R1 | 결과 열람 뒤 설계, 서술(판정어 없음). L19, L20 은 등록 | 2.8 사전 고정 서술 규칙]`. 채울 내용(FINAL_BATCH 2.8): 무작위 셀에서 kNNDM 으로 갈 때 직접 ML a cm, 재보정 Stefan b cm 증가(지역별 수치와 CI), 무작위 수치 병기, Wadoux 2021 반론과 비확률 표본 병기. XF 는 FINAL_BATCH 6절에 따라 Supplementary Table 에만 두고 Table 1 에 행을 더하지 않는다(그림 명세 D-5).

### 4.2 R2 Transfer without target labels(400단어)

**첫 문장(영문)**: "Without target labels, pseudo-labels from the Stefan model reduced RMSE relative to shuffled pseudo-labels by 1.63 cm (95% CI 1.04 to 1.98; block-equal 0.57 to 1.51; four regions; Holm-adjusted *P* = 0.001), and in a post hoc count, fewer of 30 targets lost more than 2 cm against the source-coefficient Stefan model with physics pseudo-label augmentation (2 targets) than with direct ML (18 targets) (Fig. 3b; Fig. 6a,b)."

**국문**: "대상 라벨이 없을 때 Stefan 식으로 만든 유사라벨은 순서를 섞은 유사라벨보다 RMSE를 1.63 cm 줄였다(95% 신뢰구간 1.04–1.98, 4지역, Holm 보정 P 0.001). 사후 서술로 30개 대상 가운데 원천 계수 Stefan 식보다 2 cm 넘게 나빠진 대상은 물리 유사라벨 증강 2개, 직접 ML 18개였다."

**근거 수치**

| 항목 | 값 | 판정과 표지 | 원천 |
|---|---|---|---|
| 물리 유사라벨 − 섞은 유사라벨, 라벨 0 | −1.63 [−1.98, −1.04] / −1.05 [−1.51, −0.57] | 우세, Holm *P* 0.001, 공통 CI 도 우세(단정형 가능), 재현(비맹검) | C1 README 2.3 AB3 |
| 위약 대조의 다른 행 | 상수 위약 두 종 −1.68(const_t), −3.63(const_src), √TDD 선형 −0.48(tddlin, 크기 0.5 cm 미만), 라벨 10개 √TDD 선형 −0.08(동등) | L15 '혼재' | C1 README 2.3 |
| 위험표(30대상, 독립 지역 7) | 2 cm 넘게 악화: 직접 ML 18, 잔차 가중 1.0 13, 저가중 잔차 5, 물리 유사라벨 증강 2, 증강 + 앵커 + 잔차(λ 1.0) 1 | 사후 서술, 판정어 없음 | C1 README 2.5 |
| 같은 대상의 CI 기준 악화 수 | 직접 ML 14, 증강 8, 저가중 잔차 6, 증강 + 앵커 + 잔차 5 | 사후 계산 | C1 README 2.8 의 3 |
| 직접 ML − 원천 계수 Stefan, 두 CI 판정이 같은 학습기 5종 | 랜덤 포레스트 +2.63, MLP +2.45, 다중 헤드 MLP +2.44, 축소 FT-Transformer +4.13, RealMLP +5.21 | 열세(단정형) | C1 README 2.1 #4, #7–#10 |
| 주 학습기 CatBoost | +2.25 [1.10, 3.42] / +1.62 [0.52, 2.69] | 열세, Holm *P* 0.023, 분할 독립 가정 의존(공통 CI 미결정) | C1 README 2.1 #1(AB1) |
| 나머지 4종(CatBoost 두 설정, TabPFN v2, TabICL v2) | +0.85에서 +1.94 | 열세, 분할 독립 가정 의존. TabPFN v2 는 보정 전 유의(Holm 0.1168) | C1 README 2.1 #2, #3, #5, #6a, #6b |
| 직접 ML(CatBoost) − 연도 정합 Stefan | +3.24 [1.86, 4.29] / +2.03 [0.90, 3.12] | 열세(두 CI 일치), Holm *P* 0.0050, 보조 가설, 앵커 변형 비맹검 | C1 README 2.7 |
| 연도 정합 Stefan − 원천 계수 Stefan | −1.00 [−1.35, −0.32] / −0.41 [−0.74, −0.08] | 우세, 분할 독립 가정 의존 | C1 README 2.7 |
| 물리식·제품 앙상블 − 원천 계수 Stefan | −2.73 [−3.33, −0.91] / +0.05 [−0.94, 1.01] | 미결정(AB2) | C1 README 2.7 |
| 저가중 잔차(λ 0.25) − 원천 계수 Stefan | TabPFN −0.31, CatBoost −0.05, TabICL −0.27 | 동등, 분할 독립 가정 의존 | C1 README 2.4 |
| 예외: 알래스카 증강 | +1.79 [0.37, 2.57] / +1.04 [0.34, 1.85] | 대상 곡선 worse | C1 README 2.6 |
| 예외: 티베트 직접 ML | −59.98 [−62.90, −57.15] | improve, 예측된 결과(비맹검), 교차 환경 | C1 README 2.2 |
| 예외: TabICL v2 직접 ML, 라벨 10개 | −2.91 [−3.58, −1.11](4지역, 분할 독립 가정 의존), +5.98(알래스카 포함 3지역) | 한계 행 | C1 README 2.2 |

**문단 구성**: (1) 위약 대조 문장(첫 문장). (2) 위험표(사후 서술, 2 cm 문턱과 CI 기준 수를 함께). (3) 학습기 문장: 두 CI 판정이 같은 5종은 단정형, 나머지 5종은 분할 독립 가정 의존 병기. 비교 상대를 연도 정합 Stefan 식으로 바꾸어도 직접 ML 의 오차가 컸다. (4) 예외(알래스카 증강, 티베트, 라벨 10개 TabICL). (5) 다음 절 연결("We next asked what the first ten labels changed.").

**쓰지 않을 문장**: 'safe', 'no-loss choice', '물리 증강과 저가중 잔차는 손해를 줄였다'(단정형), 'all ten learners'(단정형), 'tabular foundation models do not surpass physics'(계열 일반화), 'networks were sufficiently tuned', '30 independent regions', '저가중 잔차는 물리식과 동등하다'(단정형)(C1 README 5.2).

**새 실험**: `[XG: R2 | 결과 열람 뒤 설계, 등록 이탈(WRAPUP 10) | 2.7 사전 고정 해석 문장 XG-1, XG-2, XG-3, 공통 문장]`. 채울 내용: XG-1(라벨 0, 기존 지도 원값 − 원천 계수 Stefan), XG-3(같은 라벨로 재보정한 지도 대비 재보정 앵커 잔차), 공통 문장(ALT 정의 차이, 마스크 전후 원천 계수 Stefan RMSE, 등록 이탈 표지). 라벨 0 워크플로 행의 비교 상대가 바뀔 수 있다(C1 README 6절).

### 4.3 R3 Coefficient recalibration with three to ten labels(450단어)

**첫 문장(영문)**: "With ten target labels, recalibrating the Stefan coefficient reduced the four-region mean RMSE relative to the source coefficient by 2.45 cm (95% CI 1.88 to 3.04; block-equal 1.87 to 3.30; Holm-adjusted *P* = 0.001), with lower error in one of four regions (W Russia, 11.96 cm) and higher error in Canada (2.26 cm) (Fig. 2a)."

**둘째 문장(영문)**: "Adding residual ML to the recalibrated anchor changed RMSE by a further −0.18 cm (95% CI −0.53 to −0.01; significant before correction only, Holm-adjusted *P* = 0.136)."

**국문**: "라벨 10개로 Stefan 계수를 재보정하면 4지역 평균 RMSE가 원천 계수보다 2.45 cm 작았다(95% 신뢰구간 1.88–3.04, Holm 보정 P 0.001). 오차가 작아진 지역은 4지역 가운데 러시아 서부 하나였고(11.96 cm), 캐나다에서는 2.26 cm 커졌다. 재보정 앵커에 잔차 ML을 더한 추가 변화는 −0.18 cm였고 보정 전에만 유의하였다."

**근거 수치**

| 항목 | 값 | 판정과 표지 | 원천 |
|---|---|---|---|
| 재보정 − 원천 계수, 라벨 10(AB4) | −2.45 [−3.04, −1.88] / −2.62 [−3.30, −1.87] | 우세, Holm *P* 0.001, 지역 행 우세 1/4. 'S-a 열람 뒤 인용' 표지(Methods 등록 소절) | C2 README 2.1 A6, C3 README E36, E36a |
| 지역 행 | 레나델타 +0.22, 캐나다 +2.26 [0.71, 2.94], 러시아 서부 −11.96 [−13.62, −10.53], 러시아 동부 −0.34 | 미결정, 열세, 우세, 미결정 | C3 README E36a |
| 잔차 ML − 재보정, 라벨 10(AB5) | −0.18 [−0.53, −0.01] / −0.29 | 우세(보정 전), Holm *P* 0.136, 크기 0.5 cm 미만, 분할 독립 가정 의존 | C2 README A7, C3 README E36 |
| 라벨을 쓰는 보정의 이득과 라벨 10개 편향 크기 | ρ 0.38 [0.17, 0.55], 28대상, 10,000회, 단측 *p* 0.0002 | 지지, 결과 열람 뒤 설계, 확인적 아님 | C2 README 2.1 A3 |
| 부호 있는 편향과 이득 | ρ −0.01 [−0.17, 0.18] | 보조 | C2 README A4 |
| 재보정 이득과 최선 ML 이득 | ρ 0.66, 30대상 | 사후 서술, 최선 ML 사후 선택 | C2 README A1 |
| 재보정을 넘어선 ML 몫과 계수 오차 | 정의에 따라 ρ −0.08에서 0.39, CI 없음 | 사후 점검, '확인하지 못했다' | C2 README 2.3 |
| 러시아 서부의 분해(라벨 전량) | 원천 계수 대비 감소 13.98 cm 가운데 13.50 cm 가 재보정 몫 | 사후 서술 | C2 README 2.2 RW1, RW5 |
| 잔차 구조 − 물리 입력 구조, 라벨 10(AB7) | −3.74 [−5.23, −2.75] / [−5.14, −2.98] | 우세, Holm *P* 0.001. 물리 입력 구조가 재보정 계수를 받지 않는다는 단서 | C3 README E06 |
| 같은 비교, 구조 차이만 보는 대비(재보정 값 입력) | −2.52 [−3.26, −1.56] / [−2.90, −1.16] | 우세, 지역 행: 러시아 서부·동부 우세, 레나델타 미결정, 캐나다 열세(+3.12) | C3 README E04, E08a |
| 라벨 0 구조 비교 | −2.72 [−4.34, −1.73] / [−3.93, −1.97] | 우세(L10 지지, 맹검) | C3 README E01, E05 |
| 증강 결합, 라벨 10(AB6) | −0.13 [−0.31, 0.22] / [−0.31, 0.20] | 동등, Holm 동등성 *P* 0.002 | C3 README E23 |
| 증강 결합, 라벨 40 | +0.34 [0.25, 0.49] / [0.21, 0.50] | 열세, 크기 0.5 cm 미만(레나델타·캐나다) | C3 README E21, E22 |
| 다른 학습기의 잔차 순가치(P1 대비) | TabPFN v2 라벨 3개 −0.32(보정 전 유의), 10개 −0.42; TabICL v2 10개 −0.55, 전량 −1.00 | 우세, 크기 0.5 cm 미만 행 있음, 일부 재현(비맹검) | C3 README E38, E39 |
| 수축 재보정 대 현지 최소제곱, 라벨 3(서술) | 현지 최소제곱 − 수축: 레나델타 +2.91, 캐나다 +5.40, 러시아 동부 +3.69, 러시아 서부 −4.97, 티베트 −81.77 | 서술(판정에 쓰지 않음) | C3 README E41, E43 |

**문단 구성**: (1) 재보정 문장(첫 문장, 지역 수와 캐나다 열세). (2) 잔차 ML 의 추가 몫(보정 전 유의). (3) 라벨을 쓰는 보정의 이득과 편향 진단(ρ 0.38, 결과 열람 뒤 설계, 부호는 예측하지 않음). 이어서 XA 자리표시. (4) 결합 구조(잔차 대 물리 입력, 라벨 10개 이하. 4지역 평균의 결론이며 지역 행을 병기). (5) 증강 결합의 동등(라벨 10)과 라벨 40 의 작은 열세. (6) 수축 재보정 대 현지 최소제곱(서술, 러시아 서부와 티베트 예외).

**쓰지 않을 문장**: 'ML gain is proportional to coefficient error'(C2 README 5.2), '라벨 10개로 ML 을 더할 가치를 진단한다', '러시아 서부에서 ML 이 14 cm 줄였다', '잔차 구조가 물리 입력 구조보다 낫다'(라벨 수 조건 없이), '수축 재보정이 현지 최소제곱보다 안전하다', '증강은 라벨이 생기면 효과가 없다'(일반화), 'Recalibration with 3–10 labels is safe'(SC1w 안전성 미확인 4/10).

**새 실험**
- 본문: `[XA: R3 | 사후 분석(재현(비맹검)) | 2.1 해석 조각]`(FINAL_BATCH 0.4 의 등록 문자열 그대로). 채우는 법: 2.1 해석 조각을 XA-1, XA-2, XA-3 순서로 이어 붙이고 공통 규칙(수축 잔여 의존 표지, 주 CI 와 대상 고정 CI 가 다르면 약한 쪽, '사후 분석' 명기)을 적용한다. XA 전까지 이 자리의 임시 문장: "We could not establish a relation between coefficient error and the gain of ML beyond recalibration (post hoc rank correlations −0.08 to 0.39 depending on the definition; Supplementary Table S13)."
- SI 지시: `[XB: SI, R3 지시 문장 | 결과 열람 뒤 설계, 실행 전 등록(비맹검 부분 포함) | 2.2 사전 고정 해석 문장 XB-1, XB-2(라벨 10개 행)]`. FINAL_BATCH 6절은 XB 를 SI 로 고정했다. 지침 6.2 의 'Fig 2b 에 적층 선 추가'는 쓰지 않는다(결과 전 배치 고정이 우선).

### 4.4 R4 Method selection with tens to hundreds of labels(350단어)

**첫 문장(영문)**: "Selecting the method by five-fold block cross-validation within the target labels reduced RMSE relative to a fixed residual recipe by 1.32 cm with 160 labels (95% CI 0.68 to 1.70; block-equal 0.83 to 1.77; Lena Delta and Canada; unchanged under common resampling) and was non-inferior at 40 and 160 labels (margin 0.5 cm), with both results arising in Canada; this rule was designed after the label-grid results had been viewed (Fig. 4c)."

**국문**: "대상 라벨 안 5겹 블록 교차검증으로 방법을 선정하면 고정 잔차 레시피보다 라벨 160개에서 RMSE가 1.32 cm 작았고(95% 신뢰구간 0.68–1.70, 레나델타와 캐나다), 라벨 40개와 160개에서 비열등이었다(한계 0.5 cm). 두 결과 모두 캐나다에서 나왔다. 이 규칙은 라벨 격자 결과를 본 뒤 설계하였다."

**근거 수치**

| 항목 | 값 | 판정과 표지 | 원천 |
|---|---|---|---|
| 규칙 W − 고정 잔차(λ 0.25), 라벨 160(2지역) | −1.32 [−1.70, −0.68] / [−1.77, −0.83] | 우세, 공통 CI 도 우세, 보조 Holm *P* 0.0004, 비열등 성립 | C5 README 2.1 E3 |
| 같은 대비, 라벨 40(2지역) | −0.87 [−1.20, −0.27] / [−1.28, −0.46] | 우세, 분할 독립 가정 의존(공통 CI 미결정), 비열등 성립 | C5 README E2 |
| 같은 대비, 라벨 10(4지역) | −0.25 [−1.11, 0.73] / [−1.09, 0.69] | 미결정, 비열등 불성립 | C5 README E1 |
| 같은 대비, 라벨 전량(4지역) | −0.74 [−1.65, 0.19] / [−1.75, −0.07] | 미결정, 비열등은 분할 독립 가정 의존(공통 CI 상한 0.62 cm) | C5 README E4 |
| 지역 행 | 캐나다 라벨 40 −2.10, 160 −2.72(우세); 레나델타 +0.35, +0.07(미결정); 레나델타 라벨 10 +0.70 [0.56, 1.03](열세) | 서술 | C5 README 2.2 E6, E7, E10–E12 |
| 재보정 Stefan 보다 나은 대상 수(WF4-b) | 라벨 160: 규칙 3, 고정 잔차 4; 전량: 규칙 2, 고정 잔차 8 | 기각 | C5 README 2.3 |
| 캐나다에서 규칙이 고른 방법 | 잔차 가중 1.0 을 라벨 40 에서 22/25, 160 에서 24/25 | 서술(사후) | C5 README 2.5 E29 |
| 풀 밖 알래스카(전이 모드) 규칙 W − 원천 계수 Stefan | 라벨 10, 40, 160 에서 +0.91, +0.98, +2.14 | 곡선 표 보조 판정(열세), 등록 대비 아님 | C5 README 2.4 E27a–E27c |
| 라벨 40–160 의 결합 구조 역전 | 잔차 구조 − 물리 입력 구조 +1.85에서 +2.38(2지역 열세), 캐나다 +4.46에서 +5.14, 레나델타 미결정, 알래스카(전이) 반대 방향 | L10 보조 행 | C3 README E12–E17 |
| 더 강한 재보정 기준선 대비 | 잔차 ML − 토양 보정 Stefan, 라벨 40: +1.31, 160: +1.30 | 미결정(둘 다) | C3 README E19 |
| 원천 교차검증으로 고른 신경망(두 CI 판정이 같은 대비) | 직접 ML 오차 증가: MLP 라벨 10 +2.96, 다중 헤드 MLP 라벨 0, 10, 전량 +1.33, +2.56, +1.67. 축소 FT-Transformer 잔차 전량 −0.19(λ 0.25)과 −0.76(λ 1.0)은 분할 독립 가정 의존 | 학습기별, 강건 문장 규칙을 채운 학습기 없음 | C5 README 2.6 E38–E46 |

**문단 구성**: (1) 규칙 W 문장(첫 문장, 결과 열람 뒤 설계, 2지역, 캐나다). (2) 라벨 10개와 전량의 비열등 상태, 레나델타 라벨 10 의 열세, WF4-b 기각. (3) 이득의 기전은 Discussion 으로 보내고 여기서는 선택 횟수만 쓴다. (4) 결합 구조 역전(2지역, 캐나다에서 발생, 토양 보정 Stefan 대비 미결정). (5) 원천 교차검증 조정 결과는 학습기별로 쓰고, 규칙 W 와 직접 비교한 실험이 없다는 점을 적는다(C5 README N10). 두 결과는 계산 환경이 달라(Rescale CPU 와 로컬 GPU) 빼서 비교하지 않는다.

**쓰지 않을 문장**: '규칙 W 는 안전하다', '라벨 수와 관계없이 낫다', '4지역에서 약 1 cm 낫다', '규칙 W 는 물리식보다 낫다', '라벨 40개에서 유의하게 낫다'(단정형), '라벨 전량에서 비열등'(단정형), '알래스카에서도 낫다', '대상 라벨 교차검증이 원천 교차검증보다 낫다는 것을 보였다', '방법은 원천이 아니라 대상 라벨로 골라야 한다'(단정형 처방), 'ML 이 재보정 물리식을 넘는다'(라벨 40, 160)(C5 README 5, C3 README 5.2).

**새 실험**
- 본문: `[XC: R4 | 결과 열람 뒤 설계, 재사용 지역의 재검정(비맹검 부분 포함) | 2.3 사전 고정 해석 문장 XC-1, XC-2(+XC-2s), 뒤에 S-XC4 (b) 지역 손해 문장]`. 채울 내용: XC-1(워크플로 − 무작위 배치 + 고정 잔차, 라벨 40, 160), XC-2(재보정 Stefan 대비 비열등, 라벨 10, 40, 160, 우세면 XC-2s 문장 덧붙임). 모든 문장에 '같은 지역을 다시 무작위로 나눈 분할에서'를 넣는다. 비맹검 부분은 WF2, WF4, WF8, LGX-N4 다
- SI 지시: `[XB: SI, R4 지시 문장 | 결과 열람 뒤 설계, 실행 전 등록(비맹검 부분 포함) | 2.2 사전 고정 해석 문장 XB-1, XB-2(라벨 40, 160 행, 레나·캐나다 2지역)]`

### 4.5 R5 Gains within label-rich regions(450단어)

**첫 문장(영문)**: "Within Alaska, residual ML on the recalibrated Stefan anchor reduced RMSE relative to the Stefan model recalibrated on the same labels by 0.54 cm with 1000 labels (95% CI 0.32 to 0.69; block-equal 0.40 to 0.80; 14.41 to 13.87 cm; 25 splits), and with all labels about 99% of the reduction in squared error lay between ERA5-Land climate grid cells (Fig. 7a,c)."

**국문**: "알래스카 안에서 재보정 Stefan 앵커 위의 잔차 ML은 같은 라벨로 재보정한 Stefan 식보다 라벨 1000개에서 RMSE가 0.54 cm 작았다(95% 신뢰구간 0.32–0.69, 14.41 cm에서 13.87 cm, 분할 25회). 라벨 전량에서 오차 제곱 감소의 약 99%는 ERA5-Land 기후 격자 사이 성분이었다."

**근거 수치**

| 항목 | 값 | 판정과 표지 | 원천 |
|---|---|---|---|
| 잔차 ML − 재보정 Stefan, 알래스카 라벨 1000(25분할) | −0.54 [−0.69, −0.32] / −0.60 [−0.80, −0.40]; RMSE 13.87 / 14.41 | 우세, 공통 CI 도 우세, 결과 열람 뒤 설계 | C4 README 2.1 E1.R1.n1000 |
| 같은 대비, 라벨 500 / 전량 | −0.39 [−0.54, −0.21] / [−0.57, −0.21]; −0.52 [−0.70, −0.31] / −0.68 [−0.91, −0.44] | 우세(두 주 가중), 공통 CI 에서는 미결정 | C4 README E1.R1.n500, n전량, 2.7 O1 |
| 줄일 수 있는 오차 제곱(하한 11.31 cm 위) | 79.71 에서 64.42 cm², 19.17% 감소 | 파생값 | C4 README 2.6 D1 |
| 레나델타(약 1477개), 캐나다(약 376개) | 잔차 ML − 재보정 Stefan 모두 미결정 | WF6-a 기각 | C4 README 2.2, 2.7 O6 |
| SAR 9열 추가(알래스카, 5분할) | 라벨 500–5000 에서 동등, 전량 미결정 | WF1-b 기각(추가 가치 미확인) | C4 README 2.5 |
| 첫 시험(5분할, 최선 물리 보정식 대비) | 알래스카 기각 | 재시험의 네 변경(비교 상대, 분할 수, 레시피, 라벨 범위)을 Methods 와 문단에 적는다 | C4 README 1.2 (c) |
| 원천 계수 Stefan 대비(같은 시험) | 셀 가중 −0.70, 블록 등가중 +0.58 | 미결정 | C4 README 2.7 O2 |
| 총 / 격자 사이 / 격자 안(라벨 전량, 교차검증 λ) | −0.52 [−0.70, −0.31] / −0.68; −1.03 [−1.30, −0.59] / −0.89 [−1.16, −0.61]; −0.01 [−0.03, 0.04] / 0.04 [0.01, 0.06] | 우세, 우세, 동등. 두 격자 정의에서 소수 둘째 자리까지 같음. RMSE 차는 더하지 않는다 | C8 README 2.1 A1–A6 |
| SSE 이득의 격자 안 몫 | 0.9%(토양 정의 1.0%) | 서술(CI 없음) | C8 README 2.2 B4, B10 |
| 재보정 Stefan 오차 제곱의 격자 안 비율 | 72% | 서술 | C8 README B1 |
| ML 이 설명한 격자 안 변동 | 알래스카 1% 미만(최대 0.7%), 세 지역 모든 ML 설정 8% 미만(최대 레나델타 7.8%) | 서술 | C8 README 2.3 C2, C12–C14 |
| 격자 안 등록 가설(지역 내 층화 평균) | +0.09 [0.02, 0.40] / 0.37 [0.25, 0.50](기온 정의 전량); 토양 정의 +0.27, +0.42, +0.40 | 열세가 있는 기각, 모두 0.5 cm 미만 | C8 README 2.4 D3, 2.5 E1–E3 |
| 전이 격자 안 | 토양 정의 민감도 뒤 기각 | '근거는 확인되지 않았다' | C8 README 1.2 의 2 |
| 기후 외삽 | 판정 불가(캐나다 외삽 영역 3블록) | WF10 | C8 README 5.1 의 11 |

**문단 구성**: (1) 알래스카 문장(첫 문장, 결과 열람 뒤 설계와 네 변경). (2) 하한 대비 19%, 라벨 500 과 전량(공통 CI 의존), 레나델타·캐나다 미결정, SAR. (3) 이득의 공간 규모: 격자 사이 대 격자 안, 72% 와 1% 미만. (4) 격자 안 등록 가설의 열세가 있는 기각 문장("Within regions, the within-grid predictions of residual ML increased the error in the stratified mean (all below 0.5 cm)"). (5) 기후 외삽 판정 불가. Fig 7b 의 지도 문구는 기본판(채점 블록별 오차 변화)이면 'error change by scoring block', 1 km 판(새 적합·예측 실행 승인 뒤)이면 '1 km display of a climate-grid-scale correction'이다(그림 명세 D-11).

**쓰지 않을 문장**: '라벨이 많으면 물리 기반 ML 이 물리식보다 낫다'(조건 없이), 'exceeds within Alaska'(비교 상대 없이), '물리 계수가 맞는 지역일수록 이득이 작다', '라벨 500개 이상이면 유의하게 낫다', 'SAR 은 쓸모없다', 'ML adds spatial detail', 'high-resolution ALT map', '이득의 99% 는 격자 단위 보정이다'(지역 한정 없이), '기후 변화에서 물리 잔차가 더 안전하다', '엄격한 조건에서 대회 결과를 재현했다'(C4 README 5.2, C8 README 5.2).

**새 실험**
- `[XE: R5 SI 지시 문장 | 결과 열람 뒤 설계 | 2.5 사전 고정 해석 문장(6갈래)과 공통 문장]`. 6갈래: XE-a·XE-b 우세, XE-a 우세·XE-b 미결정, 동등, 미결정(100 m 셀 안 분산 비율 병기), 열세, 판정 불가
- `[XI: R5 SI 지시 문장 | 결과 열람 뒤 설계 | 2.9 다섯 갈래, 공간 대용과 외삽 폭]`. 문장에 '같은 시기의 따뜻한 블록을 쓴 공간 대용'과 외삽 폭을 넣고, 판정 불가가 반복되면 '기후 외삽은 판정할 수 없었다'를 유지한다
- `[XB: SI, R5 지시 문장 | 결과 열람 뒤 설계, 실행 전 등록(비맹검 부분 포함) | 2.2 사전 고정 해석 문장 XB-3, XB-4]`(XB-3 지역 내 적층 잔차, XB-4 알래스카 비열등)

### 4.6 R6 Placement of new labels(400단어)

**첫 문장(영문)**: "In Canada, spreading labels over 0.5° blocks or covariate space reduced RMSE relative to random cell sampling by 2.39 cm with 40 transfer labels (95% CI 0.49 to 3.40; block-equal 0.76 to 2.49) and by 2.54 to 3.21 cm with 50 to 100 within-region labels, whereas in the Lena Delta block stratification increased RMSE by 1.65 cm with 100 labels (95% CI 0.90 to 2.40; block-equal 0.15 to 1.57) (Fig. 5)."

**국문**: "캐나다에서 라벨을 0.5° 블록이나 공변량 공간에 퍼뜨리면 무작위 셀 추출보다 RMSE가 전이 라벨 40개에서 2.39 cm, 지역 내 라벨 50–100개에서 2.54–3.21 cm 작았다. 레나델타에서는 블록 층화가 라벨 100개에서 RMSE를 1.65 cm 키웠다."

**근거 수치**

| 항목 | 값 | 판정과 표지 | 원천 |
|---|---|---|---|
| 캐나다 전이, 공변량 최대 최소 거리 − 무작위, 라벨 40 | −2.39 [−3.40, −0.49] / −1.64 [−2.49, −0.76] | 우세(WF8-a 지지, 결과 열람 뒤 설계) | C6 README 2.5 D1 |
| 캐나다 지역 내, 라벨 50–100 | 블록 층화 −2.74, −2.54; 공변량 −2.63, −3.21 | 우세(서술 대상) | C6 README 2.1 A2, A3, A6, A7 |
| 캐나다 지역 내, 라벨 200 | 블록 층화 −1.44, 공변량 −1.25 | 우세 | C6 README A4, A8 |
| 레나델타 지역 내 블록 층화, 라벨 100 | +1.65 [0.90, 2.40] / +0.88 [0.15, 1.57] | 열세 | C6 README A17 |
| 레나델타 블록 층화, 라벨 20–500 | +1.47에서 +1.98, 셀 가중 CI 모두 0 초과 | 서술 | C6 README 1.3 |
| 알래스카 하위 지역 | AL-1 공변량(라벨 50–200) 0.32–0.77 cm, AL-2 블록 층화(라벨 50, 100) 0.70, 0.65 cm, 공변량(라벨 50) 0.39 cm 감소 | 우세, 독립 지역 아님(대상 수로 서술) | C6 README A9–A13, A15 |
| 알래스카 전체 | 블록 등가중 −1.08에서 −1.73(CI 모두 0 미만), 셀 가중 +0.28에서 +0.74 | 미결정 | C6 README A23–A25, A29–A31 |
| 전이 라벨 10(WF8-b) | 두 가중 모두 열세인 대상 없음. 레나델타 블록 층화 셀 가중 +1.89 [0.30, 3.08], 러시아 동부 공변량 블록 등가중 +0.36 [0.08, 0.62] | 미결정, 비열등 시험 아님 | C6 README D3, D10, 1.2 (d) |
| 예측 분산 능동 선정 | 알래스카, 레나델타, AL-1: +0.12에서 +2.79(라벨 50–200, 열세); 캐나다 라벨 100 −1.50, 200 −0.68(우세) | 열세·우세 혼재 | C6 README 2.2 B1–B9, B14, B15 |
| 알래스카 최선 전략의 이전 | 레나델타 +0.33 [0.04, 0.77] / +0.68 [0.36, 1.00] | 열세, WF2-b 기각 | C6 README 2.4 C1 |
| 등록 배치 대비(블록 분산 − 셀 무작위) | 라벨 10: −0.07 [−0.86, 0.89] / −0.21 [−0.81, 0.41]; 라벨 40: −0.24 [−1.19, 0.99] / −0.52 [−1.36, 0.30] | 미결정, '배치 효과는 확인되지 않았다'(L43, 비맹검 부분 포함) | C6 README 2.6 E1, E2, 2.7 |

**문단 구성**: (1) 캐나다·레나델타 문장(첫 문장, 결과 열람 뒤 설계, 실행 전 등록). (2) 알래스카 전체와 하위 지역(대상 수). (3) 전이 라벨 10 의 상태(열세 판정 없음, 한쪽 가중 CI 가 0 초과인 두 행). (4) 능동 선정. (5) 알래스카 전략의 이전 기각과 등록 대비 L43(같은 라벨 집합의 다른 집계라 모순이 아님을 한 문장으로). (6) 'We tested a pre-specified placement procedure'(배치 절차) 문장과 XC, XD 자리표시. 효과의 지역 차이 해석(레나델타 큰 블록의 E 차이)은 Discussion 으로 보낸다.

**쓰지 않을 문장**: '안전하다', '어떤 지역에서도 무작위보다 나빠지지 않는다', '능동 선택은 모든 대상에서 이득이 없었다', '1–3 cm 나빴다', 'S4 로 고르고 S5, S6, S7 은 쓰지 않는다'(처방), '라벨은 여러 블록에 흩어야 한다', 'where to measure', 관측 우선순위 지도, '이질적 지역에서는 2.5–3.2 cm 이득'(일반 법칙), 'S4 가 최적 관측 설계임을 보였다', '블록 분산 추출은 무효다'(C6 README 5.2).

**새 실험**
- `[XC: R6 | 결과 열람 뒤 설계, 재사용 지역의 재검정(비맹검 부분 포함) | 2.3 사전 고정 해석 문장 XC-5와 공통 문장]`. 채울 내용: XC-5(규칙 W 아래 배치 알고리즘 − 무작위, 라벨 40, 160)와 공통 문장(배치 절차는 알래스카 계열에서 골랐고 평가 셀은 구성 요소 설계에 쓰였다). 그림은 Fig 7e 의 'vs random, CV selection' 묶음
- `[XD: R6 SI 지시 문장 | 결과 열람 뒤 설계, 탐색(SI), XD-2 사후 설계(지역 표적) | 2.4 사전 고정 해석 문장(SI) XD-1, XD-3, 공통 한계 문장]`(공통 한계 문장: 후보 풀은 이미 측정한 셀). FINAL_BATCH 6절에 따라 XD 는 SI 다. 지침 6.5 의 'Fig 5f 에 XD 행 추가'는 쓰지 않는다.

### 4.7 R7 Independent regions(250단어)

**첫 문장(영문)**: "In Central Russia, a licence-verified new shallow region, direct ML without target labels increased RMSE relative to the source-coefficient Stefan model by 2.75 cm (95% CI 2.11 to 4.09; block-equal 3.17 to 6.60), and in the five-region shallow pool that includes it, residual ML with all labels reduced RMSE by 3.19 cm (95% CI 2.22 to 4.19; block-equal 1.53 to 3.71; cross-environment pool) (Table 1)."

**국문**: "약관이 확인된 새 얕은 레짐 지역인 러시아 중부에서 라벨 없는 직접 ML은 원천 계수 Stefan 식보다 RMSE가 2.75 cm 컸다(95% 신뢰구간 2.11–4.09). 러시아 중부를 더한 얕은 레짐 5지역 풀에서 라벨 전량의 잔차 ML은 RMSE를 3.19 cm 줄였다(95% 신뢰구간 2.22–4.19, 교차 환경 풀)."

**근거 수치**

| 항목 | 값 | 판정과 표지 | 원천 |
|---|---|---|---|
| 러시아 중부 직접 ML − 원천 계수 Stefan, 라벨 0 | +2.75 [2.11, 4.09] / +4.83 [3.17, 6.60] | worse(공통 CI 도 worse), 교차 환경 표지 | C1 README 2.2 |
| 얕은 레짐 5지역 풀, 잔차 ML − 원천 계수 Stefan, 전량 | −3.19 [−4.19, −2.22] / −2.62 [−3.71, −1.53] | improve, 교차 환경 | SI_learners README D-L8e-PE1 |
| 같은 풀, 잔차 ML − 재보정 Stefan, 라벨 10 | −0.13 [−0.42, 0.01] / [−0.41, 0.01] | 동등, 교차 환경('4지역 판정은 확장 풀에서 유지되지 않는다') | C3 README E37 |
| 러시아 중부 단독, 잔차 ML − 원천 계수 Stefan, 전량 | −5.42 [−8.96, −1.64] / −2.55 [−6.83, 1.48] | 미결정(두 가중 엇갈림), L38 '어긋난 지역' | SI_learners README 1.3 |
| 라벨 0–40 직접 ML 의 유의 개선 지역 | 4지역 0/4, 5지역 0/5, 6지역 1/6(티베트) | L1e 지지 | SI_learners README D-L1e |
| 티베트(심부 레짐) | 원천 계수 Stefan RMSE 244.51 cm, 직접 ML 라벨 0 −59.98; 잔차 ML 라벨 10 은 수축 재보정보다 20.90 cm 작으나 현지 최소제곱과 미결정(+19.11) | 비맹검, 원천 지지 밖, 독립 지역 검증 근거로 쓰지 않음 | C2 README 2.2 TB1, TB3, SI_learners README 1.3 |
| 북대서양 | SI 전체 판에만 | 약관 미확인 자료 포함 | SI_learners README 1.3 |

**문단 구성**: (1) 러시아 중부 문장. (2) 확장 풀의 유지와 불유지(라벨 10개 이하 순이득은 유지되지 않음). (3) 티베트는 따로 보고하고 검증 근거로 쓰지 않는 이유. (4) XF 결과 자리. 용어: FINAL_BATCH 1절은 '독립 지역 확인' 표지를 2026-10-04 조사 뒤 처음 확보한 지역에만 쓰게 했으므로 러시아 중부에는 'licence-verified new region'을 쓰고 'confirmation'을 붙이지 않는다. `[DECISION: 러시아 중부의 영문 표기]`

**쓰지 않을 문장**: 'replicated in new regions'(L38 (a)(b)가 모든 새 지역에서 성립할 때만), '30개 독립 지역', 지역 일반 문장(HK 구간 없이)(WRAPUP 11.2).

**새 실험**
- `[XF: R7 | 결과 열람 뒤 설계, 새 지역은 맹검(독립 지역 확인), NAtlantic v5 는 민감도 | 2.6 사전 고정 해석 문장]`. 갈래: 적격 0곳 문장, 적격 1곳 이상의 다섯 갈래, NAtlantic v5 문장, 하위 과제만 추가 문장. 결과 표는 Supplementary Table S18 이다. 사전 기대는 적격 0곳이다(QA_FOLLOWUP 1.1).
- `[XC: R7 | 실행 전 등록 확인 시험(맹검) | 2.3 XC-F3 갈래]`(F3 0곳이면 '공개 자료로 확보한 새 독립 지역이 적격 조건을 만족하지 못해 외부 계열 시험은 하지 못했다')
- `[XJ: R7 SI 한 줄 | 결과 열람 뒤 설계, 서술 | 2.10 사전 고정 해석 문장]`(지온 유도 보조 행 가중 w, 라벨 k개, L39 정의 차이 단서)

### 4.8 R8 Prediction intervals(200단어)

**첫 문장(영문)**: "Without target labels, hierarchical conformal intervals calibrated with regions as groups covered on average 0.86 of held-out observations (95% CI 0.80 to 0.92; nominal 0.90; four regions) but 0.75 in W Russia, and with four calibration regions they carry no finite-sample guarantee (Supplementary Table S12)."

**국문**: "대상 라벨이 없을 때 지역을 집단으로 보정한 계층 conformal 구간은 홀드아웃 관측의 평균 0.86을 포함하였고(95% 신뢰구간 0.80–0.92, 명목 0.90, 4지역) 러시아 서부에서는 0.75였다. 보정 집단이 4개라 유한표본 보장은 없다."

**근거 수치**

| 항목 | 값 | 판정과 표지 | 원천 |
|---|---|---|---|
| 계층 conformal, 라벨 0 | 0.86 [0.80, 0.92], 러시아 서부 0.75 | 09-26 사전 등록(F10, 띠 판정), 앞선 규약 표지 | SI_uncertainty README 1.3, 1.2 표 |
| 알래스카에서 보정한 분위 구간의 전이 | 커버리지 0.32–0.73 | 서술 | SI_uncertainty README 1.3 |
| 원천 셀 교환성 생성 구간 | 정규화 흐름 0.76, 흐름 정합 0.65 | 과소 커버리지 | SI_uncertainty README 1.3 |
| 셀별 폭 정규화기 − 위약 | +3.29 [−4.17, 8.16] | 미결정(LGU-B2, 맹검) | SI_uncertainty README 2.5 |
| 라벨 10, 계층 예측 분포 − 보정 상수 폭(구간 점수, AB10) | +9.51 [−0.75, 22.11] | 미결정, Holm *P* 0.277, 규칙 (d) | SI_uncertainty README 2.6 |
| 알래스카 지역 내 분위 구간 | 보정 전 0.45 [0.32, 0.62], 보정 뒤 0.93 [0.89, 0.97] | 서술(대회 산출 재분석) | SI_uncertainty README 1.3 |

**문단 구성**: (1) 계층 conformal 문장. 이 값은 09-26 규약의 참조값이라 Fig 6d 에 그리지 않고 Supplementary Table S12 에 둔다(`fig6_b_ref.csv` 의 `use` 열: 'LGU 와 CI 를 나란히 비교하지 않는다', 그림 명세 8.3 d). (2) 전이에서 일반 구간의 과소 커버리지와 구간 척도화 5종의 포함률 대 폭(Fig. 6d). (3) 셀별 폭과 라벨 있는 계층 분포의 미결정, 라벨 40개와 160개 잔차 구간(Fig. 6e). 지도는 SI 한 장이다(LGU-B2 미결정으로 Fig 6 지도 행 조건 불충족, 지침 6.6).

**쓰지 않을 문장**: 'coverage-guaranteed interval or map', 'labels narrow the interval', 'conditional width transfers between regions', 'the hierarchical model improves intervals'(SI_uncertainty README 5.2, 결과 초안 D6).

---

## 5. Discussion(850단어, 소제목 없음, 6문단)

순서는 의의와 워크플로를 먼저, 한계 1문단, 마지막은 적용 조건과 전망이다(지침 4.1, H11, scirep_ex 5.7). 새 분석은 하지 않는다.

| 문단 | 역할 | 목표 단어 | 내용(원천) | 인용 |
|---|---|---|---|---|
| D1 | 의의와 워크플로 | 170 | 결과가 정하는 단계(claims 4절 표, Fig. 7d): 라벨 0 은 원천 계수 Stefan 식(연도 정합 도일이 있으면 연도 정합 Stefan 식을 함께 보고)과 물리 정보를 쓴 ML 만, 라벨 3–10 은 계수 재보정과 저가중 잔차, 라벨 40–1000 은 대상 라벨 교차검증 선정(라벨 40개와 160개 비열등, 이득은 캐나다), 라벨이 수천 개인 지역 안에서는 교차검증 잔차 가중의 잔차 ML(알래스카, 격자 단위 보정, Fig. 7a–c). Fig. 7d 의 경로는 레나델타·캐나다 전이 풀 값이므로 전량 구간도 교차검증 선정 값이고, 이 풀에서 라벨 3–160 의 권고 방법은 원천 계수 Stefan 보다 0 위에 놓인다(그림 명세 9.3 내용 위험, `[DECISION: Fig 7d 경로의 y 기준과 풀, 그림 명세 D-13]`). 배치는 방법 수준 절차로만. 수치 1–2개(AB3, AB4 또는 99%) | Fig 7d 만 |
| D2 | 기전 | 150 | 이득의 출처: 계수 수준 보정(재보정 몫, 러시아 서부 97%, C2 README 2.3 비율 표의 4지역 평균 85% 는 러시아 서부 지배)과 기후 격자 단위 편향 보정(C8). E 는 열전도도, n-인자, 함수량, 격자 편향, 라벨 정의를 묶은 계수다(지원 초안 M3.2 D1–D3, 해석). 원천 E 는 알래스카 셀 78–94% 라 원천 계수 Stefan 식은 알래스카 중심 계수의 이전을 시험한다(D8). 캐나다는 지역 계수가 원천과 가장 가깝다. 그러나 라벨 위치 이질성 때문에 재보정이 나빴고 규칙 W 는 잔차 가중 1.0 을 골랐다(C4 README 1.2 (a), C5 README N6, 해석) 자리표시 `[XA: Discussion D2 \| 사후 분석(재현(비맹검)) \| 2.1 해석 조각 XA-3]` | Riseborough 2008, Shiklomanov & Nelson 2002, Nelson 1997 |
| D3 | 선행 연구 대비 | 160 | 일치: 순수 ML 이 Stefan 보다 나빴던 무작위 분할 결과(Gautam 2025)와 같은 방향. 지역 안 대회 결과(14.46 대 13.33 cm, RMSE 8% 감소, 줄일 수 있는 오차 제곱 39% 감소)와 엄격한 재시험(4%, 19%)은 같은 방향이고 크기가 작다(C4 README 2.6 D1, D2, 백분율은 지침 4.5 에 따라 정수). 물리 입력 구조 선례(Pilyugina 2025, Wang 2025)는 라벨 수에 따라 우열이 바뀐다(C3). 검증 설계 문헌의 양쪽 입장(Ploton 2020, Meyer & Pebesma 2022 대 Wadoux 2021)과 우리 라벨이 비확률 군집 표본이라는 점. 가장 가까운 최신 연구(O'Malley 2026)와 범위 차이 | Gautam 2025, Ran 2022a, Pilyugina 2025, Wang 2025, Read 2019, Jia 2021, O'Malley 2026, Ploton 2020, Meyer & Pebesma 2022, Wadoux 2021, Linnenbrink 2024 |
| D4 | 대안 설명 배제 | 110 | 위약 대조(섞은 유사라벨, 상수, √TDD 선형, C1 README 2.3), 라벨 단위와 연도 민감도(지점 대 1 km 위치 평균, 단년 라벨, S-a·S-b, 지원 초안 M5.4–M5.5), 계산 환경(물리식·CatBoost 는 환경 사이 최대 3.6e-14 cm, 신경망은 최대 5.6 cm 달라 워크플로는 CPU 결정적 학습기로 구성, RESEARCH_OVERVIEW 4절), 학습기 선택(학습기 10종, 두 CI 판정이 같은 5종), 분할 독립 가정(공통 재표집 CI 병기) | 없음 또는 Hartung & Knapp 2001 |
| D5 | 한계(Hall 2026 형식) | 160 | 아래 5.1 표 | 없음 |
| D6 | 적용 조건과 전망(마지막) | 110 | 워크플로 단계마다 적용 조건(라벨 단위는 1 km 위치 평균, 재보정은 라벨 10개 이상, 규칙 W 의 블록 교차검증은 블록 수에 의존, 배치 절차는 사전 지정 절차로 시험). 다음에 필요한 자료: 격자 안 공변량, 새 독립 지역, 같은 시험지의 기존 ALT 제품 비교. 자리표시 `[XC: Discussion D6 \| 결과 열람 뒤 설계, 재사용 지역의 재검정(비맹검 부분 포함) \| 2.3 XC-1, XC-2]`, `[XE: Discussion D6 \| 결과 열람 뒤 설계 \| 2.5 해석 문장]`, `[XF: Discussion D6 \| 결과 열람 뒤 설계, 새 지역은 맹검(독립 지역 확인) \| 2.6 해석 문장]`, `[XG: Discussion D6 \| 결과 열람 뒤 설계, 등록 이탈(WRAPUP 10) \| 2.7 공통 문장]`. 마지막 문장은 조건이나 다음 자료로 끝내고 한계 나열이나 일반론으로 끝내지 않는다 | Wei 2026(제품, 서지 확인 뒤) |

### 5.1 한계 문단(D5) 설계

scirep_ex 4.7 이 정리한 Hall 2026 한계 문단의 네 요소를 따른다. 문장마다 해석에 주는 영향을 붙이고, 자기 평가 문장으로 닫지 않는다.

| 요소 | 이 원고의 내용 | 해석에 주는 영향(문장에 함께 쓴다) | 원천 |
|---|---|---|---|
| (1) 하지 못하는 것 | 워크플로는 측정 지점의 순위나 관측 우선순위 지도를 내지 않는다. 지도는 기후 격자 단위 보정을 1 km 셀로 표시한 것이고 격자 안 상세도가 아니다. 예측 구간에는 유한표본 보장이 없다 | 배치 문장은 방법 수준 절차로만 읽어야 한다. 1 km 표시는 격자 안 변동의 예측으로 읽지 않는다 | C6 README 5.2, C8 README 4절, SI_uncertainty README 1.3 |
| (2) 시험하지 않은 조건 | 확인적 독립 지역은 주 4지역(재사용 지역의 재검정)과 새 얕은 지역 1곳이다. 원천의 78–94% 가 알래스카 셀이다. 기후 외삽은 판정하지 못했다. 심부 레짐은 티베트 하나이고 비맹검이다. 결론은 GPR 라벨을 포함한 원천 조건에 한정된다 | 지역 일반 문장을 쓰지 않는다. 원천 구성을 바꾸면 원천 계수 Stefan 식과 그에 기댄 방법의 순위가 바뀔 수 있다 | D README 1.2 의 4, E35, C8 README 5.1 의 11, 결과 초안 D5 의 4 |
| (3) 시험 방식에 따른 차이 | 지역 안 블록 시험과 전이 시험의 크기가 다르다(알래스카 지역 내 0.54 cm 대 전이 모드 라벨 전량 미결정). 무작위 셀 분할과 지역 홀드아웃의 직접 ML 차는 10.34 cm 다 | 보고 수치는 시험 방식에 묶인다. 다른 연구의 무작위 분할 수치와 직접 비교하지 않는다 | C4 README 2.7 O7, C7 README A1 |
| (4) 측정하지 않은 변수 | 격자 안 토양 수분, 유기층 두께, 적설 지속, 미지형은 현재 공변량(약 5 km 로 추출한 SoilGrids 포함)에 거의 없다. 라벨 연도(1990–2024)와 기후 기준 기간(2015–2020)이 다르다 | 격자 안 설명 비율 1% 미만은 현재 입력에서의 값이며 원리적 한계로 읽지 않는다. 연도 불일치는 계수 E 에 흡수된다 | C8 README 5.1 의 12, 방법 초안 Covariates, 지원 초안 M3.2 D3 |
| 설계 표지 | WF 실험과 XA–XJ 는 앞선 결과를 본 뒤 설계하였다(실행 전 등록). 효과 크기가 작다(라벨이 많은 지역 0.4–0.5 cm) | 해당 결과는 확인적 결과보다 약한 근거로 읽는다 | QA_FINAL_REVIEW 리뷰어 평가, C4 README 5.1 |

D5 영문 틀(scirep_ex #40, #38): "The workflow does not rank sites for measurement or predict variation within climate grid cells, its conclusions are restricted to source data that include GPR-derived labels, and its behaviour in climates warmer than the training range could not be evaluated. The Alaskan dominance of the source data matters most at small label counts, because the source coefficient carries the weight of ten labels in the recalibrated coefficient." 근거: 결과 초안 D5 의 4(GPR 라벨 조건), C8 README 5.1 의 11(외삽 판정 불가), 지원 초안 M3.2 D8($E_0$ 의 가중 10라벨).

---

## 6. Methods(약 4240단어, 소절 15개)

소제목은 명사구다. 결정마다 이유를 한 구절로 붙인다(scirep_ex 5.8, Hall·Feeney 형식). 표지: WF 와 XA–XJ 를 다루는 소절은 'designed after earlier results were viewed' 를 적는다.

| 번호 | 영문 소제목 | 국문 소제목 | 목표 단어 | 내용 | 원천 |
|---|---|---|---|---|---|
| M1 | Study regions and labels | 연구 지역과 라벨 | 400 | 라벨 자료(GTN-P/CALM, ALLena, ABoVE), 셀 집계 규칙, v3 표(17,572행, 직접 라벨 17,467셀), 지온 유도 20셀 표지, 지역 정의와 하위 지역(라벨 없는 k-평균), 라벨 단위(지점, CALM 다년 평균, 1 km 셀 평균), 약관 미확인 자료 제외 | 방법 초안 Observational data, D README E01–E08, E10–E20 |
| M2 | Covariates | 공변량 | 300 | x25(지형 6, 기후 8, 토양 9, CCI 2), ERA5-Land 2015–2020, SoilGrids 는 250 m 제품을 약 5 km 로 재표본해 추출(정정), CCI 는 모형 산출 공변량, TDD 식(식 (1)), 라벨 유래 값 입력 금지 | 방법 초안, scirep_fmt 3.5(SoilGrids 정정) |
| M3 | Stefan baselines and coefficient recalibration | Stefan 기준선과 계수 재보정 | 300 | $E_0 = \sum s y / \sum s^2$, $E_n = (n E_{ls} + 10 E_0)/(n + 10)$, 연도 정합 Stefan(라벨 0, 전량 병기), 토양 보정 Stefan(라벨 40, 160 병기), 현지 최소제곱. 수식 번호는 (1)–(3) | 지원 초안 M3.1, D README 1.3, E41–E43, E51, E52 |
| M4 | Machine-learning methods and combination structures | 기계학습 방법과 결합 구조 | 450 | 직접 ML, 물리 유사라벨 증강(비율 r = 10), 앵커 + 잔차(λ), 증강 + 앵커 + 잔차, 물리 입력, Stefan·CCI 앵커, 위약 유사라벨 3종. 학습기 10종은 이름만(세부는 Supplementary Methods). 1.4 표의 이름을 한 번 대응시킨다 | 방법 초안 Learning methods, C3 README 0절 |
| M5 | Transfer design | 전이 시험 설계 | 300 | 대상 27개(주 4지역, 알래스카 참조, 하위 지역), 원천에서 대상과 100 km 완충 제외(누설 방지 이유), A/B 블록 절반 분할 5회, 라벨 추출 5회, 라벨 격자, 공정성 원칙(모든 방법이 같은 라벨) | D README E24–E34, scirep_ex #15 |
| M6 | Within-region design and gain decomposition | 지역 내 시험 설계와 이득 분해 | 300 | 지역 내 블록 홀드아웃(원천 없음), 분할 5회(첫 시험)와 25회(재시험), 첫 시험 기각 뒤의 네 변경, 격자 묶음(√TDD 값과 0.5° 블록), 격자 안·사이 저장소, SSE 기준 분해, 격자 정의 이탈(기온 √TDD 로 실행, 토양 정의 민감도) | C4 README 1.2 (c), C8 README 기호와 5.1 의 1–3 |
| M7 | Label placement procedures | 라벨 배치 절차 | 250 | 셀 무작위, 블록 층화, 공변량 군집 층화, 공변량 최대 최소 거리, 원천 외삽 우선, 예측 분산 능동 선정, √TDD 우선의 정의. 배치 절차(Algorithm P)의 의사코드와 매개변수 후보(γ 0, 0.5, 1), 알래스카 계열에서만 고르는 규칙. 자리표시 `[XD: Methods M7 \| 결과 열람 뒤 설계, 부록 XC-0 의 기계적 선택(알래스카 계열) \| 2.4 고른 후보 이름과 γ]` | C6 README 3.1, QA_FOLLOWUP 2.4 |
| M8 | Method-selection rule and bias diagnosis | 방법 선정 규칙과 편향 진단 | 250 | 규칙 W 절차 5단계(후보 7개, 5겹 블록 교차검증, 라벨 10개 미만이면 재보정), 편향 진단값(라벨 10개에서 E0·√TDD − ALT 평균의 절댓값), 비열등 한계 0.5 cm. 워크플로 순차 적용(XC)의 팔 정의는 SI | C5 README 1.4, C2 README 기호 |
| M9 | Statistical analysis | 통계 분석 | 600 | 6.9 절 | 방법 초안 Scoring, WRAPUP 0.5, 1.1, FINAL_BATCH 1절 |
| M10 | Internal pre-registration and deviations | 내부 사전 등록과 등록 이탈 | 300 | 6.10 절 | 방법 초안 Pre-registration, 등록부 |
| M11 | Additional registered analyses | 추가 등록 분석 | 350 | XA–XJ 의 질문, 대비, 판정 규칙, 맹검 표지, 본문/SI 배치(결과 전 고정)를 실험마다 2–3문장으로. 세부는 Supplementary Methods | FINAL_BATCH 0.1, 2절, 6절 |
| M12 | Computing environments and reproducibility | 계산 환경과 재현성 | 200 | Rescale CPU(elm, hematite), 로컬 GPU, 패키지 판 고정, 환경 사이 재현(최대 3.6e-14 cm, WF9 공통 키 11,814개 ΔSSE 0, 신경망 최대 5.61 cm), 한 대비는 한 플랫폼 안에서, 교차 환경 표지 | SI_learners README 1.3, C8 README 5.1 의 14, FINAL_BATCH 1절 |
| M13 | Maps and colour scales | 지도와 색표 | 80 | 북극 평사 투영(EPSG:3413 형식), 해안선 Natural Earth 50 m, 영구동토 구역 자료 [미확인], Cartopy 판 [미확인], Crameri 색표(oslo, broc, bam, acton)와 끝 자르기 사실 | 지침 2.6, 2.11 |
| M14 | Use of large language models | 대형 언어 모델 사용 | 40 | `[DECISION: LLM 사용 기재 문구]`. 참고 틀(scirep_ex 5.9, 결정 전에는 쓰지 않는다): "Claude (Anthropic) was used to [write analysis code / edit language]. The authors checked all code, results and text and take responsibility for the content." | Sci Rep 규정(scirep_fmt 2.2) |
| M15 | Code availability | 코드 가용성 | 120 | 6.15 절 | 방법 초안 Code availability |

### 6.9 M9 Statistical analysis(600단어)

- 대비 Δ = RMSE(A) − RMSE(B)(cm). 지역 안 RMSE 는 채점 셀 가중(셀 가중)과 0.5° 블록 등가중(블록 등가중) 두 가지로 낸다. 층화 평균은 지역 등가중이다.
- 신뢰구간은 분할 안 채점 블록 재표집 95% 백분위 구간이다. 재표집은 LGX, LGF, WF, 초록 묶음 10,000회, LG 본 판정 1000회, 예측 구간(AB10) 1000회다. 두 팔의 라벨 집합이 다른 배치 대비는 2단 재표집(추출 변동 포함)을 쓴다(L43, XC, XD).
- 4분 판정: 두 가중 CI 가 모두 0 아래면 lower error, 모두 위면 higher error, 네 끝값이 모두 ±0.5 cm 안이면 equivalent, 그 밖은 undetermined. 0.5 cm 는 라벨 0 원천 계수 Stefan 식 RMSE(셀 가중, 주 4지역 21.69–42.98 cm)의 약 1–2% 다(D README '이번 계산'). 보조 한계 1.0 cm 와 상대 한계를 병기한다.
- 비열등: 단측, 두 가중 CI 상한이 모두 +0.5 cm 미만(규칙 W, SC1w, XB-4, XC-2).
- 유의 수준: 양측 α = 0.05(단측 0.025 와 대응). p 는 두 가중 분포의 양측 부트스트랩 p 가운데 큰 값이다. 모든 대비의 보정 전 p 와 Holm 보정 p 를 SI 표에 싣는다(Sci Rep 통계 지침).
- 다중성: 초록 묶음 10개(m = 10)에 Holm 보정을 하고 보정 p 로 초록 문구를 정한다(규칙 (a)–(e)). 가설 묶음 안 Holm 은 보조 열이다. XC(m = 8)와 XG(m = 6)는 Holm 보정 p 로 판정 문구를 정한다.
- 공통 재표집 보조 CI: 분할 사이 독립을 가정하지 않는 지역 블록 공통 재표집 CI 를 함께 낸다. 판정이 다르면 'depends on the split-independence assumption'을 병기하고 단정형으로 쓰지 않는다.
- 작은 표본: 지역은 4–7개다. 재표집 단위가 지역이 아니라 블록인 이유를 적고, 지역별 값과 범위를 그림과 표에 함께 둔다. 지역 일반 문장은 Hartung–Knapp 구간이 0 을 제외할 때만 쓴다.
- 순위상관: Spearman ρ, 대상 고정 결합 재표집 CI(편향 진단). XA 는 계열 군집 재표집 CI.
- 사후 서술과 결과 열람 뒤 설계의 표지, 맹검 표지 네 가지(맹검, 비맹검 부분 포함, 재현(비맹검), 등록 시점에 결과 존재)를 정의한다.
- 오차 하한: 공변량이 같은 셀 묶음 안 관측 분산의 합동 추정(알래스카 13,333셀). 효과 크기 표기의 분모로만 쓴다(D README E44, E45).
- ± 뒤의 값은 쓰지 않는다. 쓰게 되면 SD 인지 SE 인지 적는다.

### 6.10 M10 Internal pre-registration and deviations(300단어)

**등록 기록**(커밋 해시와 시각은 `paper/registry/experiments.csv`와 방법 초안 Table 2 에서 `numbers.tex` 매크로로 옮긴다)

| 묶음 | 표지 | 기록 |
|---|---|---|
| LG, LGX, LGD, LGT, LGU, LGF, WRAPUP | internally pre-registered(결과 열람 전 커밋) | 방법 초안 Table 2(LG 40be64c 2026-09-29 15:45:46 등) |
| WF1–WF5, WF6–WF8, WF9–WF10, WF9 토양 민감도 | designed after earlier results were viewed, registered before running | WF 계획서 커밋 4168331, 1b42b4d, 8180632, 7da0064(C4, C6, C8 README) |
| WF0 | post hoc description | WF 계획서 2절 |
| XA–XJ | designed after earlier results were viewed. XA 는 사후 분석, XC-F3 만 실행 전 등록 확인 시험 | FINAL_BATCH 개정 1 커밋(git 2678100, T0) |

**등록 이탈과 사후 변경**(원고에 모두 적는다)
1. WF9 격자 묶음: 등록 문구는 토양 √TDD 였다. 본 실행은 기온 √TDD 로 묶었다. 이탈을 기록한 뒤 토양 정의 민감도를 등록해 실행했다(C8 README 5.1 의 1).
2. 지역 내 충분 라벨 시험: 첫 시험 기각 뒤 비교 상대, 분할 수, 레시피, 라벨 범위를 바꾼 재시험을 등록했다(C4 README 1.2 (c)).
3. XG: WRAPUP 10 의 서술 SI 표 등록을 확인 대비 6개 판정으로 바꾸었다(등록 이탈 표지, FINAL_BATCH 0.4).
4. XC 평가 분할: 6–15 에서 201–210 으로 옮겼다(LGX-N4 가 1–200 을 사용, FINAL_BATCH 0.4). XC 판정어의 초록 사용안은 철회했다(7.1).
5. 초록 묶음의 AB4 인용에는 결과 전 민감도 계산(S-a)에서 같은 양의 분할 분포를 본 표지를 단다(C2 README A6).
6. v3 라벨 표의 지온 유도 20셀은 표 고정 뒤 발견되어 직접 라벨 등급에 남았고 표지로 처리하였다(D README E08, E50).
7. 첫 회수 뒤 묶음 해제 전에 커밋한 사후 개정 4건 `[DECISION: 수용 여부, 방법 초안 183행]`
8. 교차 환경 점검 (i)은 CatBoost 에서 통과하지 못해 확장 풀 행에 교차 환경 표지를 단다(C3 README 3절).
9. 관측 우선순위 지도는 등록 규칙에 따라 내지 않았다(C6 README 1.4).
10. 기후 외삽 시험은 설계 점검 누락(캐나다 외삽 영역 3블록)으로 판정 불가였다(C8 README 5.1 의 11).

### 6.15 M15 Code availability(120단어)

- 저장소 `[DECISION: 공개 저장소 주소와 공개 시점]`, 판 고정 DOI `[DECISION: Zenodo DOI]`.
- 적을 코드(코드 서체는 이 소절에서만): 하네스 `h40_label_grid.py`, `h41_validation_ladder.py`, `h42_label_grid_ext.py`, `h54_workflow.py`, 집계 `h39_scenarios.py`, 재분석 `wf0_reanalysis.py`, 새 묶음 `x_*.py`, 시험 `tests/`, Rescale 설정 `configs/rescale/`, 패키지 판(catboost 1.2.10, scikit-learn 1.9.1, pandas 2.3.3, scipy 1.17.1, numpy 2.1.1, FINAL_BATCH 1절).

### 6.16 Data availability(약 250단어, Methods 뒤, References 앞)

- 라벨: GTN-P/CALM(Streletskiy 2025, PANGAEA 972777), ALLena(Veremeeva 2025, PANGAEA 973813), ABoVE(Moore 2025, ORNL DAAC 2369), 새 지역 자료원(참고문헌 표의 자료 항목 20여 편).
- 공변량: ERA5-Land(Muñoz-Sabater 2021), SoilGrids 2.0(Poggio 2021), Copernicus DEM GLO-30 `[미확인: 인용 형식]`, ESA CCI Permafrost ALT(Westermann 2024; DOI 불일치 해결 필요, 9.3절), SAR(ReSALT ORNL DAAC 10.3334/ORNLDAAC/2004, PolSAR DOI `[미확인]`).
- 도출 자료와 그림 source data 기탁 `[DECISION: 저장소와 DOI]`. Sci Rep 은 그림 source data 제공 사실을 이 진술에 적게 한다.
- 약관 미확인 자료는 주 판에서 빼고 재배포하지 않는다. 북대서양 전체 판은 SI 에만 둔다. KPDC 자료 `[DECISION: SI 포함 여부와 식별자, 이용 조건]`

---

## 7. 표시 항목(본문 8개: Fig 1–7, Table 1)

그림 배치는 재구성안 B 와 지침 6절을 따른다. v2 그림(`outputs/figures/paper/v2/`)과 v2 모듈(`scripts/4_visualization/paper/`)은 고치지 않고, 재구성판은 비교 폴더 `outputs/figures/paper/v3_restructure/`에 따로 둔다(사용자 결정). 패널 내용, 배치(mm), 설명문 초안은 그림 명세 `outputs/figures/paper/v3_restructure/FIGURE_SPEC_v3.md`가 정하고, 아래 표의 설명문 첫 문장은 그 초안의 첫 문장과 같다. `figures/figure_spec.json` 의 `paper_v3_figN` 항목은 사용자가 v3 를 채택한 뒤 만든다(그림 명세 D-20). 설명문은 첫 문장이 주장(결과 그림, 20단어 이하) 또는 명사구(Fig 1, Table 1)이고, 350단어 이하(목표 150–250단어), 그림 안 수치·제목·판정 기호는 0개다(지침 4.4, H4). 설명문 단어 열은 그림 명세 16절의 값이며 X 치환 뒤에도 350 이하가 되도록 예산을 두었다.

| 항목 | 설명문 첫 문장(영문 초안, 그림 명세와 같음) | 패널 | 인용 절 | 그림 값 원천 | 새 실험 | 설명문 단어(초안) |
|---|---|---|---|---|---|---|
| Fig 1 | Study regions, Stefan coefficients and evaluation design. | a 범북극 지도(라벨 위치 면적 원, 새 지역 흰 원), b 지역별 E 분포(로그 축), c 레나델타 지역 홀드아웃 확대도(100 km 완충), d 같은 범위의 블록 홀드아웃 확대도(라벨 블록, 채점 블록), e 검증 사다리 | R1, R7 | `data/processed/paper_figs/fig1_*.csv`, `data/processed/fidelity_base_v3.csv`(c), `half_split_blocks`·`draw_cells`(d), XH 산출(e. 미리보기 `data/processed/m1/cv_scheme_comparison.csv`) | `[XH: Fig 1e \| 결과 열람 뒤 설계, 서술(판정어 없음) \| 2.8 사전 고정 서술 규칙]`. 확대도를 c 와 d 로 나누어 FINAL_BATCH 6절의 '패널 e'와 지침 6.1 의 확대도 2개를 함께 맞춘다(그림 명세 D-1) | 323(자리표시 제외 308) |
| Fig 2 | With ten labels, Stefan recalibration lowered the four-region mean error; no further difference was established for residual ML. | a 주 4지역 풀(라벨 10개 이하, 라벨 0 별도 축), b 레나델타·캐나다 풀(전체 라벨, 전량 별도 축), c–f 지역 4개(라벨 3개 이상) | R2, R3, R4 | `paper_figs/fig2_pool_curves.csv`, `fig2_pstar_pool.csv`, `fig2_region_curves.csv` | 없음(XB 는 SI) | 306 |
| Fig 3 | Physics pseudo-labels lowered error relative to shuffled pseudo-labels, and the better combination structure depended on the number of labels. | a 개념 좌표도(레나델타 실제 라벨, 원천·재보정 직선, 잔차), b 위약 포레스트(라벨 0개와 10개), c 결합 구조 곡선(라벨 0–160과 전량), d 현지 최소제곱 − 수축 재보정(서술) | R2, R3, R4 | `paper_figs/fig3_a.csv`, `fig3_b_L10_alln.csv`, `pool_fixed_curve.csv` | 없음 | 268 |
| Fig 4 | The error reduction of label-based correction was rank-correlated with the source-coefficient bias measured with ten labels. | a 대상별 최소 라벨 수(라벨 10개 편향 순, 지침 6.4 의 계수 오차 순과 다름, 그림 명세 D-8), b 라벨 10개 편향 대 대상 라벨 교차검증 선정의 원천 계수 Stefan 대비 오차 변화(라벨 전량, 28대상, 두 CI 막대), c 규칙 W − 고정 레시피 포레스트(라벨 10개와 전량은 주 4지역, 40개와 160개는 레나델타·캐나다), d XA 분해 | R3, R4 | `paper_figs/fig4_a.csv`, `results/rescale_wf/data/processed/wf/wf_tests.csv`(WF4-a, WF4-c 지역 행), `wf_curve.csv`(`method == 'W'`, `n == -1`), XA 산출(d. 미리보기 `paper/claims/C2_bias_diagnosis/tables/derived_c2_wf4c_split.csv`) | `[XA: Fig 4d \| 사후 분석(재현(비맹검)) \| 2.1 결과 범주(양, 확인하지 못함, 음)와 CE1, CE2 판]`. XA 가 없으면 d 는 SI | 310(자리표시 제외 295) |
| Fig 5 | Spreading labels across blocks and covariate space lowered error versus random cells in Canada, not in the Lena Delta. | a–d 캐나다 같은 예산의 전략별 선정 지점 지도(무작위, 블록 층화, 공변량 최대 최소, 능동 선정), e 예산 대 ΔRMSE(캐나다, 레나델타, 알래스카, 점과 두 막대), f 전이 라벨 10개와 40개 포레스트 | R6 | `results/rescale_wf/data/processed/wf/wf_tests.csv`(WF2-a), `results/rescale_wf2/data/processed/wf/wf2b_tests.csv`(WF8), 선정 지점은 라벨 값 없이 seed 로 재현(d 는 로컬 재선정 승인 뒤, 승인이 없으면 a–c 3열, 그림 명세 D-12) | 없음(XD 는 SI) | 343 |
| Fig 6 | Without target labels, 2 of 30 targets lost over 2 cm with physics pseudo-labels and 18 with direct ML. | a, b 대상별 오차 변화 지도(직접 ML, 물리 유사라벨, 공유 발산 색표, 사후 서술), c 학습기 포레스트(물리 기준선, 부스팅과 랜덤 포레스트, 표형 파운데이션 모델, 신경망), d 구간 척도화 5종의 포함률 대 구간 폭(라벨 0), e 라벨 40개와 160개 잔차 구간의 포함률 대 폭 비(관측 대 예측 패널은 셀 단위 구간 파일이 없어 쓰지 않는다, 그림 명세 D-16) | R2, R8 | `paper_figs/fig6_a.csv`, `fig6_b.csv`, `fig6_d.csv`, 대상별 Δ(`results/rescale_lg/data/processed/lg/lg_curve.csv`, `data/processed/lgd/lgd_curve_lic.csv` 를 `wf0_reanalysis.py` 로 읽기 전용 집계), 대상 중심 좌표 `[미확인]` | `[XG: Fig 6c \| 결과 열람 뒤 설계, 등록 이탈(WRAPUP 10) \| 2.7 XG-1 행]`. 계층 conformal 0.86 은 규약이 달라 Fig 6d 에 두지 않고 Supplementary Table S12 에 둔다 | 306(자리표시 제외 291) |
| Fig 7 | Within Alaska, residual ML gains over recalibrated Stefan were below 1 cm and came from climate-grid-scale bias correction. | a 지역 내 재보정 Stefan 대비 오차 변화(알래스카, 레나델타, 캐나다, 점과 두 막대)와 '오차 하한 − 재보정 Stefan RMSE' 선(그림 명세 D-21, 절대 RMSE 곡선은 SI), b 알래스카 지도(기본: 채점 블록별 오차 변화. 1 km 보정 표시는 새 적합·예측 실행 승인 뒤, 그림 명세 D-11), c 격자 사이·안 분해 점 그림, d 워크플로 경로(레나델타·캐나다 전이 풀, 그림 명세 D-13, D-14), e XC 포레스트 | R4, R5, R6 | `results/rescale_wf2/data/processed/wf/wf2b_tests.csv`(WF6-a), `data/processed/lgx/lgx_floor.csv`, `results/rescale_wf3/data/processed/wf/wf3b_tests.csv`, `wf9__*_blocksse.npz`(b 기본판), `results/rescale_wf/data/processed/wf/wf_curve.csv`(d 의 규칙 W 값) | `[XC: Fig 7e \| 결과 열람 뒤 설계, 재사용 지역의 재검정(비맹검 부분 포함) \| 2.3 XC-1, XC-2 포레스트]`. XC 가 없으면 d 를 전폭으로 | 327(자리표시 제외 310) |
| Table 1 | Target regions, labels and data sources. | 열: Region, Label source, Labels, 1 km locations, 0.5° blocks, Splits, Source coefficient $E_0$ (cm (°C d)$^{-1/2}$), Alaska share of source cells. 묶음: Main regions, Reference, New regions | R1, R7 | `paper_figs/table1_rows.csv`, `fig1_source.csv`, D README E10–E19, E35, E37 | 없음(XF 는 FINAL_BATCH 6절에 따라 Supplementary Table S18, 그림 명세 D-5) | 136 |

- 설명문 공통 요소: 구간 종류와 재표집(블록 재표집 10,000회, 두 가중), 동등 띠 ±0.5 cm, 단위 수(라벨 n, 지역 수, 대상 수), 사후 서술 표지(Fig 6a,b), 결과 열람 뒤 설계 표지(Fig 4b,c, Fig 5, Fig 7), 지도의 투영·기준 위도·해안선 자료·영구동토 구역 자료·소프트웨어 판(지침 4.4. Fig 1, 5, 6, 7 설명문에 모두 넣었다).
- 결과 지도: Fig 5, Fig 6, Fig 7 이 본문 결과 지도이고, 결과 절마다 지도 2종 이상(본문 또는 SI)을 둔다(지침 1.2 표, H5). SI 지도는 8절 목록에 있다.
- 워크플로 경로(Fig 7d)의 구간 이름은 시험 격자 값(0, 3–10, 40–160, 320–1000, 전량)이다. claims 4절의 '1–10, 10–160, 160–1,000, 1,000 이상'과 맞추는 결정이 남아 있다(지침 8절 14). `[DECISION: 워크플로 구간 이름]`

---

## 8. Supplementary Information 목록

하나의 PDF 로 묶는다. 번호는 'Supplementary Fig. S1', 'Supplementary Table S1'이고, 본문에서 SI 그림의 패널을 가리키지 않는다. 'data not shown'은 쓰지 않는다(지침 4.1).

**Supplementary Methods**
1. 라벨 파싱 규칙과 자료원별 처리(방법 초안 Observational data 이관분)
2. 새 지역 조립과 적격 판정(LGD, XF)
3. 확장 대비 목록(LGX)과 결합 구조 세부
4. 예측 구간(LGU)
5. 표형 파운데이션 모델과 원천 교차검증 신경망 조정(LGT, LGF)
6. 교차 환경 규칙과 등록 시각 표(지원 초안 M4.2)
7. 라벨 단위와 연도 민감도(S-a, S-b), 순차 멈춤 규칙(SC3w)
8. 배치 전략 S1–S8 세부와 학습된 배치 정책(XD)
9. 추가 등록 분석 XA–XJ 세부

**Supplementary Notes**: Stefan 계수 E 의 물리적 의미(지원 초안 M3.2 D1–D8), 대회 시험과 이번 시험의 조건 차이(QA_FINAL_REVIEW Q14).

**Supplementary Tables**(번호는 원고 작업에서 확정)

| 번호(안) | 내용 | 원천 |
|---|---|---|
| S1 | 하위 지역 정의(AL-1–AL-6, CA-2, CA-3, LE-1, LE-2) | 방법 초안 Regions |
| S2 | 등록 가설 전체: 대비, Δ, 두 가중 CI, 보정 전 p, Holm p, 판정어, 등록 구분, 실험 번호(지침 3.3) | `paper/registry/experiments.csv`, 각 README |
| S3 | 초록 대비 묶음 AB1–AB10(블록 등가중 CI, 보정 전 p 포함) | `data/processed/lgw/lgw_bundle.csv` |
| S4 | 라벨 0 학습기 10종(두 CI, 공통 CI, 플랫폼) | C1 README 2.1, 2.2 |
| S5 | 라벨 0 위험표(2 cm 문턱, CI 기준 수, 독립 지역만) | C1 README 2.5, 2.8 |
| S6 | 결합 구조와 증강 결합(L10, L2, L12, L17) | C3 README 2절 |
| S7 | 지역 내 라벨 수 시험(WF1, WF6, SAR) | C4 README 2.1–2.5 |
| S8 | 규칙 W 와 편향 진단(WF4-a, WF4-b, WF4-c, 선택 횟수) | C5 README 2절, C2 README 2.1 |
| S9 | 배치 전략(WF2, WF8, L43) | C6 README 2절 |
| S10 | 이득 분해와 기후 외삽(WF9, WF10) | C8 README 2절 |
| S11 | 새 지역(L1e, L4e, L8e, L38, L39) | SI_learners README 2.5 |
| S12 | 예측 구간의 단계별 커버리지(S11 → H17 → LGU-B1 → 계층 conformal) | SI_uncertainty README 4절 권고 |
| S13 | XA–XJ 결과(실행 뒤) | FINAL_BATCH 8절 |
| S14 | 배포 시나리오 SC1w–SC3w | 결과 초안 Deployment |
| S15 | 선택·철회 이력과 과거 음성 결과 | 지원 초안 M4.4, M4.5 |
| S16 | 소프트웨어와 실행 환경 | 방법 초안 Computing environments |
| S17 | 자료원과 약관, 제외 자료 | 방법 초안 Data availability |
| S18 | 공개 자료 독립 지역 조사(XF) | `docs/research/2026-10-04/open_regions.md` |

**Supplementary Figures**(지침 1.2 결과 지도 계획과 그림 재구성에서 밀려난 패널)
- R1: 검증 사다리 분할 지도, 알래스카 채점 방식별 거리 분포 짝(Meyer & Pebesma 2022 형식)
- R2: 저가중 잔차판과 앵커 + 잔차판 대상별 오차 변화 지도, 학습기별 세부
- R3: 대상별 원천 계수 오차 지도, 라벨 10개 재보정 이득 지도, 토양 보정 Stefan 곡선, 지역별 대비 포레스트
- R4: 대상별 규칙 선정 결과 지도, 규칙 − 고정 레시피 대상별 지도, 원천 교차검증 조정 행렬과 순위 산점도
- R5: 채점 블록별 오차 변화 지도(레나델타, 캐나다), XE, XI, XB 그림
- R6: 대상별 전략 − 무작위 지도, 근접·거리 대비(L23, L24), 학습된 배치 정책(XD)
- R7: 독립 지역 대상별 오차 변화 지도와 예측 지도
- R8: 예측 ALT, 90% 구간 폭, 외삽 영역 지도(레나델타, `outputs/maps/transfer_lena/`)
- v2 Fig 7 의 시나리오 행렬, 멈춤 규칙 곡선, 초록 대비 포레스트

---

## 9. 참고문헌 계획

### 9.1 원칙

- Nature 번호식(`sn-nature.bst`), 영문·국문 공용 `paper/manuscript/shared/refs.bib` 하나.
- 출발점은 방법 초안의 참고문헌 표 67행(`docs/MANUSCRIPT_DRAFT_METHODS_INTRO_2026-09-30.md` 225–297행)과 `references/INDEX.md`(262항목)다. 서지가 확인되지 않은 항목은 인용하지 않는다. 인용은 원문 PDF 나 출판사 쪽으로 확인한 뒤에만 한다.
- 권고치 60개는 엄격 적용이 아니다(Sci Rep). 예시 10편의 중앙값은 71개다(scirep_ex 4.10).

### 9.2 추가하는 4편(서지는 scirep_fmt 4.10 에서 Crossref·출판사로 확인)

| 문헌 | 서지 | 쓰는 곳 | INDEX |
|---|---|---|---|
| Ploton et al., 2020 | Ploton, P. et al. Spatial validation reveals poor predictive performance of large-scale ecological mapping models. Nat. Commun. 11, 4540 (2020). https://doi.org/10.1038/s41467-020-18321-y | I2, R1, D3 | `ploton2020_spatial_validation`(PDF) |
| Meyer & Pebesma, 2022 | Meyer, H. & Pebesma, E. Machine learning-based global maps of ecological variables and the challenge of assessing them. Nat. Commun. 13, 2208 (2022). https://doi.org/10.1038/s41467-022-29838-9 | I2, D3, SI 거리 분포 | `meyer2022_globalmaps_aoa`(PDF) |
| Wadoux et al., 2021 | Wadoux, A. M. J.-C., Heuvelink, G. B. M., de Bruin, S. & Brus, D. J. Spatial cross-validation is not the right way to evaluate map accuracy. Ecol. Model. 457, 109692 (2021). https://doi.org/10.1016/j.ecolmodel.2021.109692 | I2, R1(XH 문장), D3 | `wadoux2021_spatial_cv_map_accuracy_ecolmodel`(링크, 원문 미대조) |
| Linnenbrink et al., 2024 | Linnenbrink, J., Milà, C., Ludwig, M. & Meyer, H. kNNDM CV: k-fold nearest-neighbour distance matching cross-validation for map accuracy estimation. Geosci. Model Dev. 17, 5897–5912 (2024). https://doi.org/10.5194/gmd-17-5897-2024 | R1, M9, D3 | `linnenbrink2024_knndm`(PDF) |

Meyer & Pebesma 2021(Methods Ecol. Evol. 12, 1620–1633)은 대회 보고서에 이미 있고 scirep_fmt 4.10 의 추가 5편에 들어 있다. 넣으면 72편이 된다. `[DECISION: Meyer & Pebesma 2021 포함 여부]`

### 9.3 인용 전 고칠 항목

| 항목 | 문제 | 처리 | 근거 |
|---|---|---|---|
| Stefan 1891 | 표에 [verify] | INDEX 가 Ann. Phys. 278(2), 269–286, doi:10.1002/andp.18912780206 으로 확인. 표를 고친다 | INDEX 12절 |
| Ran et al., 2022a | 공저자 일부 오기 | INDEX 정정 목록으로 저자를 쓴다 | INDEX 경고 절 |
| Westermann et al., 2024(CCI) | 원고 표 DOI 와 INDEX 01절 DOI 가 다르다 | 어느 DOI 가 ALT 변수인지 확인 뒤 하나로 | INDEX 경고 절 |
| Streletskiy et al., 2025 | 저자 목록 5명 대 6명 | 출판 기록으로 확인 | 방법 초안 표 |
| Pilyugina et al., 2025 | 분류(열방정식 정규화 대 물리 출력 입력) | 원문으로 확인 뒤 I2 문장 확정 | 방법 초안 표 |
| Qu et al., 2026 | 학회(ICML) 미확인 | 확인 전에는 arXiv 로 인용 | 방법 초안 표 |
| Du et al., 2026; Kudryavtsev et al., 1974; Pekel et al., 2016 | 서지 미확인 | 확인하거나 SI 로 | scirep_fmt 4.10 |
| Moore et al., 2025 | DataCite 2026, ORNL 2025 | ORNL 인용 형식을 따른다 | 방법 초안 표 |
| Gautam et al., 2025 수치 | R² 0.24, 0.54 의 원문 확인 수준 | QA_FOLLOWUP 3.3 은 수준 A(원문 PDF), C7 README 는 [미확인]. 원문 쪽수를 확인해 기록한다 | QA_FOLLOWUP 3.3, C7 README 2.10 |

### 9.4 개수 계획

| 묶음 | 대략 개수 | 처리 |
|---|---|---|
| 서론·고찰의 과학·방법 선례 | 약 30 | 본문 |
| 방법(학습기, 통계, 공변량) | 약 20 | 본문. LGU 생성 모형(Durkan 2019, Ho 2020, Lipman 2023)과 학습기 세부(Gorishniy 2021, 2025, Holzmüller 2024, Bergstra 2012, Cawley 2010)는 Supplementary Methods 로 옮기는 안 |
| 자료 인용(새 지역 자료원 'meta' 20여 행) | 약 21 | 본문 Data availability 또는 SI 참고문헌 |
| 추가 4편 | 4 | 본문 |

- 67 + 4 = 71편이다. 선택지: (a) 그대로 내고 표지 편지에 자료 인용이 많다고 적는다. (b) 새 지역 자료 인용과 SI 전용 방법 인용을 SI 참고문헌으로 옮긴다[미확인: Sci Rep 의 SI 참고문헌 허용]. `[DECISION: 참고문헌 처리]`
- 후보(INDEX 에 있으나 표에 없음, 쓰기 전 확인): Riseborough 2008, Shiklomanov & Nelson 2002(D2 E 해석), Crameri 2020(M13 색표), Valavi 2019(블록 교차검증 도구), Romano 2019(CQR, SI), Tama 2025(물리 앵커 + 잔차 선례), Wei 2026(ESSD Discussions, 제품 비교 XG), Hollmann 2025(이미 표에 있음). Klene 2001, Romanovsky & Osterkamp 1997 은 INDEX 에 없다[미확인].
- 형식 예시 논문(Hall 2026, Fisher 2023 등)은 문장 형식의 참고이고 인용 대상이 아니다.

---

## 10. 단어 예산과 Sci Rep 한도 대조

| 항목 | Sci Rep 한도 | 계획 | 여유와 사용처 |
|---|---|---|---|
| Title | 20단어 | 14 | |
| Abstract | 200단어 | 199(판 B), 196(판 A) | XC 수치를 넣으면 S7 또는 S8 을 대신한다 |
| Introduction | 본문 합계에 포함 | 700(I1 170, I2 210, I3 120, I4 200) | |
| Results | 본문 합계에 포함 | 2800(R1 300, R2 400, R3 450, R4 350, R5 450, R6 400, R7 250, R8 200) | |
| Discussion | 본문 합계에 포함 | 850(D1 170, D2 150, D3 160, D4 110, D5 160, D6 110) | |
| 본문 합계 | 4500단어 | 4350 | 150단어: XA 40(R3), XC 60(R4 40, R6 20), XG 30(R2), XH 20(R1) |
| Methods | 한도 없음 | 약 4240 | 11쪽 권고를 넘으면 M4, M7, M11 세부를 Supplementary Methods 로 |
| Data availability | 별도 | 약 250 | |
| Figure legends | 그림마다 350단어 | 268–343(7절 표, 그림 명세 16절) | 목표 150–250 은 넘는다. X 치환 뒤에도 350 이하가 되도록 치환 예산을 두었고, 원고 조립 때 방법 정의 문장부터 Methods 로 옮겨 줄인다 |
| Table 1 설명문 | 표 한 쪽 이내 | 150단어 이하 | |
| Keywords | 6개 | 6 | |
| References | 60개 권고 | 71–72 | 9.4절 |
| 조판 쪽수 | 11쪽 권고 | 추정 16–17쪽(지침 4.1) | 첫 컴파일 뒤 재측정 |

- 단어 수는 렌더 PDF 에서 `pdftotext` 뒤 절 경계로 잘라 `wc -w`로 센다. TeX 소스 카운트는 쓰지 않는다(M-03, MS-06).
- 기존 초안 대비: 서론 797 + 결과 7500 = 8297단어였다(scirep_fmt 3.3). 결과 초안의 LG 가설 순서 문단은 4절 구성으로 다시 쓰고, 재사용 가능한 문단은 scirep_fmt 3.5 표를 따른다.

---

## 11. 새 실험 자리표시 등록부

결과가 나오면 FINAL_BATCH 의 해당 해석 문장으로만 바꾼다. 본문/SI 배치는 결과에 따라 바꾸지 않는다(FINAL_BATCH 6절).

| 실험 | 본문/SI(결과 전 고정) | 자리(이 문서의 절) | 초록 |
|---|---|---|---|
| XA | 본문 | R3(4.3), Fig 4d(7절), D2 | 수치만 |
| XB | SI | R3, R4, R5 의 SI 지시 문장 | 없음 |
| XC | 본문 | R4, R6(Fig 7e 의 배치 묶음), R7(XC-F3), Fig 7e, D6, 초록 S8 | 수치만 또는 제외 |
| XD | SI(알고리즘 정의는 Methods) | R6 SI 지시 문장, M7 | 없음 |
| XE | SI | R5 SI 지시 문장, D6, 초록 S8 | 없음 |
| XF | SI(Supplementary Table S18) | R7, D6. Table 1 에 행을 더하지 않는다 | 없음 |
| XG | 본문 | R2, Fig 6c, D6 | 없음 |
| XH | 본문 | R1, Fig 1e | 없음 |
| XI | SI | R5 SI 지시 문장 | 없음 |
| XJ | SI 한 줄 | R7 SI 한 줄 | 없음 |

자리표시 원문(원고 초안에 그대로 넣는다. 2, 4, 5, 6, 7절의 자리표시와 그림 명세 설명문, 덱 스펙 `placeholders_registry` 의 문자열은 이 목록과 글자 하나까지 같다. 결과 절 R3 의 XA 는 FINAL_BATCH 0.4 의 등록 문자열 그대로다)

```text
[XA: R3 | 사후 분석(재현(비맹검)) | 2.1 해석 조각]
[XA: Fig 4d | 사후 분석(재현(비맹검)) | 2.1 결과 범주(양, 확인하지 못함, 음)와 CE1, CE2 판]
[XA: Discussion D2 | 사후 분석(재현(비맹검)) | 2.1 해석 조각 XA-3]
[XB: SI, R3 지시 문장 | 결과 열람 뒤 설계, 실행 전 등록(비맹검 부분 포함) | 2.2 사전 고정 해석 문장 XB-1, XB-2(라벨 10개 행)]
[XB: SI, R4 지시 문장 | 결과 열람 뒤 설계, 실행 전 등록(비맹검 부분 포함) | 2.2 사전 고정 해석 문장 XB-1, XB-2(라벨 40, 160 행, 레나·캐나다 2지역)]
[XB: SI, R5 지시 문장 | 결과 열람 뒤 설계, 실행 전 등록(비맹검 부분 포함) | 2.2 사전 고정 해석 문장 XB-3, XB-4]
[XC: R4 | 결과 열람 뒤 설계, 재사용 지역의 재검정(비맹검 부분 포함) | 2.3 사전 고정 해석 문장 XC-1, XC-2(+XC-2s), 뒤에 S-XC4 (b) 지역 손해 문장]
[XC: R6 | 결과 열람 뒤 설계, 재사용 지역의 재검정(비맹검 부분 포함) | 2.3 사전 고정 해석 문장 XC-5와 공통 문장]
[XC: R7 | 실행 전 등록 확인 시험(맹검) | 2.3 XC-F3 갈래]
[XC: Fig 7e | 결과 열람 뒤 설계, 재사용 지역의 재검정(비맹검 부분 포함) | 2.3 XC-1, XC-2 포레스트]
[XC: Discussion D6 | 결과 열람 뒤 설계, 재사용 지역의 재검정(비맹검 부분 포함) | 2.3 XC-1, XC-2]
[XC: Abstract S8 | 결과 열람 뒤 설계, 재사용 지역의 재검정(비맹검 부분 포함) | 2.3 초록 규칙: 수치만, '개발에 쓴 지역을 다시 나눈 분할', 지역 2곳, 지역별 방향(우세 k, 열세 m)]
[XD: R6 SI 지시 문장 | 결과 열람 뒤 설계, 탐색(SI), XD-2 사후 설계(지역 표적) | 2.4 사전 고정 해석 문장(SI) XD-1, XD-3, 공통 한계 문장]
[XD: Methods M7 | 결과 열람 뒤 설계, 부록 XC-0 의 기계적 선택(알래스카 계열) | 2.4 고른 후보 이름과 γ]
[XE: R5 SI 지시 문장 | 결과 열람 뒤 설계 | 2.5 사전 고정 해석 문장(6갈래)과 공통 문장]
[XE: Discussion D6 | 결과 열람 뒤 설계 | 2.5 해석 문장]
[XE: Abstract S8 | 결과 열람 뒤 설계 | 2.5 사전 고정 해석 문장(첫째 또는 넷째 갈래)]
[XF: R7 | 결과 열람 뒤 설계, 새 지역은 맹검(독립 지역 확인), NAtlantic v5 는 민감도 | 2.6 사전 고정 해석 문장]
[XF: Discussion D6 | 결과 열람 뒤 설계, 새 지역은 맹검(독립 지역 확인) | 2.6 해석 문장]
[XG: R2 | 결과 열람 뒤 설계, 등록 이탈(WRAPUP 10) | 2.7 사전 고정 해석 문장 XG-1, XG-2, XG-3, 공통 문장]
[XG: Fig 6c | 결과 열람 뒤 설계, 등록 이탈(WRAPUP 10) | 2.7 XG-1 행]
[XG: Discussion D6 | 결과 열람 뒤 설계, 등록 이탈(WRAPUP 10) | 2.7 공통 문장]
[XH: R1 | 결과 열람 뒤 설계, 서술(판정어 없음). L19, L20 은 등록 | 2.8 사전 고정 서술 규칙]
[XH: Fig 1e | 결과 열람 뒤 설계, 서술(판정어 없음) | 2.8 사전 고정 서술 규칙]
[XI: R5 SI 지시 문장 | 결과 열람 뒤 설계 | 2.9 다섯 갈래, 공간 대용과 외삽 폭]
[XJ: R7 SI 한 줄 | 결과 열람 뒤 설계, 서술 | 2.10 사전 고정 해석 문장]
```

---

## 12. 원고 QA(제출 전, 지침 7.3)

| ID | 점검 | 기준 |
|---|---|---|
| M-01 | 연결어 대시 | em dash, en dash 0건(숫자 범위 제외). digest 5.1 정규식 |
| M-02 | 금지 표현 | 지침 4.6, 4.8 목록과 grep 0건. 'safe', 'all ten learners', 'reversal', 'most honest', 'ML adds spatial detail', 'high-resolution ALT map', 'pre-registered'(단독), 가운뎃점 숫자 나열 |
| M-03 | 분량 | 10절 표. 렌더 PDF 기준 |
| M-04 | 통계 | 'significant' 옆 *P*, Methods 에 α, 양측 여부, Holm. 평균 Δ 에 중앙값, 유의 개선·악화 대상 수, 러시아 서부를 뺀 값 `[MISSING: 중앙값과 러시아 서부 제외 값, build_numbers.py 에서 lgw_bundle.csv 지역 행으로 생성]`. 단정형은 두 CI 일치 대비만 |
| M-05 | 국문 종결 | '입니다', '합니다', '이지만' 0건 |
| M-06 | 조사 띄어쓰기 | 라틴 문자·숫자·단위 뒤 조사 앞 공백 0건, " ," 0건, 괄호 안쪽 공백 0건 |
| M-07 | 약어와 약호 | 첫 등장 풀이, 방법 약호(P0, P1, R1, D0, D1, W)와 실험 번호 0건(Methods 등록 소절과 SI 제외) |
| M-08 | 서술 방향 | 초록 첫 문장 워크플로, 첫 결과 문장 설계, R2 첫 문장 위약 대조, D6 가 적용 조건과 다음 자료 |
| M-09 | 수치 | 모든 결과 수치가 매크로, 초록 Δ·CI 쌍 2개 |
| M-10 | 서체 | 본문 굵게·기울임 강조 0건, `\texttt`는 Code availability 안 |
| M-11 | 제출 형식 | 단일 `.tex`, xelatex 와 pdflatex 오류·경고 0, PDF 3 MB 이하, SI 단일 PDF |
| M-12 | SI 인용 | 'Supplementary' 접두 100%, SI 패널 인용 0건, 'data not shown' 0건 |
| 추가 | 자리표시 | `[X?:`, `[DECISION`, `[MISSING`, `[미확인]` 0건 |

기준선(baseline 감사)의 원고 관련 결함을 넘었는지 함께 본다: 본문 8297단어 → 4500 이하, 초록의 AB 나열 → 워크플로 첫 문장, 내부 코드와 가설 번호의 본문 노출 → 0건, 덱 PDF 에서 보인 " ,"·"( " 공백 → 0건.

---

## 13. 열린 결정과 이 문서가 정한 충돌 처리

### 13.1 사용자 결정 대기

1. `[DECISION: 최종 제목]`(임시 제목 유지 중)
2. `[DECISION: 초록 판 A 또는 B]`와 판 B 의 WF 수치 개수(지침 4.2, 8절 4 와의 충돌)
3. `[DECISION: 저자, 소속, 교신 저자]`. 후보 기록만 둔다: 제1 저자 백승원(Seungwon Baek), KOPRI 공동 연구 가능자 김승희(Seung Hee Kim), 정윤택(Yoon Taek Jung). 저자 자격은 이 문서가 정하지 않는다.
4. `[DECISION: LLM 사용 기재 문구]`(M14)
5. `[DECISION: Acknowledgements, Author contributions, Competing interests 문구]`
6. `[DECISION: 참고문헌 처리]`, `[DECISION: Meyer & Pebesma 2021 포함 여부]`
7. `[DECISION: 지역 이름 표기]`, 레나델타 범위
8. `[DECISION: 워크플로 구간 이름]`
9. `[DECISION: 러시아 중부의 영문 표기]`
10. `[DECISION: 공개 저장소와 DOI]`, `[DECISION: KPDC 자료]`
11. `[DECISION: 사후 개정 4건 수용 여부]`
12. 대회 보고서의 선행 공개: 표지 편지에 적고 사본을 첨부한다(scirep_fmt 2.2). 본문 인용 방식 `[DECISION: 대회 보고서 인용 여부]`
13. 그림 배치 안 A(v2 유지)와 안 B(재구성)의 최종 선택. 이 명세는 안 B 를 전제로 썼다(지침 6.0).
14. `[DECISION: Fig 7d 경로의 y 기준과 풀]`(그림 명세 D-13, D-14). 기본값은 지침 6.7 d 의 레나델타·캐나다 전이 풀이고 라벨 3–160 의 권고 방법이 원천 계수 Stefan 보다 0 위에 놓인다. 대안은 y 를 재보정 Stefan 또는 고정 레시피 대비로, 또는 라벨 10개까지 주 4지역 풀을 쓰는 혼합 풀이다. 5절 D1 과 덱 S26 이 같은 기본값을 쓴다
15. `[DECISION: Fig 7b 판]`(그림 명세 D-11). 기본은 채점 블록별 오차 변화 지도이고, 1 km 보정 표시는 알래스카 전량 적합과 1 km 예측의 새 실행 승인 뒤다
16. `[DECISION: Fig 5d 능동 선정 지도]`(그림 명세 D-12). 로컬 1분할 재선정을 승인하지 않으면 Fig 5 는 a–c 3열이다

### 13.2 이 문서가 정한 충돌 처리(사용자가 되돌릴 수 있다)

| 충돌 | 처리 | 근거 |
|---|---|---|
| 본문 기호: 지침 H3(방법 약호 금지) 대 scirep_fmt 4.7, FINAL_BATCH 6절(P0, P1, R1 허용) | 지침을 따른다. 방법 이름을 쓴다 | 지침이 늦고 구속 문서 |
| 결과 절 구성: 재구성안 4절(R7 = C8) 대 지침 4.3(R5 = C4 + C8, R7 = 독립 지역) | 지침 4.3 | 과제 지시('noun-phrase headings per guide 4.3') |
| XB 를 Fig 2b 선으로(지침 6.2), XD 를 Fig 5f 행으로(지침 6.5) 대 FINAL_BATCH 6절(XB, XD 는 SI) | FINAL_BATCH 6절 | 결과 전 고정한 배치가 우선, 결과에 따라 올리지 않는다 |
| XH 패널: FINAL_BATCH 'Fig 1 패널 e' 대 지침 6.1 '패널 d' | 패널 e. 확대도 2개를 c, d 로 나누어 두 문서를 함께 맞춘다(그림 명세 D-1, 패널 5개) | 그림 명세와 원고 명세가 같은 패널 문자를 쓴다 |
| XG 행: FINAL_BATCH 'Fig 6a 행' 대 지침 6.6(학습기 포레스트는 c) | Fig 6c 의 새 묶음 'Existing ALT maps' | v3 배치에서 포레스트가 c(그림 명세 D-6) |
| XF: 지침 6.8 'Table 1 에 새 지역 행' 대 FINAL_BATCH 6절 'Supplementary Table' | Table 1 에 행을 더하지 않고 Supplementary Table S18 | 결과 전 배치 고정이 우선(그림 명세 D-5) |
| R8 의 계층 conformal 0.86 위치: 초판 명세 'Fig 6d' 대 그림 명세(09-26 규약 참조값은 SI) | Supplementary Table S12 를 인용하고 Fig 6d 는 구간 척도화 5종 | `fig6_b_ref.csv` 의 `use` 열('LGU 와 CI 를 나란히 비교하지 않는다') |
| Fig 7d 워크플로 경로: 지침 6.7 d(레나델타·캐나다 풀) 대 덱 초판(라벨 10개까지 주 4지역 풀) | 지침 값(레나델타·캐나다 풀)을 기본값으로, 혼합 풀과 y 기준 변경은 대안으로 | 그림 명세 D-13, D-14. 세 명세가 같은 기본값을 쓴다 |
| Fig 4b 대상 수: 지침 6.4 '30 targets' 대 C2 README A3 '28대상' | 28대상 | 원천 CSV(`wf_tests.csv` WF4-c MEAN, `n_targets` 28) |
| '독립 지역 확인' 표지: WRAPUP 1.2(LGD 새 지역) 대 FINAL_BATCH 1절(10-04 조사 뒤 지역만) | 러시아 중부에 'licence-verified new region', 확인 표지 없음 | 더 늦고 좁은 규칙 |
| 라벨 0 의 Stefan·CCI 앵커 권고(claims 4절) 대 AB2 미결정(scirep_fmt 3.6 충돌 8) | 워크플로 라벨 0 행에서 CCI 앵커를 권하지 않는다. 쓰면 '서술, 등록 대비 미결정' 병기 | scirep_fmt 3.6 |
| C4 '계수가 맞는 지역일수록 작다' | 쓰지 않는다 | C4 README 1.2 (a) |
| C6 '어떤 지역에서도 나빠지지 않았다(안전)' | 정정 문장 사용 | C6 README 1.2 (d), FINAL_BATCH 0.4 |

---

## 14. 검토 반영(2026-10-04, 원고·그림·덱 명세 교차 검토)

세 명세(이 문서, `outputs/figures/paper/v3_restructure/FIGURE_SPEC_v3.md`, `deck/deck_spec_paper_report.json`)를 수치, 그림·패널 참조, 결과 절 순서, 주장 문구, Sci Rep 한도, 지침 금지 사항 기준으로 대조하고 이 문서를 고쳤다.

**원천 CSV 재확인(42개 값, 모두 일치)**: `data/processed/lgw/lgw_bundle.csv`(AB1–AB10 주 4지역 평균, AB4 지역 행 4개), `results/rescale_wf/data/processed/wf/wf_tests.csv`(WF4-a 라벨 10, 40, 160, 전량과 지역 행, WF4-c ρ 0.38 [0.17, 0.55] 28대상과 부호 ρ, 티베트 진단 227.71과 이득 159.55, WF2-a 캐나다·레나델타 블록 층화와 공변량 분산, 능동 선정), `results/rescale_wf2/data/processed/wf/wf2b_tests.csv`(WF6-a 알래스카 라벨 500, 1000, 전량과 RMSE 13.87과 14.41, WF8 캐나다 −2.39, 레나델타 +1.89), `results/rescale_wf3/data/processed/wf/wf3b_tests.csv`·`wf3b_decomp.csv`(격자 사이 −1.03, 격자 안 −0.01, 층화 +0.09, SSE 이득 격자 안 몫 0.9 %, 격자 안 비율 72 %), `data/processed/lgx/ladder/lgv_tests.csv`·`lgv_metrics.csv`(검증 사다리 8단, +10.34, +8.22, 최소제곱 Stefan 26.81–26.96), `data/processed/lgx/lgx_tests.csv`(L15 위약 4종, L28 +3.24, L29 −1.00 Holm 0.0056, L30 랜덤 포레스트 +2.63, L10 구조 대비), `data/processed/lgf/lgfn_tests.csv`(신경망 4종), `data/processed/paper_figs/table1_rows.csv`·`fig1_z_summary.csv`·`fig1_source.csv`·`fig1_not_drawn.csv`, `data/processed/lgx/lgx_floor.csv`, `data/processed/wf/wf0_risk.csv`, `data/processed/lgd/lgd_tests_lic.csv`(러시아 중부, 5지역 풀), `data/processed/paper_figs/pool_fixed_curve.csv`, `fig2_region_curves.csv`, `data/processed/s4_residual_results.csv`. 러시아 중부 R1 − P0(−5.42 [−8.96, −1.64])는 L38 행 값이고 L8e 행은 재표집 판이 달라 CI 가 다르다(덱 스펙의 원천 필터를 고쳤다).

**고친 것**
1. 초록 판 B 의 수치를 지침 4.5 자릿수로 바꿨다(−1.32 → −1.3, 13.87과 14.41 → 13.9와 14.4, −2.54 → −2.5, +1.65 → +1.7). '이미 그 자릿수로 적었다'는 문장과 실제 값이 달랐다. 'one of four regions'를 'lower in one region'으로 바꿔 뜻을 분명히 했다(199단어 유지). 판 A 의 AB9 문장을 'lower error ... by less than 0.5 cm'로 고쳤다(196단어).
2. Fig 1 패널을 그림 명세와 맞췄다(c 지역 홀드아웃 확대도, d 블록 홀드아웃 확대도, e 검증 사다리). R1 의 그림 참조(Fig. 1c,d, Fig. 1e), XH 자리표시(Fig 1e), 13.2 충돌 표를 고쳤다.
3. 7절 표시 항목 표를 그림 명세의 패널과 설명문 첫 문장으로 다시 썼다. Fig 2 제목은 근거가 없는 'residual ML added less than 0.5 cm'(AB5 CI 하한 −0.53)를 AB5 초록 규칙 (b) 문구로 바꿨다. Fig 4b 대상 수 30 → 28(C2 README A3, `wf_tests.csv` `n_targets` 28), a 정렬 변수를 라벨 10개 편향으로(그림 명세 D-8), c 를 라벨 10개와 전량(주 4지역), 라벨 40개와 160개(2지역)로 고쳤다. Fig 6 제목은 20단어 이하의 사후 서술 문장으로, e 는 관측 대 예측이 아니라 라벨 있는 잔차 구간(그림 명세 D-16)으로 고쳤다. Fig 7 a 는 재보정 Stefan 대비 오차 변화와 하한 선(D-21), b 는 기본판 채점 블록 지도(D-11)로 고쳤다. 설명문 단어 수는 그림 명세 16절 값으로 바꿨다.
4. R8 첫 문장의 'Fig. 6d' 인용을 Supplementary Table S12 로 바꿨다. 계층 conformal 0.86 은 09-26 규약 참조값이라 그림 명세가 Fig 6d 에서 뺐다(`fig6_b_ref.csv` 의 `use` 열).
5. XF 의 Table 1 자리표시를 지웠다. FINAL_BATCH 6절이 XF 를 Supplementary Table 로 고정했다(그림 명세 D-5).
6. 모든 X 자리표시를 11절의 위치별 문자열과 같게 만들었다. 결과 절 R3 의 XA 는 FINAL_BATCH 0.4 의 등록 문자열 `[XA: R3 | 사후 분석(재현(비맹검)) | 2.1 해석 조각]` 그대로이고, 채우는 방법은 괄호 밖에 적었다. XC(F3 제외)의 표지를 '결과 열람 뒤 설계, 재사용 지역의 재검정(비맹검 부분 포함)'으로, XB 의 세 자리 표지를 하나로, XF·XG 에 '결과 열람 뒤 설계'를 맞췄다. D2 에 XA 자리표시를 더했다.
7. 고찰 D1 의 워크플로를 claims 4절 표와 그림 명세 Fig 7d 정의에 맞췄다(라벨 320–1000 은 대상 라벨 교차검증 선정, 라벨이 수천 개인 지역 안은 교차검증 잔차 가중의 잔차 ML). Fig 7d 경로가 0 위에 놓이는 내용 위험과 결정 사항을 13.1 의 14번으로 두었다.
8. 1.3 의 '증강은 라벨 0 전용으로 판정'을 C3 README 1.3 의 6에 맞게 '라벨 0 에서만 확인'으로 좁혔다. D3 의 백분율을 정수로(39 %, 4 %, 19 %), M9 의 '1.2–2.3 %'를 '약 1–2 %'로 고쳤다.
9. 네 자리 수의 쉼표를 지웠다(2800, 4240, 4350, 4500, 8297, 7500, 1000). 음수가 든 범위의 en dash 를 '에서' 표기로 바꿨다(8곳).
10. R5 의 지도 문구를 Fig 7b 판에 따라 나눠 적었다. 13.1 에 Fig 7b 판(15번), Fig 5d 재선정(16번) 결정을 더했다.

**남은 것**: Fig 7d 경로의 기본값(내용 위험), 초록 판 B 의 WF 수치 7개(지침 4.2 의 2개 한도), 지역 영문 표기, 설명문 단어 수가 목표 150–250 을 넘는 점은 사용자 결정이나 원고 조립 단계의 일이다. 덱 스펙과 그림 명세의 같은 항목은 각 문서의 검토 반영 절에 있다.
