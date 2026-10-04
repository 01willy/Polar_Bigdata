# Scientific Reports 예시 논문의 원고 형식 분석과 문장 지침 (2026-10-04)

**성격**: 2022–2026년 Scientific Reports(이하 Sci Rep) 지구과학·원격탐사·환경 ML 논문 10편의 원고 형식을 분석한 조사 문서다. 대상 항목은 제목, 초록, 서론, 결과 소제목, 그림 설명문, 통계 보고, 고찰 구조와 한계의 위치, 방법, 자료·코드 가용성 문구, 참고문헌 수다. 끝에 모방할 문장 틀 45개, 피할 표현 목록(영문·국문), 현행 초안 점검 결과를 둔다.

**관련 문서**: 투고 규정 원문, 단어 예산, 초록 문장 배치, 표시 항목 계획은 같은 날짜의 `docs/research/2026-10-04/scirep_format_and_drafts.md` 가 다룬다. 이 문서는 그 규정을 반복하지 않고, 실제 게재 논문에서 확인한 형식과 문장을 다룬다.

**파일**: 새로 받은 PDF 8편은 `references/14_scirep_exemplars/`(목록은 같은 폴더 `README.md`), 기존 2편은 `references/00_core10/gautam2025_alaska_alt.pdf`, `references/05_uq_transfer/singh2024_conformal_eo.pdf` 다.

**측정 방법**: 단어 수와 문장 수는 `pdftotext` 출력에서 셌다. 본문 절의 단어 수에는 그림 설명문과 표 문자가 섞여 있어 근사값이다. 인용한 영어 문장은 PDF 원문에서 그대로 옮겼고 참고문헌 위첨자 번호만 뺐다.

---

## 1. 요지

1. 예시 10편의 결과 소제목은 모두 명사구다(소제목이 없는 2편 제외). 주장형 문장을 소제목으로 쓴 논문은 없다. 주장은 소제목 아래 첫 문장에 수치와 함께 둔다.
2. 잘 쓴 초록(Jones 2024, Hall 2026, Isaksen 2022)은 7–8문장이다. Jones·Hall 은 결과 문장 대부분에 단위가 붙은 수치가 있고, Isaksen 은 핵심 결과 하나를 수치로 말한 뒤 비교를 말한다. 마지막 문장은 일반론이 아니라 관측 결과의 요약, 불확실성의 크기, 기존 지식과의 차이다.
3. 서론은 맥락(수치 포함) → 측정·평가의 빈틈 → 이 연구의 질문(번호 목록)의 순서이고, 좋은 예는 700–1,000단어다. Isaksen 2022 는 서론 끝에 결과 요약 문단을 둔다.
4. 한계는 고찰 끝의 한 문단에 두는 형식이 가장 흔하다(Gautam, Hall, Fisher). 한계마다 결과 해석에 미치는 영향을 함께 적은 예(Hall 2026)가 가장 쓸모 있다.
5. 자료 가용성은 저장소 DOI 를 적은 예(Jones, Hall, Khanal)와 "요청 시 제공"(Feeney, Fisher 일부), "원고와 보충자료에 있음"(Gautam)으로 갈린다. 코드 가용성 절을 따로 둔 논문은 10편 중 2편(Singh, Isaksen)이고, Gay·Feeney 는 자료 가용성 문장 안에서 코드를 언급한다.
6. Sci Rep 게재가 문체의 질을 보장하지 않는다. Gay 2026 은 'framework' 를 1,000단어당 4.1회, 'comprehensive' 를 1.9회 쓴다. Jones 2024 는 두 단어 모두 0회다. Gautam·Khanal·Fisher 의 결론부에도 일반론 문장이 있다(7.2절).
7. 예시 논문의 그림 품질은 중간 수준이다(그림 안 제목, ggplot 기본 배경, 상자형 작업 흐름도). 이 10편은 원고 형식의 기준으로 쓰고, 그림 품질의 기준은 상위 학술지 예시(`references/09_figure_exemplars/`)에서 따로 가져와야 한다.
8. 현행 영문 초안은 과장어가 적다. 고칠 점은 P 값 없이 쓴 'significant' 와 기준이 본문에 정의되지 않은 'robust' 다(8절).

---

## 2. 선정 논문

선정 기준은 (1) 2022–2026년 Sci Rep 게재, (2) 영구동토·빙권·토양·수문·공간 검증 주제, (3) 관측 자료와 모델(물리 또는 통계·ML)을 함께 다룸, (4) 오픈액세스다. 후보는 OpenAlex 에서 Sci Rep(source S196734849)로 걸러 인용 수 순으로 정렬한 뒤 주제와 저자 기관을 보고 골랐다. Sci Rep 의 환경 ML 논문 다수는 방법 나열형이어서 제외했다.

| 키 | 서지 | 주제 | 고른 이유 | 파일 |
|---|---|---|---|---|
| Gautam 2025 | Gautam et al., Sci. Rep. 15, 42420 | 알래스카 ALT, RF 대 Stefan, CMIP6 | 지정 논문. 가장 가까운 선행 연구 | `00_core10/gautam2025_alaska_alt.pdf` |
| Hall 2026 | Hall, Chipman & Lara, Sci. Rep. (Article in Press) | 알래스카 RTS 확장, GLMM, 투영 | 영구동토 통계 모델, 누설 방지·외삽 처리·한계 문단 | `14_scirep_exemplars/hall2026_rts_expansion_alaska.pdf` |
| Gay 2026 | Gay et al., Sci. Rep. 16, 28715 | 북극권 zero-curtain, 물리 정보 전이학습 | 영구동토 + 물리 정보 DL. 문체 반면교사 | `14_scirep_exemplars/gay2026_zero_curtain_ai_eo.pdf` |
| Jones 2024 | Jones et al., Sci. Rep. 14, 8499 | 산불 뒤 영구동토 안정화, LiDAR·시추 | 수치 중심 서술의 모범 | `14_scirep_exemplars/jones2024_postfire_permafrost_stabilization.pdf` |
| Isaksen 2022 | Isaksen et al., Sci. Rep. 12, 9371 | 바렌츠해 기온 상승, 관측 대 재분석 | 서론 질문 목록, 질문별 고찰, 재분석 검증 | `14_scirep_exemplars/isaksen2022_barents_warming.pdf` |
| Hatami 2022 | Hatami & Nazemi, Sci. Rep. 12, 2196 | 기온·적설 복합 변화와 동결·융해 | 주장형 제목, 검정 명시 | `14_scirep_exemplars/hatami2022_temperature_snow_freeze_thaw.pdf` |
| Feeney 2022 | Feeney et al., Sci. Rep. 12, 1379 | 국가 규모 토양 탄소 지도 8종 비교 | 지도 비교·검증, 범위 한정 문장 | `14_scirep_exemplars/feeney2022_soil_map_comparison_soc.pdf` |
| Khanal 2023 | Khanal et al., Sci. Rep. 13, 8090 | 네팔 산림 토양 탄소, QRF | 공간 CV 대 무작위 CV, AOA, 전지구 제품 비교 | `14_scirep_exemplars/khanal2023_soc_stocks_nepal.pdf` |
| Fisher 2023 | Fisher et al., Sci. Rep. 13, 8174 | 개방 수면 증발, 과정 모델 대 ML 11종 | 물리 모델 대 ML 벤치마크 논리 | `14_scirep_exemplars/fisher2023_open_water_evaporation.pdf` |
| Singh 2024 | Singh et al., Sci. Rep. 14, 16166 | 지구관측 ML 의 conformal UQ | 불확실성 용어 정의, 코드 가용성 | `05_uq_transfer/singh2024_conformal_eo.pdf` |

Hall 2026 파일은 출판사가 공개한 Article in Press 판(편집 전)이다. 최종 편집본이 나오면 교체한다. Gay 2026 과 Hall 2026 은 CC BY-NC-ND 이고 나머지는 CC BY 다.

---

## 3. 이 분석과 관련된 공식 지침

출처: https://www.nature.com/srep/author-instructions/submission-guidelines (2026-10-04 열람). 단어·분량 한도의 원문은 `scirep_format_and_drafts.md` 2.2절에 있다. 여기서는 예시 논문과 대조할 항목만 옮긴다.

| 항목 | 원문 |
|---|---|
| 본문 구성 | "Introduction / Results (with subheadings) / Discussion (without subheadings) / Methods" (권장 구성, 필수 아님) |
| 제목 | "should describe the main message of the article using a single scientifically accurate sentence, and should not contain puns or idioms" |
| 초록 | "Make sure it serves both as a general introduction to the topic and as a brief, non-technical summary of the main results and their implications. Abstract should be unstructured" |
| 그림 설명 | "Please begin your figure legends with a brief title sentence for the whole figure and continue with a short description of what is shown in each panel. Use any symbols in sequence and minimise the methodological details as much as possible." |
| 그림 오차 | "Include error bars when appropriate. Include a description of the statistical treatment of error analysis in the figure legend." |
| 그림 장식 | "Put all display items on a white background, and avoid excessive boxing, unnecessary colour, spurious decorative effects (such as three-dimensional 'skyscraper' histograms) and highly pixelated computer drawings." |
| 그림 문자 | "Use a clear, sans-serif typeface (for example, Helvetica) for figure lettering. Use the same typeface in the same font size for all figures in your paper." / "Lettering in figures should be in lower-case type, with only the first letter of each label capitalised." |
| 'significant' | "Use of the word "significant" should always be accompanied by a P value; otherwise, use "substantial," "considerable," etc." |
| ± 표기 | "You must state whether a number that follows the ± sign is a standard error (s.e.m.) or a standard deviation (s.d.)." |
| 작은 자료 | "Ranges are more appropriate than standard deviations or standard errors for small data sets." |
| LLM | "Use of an LLM should be properly documented in the Methods section" |

지침의 "avoid excessive boxing" 은 사용자가 지적해 온 상자형 문자 그림 금지와 같은 방향이다. 지역 수가 4–7개인 우리 추론은 "small data sets" 항목에 해당하므로 지역별 값과 범위를 함께 보고해야 한다.

---

## 4. 형식 분석

### 4.1 제목

| 논문 | 단어 수 | 형태 | 제목 |
|---|---|---|---|
| Isaksen 2022 | 6 | 명사구 | Exceptional warming over the Barents area |
| Fisher 2023 | 6 | 명사구 | Remotely sensed terrestrial open water evaporation |
| Hall 2026 | 8 | 동명사구 | Predicting retrogressive thaw slump expansion across northern Alaska |
| Khanal 2023 | 8 | 동명사구 | Mapping soil organic carbon stocks in Nepal's forests |
| Gay 2026 | 8 | 동명사구 | Resolving circumarctic zero-curtain phenomena with AI-integrated earth observations |
| Jones 2024 | 9 | 명사구 | Post-fire stabilization of thaw-affected permafrost terrain in northern Alaska |
| Singh 2024 | 12 | 명사구 | Uncertainty quantification for probabilistic machine learning in earth observation using conformal prediction |
| Gautam 2025 | 14 | 명사구 | Machine learning and process-based modeling of spatiotemporal changes in active layer thickness across Alaska |
| Feeney 2022 | 15 | 주장 문장 | Multiple soil map comparison highlights challenges for predicting topsoil organic carbon concentration at national scale |
| Hatami 2022 | 16 | 주장 문장 | Compound changes in temperature and snow depth lead to asymmetric and nonlinear responses in landscape freeze–thaw |

관찰:
- 6–16단어이고 중앙값은 8.5단어다. 지침의 20단어 한도를 넘은 예는 없다.
- 지침은 "single scientifically accurate sentence" 를 권하지만 실제로는 10편 중 2편만 주장 문장이다. 명사구 제목도 게재된다.
- 의문문 제목, 콜론으로 나눈 제목, 방법 이름(약어)을 앞세운 제목은 없다.
- 주장 문장 제목(Hatami)은 결과의 방향(asymmetric, nonlinear)을 제목에서 말한다. 그만큼 본문 결과가 모든 지역에서 같은 방향이어야 한다.

### 4.2 초록

| 논문 | 단어 | 문장 | 수치 든 문장 | 문장 순서 | 마지막 문장 |
|---|---|---|---|---|---|
| Hall 2026 | 197 | 7 | 4 | 정의 → 빈틈 → 자료·n·기간 → 분류 성능(민감도·특이도) → 요인(R², RMSE) → 1차 추정 → 투영(범위, "highly uncertain") | 불확실성 범위 |
| Isaksen 2022 | 172 | 7 | 2 | 맥락 → 빈틈(관측 부족) → 자료 → 핵심 수치(°C/decade) → 비교 대상 → 해석 → 기존 대비 | 기존 지식 대비 차이 |
| Jones 2024 | 250 | 8 | 6 | 사건·면적 → 초기 변화 → 1차 기간 침하 15% → 2차 기간 <1% → 지온 TDD 배율 → MAGT 변화율 → 시추 n·두께 → "Taken together" | 관측의 종합 |
| Fisher 2023 | 312 | 8 | 5 | 중요성 → 평가 공백 → "Here, we evaluated"(n = 19) → r²·bias·RMSE(평균 대비 %) → 바람 조건 민감도 → 일 단위 오차 → ML 11종 벤치마크 → 함의 | 신뢰와 불확실성 병기 |
| Gay 2026 | 273 | 8 | 6 | 맥락 → 정의 → 빈틈 → 방법(관측 수) → 정확도 → 절제 실험 → 요인 → 운영 함의 | "establishes a NISAR-ready ... protocol" |
| Gautam 2025 | 345 | 12 | 6 | 맥락 → 방법 → 시나리오 → R²(훈련·시험) → RMSE → 변수 중요도 → 기준 ALT → 위도 경향 → 투영 → 공간 → 일반론 2문장 | "highlighting the urgent need ..." |
| Singh 2024 | 225 | 10 | 1 | 맥락 → 필요 → 방법 성질 → 검토 결과(22.5%) → 도구 → 시연 → 전망 | 기대 문장 |
| Khanal 2023 | 201 | 8 | 1 | 필요 → 빈틈 → 자료 → 방법 → 결과(정성) → 기준선 → 총량(SE) | 함의 |
| Feeney 2022 | 200 | 9 | 0 | 중요성 → 동인 → DSM → 불일치 → "Here we compare" → 결과(정성) → 함의 → 활용 → 향후 | 향후 과제 |
| Hatami 2022 | 184 | 8 | 1 | 중요성 → 필요 → 방법 → 결과(정성) → 비대칭 → 비선형 → 함의 | 함의 |

관찰:
- 200단어 이하는 4편(Hall, Isaksen, Hatami, Feeney 는 정확히 200)이고 Khanal 은 201단어다. Gautam 2025(345단어)처럼 한도를 크게 넘어도 게재되지만, 한도 안에서 수치 문장이 많은 초록이 읽기 쉽다.
- 좋은 초록의 공통점: (1) 첫 두 문장 안에 빈틈을 말한다, (2) "Here, we ..." 문장에 자료 수(n), 기간, 설계를 함께 넣는다, (3) 결과 문장 대부분에 단위가 있는 수치가 있다(Jones 8문장 중 6, Hall 7문장 중 4), (4) 불확실성이나 조건을 한 문장으로 말한다(Hall: "remain highly uncertain", Fisher: "though not without uncertainty").
- 약한 초록의 공통점: 수치 없는 결과 문장(Feeney, Hatami), 끝의 일반론("highlight the complex, multifactorial nature", "urgent need"), 형용사 결과("vivid spatial divide").
- 우리 초록에 바로 쓸 점: 결과 문장은 cm 단위 차이와 95 % 구간을 넣고, 라벨 수(n)를 문장마다 밝힌다. 마지막 문장은 워크플로 규칙 또는 적용 조건을 한 문장으로 쓴다.

### 4.3 서론

| 논문 | 길이(근사) | 문단 역할 |
|---|---|---|
| Jones 2024 | 약 600단어, 4문단 | ① 툰드라 산불 증가와 선행 수치(면적 3 %가 열카르스트 11 %) ② 영구동토 해빙과 탄소 ③ 대상 사건과 자체 선행 연구, 빈틈("remained uncertain") ④ "In this study" + 번호 질문 2개 + 대상지 |
| Fisher 2023 | 약 700단어, 4문단 | ① 중요성(수지 균형) ② 측정이 어려운 두 이유(번호) ③ 원격탐사의 가능성과 한계 ④ 동기(운영 산출물), 목표, n = 19, ML 11종 벤치마크 |
| Hall 2026 | 약 850단어, 3문단 + 선행 수치 표 | ① RTS 정의와 지역별 요인 차이(선행 연구 비교 Table 1) ② 대상 지역의 기후 경사 ③ 번호 목표 3개 |
| Singh 2024 | 약 850단어, 소제목 2개 | ① 국제 협약과 자료 ② GeoAI 와 사용자 신뢰 ③ 불확실성 정의(aleatoric·epistemic, 불확실성 대 오차) ④ 목표 4개(Next, Third, Lastly) |
| Feeney 2022 | 약 950단어, 6문단 | ① SOC 중요성 ② 관리 ③ 조사·원격탐사 ④ DSM 설명 ⑤ 대상국과 빈틈("no attempt to date") ⑥ "In this study" 절차 |
| Gautam 2025 | 약 1,000단어, 5문단 | ① 영구동토와 ALT 정의 ② CALM ③·④ 모델 유형 나열 ⑤ 선행 연구 1편 + 목표 2개 |
| Hatami 2022 | 약 1,000단어, 5문단 | ① 동결·융해 중요성 ② 대상 지역 ③ 지식 공백의 원천(Firstly, Secondly) ④ "We argue" 방법 논거 ⑤ copula 방법 |
| Isaksen 2022 | 약 1,100단어, 6문단 | ① 해빙·기온 수치 ② 북극 온난화율 수치와 핫스팟 ③ 관측 부족, 재분석 의존 ④ 서부 관측 편중 ⑤ 목표 3개 + 질문 3개(목록) ⑥ "We find" 결과 요약 |
| Khanal 2023 | 약 1,400단어, 9문단 | 교과서식 배경(scorpan 식 2개 포함) → 목표. 길고 빈틈이 늦게 나온다 |
| Gay 2026 | 약 1,700단어, 7문단 | 배경·정의·한계 나열 → 기여 3개. 전문어 밀도가 높다 |

관찰:
- 좋은 서론은 3–5문단, 600–1,100단어다. 문단마다 역할이 하나다.
- 빈틈 문장은 "무엇이 측정·평가되지 않았는가"를 구체적으로 말한다(Jones: 장기 관측이 없어 궤적이 불확실, Isaksen: 북동부 관측이 없어 재분석 의존, Feeney: 지도 간 비교가 없음).
- 질문을 번호로 적는 형식(Jones 2개, Isaksen 3개, Hall 3개)이 고찰 구성과 연결된다. Isaksen 은 고찰 소제목을 서론의 질문 문장으로 그대로 쓴다.
- 모델 유형을 나열만 하는 문단(Gautam ③·④)은 빈틈을 흐린다. 선행 연구는 "평가 조건"의 차이로 묶어야 한다.

### 4.4 결과 소제목

| 논문 | 결과 소제목 |
|---|---|
| Jones 2024 | LiDAR change detection in terrain units / Post-fire ground thermal regimes / Changes to near-surface permafrost |
| Hall 2026 | ArcticDEM and kriging reconstruction / RTS variability and change / Drivers of RTS activity / Future projections of RTS dynamics / RTS nutrient concentrations |
| Isaksen 2022 | Recent surface air temperature (SAT) development (하위: SAT development from reanalyses, SAT development from instrumental observations, Comparing observed SAT trends with reanalyses) / Evaluation of ERA5 and CARRA reanalyses / Trends in SIC and SST / Observed warming related to sea ice and SST |
| Khanal 2023 | Spatial distribution of forest SOC stocks / Evaluation of uncertainty / Area of applicability of spatial prediction model / Variable importance / Comparison of model predictions with existing global SOC data products |
| Feeney 2022 | Inter-comparison of eight topsoil organic carbon concentration maps / Evaluation of modelled topsoil organic carbon concentrations against survey data |
| Gautam 2025 | (Results and discussion 결합) Observed environmental controls of active layer thickness / Estimation of ALT under current climate / Projection of ALT under future climate |
| Gay 2026 | Principal Insights / Physics-informed zero-curtain candidate detection / Remote sensing / GeoCryoAI model / Model validation and performance benchmarking / Uncertainty quantification / Temporal trend analysis and methodological convergence |
| Singh 2024 | Literature review 와 사례 연구 소제목(Global google dynamic world (classification) 등) |
| Fisher 2023, Hatami 2022 | 소제목 없음 |

관찰:
- 주장형 소제목(예: "Physics features reduce transfer error")은 한 편도 없다. 모두 대상·분석 이름의 명사구다.
- 주장은 소제목 아래 첫 문장에 둔다. 예: Jones "Between 2009 and 2014, permafrost thaw subsidence was detected across 15% of the 50 km2 terrestrial landscape area at a rate of 136.6 ha/yr (Table 1)."
- 결론: 영문 원고의 결과 소제목은 명사구로 쓰고(예시 논문 관행, 사용자 문체 규칙과 일치), 첫 문장을 주장 문장으로 쓴다.

### 4.5 그림 설명문

| 논문 | 길이(근사) | 첫 문장 형태 | 패널 서술 | 오차·통계 정의 | 소프트웨어 표기 |
|---|---|---|---|---|---|
| Fisher 2023 | 15–60단어 | 주장 문장("All raw unfiltered data showed high in situ instantaneous open water evaporation associated with high wind events.") | 짧음 | 그림 안 R², RMSE | 있음 |
| Jones 2024 | 60–110단어 | 명사구("Post-fire stabilization of the thaw-affected permafrost landscape.") | (a)(b) 순서 | 핵심 수치를 설명문에 다시 씀(15 %, <1 %) | 지도에 있음 |
| Isaksen 2022 | 60–150단어 | 명사구 | (a)–(f) 순서, 기간·단위 | 필터("Gaussian filter with a standard deviation of three years"), 표 설명에 Mann–Kendall 유의수준 | 있음 |
| Hall 2026 | 30–80단어 | 명사구 | (a)–(h) | "Error bars represent 95% Monte Carlo uncertainty ranges" | 지도에 있음 |
| Feeney 2022 | 50–110단어 | 명사구 | (a)–(d), 통계 정의(CV, SNR) | "± 1 standard error of the mean" | 없음 |
| Khanal 2023 | 20–50단어 | 명사구 | "Panel (A) shows ..." | n(1156), RMSE 단위 | 있음 |
| Gautam 2025 | 20–45단어 | "Comparison of ..." | (a)(b) | 대부분 없음(Fig. 5 만 중앙값·IQR) | 있음 |
| Gay 2026 | 150–200단어 이상 | "This figure illustrates ..." | 길고 방법 세부 포함 | 일부 | 일부 |

관찰:
- 지침대로 "제목 문장 + 패널 설명 + 오차 정의"를 갖춘 예는 Hall, Isaksen, Feeney 다.
- 제목 문장을 주장으로 쓴 예(Fisher)는 그림 하나에 메시지 하나를 강제하는 효과가 있다. 사용자 시각 규칙("one dominant message per figure")과 맞는다.
- 피할 형태: "This figure illustrates ..."(Gay), 오차 막대·음영의 정의가 없는 설명문(Gautam Fig. 2–4), 방법 세부를 설명문에 반복하는 형태(Gay).

### 4.6 통계 보고

| 논문 | 보고 방식 |
|---|---|
| Hall 2026 | 귀무 모델 대비 ΔAIC, 주변·조건부 R², 민감도·특이도와 분자·분모(14/23), 계수·z·p, 95 % Monte Carlo 범위, 집단별 n |
| Isaksen 2022 | 추세 °C/decade, Mann–Kendall 1 %·5 % 수준(표에서 굵게·기울임), 편향·SDE, R² |
| Khanal 2023 | 공간 10겹 CV 대 무작위 CV 의 RMSE·R², 잔차 변동도와 99회 순열 포락선, AOA, 총량과 SE |
| Fisher 2023 | r², RMSE·bias 를 절대값과 평균 대비 %로 함께, 사이트·장면 수(686), 회귀 계수의 p |
| Hatami 2022 | 상대 오차 %, Kendall τ, Bonferroni 보정 일원 ANOVA, 1,000회 재표본 |
| Singh 2024 | 경험적 포함률 95.15 % ± 0.07, 평균 구간 폭 9.28 m ± 0.03, 자료 수 n = 243 |
| Jones 2024 | 기술 통계 중심, 시추 집단별 n(n = 41, n = 8), 변화 배율(~30) |
| Feeney 2022 | 격자 n = 208,752, 점 n = 2,614, Pearson r 범위, Taylor 도표, ± 1 SE |
| Gautam 2025 | 훈련·시험 R²·RMSE, 평균 ± SD. 신뢰구간과 검정 없음. "median values of 13 ± 2.6 cm" 처럼 중앙값과 ± 를 섞은 표기 |
| Gay 2026 | 정확도 %, r, 왜도·첨도, 절제 실험, 민감도 분석 |

관찰:
- 모범 형식은 효과 크기 + 구간 + n + 검정 이름 + 정확한 P 다(Hall). 구간을 Monte Carlo, 부트스트랩 중 무엇으로 냈는지를 설명문에 적는다.
- 상대 오차를 평균 대비 %로 병기하는 Fisher 형식은 지역별 ALT 수준이 다른 우리 자료에도 쓸 만하다.
- 무작위 CV 와 공간 CV 를 나란히 보고하는 Khanal 형식은 우리 C7(무작위 분할 과대평가)을 보조 결과로 쓸 때의 틀이다.
- Gautam 2025 는 무작위 70/30 분할을 쓰고 "Due to the limited size of the dataset, we performed parameter tuning using the full dataset, which may introduce a degree of information leakage." 라고 적는다. 우리 원고가 이 연구와 비교할 때 평가 설계 차이를 사실대로 적을 근거다.

### 4.7 고찰 구조와 한계의 위치

| 논문 | 고찰 구성 | 한계의 위치 |
|---|---|---|
| Jones 2024 | 소제목 없음. 3단계 해석 → 기전 → 선행 연구 비교 → 개념도(Fig. 6) → 탄소 함의 | 명시 문단 없음(관측 범위로 한정) |
| Feeney 2022 | 소제목 없음. 지도별 차이의 원인 → 방법의 확장 가능성 + 별도 Conclusion | 범위 한정 문장은 Methods 에("not a validation study") |
| Fisher 2023 | 선행 연구 수치 비교 → 오차 원천 5개 열거 → 한계 → 전망 + Conclusion | "Taken together, these challenges and limitations ..." 문단 |
| Hall 2026 | 소제목 2개(요인, 극한 사건) + Conclusion | 고찰 끝 문단: 적용 범위, 훈련·시험 R² 차이, 측정 안 된 변수, 관측 빈도 |
| Gautam 2025 | 결과와 결합 + Conclusions | 결합 절 끝: "Our study has certain limitations and underlying assumptions. First, ..." |
| Isaksen 2022 | 서론 질문 3개를 소제목으로 + Summary and conclusions | 해당 질문 안에서 처리 |
| Khanal 2023 | 목표 재진술로 시작 → 요인별 해석 → CV·AOA → 전지구 제품 비교 + Conclusions | AOA·표본 배치 문단 안에 분산 |
| Gay 2026 | 'Limitations' 소제목, 번호 6개 | 별도 소절 |
| Singh 2024 | 주제별 문단 + 'Future research directions' | "A limitation impeding ..." 등 분산 |
| Hatami 2022 | 추가 분석(비선형성)을 고찰에서 수행 + Summary | 명시 없음 |

관찰:
- 지침은 고찰에 소제목을 두지 않도록 권한다. 지킨 예는 Jones, Feeney, Fisher 다.
- 쓸모 있는 한계 문단의 요소(Hall): (1) 모델이 하지 못하는 것("they do not predict the timing or precise location of new (uninitiated) thaw slumps"), (2) 검증하지 않은 조건("remain untested in dissimilar permafrost environments"), (3) 훈련·시험 성능 차이와 그 원인, (4) 측정되지 않은 변수.
- 피할 형태: 한계를 적은 뒤 "the net gain of insight from this analysis far outweighs these limitations, making these results a significant contribution"(Fisher)처럼 스스로 가치를 판정하는 문장. 한계의 끝을 자기 평가로 닫지 않는다.
- Hatami 처럼 고찰에서 새 분석을 하는 구성은 피한다. 새 분석은 결과에 둔다.

### 4.8 방법의 위치·분량·하위 절

| 논문 | 위치 | 분량(근사) | 하위 절 |
|---|---|---|---|
| Jones 2024 | 고찰 뒤 | 약 950단어 | Repeat LiDAR analysis / Terrain unit mapping / Ground temperature measurements / Permafrost borehole coring campaigns |
| Gautam 2025 | 서론 뒤 | 약 1,250단어 | Study area, ALT observations and environmental datasets / Earth system model and emission scenarios / Stefan model / Machine learning model |
| Hatami 2022 | 고찰 뒤 | 약 1,300단어 | Data support / Proposed copula-based impact assessment framework |
| Isaksen 2022 | 고찰 뒤 | 약 1,700단어 | 기간 선택 근거 / Instrumental SAT data / Reanalyses / SIC and SST |
| Fisher 2023 | 서론 뒤 | 약 2,000단어 | Data: in situ / Data: satellite / Model: AquaSEBS / Model: machine learning / Software packages |
| Khanal 2023 | 서론 뒤 | 약 2,250단어 | Study area / Field data / Selection of predictors / Prediction and uncertainty / Representativeness(AOA) / Comparison with global estimates |
| Feeney 2022 | 고찰 뒤 | 약 2,600단어 | Inter-comparison / Evaluation against survey data |
| Hall 2026 | 서론 뒤 | 약 3,350단어 | Study region / Data acquisition / Modeling RTS change / Projecting RTS change / Nutrient mobilization |
| Gay 2026 | 고찰 뒤 | 약 7,200단어 | 10개 이상 |
| Singh 2024 | 없음 | 해당 없음 | 사례 연구 안에 기술 |

관찰:
- 방법 위치는 5편이 고찰 뒤(지침 권장), 4편이 서론 뒤다. 둘 다 게재된다. 지침 권장인 고찰 뒤에 둔다.
- 하위 절은 자료 → 모델 → 평가 → 투영·응용의 순서이고 이름은 명사구다.
- 좋은 방법 문장은 결정의 이유를 함께 적는다(Hall: 누설 방지, 외삽 시 변수 고정. Feeney: 자료가 적어 층화를 줄임).

### 4.9 자료·코드 가용성 문구

| 논문 | 자료 | 코드 |
|---|---|---|
| Jones 2024 | "The data that support the findings of this study are published at the Arctic Data Center." (자료 인용 번호) | 별도 없음 |
| Hall 2026 | 저장소와 DOI(NSF Arctic Data Center) | 별도 없음 |
| Khanal 2023 | figshare DOI + "The spatial layers are in GeoTIFF format with a spatial resolution of 30 m and ESPG:32644 spatial reference system." | 별도 없음 |
| Isaksen 2022 | 자료원별 목록(노르웨이·러시아 기관 저장소) | "Information about the codes can be obtained from the corresponding author upon request." |
| Singh 2024 | 공개 자산(GEE) | GitHub 저장소와 GEE 코드 링크 |
| Gay 2026 | 자료원 나열, 산출물은 "archived on Drive via GitHub" | GitHub(재현 문서 이름까지 기술), DOI 없음 |
| Hatami 2022 | 지역 자료 포털 | 별도 없음 |
| Fisher 2023 | 위성·재분석은 공개 경로, 현장 자료는 "available from the corresponding author on reasonable request" | 별도 없음 |
| Feeney 2022 | "Some of our data, including code is available on request" | 같은 문장 |
| Gautam 2025 | "Data is provided within the manuscript & supplementary information files." | 없음 |

관찰:
- 최근 영구동토 논문(Jones, Hall)은 Arctic Data Center DOI 를 적는다. 도출 자료를 DOI 저장소에 두는 것이 이 분야 관행으로 굳어지고 있다.
- 코드는 GitHub 링크만 적은 예가 많다. 링크는 바뀔 수 있으므로 버전 고정 DOI(예: Zenodo)를 함께 적는 편이 낫다. 이것은 예시 논문에서 온 관찰이 아니라 권고다.
- 형식 요소로는 Khanal 의 파일 형식·해상도·좌표계 명시가 가장 구체적이다.

### 4.10 참고문헌 수

PDF 번호 기준: Jones 58, Gautam 62, Singh 62, Isaksen 64, Feeney 67, Hatami 75, Hall 109, Khanal 116, Fisher 142, Gay 187. 중앙값은 71이다. 지침의 60개 한도는 엄격하지 않다. 영구동토 ML 원고는 55–75개가 예시 범위의 중간이다.

### 4.11 그림 자체에 대한 관찰

PDF 8쪽(그림 10개: Jones Fig. 3, Hall Fig. 2·4·5, Isaksen Fig. 6, Gay Fig. 1, Khanal Fig. 3·4, Feeney Fig. 4, Gautam Fig. 5)을 렌더링해 보았다.
- Gautam Fig. 5: 그림 안 제목, ggplot 회색 배경, 소수 둘째 자리 기준선 표기("59.63"), 범례 중복.
- Hall Fig. 2: 상자와 화살표로 된 자료 처리 흐름도에 작은 지도를 끼운 형태. 사용자가 피하라고 한 상자형 그림의 예다.
- Gay Fig. 1: 그림 안 제목과 문자 상자, 개념적 이상화 자료.
- Khanal Fig. 3: 육각 밀도 산점도 + 1:1 선 + 패널 안 RMSE·R². 단순하고 읽기 쉽다.
- Isaksen Fig. 6: 산점도·시계열 6패널, 패널 안 R². 정보량은 많으나 범례가 길다.
- Feeney Fig. 4: 작은 다중 상자그림 + 밝은 범주 색. 밀도가 높다.

이 표본의 그림은 Sci Rep 의 하한에 가깝다. 우리 그림은 지침의 그림 조항(흰 배경, 상자·장식 금지, 같은 sans-serif 크기, 소문자 굵은 패널 기호)과 사용자 시각 규칙을 기준으로 하고, 품질 비교 대상은 상위 학술지 예시에서 고른다.

---

## 5. 본 원고 적용 규칙

단어 예산·표시 항목·초록 문장 배치는 `scirep_format_and_drafts.md` 4.2–4.6절을 따른다. 아래는 예시 논문 분석에서 나온 추가 규칙이다.

### 5.1 제목
- 의문문 제목(`MANUSCRIPT_RESTRUCTURE_PLAN_2026-10-02.md` 후보 1, "How many labels ...")은 쓰지 않는다. 지침의 "single scientifically accurate sentence" 와 맞지 않고, 예시 10편에도 없다. 'how many' 는 RESEARCH_FRAME A.1 의 금지어이기도 하다(`scirep_format_and_drafts.md` 4.2절).
- 명사구(후보 2·3 형태) 또는 결과가 모든 지역에서 같은 방향일 때만 주장 문장을 쓴다.

### 5.2 초록
- 7–8문장, 200단어 이하. 결과 문장 3–4개에 cm 단위 차이와 95 % 구간, 라벨 수 n 을 넣는다.
- 설계 문장은 Fisher·Hall 형식으로 쓴다: "Here we [compare ...] across [k] regions and label budgets of [0–N] sites, holding out each region with a 100 km buffer."
- 마지막 문장은 적용 조건이나 불확실성의 크기를 쓴다. "highlight the importance", "pave the way" 류로 닫지 않는다.

### 5.3 서론(4문단, 700–900단어)
1. 문제와 수치: 새 지역의 ALT 예측, 관측 수의 제약, 지역마다 다른 Stefan 계수 E.
2. 평가 조건의 빈틈: 선행 연구를 모델 종류가 아니라 평가 조건(무작위 분할, 단일 라벨 조건, 물리 기준선 유무)으로 묶는다. Gautam 2025 의 시험 R²(RF 0.24, Stefan 0.54)와 무작위 70/30 분할을 사실대로 적는다.
3. 질문: 번호 2–3개(Jones·Isaksen 형식).
4. 설계와 결과 요약: Isaksen 처럼 "We find ..." 로 핵심 수치 1–2개를 먼저 말한다.

### 5.4 결과
- 소제목은 명사구로 한다. `MANUSCRIPT_RESTRUCTURE_PLAN_2026-10-02.md` 4절 R1–R8 에 대응하는 예: "Evaluation design and error floor", "Transfer without target labels", "Recalibration and residual learning with 1–10 labels", "Method choice with tens to hundreds of labels", "Gains within label-rich regions", "Placement of new observations", "Source of the gain relative to the Stefan model", "Prediction intervals". 최종 문구는 `scirep_format_and_drafts.md` 4.4절 표와 맞춘다.
- 각 절의 첫 문장은 주장과 수치다. 틀: "[Model] [reduced/increased] [metric] relative to [baseline] by [x] cm (95% CI [a, b]; n = [labels]; [k] regions) (Fig. [n])."
- 절 사이 전환은 질문으로 한다(Fisher: "We next asked ...").
- 유의하지 않은 결과도 같은 형식으로 쓴다(Isaksen: "we did not find evidence indicating that ...").

### 5.5 그림 설명문
틀(200–300단어 이내):
1. 제목 문장 1개: 그림의 메시지(주장 문장) 또는 명사구.
2. 패널 (a)–(d) 순서의 설명, 패널마다 1–2문장.
3. 표기 정의: 점·선·음영이 무엇인지, 오차가 무엇인지("Shading shows the 95% block-bootstrap interval across [units]").
4. n 과 자료 기간.
5. 지도에는 소프트웨어와 버전.

금지: "This figure illustrates", 방법 반복, 그림 안 제목, 오차 정의 없는 음영.

### 5.6 통계 문장
- 형식: 차이(cm) + 95 % 구간 + 단위 수(n, 지역 수) + 검정 이름 + 보정 방법 + 정확한 P.
- 'significant' 는 P 값과 함께만 쓴다. 지침은 P 값이 없을 때 'substantial', 'considerable' 을 허용하지만, 이 원고는 그 경우에도 수치로 말한다.
- ± 뒤의 값이 SD 인지 SE 인지 적는다.
- 지역이 4–7개인 비교는 지역별 값과 범위를 표나 그림에 함께 둔다(지침의 small data sets 조항).
- 상대 크기가 필요하면 Fisher 형식으로 평균 대비 %를 병기한다.

### 5.7 고찰(소제목 없음, 5–6문단)
1. 질문에 대한 답을 수치 1–2개로 다시 말한다(Jones 형식, 목표 재진술로 시작하지 않는다).
2. 기전 해석: 이득이 어디서 오는가(계수 재보정, 격자 단위 편향 보정).
3. 선행 연구 대비: 일치("is consistent with"), 불일치와 원인("This contrasts with ... We attribute this divergence to ...").
4. 대안 설명 배제: "This [pattern] is unlikely to reflect [artefact], as [control]."
5. 한계 1문단: 하지 못하는 것, 검증하지 않은 조건, 측정하지 않은 변수, 훈련·시험 차이. 문장마다 해석에 대한 영향을 적는다.
6. 워크플로와 적용 조건. 자기 평가 문장으로 끝내지 않는다.

### 5.8 방법(고찰 뒤, 하위 절 명사구)
Study regions and ALT observations / Covariates / Stefan baseline and coefficient recalibration / Machine-learning models and physics features / Evaluation design (region hold-out with buffer, label-budget grid, in-region blocks) / Statistical analysis / Preregistration and deviations / Use of large language models. 결정마다 이유를 한 구절로 붙인다(Hall·Feeney 형식).

### 5.9 자료·코드 가용성과 LLM 사용 기재
- 자료: 도출 표와 예측 산출물은 DOI 저장소에 두고 형식·해상도·좌표계를 적는다(Khanal 형식). 제3자 자료(CALM, GTN-P, KPDC, ERA5-Land 등)는 기관명과 접근 조건을 적는다. 각 URL·DOI 는 원고 작성 시 확인한다[미확인].
- 코드: 저장소 링크와 버전 고정 DOI.
- LLM: 지침상 Methods 에 기재해야 한다. 문구는 사용자 결정 사항이다. 초안 틀: "Claude (Anthropic) was used to [write analysis code / edit language]. The authors checked all code, results and text and take responsibility for the content."

---

## 6. 모방할 문장 틀 (45개)

원문은 예시 논문의 문장을 그대로 옮긴 것이다(참고문헌 번호만 제거). 틀은 우리 원고에 맞춘 일반형이다. 원문에 과장어가 있으면 틀에서 뺐다.

### 6.1 초록

| # | 틀 | 원문 예 | 출처 |
|---|---|---|---|
| 1 | Here we [compare/evaluate] [object] against [n] [sites/regions] using [design]. | "Here, we evaluated the open water evaporation algorithm, AquaSEBS, used by ECOSTRESS and OpenET against 19 in situ open water evaporation sites from around the world using MODIS and Landsat data" | Fisher 2023 |
| 2 | [Metric] [increased] by [x unit] [period], but [then decreased] by [y unit] [period]. | "Mean annual ground temperature of the near-surface permafrost increased by 0.33 °C/yr in the burn site up to 7-years post-fire, but then cooled by 0.15 °C/yr in the subsequent eight years" | Jones 2024 |
| 3 | [Data-driven models] did not improve on [process-based model], suggesting that the remaining error comes from [sources]. | "To benchmark AquaSEBS, we ran a suite of 11 machine learning models, but found that they did not significantly improve on the process-based formulation of AquaSEBS suggesting that the remaining error is from a combination of the in situ evaporation measurements, forcing data, and/or scaling mismatch" | Fisher 2023 |
| 4 | [Model] distinguished [A] from [B] ([x]% sensitivity, [y]% specificity). | "Generalized linear mixed models distinguished historically active from inactive RTSs (60% sensitivity, 66% specificity)" | Hall 2026 |
| 5 | ..., although [estimate] remains uncertain, ranging from [a] to [b]. | "although projected expansion rates remain highly uncertain, ranging from approximately 1- to 34-fold greater than present-day rates by century's end." | Hall 2026 |
| 6 | Taken together, our [observations/results] show that [summary]. | "Taken together, our observations highlight that the initial degradation of ice-rich permafrost following the Anaktuvuk River tundra fire has been followed by a period of thaw cessation, permafrost aggradation, and terrain stabilization." | Jones 2024 (highlight 는 show 로 바꾼다) |

### 6.2 서론

| # | 틀 | 원문 예 | 출처 |
|---|---|---|---|
| 7 | However, because [observations] were lacking, [quantity] remained uncertain. | "However, due to a lack of previous long-term observations on the long-term effects of fire-induced permafrost thaw, the trajectory of post-fire permafrost degradation in Arctic tundra remained uncertain." | Jones 2024 |
| 8 | No study has [compared X under condition Y]. This matters because [consequence]. | "no attempt to date has been made to compare published SOC concentration maps with each other. This is an important limitation as predictions from each of these maps may differ substantially from one to another" | Feeney 2022 ('to date' 주장은 문헌 확인 뒤에만) |
| 9 | We address two questions: (1) [...]; and (2) [...]? | "Here we were interested in addressing two primary questions: (1) what is the trajectory and long-term effect of fire-induced permafrost degradation in the Arctic, and (2) can permafrost in burned tundra landscapes stabilize and recover under the current climate?" | Jones 2024 |
| 10 | Our objectives are threefold: i) [...], ii) [...], and iii) [...]. Specifically, we address the following questions: | "Our main objectives are threefold: i) establish an extended and consistent high-quality SAT-dataset covering the NBS region, ii) study the recent warming and its spatial and temporal variability over the NBS, and iii) relate the trend in SAT pattern to variations in SIC and SST. Specifically, we address and discuss the following questions:" | Isaksen 2022 |
| 11 | We find [result with number]. [Second result]. | "We find an unprecedented increase in SAT over the NBS of up to 2.7 °C per decade annually at Karl XII-øya in northeastern Svalbard, with a maximum in autumn of up to 4.0 °C per decade." | Isaksen 2022 ('unprecedented' 는 수치 비교가 있을 때만) |
| 12 | [Task] is difficult for two reasons: (1) [...]; and (2) [...]. | "Measuring open water evaporation is challenging for two primary reasons: (1) instrumentation; and, (2) representation." | Fisher 2023 |
| 13 | [Term A] differs from [term B]: [A is ...]; [B is ...]. | "Uncertainty is distinct from error/accuracy, uncertainty quantification defines the estimated distribution within which the true value lies and corresponds to the confidence in a prediction. Conversely, error is defined as the difference between an observed true value and a model prediction." | Singh 2024 |
| 14 | [Author] reported that [quantity] [value] in [context]. | "Chen et al. suggested tundra fire disturbance, while representing just 3% of the overall area, was responsible for 11% of the areal extent of all thermokarst detected between 1950 and 2015 across 4700 km2 area in arctic Alaska." | Jones 2024 |

### 6.3 방법

| # | 틀 | 원문 예 | 출처 |
|---|---|---|---|
| 15 | To prevent information leakage between training and evaluation, [test set construction]. | "To prevent information leakage between model training and evaluation, testing datasets were generated by randomly withholding individual RTS observations stratified by pre-expansion size classes" | Hall 2026 |
| 16 | [Variable] was excluded from [model], which uses only [information available at prediction time]. | "Elevation change was defined as the difference in mean elevation between successive RTS extents and was excluded from predictive models, which only rely on pre-expansion topographic conditions (Table 2)." | Hall 2026 |
| 17 | Because [inputs] extend beyond the range of the training data, [variable] was held constant at [value]. | "Since projected climate data extend beyond the range of the historic training data, climate predictors (SWI and precipitation) were held constant at their most recently observed (2021) values" | Hall 2026 |
| 18 | A more [rigorous] approach would be to [X]; without it, [expected effect]. | "We note that a more sophisticated approach would be to conduct a temporally dynamic footprint-aware analysis for each site, and adjust the corresponding pixels accordingly; the absence of this approach may reduce the goodness of fit in some instances." | Fisher 2023 |
| 19 | The [ML models] serve as a benchmark: if [physics model] explains [a]% and the best ML model [b]%, [interpretation]. | "This creates a benchmark to differentiate error between AquaSEBS and the in situ data that can be attributed to the remote sensing or the in situ data. For example, if AquaSEBS explained 50% of the variation in the in situ data, and the best machine learning model predicted 60%, then this suggests that AquaSEBS predicted most of the explainable variation." | Fisher 2023 |
| 20 | What we present here is not [X] but [Y]. | "It is also important that we emphasise that what we present here is not a validation study of several maps, but an investigation into what differences exist in map predictions" | Feeney 2022 |
| 21 | For [comparison], we assume that [reference] represents [...]. | "For the purpose of comparing the maps, we will assume that CS 2007, with its advantages over other soil surveys, represents the most accurate and complete picture of SOC concentrations available for GB." | Feeney 2022 |
| 22 | Because [data] are limited, we avoided [finer stratification]. | "Due to the limited number of CS 2007 data points, we wanted to avoid fractionating the land surface too finely." | Feeney 2022 |
| 23 | Values that differ from [baseline] at the 1% and 5% levels ([test]) are marked in bold and italics. | "Trends that are statistically significant according to a Mann-Kendall test at levels 1% and 5% are marked in bold and italics, respectively." | Isaksen 2022 |
| 24 | Uncertainty was estimated with [method] and is reported as 95% [interval type]. | "Uncertainty was assessed using a Monte Carlo framework to simulate RTS areal and volumetric change across climate scenarios and size classes, with uncertainty reported as 95% confidence ranges." | Hall 2026 |

### 6.4 결과

| # | 틀 | 원문 예 | 출처 |
|---|---|---|---|
| 25 | Between [t1] and [t2], [quantity] was [x]% of [domain] at [rate] (Table [n]). | "Between 2009 and 2014, permafrost thaw subsidence was detected across 15% of the 50 km2 terrestrial landscape area at a rate of 136.6 ha/yr (Table 1)." | Jones 2024 |
| 26 | ..., which corresponds to a decrease by a factor of ~[k] (Table [n]). | "the total estimated volumetric land surface subsidence declined to 1.9 ha-m/yr, which corresponds to a decrease by a factor of ~ 30 (Table 1)." | Jones 2024 |
| 27 | [Model] improved fit relative to [null/baseline] in all cases (ΔAIC: [...]). | "Fixed effects improved model fit relative to null models in all cases (ΔAIC: binomial = 25.81, area = 14.80, volume = 16.73)." | Hall 2026 |
| 28 | ..., with a sensitivity of [x]% ([a]/[b] [units] correctly identified). | "with a sensitivity of 60% (14/23 active slumps correctly identified) and specificity of 66% (16/24 stable slumps correctly identified)." | Hall 2026 |
| 29 | We next asked [question]. | "We next asked what the change in accuracy is with increasing spatial resolution from MODIS to Landsat." | Fisher 2023 |
| 30 | Although the sample was small and [caveat], [cautious statement]. | "Although the sample size was small and the comparison was not 1-to-1, it appeared that there may be a modest increase in correlation with Landsat" | Fisher 2023 |
| 31 | [k] outliers in [site] reduced [metric] from [a] to [b]. | "The small sample size was sensitive to outliers; the three outlier points in the Five-O data caused a reduction in r2 from 0.71 to 0.56." | Fisher 2023 |
| 32 | [Scheme A] gave lower agreement than [scheme B], indicating [interpretation]. | "Despite reasonable prediction accuracy, the spatial CV showed lower agreement with observed SOC compared to random CV indicating pessimistic map accuracy results (Fig. 3)." | Khanal 2023 |
| 33 | Part of the difference between [grid] and [point] values reflects what they represent, not error. | "It should be noted when comparing reanalysis (grid values) with point observations that some of the differences are due to what they represent and not necessarily errors in the data sets." | Isaksen 2022 |

### 6.5 고찰

| # | 틀 | 원문 예 | 출처 |
|---|---|---|---|
| 34 | In [scope], [system] evolved in [k] phases. | "In the first ~ 15 years since the 2007 Anaktuvuk River tundra fire, the permafrost-influenced landscape appears to have evolved in three phases." | Jones 2024 |
| 35 | [Result] is consistent with [study], supporting [interpretation]. | "The positive relationship between SWI and volumetric loss is consistent with Dai et al., supporting a broader role for temperature in regulating RTS dynamics." | Hall 2026 |
| 36 | This contrasts with [studies], where [...]. We attribute this divergence to [cause]: [explanation]. | "This contrasts with broader-scale studies, where volumetric losses are primarily attributed to latitudinal climate gradients. We attribute this divergence to scale dependency: within northern Alaska, variation in RTS size and morphology better explain differences in annual rates of change compared to climate." | Hall 2026 |
| 37 | This [pattern] is unlikely to reflect [artefact], as [control]. | "This inverse relationship is unlikely to reflect regional bias, as spatial differences between the Brooks Foothills and Noatak were explicitly accounted for in the model, and similar patterns have been observed in northwestern Canada." | Hall 2026 |
| 38 | These results should be interpreted cautiously given [source], which is largest for [condition]. | "These results should therefore be interpreted cautiously given substantial uncertainty, which is greatest for large slumps under RCP8.5." | Hall 2026 |
| 39 | This is a common limitation of data-driven models, which cannot represent [process]. | "This is a common limitation of data-driven models, which cannot explicitly account for site-level process limitations that may constrain continued change, such as ground ice depletion limiting RTS expansion." | Hall 2026 |
| 40 | Our models [do X], but they do not [Y] and remain untested in [Z]. | "While our models provide new insights into past and projected drivers of change in active RTSs, they do not predict the timing or precise location of new (uninitiated) thaw slumps and remain untested in dissimilar permafrost environments (e.g., High Arctic or alpine regions)." | Hall 2026 ('new insights' 는 뺀다) |
| 41 | The error has several sources: (1) [...]; (2) [...]; ... | "Although we often attribute errors entirely to the model, the error is in fact a combination of multiple sources including: (1) the model; (2) the in situ data; (3) the forcing data; (4) scale mismatch; and, (5) user error." | Fisher 2023 |
| 42 | [Estimates] should be interpreted as first-order approximations. | "and should be interpreted as first-order approximations." | Hall 2026 |
| 43 | We did not find evidence that [X] for [the cases examined]. | "we did not find evidence indicating that observations used periodically had a large impact on trends in CARRA for the few investigated locations here." | Isaksen 2022 |

### 6.6 자료 가용성

| # | 틀 | 원문 예 | 출처 |
|---|---|---|---|
| 44 | The data that support the findings of this study are published at [repository] ([DOI]). | "The data that support the findings of this study are published at the Arctic Data Center." | Jones 2024 |
| 45 | [Product] is available at [DOI]. The layers are in [format] at [resolution] in [CRS]. | "The spatial layers are in GeoTIFF format with a spatial resolution of 30 m and ESPG:32644 spatial reference system." | Khanal 2023 |

---

## 7. 피할 표현

### 7.1 영문

| 범주 | 피할 표현 | 대신 쓸 형태 | 예시 논문 실례 |
|---|---|---|---|
| 신규성 선언 | "In this study, we propose a novel ...", "To the best of our knowledge, this is the first ..."(문헌 확인 없이), "groundbreaking", "pioneering" | "We compare ...", "We evaluate ...". 처음이라는 주장은 문헌 확인 근거가 있을 때 Methods·Discussion 에서 한정해 쓴다 | Hall 2026: "to our knowledge, this is the first study to demonstrate ..." (근거가 있을 때만 허용) |
| 과장 동사 | leverage, harness, unlock, unleash, empower, revolutionize, delve into, shed light on, pave the way, bridge the gap, push the boundaries | use, apply, test, compare, estimate, show | Gay 2026: "no existing framework or methodology effectively harnesses, processes, simulates, captures, and reconciles zero-curtain dynamics"; Singh 2024: "This integration empowers users to identify instances" |
| 근거 없는 형용사 | comprehensive, robust(정의 없이), holistic, seamless, cutting-edge, powerful, sophisticated, remarkable, striking, crucial, pivotal, critical, paramount, vital, invaluable, unprecedented(수치 비교 없이) | 수치로 말한다. robust 는 "unchanged under [variants]" 처럼 조건을 적는다 | Gay 2026: "The transfer learning implementation represents a sophisticated domain adaptation methodology that leverages ..."; Khanal 2023: "provides a crucial baseline for evaluating the potential impacts of a changing climate on these critical carbon reservoirs" |
| 'significant' 오용 | P 값 없는 significant/significantly | 수치 차이와 구간, 또는 P 값 병기 | Khanal 2023: "a significant underrepresentation of these stocks in global-scale assessments"(초록, 검정 없음) |
| 신호 문구 | It is worth noting that, It is important to note, Notably, Importantly, Interestingly, In this regard, In the realm of, plays a crucial role in | 지우고 바로 내용을 쓴다 | Hall 2026: "Interestingly, RTS I41 has been stable for over 70 years, yet had the highest C and N concentration" |
| 접속사 연쇄 | 연속 문장 첫머리의 Furthermore, / Moreover, / Additionally, | 접속사 없이 쓰거나 논리 관계(because, whereas, therefore)를 쓴다 | Isaksen 2022: furthermore·moreover·additionally 12회(1,000단어당 1.1회) |
| 일반론 결말 | "These findings highlight the importance of ...", "underscore the need for ...", "offer valuable insights", "has important implications for"(내용 없이), "paves the way for future research" | 조건과 수치가 있는 함의 한 문장 또는 끝내기 | Gautam 2025: "highlighting the urgent need for improved predictive capabilities to inform adaptation strategies in the Arctic."; Khanal 2023: "offer valuable insight into the future of these carbon stocks" |
| 자기 평가 | "making these results a significant contribution to the scientific literature", "demonstrates the effectiveness of the proposed framework" | 평가는 독자에게 맡긴다 | Fisher 2023: "the net gain of insight from this analysis far outweighs these limitations, making these results a significant contribution to the scientific literature." |
| 메타 서술 | "This figure illustrates ...", "As shown in Figure X, it can be seen that ..." | 그림이 보여 주는 결과를 쓰고 괄호로 그림을 가리킨다 | Gay 2026: "This figure illustrates zero-curtain thermodynamics across one and two dimensions, synchronized per annum." |
| 수사·비유 | 비유 명사(riddle, mercurial nature), 감탄, 외국어 인사 | 기술 용어 | Gay 2026: "an inverse riddle", "the mercurial nature of the zero-curtain"; Hatami 2022: "donc à bientôt!", "We note a vivid spatial divide" |
| 구조 표지 | 세 단어 나열("accurate, robust, and scalable"), 수사 의문문, 본문 글머리표 남용, "X: a Y framework for Z" 제목, 단순 파이프라인에 약어 이름 붙이기 | 서술문. 이름은 재사용될 산출물에만 | Gay 2026: 'framework' 70회(1,000단어당 4.1회) |
| 대시 접속 | 문장 중간의 em-dash 삽입·부연 | 쉼표, 괄호, 문장 분리. 수치 범위의 en-dash 는 허용 | Gay 2026 초록 첫 문장: em-dash 삽입구 1쌍("nearly twice atmospheric levels"), 서론 정의 문장의 em-dash 연쇄 |
| 겹친 헤지 | "may potentially", "could possibly suggest" | 헤지는 한 번만 | 예시 논문 중 뚜렷한 예 없음 |
| 'framework' 남용 | 모델 하나·절차 하나를 framework 로 부름 | model, procedure, design, workflow 중 정확한 것 | Hatami 2022: framework 12회 |

### 7.2 예시 논문의 과장어 빈도

본문(참고문헌 제외) 단어 1,000개당 횟수다. 괄호 안은 총 횟수다.

| 논문 | 단어 | comprehensive | robust | framework | critical·crucial·pivotal | highlight·underscore | furthermore·moreover·additionally | significant(ly) |
|---|---|---|---|---|---|---|---|---|
| Jones 2024 | 6,421 | 0.0 (0) | 0.0 (0) | 0.0 (0) | 0.2 (1) | 0.5 (3) | 0.0 (0) | 0.3 (2) |
| Feeney 2022 | 7,363 | 0.0 (0) | 0.1 (1) | 0.0 (0) | 0.3 (2) | 0.3 (2) | 0.4 (3) | 0.5 (4) |
| Hall 2026 | 9,663 | 0.0 (0) | 0.1 (1) | 0.2 (2) | 0.1 (1) | 0.5 (5) | 0.2 (2) | 1.1 (11) |
| Fisher 2023 | 5,745 | 0.0 (0) | 0.2 (1) | 0.0 (0) | 0.3 (2) | 0.2 (1) | 0.7 (4) | 1.0 (6) |
| Isaksen 2022 | 10,691 | 0.1 (1) | 0.0 (0) | 0.0 (0) | 0.1 (1) | 0.2 (2) | 1.1 (12) | 1.4 (15) |
| Khanal 2023 | 7,988 | 0.3 (2) | 0.4 (3) | 0.3 (2) | 0.5 (4) | 0.4 (3) | 0.3 (2) | 1.0 (8) |
| Singh 2024 | 7,720 | 0.0 (0) | 0.1 (1) | 0.8 (6) | 0.4 (3) | 0.5 (4) | 1.2 (9) | 0.4 (3) |
| Hatami 2022 | 7,602 | 0.0 (0) | 0.0 (0) | 1.6 (12) | 0.5 (4) | 0.7 (5) | 0.3 (2) | 1.4 (11) |
| Gautam 2025 | 5,260 | 0.4 (2) | 0.6 (3) | 0.8 (4) | 0.6 (3) | 1.5 (8) | 1.1 (6) | 1.5 (8) |
| Gay 2026 | 17,253 | 1.9 (33) | 0.9 (16) | 4.1 (70) | 1.0 (17) | 0.3 (6) | 0.8 (13) | 1.1 (19) |

해석: 수치 서술이 많은 논문(Jones, Feeney, Hall, Fisher)은 과장어 빈도가 낮다. Gay 2026 은 'comprehensive' 와 'framework' 가 다른 논문의 수 배다. 'significant' 는 검정을 쓰는 논문에서 정당하게 많이 나오므로 빈도만으로 판단하지 않고 P 값 병기 여부로 판단한다. 이 표는 자동 계수이고 문맥 판단을 대신하지 않는다.

### 7.3 국문(원고·PPT)

사용자 문체 규칙(`~/.claude/rules/writing-tone.md`)을 원고·발표 자료에 적용한 목록이다.

| 유형 | 피할 표현 | 대신 쓸 형태 |
|---|---|---|
| 구어·권유 | ~해봅시다, ~살펴보자, ~알아보자, 함께 ~해요, 자 이제, 이제 ~해 본다 | ~를 비교한다, ~를 정리한다, 명사형 제목(실습, 비교, 결과) |
| 존댓말 | ~입니다, ~합니다 | ~이다, ~한다 |
| 과장 | 획기적, 혁신적, 놀라운, 압도적, 완벽한, 게임 체인저, 판도를 바꾸는, 대폭 향상 | 수치(오차 2.4 cm 감소, 95 % 구간 [a, b]) |
| 서사형 제목 | "여기까지 왔고, 하나가 남았다", "~는 이렇게 생겼다", "~를 직접 계산해 본다", "무너질 때 나타나는 세 가지 신호" | 명사형 제목: "지난 결과 요약과 남은 문제", "학습 곡선 해석", "계산 예시", "학습 불안정 신호" |
| 카드식 요약 | 이모지, 체크 기호, "핵심 포인트 3가지", "한 줄 요약", "오늘의 네 문장", 굵은 키워드 + 콜론 나열, 상자 안 문장 카드 | 표 하나 또는 그림 하나에 주장 한 문장. 글머리표는 4개 이하 |
| 번역투 상투구 | ~에 있어서, ~함에 있어, ~를 통해 ~를 도모한다, 중요한 역할을 한다, 다양한, 효과적으로, 시사하는 바가 크다, ~라고 할 수 있다, ~것으로 사료된다 | 주어·동사가 분명한 짧은 문장. "다양한" 은 개수로 바꾼다(학습기 10종) |
| 결말 상투구 | 본 연구는 ~의 가능성을 보여주었다, 향후 ~에 기여할 것으로 기대된다, 중요한 시사점을 제공한다, 의미 있는 결과를 도출하였다 | 결과 수치와 적용 조건 한 문장 |
| 신규성 선언 | 새로운 프레임워크를 제안한다, 최초로 | "~를 비교한다". 처음이라는 주장은 문헌 근거와 함께 한정한다 |
| 대시 접속 | "A — B", "결과 – 원인" | 문장 분리, 쉼표, 쌍점, 가운뎃점(·), 괄호. 수치 범위의 en-dash 는 허용 |
| 접속 연쇄 | 또한, 나아가, 뿐만 아니라, 아울러 의 연속 | 문장 순서로 논리를 보인다 |
| 비유 남용 | 줄다리기, 돋보기 심사, 완전 해부 | 비유는 본문 설명에서 한 번, 평이하게("생성자를 위조범에 비유하면 ...") |

---

## 8. 현행 초안 점검

대상: `docs/PAPER_SCIREP_FRONTMATTER_DRAFT.md`, `docs/MANUSCRIPT_DRAFT_METHODS_INTRO_2026-09-30.md`, `docs/MANUSCRIPT_DRAFT_RESULTS_ABSTRACT_2026-09-30.md`, `docs/MANUSCRIPT_DRAFT_SUPPORT_2026-09-30.md`. 7.1절 목록으로 자동 검색한 뒤 문맥을 보았다. 파일은 고치지 않았다.

| 항목 | 결과 | 조치 |
|---|---|---|
| em-dash | 4개 파일 모두 0개 | 없음 |
| 'novel' | 30회 검색되나 대부분 문서 이름 NOVELTY(참조 표기) | 원고로 옮길 때 참조 표기를 지운다 |
| 'harness' | 실험 도구 이름(h39–h52 harness) | 원고 문장에서는 "analysis scripts" 로 쓴다 |
| 'state-of-the-art' | ERA5-Land 논문 제목 인용 | 인용 제목이므로 유지 |
| 'significant' | 결과·초록 초안 11회. 예: "This was significant in one of four regions", "significant before correction only" | 지역별 P 값과 보정 방법을 함께 적는다(지침 Statistical guidelines). `scirep_format_and_drafts.md` 2.2절의 [MISSING] 19 와 같은 작업이다 |
| 'robust' | 결과·초록 초안 8회, 지원 문서 3회. 예: "Tibet was robust for variants (b) and (d)–(g)" | 사전 등록된 판정 기준을 Methods 에 정의하고 본문에서는 "unchanged under variants (b) and (d)–(g)" 처럼 조건을 쓴다 |
| 국문 상투구 | 7.3절 목록 검색 결과 0회 | 없음 |

---

## 9. 미확인·후속 사항

1. Hall 2026 은 Article in Press 판이다. 최종 편집본의 권·논문 번호는 [미확인]이다. 최종본이 나오면 파일을 교체하고 README 를 고친다.
2. Sci Rep 의 필수 한도 목록은 지침 본문이 아니라 내려받는 점검표 파일에 있다. 이 문서에서는 열지 않았다. `scirep_format_and_drafts.md` 2.3절이 점검표 PDF 를 요약했다.
3. 같은 시각에 다른 작업이 `references/08_recent_alt_2022_2026/` 를 채우는 중이다. 그쪽에 hall2026·gay2026 이 다시 받아지면 `references/INDEX.md` 통합 때 중복을 정리한다. 이 문서는 `references/INDEX.md` 를 고치지 않았다.
4. LLM 사용 기재 문구(5.9절)는 사용자 결정 사항이다.
5. 제목 형태(명사구 또는 주장 문장)는 사용자 결정 사항이다. 의문문 후보는 제외를 권한다(5.1절).
6. 예시 논문 그림은 품질 기준으로 쓰기에 부족하다. 그림 품질 기준은 `references/09_figure_exemplars/`, `references/13_figure_and_slide_design/` 의 자료로 따로 정한다.
7. 본문 절 단어 수는 `pdftotext` 기반 근사값이다(그림 설명문·표 문자 포함). 초록 단어 수도 하이픈 분리에 따라 몇 단어 다를 수 있다.
8. 제3자 자료(CALM, GTN-P, KPDC, ERA5-Land)의 URL·DOI 는 원고 작성 시 출처 페이지에서 확인한다[미확인].

---

## 부록. 재현 방법

- 후보 검색: OpenAlex `works?filter=primary_location.source.id:S196734849,from_publication_date:2022-01-01,title_and_abstract.search:<주제어>,type:article&sort=cited_by_count:desc`.
- 다운로드: `https://www.nature.com/articles/<id>.pdf`(Hall 2026 은 `<id>_reference.pdf`, Article in Press). `curl -L --max-time 120 -A 'Mozilla/5.0'`, `file`·`pdfinfo` 로 PDF 여부와 쪽수 확인(HTML 응답 1건은 버리고 Article in Press 판으로 다시 받음).
- 분석: `pdftotext` 로 본문 추출, 절 표지 사이 단어 수, 초록 문장 수, 과장어 정규식 계수, 그림 쪽 `pdftoppm -r 55` 렌더링 후 육안 확인.
- 공식 지침: https://www.nature.com/srep/author-instructions/submission-guidelines (2026-10-04 열람).
