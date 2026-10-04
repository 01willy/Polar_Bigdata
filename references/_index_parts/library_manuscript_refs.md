# 원고 참고문헌 라이브러리 (11 · 12 폴더)

**작성일** 2026-10-04
**대상** `docs/MANUSCRIPT_DRAFT_METHODS_INTRO_2026-09-30.md` 끝의 References 표(67항목), `docs/NOVELTY_POSITIONING_2026-09-29.md` 인용 문헌, 과제에서 지정한 공간 검증·방법론 문헌.
**폴더 규칙** `references/11_validation_and_methods/` 는 검증 설계, 통계, ML 학습기, 물리 유도 ML 방법(타 분야 포함)을 둔다. `references/12_manuscript_refs/` 는 영구동토 과학, ALT 제품, 공변량 자료원, 관측 자료 논문, 영구동토 물리-ML 선례를 둔다.
**내려받기 규칙** 합법 공개본만 받았다(출판사 OA, arXiv, bioRxiv, Copernicus, 기관 저장소). OA 위치는 OpenAlex `locations`(2026-10-04 조회)로 찾았고, 일부는 Semantic Scholar `openAccessPdf` 와 HAL API로 보완했다. 파일마다 `file`, `pdfinfo` 와 첫 3쪽 제목 일치를 확인했다. 이미 `references/` 아래에 있는 문헌은 새로 받지 않고 기존 경로를 적었다.
**표기** [PDF] 로컬 사본 있음. [차단] 공개본은 있으나 이 서버에서 자동 내려받기가 막힘(브라우저로 수동 내려받기 가능). [링크] 구독 필요. OA 상태는 OpenAlex 분류(gold, hybrid, bronze, green, closed)다.

## 집계

| 구분 | 항목 수 | 로컬 PDF | 차단 | 구독 필요 | 비고 |
|---|---|---|---|---|---|
| 11 검증·통계·ML 방법 | 40 | 31 | 3 | 6 | 새로 받은 22편 + 기존·동시 수집 사본 9편 |
| 12 영구동토·자료·응용 | 39 | 26 | 6 | 7 | 새로 받은 11편 + 기존·동시 수집 사본 15편 |
| 데이터셋·단행본 인용 | 25 | 해당 없음 | 해당 없음 | 해당 없음 | 3절 표 |
| 합계 | 104 | 57 | 9 | 13 | |

경로는 2026-10-04 13시대 기준이다. 다른 수집 작업이 같은 시각에 `08`, `09`, `14` 폴더를 정리하고 있었으므로, 경로가 맞지 않으면 파일 이름의 저자·연도로 찾는다. 동시에 진행된 `08_recent_alt_2022_2026` 수집과 8편이 겹쳤다. 같은 파일을 두 번 두지 않도록 먼저 받아진 사본 하나만 남겼다(Feng 2023, Tama 2025, Liu Z. 2024, Streletskiy 2026, Wang G. 2025, Du Q. 2026 은 08 폴더 사본, Garibaldi 2026 과 Ran 2022b 는 12 폴더 사본).

---

## 11 · 검증·통계·ML 방법 (`references/11_validation_and_methods/`)

### 11.1 공간 검증과 적용 범위

#### [PDF] Ploton et al., 2020
- **서지** Ploton, P. et al. Spatial validation reveals poor predictive performance of large-scale ecological mapping models. Nat. Commun. 11, 4540 (2020).
- **DOI** 10.1038/s41467-020-18321-y · **OA** gold (CC BY)
- **로컬** `references/09_figure_exemplars/ploton2020_spatial_validation.pdf` (기존 사본)
- **요약** 중앙아프리카 산림 조사 자료(수목 1,180만 그루)로 대규모 지상부 생물량 지도의 랜덤 포레스트 절차를 재현했다. 비공간 검증은 변동의 절반 이상을 설명한다고 나오지만, 공간 자기상관을 고려한 검증에서는 예측력이 거의 0이다.
- **관련성** 무작위 분할의 과대평가(보조 주장)의 대표 선례다. NOVELTY 9절 ML 방법론 문단에 인용되어 있다.

#### [PDF] Meyer and Pebesma, 2021
- **서지** Meyer, H. & Pebesma, E. Predicting into unknown space? Estimating the area of applicability of spatial prediction models. Methods Ecol. Evol. 12, 1620–1633 (2021).
- **DOI** 10.1111/2041-210X.13650 · **OA** hybrid (CC BY)
- **로컬** `references/00_core10/meyer2021_aoa.pdf`, `references/05_uq_transfer/meyer2021_aoa.pdf` (기존 사본)
- **요약** 중요도로 가중한 예측 변수 공간에서 학습 자료까지의 최소 거리로 비유사도 지수(DI)를 정의한다. 교차검증에서 얻은 DI 문턱값으로 적용 범위(AOA)를 구획하고, DI와 교차검증 성능의 관계로 성능 지도를 추정한다.
- **관련성** 제외 지역이 원천 공변량 범위 밖에 있는지 진단하는 도구다. 지도 마스크와 외삽 진단 문단의 인용 후보다.

#### [PDF] Meyer and Pebesma, 2022
- **서지** Meyer, H. & Pebesma, E. Machine learning-based global maps of ecological variables and the challenge of assessing them. Nat. Commun. 13, 2208 (2022).
- **DOI** 10.1038/s41467-022-29838-9 · **OA** gold (CC BY)
- **로컬** `references/05_uq_transfer/meyer2022_globalmaps_aoa.pdf` (기존 사본)
- **요약** 전 지구 생태 변수 ML 지도에 쓰인 자료와 방법을 검토하는 관점 논문이다. 희소하고 편중된 표본에서 예측값의 품질을 전역·국지적으로 평가할 수 있는지를 논의한다.
- **관련성** 고찰에서 "학습 자료가 없는 지역의 지도 정확도는 직접 평가할 수 없다"는 한계 서술의 근거다.

#### [링크] Wadoux et al., 2021
- **서지** Wadoux, A. M. J.-C., Heuvelink, G. B. M., de Bruin, S. & Brus, D. J. Spatial cross-validation is not the right way to evaluate map accuracy. Ecol. Model. 457, 109692 (2021).
- **DOI** 10.1016/j.ecolmodel.2021.109692 · **OA** OpenAlex green 표기이나 WUR 저장소 기록에 파일이 첨부되어 있지 않다
- **로컬** 구독 필요 (https://doi.org/10.1016/j.ecolmodel.2021.109692)
- **요약** 표준 교차검증과 공간 교차검증 모두 지도 정확도를 편향 추정할 수 있다고 주장한다. 모형에 의존하지 않는 비편향 정확도 평가는 확률 표본과 설계 기반 추론으로 얻는다고 결론짓는다(초록 요지는 검색 결과 기준, 원문 미대조).
- **관련성** 블록 절반 분할 채점이 "지도 전체 정확도"가 아니라 "제외 지역 채점 블록의 예측 오차"임을 명시하는 근거다.

#### [차단] Roberts et al., 2017
- **서지** Roberts, D. R. et al. Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography 40, 913–929 (2017). 온라인 2016.
- **DOI** 10.1111/ecog.02881 · **OA** OpenAlex closed, Semantic Scholar bronze(출판사 무료 열람)
- **로컬** 없음. Wiley PDF 요청이 403으로 막혔다 (https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/ecog.02881)
- **요약** 구조가 있는 자료에서 무작위 교차검증이 예측 오차를 과소추정하고 비인과 예측 변수의 과적합을 허용함을 보인다. 블록 교차검증을 권하되, 블록이 예측 변수 범위를 제한해 외삽을 유발할 수 있음을 함께 지적한다.
- **관련성** 원고 Methods에서 라벨 블록과 채점 블록 분리의 근거로 인용된다.

#### [PDF] Valavi et al., 2019
- **서지** Valavi, R., Elith, J., Lahoz-Monfort, J. J. & Guillera-Arroita, G. blockCV: an R package for generating spatially or environmentally separated folds for k-fold cross-validation of species distribution models. Methods Ecol. Evol. 10, 225–232 (2019). 온라인 2018.
- **DOI** 10.1111/2041-210X.13107 · **OA** green (게재본은 Digital.CSIC 저장소, 자동 접근 차단)
- **로컬** `references/11_validation_and_methods/valavi2019_blockcv.pdf` (bioRxiv 10.1101/357798 v1 사전 공개본, CC BY-NC-ND. 게재본과 문구가 다를 수 있다)
- **요약** 공간 또는 환경으로 분리된 k-겹을 만드는 R 패키지를 제시한다. 공변량의 공간 자기상관 거리 측정, 블록 생성과 겹 시각화 기능을 포함한다.
- **관련성** 0.5° 블록 분할의 도구 선례다. 원고 Methods 인용 후보다.

#### [PDF] Linnenbrink et al., 2024
- **서지** Linnenbrink, J., Milà, C., Ludwig, M. & Meyer, H. kNNDM CV: k-fold nearest-neighbour distance matching cross-validation for map accuracy estimation. Geosci. Model Dev. 17, 5897–5912 (2024).
- **DOI** 10.5194/gmd-17-5897-2024 · **OA** gold (CC BY)
- **로컬** `references/00_core10/linnenbrink2024_knndm.pdf`, `references/05_uq_transfer/linnenbrink2024_knndm.pdf` (기존 사본)
- **요약** NNDM LOO 교차검증을 k-겹으로 확장한다. 시험-학습 최근접 거리의 경험 분포를 예측-학습 거리 분포에 맞추는 겹 구성을 찾는다.
- **관련성** 우리 설계(지역 제외, 100 km 버퍼, 블록 절반 분할)와 대비되는 지도 정확도 추정 방식이다. 고찰에서 설계 선택의 이유를 설명할 때 쓴다.

### 11.2 물리 유도 ML과 미관측 대상 전이

#### [PDF] Karpatne et al., 2017
- **서지** Karpatne, A. et al. Theory-guided data science: a new paradigm for scientific discovery from data. IEEE Trans. Knowl. Data Eng. 29, 2318–2331 (2017).
- **DOI** 10.1109/TKDE.2017.2720168 · **OA** green (arXiv:1612.08544)
- **로컬** `references/11_validation_and_methods/karpatne2017_theory_guided_data_science.pdf` (arXiv 판)
- **요약** 과학 지식을 데이터 과학 모형에 넣는 이론 유도 데이터 과학(TGDS)을 개념화하고 연구 주제를 분류한다. 과학적 일관성을 일반화 가능한 모형의 필수 요소로 둔다.
- **관련성** 물리 증강 ML 용어의 출발점이다. 서론 첫 인용 후보다.

#### [PDF] Willard et al., 2022
- **서지** Willard, J., Jia, X., Xu, S., Steinbach, M. & Kumar, V. Integrating scientific knowledge with machine learning for engineering and environmental systems. ACM Comput. Surv. 55(4), 66 (2022).
- **DOI** 10.1145/3514228 · **OA** bronze (ACM)
- **로컬** `references/06_physics_ml/willard2020_physics_ml_survey.pdf` (기존 사본, arXiv:2003.04919v6. 게재본과 제목이 같다)
- **요약** 물리 기반 모형과 ML의 결합 기법을 응용 목적별로 정리하고, 물리 유도 ML과 하이브리드 틀의 분류 체계를 제시한다.
- **관련성** 원고 서론에서 사전학습, 잔차 학습, 재보정 물리 모형과의 비교를 묶어 인용한다.

#### [PDF] Jia et al., 2021
- **서지** Jia, X. et al. Physics-guided machine learning for scientific discovery: an application in simulating lake temperature profiles. ACM/IMS Trans. Data Sci. 2(3), 20 (2021).
- **DOI** 10.1145/3447814 · **OA** bronze (ACM, 자동 차단). arXiv:2001.11086 판을 받았다
- **로컬** `references/11_validation_and_methods/jia2021_pgml_lake_temperature.pdf` (arXiv 판)
- **요약** 물리 모형(General Lake Model) 출력과 순환 신경망을 결합한 PGRNN을 제안한다. 물리 모의값 사전학습과 에너지 보존 손실로 관측이 적을 때도 물리 모형보다 정확하고 물리적으로 일관된 수온 예측을 얻는다.
- **관련성** 두 기준선(원천 물리, 같은 라벨로 재보정한 물리)과 비교하는 보고 형식의 직접 선례다(원고 Methods "Following Read et al. (2019) and Jia et al. (2021)").

#### [PDF] Read et al., 2019
- **서지** Read, J. S. et al. Process-guided deep learning predictions of lake water temperature. Water Resour. Res. 55, 9173–9190 (2019).
- **DOI** 10.1029/2019WR024922 · **OA** hybrid (CC BY-NC)
- **로컬** `references/00_core10/read2019_pgdl_lake_temp.pdf`, `references/03_alt_forecasting/read2019_pgdl_lake_temp.pdf` (기존 사본)
- **요약** LSTM, 에너지 보존 위반 벌점, 과정 모형 모의값 사전학습을 결합한 PGDL을 수심별 호수 수온에 적용한다. 두 상세 호수에서 PGDL의 RMSE가 DL과 과정 모형보다 작았으나, 사전학습 자료의 변동성이 클 때에 한했다.
- **관련성** 물리 모의값 사전학습 범주와 두 기준선 형식의 선례다(원고 서론, NOVELTY 2·3절).

#### [PDF] Willard et al., 2021
- **서지** Willard, J. D. et al. Predicting water temperature dynamics of unmonitored lakes with meta-transfer learning. Water Resour. Res. 57, e2021WR029579 (2021).
- **DOI** 10.1029/2021WR029579 · **OA** hybrid (CC BY, Wiley 자동 차단). arXiv:2011.05369 판을 받았다
- **로컬** `references/11_validation_and_methods/willard2021_meta_transfer_unmonitored_lakes.pdf` (arXiv 판)
- **요약** 관측 호수 145곳의 원천 모형 가운데 대상 호수에 맞는 모형을 메타 모형으로 골라 미관측 호수 305곳의 수온을 예측한다. 미보정 과정 모형의 중앙 RMSE 2.52 °C에 대해 PGDL 메타 전이 앙상블은 1.88 °C였다.
- **관련성** 미관측 대상에서 라벨 수(1–50)에 따른 평가의 인접 분야 선례다(원고 서론, NOVELTY 3절).

#### [PDF] Feng et al., 2023
- **서지** Feng, D., Beck, H., Lawson, K. & Shen, C. The suitability of differentiable, physics-informed machine learning hydrologic models for ungauged regions and climate change impact assessment. Hydrol. Earth Syst. Sci. 27, 2357–2373 (2023).
- **DOI** 10.5194/hess-27-2357-2023 · **OA** gold (CC BY)
- **로컬** `references/08_recent_alt_2022_2026/feng2023_differentiable_hydrology_ungauged_hess.pdf` (동시 수집 사본)
- **요약** 신경망이 매개변수를 정하는 미분 가능 과정 수문 모형(δ 모형)을 미계측 유역(PUB)과 미계측 지역(PUR, 지역 제외) 조건에서 LSTM과 비교한다. 일 유량 지표와 10년 규모 추세 재현을 함께 평가한다.
- **관련성** 하이브리드 모형의 지역 홀드아웃 평가 선례다. 대상 라벨 수가 0으로 고정되어 있다는 점이 본 연구와 다르다(NOVELTY 3절).

#### [차단] Pool et al., 2019
- **서지** Pool, S., Viviroli, D. & Seibert, J. Value of a limited number of discharge observations for improving regionalization: a large-sample study across the United States. Water Resour. Res. 55, 363–377 (2019). 온라인 2018.
- **DOI** 10.1029/2018WR023855 · **OA** bronze, 취리히대 ZORA 수락본 green
- **로컬** 없음. Wiley(403)와 ZORA(봇 확인)가 모두 막혔다 (https://www.zora.uzh.ch/id/eprint/169290/1/Pool_et_al-2019-Water_Resources_Research.pdf)
- **요약** 미계측 유역에서 단기 관측 3–24개로 다수 공여 유역의 매개변수 예측에 가중을 주는 정보 지역화를 제안한다. 미국 579개 유역 leave-one-out 시험에서 표본 연도 10개 중 8개 연도에 최대 94 % 유역의 지역화가 개선되었다.
- **관련성** 라벨 수 축 평가의 직접 선례다(원고 서론, NOVELTY 8절).

#### [링크] Oudin et al., 2008
- **서지** Oudin, L., Andréassian, V., Perrin, C., Michel, C. & Le Moine, N. Spatial proximity, physical similarity, regression and ungaged catchments: a comparison of regionalization approaches based on 913 French catchments. Water Resour. Res. 44, W03413 (2008). 논문 번호 [미확인]
- **DOI** 10.1029/2007WR006240 · **OA** closed (HAL 기록 hal-02590551 은 서지만 있다)
- **로컬** 구독 필요
- **요약** 프랑스 913개 유역에서 회귀, 공간 근접, 물리 유사성 기반 매개변수 지역화를 비교한다. 공간 근접이 가장 낫고, 회귀 방식은 전국 단일 중앙값 매개변수와 거의 같은 수준이다.
- **관련성** 공변량 회귀로 만든 계수 지도가 지역 간에 실패한다는 결과(N7)의 수문학 선례다.

#### [PDF] Staudinger et al., 2025
- **서지** Staudinger, M. et al. How well do process-based and data-driven hydrological models learn from limited discharge data? Hydrol. Earth Syst. Sci. 29, 5005–5029 (2025).
- **DOI** 10.5194/hess-29-5005-2025 · **OA** gold (CC BY)
- **로컬** `references/11_validation_and_methods/staudinger2025_limited_discharge_models.pdf`
- **요약** 독일 중규모 유역 3곳에서 훈련 자료의 수와 선택 방식을 바꿔 과정 기반 모형 3종과 자료 기반 모형 4종의 학습 거동을 비교한다. 결합·조건부 엔트로피 같은 정보 측도로 평가한다.
- **관련성** "자료가 적으면 과정 모형이 유리하다"는 통념을 시험한 선례다. NOVELTY 9절 ML 방법론 문단에 인용되어 있다.

#### [차단] Portes et al., 2026
- **서지** Portes, C., Ienco, D. & Gabriel, E. When machine learning extrapolates in space: how local data shape spatial transferability. Spat. Stat. 74, 101009 (2026).
- **DOI** 10.1016/j.spasta.2026.101009 · **OA** hybrid (CC BY), HAL 사전본 hal-05697915
- **로컬** 없음. ScienceDirect와 HAL(https://hal.inrae.fr/hal-05697915/document)이 봇 확인 페이지를 반환했다
- **요약** 프랑스 Xylella fastidiosa 감시 자료(2015–2023)로 공간 외삽 조건의 예측 성능을 분석한다. 미관측 지역에 현지 자료를 단계적으로 넣는 공간 전이 평가로 외삽에서 내삽으로의 비선형 전이를 보인다.
- **관련성** 원고 서론의 "spatial prediction in general" 선례이고, 설계의 일반적 신규성을 주장하지 않는 근거다.

#### [링크] Steyerberg et al., 2004
- **서지** Steyerberg, E. W. et al. Validation and updating of predictive logistic regression models: a study on sample size and shrinkage. Stat. Med. 23, 2567–2586 (2004).
- **DOI** 10.1002/sim.1844 · **OA** closed
- **로컬** 구독 필요
- **요약** 다른 기관에서 개발한 예측 모형을 표본 크기별로 검증하고 갱신한다(GUSTO-I 자료). 절편·기울기 재보정 같은 간소한 갱신이 계수 재추정보다 낫고, 구조적 수정은 큰 표본에서만 권한다.
- **관련성** 적은 라벨의 재보정이 오차를 키울 수 있다는 원고 Methods 문장과, 원천 계수로 수축하는 P1 재보정의 근거다.

#### [PDF] Tama et al., 2025
- **서지** Tama, B. A., Wang, J., Janeja, V. et al. Learning subglacial bed topography from sparse radar with physics-guided residuals. arXiv:2511.14473 (2025). WACV 2026 게재(DOI 10.1109/WACV61042.2026.00528, Crossref 확인).
- **DOI** 10.48550/arXiv.2511.14473 · **OA** green
- **로컬** `references/08_recent_alt_2022_2026/tama2025_physics_guided_residual_bed_topography_wacv.pdf` (동시 수집 사본, arXiv 판)
- **요약** BedMachine 사전값 위의 빙하 두께 잔차를 질량 보존 등 물리 항과 자료 항으로 학습한다. 버퍼를 둔 블록 홀드아웃으로 그린란드 소지역 2곳에서 평가한다.
- **관련성** "물리 기준선 + 잔차" 구조의 인접 분야 선례다(NOVELTY 2절).

#### [PDF] Tahvildari et al., 2026
- **서지** Tahvildari, S. P., Shojaei, S. & Masihi, M. A physics-informed hybrid ML framework for pore pressure and fracture gradient prediction in carbonate reservoirs. Sci. Rep. 16, 8925 (2026). 논문 번호는 NOVELTY 기록, OpenAlex에는 없다
- **DOI** 10.1038/s41598-026-41773-z · **OA** gold (CC BY-NC-ND)
- **로컬** `references/11_validation_and_methods/tahvildari2026_physics_hybrid_pore_pressure.pdf`
- **요약** 고전 경험식 출력에 적응 보정층과 불확실성 모듈을 결합한 기울기 부스팅 하이브리드로 공극압과 파쇄 압력 구배를 예측한다. 해상 탄산염 가스전 시추공 6곳에서 보정된 고전 모형 RMSE 0.7–1.1 MPa에 대해 하이브리드는 0.45 MPa였다.
- **관련성** 보정된 물리식과 비교하는 형식의 Sci Rep 선례다. 영구동토 문헌은 아니다(NOVELTY 10절).

#### [PDF] O'Malley et al., 2026
- **서지** O'Malley, D. et al. In-context learning enables continental-scale subsurface temperature prediction from sparse local observations. arXiv:2605.16665 (2026).
- **DOI** 10.48550/arXiv.2605.16665 · **OA** green
- **로컬** `references/05_uq_transfer/omalley2026_incontext_subsurface_temp.pdf` (기존 사본)
- **요약** 희소 시추공 관측을 문맥으로 쓰는 트랜스포머로 미국 본토 지중 온도장을 예측한다(MAE 4.7 °C). 미세조정 없이 현지 관측 20개만으로 앨버타, 호주, 영국에 적응한다.
- **관련성** "학습 제외 지역 × 현지 관측 수" 평가의 가장 가까운 선례다. 신규성 문장 N1의 범위를 ALT·MAGT로 좁힌 근거다.

### 11.3 불확실성 구간과 채점

#### [PDF] Dunn et al., 2023
- **서지** Dunn, R., Wasserman, L. & Ramdas, A. Distribution-free prediction sets for two-layer hierarchical models. J. Am. Stat. Assoc. 118, 2491–2502 (2023). 온라인 2022.
- **DOI** 10.1080/01621459.2022.2060112 · **OA** green (arXiv:1809.07441)
- **로컬** `references/11_validation_and_methods/dunn2023_hierarchical_conformal.pdf` (arXiv 판)
- **요약** 집단마다 분포가 달라 교환 가능성이 깨지는 2층 계층 자료의 conformal 예측 집합을 다룬다. CDF pooling, 단일·반복 부분표본 방식을 제안하고, 커버리지 보장이 필요하면 반복 부분표본을 권한다.
- **관련성** 지역을 집단으로 둔 계층 conformal 구간(원고 Methods, 신규성 문장 N9)의 직접 근거다.

#### [PDF] Romano et al., 2019
- **서지** Romano, Y., Patterson, E. & Candès, E. J. Conformalized quantile regression. NeurIPS 32, 3538–3548 (2019).
- **DOI** 10.48550/arXiv.1905.03222 · **OA** green
- **로컬** `references/11_validation_and_methods/romano2019_conformalized_quantile_regression.pdf`
- **요약** conformal 예측과 분위 회귀를 결합해 이분산에 적응하는 구간을 만든다. 유한 표본 커버리지를 보장하고 다른 conformal 방법보다 짧은 구간을 보인다.
- **관련성** 분위 기반 구간 보정의 표준 참고문헌이다. 현재 원고 초안에는 인용이 없다(과제 지정 추가분).

#### [PDF] Gneiting and Raftery, 2007
- **서지** Gneiting, T. & Raftery, A. E. Strictly proper scoring rules, prediction, and estimation. J. Am. Stat. Assoc. 102, 359–378 (2007).
- **DOI** 10.1198/016214506000001437 · **OA** OpenAlex closed, 저자(워싱턴대 Raftery) 누리집 공개본 green
- **로컬** `references/11_validation_and_methods/gneiting2007_proper_scoring_rules.pdf` (저자 누리집 사본)
- **요약** 확률 예보 채점 규칙의 적정성 이론을 일반 확률 공간에서 정리한다. CRPS, 분위 점수, 구간 점수 등 예를 제시한다.
- **관련성** 원고 Methods의 CRPS, pinball 손실, 구간 점수 계산 근거다.

#### [PDF] Kakhani et al., 2024
- **서지** Kakhani, N. et al. Uncertainty quantification of soil organic carbon estimation from remote sensing data with conformal prediction. Remote Sens. 16, 438 (2024).
- **DOI** 10.3390/rs16030438 · **OA** gold (CC BY)
- **로컬** `references/11_validation_and_methods/kakhani2024_conformal_soc.pdf`
- **요약** 유럽 LUCAS 토양 자료에서 원격탐사 공변량과 랜덤 포레스트에 conformal prediction을 결합해 토양 유기탄소 예측 구간을 만든다. 기존 불확실성 정량화 방법과 여러 지표로 비교한다.
- **관련성** 환경 매핑의 conformal 커버리지 선례(N9 인접 선례)다.

### 11.4 표 형식 학습기와 생성 모형

#### [PDF] Hollmann et al., 2025
- **서지** Hollmann, N. et al. Accurate predictions on small data with a tabular foundation model. Nature 637, 319–326 (2025).
- **DOI** 10.1038/s41586-024-08328-6 · **OA** hybrid (CC BY)
- **로컬** `references/11_validation_and_methods/hollmann2025_tabpfn_nature.pdf`
- **요약** 수백만 개 합성 자료로 사전학습한 표 형식 기반 모형 TabPFN을 제시한다. 표본 1만 개 이하 자료에서 기존 방법보다 정확하고, 분류에서는 2.8 s 만에 4 h 조정한 앙상블을 넘는다.
- **관련성** TabPFN v2 실행(LGT, LGF)의 출처다(원고 Methods).

#### [PDF] Qu et al., 2025 (TabICL)
- **서지** Qu, J., Holzmüller, D., Varoquaux, G. & Le Morvan, M. TabICL: a tabular foundation model for in-context learning on large data. arXiv:2502.05564 (2025). 학회 게재 [미확인]
- **DOI** 10.48550/arXiv.2502.05564 · **OA** green
- **로컬** `references/11_validation_and_methods/qu2025_tabicl.pdf`
- **요약** 열-행 어텐션으로 고정 차원 행 임베딩을 만든 뒤 트랜스포머로 문맥 내 학습을 하는 표 형식 분류 모형이다. TALENT 분류 자료 200개에서 TabPFNv2와 대등하고 최대 10배 빠르다.
- **관련성** TabICL 계열의 원 논문이다. 원고는 v2(아래)를 인용한다.

#### [PDF] Qu et al., 2026 (TabICLv2)
- **서지** Qu, J., Holzmüller, D., Varoquaux, G. & Le Morvan, M. TabICLv2: a better, faster, scalable, and open tabular foundation model. arXiv:2602.11139 (2026).
- **DOI** 10.48550/arXiv.2602.11139 · **OA** green
- **로컬** `references/11_validation_and_methods/qu2026_tabiclv2.pdf`
- **요약** 합성 자료 생성기, 확장 가능한 softmax 어텐션, Muon 최적화기를 도입한 회귀·분류 기반 모형이다. 조정 없이 TabArena와 TALENT에서 RealTabPFN-2.5를 넘는다고 보고한다.
- **관련성** 원고 Methods의 TabICL v2(tabicl 2.0.2) 출처다. ICML 2026 게재 여부는 [미확인]으로 남긴다.

#### [PDF] Prokhorenkova et al., 2018
- **서지** Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V. & Gulin, A. CatBoost: unbiased boosting with categorical features. NeurIPS 31, 6639–6649 (2018).
- **DOI** 10.48550/arXiv.1706.09516 · **OA** green
- **로컬** `references/11_validation_and_methods/prokhorenkova2018_catboost.pdf` (arXiv 판)
- **요약** 순서 부스팅과 범주형 특징 처리로 기존 기울기 부스팅의 목표 누설에 따른 예측 이동을 줄인다.
- **관련성** 방법 축 기본 학습기 CatBoost의 출처다.

#### [PDF] Gorishniy et al., 2021
- **서지** Gorishniy, Y., Rubachev, I., Khrulkov, V. & Babenko, A. Revisiting deep learning models for tabular data. NeurIPS 34 (2021).
- **DOI** 10.48550/arXiv.2106.11959 · **OA** green
- **로컬** `references/11_validation_and_methods/gorishniy2021_ft_transformer.pdf` (arXiv 판)
- **요약** 표 형식 DL 구조를 같은 규약으로 비교하고 ResNet형 기준선과 FT-Transformer를 제시한다. GBDT와 비교해 보편적으로 우월한 방법은 없다고 결론짓는다.
- **관련성** 축소 FT-Transformer형 학습기의 출처다.

#### [PDF] Gorishniy et al., 2025 (TabM)
- **서지** Gorishniy, Y., Kotelnikov, A. & Babenko, A. TabM: advancing tabular deep learning with parameter-efficient ensembling. ICLR (2025). arXiv 2024.
- **DOI** 10.48550/arXiv.2410.24210 · **OA** green
- **로컬** `references/11_validation_and_methods/gorishniy2025_tabm.pdf` (arXiv 판)
- **요약** 하나의 모형이 매개변수를 공유하는 다수 MLP 앙상블을 모방하는 TabM을 제안한다. 대규모 평가에서 MLP 계열이 어텐션·검색 기반 구조보다 실용적임을 보인다.
- **관련성** 원고의 다중 헤드 MLP(코드명 `tabm`)가 TabM과 다른 구조라는 명시 문장의 근거다.

#### [PDF] Holzmüller et al., 2024
- **서지** Holzmüller, D., Grinsztajn, L. & Steinwart, I. Better by default: strong pre-tuned MLPs and boosted trees on tabular data. NeurIPS 37 (2024).
- **DOI** arXiv:2407.04491 (OpenAlex에 arXiv DOI 등록 없음) · **OA** green
- **로컬** `references/11_validation_and_methods/holzmuller2024_realmlp_better_by_default.pdf` (arXiv 판)
- **요약** 개선 MLP인 RealMLP와, GBDT·RealMLP의 메타 조정 기본값을 제시한다. 메타 훈련 자료 118개로 조정하고 별도 자료 90개에서 검증한다.
- **관련성** RealMLP(pytabkit 1.7.3 기본값) 학습기의 출처다.

#### [PDF] Bergstra and Bengio, 2012
- **서지** Bergstra, J. & Bengio, Y. Random search for hyper-parameter optimization. J. Mach. Learn. Res. 13, 281–305 (2012).
- **DOI** 없음 (JMLR) · **OA** 공개 (https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)
- **로컬** `references/11_validation_and_methods/bergstra2012_random_search.pdf`
- **요약** 하이퍼파라미터 최적화에서 무작위 탐색이 격자 탐색보다 효율적임을 경험과 이론으로 보인다.
- **관련성** LGF 조정 실험의 무작위 탐색(최대 34개 설정) 근거다.

#### [PDF] Cawley and Talbot, 2010
- **서지** Cawley, G. C. & Talbot, N. L. C. On over-fitting in model selection and subsequent selection bias in performance evaluation. J. Mach. Learn. Res. 11, 2079–2107 (2010).
- **DOI** 없음 (JMLR) · **OA** 공개 (https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf)
- **로컬** `references/11_validation_and_methods/cawley2010_model_selection_overfitting.pdf`
- **요약** 모형 선택 기준 추정량의 분산이 선택 단계의 과적합을 일으키고, 그 영향이 학습 알고리즘 간 성능 차이와 비슷한 크기일 수 있음을 보인다.
- **관련성** 기본값보다 겹별로 일관되게 나을 때만 비기본 설정을 고르는 규칙의 근거다.

#### [PDF] Durkan et al., 2019
- **서지** Durkan, C., Bekasov, A., Murray, I. & Papamakarios, G. Neural spline flows. NeurIPS 32 (2019).
- **DOI** 10.48550/arXiv.1906.04032 · **OA** green
- **로컬** `references/11_validation_and_methods/durkan2019_neural_spline_flows.pdf`
- **요약** 단조 유리 2차 스플라인 변환으로 결합·자기회귀 흐름의 유연성을 높이고 해석적 역변환을 유지한다.
- **관련성** 조건부 스플라인 정규화 흐름 학습기의 출처다.

#### [PDF] Ho et al., 2020
- **서지** Ho, J., Jain, A. & Abbeel, P. Denoising diffusion probabilistic models. NeurIPS 33, 6840–6851 (2020).
- **DOI** 10.48550/arXiv.2006.11239 · **OA** green
- **로컬** `references/11_validation_and_methods/ho2020_ddpm.pdf`
- **요약** 확산 확률 모형을 노이즈 제거 점수 정합과 연결된 가중 변분 하한으로 학습해 고품질 이미지 생성을 보인다. (OpenAlex 초록 필드가 다른 문서로 오염되어 있어 요약은 PDF 초록 기준이다.)
- **관련성** 200단계 노이즈 제거 확산 학습기의 출처다.

#### [PDF] Lipman et al., 2023
- **서지** Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M. & Le, M. Flow matching for generative modeling. ICLR (2023). arXiv 2022.
- **DOI** 10.48550/arXiv.2210.02747 · **OA** green
- **로컬** `references/11_validation_and_methods/lipman2023_flow_matching.pdf`
- **요약** 고정된 조건부 확률 경로의 벡터장을 회귀해 연속 정규화 흐름을 시뮬레이션 없이 학습한다. 확산 경로와 최적 수송 경로를 포함한다.
- **관련성** 조건부 flow matching 학습기의 출처다.

### 11.5 다중 비교와 동등성 검정

#### [링크] Hartung and Knapp, 2001
- **서지** Hartung, J. & Knapp, G. A refined method for the meta-analysis of controlled clinical trials with binary outcome. Stat. Med. 20, 3875–3889 (2001).
- **DOI** 10.1002/sim.1009 · **OA** closed
- **로컬** 구독 필요
- **요약** 랜덤효과 메타분석에서 처리 효과 분산의 개선 추정량에 기반한 검정을 제안한다. 모의실험에서 명목 유의수준을 기존 검정보다 잘 지킨다.
- **관련성** 지역 간 Δ의 Hartung-Knapp 랜덤효과 평균(LGX)의 근거다.

#### [링크] Holm, 1979
- **서지** Holm, S. A simple sequentially rejective multiple test procedure. Scand. J. Stat. 6, 65–70 (1979).
- **DOI** 10.2307/4615733 (JSTOR. 원고 표에는 DOI가 없으므로 추가할 수 있다) · **OA** closed
- **로컬** 구독 필요
- **요약** 가설을 하나씩 기각하는 순차 기각형 다중 검정으로, 참 가설의 어떤 조합에서도 1종 오류율을 통제한다.
- **관련성** 사전 지정 대비 10개의 Holm 보정 근거다.

#### [링크] Schuirmann, 1987
- **서지** Schuirmann, D. J. A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. J. Pharmacokinet. Biopharm. 15, 657–680 (1987).
- **DOI** 10.1007/BF01068419 · **OA** OpenAlex green(Zenodo 1232484, 'public-domain' 표기). 1987년 학술지 논문의 공공 영역 표기는 권리 근거를 확인할 수 없어 내려받지 않았다
- **로컬** 구독 필요
- **요약** 생물학적 동등성 판정에서 두 단측 검정(TOST)과 검정력 접근을 비교한다. 단측 α = 0.05의 TOST가 대부분 경우 검정력 접근보다 균일하게 우월하다.
- **관련성** ±δ 동등성 판정 규칙의 근거다.

---

## 12 · 영구동토·자료·응용 (`references/12_manuscript_refs/`)

### 12.1 Stefan 해와 ALT 지역 매핑의 고전

#### [차단] Stefan, 1891
- **서지** Stefan, J. Über die Theorie der Eisbildung, insbesondere über die Eisbildung im Polarmeere. Ann. Phys. 278(2), 269–286 (1891).
- **DOI** 10.1002/andp.18912780206 · **OA** green (Zenodo 2042312, 공공 영역)
- **로컬** 없음. Zenodo가 이 서버 네트워크에 403(비정상 트래픽 제한)을 반환했다 (https://zenodo.org/record/2042312)
- **요약** 결빙 경계가 이동하는 열전도 문제를 다루고, 극해 얼음 두께 성장의 이론을 제시한 원 논문이다. 얼음 두께가 누적 냉각량의 제곱근에 비례한다는 해의 출발점이다.
- **관련성** 원고 Stefan 식의 원전이다. 원고 표의 [verify]는 이번 OpenAlex 조회로 권·쪽·DOI가 확인되었다.

#### [링크] Nelson et al., 1997
- **서지** Nelson, F. E. et al. Estimating active-layer thickness over a large region: Kuparuk River basin, Alaska, U.S.A. Arct. Alp. Res. 29, 367–378 (1997).
- **DOI** 10.1080/00040851.1997.12003258 · **OA** closed
- **로컬** 구독 필요
- **요약** 알래스카 Kuparuk 유역 26,278 km²에서 탐침, 온도 기록계, GIS를 결합해 1995년 여름 주 단위 ALT 지도를 만든다. 대표 1 km 단위에서 예측값이 측정 평균과 약 6 cm 이내다.
- **관련성** Stefan 계수 E를 현지 측정으로 보정하는 관행의 출처다(원고 서론 첫 문단).

#### [링크] Shiklomanov and Nelson, 2002
- **서지** Shiklomanov, N. I. & Nelson, F. E. Active-layer mapping at regional scales: a 13-year spatial time series for the Kuparuk region, north-central Alaska. Permafr. Periglac. Process. 13, 219–230 (2002).
- **DOI** 10.1002/ppp.425 · **OA** closed
- **로컬** 구독 필요
- **요약** 기온을 강제력으로, 피복별 edaphic 계수를 국지 반응으로 둔 Stefan 해로 Kuparuk 지역의 13년 연별 ALT 지도를 만든다. 적당한 기후·edaphic 정보로 지역 규모 ALT 지도가 가능하고, 연간 기후 변동이 ALT를 크게 바꾼다.
- **관련성** 피복별 E로 지역 ALT를 그리는 방식의 선례다. 계수 지도 실험(N7) 서술의 배경이다.

#### [링크] Riseborough et al., 2008
- **서지** Riseborough, D., Shiklomanov, N., Etzelmüller, B., Gruber, S. & Marchenko, S. Recent advances in permafrost modelling. Permafr. Periglac. Process. 19, 137–156 (2008).
- **DOI** 10.1002/ppp.615 · **OA** closed
- **로컬** 구독 필요
- **요약** 2003년 이후의 영구동토 모델링 진전을 시간·열·공간 기준으로 분류해 검토한다. 공간 모형 안의 열 모형 적용 확대와 GCM 지표 방안에의 영구동토 포함을 진전으로, 아격자 변동성 매개변수화를 과제로 든다.
- **관련성** Stefan, Kudryavtsev 같은 단순 해석 모형의 위치를 설명하는 리뷰다.

### 12.2 ALT·지온 통계·ML 지도와 기존 제품

#### [PDF] Aalto et al., 2018
- **서지** Aalto, J., Karjalainen, O., Hjort, J. & Luoto, M. Statistical forecasting of current and future circum-Arctic ground temperatures and active layer thickness. Geophys. Res. Lett. 45, 4889–4898 (2018).
- **DOI** 10.1029/2018GL078007 · **OA** bronze (Wiley 자동 차단), 헬싱키대 HELDA 저장소 사본
- **로컬** `references/12_manuscript_refs/aalto2018_circumarctic_ground_temp_alt.pdf` (HELDA 사본)
- **요약** 다중 통계 기법 앙상블로 환북극 MAGT와 ALT를 예측하고 거리 블록 교차검증으로 전이성을 평가한다. 현재 영구동토가 유지될 조건의 면적을 15.1 ± 2.8 × 10⁶ km²로 추정한다.
- **관련성** ALT 거리 블록 검증의 선례다(원고 서론, NOVELTY 8절).

#### [PDF] Karjalainen et al., 2019
- **서지** Karjalainen, O. et al. Circumpolar permafrost maps and geohazard indices for near-future infrastructure risk assessments. Sci. Data 6, 190037 (2019).
- **DOI** 10.1038/sdata.2019.37 · **OA** gold
- **로컬** `references/09_figure_exemplars/karjalainen2019_circumpolar_geohazard_maps.pdf` (기존 사본)
- **요약** 30초 해상도 지리 자료로 지온과 ALT를 통계 모형으로 예측하고, 지반 얼음·입도·경사와 결합해 지반 위험 지수를 만든다. RCP 2.6, 4.5, 8.5로 2041–2060년과 2061–2080년을 투영한다.
- **관련성** ALT 거리 제외(500 km) 검증의 선례다(원고 서론, NOVELTY 8절).

#### [PDF] Ran et al., 2022a
- **서지** Ran, Y. et al. New high-resolution estimates of the permafrost thermal state and hydrothermal conditions over the Northern Hemisphere. Earth Syst. Sci. Data 14, 865–884 (2022).
- **DOI** 10.5194/essd-14-865-2022 · **OA** gold (CC BY)
- **로컬** `references/00_core10/ran2022_panarctic.pdf`, `references/01_benchmark/ran2022_panarctic.pdf` (기존 사본)
- **요약** 시추공 1,002곳 MAGT와 452개 지점 ALT, 원격탐사 자료를 앙상블 통계 학습으로 결합해 북반구 1 km MAGT·ALT(2000–2016)를 만든다. ALT RMSE는 86.93 ± 19.61 cm다.
- **관련성** 공변량 회귀 ALT 지도의 대표 사례이고 기존 제품 비교 후보(TPDC 계정 필요)다.

#### [PDF] Ran et al., 2022b
- **서지** Ran, Y. et al. Permafrost degradation increases risk and large future costs of infrastructure on the Third Pole. Commun. Earth Environ. 3, 238 (2022).
- **DOI** 10.1038/s43247-022-00568-6 · **OA** gold
- **로컬** `references/12_manuscript_refs/ran2022b_third_pole_infrastructure.pdf`
- **요약** 자료 기반 투영, 다중 위험 지수, 수명 교체 모형을 결합해 청장고원 기반시설의 영구동토 열화 비용을 추정한다. SSP245에서 2090년까지 약 63.1억 달러의 추가 비용이 필요하다.
- **관련성** E를 공변량에서 ML로 추정한 선례다(원고 서론). ALT 오차는 보고하지 않았다(NOVELTY 2절).

#### [PDF] Gautam et al., 2025
- **서지** Gautam, S., Mishra, U., Scott, S. et al. Machine learning and process-based modeling of spatiotemporal changes in active layer thickness across Alaska. Sci. Rep. 15, 42420 (2025).
- **DOI** 10.1038/s41598-025-26586-w · **OA** gold (CC BY)
- **로컬** `references/00_core10/gautam2025_alaska_alt.pdf`, `references/01_benchmark/gautam2025_alaska_alt.pdf` (기존 사본)
- **요약** 알래스카 ALT를 랜덤 포레스트와 Stefan 모형으로 비교하고 CMIP6 SSP2-4.5, SSP5-8.5로 투영한다. 훈련 R²는 0.84 대 0.53, 시험 R²는 0.24 대 0.54다.
- **관련성** 대상 저널의 가장 가까운 선례다. 무작위 70/30 1회 분할이라는 설계 차이를 서론에 적는다.

#### [PDF] Liu Z. et al., 2024
- **서지** Liu, Z., Kimball, J. S., Ballantyne, A. P. et al. Widespread deepening of the active layer in northern permafrost regions from 2003 to 2020. Environ. Res. Lett. 19, 014020 (2024). 온라인 2023.
- **DOI** 10.1088/1748-9326/ad0f73 · **OA** gold (CC BY)
- **로컬** `references/08_recent_alt_2022_2026/liu2024_widespread_alt_deepening_2003_2020_erl.pdf` (동시 수집 사본)
- **요약** 현지 ALT 2,966 지점-연도로 학습한 ML로 2003–2020년 북부 영구동토 지역 1 km 연별 ALT를 만든다. 관측 대비 R² 0.97이고 토양 물성, 고도, 지표 온도가 주요 인자다.
- **관련성** 기존 ALT 제품 비교(XG) 후보다. R² 0.97은 무작위 검증 수치라는 점을 비교 시 함께 적는다(EXPERIMENT_LOG 기록).

#### [PDF] Wei et al., 2026 (Wei & Chen 2026)
- **서지** Wei, Y., Wu, Z., Wang, J. et al. A 1 km resolution dataset of Northern Hemisphere permafrost active layer thickness (2000–2024). Earth Syst. Sci. Data Discuss., essd-2026-447 (2026).
- **DOI** 10.5194/essd-2026-447 · **OA** 공개 사전본
- **로컬** `references/08_recent_alt_2022_2026/wei2026_nh_alt_1km_2000_2024_essdd.pdf` (기존 사본)
- **요약** 연별 ALT 관측 2,196개로 2000–2024년 북반구 1 km 연별 ALT를 재구성한다. 지점 제외 교차검증의 앙상블 R² 0.76, RMSE 60.48 cm다.
- **관련성** 경쟁 제품이자 커버리지 비교 대상이다(N9). 학습 표가 CALM 지점이라 표본 중복 표시가 필요하다.

#### [링크] Li C. et al., 2022
- **서지** Li, C. et al. Active layer thickness in the Northern Hemisphere: changes from 2000 to 2018 and future simulations. J. Geophys. Res. Atmos. 127, e2022JD036785 (2022).
- **DOI** 10.1029/2022JD036785 · **OA** closed
- **로컬** 구독 필요
- **요약** 관측 자료와 ERA5-Land 기온으로 Stefan 모형을 써서 2000–2018년 북반구 1 km ALT를 모의하고 CMIP6로 미래를 투영한다. 평균 ALT는 127.19 cm에서 145.37 cm로 연 0.65 cm 증가했다.
- **관련성** 현지 관측으로 E를 보정하는 관행의 출처다(NOVELTY 4절). E 지도 공개 여부는 [미확인]이다.

#### [PDF] Peng et al., 2018
- **서지** Peng, X. et al. Spatiotemporal changes in active layer thickness under contemporary and projected climate in the Northern Hemisphere. J. Clim. 31, 251–266 (2018). 온라인 2017.
- **DOI** 10.1175/JCLI-D-16-0721.1 · **OA** bronze (AMS 공개)
- **로컬** `references/12_manuscript_refs/peng2018_nh_alt_changes.pdf`
- **요약** 현장 관측, 융해 지수 기반 Stefan 해, E 계수로 북반구 1971–2000년 ALT 기후값과 1850–2100년 변화를 계산한다. 현장 ALT는 대체로 40–320 cm다.
- **관련성** E 보정 관행의 출처다. NOVELTY의 "Peng 2018"은 검색 요약 수준 기록이었고, 이 논문으로 특정한 것은 이번 작업의 판단이다 [원문 대조 필요].

#### [링크] Shen et al., 2023
- **서지** Shen, T. et al. Changes in permafrost spatial distribution and active layer thickness from 1980 to 2020 on the Tibet Plateau. Sci. Total Environ. 859, 160381 (2023). 온라인 2022.
- **DOI** 10.1016/j.scitotenv.2022.160381 · **OA** closed
- **로컬** 구독 필요
- **요약** 티베트 고원 영구동토 분포와 ALT를 1980–2020년 10년 단위 1 km로 그린다. 시추공 검증에서 분포는 ML이, ALT는 경험 모형(Stefan)이 더 잘 맞았고, 2010년대 지역 평균 ALT는 1980년대보다 18.94 cm 두껍다.
- **관련성** "ALT에서 Stefan 우위" 결론의 방향 선례다(N6). 원고 영문 문장 N6에 인용되어 있다.

#### [PDF] Du J. et al., 2026
- **서지** Du, J. et al. Assessing spatial heterogeneity of active layer thickness over Arctic-foothills tundra, North Slope Alaska. Cryosphere 20, 4277–4291 (2026).
- **DOI** 10.5194/tc-20-4277-2026 · **OA** gold (CC BY)
- **로컬** `references/08_recent_alt_2022_2026/du2026_alt_heterogeneity_arctic_foothills_tc.pdf` (기존 사본)
- **요약** North Slope 구릉 툰드라의 90 m × 90 m 구획 4곳 집중 조사와 원격탐사·ML로 ALT 공간 이질성을 분석한다. 5 m 규모에서는 식생, 습윤, 지하 암석, 미지형이, 10 m 규모에서는 지형(약 65 %)이 지배한다.
- **관련성** 점 라벨의 대표성 오차와 축척 차이를 서술할 때 쓴다.

#### [PDF] Du Q. et al., 2026
- **서지** Du, Q., Wang, F., Li, G. et al. Seasonal precipitation provides modest incremental information for retrospective estimation of active-layer thickness at monitored sites along the Qinghai–Tibet engineering corridor, China. Buildings 16, 3023 (2026).
- **DOI** 10.3390/buildings16153023 · **OA** gold (CC BY)
- **로컬** `references/08_recent_alt_2022_2026/du2026_seasonal_precip_alt_retrospective_buildings.pdf` (동시 수집 사본)
- **요약** 청장 공학 회랑 시추공 54곳(2001–2020)의 연별 ALT에 대해 롤링 원점 설계로 여름 강수의 추가 정보를 평가한다. RMSE 감소는 0.2035 m에서 0.1993 m로 2.05 %다.
- **관련성** 티베트 고원 안에서 공간 블록·시간 분할로 평가한 사례다(NOVELTY 9절, 사용자 PDF 대조).

#### [차단] Ahajjam et al., 2025
- **서지** Ahajjam, A., Wilcox, A., Soaper, M. et al. Multi-horizon active layer thickness prediction across circum-Arctic permafrost regions using geospatial machine learning. J. Geophys. Res. Mach. Learn. Comput. 2(4), e2025JH000969 (2025).
- **DOI** 10.1029/2025JH000969 · **OA** gold (CC BY)
- **로컬** 없음. Wiley(AGU) PDF가 403으로 막혔다 (https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1029/2025JH000969)
- **요약** 지리 공간 특징 80개에서 다단계 특징 선택을 거친 뒤 CatBoost, Extra Trees, Bagging 가중 앙상블로 CALM 115지점의 연 최대 ALT를 최대 5년 앞까지 예측한다.
- **관련성** 검증 방식(지점·지역 홀드아웃 여부)을 투고 전 본문으로 확인해야 하는 위협 문헌이다(NOVELTY 8절). `08_recent_alt_2022_2026/ahajjam2026_..._egusphere.pdf` 는 별개 논문이다.

#### [PDF] Streletskiy et al., 2026
- **서지** Streletskiy, D. A. et al. Long-term monitoring of active layer thickness confirms global permafrost degradation. Commun. Earth Environ. 7, 671 (2026).
- **DOI** 10.1038/s43247-026-03824-1 · **OA** gold (CC BY)
- **로컬** `references/08_recent_alt_2022_2026/streletskiy2026_calm_long_term_alt_cee.pdf` (동시 수집 사본)
- **요약** 북극, 남극, 산악 영구동토의 관측지 156곳 ALT(2000–2024)를 분석해 북극 55 %, 남극 38 % 지점에서 유의한 증가를 보인다. 북극의 변화는 융해 도일 증가, 다음으로 강우 증가로 설명된다.
- **관련성** 서론 배경과 지점 내 √DDT 계수 비교(NOVELTY 10절)의 근거다. "156곳"은 이 PDF로 대조할 수 있다.

#### [PDF] Hasler et al., 2015
- **서지** Hasler, A., Geertsema, M., Foord, V., Gruber, S. & Noetzli, J. The influence of surface characteristics, topography and continentality on mountain permafrost in British Columbia. Cryosphere 9, 1025–1038 (2015).
- **DOI** 10.5194/tc-9-1025-2015 · **OA** gold (CC BY)
- **로컬** `references/12_manuscript_refs/hasler2015_mountain_permafrost_bc.pdf`
- **요약** 브리티시컬럼비아 현장 7곳에서 지표·열 offset을 측정한다. 대기후 조건이 과정의 효과를 바꾸므로 지표·지형별 offset은 지역 간에 옮기기 어렵다고 결론짓는다.
- **관련성** 경험 계수가 지역을 넘어 옮겨지지 않는다는 결과(N7)의 영구동토 선례다. NOVELTY에서 요약 도구 경유로 적은 수치는 이 PDF로 대조한다.

#### [PDF] Garibaldi et al., 2026
- **서지** Garibaldi, M. C., Bonnaventure, P. P., Way, R. G. et al. Determining TTOP model parameter importance and overall performance across northern Canada. Cryosphere 20, 2375–2392 (2026).
- **DOI** 10.5194/tc-20-2375-2026 · **OA** gold (CC BY)
- **로컬** `references/12_manuscript_refs/garibaldi2026_ttop_parameter_importance.pdf`
- **요약** 캐나다 북부 330개 지점의 지온·기온 자료로 TTOP 매개변수의 영향을 leave-one-out과 랜덤 포레스트로 평가한다. 결빙기 n-계수와 결빙 도일이 성능을 지배하고 민감도 양상은 지역마다 다르다.
- **관련성** 지역별 매개변수 차이(N7)의 근거 문헌이다.

#### [PDF] Parsekian et al., 2021
- **서지** Parsekian, A. D. et al. Validation of permafrost active layer estimates from airborne SAR observations. Remote Sens. 13, 2876 (2021).
- **DOI** 10.3390/rs13152876 · **OA** gold (CC BY)
- **로컬** `references/12_manuscript_refs/parsekian2021_alt_airborne_sar_validation.pdf`
- **요약** 알래스카 3개 지역에서 항공 SAR 유도 ALT를 보정 GPR 자료로 검증한다. 79 % 지점에서 불확실성 범위 안에서 일치했고, 평균 불확실성은 GPR ALT 0.14 m, SAR ALT 0.19 m다.
- **관련성** 라벨 측정 불확실성(0.14 m)과 잔차 ML 이득의 크기 비교 근거다(NOVELTY 9절).

#### [PDF] Nitze et al., 2021
- **서지** Nitze, I., Heidler, K., Barth, S. & Grosse, G. Developing and testing a deep learning approach for mapping retrogressive thaw slumps. Remote Sens. 13, 4294 (2021).
- **DOI** 10.3390/rs13214294 · **OA** gold (CC BY)
- **로컬** `references/12_manuscript_refs/nitze2021_rts_deep_learning.pdf`
- **요약** PlanetScope, ArcticDEM으로 후퇴성 융해 사태(RTS)를 분할하는 딥러닝을 캐나다·러시아 6개 지역의 지역 교차검증으로 평가한다. 4개 지역은 maxIoU 0.39–0.58이었고 2개 지역에서는 실패했다.
- **관련성** 영구동토 ML의 지역 홀드아웃 선례다. 신규성 문장의 범위를 "ALT 또는 MAGT 매핑"으로 한정하는 근거다.

### 12.3 영구동토 물리-ML 결합 선례

#### [차단] Pilyugina et al., 2025
- **서지** Pilyugina, P. et al. A physics-informed machine learning framework for permafrost stability assessment. IEEE Access 13, 96423–96433 (2025).
- **DOI** 10.1109/ACCESS.2025.3573072 · **OA** gold (CC BY-NC-ND)
- **로컬** 없음. IEEE Xplore가 자동 요청을 막았다. 사전본은 아래 2023 arXiv 판이 로컬에 있다
- **요약** 물리 기반 모형을 ML에 통합해 특징 집합을 늘리는 방식으로 영구동토 융해 속도를 예측한다.
- **관련성** 물리 출력을 입력 특징으로 쓴 선례다. 분류는 [verify] 상태다: 2023 초록은 열 방정식 정칙화로, NOVELTY 본문 판독은 Kudryavtsev 출력의 CatBoost 입력으로 서술한다.

#### [PDF] Pilyugina et al., 2023
- **서지** Pilyugina, P. et al. Assessing the risk of permafrost degradation with physics-informed machine learning. arXiv:2310.02525 (2023).
- **DOI** 10.48550/arXiv.2310.02525 · **OA** green
- **로컬** `references/06_physics_ml/pilyugina2023_pinn_permafrost_risk.pdf` (기존 사본)
- **요약** 열 방정식으로 영구동토 관측과 기후 투영에 학습한 자료 기반 모형을 정칙화한다고 초록에 서술한다.
- **관련성** 위 게재본의 사전본이다. 원고는 게재본을 인용하고 분류는 본문 판독을 따른다.

#### [PDF] Wang G. et al., 2025
- **서지** Wang, G. et al. Simulation of active layer thickness based on multi-source remote sensing data and integrated machine learning models: a case study of the Qinghai-Tibet Plateau. Remote Sens. 17, 2006 (2025).
- **DOI** 10.3390/rs17122006 · **OA** gold (CC BY)
- **로컬** `references/08_recent_alt_2022_2026/wang2025_alt_stefan_catboost_et_qtp_rs.pdf` (동시 수집 사본)
- **요약** 다중 원격탐사 자료, Stefan 식 출력, 실측 ALT로 CatBoost-Extra Trees 블렌딩 모형을 만들어 청장고원 ALT를 추정한다. 10-겹 교차검증 RMSE 32.68 cm, R² 0.873이고 1958–2022년을 역산한다.
- **관련성** 물리 출력을 입력 특징으로 쓴 선례다(원고 서론). 같은 자료의 Stefan 단독 오차는 없다(NOVELTY 3절).

#### [차단] Zhang C. et al., 2024
- **서지** Zhang, C., Douglas, T. A., Brodylo, D., Bosche, L. V. & Jorgenson, M. T. Combining a climate-permafrost model with fine resolution remote sensor products to quantify active-layer thickness at local scales. Environ. Res. Lett. 19, 044030 (2024).
- **DOI** 10.1088/1748-9326/ad31dc · **OA** gold (CC BY)
- **로컬** 없음. IOPscience가 봇 관리 페이지를 반환했다 (https://iopscience.iop.org/article/10.1088/1748-9326/ad31dc/pdf)
- **요약** 기후-영구동토 모형과 ML을 연결해 고해상도 원격탐사 자료로 edaphic 계수를 추정하고 국지 규모 ALT를 추정한다. 내륙 알래스카 실험지 2곳의 2014–2022년 현장 측정에서 ALT 분산의 60 % 이상을 설명했다.
- **관련성** E를 ML로 추정한 선례다(원고 서론). NOVELTY에서 요약 도구 경유로 적은 수치는 원문 대조가 필요하다.

#### [PDF] Gay et al., 2026
- **서지** Gay, B. A., Miner, K. R., Rietze, N. et al. Resolving circumarctic zero-curtain phenomena with AI-integrated earth observations. Sci. Rep. 16, 28715 (2026).
- **DOI** 10.1038/s41598-026-61719-9 · **OA** gold (CC BY-NC-ND)
- **로컬** `references/14_scirep_exemplars/gay2026_zero_curtain_ai_eo.pdf` (기존 사본)
- **요약** 물리 정보 전이 학습 틀(GeoCryoAI)로 현장 측정 6,271만 건과 원격탐사 33억 건을 통합해 환북극 zero-curtain의 계절·경도 비대칭을 분석한다.
- **관련성** 물리 규칙 라벨로 학습한 선례이고(원고 서론), 대상 저널의 그림·서술 형식 참고 대상이다.

#### [PDF] Huang S. et al., 2025
- **서지** Huang, S. et al. A physics-informed machine learning (PIML) framework for projecting 21st-century permafrost extent in Northeast China. EGUsphere, egusphere-2025-4544 (2025).
- **DOI** 10.5194/egusphere-2025-4544 · **OA** 공개 사전본
- **로컬** `references/06_physics_ml/piml2025_ne_china_permafrost_extent.pdf` (기존 사본)
- **요약** TTOP 모형, 토지 이용·피복 변화, CMIP6 투영을 통합한 PIML 틀로 중국 동북부 21세기 영구동토 범위를 투영한다. SSP5-8.5에서 세기말 범위가 90 % 이상 줄어든다.
- **관련성** TTOP 출력을 학습 목표로 쓴 물리 출력 라벨 선례다(NOVELTY 2·6·10절).

#### [링크] Liu Y. et al., 2023
- **서지** Liu, Y., Ran, Y., Li, X., Che, T. & Wu, T. Multisite evaluation of physics-informed deep learning for permafrost prediction in the Qinghai-Tibet Plateau. Cold Reg. Sci. Technol. 216, 104009 (2023).
- **DOI** 10.1016/j.coldregions.2023.104009 · **OA** closed
- **로컬** 구독 필요
- **요약** GIPL2 모의값으로 사전학습한 LSTM을 시추공 지온으로 미세조정하는 물리 정보 LSTM을 청장고원 다지점에서 평가한다(기존 INDEX 기록 기준, 원문 미대조).
- **관련성** 물리 모의값 사전학습 후 관측 미세조정의 영구동토 선례다(NOVELTY 2절). 첫 저자 이름은 기존 INDEX가 Yuandong, OpenAlex·Semantic Scholar가 Yibo로 적는다 [미확인].

#### [PDF] Li J. and Jia, 2026
- **서지** Li, J. & Jia, J. Freezing depth prediction of surrounding rock in seasonally frozen tunnels based on bayes-optimized XGBoost. Sci. Rep. 16 (2026). 논문 번호 [미확인]
- **DOI** 10.1038/s41598-026-49818-z · **OA** gold (CC BY-NC-ND)
- **로컬** `references/12_manuscript_refs/li2026_tunnel_freezing_depth_xgboost.pdf`
- **요약** 계절 동결 터널 주변 암반의 수열 결합 모형과 라틴 초입방 표본 수치 모의로 동결 깊이 자료를 만들고, 베이즈 최적화 XGBoost로 예측한다.
- **관련성** 수치 모의 라벨로 학습한 선례로 NOVELTY 6절 표에 있다. 영구동토 ALT 문헌은 아니다.

#### [차단] Herrington et al., 2026
- **서지** Herrington, T. C., Erler, A. R. & Fletcher, C. G. Application of machine learning to improve reanalysis soil temperatures over the extratropical northern hemisphere. Theor. Appl. Climatol. 157(7) (2026). 논문 번호 [미확인]
- **DOI** 10.1007/s00704-026-06338-0 · **OA** hybrid (CC BY-NC-ND)
- **로컬** 없음. Springer가 봇 확인 페이지를 반환했다 (https://link.springer.com/content/pdf/10.1007/s00704-026-06338-0.pdf)
- **요약** ERA5-Land와 FLDAS 토양 온도의 편향을 평균 편향 제거, 다중 회귀, 랜덤 포레스트로 보정한다. 랜덤 포레스트가 적설기 RMSE를 46–77 % 줄였고, ERA5-Land는 영구동토 지역에서 온난 편향을 보인다.
- **관련성** 재분석 + ML 보정을 단순 보정 기준선과 비교한 선례다(N2 인접). ERA5-Land 편향 기록은 우리 TDD 편향 논의에도 쓸 수 있다.

### 12.4 공변량 자료원과 배경

#### [PDF] Muñoz-Sabater et al., 2021
- **서지** Muñoz-Sabater, J. et al. ERA5-Land: a state-of-the-art global reanalysis dataset for land applications. Earth Syst. Sci. Data 13, 4349–4383 (2021).
- **DOI** 10.5194/essd-13-4349-2021 · **OA** gold (CC BY)
- **로컬** `references/12_manuscript_refs/munozsabater2021_era5_land.pdf`
- **요약** ERA5의 지표 성분을 9 km 해상도로 다시 계산한 ERA5-Land의 구성과 검증을 기술한다. 근지표 상태량에 고도 보정을 적용한다.
- **관련성** TDD, FDD, 토양 온도, 적설 수당량 공변량의 출처다(원고 Methods).

#### [PDF] Poggio et al., 2021
- **서지** Poggio, L. et al. SoilGrids 2.0: producing soil information for the globe with quantified spatial uncertainty. SOIL 7, 217–240 (2021).
- **DOI** 10.5194/soil-7-217-2021 · **OA** gold (CC BY)
- **로컬** `references/12_manuscript_refs/poggio2021_soilgrids2.pdf`
- **요약** 약 24만 지점의 토양 관측과 400개 이상의 공변량으로 250 m 전 지구 토양 물성 지도와 공간 불확실성을 만든다. 고위도 관측 부족을 한계로 든다.
- **관련성** 토양 공변량 9개의 출처다. 고위도 관측 부족은 토양 공변량의 정보 한계 서술에 쓴다.

#### [차단] Pekel et al., 2016
- **서지** Pekel, J.-F., Cottam, A., Gorelick, N. & Belward, A. S. High-resolution mapping of global surface water and its long-term changes. Nature 540, 418–422 (2016).
- **DOI** 10.1038/nature20584 · **OA** closed, GEOMAR OceanRep 수락본 green
- **로컬** 없음. OceanRep 수락본(http://oceanrep.geomar.de/35170/7/Pekel.pdf)이 봇 확인 페이지를 반환했다
- **요약** Landsat 영상 300만 장 이상으로 1984–2015년 30 m 월별 지표수 변화를 정량화한다. 영구 지표수 약 9만 km²가 사라지고 18.4만 km²가 새로 생겼다.
- **관련성** 지도 마스크에 쓴 JRC 지표수 출현 빈도층의 출처다. 원고 표의 [verify]는 이번 조회로 권·쪽·DOI가 확인되었다.

#### [PDF] Biskaborn et al., 2019
- **서지** Biskaborn, B. K. et al. Permafrost is warming at a global scale. Nat. Commun. 10, 264 (2019).
- **DOI** 10.1038/s41467-018-08240-4 · **OA** gold
- **로컬** `references/07_context/biskaborn2019_gtnp_warming.pdf` (기존 사본)
- **요약** GTN-P 시추공 지온 시계열로 2007–2016년 전 지구 영구동토 온도가 0.29 ± 0.12 °C 상승했음을 보인다.
- **관련성** 원고 서론 첫 문단의 배경이다.

### 12.5 관측 자료 논문

#### [링크] Åkerman and Johansson, 2008
- **서지** Åkerman, H. J. & Johansson, M. Thawing permafrost and thicker active layers in sub-arctic Sweden. Permafr. Periglac. Process. 19, 279–292 (2008).
- **DOI** 10.1002/ppp.626 · **OA** closed
- **로컬** 구독 필요
- **요약** 스웨덴 Torneträsk 지역 9개 지점(최대 29년 격자 측정)의 ALT가 연 0.7–1.3 cm 증가했고 최근 10년에 가속했다.
- **관련성** Abisko 하위 지점 CALM 누리집 자료의 출처 논문이다. 재배포 허가가 확인되지 않은 자료로 원고에 기록되어 있다.

#### [PDF] Pohl et al., 2026
- **서지** Pohl, E. et al. Thermo-hydrological river valley observatory in Yedoma permafrost from 2012 through 2022 in Syrdakh, Central Yakutia. Earth Syst. Sci. Data 18, 3525–3557 (2026).
- **DOI** 10.5194/essd-18-3525-2026 · **OA** gold (CC BY)
- **로컬** `references/12_manuscript_refs/pohl2026_syrdakh_observatory_essd.pdf`
- **요약** 중앙 야쿠티아 Syrdakh에서 열카르스트 호수를 잇는 하천 단면에 지중 온도 사슬을 설치해 2012–2022년 열·수문 관측 자료를 제공한다.
- **관련성** Syrdakh 융해 깊이 라벨(Zenodo 19890671)의 자료 논문이다. 원고 표의 서지 제목은 Zenodo 기록의 제목이고, 게재 논문 제목은 위와 같다.

#### [PDF] Talucci et al., 2025
- **서지** Talucci, A. C. et al. Permafrost–wildfire interactions: active layer thickness estimates for paired burned and unburned sites in northern high latitudes. Earth Syst. Sci. Data 17, 2887–2909 (2025).
- **DOI** 10.5194/essd-17-2887-2025 · **OA** gold (CC BY)
- **로컬** `references/08_recent_alt_2022_2026/talucci2025_firealt_paired_burned_essd.pdf` (기존 사본)
- **요약** 기여자 18명의 연소·미연소 짝 지점 융해 깊이 52,466건을 모으고, 수정 Stefan 식으로 계절 말 ALT 48,669건을 추정한 FireALT 자료를 기술한다.
- **관련성** FireALT 라벨의 자료 논문이다. ALT가 Stefan 식으로 외삽한 추정값이라는 점은 라벨 정의 차이로 기록할 사항이다.

#### [PDF] Du E. et al., 2026
- **서지** Du, E., Wu, T., Dai, L. et al. A first 90 m resolution active layer moisture dataset across the Qinghai–Tibet Plateau permafrost region. Earth Syst. Sci. Data Discuss., essd-2026-330 (2026).
- **DOI** 10.5194/essd-2026-330 · **OA** 공개 사전본
- **로컬** `references/08_recent_alt_2022_2026/du2026_qtp_active_layer_moisture_90m_essdd.pdf` (기존 사본)
- **요약** 2009–2024년 현장 표본 342개와 원격탐사·지형·토양 자료로 청장고원 영구동토 지역 90 m 활동층 수분 자료를 만든다. 무작위 5-겹 교차검증 R²는 0.62–0.63이다.
- **관련성** 원고가 쓴 GPR·피트 ALT 조사점(Zenodo 21999366)의 관련 논문이다. 원고 표의 "article DOI [verify]"는 이 DOI로 채울 수 있다. 최종 게재 여부는 [미확인]이다.

---

## 3. 데이터셋·단행본 인용 (PDF 대상 아님)

원고 References 표의 자료 인용이다. 요약은 내려받을 때 기록한 `data/processed/ext_labels/*_meta.json` 의 `name` 필드와 원고 본문을 따랐다. 모두 로컬 PDF가 해당되지 않는다.

| 인용 | DOI 또는 위치 | 상태 | 내용 | 관련성 |
|---|---|---|---|---|
| Åkerman, 1998 | 없음 (CAPS v1.0 CD-ROM, NSIDC. CALM 누리집 https://www2.gwu.edu/~calm/data/north.htm) | 자료, 재배포 허가 미확인 | Abisko S2, Kapp Linné S1 하위 지점 탐침 원자료 | 확인적 풀 밖 라벨 출처 |
| Boike et al., 2024 | 10.1594/PANGAEA.971586 | 공개 자료 | T-MOSAiC 2022 myThaw 표준 규약 융해 깊이 | 추가 지역 라벨 |
| Du et al., 2026 (Zenodo) | 10.5281/zenodo.21999366 | 공개 코드·자료 | 청장고원 활동층 수분 코드 묶음의 GPR·인력 시추 ALT 조사점 | 티베트 라벨. 논문은 12.5 Du E. 2026 |
| Fu, 2025 | 10.6084/m9.figshare.29206613.v1 | 공개 자료 (CC BY 4.0) | 청장고원 지온과 ALT 2001–2020 | 티베트 라벨 |
| Grosse, 2007 | 10.1594/PANGAEA.611409 | 공개 자료 | Cape Mamontov Klyk 주변 활동층 자료 | 러시아 라벨 |
| Hammar et al., 2025 | 10.1594/PANGAEA.974461 | 공개 자료 | 2023 myThaw 융해 깊이 | 추가 지역 라벨 |
| Jorgenson and Kanevskiy, 2025 | 10.18739/A27P8TG0G | 공개 자료 | Alaska Permafrost Soils Inventory and Thermokarst Monitoring Database 2024 Update (CUSP v1.1 경유) | 알래스카 보조 라벨 |
| Kudryavtsev et al., 1974 | 없음 (모스크바대 출판부 단행본, CRREL Draft Translation 606, 1977) | 단행본, 공개본 [미확인] | Kudryavtsev 해석해의 원전 | 원고 표 [verify] 유지 |
| Liang et al., 2023 | 10.1594/PANGAEA.961876 | 공개 자료 | 시베리아 북동부 낙엽송 둔덕 융해 깊이 | 러시아 라벨 |
| Makarieva et al., 2017 | 10.1594/PANGAEA.881754 | 공개 자료 | 콜리마 물수지 관측소 융해 깊이·적설 시계열 1954–1997 | 러시아 라벨 |
| Martin et al., 2023 | 10.1594/PANGAEA.956039 | 공개 자료 | T-MOSAiC 2021 myThaw | 추가 지역 라벨 |
| Moore et al., 2025 | 10.3334/ORNLDAAC/2369 | 공개 자료 | ABoVE 토양 수분·ALT 2005–2024, 버전 2 | 주 지역 라벨(알래스카·캐나다). DataCite 연도 2026 표기 차이 |
| Nixon, 2003 | 10.7265/7m84-k262 | 공개 자료 | NSIDC GGD353 캐나다 활동층 관측(매켄지 계곡 융해관) | 캐나다 라벨 |
| Palmtag et al., 2022 | 10.17043/palmtag-2022-pedon-1 | 공개 자료 | 북부 영구동토 지역 토양 단면 자료 | 토양 보조 자료 |
| Petrone et al., 2016 | 10.1594/PANGAEA.845258 | 공개 자료 | 서그린란드 GPR·지형·식생 기반 퇴적층·ALT 모형 | 그린란드 라벨(CUSP 경유) |
| Pohl et al., 2026 (Zenodo) | 10.5281/zenodo.19890671 | 공개 자료 | Syrdakh 열·수문 관측소 융해 깊이 2012–2018 | 러시아 라벨. 논문은 12.5 |
| Sannel, 2020 | 10.17043/sannel-2020-temperature-1 | 공개 자료 | Tavvavuoma 이탄지 지온, 융해 깊이, 적설 | 스웨덴 라벨 |
| Scheer et al., 2024 | 10.1594/PANGAEA.964306 | 공개 자료 | 일루리사트 ALT 탐침 측정 2020–2021 | 그린란드 라벨 |
| Streletskiy et al., 2025 | 10.1594/PANGAEA.972777 | 공개 자료 | GTN-P CALM 35년 ALT 집계 | 주 라벨 출처. 저자 목록 [verify] 유지 |
| Talucci et al., 2024 | 10.18739/A2RN3092P | 공개 자료 | FireALT 자료 원본(Arctic Data Center) | 논문은 12.5 |
| Veremeeva et al., 2025 | 10.1594/PANGAEA.973813 | 공개 자료 | ALLena 레나 델타 융해 깊이 1998–2022 | 주 지역(레나) 라벨 |
| Walker et al., 2009 | 10.1594/PANGAEA.842711 | 공개 자료 | 야말 측선 융해 깊이 2007–2008 | 러시아 라벨 |
| Westermann et al., 2024 | 10.5285/d34330ce3f604e368c06d76de1987ce5 | 공개 자료 (CEDA) | ESA CCI Permafrost ALT v4.0 | 모형 기반 공변량. 제품 문서는 `references/01_benchmark/esa_cci_permafrost_v4.pdf` |
| Yang and Qiu, 2026 | 10.5281/zenodo.18150789 | 공개 자료 | 티베트 고원 ALT·MAGT 현장 관측 집계표 | 기술 통계용, 분석 미사용 |
| Zastruzny et al., 2024 | 10.1594/PANGAEA.967139 | 공개 자료 | 디스코섬 사면 동결면 깊이 2015 | 그린란드 라벨 |

---

## 4. 공개본이 있으나 자동 내려받기가 막힌 항목

아래 9편은 합법 공개본이 있으나 이 서버의 요청이 출판사·저장소의 봇 차단에 걸렸다. 브라우저에서 받아 해당 폴더에 같은 파일 이름으로 두면 된다.

| 문헌 | 공개 위치 | 저장할 경로 |
|---|---|---|
| Roberts et al., 2017 | https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/ecog.02881 | `11_validation_and_methods/roberts2017_cv_structured_data.pdf` |
| Pool et al., 2019 | https://www.zora.uzh.ch/id/eprint/169290/1/Pool_et_al-2019-Water_Resources_Research.pdf | `11_validation_and_methods/pool2019_limited_discharge_regionalization.pdf` |
| Portes et al., 2026 | https://hal.inrae.fr/hal-05697915/document 또는 https://doi.org/10.1016/j.spasta.2026.101009 | `11_validation_and_methods/portes2026_ml_spatial_transferability.pdf` |
| Stefan, 1891 | https://zenodo.org/record/2042312 | `12_manuscript_refs/stefan1891_eisbildung.pdf` |
| Pilyugina et al., 2025 | https://doi.org/10.1109/ACCESS.2025.3573072 | `12_manuscript_refs/pilyugina2025_piml_permafrost_ieee_access.pdf` |
| Zhang C. et al., 2024 | https://iopscience.iop.org/article/10.1088/1748-9326/ad31dc/pdf | `12_manuscript_refs/zhang2024_climate_permafrost_model_rs_alt.pdf` |
| Pekel et al., 2016 | http://oceanrep.geomar.de/35170/7/Pekel.pdf (수락본) | `12_manuscript_refs/pekel2016_global_surface_water.pdf` |
| Ahajjam et al., 2025 | https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1029/2025JH000969 | `12_manuscript_refs/ahajjam2025_multihorizon_alt_circumarctic.pdf` |
| Herrington et al., 2026 | https://link.springer.com/content/pdf/10.1007/s00704-026-06338-0.pdf | `12_manuscript_refs/herrington2026_ml_reanalysis_soil_temp.pdf` |

구독 필요 13편: Wadoux 2021, Oudin 2008, Steyerberg 2004, Hartung 2001, Holm 1979, Schuirmann 1987, Nelson 1997, Shiklomanov 2002, Riseborough 2008, Åkerman 2008, Li C. 2022, Shen 2023, Liu Y. 2023. 기관 구독으로 받으면 같은 규칙의 파일 이름으로 둔다.

---

## 5. 서지 확인 결과와 원고 반영 사항

이번 조회(OpenAlex, Crossref, 출판사 PDF)로 원고 References 표의 다음 항목을 고칠 수 있다.

- **Stefan, 1891** [verify] 해소: Ann. Phys. 278(2), 269–286, DOI 10.1002/andp.18912780206.
- **Pekel et al., 2016** [verify] 해소: Nature 540, 418–422, DOI 10.1038/nature20584.
- **Du et al., 2026 article DOI** 후보: Du, E. et al., ESSD Discuss., DOI 10.5194/essd-2026-330. 사전본이므로 게재 확정 여부는 [미확인]이다.
- **Holm, 1979** DOI 추가 가능: 10.2307/4615733.
- **Tama et al., 2025** 게재처: WACV 2026 논문집, DOI 10.1109/WACV61042.2026.00528(Crossref). 원고에서 arXiv 대신 게재본을 인용할 수 있다.
- **온라인 연도와 권 연도가 다른 항목**: Roberts 2017(온라인 2016), Valavi 2019(2018), Pool 2019(2018), Peng 2018(2017), Dunn 2023(2022), Liu Z. 2024(2023), Shen 2023(2022), Prokhorenkova 2018(arXiv 2017), Lipman 2023(arXiv 2022), Gorishniy 2025(arXiv 2024). 원고는 권 연도를 쓰고 있으므로 그대로 둔다.
- **Pohl et al., 2026** 게재 논문 제목은 "Thermo-hydrological river valley observatory in Yedoma permafrost from 2012 through 2022 in Syrdakh, Central Yakutia"다. 원고 표의 제목은 Zenodo 자료 제목이다.
- **Westermann et al., 2024** 원고 DOI(d34330ce…)와 기존 INDEX의 CCI DOI(7479606004…)가 다르다. 두 DOI가 각각 어느 CCI 변수(ALT, 지온, 범위)인지 확인이 필요하다 [미확인].
- **Ran 2022 저자** 기존 INDEX의 Ran 2022 ESSD 항목에 적힌 공저자(Ran, Cheng, Dong, Hjort, Lovecraft, Kang, Tan, Li)는 OpenAlex 기준 CEE 논문(Ran et al., 2022b)의 저자 8명과 같다. ESSD 논문 저자는 INDEX 머리말의 정정 목록(Ran, Li, Cheng, Che 등 13명)을 따른다.

---

## 6. 미확인 항목

- **Adjei 2026** (NOVELTY 8절 "환경 매핑의 conformal 커버리지"): 서지를 특정하지 못했다. Crossref 후보 Adjei-Yeboah 외 2026(Smart Construction and Sustainable Cities, 사면 안정 conformal 예측, DOI 10.1007/s44268-026-00094-w)은 주제가 "환경 매핑"과 맞지 않아 채택하지 않았다. 원 조사 기록을 확인한 뒤 인용한다.
- **Peng 2018, Shen 2023** NOVELTY의 저자·연도 표기를 이번 작업에서 각각 J. Clim. 31, 251–266 과 Sci. Total Environ. 859, 160381 로 특정했다. NOVELTY의 분류(E 보정 관행, Stefan 우위)와 초록은 맞지만 원 조사가 같은 논문을 가리켰는지는 원문 대조가 필요하다.
- **Liu Y. 2023 첫 저자 이름** Yuandong 과 Yibo 가 문헌 DB마다 다르다.
- **Li J. and Jia 2026, Herrington 2026 논문 번호** OpenAlex에 없다.
- **Qu 2025, Qu 2026 학회 게재** 둘 다 arXiv만 확인했다.
- **Wadoux 2021 요약** 초록 원문이 아니라 검색 결과 요지에 기반했다.
- **Liu Y. 2023 요약** 기존 INDEX 기록에 기반했고 원문은 대조하지 않았다.
