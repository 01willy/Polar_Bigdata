# 문헌 학습 안내 (읽는 순서)

- **작성일**: 2026-10-04
- **대상**: `references/` 아래 262개 항목(PDF 171편) 가운데 원고·그림·방법·협업 준비에 먼저 필요한 30개 항목(31편)
- **사용법**: 각 줄은 `키`(폴더 번호) 와 읽는 목적이다. 파일 위치, 서지, DOI, 요약은 `references/INDEX.md` 에서 키로 찾는다. 🔗 표시 항목은 로컬 PDF 가 없으므로 브라우저로 받거나(OA, 자동 차단) 기관 구독으로 받는다.
- **순서 원칙**: 각 묶음 안에서 위에서 아래로 읽는다. 원고 묶음을 먼저 읽고, 그림·방법·협업 묶음은 작업 시점에 맞춰 읽는다.

## 1. 원고 (Scientific Reports 투고 준비)

1. `gautam2025_alaska_alt` (01): 대상 저널의 가장 가까운 선행 연구다. RF 와 Stefan 비교, 무작위 70/30 분할, 시험 R² 0.24 대 0.54 를 확인하고 우리 설계와의 차이를 정리한다.
2. `omalley2026_incontext_subsurface_temp` (05): "학습 제외 지역 × 현지 관측 수" 평가의 가장 가까운 선례다. 표 S2(관측 1–40개별 MAE)와 물리 모형 비교 범위를 읽는다.
3. 🔗 `ahajjam2025_multihorizon_alt_circumarctic_jgrmlc` (08): 검증 방식이 지점·지역 홀드아웃인지에 따라 신규성 문장 N1 을 고쳐야 하므로 투고 전 본문을 확인한다.
4. `wei2026_nh_alt_1km_2000_2024_essdd` (08): 경쟁 ALT 제품이다. LOSO RMSE 60.48 cm 와 CALM 학습 표의 중복 가능성을 확인한다.
5. `liu2024_widespread_alt_deepening_2003_2020_erl` (08): 무작위 10-fold R² 0.97 제품이다. Wei 2026 과 대비해 무작위 분할 과대평가 서술에 쓴다.
6. `garibaldi2026_ttop_parameter_importance` (12): 경험 계수가 지점 간에 전이되지 않는다는 직접 선례(N7)다.
7. `read2019_pgdl_lake_temp` (03) 와 `jia2021_pgml_lake_temperature` (11): 원천 물리와 재보정 물리, 두 기준선과 비교하는 보고 형식을 확인한다.
8. `streletskiy2026_calm_long_term_alt_cee` (08): 서론 배경 수치(156지점, 북극 55 % 증가)와 관측망 불균등 서술의 근거다.
9. `jones2024_postfire_permafrost_stabilization` (14): 초록과 결과 문장의 수치 밀도를 따라 쓸 모범이다.
10. `hall2026_rts_expansion_alaska` (14): 누설 방지·외삽 처리 서술과 고찰 끝 한계 문단의 형식을 따른다.

## 2. 그림과 발표 자료

11. `rougier2014_ten_simple_rules_better_figures` (13): 그림 설계 원칙 전체를 먼저 훑는다.
12. `crameri2020_misuse_of_colour` (13): 순차형·발산형 색지도 선택 규칙과 rainbow 금지의 근거다.
13. `ploton2020_spatial_validation` (09): Fig 5 의 "성능 곡선과 영 모형 기준선의 교차"를 라벨 수 곡선(Fig 2) 설계에 옮긴다.
14. `langer2024_cryogridlite_ensemble` (09): Fig 3 의 예측 지도 위 관측 잔차 원과 5–95 % 오차 막대 산점도를 Fig 6 에 옮긴다.
15. `nitzbon2024_no_permafrost_tipping_point_preprint` (09): Fig 3 의 "상자 없는 개념도 위, 결과 아래" 구성을 워크플로 그림(Fig 7)에 옮긴다.
16. `karjalainen2019_circumpolar_geohazard_maps` (09): 자료 종류를 색 대신 모양으로 구분한 지도(Fig 2)와 상자형 흐름도(Fig 1, 반면교사)를 함께 본다.
17. `shen2023_differentiable_modelling_review` (09): Fig 2 함수 공간 도식을 물리 기준선과 잔차 ML 개념 패널(Fig 3)에 쓴다.
18. `tsai2021_parameter_learning_scaling` (09): Fig 7b 의 자료량 곡선과 외부 기준 도달점 표시를 확인한다.
19. `alley2005_sentence_headlines_visual_evidence` (13): 주장 헤드라인과 그림 증거로 구성하는 슬라이드 구조, 글자 크기 기준을 확인한다.
20. `naegle2021_ten_simple_rules_presentation_slides` (13): 1장 1생각, 1장 1분 원칙으로 덱 분량을 정한다.

## 3. 방법 (검증 설계·물리 결합·불확실성)

21. `willard2020_physics_ml_survey` (06): 물리-ML 결합 분류(사전학습, 잔차, 손실 등)로 우리 설계의 위치를 정한다.
22. `feng2023_differentiable_hydrology_ungauged_hess` (08): 지역 홀드아웃(PUR)에서 하이브리드 모형을 평가한 방식과 대상 라벨 0 고정이라는 차이를 확인한다.
23. `willard2021_meta_transfer_unmonitored_lakes` (11): 미관측 대상에서 라벨 수(1–50)에 따른 평가 설계를 확인한다.
24. `meyer2021_aoa` (05): 제외 지역이 원천 공변량 범위 밖인지 진단하는 AOA 를 확인한다.
25. `dunn2023_hierarchical_conformal` (11): 지역을 집단으로 둔 계층 conformal 구간(N9)의 이론 근거다.
26. `gneiting2007_proper_scoring_rules` (11): CRPS, 분위 점수, 구간 점수의 정의를 확인한다.

## 4. KOPRI 협업 준비 (김승희·정윤택)

27. 🔗 `jung2025_phd_thesis_polinsar_boreal_deformation_sejong` (10): 동결기 InSAR 융기와 RVoG 모의 자료로 학습한 딥러닝 역산을 읽고 공동 원고의 방법 용어를 맞춘다(dCollection 웹 열람).
28. 🔗 `jung2023_yakutia_seasonal_deformation_insar_rse` (10): Stefan 해 기반 3구간 선형 모형으로 계절 변위를 해석한 논문이다. 첫 협의 때 저자 원고 공유를 요청한다.
29. `kimhc2025_kopri_report_arctic_cryosphere_remote_sensing` (10): "3-2-4. 동토지역 초분광/라이다 자료 구축" 의 콘슬 무인기 자료(위치 오차 0.02–0.05 m)와 KOPRI-KPDC-00001470 을 확인한다.
30. `chang2024_deformation_alt_nonlinear_npjclim` (08): InSAR 변형과 ALT 의 관계가 지역마다 부호까지 달라진다는 결과로, 변위를 공변량으로 쓸 때의 위험을 협의 전에 정리한다.

## 이어서 읽을 곳

- 최근 ALT 문헌 전체와 신규성 대조표: `references/INDEX.md` 08 절, 원 조사 기록 `references/_index_parts/library_recent_alt.md`
- 그림별 분석과 우리 Fig 1–7 설계 권고: `docs/research/2026-10-04/alt_figure_exemplars.md`
- 그림·슬라이드 규격과 인용문: `docs/research/2026-10-04/figure_slide_standards.md`
- Sci Rep 형식 분석과 문장 틀 45개: `docs/research/2026-10-04/scirep_manuscript_exemplars.md`
- 협업자 신원 판정 근거와 첫 협의 확인 사항: `references/10_kopri_collaborators/README.md` 6–7절
