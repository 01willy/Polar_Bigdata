# Scientific Reports 투고 원고 필수 요소 초안 (Data/Code availability · Declarations · 통계 명세 · Reporting Summary · Gautam 2025 대비 · 용어 정정)

**작성** 2026-09-21. **상태** 초안(계산 없음, 문서·스크립트·원자료 헤더·외부 DOI 페이지 확인 결과만 반영).
**대상** `outputs/report/main.tex`(국문 보고서 692행)를 영문 원고로 재구성할 때 감사(09-18, 79건)가 지적한 누락 요소.
**근거 문서** `docs/PAPER_PLAN_SCIREP.md`, `docs/EXPERIMENT_PLAN_PAPER_2026-09-08.md`, `docs/EXPERIMENT_PLAN_TRANSFER_2026-09-15.md`, `docs/EXPERIMENT_PLAN_MASTER_2026-09-16.md`(개정 09-21 포함), `docs/RESULTS_RECONCILIATION_2026-09-14.md` §5·§6, `docs/EXPERIMENT_LOG.md` 2026-09-18/21 항목, `data/processed/fidelity_base_v3_meta.json`, `data/processed/m1/label_year_sensitivity{.csv,_meta.json}`, `data/processed/m1/m1_gate_gate.csv`, `references/01_benchmark/gautam2025_alaska_alt.pdf`.
**표기 규칙** 영문 선언문은 제출용 정확 문구, 국문은 보고서 톤 병기. 확인하지 못한 식별자는 **[확인 필요]**, 저자·자금 등은 **[자리표시자]**로 표시한다.

---

## 0. 요약: 이 문서가 채우는 빈칸

| 감사 지적 | 본 문서 절 | 상태 |
|---|---|---|
| Data availability 성명(원자료 DOI·버전·접근일·라이선스) | §1 | 8건 중 7건 식별자 확정, KPDC 3종·GTN-P 라이선스·SAR 2건 [확인 필요] |
| Code availability(영구 식별자·환경 고정) | §2 | GitHub 공개 전환·Zenodo DOI 예정, `environment-lock.txt` 생성 완료(517행) |
| CRediT·이해상충·자금·감사·윤리 | §3 | 영문 정확 문구, 저자명·자금 [자리표시자] |
| 통계 검정 명세표 | §4 | H1–H17, 사전 등록 커밋 이력 정직 기록(09-21 개정은 미커밋) |
| Reporting Summary(제외·무작위화·반복·시공간 규모) | §5 | 초안 |
| Gautam et al. 2025 대비·선행 공개 고지 | §6 | 서론 문단·SI 표·커버레터 문장 |
| 용어 정정 목록 | §7 | RESULTS_RECONCILIATION §5·§6 이관, main.tex 행 번호 부여 |

---

## 1. Data availability

### 1.1 원자료 식별자 표

접근일은 `data/raw/` 파일 시각(mtime)으로 추정한 값이며, 원고에는 월 단위로 적어도 된다. 라이선스 문구는 원자료 헤더 또는 배포 페이지에서 확인한 것만 적었다.

| # | 자료 | 용도 | 식별자·인용 | 버전·기간 | 접근일(추정) | 라이선스 | 확인 상태 |
|---|---|---|---|---|---|---|---|
| 1 | GTN-P/CALM ALT 편집본 | 알래스카 학습 라벨 일부, 전이 대상 지역(러시아 W/E/C·그린란드·캐나다 동부·심부 5지역) | Streletskiy, D. A.; CALM; GTN-P; Wieczorek, M.; Heim, B.; Bartsch, A. (2025). GTN-P CALM: 35 years of Active Layer Thickness (ALT) across latitudinal and elevational gradients in the Northern Hemisphere. PANGAEA. https://doi.org/10.1594/PANGAEA.972777 | 263 시계열, 1990–2024 | 2026-06-29 | CC-BY 4.0(헤더 명시) | 확정(파일 헤더) |
| 2 | ALLena 융해 깊이 편집본(레나델타) | 전이 대상 레나델타 3,037셀 | Veremeeva, A.; Morgenstern, A.; Gottschalk, M.; … Grosse, G. (2025). ALLena: Thaw depth measurements of the active layer in the Lena River Delta region from 1998 to 2022, Northeastern Siberia. PANGAEA. 주표 https://doi.org/10.1594/PANGAEA.973813, 묶음 https://doi.org/10.1594/PANGAEA.974408 | 1998–2022 | 2026-07-13 | CC-BY 4.0(헤더 명시) | 확정(파일 헤더) |
| 3 | ABoVE 토양수분·활동층 두께(알래스카·캐나다) | 알래스카 학습 라벨 주원(GPR 유도 86%), 캐나다 서부 셀 | Moore, M. A.; Schaefer, K.; Clayton, L. K.; … (2025). ABoVE: Soil Moisture and Active Layer Thickness in Alaska, USA and Canada, 2005-2024, Version 2. ORNL DAAC. https://doi.org/10.3334/ORNLDAAC/2369 | v2, 2005-01-10 – 2024-08-25; 파일 `ABoVE_Soil_ThawDepth_Moisture_Validation_V2.csv` | 2026-07-01 | NASA EOSDIS 공개 자료(제한 없음, 인용 의무) | 확정(웹, 2026-09-21) |
| 4 | GTN-P 시추공 지중온도(알래스카) | 지온 유도 ALT(보충자료), 표층 3차원 온도장 라벨 | GTN-P Database API https://data.gtn-p.org/api (`policy: Open` 데이터셋만 취득); 사이트 37지점; 데이터베이스 인용 Biskaborn et al. 2019 doi:10.1038/s41467-018-08240-4는 맥락 인용 | 취득 시점 스냅샷(매니페스트 `data/raw/gtnp/*_manifest.json`) | 2026-06-24 – 06-29 | **[확인 필요]** GTN-P 데이터 정책 문구·권장 인용 형식 | 부분 확인 |
| 5 | KPDC 콘슬(수어드반도) 현장 자료 3종 | 현장 검증(보충자료) | 코어 융해 깊이·토양: **KOPRI-KPDC-00002125**; 일별 지중온도 프로파일 2024: **KOPRI-KPDC-00002707**; 2025: **KOPRI-KPDC-00002955**; 자동기상관측(AWS, 2019·2021 등)은 위 식별자 중 어디에 귀속되는지 **[확인 필요]** | 2021–2025 | 2026-07-21(로컬 정리일) | **[확인 필요]** KPDC 자료 이용 정책(Disclosure Request 승인 조건, 재배포 가능 범위) | 식별자는 09-08 계획·08-26 검증 기록 기준, 데이터셋 제목·라이선스 미확인 |
| 6 | ERA5-Land 월별 평균 | 기후 공변량 8종(2015–2020 평년), 연별 지도(2010–2024) | Muñoz-Sabater, J. et al. (2021). ERA5-Land: a state-of-the-art global reanalysis dataset for land applications. ESSD 13, 4349–4383. 데이터셋 DOI https://doi.org/10.24381/cds.68d2bb30 | 변수 t2m·sd·stl1, 2015–2020(평년) 및 2010–2024(연별), 25–84°N 전 경도, 0.1° | 2026-07-01(2015–2020), 2026-07-08(2010–2024) | Copernicus 라이선스(CC-BY 계열, CDS 페이지 표기) | 확정(웹) |
| 7 | ESA CCI Permafrost ALT v4.0 | 위성 기반 제품(준독립) 공변량·앵커 | Westermann, S.; Barboux, C.; Bartsch, A.; … Wiesmann, A. (2024). ESA Permafrost Climate Change Initiative (Permafrost_cci): Permafrost active layer thickness for the Northern Hemisphere, v4.0. NERC EDS CEDA. https://doi.org/10.5285/d34330ce3f604e368c06d76de1987ce5 | v4.0, 1997–2021 연별 25파일, 1 km, 다년 평균 사용 | 2026-07-02 | ESA CCI Permafrost 이용 약관(등록·비등록 이용 가능, 인용 의무) | 확정(웹, 2026-09-21). `references/INDEX.md`의 10.5285/7479606004… 는 컬렉션 레코드이므로 원고에는 위 ALT 데이터셋 DOI를 쓴다 |
| 8 | SoilGrids 2.0 | 토양 공변량 9종(clay·sand·silt·soc·bdod·cfvo·phh2o, 5–15 cm; soc 0–5·15–30 cm) | Poggio, L.; de Sousa, L. M.; Batjes, N. H.; Heuvelink, G. B. M.; Kempen, B.; Ribeiro, E.; Rossiter, D. (2021). SoilGrids 2.0: producing soil information for the globe with quantified spatial uncertainty. SOIL 7, 217–240. https://doi.org/10.5194/soil-7-217-2021. 취득 경로 ISRIC WCS 2.0.1 (`https://maps.isric.org/mapserv`) 및 WebDAV VRT | 250 m, "latest" 릴리스(취득 시점 버전 태그 **[확인 필요]**) | 2026-07-20, 2026-09-08 | CC-BY 4.0 | 논문 DOI 확정(웹), 데이터 버전 태그 미확인 |
| 9 | Copernicus DEM GLO-30 | 지형 공변량 6종 | Copernicus DEM, Global and European Digital Elevation Model. https://doi.org/10.5270/ESA-c5d3d65 (AWS 공개 버킷 `copernicus-dem-30m`) | 30 m, 177타일 | 2026-07-14, 2026-09-08 | 무료 라이선스, 귀속 문구 필수: "© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved" | 확정(웹) |
| 10 | (알래스카 한정) ABoVE ReSALT InSAR ALT·침하 | SAR 공변량 5종(지역 내 34종 입력에만 사용) | ORNL DAAC https://doi.org/10.3334/ORNLDAAC/2004 (`scripts/0_download/resalt_insar.py`) | 2017, 30 m, 52 granule | 2026-07-01 | NASA EOSDIS 공개 자료 **[확인 필요]** 페이지 확인 | 부분 확인 |
| 11 | (알래스카 한정) ABoVE upscaled ALT 2014–2017(PolSAR 유도) 및 ds2332 | PolSAR 공변량 3종 | Whitcomb, J. et al. ABoVE: Upscaled Active Layer Thickness in Northern Alaska, 2014–2017. ORNL DAAC. **DOI·데이터셋 번호 [확인 필요]**; `data/raw/ornl_ds2332/`(PDO ReSALT VWC) DOI 10.3334/ORNLDAAC/2332 **[확인 필요]** | 2014·2015·2017 | 2026-07-03, 07-13 | **[확인 필요]** | 미확인 |

원고 본문에서 자료 1–3·6–9는 필수, 4·5는 보충자료 사용분, 10·11은 지역 내 34종 입력을 보고할 때만 필요하다. WorldClim 2.1은 초기 단계에서만 쓰였고 최종 결과에는 들어가지 않으므로 성명에서 제외한다.

### 1.2 파생 자료 기탁 계획(Zenodo)

| 파생 자료 | 내용 | 규모 | 기탁 형식 |
|---|---|---|---|
| `data/processed/fidelity_base_v3.csv` + `_meta.json` | 셀 단위 통합 스키마(고유 좌표, 다년 평균 ALT, 공변량 34종, 지역·등급·관측법 태그) | 17,572행 × 45열 | CSV + 메타 JSON(git 커밋 해시 포함) |
| `data/processed/e5_soil_tdd_v3.csv` | 토양 도일 공변량(0 셀 결측 처리) | 17,572행 | CSV |
| M1 결과 데이터베이스 `data/processed/m1/m1_*_shard*.csv`, `*_meta.json`, `*_preds.npz` | (조건, 지역, fold, 분할, seed, 구성)별 RMSE·bias와 셀별 예측 벡터 | 축별 shard 12개 이상 | 장형 CSV + npz(npz는 git 미추적이므로 Zenodo에만) |
| 게이트·검정 CSV `m1_*_gate.csv`, `m1_*_nested.csv`, `m1_*_tests.csv`, `label_year_sensitivity*.csv`, `cv_scheme_comparison*.csv` | 재현 게이트, 중첩 선택, 가설 채점, 라벨 연도 민감도 | 소형 | CSV |
| 옛 단계 결과 `e1_*`, `e2_*`, `e4_*`, `s11_conformal_*`, `s12_hybrid_*` | 보충자료 표의 근거 | 소형 | CSV |

주의: 셀 스키마에는 원자료 1–3의 좌표·ALT 값이 재배포되므로 CC-BY 4.0 귀속 문구를 메타에 넣는다. KPDC 파생 값(콘슬 유도 ALT 등)은 KPDC 정책 확인 전까지 요약 통계만 기탁한다 **[확인 필요]**.

### 1.3 영문 성명 초안

> **Data availability.** All observational and covariate data used in this study are available from their original public archives. Active-layer thickness (ALT) observations were obtained from the GTN-P/CALM compilation (Streletskiy et al., 2025; PANGAEA, https://doi.org/10.1594/PANGAEA.972777; CC-BY 4.0; accessed June 2026), the ALLena thaw-depth compilation for the Lena River Delta (Veremeeva et al., 2025; PANGAEA, https://doi.org/10.1594/PANGAEA.973813 and https://doi.org/10.1594/PANGAEA.974408; CC-BY 4.0; accessed July 2026), and the ABoVE soil moisture and active layer thickness dataset for Alaska and Canada, 2005–2024, version 2 (Moore et al., 2025; ORNL DAAC, https://doi.org/10.3334/ORNLDAAC/2369; accessed July 2026). Borehole ground-temperature profiles used in the Supplementary Information were retrieved from the GTN-P database (https://data.gtn-p.org; open-access datasets only; accessed June 2026). Field observations at the Council site (Seward Peninsula, Alaska) were provided by the Korea Polar Data Center (KPDC) of the Korea Polar Research Institute under datasets KOPRI-KPDC-00002125 (soil cores), KOPRI-KPDC-00002707 (daily soil-temperature profiles, 2024) and KOPRI-KPDC-00002955 (daily soil-temperature profiles, 2025); these are available through the KPDC data request procedure (https://kpdc.kopri.re.kr) [확인 필요: KPDC 정책 문구]. Climate covariates were derived from ERA5-Land monthly averaged data (Muñoz-Sabater et al., 2021; Copernicus Climate Data Store, https://doi.org/10.24381/cds.68d2bb30; 2015–2020 climatology and 2010–2024 annual fields). The satellite-based ALT product is ESA CCI Permafrost active layer thickness v4.0 (Westermann et al., 2024; CEDA, https://doi.org/10.5285/d34330ce3f604e368c06d76de1987ce5; 1997–2021). Soil covariates were taken from SoilGrids 2.0 (Poggio et al., 2021; https://doi.org/10.5194/soil-7-217-2021; ISRIC WCS service; CC-BY 4.0) and terrain covariates from the Copernicus DEM GLO-30 (https://doi.org/10.5270/ESA-c5d3d65; © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under COPERNICUS by the European Union and ESA). SAR-derived covariates used only in the within-Alaska analysis were taken from ABoVE ReSALT (ORNL DAAC, https://doi.org/10.3334/ORNLDAAC/2004) and the ABoVE upscaled ALT product [확인 필요]. The assembled cell-level dataset (17,572 cells), the complete factorial results database with per-cell predictions, and all reproduction-gate and hypothesis-test tables will be deposited in Zenodo (DOI to be assigned) upon acceptance; values derived from KPDC data are redistributed only as aggregated statistics in accordance with the KPDC data policy.

### 1.4 국문 요지

학습·검증에 쓴 관측 라벨(CALM·ALLena·ABoVE), 지중온도(GTN-P), 현장 자료(KPDC 3종), 공변량(ERA5-Land·CCI ALT v4.0·SoilGrids 2.0·Copernicus DEM GLO-30)의 식별자·버전·접근 시기·라이선스를 위 표대로 기재한다. 파생 자료(셀 스키마 v3, M1 결과 DB, 게이트·검정 CSV)는 게재 확정 시 Zenodo에 기탁하고 DOI를 성명에 삽입한다.

---

## 2. Code availability

### 2.1 공개 범위

저장소 `git@github.com:01willy/Polar_Bigdata`(현재 비공개, 투고 시 공개 전환)에서 아래 경로를 공개 범위로 한다. `submission/`(예선 동결본)은 저장소 공개본에서 제외한다(`docs/PAPER_PLAN_SCIREP.md` §5).

| 경로 | 파일 | 역할 |
|---|---|---|
| `src/polar/` | `acquire.py, alt_dataset.py, alt_model.py, config.py, covariates.py, eval_metrics.py, fidelity.py, geomap.py, geo.py, gridding.py, interpolate.py, m1_core.py, m1_ext.py, model.py, outputs.py, physics.py, plotstyle.py, preprocessing.py, preprocess.py, source_aware.py, tab_models.py, visualize.py, viz3d.py, viz_suite.py` | 라이브러리(공변량 집합 정본 `fidelity.py`, 물리식 `physics.py`, 하네스 코어 `m1_core.py`, 모델 `tab_models.py`) |
| `scripts/0_download/` | `copernicus_dem.py, era5land_monthly.py, era5land_monthly_multiyear.py, era5land_test.py, gridded_covariates.py, gtnp_boreholes.py, resalt_insar.py, soilgrids_alaska.py, soilgrids_multiregion.py` | 원자료 취득 |
| `scripts/1_data_prep/` | `aggregate_alt_cell.py, assemble_cell_v2.py, assemble_dl_dataset.py, assemble_pretrain_dataset.py, build_fidelity_base_v3.py, build_fidelity_schema.py, derive_alt_gtnp_envelope.py, enrich_cci_cell.py, enrich_covariates.py, enrich_new_regions.py, enrich_soilgrids_cell.py, enrich_soilgrids_wcs.py, era5land_covariates.py, era5land_soil_tdd.py, era5land_temporal_covariates.py, expand_calm_regions.py, extract_resalt_weaklabels.py, gtnp_alt_inventory.py, parse_above_alt.py, parse_allena.py, parse_ground_temp.py, parse_gtnp_global.py, parse_kpdc_met.py, parse_qtec.py, preprocess_boreholes.py, s7_parse_kpdc_council.py, terrain_features_dem.py` | 파싱·QC·공변량 부착·셀 스키마 조립 |
| `scripts/2_evaluation/` | `l2_common.py, a2_common.py`(공용 계층), `aoa_conformal_alt.py, aoa_transfer.py, s11_comparison_tables.py, kpdc_era5_validation.py, rescore_results.py, rescore_worldclim_vs_era5.py, restructure_gate.py, map_model_gate.py, temporal_gate.py, areal_evaluation.py, analyze_error_structure.py, diagnose_apparent_floor.py, grid_support_experiment.py, overnight_cell_experiments.py` | 평가·보조 분석(감사 지적의 `l2_*`·`a2_*`는 현재 공용 모듈 2개만 존재하며 개별 `l2_*.py`·`a2_*.py` 분석 스크립트는 미작성 **[확인 필요]**) |
| `scripts/3_deep_learning/` | `e1_unified_factorial.py, e1_analysis.py, e1_kriging_baseline.py, e2_physics_ladder.py, e2_seasonal_dt.py, e3_adaptive_E.py, e4_nested_selection.py, e4_interval_score.py, e4_label_sensitivity.py, m1_master_factorial.py, m1_analysis.py, m1_cv_scheme_comparison.py, m1_label_year_sensitivity.py, m1_run_sc.sh, m1_run_queue.sh, s11_conformal_uq.py, s12_hybrid_transfer.py` | 실험 하네스·분석 |
| `configs/` | `m1_gate_configs.json, m1_gate_canada_evalall.json` | 재현 게이트 구성(현재 git 미추적, 커밋 필요) |
| `tests/` | `test_leakage.py`(18), `test_fidelity_v3.py`(7) | 누설 통제·스키마 무결성 pytest 25건 |
| 문서 | `docs/EXPERIMENT_PLAN_PAPER_2026-09-08.md, docs/EXPERIMENT_PLAN_TRANSFER_2026-09-15.md, docs/EXPERIMENT_PLAN_MASTER_2026-09-16.md, docs/RESULTS_RECONCILIATION_2026-09-14.md, docs/EXPERIMENT_LOG.md` | 사전 등록·개정 이력·로그 |
| 루트 | `environment-lock.txt`, `requirements.txt` | 환경 고정 |

### 2.2 환경 고정

- 실행: `/home/anaconda3/bin/python -m pip freeze > environment-lock.txt` (2026-09-21, 517행). 실험 스크립트가 호출하는 `python3`도 같은 인터프리터(`/home/anaconda3/bin/python3`)이다.
- 주의: 517행 중 310행이 conda 설치분(`name @ file:///croot/...`)이라 그대로는 `pip install -r`로 복원되지 않는다. 공개 전 `pip list --format=freeze`(이름==버전 형식) 또는 `conda env export --no-builds > environment.yml`로 보완본을 만들고, 원고에는 아래 버전 표를 인용한다.
- GPU: NVIDIA GeForce RTX 3090, 드라이버 530.41.03, CUDA 12.4(torch 빌드 기준).

| 패키지 | 버전 | 근거 |
|---|---|---|
| Python | 3.9.18 | `python --version` |
| numpy | 1.26.4 | environment-lock.txt |
| pandas | 2.1.4 | environment-lock.txt(wheel 파일명)·import 확인 |
| scikit-learn | 1.3.0 | import 확인(lock에는 경로만) |
| SciPy | 1.11.4 | environment-lock.txt·import 확인 |
| CatBoost | 1.2.10 | environment-lock.txt |
| LightGBM | 4.6.0 | environment-lock.txt |
| XGBoost | 2.1.4 | environment-lock.txt |
| PyTorch | 2.6.0+cu124 | environment-lock.txt·import 확인 |
| pytabkit(RealMLP) | 1.7.3 | environment-lock.txt |
| tabpfn | 8.0.7(설치됨, 라이선스 토큰 부재로 실험 제외) | environment-lock.txt·EXPERIMENT_LOG 09-18/21 |
| matplotlib / xarray / netCDF4 / rasterio / cdsapi / earthaccess | 3.8.0 / 2023.6.0 / 1.7.2 / 1.4.3 / 0.7.7 / 0.11.0 | import·lock |

### 2.3 영문 성명 초안

> **Code availability.** All code for data acquisition, preprocessing, model training, evaluation and figure generation is available at https://github.com/01willy/Polar_Bigdata (public from submission) and is archived at Zenodo (https://doi.org/10.5281/zenodo.XXXXXXX [DOI to be assigned]) under the release tag corresponding to the analyses reported here. The repository contains the analysis library (`src/polar/`), acquisition and preprocessing scripts (`scripts/0_download/`, `scripts/1_data_prep/`), the pre-registered experiment harness and analysis scripts (`scripts/3_deep_learning/`: `e1_*`, `e2_*`, `e3_*`, `e4_*`, `m1_*`, `s11_*`, `s12_*`), evaluation utilities (`scripts/2_evaluation/`), unit tests for leakage control and dataset integrity (`tests/`, 25 tests), the pre-registration documents with their revision histories (`docs/EXPERIMENT_PLAN_*.md`), and a frozen environment specification (`environment-lock.txt`: Python 3.9.18, NumPy 1.26.4, pandas 2.1.4, scikit-learn 1.3.0, SciPy 1.11.4, CatBoost 1.2.10, LightGBM 4.6.0, XGBoost 2.1.4, PyTorch 2.6.0 with CUDA 12.4, pytabkit 1.7.3). Each results file records the git commit hash of the code that produced it.

### 2.4 국문 요지

코드는 GitHub 공개 전환과 Zenodo 아카이브(DOI 배정 예정)로 영구 식별자를 확보한다. 공개 범위는 §2.1 표, 환경은 `environment-lock.txt`와 §2.2 버전 표로 고정한다. 결과 파일 메타(`*_meta.json`)에 git 커밋 해시가 기록되어 있음을 명시한다.

---

## 3. Declarations (영문 정확 문구)

> **Acknowledgements.** We thank the Korea Polar Research Institute (KOPRI) and the Korea Polar Data Center (KPDC) for providing the Council site observations (KOPRI-KPDC-00002125, KOPRI-KPDC-00002707 and KOPRI-KPDC-00002955). Part of this work was initiated as an entry to the [2026 Polar Big Data and Artificial Intelligence Contest organised by KOPRI/KPDC; 자리표시자, 공식 영문 명칭 확인]. We acknowledge the CALM/GTN-P, ALLena, ABoVE, ESA CCI Permafrost, Copernicus (ECMWF ERA5-Land and Copernicus DEM) and ISRIC SoilGrids teams for making their data openly available. Terrain covariates were produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.

> **Author contributions.** [Author A]: Conceptualization, Methodology, Software, Formal analysis, Investigation, Data curation, Validation, Visualization, Writing (original draft), Writing (review and editing). [Author B]: Conceptualization, Supervision, Resources, Funding acquisition, Writing (review and editing). [Author C]: Data curation (KPDC field data), Validation, Writing (review and editing). All authors reviewed and approved the final manuscript. [자리표시자: CRediT 14개 역할(Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Resources, Data curation, Writing (original draft), Writing (review and editing), Visualization, Supervision, Project administration, Funding acquisition) 중 해당 항목만 저자별로 배정]

> **Funding.** This work was supported by [funding agency, grant number; 자리표시자]. / The authors received no specific funding for this work. [둘 중 하나 선택]

> **Competing interests.** The authors declare no competing interests.

> **Ethics declarations.** This study used only publicly available or institutionally archived observational datasets; no human participants or animals were involved. Ethics approval and informed consent were therefore not required.

> **Pre-registration statement (Methods 절 삽입용).** The analysis plan, primary estimands and decision rules were documented before the corresponding results were inspected (`docs/EXPERIMENT_PLAN_PAPER_2026-09-08.md`, `docs/EXPERIMENT_PLAN_TRANSFER_2026-09-15.md`, `docs/EXPERIMENT_PLAN_MASTER_2026-09-16.md` including the revision of 21 September 2026). The documents were committed to the repository at commits ff6504a (15 September 2026), 2b6d969 and 291e2c7 (16 September 2026) and [해시 미정, 09-21 개정 커밋 필요]; the first commit of the 8 September plan post-dates the E1–E4 results, so the git history does not itself certify the pre-registration of H1–H6, and all deviations are listed in the revision histories and in Supplementary Table S-x.

> **Use of AI-assisted tools (필요 시 Methods 또는 Acknowledgements).** [자리표시자] Large language model based coding assistants were used for code drafting and text editing; all analyses, results and conclusions were verified by the authors. (Springer Nature 정책상 LLM 사용은 방법 절에 기재, 저자 등재 불가. 대회 규정의 생성형 AI 사용 내역 기재와 정합.)

국문 요지: 감사는 KOPRI·KPDC 자료 제공과 대회 주관(자리표시자), 자료 제공 기관, Copernicus DEM 귀속 문구를 포함한다. 저자 기여는 CRediT 용어로 저자별 배정, 자금은 자리표시자, 이해상충 없음, 윤리 해당 없음(공개·기관 보관 관측 자료만 사용). 사전 등록 성명은 git 이력의 한계(09-08 계획의 최초 커밋이 결과 이후)를 숨기지 않는다.

---

## 4. 통계 검정 명세표 (SI Table S-x 초안)

### 4.1 공통 규약(개정 09-21 A·B·C 반영)

| 항목 | 규약 |
|---|---|
| 주 채점 | 셀 가중 RMSE(cm). 보조 채점 (a) 블록별 RMSE 후 블록 등가중 평균, (b) 블록당 셀 수 상한 100 가중 RMSE. 모든 지역 표에 세 값과 블록 다수결(k/n) 병기, 부호가 갈리면 "집계 의존" |
| 표본 단위 | 셀 = 고유 좌표(다년 평균). 재표집 단위 = 0.5° 블록. 지역 수준 추론 단위 = 지역 |
| 짝지음 | 같은 평가 셀에서 방법 A·B의 seed별 Δ(ΔRMSE = RMSE_A − RMSE_B)를 계산해 seed 평균. seed 앙상블 Δ는 보조 열 |
| 셀·블록 수준 CI | 짝지은 블록 부트스트랩 1,000회(seed 평균 Δ를 통계량), 백분위 95% CI. 블록 8개 미만 지역은 CI 미산출·플래그(러시아 C·그린란드). MDE(80% 검정력 근사) = 2.8 × 부트스트랩 SD |
| 지역 수준 주 추정치 | 지역 층화 블록 부트스트랩(주 집합 각 지역의 블록을 동시에 재표집) 1,000회로 얻는 지역 비가중 평균 Δ의 95% CI |
| 보조 검정 | 블록 단위 층화 순열검정(지역 내 부호 뒤집기), Wilcoxon 부호순위. 지역 부호검정(n=6, 만장일치만 p=0.031)은 검정력 부족(관측 효과 −1.25 cm에서 약 13%)으로 기술 통계로 강등 |
| 조건 | (i) 공변량만: 셀 수가 충분한 4지역(레나·캐나다·러시아 W·러시아 E), A/B 블록 분할 3회 반복(split seed 0–2)의 분할 반복 CI. (ii) 정보 없음·배포형: 주 6지역(레나·캐나다·러시아 W/E/C·그린란드), 층화 부트스트랩 CI |
| 다중 비교 | 주 가설 = H12 하나(무보정). 가족별 Holm 보정: 전이 가족(H1·H7–H11·H14), 지역 내 가족(H6·H13), UQ 가족(H15). 축 스크린(X-*)은 탐색적 표로 분리 |
| 선택 | 구성 선택은 중첩만(주 지역 중 한 지역을 빼고 선택 → 남은 지역 채점). in-sample 최소는 보조 열 |
| 알래스카 fold 민감도 | 6-fold 공간블록의 fold 0이 단일 블록 4,716셀(35%)이므로 fold별 값·대블록 제외 값 병기, 무작위 블록 fold 배정 10회 반복 민감도를 보충자료에 |

### 4.2 지역별 표본 수(v3, F4_direct 탐침 라벨, 평가 셀 = y·CCI·토양 도일 유효)

| 지역 | 셀 | 평가 셀 | 0.5° 블록 | 사용 조건 | CI |
|---|---|---|---|---|---|
| 알래스카 | 13,606 | 13,606 | 74 | 라벨 있음 6-fold | 산출 |
| 레나델타 | 3,037 | 2,958 | 20 | (i)(ii) | 산출 |
| 캐나다 | 750 | 747 | 39 | (i)(ii) | 산출 |
| 러시아 W | 31 | 28 | 21 | (i)(ii) | 산출 |
| 러시아 E | 30 | 27 | 21 | (i)(ii) | 산출 |
| 러시아 C | 7 | 7 | 6 | (ii) | 미산출·플래그 |
| 그린란드 | 3 | 1 | 2 | (ii) | 미산출·플래그 |

공변량만 조건의 채점 셀은 B블록(약 절반, 예: 캐나다 371셀)이며 분할 3회의 평균과 분할 간 SD를 함께 적는다. 심부 집합(스발바르·몽골·알프스·티베트·스칸디나비아)은 확인적 검정에서 제외하고 기술 통계만 보고한다.

### 4.3 가설별 명세

| ID | 비교 A − B | 조건·입력 | 표본 단위 / n | 짝지음 | 검정·CI | 가족(보정) | 사전 등록 문서 | 비고 |
|---|---|---|---|---|---|---|---|---|
| H1 | Stefan+CCI 앵커(무보정, 등가중 0.5) − Stefan 앵커, λ=0 | (ii) 정보 없음·배포형, 25종 | 블록 / 6지역(§4.2) | seed 없음(해석적 앵커) | 짝지은 블록 부트스트랩 + 지역 층화 부트스트랩; 보조 순열·Wilcoxon | 전이(Holm) | 09-08 §7 | 이탈: 가중은 "알래스카 최소제곱" 대신 고정 0.5(최소제곱 가중이 1.0으로 클리핑되어 Stefan과 동치). 09-08 결과: 방향 5/6, 셀 가중 평균 −1.25, 부호검정 p=0.22(비유의) |
| H2 | 앵커 + ridge 잔차(λ=0.25) − 앵커 단독 | (i)(ii), 25종 | 블록 / 6지역 | seed별 Δ 후 평균(3 seed) | 동일 | 전이(Holm) | 09-08 §7 | 09-08 결과: 주 6/6 악화(p=0.031), 외삽 폭주(그린란드 +650). 기각 |
| H3 | Stefan 유사라벨(r=10) − {상수, 셔플, TDD 선형, 대상 수준 상수} 유사라벨 | (i) 공변량만 4지역, catboost_lo, 25종 | 블록 / 4지역, 분할 3회 | seed별 Δ 후 평균(개정 09-21). 09-08 실행은 seed 앙상블 후 짝지음(이탈 기록) | 짝지은 블록 부트스트랩, 분할 반복 CI | 증강(제안: H16과 묶어 Holm) | 09-08 §7 | 셔플 대조는 복제 행 순열이라 대상 수준 상수와 동치(셀 단위 셔플은 S-D 추가). 대조군 사다리 분해 결과는 §7 참조 |
| H4 | 같은 보정 자유도 k에서 Stefan − 대안 물리식(멱법칙·TDD 선형·MAAT 선형·edaphic·Kudryavtsev·토양 도일 Stefan·2층·눈 보정) | 지역 내 6-fold, LORO | 블록 / 알래스카 74, 전이 지역별 | 계수는 학습 fold·학습 지역에서만 적합 | RMSE·bias 표 + 짝지은 블록 부트스트랩 | 기술적(가족 미배정) | 09-08 §7 | 09-08 결과: 3지역 기준 k=1 Stefan 최저·k=2 동급; 격차 원인은 배율 편향 c≈0.53–0.71 |
| H5 | 멱지수 b(a·TDD^b)의 fold 분포와 0.5 포함 여부 | 지역 내 6-fold | fold / 6 | 해당 없음 | fold 분포 기술, 부트스트랩 CI | 기술적 | 09-08 §7 | 09-08 결과: b 0.35–0.42(0.5 미포함) |
| H6 | Stefan+CCI 앵커 − Stefan 앵커(지역 내), λ=0 및 λ=0.25 ridge | 라벨 있음, 알래스카 6-fold | 블록 / 13,606셀·74블록 | fold·seed별 Δ | 짝지은 블록 부트스트랩 | 지역 내(Holm, H13과) | 09-08 §7 | 09-08 결과 +2.02 [0.47, 3.38] 악화는 무보정 등가중 구성 한정, 보정 CCI 결합은 −0.24(악화 없음) |
| H7 | T2 게이팅 결합 앵커(중첩 선택) − 등가중 Stefan+CCI | (i) 4지역, (ii) 6지역 | 블록 / §4.2 | seed별 Δ 후 평균 | 중첩 채점 + 층화 부트스트랩 | 전이(Holm) | 09-15 §2 | 게이팅 가중은 중첩 선택 |
| H8 | T3 중요도 가중 MLP(직접) − 순수 MLP; 동시에 Stefan 앵커 대비 |Δ| ≤ 1 cm 동급성 | (i)(ii) | 블록 / §4.2 | seed별 Δ 후 평균 | 차이 CI + 동급성은 CI가 [−1, +1] 안에 들어오는지 | 전이(Holm) | 09-15 §2 | 밀도비는 알래스카 대 대상 공변량 분류기로 추정 |
| H9 | T1 최적(중첩) 앵커+잔차 구성 − 앵커 단독 | (i)(ii) | 블록 / §4.2 | seed별 Δ 후 평균 | 중첩 채점 + 층화 부트스트랩 | 전이(Holm) | 09-15 §2 | in-sample 최소는 보조 열 |
| H10 | T0 물리 계수 지도 앵커 Ê(x)√TDD − 상수 E Stefan | (i) 공변량만 4지역 | 블록 / §4.2 | 해당 없음(회귀 seed 없음) 또는 catboost seed | 분할 반복 CI + 층화 부트스트랩 | 전이(Holm) | 09-15 §2(09-16 개정 추가) | E(x) 회귀는 블록 단위 LORO |
| H11 | T0b 수준·구조 분리 잔차 − 앵커 단독 | (i)(ii) | 블록 / §4.2 | seed별 Δ 후 평균 | 동일 | 전이(Holm) | 09-15 §2(09-16 개정 추가) | 목표 {y − 앵커, y − μ_region} |
| **H12(주)** | "앵커+증강+잔차" 중첩 선택 레시피 − Stefan+CCI 앵커 단독 | (i) 공변량만 4지역(분할 3회), 병기 (ii) 6지역 | 블록 / §4.2 | seed별 Δ 후 평균 | 분할 반복 CI + 지역 층화 부트스트랩 CI(주), 순열·Wilcoxon(보조) | **무보정(주 가설)** | 09-16 §3 S-E, 개정 09-21 A–D | 결정 규칙 A/B/C 판정의 근거 |
| H13 | 잔차 결합 − 앵커 단독(라벨 있음) | 라벨 있음: 4지역 A블록 실측 포함 + 알래스카 6-fold | 블록 / §4.2 | seed별 Δ 후 평균 | 동일 | 지역 내(Holm, H6과) | 09-16 §3 S-E | |
| H14 | 학습 지역 집합 확장(알래스카+대상 외 전 지역) − 알래스카만: Δ ≤ 0(개선 없음) | (ii) 정보 없음 6지역 | 블록 / §4.2 | seed별 Δ 후 평균 | 층화 부트스트랩 CI가 0 이하를 포함하는지 | 전이(Holm) | 09-16 §3 S-E | P1 결론 재검 |
| H15 | 생성 모델(조건부 플로 매칭·DDPM·정규화 플로) 잔차의 90% 구간 interval score − CQR(68.9) | 라벨 있음, 알래스카 6-fold OOF | 블록 / 13,606셀·74블록 | seed별 Δ 후 평균 | 짝지은 블록 부트스트랩(proper scoring rule 차이) | UQ(Holm, H17과 제안) | 09-16 개정(09-16 사용자 요청) | 점예측 정확도는 탐색적 |
| H16 | Stefan 유사라벨 − 자기훈련(`self`) 유사라벨 | (i) 공변량만 4지역 | 블록 / §4.2 | seed별 Δ 후 평균 | 분할 반복 CI + 층화 부트스트랩 | 증강(제안, 개정 문서 미지정) | 개정 09-21 F | C1 "물리 정보" 해석은 이 대조를 넘을 때만 유지 |
| H17 | 알래스카 보정 CQR 구간의 대상 지역 커버리지 < 명목 90%; 밀도비 가중 등순응·대상 유사라벨 보정의 회복은 탐색적 | (ii) 대상 지역 전체 셀 | 셀·블록 / 지역별 평가 셀 | seed별 커버리지 평균 | 커버리지 블록 부트스트랩 95% CI가 0.90 미만인지(확인적), 회복은 방향만 | UQ(제안) | 개정 09-21 F | S-D·S-F에서 실행 |

### 4.4 사전 등록 문서와 커밋 이력(정직 기록)

| 문서 | 내용 | 작성일(문서 내) | git 커밋 | 결과 열람과의 순서 |
|---|---|---|---|---|
| `EXPERIMENT_PLAN_PAPER_2026-09-08.md` | H1–H6, E1–E4 설계 | 2026-09-08 | **ff6504a** (2026-09-15 15:54, 09-14 대조·09-15 전이 계획과 함께 최초 커밋) | 최초 커밋이 E1–E4 결과 확인 후. 문서 내 개정 이력에 "E4.1·E2 결과 확인 후 E1 앵커 수준 2개 추가", "E3 후 주 집합 150 cm 기준 재정의", "H2 기각 기록" 등 결과 확인 후 변경이 기록됨. 원고에는 "계획을 문서화한 뒤 실행하였으나 git 이력으로는 선후를 증명할 수 없다"고 적는다 |
| `EXPERIMENT_PLAN_TRANSFER_2026-09-15.md` | V1–V4, T0–T6, H7–H11, 결정 규칙 | 2026-09-15 | ff6504a(초판), **b13f4c4**(09-16, §0b 주 축 재정의·H10·H11 추가) | 실행 전 커밋 |
| `EXPERIMENT_PLAN_MASTER_2026-09-16.md` | 불변 규약, 축·수준, S-A~S-G, H12–H14, 결정 규칙 | 2026-09-16 | **2b6d969**(09-16 16:11), **291e2c7**(09-16 16:22, 모델 축 확장·H15) | S-A 착수(09-18) 전 커밋. 결과 DB 메타(`m1_gate_shard0_meta.json`)의 `git_commit` = 291e2c7 |
| 같은 문서의 개정 2026-09-21(A–F, H16·H17) | 채점·짝지음·층화 부트스트랩·조건별 지역 집합·Holm·결정 규칙 재정의 | 2026-09-21(S-C 시작 09-21 10:27, 결과 열람 전 명시) | **미커밋**(작업 트리 `M docs/EXPERIMENT_PLAN_MASTER_2026-09-16.md`, +32행) | S-C 결과 열람 전에 커밋해야 사전 등록 주장이 성립한다. **즉시 커밋 권고** |

### 4.5 원고 각주로 옮길 사전 등록 이탈 목록

1. H1 가중: 알래스카 최소제곱 → 고정 0.5·무보정 CCI(최소제곱 가중 1.0 클리핑).
2. H3 짝지음: 09-08 실행은 seed 앙상블 후 짝지음. 09-21 규약(seed별 Δ)으로 재산출.
3. E1 앵커 수준 `stefan_soil`·`stefan_soil_cci`는 E2(레나·캐나다 포함) 결과 확인 후 추가 → 레나·캐나다 F7 결과는 탐색적, 확인은 신규 지역에서.
4. 주 전이 집합 기준: ALT 평균 300 cm → 150 cm(09-08 개정) → v3에서 라벨 관측법(지온·융해관 유도) 기준으로 서술 전환.
5. S12 옛 채점: 기계학습 계열은 CCI 결측 셀 포함 전체 셀, 21.32는 캐나다 단일 seed 구성(seed 완비 최소 21.42). 부트스트랩 횟수 400(E1)에서 1,000(M1)으로 통일.
6. `ku_cal`·`stefan_ku`·`stefan_ku_cci` 앵커는 결과 확인 후 추가(탐색적).
7. TabPFN 제외(라이선스 토큰), RealMLP seed 1·별도 shard.

---

## 5. Reporting Summary 대응 초안

Scientific Reports는 생명과학·행동과학·생태·진화·환경과학 논문에 Nature Portfolio Reporting Summary를 요구한다. 본 원고가 환경과학 항목으로 분류될 경우를 대비해 해당 양식의 항목 순서로 초안을 둔다. 분류가 물리과학이면 통계 요건(효과크기·CI·정확 p·표본 단위)만 본문에 충족하면 된다 **[편집부 분류 확인 필요]**.

### 5.1 Study description · Research sample

- 설계: 관측 자료의 재분석·벤치마크(실험 개입 없음). 세 조건(라벨 있음·공변량만·정보 없음) × 여섯 축(물리식·증강·잔차·모델·학습 지역 집합·앵커 제품) 요인 설계, 단일 하네스·단일 결과 DB.
- 표본: 고유 좌표 셀 17,572(v3), 그중 탐침·GPR 실측 라벨(F4_direct) 17,467셀. 알래스카 13,606셀(0.5° 블록 74; 라벨의 86%가 ABoVE GPR 유도, 7.7% 탐침), 전이 주 집합 6지역 3,858셀, 심부 5지역(지온·융해관 유도 68셀은 등급 3으로 분리).
- 표본 크기 근거: 사전 검정력 계산 없음(관측 자료 전수 사용). 사후 MDE(2.8 × 부트스트랩 SD)를 모든 비유의 결과에 병기. 지역 수준 부호검정(n=6)은 검정력 약 13%로 기술 통계로 강등.

### 5.2 Data exclusions(자료 제외 규칙 표)

| 규칙 | 기준 | 영향 | 사전 지정 여부 | 근거 |
|---|---|---|---|---|
| ABoVE 원자료 QC | ALT 결측(−9999) 제거, 0 < ALT < 300 cm, 좌표 45–80°N·−170~−60°E | 점관측 22.4만 → site-year 집계 | 파싱 시점 고정 | `parse_above_alt.py` |
| ALLena QC | 품질 플래그(QF TD) 값 제외·표시, 8–9월 측정 사용 | 레나델타 3,037셀 | 파싱 시점 고정 **[세부 규칙 확인 필요]** | `parse_allena.py` |
| 평가 셀 집합 | CCI 유효 ∩ 토양 도일 유효 ∩ y 유효. 모든 방법 동일 셀 | 레나 3,037→2,958, 캐나다 750→747, 러시아 W 31→28, E 30→27, 그린란드 3→1(옛 S12는 두 지역 결측 80셀) | 09-16 §1 사전 지정 | `m1_core.eval_mask`, `l2_common.py` |
| 완전 사례 보조 채점 | SoilGrids·CCI 결측 셀의 NaN 라우팅 확인을 위해 지역별 결측 셀 수 기재, 완전 사례 재채점 병기 | 표 각주 | 개정 09-21 E | |
| 심부 레짐 분리 | v3: 라벨 관측법(시추공 지온·융해관 유도 = `F4_calm_temp`, 등급 3)을 기준으로 주 집합에서 제외. 초판의 ALT 평균 300 cm·개정 150 cm 기준은 역사 기록 | 심부 5지역(스발바르·몽골·알프스·티베트·스칸디나비아), 알래스카 계수로 RMSE 180–350 cm | 09-08 §3.3 → 09-08 개정 → 09-15 §4·S-A | `fidelity_base_v3_meta.json` |
| 근접 중복 제거 | 기존 셀과 0.01° 이내 신규 좌표는 기존 우선 | 8셀(알프스 2·러시아 4·스발바르 2) | 09-08 §3.2 | `fidelity_base_v2_meta.json` (`dedup_dropped: 8`) |
| 토양 도일 0 셀 | `e5_tdd_soil ≤ 0`(연중 토양 0 °C 미만, 빙상 격자) → 결측 처리, 토양 도일 Stefan은 대기 도일 폴백 | 8행(그린란드 2·캐나다 1·스발바르 시추공 3 등) | 09-15 §4-3 | `e5_soil_tdd_v3.csv` |
| 지온 유도 라벨 강등 | CALM 이벤트 헤더 `Method:` 파싱(261/263 유효): thaw tube·temperature → `F4_calm_temp` | 68셀(알프스 9·몽골 46·QTP 6·스발바르 3·스칸디나비아 2·캐나다 2). 주 집합 탐침 셀 보존(캐나다 752→750). C20 Baker Lake는 Method 없음(유지·플래그) | 09-15 §4-2 | `build_fidelity_base_v3.py` |
| 스발바르 매크로 분리 | `CALM_Svalbard` & lat < 76° → `Scandinavia`(심부) | 3사이트(Juvvasshøe·Snøheim·Abisko) | 09-15 §4-1 | 동일 |
| A/B 분할 불가 지역 | A블록 0–1셀인 그린란드는 공변량만·라벨 있음 조건에서 제외, 정보 없음·배포형만 보고. 러시아 C(7셀)는 CI 미산출 | 확인적 조건 (i) = 4지역 | 개정 09-21 B | |
| 캐나다 셀 판 변경 | v1 742 → v2 752 → v3 750(중복·강등) | 옛 표(34.0·31.8) 대 v3 값 차이의 원인, 각주 | S-B 게이트 | `m1_gate_gate.csv` |
| GTN-P ALT 106 데이터셋 | CALM 중복 87·좌표 미상 11·신규 5 → 편입 보류 | 없음 | S-A | `gtnp_alt_inventory.csv` |

### 5.3 Reproducibility(반복)

- 무작위 반복: 모델 초기화·증강 표본 seed 3(0·1·2), A/B 블록 분할 3회(split seed 0·1·2), CQR 보정 블록 분할 3 seed, 알래스카 fold 배정 10회 반복 민감도(보충자료).
- 재현 게이트(S-B): 옛 표의 대표 수치 18개를 새 하네스에서 허용 오차 0.1 cm로 재산출. 13개 통과, 캐나다 5개 불일치는 셀 판(v1/v2/v3)과 채점 셀 집합 차이로 규명(`--eval-all` v1 재실행 시 33.96/31.66/10.35로 재현). 공용 계층 `l2_common.py`도 동일 값(14.457·13.330·13.620·21.624/20.761·26.504/25.719) 재현.
- 누설 통제 테스트: pytest 25건(대상 라벨이 학습·계수 적합에 미사용, 평가 셀 동일, 기존 행 불변).
- 실패한 재현 시도: 없음(불일치는 전부 원인 규명). 폐기 실험(S13·S5·S8·B0/B1 등)은 보충자료 목록에 사유와 함께.

### 5.4 Randomization(무작위화)

| 대상 | 방식 | seed |
|---|---|---|
| 알래스카 6-fold | 0.5° 블록 GroupKFold(블록 ID 기준, 결정적) | 없음(민감도: 무작위 fold 배정 10회) |
| 대상 지역 A/B 분할 | split seed 0: GroupKFold n=2(E1·S3·S12 규약); seed 1·2: 블록 무작위 순열 후 셀 수 균형 탐욕 배정 | 0·1·2 |
| 증강 표본·모델 초기화 | seed별 독립 | 0·1·2 |
| CQR 보정 표본 | 각 fold의 학습 블록 중 25%를 seed별 난수(rng 1000+seed)로 분리, 시험 블록 무접촉(assert) | 0·1·2 |
| 부트스트랩 | 블록 재표집 1,000회 | 0 |
| 관측 자료 배정 | 해당 없음(관측 자료 재분석, 처리군 없음) | |

### 5.5 Blinding(블라인딩)

해당 없음. 관측 자료의 재분석이며 사람의 판정이 개입하는 측정이 없다. 대신 선택 편향은 사전 등록 주 추정치·중첩 선택·재현 게이트로 통제한다.

### 5.6 Timing and spatial scale(시공간 규모)

| 항목 | 값 | 근거 |
|---|---|---|
| 분석 단위 | 셀 = 고유 좌표(좌표 반올림 단위 집계, 다년 평균 ALT). 위치 내 변동 평균 3.8 cm(중앙값 2.0) | `aggregate_alt_cell.py`, CONTEST_REPORT §2.1 |
| 교차검증 블록 | 0.5° 격자(알래스카 74, 지역별 §4.2) | `fidelity.py` |
| 공변량 해상도 | ERA5-Land 0.1°(약 9 km) 월별 → 2015–2020 평년; DEM 30 m; SoilGrids 250 m; CCI 1 km(1997–2021 다년 평균) | 각 스크립트 |
| 라벨 연도 | 알래스카 중앙 연도 평균 2014.9(2015–2020과 겹침 34%, 이전 65%, 이후 1%), 레나 2012.9, 캐나다 2019.1, 러시아 W 2015.2, E 2009.4, C 2007.6, 그린란드 2006.2. 전체 F4_direct 17,467셀 중 겹침 6,417(완전 포함 5,581)·이전 10,434·이후 616 | `label_year_sensitivity.csv` counts 행, `_meta.json` |
| 라벨 연도 대 평년 불일치 민감도 | 학습을 겹치는 셀로 제한해도 겹치는 셀 오차 불변: Stefan −0.02 [−0.41, 0.29], ridge +0.58 [−0.73, 2.17], catboost_lo +0.67 [−0.34, 1.55] cm. 잔차(관측 − Stefan)의 연도 기울기 알래스카 −0.69 [−1.84, 0.91], 주 6지역 +0.39 [−0.61, 1.37] cm/yr. 문헌 추세 0.8 cm/yr가 성립해도 RMSE 기여 0.07(알래스카)·0.44(전이) cm | 동일 파일 indomain·tests 행, EXPERIMENT_LOG 09-21 |
| 다년 평균 규약의 선례 | Karjalainen 2019·Aalto 2018·Hjort 2018(2000–2014 평균), Ran 2022(2000–2016 대표값); 연별 라벨 연구는 같은 해 강제 결합. S14 연별 실험 anomaly R² ≤ 0.0015 | EXPERIMENT_LOG 09-21 |
| 현장 검증 시기 | KPDC 콘슬 2019·2021–2025, c1 2016·2018 | main.tex §2.4 |

### 5.7 Field work

해당 없음(저자 현장 조사 없음; KPDC 관측은 KOPRI가 수행).

---

## 6. Gautam et al. 2025 대비 서술과 선행 공개 고지

### 6.1 Gautam et al. 2025 설계 요약(로컬 PDF 본문 확인)

Gautam et al. (2025), *Scientific Reports* 15, 42420, doi:10.1038/s41598-025-26586-w. CALM 알래스카 68지점의 2014년 단일 연도 ALT(평균 54 ± 23 cm, 25–168 cm)를 WorldClim 1970–2000 평년·토지피복 등과 결합, 무작위 70/30 분할로 Random Forest를 학습·검정(학습 R² 0.84, 검정 R² 0.24, RMSE 14–22 cm), Stefan 모형은 학습 0.53·검정 0.54(RMSE 17–18 cm). CMIP6 SSP2-4.5·SSP5-8.5로 2100년까지 ALT 변화를 투영.

### 6.2 서론 마지막 문단 초안(국문)

같은 지면에 게재된 Gautam et al. (2025)은 알래스카 CALM 68지점의 2014년 탐침 관측으로 Random Forest와 Stefan 모형을 비교하여, 무작위 70/30 분할에서 Random Forest의 결정계수가 학습 0.84에서 검정 0.24로 떨어지는 반면 Stefan 모형은 0.54를 유지함을 보고하였다. 본 연구는 같은 지역·같은 물리식을 다루지만 질문과 설계가 네 가지 점에서 다르다. 첫째, 표본은 CALM과 ABoVE의 고유 좌표 13,606셀(다년 평균, 86%가 지표투과레이더 유도)로 지점 수가 두 자릿수 크며, 라벨 정의·연도 범위의 민감도를 별도로 검정한다. 둘째, 검증은 0.5° 공간블록 교차검증(74블록)과 중첩 선택으로 공간 자기상관과 구성 선택에 의한 낙관 편향을 통제하며, 무작위 분할은 쓰지 않는다. 셋째, 알래스카 내부 정확도가 아니라 관측이 없는 지역으로의 전이가 주 질문이며, 주 6지역과 심부 4지역의 지역 단위 홀드아웃(LORO)에서 배포 조건(대상 지역 지도 자료만으로 적응)과 비적응 기준선을 함께 평가한다. 넷째, 예측구간은 등순응 분위 회귀로 보정해 커버리지를 검증하고 적용가능 영역을 병기하며, 이는 Gautam et al.에는 없다. 두 연구의 절대 RMSE는 라벨 연도·집계 단위·검증 방식이 달라 직접 비교할 수 없으므로, 본 연구는 수치 우위를 주장하지 않고 설계 차이를 보고한다.

### 6.3 같은 문단의 영문 초안

> Gautam et al. (2025), published in this journal, compared a random forest with a Stefan model at 68 CALM sites in Alaska using single-year (2014) probe observations and a random 70/30 split, and reported that the random forest's coefficient of determination fell from 0.84 (training) to 0.24 (test) while the Stefan model retained 0.54. The present study addresses the same region and the same physical formula but differs in four respects. First, the sample comprises 13,606 unique-coordinate cells from CALM and ABoVE (multi-year means; 86% derived from ground-penetrating radar), two orders of magnitude more locations, and the sensitivity to label definition and observation years is tested explicitly. Second, validation uses 0.5° spatial-block cross-validation (74 blocks) and nested configuration selection rather than a random split, so that spatial autocorrelation and selection do not inflate the reported skill. Third, the primary question is transfer to regions without observations: models are scored by leave-one-region-out evaluation on six main and four deep-regime regions, under both a deployment condition (adaptation with target-region map data only) and a non-adaptive baseline. Fourth, prediction intervals are calibrated by conformalized quantile regression with verified coverage and are accompanied by an area-of-applicability mask. Because label years, aggregation units and validation schemes differ, the absolute RMSE values of the two studies are not comparable, and no claim of numerical superiority is made.

### 6.4 SI '설계 비교' 표 초안

| 항목 | Gautam et al. 2025 | 본 연구 |
|---|---|---|
| 라벨 | CALM 68지점, 2014 단일 연도, 탐침 | CALM + ABoVE 알래스카 13,606셀(고유 좌표 다년 평균; GPR 유도 86%·탐침 7.7%) + 전이 대상 10지역(레나델타 ALLena 3,037, 캐나다 750, 러시아 W/E/C, 그린란드, 심부 4/5) |
| 공변량 | WorldClim 1970–2000 평년, 토지피복 등 | ERA5-Land 2015–2020 평년 8종, DEM 6종, SoilGrids 9종, CCI 2종(전이 공통 25종); 알래스카 내 SAR 8종 추가 |
| 라벨·공변량 시기 | 2014 라벨 대 1970–2000 평년 | 다년 평균 라벨 대 2015–2020 평년, 연도 민감도 검정(§5.6) |
| 검증 | 무작위 70/30 분할 | 0.5° 공간블록 6-fold(74블록), A/B 블록 분할 3회, 중첩 선택, seed 3 |
| 전이 | 없음(알래스카 내) | LORO 주 6 + 심부 4(v3 5), 배포 조건(공변량만)·비적응 기준선·라벨 있음 |
| 물리 모형 취급 | Stefan 대 RF 비교 | 물리식 사다리(같은 보정 자유도 k), 유사라벨 증강, 앵커 결합, 대조군 사다리(상수·셔플·TDD 선형·자기훈련) |
| 불확실성 | 기준선에 없음(투영은 앙상블 평균·범위) | CQR 보정 구간(커버리지 93.4%, interval score 병기), AOA |
| 통계 | RMSE·R² 점추정 | 짝지은 블록 부트스트랩 95% CI, 지역 층화 부트스트랩, Holm, MDE, 사전 등록 |
| 보고 성능 | RF 검정 R² 0.24·RMSE 14–22 cm; Stefan 0.54·17–18 cm | 지역 내 Stefan 14.46, Stefan+ridge 13.33(풀링), 앵커 없는 ridge 13.62, MLP 14.39; 전이 Stefan 21–24, 결합 앵커 23.27(정보 없음 2지역 평균). 직접 비교 불가 |
| 미래 투영 | CMIP6 SSP 2100 | 범위 밖 |

### 6.5 선행 공개 고지(커버레터 문장)

국문 요지: 본 원고의 초기 결과는 2026 극지 빅데이터·인공지능 활용 경진대회(KOPRI/KPDC)에 국문 분석 보고서(예선, 2026-07-30 제출, `submission/분석보고서_ALT_Ctrl.pdf`; 제목 "희소 관측 조건에서 물리경험식 유사라벨 증강과 기계학습을 이용한 영구동토 활동층 두께 예측 및 독립 관측 결합에 의한 지역 간 전이")와 본선 발표 자료(2026-09)로 제출되었다. 동료 심사를 거치지 않았고 학술지·프로시딩에 게재되지 않았다. 본 원고는 사전 등록한 재설계 실험(주 전이 지역 2 → 6, 대조군 사다리, 중첩 선택, 통계 규약)으로 결과를 갱신한 것이며, 헤드라인 수치(예: 22.92 → 23.27 cm)와 해석이 대회 보고서와 다르다.

> **Cover letter (prior dissemination).** Preliminary results of this work were presented, in Korean, as an entry to the 2026 Polar Big Data and AI Contest organised by the Korea Polar Research Institute and the Korea Polar Data Center (analysis report submitted 30 July 2026 and a final-round presentation in September 2026). Neither document was peer reviewed or published in a journal or conference proceedings. The present manuscript reports a redesigned, pre-registered set of experiments (expanded transfer regions, control-ladder comparisons, nested selection and revised statistical protocol); the headline estimates and interpretations therefore differ from the contest report. We are happy to provide the contest report to the editors on request.

---

## 7. 용어 정정 목록(원고 적용용)

`docs/RESULTS_RECONCILIATION_2026-09-14.md` §5(원고 변경 표)·§6(서술 정정)을 main.tex 행 번호와 함께 옮긴다. 값은 09-14 대조 시점 기준이며 M1 결과 DB 확정 후 v3 값으로 갱신한다.

### 7.1 §5 원고 변경 표(위치 부여)

| 위치(main.tex) | 현재 | 변경 |
|---|---|---|
| 제목 85행, 4.7절 제목 527행, 그림 캡션 531행 | "독립 관측(의) 결합" | "위성 기반 제품(준독립)과의 결합". 영문 제목·절 제목에서 independent observation을 satellite-based product(CryoGrid 산출물, ERA5 강제력 일부 공유)로 |
| 초록 101행, 결론 654행, 표 tab:s12 565행, 본문 546–550행 | "24.11에서 22.92 cm", "+ 잔차 학습 21.32/22.92", "네 조합 모두 오차 감소" | "24.11에서 23.27 cm(등가중 앵커, 사전 지정)". 잔차 학습 행은 삭제하거나 "in-sample 최소, 중첩 선택 25.49/25.32, 캐나다 단일 seed" 병기. "방향 일관(셀 가중 5/6, 8/10), 95% CI 0 포함, 블록 다수결 혼재"로. 캡션 555행의 "같은 셀 집합" 진술 정정(기계학습 계열은 결측 80셀 포함 전체 셀) |
| 초록 101행, 기여 133행, 식 라벨 244–245행, 캡션 421행, 표 헤더 448행, 456행, 논의 642행, 결론 654행 | "정확한 물리식/부정확한 물리식(Kudryavtsev)" | "보정한 물리식(Stefan, E 적합)/무보정 물리식(Kudryavtsev, pedotransfer 기본값)". 같은 자유도로 보정한 Kudryavtsev 라벨은 시험되지 않았음(옛 실험) 또는 M1 결과로 갱신 |
| 4.3절 tab:aug 426·439·448–456행 | 상수 대조 순가치 10.2/1.7 | 대조군 사다리(없음·상수·셔플·TDD 선형·Stefan): 캐나다 10.8 = 수준 6.2 + 기후 단조 3.6 + √ 함수형 1.0 [0.4, 1.4]; 레나 1.7 = 1.0 + 0.6 + 0.1 [−0.05, 0.20]. 마지막 항은 "√ 함수형의 추가 기여"(TDD 선형 대조도 보정 도일식). S-C 예비: 대상 수준 상수 대비 레나 −0.31 [−0.86, 0.45], 캐나다 −3.82 [−4.68, −1.84] |
| 4.5절 475–478행, 표 tab:transfer 493행 | 13.33 "물리와 데이터가 보완" | 앵커 없는 능형 13.62 병기, 앵커 추가 기여 −0.29 [−0.61, +0.08] 비유의, 개선 1.13의 약 74%는 선형 회귀 몫. 14.37(신경망)은 34종·3-seed 규약 각주 |
| 4.5절 478행·493행 | 35.45, λ 단조 증가, "알래스카의 공변량-잔차 관계" | H2: 주 6/6 악화(p=0.031), 외삽 폭주. 단조 악화는 능형·MLP의 성질, 저용량 CatBoost 평탄(|Δ| ≤ 0.5). "학습 지역의 잔차 관계" |
| 4.2절 tab:physics 397행, 캡션 416행, 433행 | "Stefan만 정확", "불확실성 전파" | k 사다리 표. 격차 원인은 전역 배율 편향(c ≈ 0.53–0.71). 3지역 기준 k=1 Stefan 최저·k=2 동급. "계수 E만 적합"은 Stefan에만 해당했음을 캡션에 명시 |
| 4.6절 517행 캡션 및 본문 | ML 34–38 붕괴 | fold별 풀어 쓰기(알래스카·레나 홀드아웃 41.5·47.5, 캐나다 격차 없음) + 알래스카 고정 학습 시 신규 5지역 +2–7 cm |
| 4.9절 594–631행, 캡션 619행, 626행 | 커버리지 93.4%, 구간 폭 지도로 신뢰 판단 | 커버리지 유지(CI [0.89, 0.96]) + interval score CQR 68.9 vs 상수 폭 참조 65.4(사후 보정 oracle), 폭·오차 순위 상관 −0.07, 양극단 5분위 0.83–0.85. "폭 지도로 어느 지역을 신뢰할지 판단" 문장 삭제(619·626행 주변 **[정확한 문장 위치 확인 필요]**). AOA 표기는 유지 |
| 초록 101행, 기여 134행, 논의 639–642행, 결론 654행 | "전이의 지렛대는 정보원의 다양성" 단정 | 확인적: H1 방향 일관(셀 가중) 비유의, 블록 다수결 혼재. 탐색적: 우세 앵커가 지역에 따라 갈림(레나·캐나다 Stefan+CCI, 러시아 W/E 보정 Kudryavtsev), 3결합 견고(Ku는 사후 추가). "지렛대"는 "가설"로 |

### 7.2 §6 정정: 원고에 쓰지 말아야 할 서술 5건

| 피해야 할 서술 | 감사 결과 | 쓸 서술 |
|---|---|---|
| CCI 결합 앵커는 전이에서만 이득, 지역 내에서는 해(+2.02) | 무보정 등가중 구성에만 해당. 보정 결합 −0.24 | "무보정 등가중 결합은 지역 내 스케일 불일치로 악화한다. 보정 결합은 악화가 없고 전이 이득은 줄어든다" |
| 물리 순가치 대부분은 수준+단조성, √ 고유 기여 ≤ 1 cm | TDD 선형 대조도 보정 도일식(상관 0.999) | "√ 함수형의 추가 기여 ≤ 1 cm. 도일 관계 자체의 가치는 상수 대조 대비 3.6·0.6" |
| 7지역 평균은 보정 Kudryavtsev 우세 | 비가중 평균, 우세의 81%가 그린란드 3셀 | "러시아 서부에서 보정 Kudryavtsev가 유의하게 낫다(−12.7). 평균 우위는 집계 방식 의존" |
| 토양 도일 Stefan 확인 실패(4/6 악화) | 그린란드 TDD_stl1=0 아티팩트가 82% | "알래스카·레나 우세, 러시아 열세, 그린란드는 공변량 퇴화로 판정 불가" |
| H1 방향 일관 5/6·8/10 | 셀 가중 진술, 블록 다수결 반대(레나 5/11, 캐나다 9/15) | "셀 가중 기준 5/6, 블록 다수결로는 대지역에서 혼재" |

### 7.3 기타 표기 정정

- 자료 표 캡션 186행·참고문헌 670행: KPDC 식별자 00002125만 표기 → 00002125(코어)·00002707(지온 2024)·00002955(지온 2025) 구분.
- 참고문헌 680행: CCI ALT v4.0 DOI 10.5285/d34330ce3f604e368c06d76de1987ce5(확인됨). 저자 표기 "Westermann, S. et al. (2024)" 유지.
- 자료 표 194행 캐나다 742셀 → v3 750셀(각주에 판 변경).
- r=10 증강이 실측 없는 셀의 반복 복제(가중 조절)임을 방법 절에 명시(PAPER_PLAN §2-3).
- "위성 관측 기반 ALT 제품" 542행 → "위성 지표온도 강제 CryoGrid 모형 산출물(ESA CCI ALT v4.0)".

---

## 8. 확인 필요 항목과 주의 사항

### 8.1 확인하지 못한 식별자·문구

1. KPDC 3종(KOPRI-KPDC-00002125·00002707·00002955)의 공식 데이터셋 제목·발행 연도·이용 정책 문구, AWS 자료의 귀속 식별자.
2. GTN-P 데이터베이스 권장 인용 형식·데이터 정책 문구(매니페스트에는 `policy: Open`만 존재).
3. SoilGrids 취득 시점의 데이터 버전 태그(WCS "latest").
4. ABoVE ReSALT(ds2004) 페이지 인용 정보, ABoVE upscaled ALT 2014–2017(Whitcomb et al.) 데이터셋 번호·DOI, `ornl_ds2332`의 정확한 제목·DOI.
5. ALLena 파싱의 품질 플래그 제외 규칙 세부(`parse_allena.py` 재확인).
6. 대회 공식 영문 명칭, 저자명·소속·자금 정보.
7. Reporting Summary 적용 여부(편집부 분류).
8. `scripts/2_evaluation/`의 `l2_*`·`a2_*` 개별 분석 스크립트 존재 여부(현재 공용 모듈 2개만).

### 8.2 주의 사항

- 09-21 개정(H16·H17, 규약 A–F)은 작업 트리에만 있고 미커밋이다. S-C 결과를 열람하기 전에 커밋해야 사전 등록 서술이 성립한다.
- 09-08 계획의 최초 커밋(ff6504a, 09-15)은 E1–E4 결과 이후이므로 H1–H6은 "문서화 후 실행"으로만 서술하고 git 이력으로 입증한다고 쓰지 않는다.
- `environment-lock.txt`는 `pip freeze` 원문(517행)이며 conda 설치분 310행이 로컬 경로 형식이다. 공개 전 이름==버전 형식 보완본 또는 `environment.yml`이 필요하다.
- `configs/`, `data/processed/fidelity_base_v3*.csv`, `e5_soil_tdd_v3.csv`, `gtnp_alt_inventory*`는 git 미추적 상태다. 공개 범위에 넣으려면 커밋해야 한다.
- 본 문서의 수치는 09-14 대조·S-B 게이트·S-C 예비 값이며, 원고 확정값은 M1 결과 DB(개정 09-21 규약 분석) 산출 후 교체한다.
- KPDC 파생 값의 재배포 범위는 KPDC 정책 확인 전까지 요약 통계로 제한한다.
