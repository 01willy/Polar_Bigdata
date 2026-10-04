# References — 문헌 인벤토리 (Polar_Bigdata)

> 총 **262개 항목**(로컬 PDF **171편**, 링크전용 91개). 1차(2026-07-06) 49편과 2차(2026-10-04) 213개를 합한 수다.
> - 1차: 총 **49편**(web-검증 완료, 2026-07-06 문헌 워크플로 57 에이전트). 오픈액세스 PDF **36편** 카테고리 폴더(01–07)에 다운로드, 나머지 13편은 링크전용(페이월/자동차단).
> - 2차: 총 **213개**(폴더 08–14). PDF **135편**, 링크전용 78개(구독 필요, OA 자동 차단, 웹 열람). PDF 는 모두 `file`·`pdfinfo` 로 확인했다(손상 0건, 삭제 0건).
> - `00_core10/` 은 1차 핵심 10편의 사본 폴더라 항목 수에 넣지 않는다. 같은 논문이 두 키로 들어간 1차 항목 1쌍(03 절, 아래 경고)은 정리하지 않고 그대로 센다.
> PDF는 저작권상 git 제외(`references/**/*.pdf`), 본 INDEX(.md)만 추적. 1차 종합 분석·모델 로드맵·T1/T2 판정은 `gpt/handoff/20260706_1717-lit-review-forecasting-4d-tracks.md`. 2차 조사 문서는 `docs/research/2026-10-04/`, 읽는 순서는 `references/README_STUDY_GUIDE.md`.

**범례**: 📄=PDF 로컬 다운로드됨 · 🔗=링크전용(페이월/자동차단) · 관계=[벤치마크|경쟁자|선례|방법|맥락|off-direction]

**2차 추가 표기**: 🔗 는 구독 필요, OA 자동 차단(합법 공개본이 있으나 이 서버의 내려받기가 봇 차단에 막힘), 웹 열람(PDF 없음)을 모두 포함하고 사유를 링크 줄에 적는다. 추가 관계 = [자료](자료 논문·라벨·공변량 출처) · [협업자](KOPRI 협업 예정 연구자의 저작) · [그림](그림 설계 모범 사례) · [그림·반면교사](피할 그림 설계의 예) · [형식](Sci Rep 원고 형식 예시). 확인 수준 = [본문]·[초록]·[제목]·[미확인].

**절별 집계**

| 절 | 폴더 | 항목 | PDF | 링크 |
|---|---|---|---|---|
| 01 기준·경쟁 산출물 | `01_benchmark/` | 3 | 3 | 0 |
| 02 ALT 공간 매핑 | `02_alt_dl_mapping/` | 5 | 5 | 0 |
| 03 시계열 예측 | `03_alt_forecasting/` | 10 | 5 | 5 |
| 04 시공간/4D | `04_spatiotemporal_4d/` | 6 | 4 | 2 |
| 05 불확실성·전이 | `05_uq_transfer/` | 13 | 10 | 3 |
| 06 물리결합 ML | `06_physics_ml/` | 11 | 8 | 3 |
| 07 맥락 | `07_context/` | 1 | 1 | 0 |
| 08 최근 ALT 2022–2026 | `08_recent_alt_2022_2026/` | 54 | 30 | 24 |
| 09 그림 모범 사례 | `09_figure_exemplars/` | 17 | 10 | 7 |
| 10 KOPRI 협업 연구자 | `10_kopri_collaborators/` | 63 | 36 | 27 |
| 11 검증 설계·통계·ML 방법 | `11_validation_and_methods/` | 31 | 22 | 9 |
| 12 원고 참고문헌 | `12_manuscript_refs/` | 17 | 11 | 6 |
| 13 그림·슬라이드 설계 | `13_figure_and_slide_design/` | 23 | 18 | 5 |
| 14 Sci Rep 형식 예시 | `14_scirep_exemplars/` | 8 | 8 | 0 |
| 합계 | | 262 | 171 | 91 |

## ⚠️ 인용 전 수정할 메타데이터 오류 (검증에서 적발)
- **Ran2022 공저자 일부 날조** → 정정: Ran Y., Li X., Cheng G., Che J., Aalto J., Karjalainen O., Hjort J., Luoto M., Jin H., Obu J., Hori M., Yu Q., Chang X.
- **`rahaman_northslope_ai_temps`(Earth Sci Inf 2024) 저자 오귀속** → 실제 저자 = **Chance, Ahajjam, Putkonen, Pasch** (MERRA-2 기반 GBDT/RF/SVR). Rahaman2025(arXiv:2510.06258)와 **다른 그룹**.
- **`ieki2025_neuralkriging_japan`** = 실제 저자 **Ieki et al.**(과거 노트의 'Suzuki 2025'는 오귀속).
- (2026-10-04 추가) **`koven_review2025_ml_permafrost`(05 절, `00_core10` 사본 포함) 저자 오귀속** → 실제 = Memiş, M. A., Keskin, I., Demir, S. & Ulus Memiş, Ş. Machine learning-based prediction of permafrost degradation and its implications on geotechnical infrastructure: a comprehensive review. *AI Civ. Eng.* 4, 28 (2025). doi:10.1007/s43503-025-00080-8 (PDF 1쪽과 Crossref 확인). 파일명은 바꾸지 않았다.
- (2026-10-04 추가) **`yin2024_alt_upscaling_airborne_gee`(02 절) 저자** → 실제 = Merchant, M. A. & McBlane, L. (PDF 1쪽 확인). 파일명은 바꾸지 않았다.
- (2026-10-04 추가) **`luo2022_qtp_thawdepth` 와 `zhang2022_qtp_alt_lstm_cnn_rf`(03 절)는 같은 논문이다**(DOI 10.1016/j.scitotenv.2022.155886). Crossref 기준 첫 저자는 Liu, Qi(공저 Niu, J., Lu, P., Dong, F., Zhou, F., Meng, X. 등)이므로 두 키의 저자 표기가 모두 틀렸다. 인용은 Liu, Q. et al. (2022) 하나로 한다.
- (2026-10-04 추가) **`liu2023_pilstm_permafrost_gipl2`(06 절) 첫 저자 이름** → Crossref 기준 Yibo Liu 다(본문의 Yuandong 은 오기).
- (2026-10-04 추가) **ESA CCI DOI 불일치**: 원고 References 표의 Westermann et al. 2024 DOI(10.5285/d34330ce3f604e368c06d76de1987ce5)와 01 절 `esa_cci_permafrost_v4` 의 DOI(10.5285/7479606004d9465bad949671501e5f21)가 다르다. 각 DOI 가 어느 CCI 변수(ALT, 지온, 범위)인지는 [미확인]이다.
- (2026-10-04 추가) **01 절 `ran2022_panarctic` 본문의 저자 나열**(Ran, Cheng, Dong, Hjort, Lovecraft, Kang, Tan, Li)은 OpenAlex 기준 CEE 논문 `ran2022b_third_pole_infrastructure`(12 절)의 저자와 같다. ESSD 논문 저자는 위 첫 줄의 정정 목록을 따른다.

## 01 · Benchmark / Incumbent (기준·경쟁 산출물)  (3/3 PDF)

폴더: `references/01_benchmark/`

### 📄 `esa_cci_permafrost_v4` — [벤치마크]
**ESA Climate Change Initiative (Permafrost_cci): Ground Temperature, Active Layer Thickness and Permafrost Extent for the Northern Hemisphere** — Westermann, S.; Bartsch, A.; Obu, J.; and Permafrost_cci team (2024, ESA CCI Permafrost v4.0 data products, NERC EDS Centre for Environmental Data Analysis (CEDA) Archive; underpinned by CryoGrid model (Westermann et al. 2017)). doi:10.5285/7479606004d9465bad949671501e5f21
- **무엇**: Operational pan-Arctic 1km gridded permafrost PRODUCT (ground temperature at fixed depths, ALT, extent, 1997-2021, annual files) from the CryoGrid forward model; a standing incumbent PRODUCT we must benchmark against for both ALT and shallow subsurface temperature.
- **우리와의 차별/활용**: CCI is a forward CryoGrid simulation delivering annual maps at fixed depths (0/1/2/5/10m) without observation-based cell-wise UQ or cross-region transfer evaluation; it is a product, not an ML transfer/UQ study. We differentiate as observation-based interpolation + transfer + cell-wise UQ + finer 3D query. NOTE: v4 (1997-2021) HAS a time axis in its annual releases, so CCI partially covers the descriptive-map side of T2; we must beat it on UQ/transfer and native fine-grid structure rather than on merely having yearly maps.
- 파일: `references/01_benchmark/esa_cci_permafrost_v4.pdf`

### 📄 `ran2022_panarctic` — [벤치마크]
**New high-resolution estimates of the permafrost thermal state and hydrothermal conditions over the Northern Hemisphere** — Ran, Youhua; Cheng, Guodong; Dong, Yuanhe; Hjort, Jan; Lovecraft, Amy Lauren; Kang, Shichang; Tan, Meibao; Li, Xin (2022, Earth System Science Data (ESSD), 14(2), 865-884). doi:10.5194/essd-14-865-2022
- **무엇**: Pan-Arctic 1km ML ensemble (GAM+SVR+RF+XGBoost) fusing 1002 MAGT boreholes + 452 ALT sites + geospatial predictors to map MAGT/ALT/permafrost probability for 2000-2016; our headline benchmark for the 2D mapping + data-fusion + UQ direction.
- **우리와의 차별/활용**: Ran2022 is coarse 1km distance-blocked (RMSE MAGT 1.32C, ALT 86.9cm) with a single-realization ensemble mean; it does NOT do explicit region-to-region TRANSFER (LORO) evaluation, cell-wise quantile UQ, or shallow 3D thermal volume. We differentiate on (a) an explicit Alaska->Siberia/QTP transfer benchmark, (b) cell-wise UQ, and (c) 0-20m 3D structure. Note direct RMSE comparison is invalid (scale mismatch: their 1km cells vs our fine-grid point ALT).
- 파일: `references/01_benchmark/ran2022_panarctic.pdf`

### 📄 `gautam2025_alaska_alt` — [경쟁자]
**Machine learning and process-based modeling of spatiotemporal changes in active layer thickness across Alaska** — Gautam et al. (2025, Scientific Reports, 15, Article 42420). doi:10.1038/s41598-025-26586-w
- **무엇**: Alaska ALT: Random Forest vs physically-based Stefan model on CALM sites, PLUS CMIP6 SSP2-4.5/SSP5-8.5 projection of ALT change to 2100; the single closest competitor to our exact task AND partially pre-empts our T2 (future-projection) track.
- **우리와의 차별/활용**: This is our nearest-neighbor competitor. RF test R2=0.24 (train 0.84 -> classic transfer/overfit collapse) vs Stefan R2=0.54; our 17cm spatial-block CV is competitive and honest about the floor. CRITICAL for T2: they already do a to-2100 SSP projection, so a plain 'project ALT to 2100' framing is pre-empted. We must differentiate via (a) explicit cross-region TRANSFER benchmark (they stay in Alaska), (b) cell-wise quantile UQ on the projections (they give only ensemble mean +/- SD), (c) spatiotemporal/4D structure rather than static-map extrapolation, and (d) representativeness-floor framing. Their RF-collapse result is strong evidence FOR our transfer/UQ novelty being an open problem.
- 파일: `references/01_benchmark/gautam2025_alaska_alt.pdf`

## 02 · ALT/영구동토 공간 매핑 (DL/ML)  (5/5 PDF)

폴더: `references/02_alt_dl_mapping/`

### 📄 `obu2019_ttop_nh` — [벤치마크]
**Northern Hemisphere permafrost map based on TTOP modelling for 2000-2016 at 1 km2 scale** — Obu, J.; Westermann, S.; Bartsch, A.; Berdnikov, N.; Christiansen, H. H.; Dashtseren, A.; Delaloye, R.; Elberling, B.; Etzelmuller, B.; Kholodov, A.; and others (2019, Earth-Science Reviews, 193, 299-316). doi:10.1016/j.earscirev.2019.04.023
- **무엇**: The equilibrium TTOP model NH permafrost/MAGT map (1km, LST + downscaled ERA-Interim + landcover CCI); the standard MAGT/permafrost-extent baseline (TTOP RMSE ~1.48C) and the temperature-mapping incumbent alongside Ran2022.
- **우리와의 차별/활용**: Obu2019 maps top-of-permafrost temperature and extent, NOT ALT directly, and is an equilibrium (steady-state) physically-parameterized model with no cell-wise ML UQ and no transfer evaluation. It is a MAGT baseline (secondary to Ran2022 for our ALT focus). We differentiate on ALT + transfer + cell-wise UQ + shallow 3D. Weakly relevant to forecasting tracks (steady-state, no time evolution).
- 파일: `references/02_alt_dl_mapping/obu2019_ttop_nh.pdf`

### 📄 `yin2024_alt_upscaling_airborne_gee` — [경쟁자]
**Machine Learning-Based Active Layer Thickness Estimation over Permafrost Landscapes by Upscaling Airborne Remote Sensing Measurements with Cloud-Computing Geotechnologies** — et al. (IntechOpen chapter) (2024, IntechOpen book chapter, in "Revolutionizing Earth Observation - New Technologies and Insights"). doi:10.5772/intechopen.1004315
- **무엇**: Upscales airborne/hyperspectral ALT measurements to multisource satellite covariates via ML (RF/XGBoost/LightGBM ensembles) on Google Earth Engine to make spatial ALT maps; reports best R2 ~0.476 with ecosystem-dependent skill.
- **우리와의 차별/활용**: A genuine but modest competitor for the 2D-ALT-mapping-from-RS niche. Overlaps in goal (spatial ALT from ML+RS) but: (1) it is airborne-upscaling with tree ensembles, NO deep learning of note, NO transfer benchmark, NO per-cell UQ, NO 3D. (2) Its R2~0.48 (localized) vs our ~0.2 (broad, harder site-year target) reflects easier range/scope, not superiority — worth framing carefully. We differentiate on transfer (LORO), calibrated cell-wise UQ, shallow 3D thermal volume, and our finding that DL ties GBM under climatological covariates (covariate bottleneck).
- 파일: `references/02_alt_dl_mapping/yin2024_alt_upscaling_airborne_gee.pdf`

### 📄 `donahue2023_nh_freeze_thaw_unet` — [맥락]
**Deep learning estimation of northern hemisphere soil freeze-thaw dynamics using satellite multi-frequency microwave brightness temperature observations** — Kellen Donahue, John S. Kimball, Jinyang Du, Fredrick Bunt, Andreas Colliander, Mahta Moghaddam, Jesse Johnson, Youngwook Kim, Michael A. Rawlins (2023, Frontiers in Big Data, Volume 6). doi:10.3389/fdata.2023.1243559
- **무엇**: U-Net CNN on SMAP L-band + AMSR2 18.7/36.5 GHz brightness temperatures, trained against ERA5/station soil temps, producing twice-daily NH soil freeze-thaw state at 9 km including a continuous 0-1 frozen-probability output (2016-2020).
- **우리와의 차별/활용**: CONTEXT, not a competitor: it classifies SURFACE freeze/thaw STATE (binary + probability), not ALT depth or subsurface structure. Its continuous probability output is a useful precedent for probabilistic/UQ-flavored outputs and for RS-driven CNN mapping, but it is a different physical variable at 9 km. We differentiate by regressing continuous ALT depth + 3D thermal column with calibrated per-cell UQ and cross-region transfer. Good to cite as 'DL + microwave RS for permafrost surface state' to frame the RS-driven landscape without threatening our niche.
- 파일: `references/02_alt_dl_mapping/donahue2023_nh_freeze_thaw_unet.pdf`

### 📄 `li2025_vit_rts_geoai` — [off-direction]
**A multi-scale vision transformer-based multimodal GeoAI model for mapping Arctic permafrost thaw** — Li, Wenwen; Hsu, Chia-Yu; Wang, Sizhe; Gu, Zhining; Yang, Yili; Rogers, Brendan M.; Liljedahl, Anna (2025, arXiv:2504.17822 (cs.CV), submitted April 23, 2025). arXiv:2504.17822
- **무엇**: Multi-scale vision-transformer multimodal GeoAI that detects/delineates Retrogressive Thaw Slumps (RTS) from satellite imagery; a modern permafrost-thaw DL paper but it maps thaw LANDFORMS, not ALT or subsurface thermal state.
- **우리와의 차별/활용**: Off our direction: it is a computer-vision landform-DETECTION task (RTS polygons), not ALT-in-cm regression, subsurface thermal structure, transfer, or UQ, and does no time-series forecasting. Relevant only as evidence that transformer-based GeoAI is active in permafrost; it neither pre-empts nor enables T1/T2. Do not benchmark against it; cite at most as adjacent context.
- 파일: `references/02_alt_dl_mapping/li2025_vit_rts_geoai.pdf`

### 📄 `pastick2015_nsp_alaska` — [선례]
**Distribution of near-surface permafrost in Alaska: Estimates of present and future conditions** — Pastick, N. J.; Jorgenson, M. T.; Wylie, B. K.; Nield, S. J.; Johnson, K. D.; Finley, A. O. (2015, Remote Sensing of Environment, v. 168, p. 301-315). doi:10.1016/j.rse.2015.07.019
- **무엇**: Statewide Alaska 30m near-surface-permafrost PRESENCE/ABSENCE map via Random Forest on terrain+climate covariates + field obs; a methodological precedent for Alaska RF permafrost mapping with a covariate stack very similar to ours (terrain + climate indices).
- **우리와의 차별/활용**: Pastick2015 predicts BINARY near-surface permafrost presence, not continuous ALT in cm and not subsurface thermal structure, and gives no cell-wise regression UQ or cross-region transfer. It is a methodological precedent (RF + terrain/climate covariates over Alaska) that legitimizes our covariate design but leaves our ALT-regression + transfer + UQ + 3D niche open. Does not touch T1/T2 forecasting.
- 파일: `references/02_alt_dl_mapping/pastick2015_nsp_alaska.pdf`

## 03 · 시계열 예측 (T1: forecasting) · physics-guided temporal  (5/10 PDF)

폴더: `references/03_alt_forecasting/`

### 🔗 `luo2022_qtp_thawdepth` — [경쟁자]
**Interannual and seasonal variations of permafrost thaw depth on the Qinghai-Tibetan Plateau: A comparative study using long short-term memory, convolutional neural networks, and random forest** — Luo et al. (2022, Science of The Total Environment, vol. 838, article 155886). doi:10.1016/j.scitotenv.2022.155886
- **무엇**: Directly compares LSTM/CNN/RF to estimate interannual and seasonal permafrost thaw depth on the QTP from meteorological series + in-situ ALT + geospatial predictors, showing CNN/LSTM with longer lagging windows beat RF for thaw-depth prediction. This is the single closest precedent to our T1 forecasting track.
- **우리와의 차별/활용**: PRE-EMPTS the core T1 idea (temporal DL for thaw depth) but ONLY on the Qinghai-Tibetan Plateau, and it is more spatiotemporal-estimation than true future forecasting with held-out future years. We differentiate by: (a) Alaska/CALM+borehole domain instead of QTP, (b) explicit multi-region TRANSFER (LORO / Alaska->other) which they do not do, (c) cell-wise UQ which they lack, and (d) joint spatial+temporal + shallow 3D well profiles rather than thaw depth alone. Cite as the paper we must beat/position against for T1.
- 링크(페이월/자동차단): https://doi.org/10.1016/j.scitotenv.2022.155886

### 📄 `rahaman2025_sequential_dl_thaw` — [경쟁자]
**Developing a Sequential Deep Learning Pipeline to Model Alaskan Permafrost Thaw Under Climate Change** — Rahaman, Addina (2025, arXiv:2510.06258 (physics.ao-ph; cs.LG)). arXiv:2510.06258
- **무엇**: NEW (Oct 2025) sequential DL pipeline benchmarking TCN/Transformer/Conv1DLSTM/GRU/BiLSTM to forecast yearly soil temperatures at multiple depths in Alaska from ERA5-Land + static geology + CMIP5 RCP future scenarios (GRU best); a direct pre-emption of our T1 time-series forecasting track.
- **우리와의 차별/활용**: This is the most dangerous new competitor for T1: it already does exactly the sequential-DL model bake-off (GRU/TCN/Transformer/BiLSTM) on Alaskan soil-T at multiple depths with ERA5-Land features and future (CMIP5 RCP) projection. To stay novel we must NOT re-run a plain LSTM/GRU vs Transformer bake-off. Remaining gaps to exploit: (a) it forecasts SOIL TEMPERATURE, not ALT-in-cm from historical ALT at CALM/well sites (our T1 target is different); (b) point/latitude-band predictions, no SPATIAL map or 4D volume (our T2); (c) no cell-wise UQ; (d) no cross-region TRANSFER (Alaska-only); (e) uses older CMIP5, we can use CMIP6. Position T1 as ALT-target forecasting + UQ + transfer, and T2 as genuinely spatiotemporal/4D.
- 파일: `references/03_alt_forecasting/rahaman2025_sequential_dl_thaw.pdf`

### 📄 `rahaman_northslope_ai_temps` — [경쟁자]
**Artificial intelligence for predicting arctic permafrost and active layer temperatures along the Alaskan North Slope** — Rahaman, Addina; and co-authors (Alaskan North Slope study) (2024, Earth Science Informatics, vol. 17, pp. 6055-6073). doi:10.1007/s12145-024-01486-1
- **무엇**: AI/deep-learning prediction of permafrost and active-layer TEMPERATURES along the Alaskan North Slope from reanalysis covariates; an earlier point-scale precedent for the same T1 sequential-prediction idea (paywalled Springer).
- **우리와의 차별/활용**: Point-scale temperature prediction along the North Slope, not spatial ALT-in-cm mapping, no cell-wise UQ, no cross-region transfer, Alaska-only. Together with Rahaman2025 it shows the same author group is actively occupying the Alaska T1 sequential-DL space. We differentiate identically: ALT-target (not temperature) forecasting at CALM/well sites WITH UQ and Alaska->other-region transfer, plus 4D spatial extension for T2. Paywalled: no open PDF found (Springer IDP redirect).
- 파일: `references/03_alt_forecasting/rahaman_northslope_ai_temps.pdf`

### 🔗 `zhang2022_qtp_alt_lstm_cnn_rf` — [경쟁자]
**Interannual and seasonal variations of permafrost thaw depth on the Qinghai-Tibetan Plateau: A comparative study using long short-term memory, convolutional neural networks, and random forest** — et al. (Science of the Total Environment) (2022, Science of the Total Environment, vol. 838, article 155886). doi:10.1016/j.scitotenv.2022.155886
- **무엇**: Head-to-head LSTM vs CNN vs RF for interannual ALT and seasonal thaw-depth prediction on the QTP with lagged air-temperature inputs; CNN/LSTM beat RF at longer lag times, temp-thaw lag up to ~32 days.
- **우리와의 차별/활용**: This is the most direct methodological PRE-EMPTION of our T1 (LSTM/CNN vs RF forecasting of ALT/thaw depth with lagged climate). It already shows deep temporal models beat RF once you feed lagged climate — a warning that our T1 could look derivative on method alone. We still differentiate: (a) region = Alaska/pan-Arctic not QTP, (b) we forecast at CALM/well sites with a formal transfer (Alaska->other-region) protocol they lack, (c) we add cell-wise UQ, (d) we can contrast with our own finding that under STATIC climatology RF ties DL, so the value-add is precisely the temporal signal they exploit. Paywalled — no direct PDF.
- 링크(페이월/자동차단): https://doi.org/10.1016/j.scitotenv.2022.155886

### 📄 `jia2020_pg_recurrent_graph` — [선례]
**Physics-Guided Recurrent Graph Networks for Predicting Flow and Temperature in River Networks** — Xiaowei Jia, Jared Willard, Anuj Karpatne, Jordan S. Read, Jacob A. Zwart, Michael Steinbach, Vipin Kumar (2020, arXiv:2009.12575 (physics.geo-ph); published version: Proceedings of the 2021 SIAM International Conference on Data Mining (SDM 2021), pp. 612-620, titled "Physics-Guided Recurrent Graph Model for Predicting Flow and Temperature in River Networks"). arXiv:2009.12575
- **무엇**: Recurrent GRAPH network for spatially connected temperature/flow prediction: physics-based pretraining plus a graph over river segments captures spatial interactions, beating standalone LSTM and process models by 24%/14%.
- **우리와의 차별/활용**: Extends the Read PGDL idea from independent sites to a SPATIAL GRAPH of stations, which is the natural bridge from our T1 (per-site LSTM) to a spatially-aware forecaster - a graph over CALM/borehole sites (or terrain-similarity edges) could share information for transfer to unmonitored cells. We differentiate by the physical process (heat + latent-heat freeze-thaw, not streamflow routing), by static/climatological covariates, and by adding UQ. Useful as the 'spatial recurrence' precedent that stops us over-claiming novelty on graph-coupled temperature forecasting. ENABLES a spatially-regularized T1 that partially addresses transfer.
- 파일: `references/03_alt_forecasting/jia2020_pg_recurrent_graph.pdf`

### 🔗 `ran2021_qtp_shallow_permafrost_cmip6` — [선례]
**Data-driven spatiotemporal projections of shallow permafrost based on CMIP6 across the Qinghai-Tibet Plateau at 1 km2 scale** — Youhua Ran, Xin Li, Guodong Cheng, et al. (2021, Advances in Climate Change Research, 12(6), 814-827 (Elsevier / KeAi, on behalf of National Climate Center)). doi:10.1016/j.accre.2021.08.009
- **무엇**: Physically-analytical + data-driven model projects QTP shallow permafrost area, MAGT and ALT for 1980-2100 at 1 km under 8-model CMIP6 SSP126/245/585 ensemble; the reference design for climate-scenario permafrost projection.
- **우리와의 차별/활용**: This is the canonical 'project permafrost to 2100 under CMIP6 SSP' template our T2 competes with (same Ran group as our Ran2022 benchmark). ENABLES our framing (shows the community values SSP-driven spatiotemporal projection) but is on QTP not Alaska, uses a physical/analytical (TTOP-like) model not deep learning, is 2D map+time not 3D volume, and reports ensemble spread rather than cell-wise learned UQ. We differentiate by DL spatiotemporal model, Alaska+transfer, 3D subsurface, and per-cell UQ.
- 링크(페이월/자동차단): https://www.sciencedirect.com/science/article/pii/S167492782100126X/pdfft?md5=&pid=1-s2.0-S167492782100126X-main.pdf

### 📄 `read2019_pgdl_lake_temp` — [선례]
**Process-Guided Deep Learning Predictions of Lake Water Temperature** — Jordan S. Read, Xiaowei Jia, Jared Willard, Alison P. Appling, Jacob A. Zwart, Samantha K. Oliver, Anuj Karpatne, Gretchen J. A. Hansen, Paul C. Hanson, William Watkins, Michael Steinbach, Vipin Kumar (2019, Water Resources Research, vol. 55, issue 11, pp. 9173-9190). doi:10.1029/2019WR024922
- **무엇**: The template physics-guided LSTM: an energy-conservation loss penalty plus pretraining on a process-based model's synthetic output lets an LSTM predict depth-resolved lake temperature profiles that stay physically consistent and generalize with sparse labels.
- **우리와의 차별/활용**: Direct methodological template for T1 (ALT/thermal time-series forecasting at CALM/well sites) and even mirrors our data situation: physics pretraining = pretrain on GIPL2 synthetic profiles, then fine-tune on sparse observed ALT/borehole temps, with an energy/heat-conservation penalty. Their sparse-label transfer story is exactly our Alaska->other-region transfer motivation. We differentiate on domain (frozen ground with latent heat + Stefan front vs open-water column), on the named-open-gap cell-wise UQ, and on quantifying transfer as a benchmark. PRE-EMPTS the generic 'physics-guided LSTM for temperature' novelty claim, so our T1 novelty must be permafrost-transfer + UQ, not the architecture itself.
- 파일: `references/03_alt_forecasting/read2019_pgdl_lake_temp.pdf`

### 🔗 `shi2025_cnnlstm_soil_temperature` — [맥락]
**A spatiotemporal CNN-LSTM deep learning model for predicting soil temperature in diverse large-scale regional climates** — See Science of the Total Environment 2025 (S0048969725005364) (2025, Science of the Total Environment, Vol. 968, Article 178901). doi:10.1016/j.scitotenv.2025.178901
- **무엇**: CNN-LSTM forecasts hourly near-surface soil temperature spatiotemporally across diverse US/Canada climate zones; an adjacent-field precedent for spatiotemporal soil-thermal DL that we can borrow architecture from.
- **우리와의 차별/활용**: CONTEXT/METHOD: shows CNN-LSTM (a ConvLSTM-family spatiotemporal model) works for gridded soil-temperature over large regions and diverse climates — supporting feasibility of our T2 backbone. But it is near-surface soil temperature (not deep permafrost/ALT), no future SSP projection, no subsurface 3D volume, no UQ, no transfer benchmark. We differentiate on permafrost ALT + 0-20 m volume, climate-scenario projection, transfer, and UQ. Mainly a source of architecture justification, not a competitor.
- 링크(페이월/자동차단): https://doi.org/10.1016/j.scitotenv.2025.178901

### 🔗 `wang2025_artificial_permafrost_table_hybrid` — [맥락]
**Enhancing artificial permafrost table predictions using integrated climate and ground temperature data: A case study from the Qinghai-Xizang highway** — (Cold Regions Sci Tech authors) (2024, Cold Regions Science and Technology, Volume 229, article 104341). doi:10.1016/j.coldregions.2024.104341
- **무엇**: Hybrid RF-LSTM-XGBoost model predicting the artificial permafrost table (thaw depth under engineered subgrade) along the Tuotuo River section of the Qinghai-Xizang Highway from climate + multi-depth/position ground-temperature data, with grid-search/CV hyperparameter tuning. Engineering-scale thaw-depth forecasting with an LSTM component.
- **우리와의 차별/활용**: Shows LSTM-in-hybrid for thaw-depth/permafrost-table prediction is established, but it is engineering subgrade (artificial permafrost table under a highway), QTP not Alaska, natural-ALT-irrelevant boundary conditions, no transfer or UQ, no spatial mapping. Weak overlap with our natural-ALT T1. Context citation confirming hybrid-DL thaw-depth prediction exists; our natural-tundra Alaska + transfer + UQ + spatial framing remains distinct.
- 링크(페이월/자동차단): https://doi.org/10.1016/j.coldregions.2024.104341

### 📄 `yurtsever2023_transformer_soiltemp` — [방법]
**A novel transformer-based approach for soil temperature prediction** — Yurtsever, Kucukmanisa, Kilimci (2023, arXiv (cs.LG)). arXiv:2311.11626
- **무엇**: Benchmarks five transformer forecasting architectures (Vanilla Transformer, Informer, Autoformer, Reformer, ETSformer) against deep-learning baselines for soil-temperature time-series forecasting at six FLUXNET stations, claiming SOTA. Not permafrost, but a direct methods menu for our temporal-transformer choice in T1.
- **우리와의 차별/활용**: Not permafrost (general FLUXNET agricultural/eco soil temperature) and no transfer/UQ/ALT — so it does NOT pre-empt our application at all. Value is purely as a METHOD reference: it tells us which transformer variants (Informer/Autoformer/ETSformer) to benchmark for T1 and provides a baseline design. We differentiate trivially by domain (permafrost ALT), region (Alaska), transfer, and UQ. Cite to justify architecture selection for the forecasting track.
- 파일: `references/03_alt_forecasting/yurtsever2023_transformer_soiltemp.pdf`

## 04 · 시공간/4D (T2: 3D+time)  (4/6 PDF)

폴더: `references/04_spatiotemporal_4d/`

### 📄 `kriuk2025_panarctic_hybrid_risk` — [경쟁자]
**Hybrid Physics-ML Framework for Pan-Arctic Permafrost Infrastructure Risk at Record 2.9-Million Observation Scale** — Boris Kriuk (2025, arXiv (stat.ML; cs.LG)). arXiv:2510.02189
- **무엇**: Stacked ensemble (RF + HistGBM + ElasticNet) with physical adjustment factors over 2.9M annual obs / 171,605 Arctic-Russia locations (2005-2021), projecting permafrost-fraction decline and infrastructure risk to 10-year horizons under RCP2.6/4.5/8.5 with spatially-explicit ensemble-std UQ. Combines physics+ML+temporal projection+UQ at huge scale.
- **우리와의 차별/활용**: Notable because it ALREADY pairs future projection with spatially-explicit UQ (ensemble std) at pan-Arctic scale — partially pre-empts our 'UQ is an open gap' claim. BUT target is permafrost FRACTION / infrastructure risk class, NOT ALT in cm or 3D thermal structure; region is Arctic Russia not Alaska; it uses tabular ensembles (no sequence DL, no transfer learning — it explicitly notes cross-validation not transfer); UQ is raw ensemble spread, not calibrated. We differentiate by ALT/3D-thermal target, Alaska->other transfer benchmark, calibrated (coverage-checked) UQ, and shallow 3D well profiles. Cite to pre-empt reviewer 'UQ already done' — show ours is calibrated + ALT-specific + transfer-tested.
- 파일: `references/04_spatiotemporal_4d/kriuk2025_panarctic_hybrid_risk.pdf`

### 🔗 `ieki2025_neuralkriging_japan` — [방법]
**Deep learning-based three-dimensional terrestrial temperature modeling throughout Japan incorporating multiple crustal properties and spatial correlation with an application to critical point distribution** — Yusei Ieki, Katsuaki Koike, Taiki Kubo (2025, Geothermics, Vol. 131, Art. 103403 (Elsevier)). doi:10.1016/j.geothermics.2025.103403
- **무엇**: LEAD CORRECTED / 'Suzuki' NOT FOUND. The real 2025 Japan 'neural kriging' paper is by Ieki, Koike & Kubo (Geothermics), combining DNN + neural kriging for 3D terrestrial (crustal/geothermal) temperature modeling with spatial correlation and deep extrapolation. No author named Suzuki; topic is geothermal crustal temperature, NOT permafrost. DOI verified via Crossref.
- **우리와의 차별/활용**: Our note's 'Suzuki 2025 neural kriging Japan' does not exist as stated — treat as a mis-attribution and cite Ieki et al. 2025 instead. Relevant as a 3D-volume METHOD precedent (neural kriging = DL spatial encoder + kriging decoder for extrapolating temperature to depth), directly analogous to our shallow-3D thermal-structure engine and a candidate baseline vs our GBM/IDW conditioning field. It is geothermal, not permafrost/ALT, so it does not pre-empt T2; WE differentiate by permafrost domain + Alaska transfer + UQ + adding the TIME axis (4D).
- 링크(페이월/자동차단): https://doi.org/10.1016/j.geothermics.2025.103403

### 📄 `liu2024_polar_ice_layers_pignn` — [방법]
**Learning Spatio-Temporal Patterns of Polar Ice Layers With Physics-Informed Graph Neural Network** — Zesheng Liu, Maryam Rahnemoonfar (2024, arXiv (cs.LG) preprint). arXiv:2406.15299
- **무엇**: Physics-informed GraphSAGE+LSTM GNN learns spatiotemporal ice-layer thickness and predicts DEEP layers from SHALLOW layers using MAR weather-model physical node features; an analogous cryosphere 4D 'shallow->deep + time' method.
- **우리와의 차별/활용**: ENABLES our T2 as a transferable methodology: it is the cleanest analog of 'predict deep subsurface structure from shallow observations across space and time' with physics features on irregular points — exactly our 3D-volume-from-surface problem, but for ice sheets not permafrost. Not a competitor (different variable/domain). We can adopt the shallow->deep spatiotemporal-GNN framing and MAR-style physical-feature injection for our 0-20 m permafrost thermal volume, adding UQ and transfer which this paper lacks.
- 파일: `references/04_spatiotemporal_4d/liu2024_polar_ice_layers_pignn.pdf`

### 📄 `mcmillen2026_3d_to_2d_projection_thaw` — [선례]
**Preserving Vertical Structure in 3D-to-2D Projection for Permafrost Thaw Mapping** — Justin McMillen, Robert Van Alphen, Taha Sadeghi Chorsi, Jason Shabaga, Mel Rodgers, Rocco Malservisi, Timothy Dixon, Yasin Yilmaz (2026, arXiv (2603.16788)). arXiv:2603.16788
- **무엇**: Point Transformer V3 encoder + learned-height-embedding projection decoder that turns 3D UAV lidar point clouds into 2D thaw-depth grids while preserving vertical (ground/understory/canopy) structure, in interior-Alaska boreal forest.
- **우리와의 차별/활용**: Closest thing to our 3D->2D structure-preserving idea and the most relevant precedent for the volume/vertical part of T2. BUT its 'vertical structure' is ABOVE-ground vegetation, projected to a 2D thaw-depth map — it does not model the SUBSURFACE 0-20 m thermal volume, has NO time dimension (no 4D), NO transfer across regions, NO UQ. We differentiate by owning the subsurface thermal volume (not canopy), adding TIME (4D), transfer, and cell-wise UQ. Good to cite as 'vertical-structure-aware projection exists, but for canopy not soil column.'
- 파일: `references/04_spatiotemporal_4d/mcmillen2026_3d_to_2d_projection_thaw.pdf`

### 🔗 `review2024_dl_spatiotemporal_earthsystem` — [맥락]
**Deep learning for spatiotemporal forecasting in Earth system science: a review** — Review article (International Journal of Digital Earth, Taylor & Francis) (2024, International Journal of Digital Earth, Vol. 17, Issue 1, Article 2391952). doi:10.1080/17538947.2024.2391952
- **무엇**: Reviews 69 studies of deep learning (ConvLSTM, ST-LSTM, SimVP, transformers) for spatiotemporal forecasting across climate/hydrology/ocean; a landscape/context anchor and architecture menu for our T2.
- **우리와의 차별/활용**: CONTEXT: establishes the spatiotemporal-DL toolbox (ConvLSTM/ST-LSTM/SA-LSTM/SimVP, video-prediction transfer to earth science) and confirms permafrost is under-represented in this literature — an opening for us. Not a competitor. Useful to cite for (a) justifying model choices in T2 and (b) arguing the gap that no reviewed study does 3D-volume+time permafrost with transfer+UQ.
- 링크(페이월/자동차단): https://www.tandfonline.com/doi/pdf/10.1080/17538947.2024.2391952

### 📄 `sitzmann2020_siren` — [방법]
**Implicit Neural Representations with Periodic Activation Functions (SIREN)** — Vincent Sitzmann, Julien N. P. Martel, Alexander W. Bergman, David B. Lindell, Gordon Wetzstein (2020, NeurIPS 2020 (arXiv:2006.09661)). arXiv:2006.09661
- **무엇**: Sinusoidal-activation MLPs that represent continuous signals and, crucially, their spatial/temporal derivatives accurately, enabling neural solutions to PDE boundary-value problems (Poisson, Helmholtz, wave, Eikonal).
- **우리와의 차별/활용**: The implicit-neural-field backbone for a continuous 3D/4D thermal representation T(x,y,z,t): because SIREN gives clean derivatives, one can impose the heat/Stefan PDE residual directly on a coordinate network - a coordinate-based alternative to voxel grids for the subsurface volume. HOWEVER this is a caution flag: our own 3D neural field already LOST to GBM/IDW and was killed, so a naive SIREN field would likely repeat that. We differentiate/justify only if we use SIREN as a physics-constrained interpolator with PDE + UQ, conditioned on the GBM field, rather than a from-scratch fit. Method/context for T2, and a documented failure-mode warning.
- 파일: `references/04_spatiotemporal_4d/sitzmann2020_siren.pdf`

## 05 · 불확실성(UQ) · 전이(Transfer) — 우리 방어축  (10/13 PDF)

폴더: `references/05_uq_transfer/`

### 📄 `lou2025_geoconformal` — [경쟁자]
**GeoConformal Prediction: A Model-Agnostic Framework for Measuring the Uncertainty of Spatial Prediction** — Xiayin Lou, Peng Luo, Liqiu Meng (2025, Annals of the American Association of Geographers, vol. 115, issue 8, pp. 1971-1998 (arXiv preprint 2412.08661)). arXiv:2412.08661
- **무엇**: Extends conformal prediction with geographic weighting to handle spatial heterogeneity/covariate shift between calibration and prediction locations; demonstrated on spatial regression (XGBoost house prices) and interpolation. Bridges exactly our two axes: UQ that is aware of the spatial-transfer problem.
- **우리와의 차별/활용**: This is the most dangerous overlap because it fuses spatial-transfer awareness WITH conformal UQ - our two 'defensible axes' in one framework. We differentiate by (a) applying it to permafrost ALT and shallow 3D thermal fields (a new, higher-noise, covariate-bottlenecked domain vs. house prices), (b) a genuine inter-region transfer test (Alaska->other Arctic) rather than within-city interpolation, and (c) integrating with a physics forward model (GIPL2) and AOA masking. Method is portable to T1/T2 but the paper is purely static 2D.
- 파일: `references/05_uq_transfer/lou2025_geoconformal.pdf`

### 📄 `singh2024_conformal_eo` — [경쟁자]
**Uncertainty quantification for probabilistic machine learning in earth observation using conformal prediction** — Geethen Singh, Glenn Moncrieff, Zander Venter, Kerry Cawse-Nicholson, Jasper Slingsby, Tamara B. Robinson (2024, Scientific Reports 14:16166 (2024); preprint arXiv:2401.06421). arXiv:2401.06421
- **무엇**: Model-agnostic conformal prediction for Earth observation giving statistically valid prediction sets/intervals without model or training-data access; demonstrated on canopy height regression (GEDI) and land cover, noting only ~22% of EO datasets carry any uncertainty. The strongest direct precedent for our cell-wise UQ axis.
- **우리와의 차별/활용**: PRE-EMPTS 'conformal UQ for environmental raster mapping' in general, but never touches permafrost/subsurface/ALT and uses standard exchangeable conformal that breaks under the spatial covariate shift central to our Alaska->other-region setting. We differentiate by (a) permafrost ALT + shallow 3D thermal target, (b) spatially-adaptive/transfer-aware conformal (localized or AOA-conditioned) validated under LORO, and (c) benchmarking interval calibration (PICP/interval width) against quantile-GBM and deep ensembles on our covariate-bottlenecked data. Method transfers to T1 (per-horizon forecast intervals) and T2.
- 파일: `references/05_uq_transfer/singh2024_conformal_eo.pdf`

### 📄 `groenke2023_bayesian_heat` — [선례]
**Investigating the thermal state of permafrost with Bayesian inverse modeling of heat transfer** — Groenke, B.; Langer, M.; Nitzbon, J.; Westermann, S.; Gallego, G.; Boike, J. (2023, The Cryosphere, 17, 3505-3533). doi:10.5194/tc-17-3505-2023
- **무엇**: 1D Bayesian inverse heat-transfer model that infers permafrost thermal state WITH posterior uncertainty from borehole temperatures (with latent-heat/phase change); the clearest permafrost-domain precedent for principled UQ, directly relevant to our cell-wise UQ novelty.
- **우리와의 차별/활용**: Groenke2023 is 1D per-borehole Bayesian inversion (point-scale posterior), NOT a spatial 2D/3D map, and does no cross-region transfer or ML upscaling. It is the key UQ PRECEDENT that both validates and bounds our UQ claim: we must show our contribution is SPATIAL cell-wise UQ over a mapped domain + transfer, not single-site posterior inference. It enables T1/T2 conceptually (their 2024 JGR follow-up reconstructs historical climate from boreholes), so we differentiate by spatial coverage + data-driven upscaling rather than site-by-site physical inversion.
- 파일: `references/05_uq_transfer/groenke2023_bayesian_heat.pdf`

### 📄 `koven_review2025_ml_permafrost` — [맥락]
**Machine learning-based prediction of permafrost degradation and its implications on geotechnical infrastructure: a comprehensive review** — (review authors) (2025, AI in Civil Engineering, vol. 4, article 28 (SpringerOpen)). doi:10.1007/s43503-025-00080-8
- **무엇**: 2025 review of ML for permafrost degradation that EXPLICITLY names (a) permafrost transfer-learning being under-evaluated and (b) most ML lacking embedded UQ as open gaps; the citation that authorizes our two core novelty claims.
- **우리와의 차별/활용**: This is the review we cite to establish that TRANSFER and cell-wise UQ are named open gaps -- our positioning anchor, not a competitor. It confirms Alaska-trained models generalize poorly to Siberia/QTP and that single-point ML lacks reliability estimates. We differentiate by actually delivering the LORO transfer benchmark + quantile UQ it calls for. Enables framing of both T1/T2 as filling named gaps.
- 파일: `references/05_uq_transfer/koven_review2025_ml_permafrost.pdf`

### 📄 `linnenbrink2024_knndm` — [방법]
**kNNDM CV: k-fold nearest-neighbour distance matching cross-validation for map accuracy estimation** — Jan Linnenbrink, Carles Milà, Marvin Ludwig, Hanna Meyer (2024, Geoscientific Model Development, Volume 17, Issue 15, pages 5897-5912). doi:10.5194/gmd-17-5897-2024
- **무엇**: Scalable k-fold version of NNDM (matches ECDF of test-train vs prediction-train NN distances) that keeps honest transfer error estimation while cutting cost from days to minutes on thousands of clustered points. The practical CV protocol for our transfer benchmark.
- **우리와의 차별/활용**: ENABLES the transfer axis at our data scale (many CALM/borehole site-years, strongly clustered). Directly usable off-the-shelf (CAST R package). We differentiate by embedding kNNDM as the scoring backbone of a permafrost transfer benchmark and by combining it with AOA masking and cell-wise UQ; the method paper itself has no permafrost/thermal/time application. Supports honest spatial folds for T2.
- 파일: `references/05_uq_transfer/linnenbrink2024_knndm.pdf`

### 🔗 `lu2022_qrf_dsm` — [방법]
**Quantile regression as a generic approach for estimating uncertainty of digital soil maps produced from machine-learning** — Qiuyuan Lu, Songchao Chen, Bifeng Hu, et al. (2021, Environmental Modelling & Software, Volume 144, Article 105139). doi:10.1016/j.envsoft.2021.105139
- **무엇**: Establishes quantile regression / Quantile Regression Forests as the standard, model-generic route to cell-wise prediction intervals (PIW maps) and interval-coverage (PICP) evaluation in environmental raster mapping. The baseline UQ method our conformal approach must beat or complement.
- **우리와의 차별/활용**: ENABLES our UQ baseline: quantile-GBM/QRF gives per-cell intervals cheaply and pairs naturally with the GBM engine we already use (GBM ties DL here). We differentiate by adding coverage-guaranteed conformal calibration on top (quantile intervals are not coverage-valid under our transfer shift) and by benchmarking QRF-vs-conformal-vs-deep-ensemble calibration specifically on permafrost ALT under Alaska->other-region transfer. Provides per-quantile intervals directly reusable for T1 forecasting horizons.
- 링크(페이월/자동차단): https://doi.org/10.1016/j.envsoft.2021.105139

### 🔗 `ludwig2023_transferability` — [선례]
**Assessing and improving the transferability of current global spatial prediction models** — Marvin Ludwig, Alvaro Moreno-Martinez, Norbert Hölzel, Edzer Pebesma, Hanna Meyer (2023, Global Ecology and Biogeography 32(3):356-368). doi:10.1111/geb.13635
- **무엇**: Directly measures how poorly global ML maps transfer to unsampled regions using the AOA/DI framework and proposes sampling/feature strategies to improve transferability. The closest methodological precedent for what we call our 'permafrost transfer learning benchmark'.
- **우리와의 차별/활용**: PRE-EMPTS the generic claim of novelty for 'assessing spatial transferability' - this already does it for global ecological variables. We must differentiate on domain and mechanism: permafrost ALT/thermal (not vegetation/soil), an explicit source(Alaska)->target(other Arctic) protocol benchmarked against Ran2022/GIPL2/CCI, and coupling transfer assessment with a physics forward model (GIPL2) plus calibrated cell-wise UQ - none of which Ludwig does. Static; context for T2 spatial generalization.
- 링크(페이월/자동차단): https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/geb.13635

### 🔗 `ma2024_tl_env_rs_review` — [맥락]
**Transfer learning in environmental remote sensing** — Yuchi Ma, Shuo Chen, Stefano Ermon, David B. Lobell (2024, Remote Sensing of Environment 301:113924). doi:10.1016/j.rse.2023.113924
- **무엇**: First systematic review of transfer learning in environmental remote sensing (1,676 papers, 2017-2022): defines domain-shift types and five TL techniques (instance/feature/parameter/relational/adversarial), documenting ~10x growth. The framing reference that legitimizes and organizes our transfer-learning novelty axis.
- **우리와의 차별/활용**: Confirms transfer learning is established generally in environmental RS (so we cannot claim 'TL is new'), but the review shows it is dominated by classification/land-cover and imagery tasks - not tabular climate+terrain regression to a subsurface geophysical target like ALT/permafrost, and not paired with spatial-CV honesty (AOA/kNNDM) or calibrated UQ. We differentiate by positioning our contribution as TL for a physics-linked regression target under sparse clustered ground truth, scored with distance-matched CV. Method vocabulary (adversarial/feature-based DA) is directly reusable for T1/T2 climate-driven extrapolation.
- 링크(페이월/자동차단): https://doi.org/10.1016/j.rse.2023.113924

### 📄 `meyer2021_aoa` — [선례]
**Predicting into unknown space? Estimating the area of applicability of spatial prediction models** — Hanna Meyer, Edzer Pebesma (2021, Methods in Ecology and Evolution 12(9):1620-1633). arXiv:2005.07939
- **무엇**: Foundational method defining the Area of Applicability (AOA) via a predictor-importance-weighted dissimilarity index (DI): masks map cells where feature-space distance to training data exceeds a threshold, so CV error no longer holds. This is the canonical spatial-generalization tool our transfer axis must adopt and cite.
- **우리와의 차별/활용**: PRE-EMPTS the naive framing of 'cell-wise trust map' but ENABLES our transfer axis: AOA gives a binary in/out mask, not a calibrated numeric uncertainty. We differentiate by (a) applying AOA specifically to permafrost ALT (never done in this literature) and (b) pairing AOA with a calibrated cell-wise UQ (conformal/quantile) to turn the binary mask into a continuous prediction-interval field, and (c) using Alaska->other-region LORO to empirically validate that AOA predicts the observed transfer degradation we already measured (108.5->87.3cm). Neutral to T1/T2 (it is static/spatial).
- 파일: `references/05_uq_transfer/meyer2021_aoa.pdf`

### 📄 `meyer2022_globalmaps_aoa` — [맥락]
**Machine learning-based global maps of ecological variables and the challenge of assessing them** — Hanna Meyer, Edzer Pebesma (2022, Nature Communications, volume 13, article number 2208 (2022)). doi:10.1038/s41467-022-29838-9
- **무엇**: High-visibility argument that ML global maps built on clustered/sparse reference data extrapolate silently; formalizes the DI/AOA and calls for graying out unreliable regions. Frames the exact motivation (clustered CALM sites, poor coverage outside training regions) for our transfer + cell-wise UQ contribution.
- **우리와의 차별/활용**: Provides the 'why it matters' citation and the AOA concept at Nature Communications visibility, but is a general ecological-mapping viewpoint with no permafrost, no thermal/ALT target, no time dimension. We differentiate by instantiating this critique quantitatively for permafrost ALT with a real held-out region transfer benchmark (Ran2022/GIPL2/CCI comparison) rather than a qualitative call-to-action. Context for both T1/T2 but does not touch time.
- 파일: `references/05_uq_transfer/meyer2022_globalmaps_aoa.pdf`

### 📄 `mila2022_nndm` — [방법]
**Nearest neighbour distance matching Leave-One-Out Cross-Validation for map validation** — Carles Milà, Jorge Mateu, Edzer Pebesma, Hanna Meyer (2022, Methods in Ecology and Evolution 13(6):1304-1316). doi:10.1111/2041-210X.13851
- **무엇**: Introduces NNDM LOO-CV: matches the test-to-train nearest-neighbour distance distribution to the prediction-to-train distribution so CV error honestly reflects the true prediction (interpolation vs extrapolation) task. The correct way to score our Alaska->other-region transfer instead of leaky random CV.
- **우리와의 차별/활용**: ENABLES a rigorous transfer benchmark: our current LORO/spatial split can be upgraded to NNDM/kNNDM to get honest, publishable transfer error estimates. We differentiate by being the first to apply distance-matched CV to permafrost ALT and by reporting how much random-CV over-optimism inflated prior pan-Arctic ML claims (Ran2022). Static/spatial; relevant to T2 spatial folds but not the time axis.
- 파일: `references/05_uq_transfer/mila2022_nndm.pdf`

### 📄 `nyland2023_transparent_earth` — [맥락]
**The Transparent Earth: A Multimodal Foundation Model for the Earth's Subsurface** — et al. (Los Alamos subsurface-FM group) (2025, arXiv (cs.LG; physics.geo-ph); accepted at NeurIPS 2025 AI4Science Workshop). arXiv:2509.02783
- **무엇**: BONUS (encountered). Multimodal foundation model for Earth's subsurface supporting in-context learning: generates predictions from zero or arbitrarily many observations across any subset of modalities. Predecessor/companion to In-Context Earth (2605.16665). 2025. Author list not verified.
- **우리와의 차별/활용**: Foundation-model precedent for multimodal subsurface prediction with in-context conditioning — the paradigm our transfer+UQ pitch competes against at the framing level. Not permafrost/ALT specific and no explicit 4D-to-2100 projection. Relevant only as context / a baseline that In-Context Earth already beats; verify authors before citing.
- 파일: `references/05_uq_transfer/nyland2023_transparent_earth.pdf`

### 📄 `omalley2026_incontext_subsurface_temp` — [방법]
**In-context learning enables continental-scale subsurface temperature prediction from sparse local observations** — O'Malley, Johnson, Santos, Lara, Malusa, Srikishan, Kath, Mazumder, Mehana, Coblentz, DeBardeleben, Lawrence, Viswanathan (2026, arXiv (cs.LG / physics.geo-ph)). arXiv:2605.16665
- **무엇**: Foundation-model / in-context-learning approach that predicts continental-scale subsurface temperature from sparse local borehole observations (contiguous US), adapting to new locations without retraining. A modern TRANSFER-learning precedent for subsurface thermal fields.
- **우리와의 차별/활용**: Directly relevant to our TRANSFER novelty: shows in-context learning generalizing subsurface temperature from sparse local data to a continent — a template for Alaska->other-region transfer. BUT it is spatial interpolation/extrapolation of a static-ish thermal field (not temporal ALT forecasting), contiguous-US (not permafrost/Arctic), and does not target ALT or seasonal thaw. We differentiate by permafrost domain, ALT target, temporal forecasting, and permafrost-specific transfer benchmark. Best-in-class METHOD to borrow (in-context/foundation-model transfer) while the application gap (permafrost ALT) stays open — this could actually strengthen our transfer track design.
- 파일: `references/05_uq_transfer/omalley2026_incontext_subsurface_temp.pdf`

## 06 · Physics-informed / Operator learning (물리결합)  (8/11 PDF)

폴더: `references/06_physics_ml/`

### 📄 `jafarov2012_gipl2_alaska` — [벤치마크]
**Numerical modeling of permafrost dynamics in Alaska using a high spatial resolution dataset** — Jafarov, E. E.; Marchenko, S. S.; Romanovsky, V. E. (2012, The Cryosphere, 6, 613-624). doi:10.5194/tc-6-613-2012
- **무엇**: The canonical GIPL2 transient forward physics model applied to all-Alaska (1km, multilayer soil column, monthly air-T/precip forcing) producing ground temperature profiles + ALT; our forward-physics benchmark and the incumbent process model we position observation-based interpolation against.
- **우리와의 차별/활용**: GIPL2 is a FORWARD climate-driven simulation (boreholes used only for validation), requires prescribed soil thermal properties/water content per class, and is deterministic (no cell-wise UQ). We differentiate as OBSERVATION-BASED interpolation with UQ and cross-region transfer, and can use GIPL2 as (a) a head-to-head baseline and (b) a synthetic pretraining corpus (PI-LSTM-style) for our forecasting tracks. It enables T2 as a physics reference but does not itself do data-driven UQ/transfer.
- 파일: `references/06_physics_ml/jafarov2012_gipl2_alaska.pdf`

### 📄 `piml2025_ne_china_permafrost_extent` — [경쟁자]
**A physics-informed machine learning (PIML) framework for projecting 21st-century permafrost extent in Northeast China** — EGUsphere preprint 2025 (egusphere-2025-4544) (2025, EGUsphere (Copernicus) preprint, in review). doi:10.5194/egusphere-2025-4544
- **무엇**: PIML couples a TTOP permafrost model, observed land-use change, and CMIP6 to project permafrost extent to 2100 (>90% loss under SSP5-8.5) in NE China; another climate-scenario permafrost-projection competitor blending physics + ML.
- **우리와의 차별/활용**: COMPETITOR for the T2 'projection' concept and the physics-ML branding, showing PIML + CMIP6 permafrost projection is an active 2025 theme. BUT it targets permafrost EXTENT (presence/absence) not continuous ALT or a 3D thermal volume, is NE China not Alaska, is 2D map+time, and shows no cell-wise UQ or cross-region transfer benchmark. We differentiate by continuous ALT + 3D subsurface + time, Alaska/transfer, and calibrated per-cell UQ rather than a binary extent map.
- 파일: `references/06_physics_ml/piml2025_ne_china_permafrost_extent.pdf`

### 📄 `aljubran2024_interpignn` — [방법]
**Thermal Earth model for the conterminous United States using an interpolative physics-informed graph neural network (InterPIGNN)** — Aljubran, Mohammad J.; Horne, Roland N. (2024, arXiv:2403.09961; published in Geothermal Energy 12, 25 (2024), SpringerOpen). arXiv:2403.09961
- **무엇**: Physics-informed GRAPH neural network that interpolates sparse borehole bottomhole temperatures into a 3D temperature-at-depth volume (0-7km) for the US by softly enforcing Fourier's conductive-heat law; the structural method precedent for our 3D PDE-constrained thermal-field idea.
- **우리와의 차별/활용**: InterPIGNN is a geothermal (deep, warm) steady-state conductive interpolation with no cryosphere phase-change/latent-heat term, no 0C permafrost base target, and no transfer or time dimension. It is a METHOD precedent we cite for the PDE-constrained interpolation idea. We differentiate by (a) applying/transferring the PDE-neural-field to cold 0C-target permafrost with latent-heat, (b) cell-wise UQ, (c) shallow 0-20m focus. RELEVANCE CAVEAT: our own 3D neural field LOST to GBM/IDW and was killed, so InterPIGNN now enables at most a physics-regularization ablation, not our main line -- treat as method precedent, borderline off_direction for the current GBM-conditioning-field engine.
- 파일: `references/06_physics_ml/aljubran2024_interpignn.pdf`

### 🔗 `chen2023_pi_lstm_gipl2_qtp` — [선례]
**Multisite evaluation of physics-informed deep learning for permafrost prediction in the Qinghai-Tibet Plateau** — et al. (Cold Regions Science and Technology) (2023, Cold Regions Science and Technology, vol. 216, article 104009). doi:10.1016/j.coldregions.2023.104009
- **무엇**: PI-LSTM: LSTM pretrained on GIPL2 physical-model output then fine-tuned on borehole ground-temperature profiles; multisite QTP evaluation shows better generalization/transferability/efficiency than plain LSTM or GIPL2 alone.
- **우리와의 차별/활용**: This is the key PRECEDENT/near-competitor for both our physics-ML story and our transfer framing, and it ENABLES T1 (time-series ground-temp/ALT prediction) — but on the Qinghai-Tibet Plateau, at borehole POINTS (1D profiles), not spatial ALT maps. It couples GIPL2 (which we also cite) as a physics prior — mirrors our 'GIPL2 forward physics' benchmark. We differentiate by: (a) Alaska/pan-Arctic not QTP, (b) 2D spatial mapping + shallow 3D volume, not single-column profiles, (c) explicit cross-REGION transfer benchmark (LORO) and cell-wise UQ rather than cross-site fine-tuning. Not open access — flag PDF as paywalled.
- 링크(페이월/자동차단): https://doi.org/10.1016/j.coldregions.2023.104009

### 🔗 `koric2023_deeponet_heat` — [방법]
**Data-driven and physics-informed deep learning operators for solution of heat conduction equation with parametric heat source** — Seid Koric, Diab W. Abueidda (2023, International Journal of Heat and Mass Transfer, vol. 203, article 123809). doi:10.1016/j.ijheatmasstransfer.2022.123809
- **무엇**: Head-to-head data-driven vs physics-informed DeepONet for the (Poisson) heat conduction equation with a spatially varying parametric source; near-instant parametric solves, orders of magnitude faster than numerical solvers.
- **우리와의 차별/활용**: DeepONet is the alternative operator backbone to FNO for our thermal surrogate: its branch/trunk split naturally ingests a variable forcing function (branch) and queries arbitrary (x,z,t) points (trunk), which fits irregular borehole/CALM sampling better than FNO's regular grid. We adopt the trunk-net query idea for continuous-depth thermal profiles. We differentiate by adding time (transient trunk) for T2, real climate forcing, and UQ. Not open access (paywalled), no arXiv preprint located. ENABLES a mesh-free continuous 3D/4D thermal field; context/method, not a permafrost competitor.
- 링크(페이월/자동차단): https://doi.org/10.1016/j.ijheatmasstransfer.2022.123809

### 📄 `li2020_fno` — [방법]
**Fourier Neural Operator for Parametric Partial Differential Equations** — Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar (2020, arXiv (cs.LG); published at ICLR 2021). arXiv:2010.08895
- **무엇**: Foundational Fourier Neural Operator: learns a resolution-invariant mapping between function spaces (parameter field -> PDE solution) via spectral convolutions, up to 1000x faster than classical solvers with zero-shot super-resolution.
- **우리와의 차별/활용**: The leading candidate operator-learning backbone for a fast 4D thermal surrogate: map a forcing/property field (surface temperature, snow, soil parameters) to the subsurface thermal solution over a grid, amortizing GIPL2-style physics. We differentiate by conditioning on real ERA5-Land + terrain covariates and by attaching cell-wise UQ (an open gap in the 2025 review), which vanilla FNO lacks. Risk given our findings: FNO shines when there is a genuine field-to-field PDE map with rich covariates; our covariate-information bottleneck means a static-feature FNO may still tie GBM. Best positioned as the ENABLER of T2 4D volumetric forecasting to 2100, not T1 point forecasting.
- 파일: `references/06_physics_ml/li2020_fno.pdf`

### 🔗 `liu2023_pilstm_permafrost_gipl2` — [선례]
**Multisite evaluation of physics-informed deep learning for permafrost prediction in the Qinghai-Tibet Plateau** — Yuandong Liu, Youhua Ran, Xin Li, Tao Che, Tonghua Wu (2023, Cold Regions Science and Technology, vol. 216, art. 104009). doi:10.1016/j.coldregions.2023.104009
- **무엇**: PI-LSTM pretrained on GIPL2 forward-model output then fine-tuned on borehole ground temperatures predicts vertical-profile permafrost temperature over time, beating LSTM and GIPL2 alone; a transfer + physics-informed time-series precedent.
- **우리와의 차별/활용**: Strong PRECEDENT/METHOD for our T1 and for the physics-ML angle: it already does (a) GIPL2->DL pretraining and (b) site-to-site transfer via fine-tuning — two things on our roadmap. BUT it is QTP not Alaska, point/borehole vertical-profile in time (not a spatial 2D/3D field), predicts ground temperature not ALT, and has no explicit cell-wise UQ. We differentiate by ALT + spatial 3D field, Alaska->other-region transfer as a benchmark (not just per-site fine-tune), and calibrated cell-wise UQ. Also useful as the design pattern to make our GBM 3D-conditioning field physics-informed.
- 링크(페이월/자동차단): https://doi.org/10.1016/j.coldregions.2023.104009

### 📄 `madir2024_pinn_phasechange` — [방법]
**Physics Informed Neural Networks for heat conduction with phase change** — Bahae-Eddine Madir, Francky Luddens, Corentin Lothode, Ionut Danaila (2024, arXiv preprint (2410.14216); published in International Journal of Heat and Mass Transfer, 2025). arXiv:2410.14216
- **무엇**: PINN strategies for liquid-solid phase-change heat conduction (Stefan) that specifically tackle the learning difficulty at the discontinuous phase-change interface, benchmarked against finite-difference solvers.
- **우리와의 차별/활용**: Directly relevant recent recipe for the numerical pain point we would hit: gradient discontinuity at the freeze-thaw interface. Their interface-handling tricks (loss weighting / interface tracking near the front) are adoptable for a permafrost latent-heat term. We differ by targeting real ground with heterogeneous soil columns and climatological forcing rather than a clean canonical Stefan benchmark, and by needing it at scale over a 2D map (thousands of columns) rather than a single simulation. ENABLES the physics-guided branch of T2; not a competitor since it never touches permafrost or geospatial transfer.
- 파일: `references/06_physics_ml/madir2024_pinn_phasechange.pdf`

### 📄 `pilyugina2023_pinn_permafrost_risk` — [선례]
**Assessing the Risk of Permafrost Degradation with Physics-Informed Machine Learning** — Pilyugina, Chernikov, Zaytsev, Bulkin, Burnaev, et al. (2023, arXiv (physics.geo-ph), submitted October 4, 2023). arXiv:2310.02525
- **무엇**: (2026-09-29 정정, 본문 확인) Kudryavtsev 모델을 토양 4종 초기값으로 돌려 얻은 ALT·MAGT 값을 기후·식생 자료와 함께 CatBoost 의 입력 특징으로 쓴다. 손실에 물리 항은 없다. ALT(cm, 2,729건)와 MAGT(°C, 961건)를 직접 예측한다. 검증은 시간 분할(2013년 이전 학습)과 무작위 5-fold 다. 이전 기재(열방정식 정칙화, 대상은 위험 지표)는 초록 표현을 따른 오분류였다.
- **우리와의 차별/활용**: Establishes heat-equation-constrained temporal forecasting for permafrost degradation, so 'physics-informed temporal permafrost DL' is already claimed (reinforces that our novelty cannot be 'PINN for permafrost'). BUT target is a generic degradation-risk metric (not ALT/temperature profiles), no cell-wise calibrated UQ, no explicit region transfer benchmark, and no shallow-3D thermal reconstruction. We differentiate through concrete ALT/thermal targets, transfer, and calibrated UQ. Method/context citation for the physics-constraint design of T1/T2.
- 파일: `references/06_physics_ml/pilyugina2023_pinn_permafrost_risk.pdf`

### 📄 `wang2020_stefan_pinn` — [방법]
**Deep learning of free boundary and Stefan problems** — Sifan Wang, Paris Perdikaris (2020, Journal of Computational Physics, Volume 428, article 109914 (published 2021; arXiv preprint 2020)). arXiv:2006.05311
- **무엇**: Canonical PINN framework for forward and inverse Stefan (moving free-boundary phase-change) problems: two coupled networks approximate the temperature field and the moving interface, including a data-driven inverse variant needing no IC/BC.
- **우리와의 차별/활용**: This is THE method precedent for a physics-informed freeze-thaw engine: the thaw front separating active layer from permafrost is exactly a Stefan moving boundary, so ALT is literally the free-boundary location. We would differentiate by (a) driving it with real ERA5-Land surface forcing rather than analytic BCs, (b) operating in a 3D/4D geospatial setting instead of 1D toy domains, and (c) coupling the inverse Stefan idea to invert soil thermal/latent properties from sparse borehole temperatures. Caveat: pure PINN Stefan solvers are per-domain and data-hungry near the interface; our covariate-bottleneck finding suggests they help most as a physics REGULARIZER on the GBM conditioning field, not as a standalone predictor. ENABLES a physics-guided version of T2 (4D thermal).
- 파일: `references/06_physics_ml/wang2020_stefan_pinn.pdf`

### 📄 `willard2020_physics_ml_survey` — [맥락]
**Integrating Physics-Based Modeling with Machine Learning: A Survey** — Jared Willard, Xiaowei Jia, Shaoming Xu, Michael Steinbach, Vipin Kumar (2020, arXiv:2003.04919 (physics.comp-ph); published in ACM Computing Surveys, Vol. 55, No. 4, Article 66 (Nov 2022)). arXiv:2003.04919
- **무엇**: Structured survey defining a 5-way taxonomy for physics+ML integration: physics-guided loss, physics-guided initialization/pretraining, physics-guided architecture, residual modeling, and hybrid physics-ML models.
- **우리와의 차별/활용**: The organizing framework we should cite to name our chosen integration strategy precisely: for permafrost we most plausibly use (ii) physics-guided pretraining (GIPL2 synthetic -> fine-tune) + (i) energy/Stefan loss penalty + (iv) residual modeling on top of the GBM conditioning field, which is a defensible, less-explored combination for frozen ground. Not a competitor - it is the meta-map that lets us justify why we picked residual/pretraining hybrids over a pure PINN (consistent with our covariate-bottleneck finding). Context for both T1 and T2.
- 파일: `references/06_physics_ml/willard2020_physics_ml_survey.pdf`

## 07 · Context / Off-direction  (1/1 PDF)

폴더: `references/07_context/`

### 📄 `biskaborn2019_gtnp_warming` — [off-direction]
**Permafrost is warming at a global scale** — Biskaborn, B. K.; Smith, S. L.; Noetzli, J.; Matthes, H.; Vieira, G.; Streletskiy, D. A.; and others (2019, Nature Communications, 10, Article 264). doi:10.1038/s41467-018-08240-4
- **무엇**: GTN-P borehole synthesis showing global permafrost warmed +0.29C over a decade; a motivation/context citation and the authority defining the GTN-P borehole standard we use for MAGT validation, but not a mapping/ML method.
- **우리와의 차별/활용**: This is an observational warming-trend synthesis at depth >10m, NOT a spatial ALT map, ML method, or subsurface interpolation. It is off our current direction (ALT mapping + transfer + UQ + shallow 3D + forecasting) except as (a) motivation and (b) definer of the GTN-P validation standard. It neither pre-empts nor enables T1/T2 methodologically; cite only in intro/motivation.
- 파일: `references/07_context/biskaborn2019_gtnp_warming.pdf`

---

# 2차 추가분 (2026-10-04, 폴더 08–14)

2차 항목은 1차와 같은 필드(상태 표기, 키, 서지, DOI, 무엇, 우리와의 차별/활용, 파일 또는 링크)를 쓴다. 표기 규칙은 아래와 같다.

- 키는 PDF 파일 이름(확장자 제외)이다. 링크전용 항목의 키는 같은 규칙(첫저자연도_주제)으로 붙였고, 나중에 브라우저로 받으면 그 항목이 실린 절의 폴더에 `키.pdf` 로 둔다.
- 확인 수준: [본문] PDF 본문 확인, [초록] 출판사·Crossref 초록만 확인, [제목] 서지만 확인, [미확인] 확인하지 못한 사실.
- 링크 사유: "구독 필요" 는 구독 문헌, "OA, 자동 차단" 은 합법 공개본이 있으나 이 서버의 자동 내려받기가 출판사·저장소의 봇 차단에 막힌 경우, "웹 열람" 은 PDF 파일 없이 웹 뷰어만 있는 경우다.
- 한 문헌은 한 절에만 항목으로 둔다. 다른 절에도 관련되면 "다른 절에 둔 관련 항목" 줄에 키만 적는다.
- 2차 PDF 135편은 모두 `file` 과 `pdfinfo` 로 PDF 여부와 쪽 수를 확인했다(2026-10-04). 손상·HTML 파일은 없었고 삭제한 파일도 없다. 폴더 08–14 안에서 바이트가 같은 중복 사본은 없다.
- 원 조사 기록: `references/_index_parts/`(4개 파일), 폴더별 `README.md`, `docs/research/2026-10-04/`.

## 08 · 최근 ALT·영구동토 ML 문헌 (2022–2026)  (30/54 PDF)

폴더: `references/08_recent_alt_2022_2026/` · 원 조사 기록: `references/_index_parts/library_recent_alt.md`

신규성 문장과의 관계: 우리 문장은 "학습에서 제외한 지역의 ALT 오차를, 대상 지역 라벨 수의 함수로, 같은 라벨로 재보정한 물리 기준선 대비 보고한다"이다. 세 요소(지역 홀드아웃, 라벨 수 축, 재보정 물리 기준선)를 모두 갖춘 ALT·MAGT 문헌은 이번 조사에서 확인되지 않았다. 요소가 가장 많이 겹치는 문헌은 O'Malley 2026(`05_uq_transfer`, 지중 온도, 미국 학습 후 앨버타·호주·영국에 문맥 관측 1–40개로 적용, 위협 중간)과 Ahajjam 2025(아래, 검증 방식 [미확인])이다.

다른 절에 둔 관련 항목: `omalley2026_incontext_subsurface_temp`(05 절, 표 S2 본문 확인: 앨버타 MAE 관측 1개 3.65 °C, 5개 2.50 °C, 20개 2.22 °C, 40개 2.19 °C), `gay2026_zero_curtain_ai_eo`(14 절), `garibaldi2026_ttop_parameter_importance`·`peng2018_nh_alt_changes`(12 절), `portes2026_ml_spatial_transferability`(11 절), `liu2023_pilstm_permafrost_gipl2`(06 절), `pilyugina2023_pinn_permafrost_risk`(06 절), `kriuk2025_panarctic_hybrid_risk`(04 절), `koven_review2025_ml_permafrost`(05 절, 실제 저자 Memiş et al. 2025, 상단 경고 참조).

**(1) ALT 예측 ML과 직접 비교 대상**

### 🔗 `ahajjam2025_multihorizon_alt_circumarctic_jgrmlc` [경쟁자]
**Multi-horizon active layer thickness prediction across circum-Arctic permafrost regions using geospatial machine learning**. Ahajjam, A., Wilcox, A., Soaper, M., Chance, R. & Pasch, T. (2025, *J. Geophys. Res. Mach. Learn. Comput.* 2(4), e2025JH000969). doi:10.1029/2025JH000969
- **무엇** [초록]: CALM 115지점의 연 최대 ALT를 예측한다. 지리공간 특징 80개에서 다단계 선택으로 변수를 고르고, CatBoost·Extra Trees·Bagging 가중 앙상블을 예측 시차(당해, +1, +2, +5년)마다 따로 학습한다. 사전 공개본 초록은 R² 0.8 이상, RMSE 25 cm 미만을 보고한다.
- **우리와의 차별/활용**: 위협 미확인. 초록에는 대상 지역 라벨 수 축과 물리 기준선이 없다. "다양한 북극 환경으로의 일반화"를 주장하므로, 지점·지역 홀드아웃이 있다면 N1 의 "지역 홀드아웃 ALT 연구 없음" 서술을 고쳐야 한다. 투고 전 본문 확인 필수. 원고 인용 문헌이다.
- 링크(OA CC BY 4.0, Wiley 자동 차단): https://doi.org/10.1029/2025JH000969

### 🔗 `wilcox2026_circumarctic_alt_geospatial_ml_essoar` [경쟁자]
**A data-driven framework for active layer thickness prediction across circum-Arctic permafrost regions using geospatial machine learning**. Wilcox, A., Ahajjam, A., Soaper, M., Chance, R. & Pasch, T. (2026, *ESS Open Archive* preprint). doi:10.22541/essoar.177306896.67279112/v1
- **무엇** [초록]: Ahajjam 2025 와 같은 틀의 사전 공개본이다. 예측 시차 4종, 다단계 변수 선택, 앙상블, CALM 115지점, R² 0.8 이상과 RMSE 25 cm 미만을 보고한다.
- **우리와의 차별/활용**: Ahajjam 2025 의 검증 방식을 본문으로 확인하는 보조 경로다.
- 링크(OA CC BY 4.0, ESSOAr 자동 차단 403): https://doi.org/10.22541/essoar.177306896.67279112/v1

### 📄 `wang2025_alt_stefan_catboost_et_qtp_rs` [선례]
**Simulation of active layer thickness based on multi-source remote sensing data and integrated machine learning models: a case study of the Qinghai-Tibet Plateau**. Wang, G. et al. (2025, *Remote Sens.* 17, 2006). doi:10.3390/rs17122006
- **무엇** [본문]: Stefan 식 결과를 입력 특징으로 넣고 CatBoost 와 Extra Trees 를 블렌딩한 SCE 모형으로 칭하이-티베트 고원 ALT 를 추정한다. 무작위 10-fold 교차검증 MAE 20.7 cm, RMSE 32.7 cm, R² 0.873 이다. 1958–2022년을 역산해 1998년 전후 증가율 0.25 → 1.26 cm/년을 보고한다.
- **우리와의 차별/활용**: 물리 출력을 입력 특징으로 쓰는 구조의 선례다(원고 서론). 지역 홀드아웃, 라벨 수 축, 같은 자료의 Stefan 단독 오차가 없다. 위협 없음.
- 파일: `references/08_recent_alt_2022_2026/wang2025_alt_stefan_catboost_et_qtp_rs.pdf`

### 🔗 `zhang2024_climate_permafrost_model_efactor_ml_erl` [선례]
**Combining a climate-permafrost model with fine resolution remote sensor products to quantify active-layer thickness at local scales**. Zhang, C., Douglas, T. A., Brodylo, D., Bosche, L. V. & Jorgenson, M. T. (2024, *Environ. Res. Lett.* 19, 044030). doi:10.1088/1748-9326/ad31dc
- **무엇** [초록]: 기후-영구동토 모형의 토양 계수(edaphic factor, E)를 고해상도 초분광·라이다·위성 자료로 ML 추정해 국지 규모 ALT 를 산출한다. 내륙 알래스카 실험지 2곳의 2014–2022년 현장 관측에서 ALT 분산의 60 % 이상을 설명한다.
- **우리와의 차별/활용**: E 를 ML 로 추정한 선례다(원고 서론). 지역 간 전이와 라벨 수 축은 없다. NOVELTY 문서의 수치는 요약 도구 경유 기록이므로 원문 대조가 필요하다.
- 링크(OA CC BY 4.0, IOP 캡차 차단): https://iopscience.iop.org/article/10.1088/1748-9326/ad31dc

### 📄 `du2026_seasonal_precip_alt_retrospective_buildings` [선례]
**Seasonal precipitation provides modest incremental information for retrospective estimation of active-layer thickness at monitored sites along the Qinghai–Tibet Engineering Corridor**. Du, Q., Wang, F., Li, G. et al. (2026, *Buildings* 16, 3023). doi:10.3390/buildings16153023
- **무엇** [본문]: 칭하이-티베트 공학 회랑 시추공 54개의 연 ALT(2001–2020)에 지점 고정효과, 2차 시간 추세, 전년 ALT, 기온 지연을 넣은 뒤 여름 강수의 추가 정보를 시험한다. 9개 보류 연도(2012–2020) 롤링 원점 평가에서 RMSE 가 0.2035 m 에서 0.1993 m 로 2.05 % 줄었고, 공간 블록 × 시간 홀드아웃에서는 전이되는 이득이 없었다.
- **우리와의 차별/활용**: 공간 블록 × 시간 홀드아웃과 작은 효과 크기 보고 방식이 우리 음성 결과 서술과 같은 유형이다. 라벨 수 축과 물리 기준선은 없다. 위협 없음.
- 파일: `references/08_recent_alt_2022_2026/du2026_seasonal_precip_alt_retrospective_buildings.pdf`

### 🔗 `shen2023_qtp_permafrost_alt_1980_2020_stoten` [선례]
**Changes in permafrost spatial distribution and active layer thickness from 1980 to 2020 on the Tibet Plateau**. Shen, T., Jiang, P., Ju, Q. et al. (2023, *Sci. Total Environ.* 859, 160381; 온라인 2022). doi:10.1016/j.scitotenv.2022.160381
- **무엇** [초록]: 티베트 고원 영구동토 분포와 ALT 를 1980–2020년 10년 단위 1 km 로 그린다. 시추공 검증에서 분포는 ML 이, ALT 는 경험 모형(Stefan)이 더 잘 맞았고, 2010년대 지역 평균 ALT 는 1980년대보다 18.94 cm 두껍다.
- **우리와의 차별/활용**: "ALT 에서 Stefan 우위" 결론의 방향 선례다(N6, 원고 영문 문장 N6 인용). NOVELTY 문서의 "Shen 2023" 이 이 논문인지는 원문 대조가 필요하다.
- 링크(구독 필요): https://doi.org/10.1016/j.scitotenv.2022.160381

### 🔗 `herrington2026_ml_reanalysis_soil_temperature_taac` [선례]
**Application of machine learning to improve reanalysis soil temperatures over the extratropical northern hemisphere**. Herrington, T. C., Erler, A. R. & Fletcher, C. G. (2026, *Theor. Appl. Climatol.* 157(7), 논문 번호 [미확인]). doi:10.1007/s00704-026-06338-0
- **무엇** [초록]: ERA5-Land 와 FLDAS 토양 온도의 편향을 평균 편향 제거, 다중 회귀, 랜덤 포레스트로 보정한다. 랜덤 포레스트가 적설기 RMSE 를 46–77 % 줄였고, ERA5-Land 는 영구동토 지역에서 온난 편향을 보인다.
- **우리와의 차별/활용**: 재분석 + ML 보정을 단순 보정 기준선과 비교한 선례다(N2 인접). ERA5-Land 편향 기록은 TDD 편향 논의에도 쓴다. 원고 인용 문헌이다.
- 링크(OA CC BY-NC-ND, Springer 자동 차단): https://doi.org/10.1007/s00704-026-06338-0

### 📄 `du2026_alt_heterogeneity_arctic_foothills_tc` [맥락]
**Assessing spatial heterogeneity of active layer thickness over Arctic-foothills tundra, North Slope Alaska**. Du, J., Endsley, K. A., Bakian Dogaheh, K. et al. (2026, *The Cryosphere* 20, 4277–4291). doi:10.5194/tc-20-4277-2026
- **무엇** [본문]: 노스슬로프 구릉 툰드라의 90 m × 90 m 표본구 4개에서 탐침 ALT 를 약 1700회 측정하고, 드론·항공 자료와 랜덤 포레스트로 0.1 m 해상도 ALT 지도(5 km × 5 km)를 만든다. 해상도를 0.1 m 에서 1000 m 로 낮출수록 오차가 단계적으로 커지고, 10 m 해상도에서는 지형 요인의 기여가 약 65 % 다.
- **우리와의 차별/활용**: 위협 없음. 1 km 셀 라벨의 대표성 한계와 격자 안 이질성 논의(WF 3차 "격자 안 상세도 없음")의 근거다.
- 파일: `references/08_recent_alt_2022_2026/du2026_alt_heterogeneity_arctic_foothills_tc.pdf`

### 🔗 `brodylo2024_alt_multiscale_interior_alaska_erl` [맥락]
**Quantification of active layer depth at multiple scales in Interior Alaska permafrost**. Brodylo, D., Douglas, T. A. & Zhang, C. (2024, *Environ. Res. Lett.* 19, 034013). doi:10.1088/1748-9326/ad264b
- **무엇** [초록]: 현장(1 m²) ALT 를 항공 초분광·라이다로 1 km² 까지 올리고, 다시 위성 자료로 100 km² 규모로 올리는 단계적 상향 규모화 틀이다.
- **우리와의 차별/활용**: 위협 없음. 점 라벨과 격자 규모 불일치 서술의 근거다.
- 링크(OA CC BY 4.0, IOP 캡차 차단): https://iopscience.iop.org/article/10.1088/1748-9326/ad264b

### 🔗 `hantson2025_scaling_arctic_features_ald_ere` [맥락]
**Scaling Arctic landscape and permafrost features improves active layer depth modeling**. Hantson, W., Yang, D., Serbin, S. P. et al. (2025, *Environ. Res. Ecol.* 4, 015001). doi:10.1088/2752-664X/ad9f6c
- **무엇** [초록]: 지상, 무인기, 항공(AVIRIS-NG), 위성(Sentinel-2) 자료로 활동층 깊이의 미세 이질성과 관측 규모의 관계를 분석한다. 경관 규모 평균 ALD 는 항공·중해상도 위성 자료로 포착된다.
- **우리와의 차별/활용**: 위협 없음. 규모 의존성 논의용이다.
- 링크(OA CC BY 4.0, IOP 캡차 차단): https://iopscience.iop.org/article/10.1088/2752-664X/ad9f6c

### 🔗 `whitcomb2023_pband_polsar_alt_alaska_erl` [맥락]
**Maps of active layer thickness in northern Alaska by upscaling P-band polarimetric synthetic aperture radar retrievals**. Whitcomb, J., Chen, R., Clewley, D. et al. (2024, *Environ. Res. Lett.* 19, 014046; 온라인 2023). doi:10.1088/1748-9326/ad127f
- **무엇** [제목]: P밴드 편광 SAR(AirMOSS) ALT 산출값을 북알래스카로 상향 규모화한 지도다.
- **우리와의 차별/활용**: 위협 없음. 원격탐사 ALT 산출의 맥락이다. ALT 로 학습한 제품이므로 공변량 규칙상 예측 변수로 쓰지 않는다.
- 링크(OA CC BY, IOP 캡차 차단): https://iopscience.iop.org/article/10.1088/1748-9326/ad127f

### 📄 `lin2026_regional_drivers_alt_site_scale_egusphere` [맥락]
**Regional variations in drivers of active layer thickness: a site-scale analysis across Northern Hemisphere permafrost**. Lin, Y., Zhang, B., Suo, H. et al. (2026, *EGUsphere* preprint). doi:10.5194/egusphere-2026-841
- **무엇** [본문]: ALT 관측 지점 785개를 환북극(CAP), 아환북극(SCAP), 칭하이-티베트(QTP)로 나눈다. 평균 ALT 는 84.9, 200, 224 cm 이고, 5년 이상 기록이 있는 291지점의 60 % 가 증가 추세다. PLS 경로 모형에서 토양 특성이 기온보다 영향이 크다. 같은 저자의 SSRN 판(doi:10.2139/ssrn.6044956)이 있다.
- **우리와의 차별/활용**: 예측 평가가 아니어서 위협 없음. 785지점 편집 목록은 라벨 출처 점검에 쓸 수 있다.
- 파일: `references/08_recent_alt_2022_2026/lin2026_regional_drivers_alt_site_scale_egusphere.pdf`

### 📄 `xiao2025_gpr_augmented_ml_mountain_permafrost_rs` [선례]
**Mapping mountain permafrost via GPR-augmented machine learning in the northeastern Qinghai–Tibet Plateau**. Xiao, Y., Liu, G., Hu, G. et al. (2025, *Remote Sens.* 17, 2015). doi:10.3390/rs17122015
- **무엇** [본문]: 시추공, 토양 단면, GPR 횡단선 128개, 고지대 경험점 22개로 만든 존재·부재 표본 1037개로 분류기 13종을 비교한다(5-fold × 40회 반복). LightGBM·CatBoost 가 F1 0.98 이상이다. GPR 과 고지대 표본을 더하면 대표성이 낮은 지형에서 일반화가 좋아진다.
- **우리와의 차별/활용**: 위협 없음. "대표성이 낮은 곳에 관측을 더하면 일반화가 좋아진다"는 관측 설계 논의의 사례다. 분류 문제이고 무작위 교차검증이라는 점을 함께 적는다.
- 파일: `references/08_recent_alt_2022_2026/xiao2025_gpr_augmented_ml_mountain_permafrost_rs.pdf`

### 📄 `lin2025_miso_alaska_soil_permafrost_mapping_arxiv` [맥락]
**Fine-scale soil mapping in Alaska with multimodal machine learning**. Lin, Y., Chen, T., Brungard, C. et al. (2025, arXiv:2506.17302; SIGSPATIAL 2025 투고, arXiv 주석 기준). arXiv:2506.17302
- **무엇** [초록]: 지리공간 기반 모델 특징, 암묵 신경 표현, 대조 학습을 결합한 MISO 로 알래스카 전역의 지표 근처 영구동토 존재와 토양 분류 지도를 만든다. 공간 교차검증과 영구동토대·MLRA 별 분석에서 랜덤 포레스트보다 미관측 위치 일반화가 좋다.
- **우리와의 차별/활용**: ALT 회귀가 아니어서 위협 없음. 기반 모델 특징을 쓴 영구동토 지도의 공간 CV 사례다.
- 파일: `references/08_recent_alt_2022_2026/lin2025_miso_alaska_soil_permafrost_mapping_arxiv.pdf`

### 📄 `ahajjam2026_alt_trajectory_causal_calm_egusphere` [맥락]
**Beyond permafrost observation: long-term ALT trajectory classification and causal inference across the circum-Arctic CALM network**. Ahajjam, A., Soaper, M., Gupta, U., Wilcox, A., Caparó Bellido, A., Weaver, S., Parker, S., Kidanu, S. & Pasch, T. (2026, *EGUsphere* preprint). doi:10.5194/egusphere-2026-2639
- **무엇** [본문]: CALM 129지점(1990–2024)의 ALT 시계열을 방향과 속도로 6개 범주(ALDI)로 나누고, 지점 고정효과와 1차 차분 추정으로 범주별 동인을 추정한다. 두꺼워지는 지점과 얇아지는 지점의 비는 3.5 : 1 이고, 열 강제는 두꺼워지는 지점에서만 지배적이다.
- **우리와의 차별/활용**: 예측·전이 평가가 아니어서 위협 없음. Ahajjam 연구진의 CALM 자료 처리 흐름을 확인하는 용도다. Ahajjam 2025(위)와 다른 논문이다.
- 파일: `references/08_recent_alt_2022_2026/ahajjam2026_alt_trajectory_causal_calm_egusphere.pdf`

### 🔗 `kang2026_qtp_depth_resolved_permafrost_dl_climdyn` [맥락]
**Real-time, depth-resolved permafrost thermal monitoring across the Qinghai-Tibet Plateau enabled by bayesian-optimized deep learning**. Kang, X., Li, Z., Wan, J. et al. (2026, *Clim. Dyn.* 64, 339). doi:10.1007/s00382-026-08304-y
- **무엇** [제목]: 베이즈 최적화 딥러닝으로 칭하이-티베트 고원의 깊이별 지온을 추정한다.
- **우리와의 차별/활용**: 지온 대상이고 ALT 가 아니다. 신규성 관계는 초록 확인 후 분류한다 [미확인].
- 링크(구독 필요, OA 여부 [미확인]): https://doi.org/10.1007/s00382-026-08304-y

### 🔗 `zhang2026_qtp_permafrost_cmip6_ml_jclim` [맥락]
**Long-term evolution of permafrost across the Qinghai–Tibet Plateau: perspectives from multimodel ensembles and machine learning**. Zhang, T., Zhang, T., Zhou, X. et al. (2026, *J. Clim.* 39, 2385–2400). doi:10.1175/JCLI-D-25-0473.1
- **무엇** [초록]: CMIP6 자료와 SVR, CatBoost, LightGBM, DNN 으로 칭하이-티베트 고원의 영구동토 범위와 최대 계절 동결 깊이를 예측하고 SSP 4종으로 2100년까지 투영한다. DNN 이 R² 0.961 로 가장 높다.
- **우리와의 차별/활용**: 범위·동결 깊이 대상이고 전이 평가가 없어 위협 없음.
- 링크(구독 필요): https://doi.org/10.1175/JCLI-D-25-0473.1

### 🔗 `jiang2026_qtp_ml_reconstruction_multiscale_gpc` [맥락]
**Multi-scale pattern analysis of permafrost dynamics on the Qinghai–Tibet Plateau based on machine-learning reconstruction**. Jiang, P., Ding, K., Ni, J. et al. (2026, *Glob. Planet. Change* 259, 105341). doi:10.1016/j.gloplacha.2026.105341
- **무엇** [제목]: ML 재구성 자료로 칭하이-티베트 고원 영구동토 변화의 다중 규모 양상을 분석한다.
- **우리와의 차별/활용**: 맥락 인용 후보. 신규성 관계 [미확인].
- 링크(구독 필요): https://doi.org/10.1016/j.gloplacha.2026.105341

### 🔗 `ding2025_qtp_ml_climate_memory_jhydrol` [맥락]
**Machine learning uncovers a multi-year climate memory in permafrost degradation on the Qinghai–Tibet Plateau: the critical roles of precipitation and [제목 이하 원문 확인 필요]**. Ding, K., Jiang, P., Ni, J. et al. (2025, *J. Hydrol.* 663, 134272). doi:10.1016/j.jhydrol.2025.134272
- **무엇** [제목]: ML 로 영구동토 퇴화의 다년 기후 기억(강수 등)을 분석한다.
- **우리와의 차별/활용**: 위협 낮음으로 추정 [미확인].
- 링크(구독 필요): https://doi.org/10.1016/j.jhydrol.2025.134272

### 🔗 `kriuk2026_panarctic_hybrid_risk_m2garss` [맥락]
**Hybrid physics-ML framework for pan-Arctic permafrost infrastructure risk at record 2.9-million observation scale**. Kriuk, B. (2026, *2026 IEEE M2GARSS*, 214–218). doi:10.1109/M2GARSS67833.2026.11582706
- **무엇**: 1차 항목 `kriuk2025_panarctic_hybrid_risk`(arXiv:2510.02189, 04 절)의 학회 게재판이다. 대상은 영구동토 비율과 기반시설 위험이다.
- **우리와의 차별/활용**: 위협 없음. 인용할 때 게재판 서지로 바꾼다.
- 링크(구독 필요): https://doi.org/10.1109/M2GARSS67833.2026.11582706

**(2) 물리 결합·하이브리드·미분가능 모형**

### 🔗 `pilyugina2025_piml_permafrost_stability_ieeeaccess` [선례]
**A physics-informed machine learning framework for permafrost stability assessment**. Pilyugina, P., Chernikov, T., Smirnova, M. et al. (2025, *IEEE Access* 13, 96423–96433). doi:10.1109/ACCESS.2025.3573072
- **무엇**: 1차 항목 `pilyugina2023_pinn_permafrost_risk`(06 절, arXiv)의 게재판이다. 사전 공개본 본문 기준으로 Kudryavtsev 모형 출력을 CatBoost 입력으로 넣어 ALT 와 MAGT 를 직접 예측하고, 시간 분할과 무작위 5-fold 로 검증한다.
- **우리와의 차별/활용**: 물리 출력 입력형 선례다(원고 인용). 지역 홀드아웃과 라벨 수 축이 없다. 게재판에서 검증이 바뀌었는지는 [미확인]이므로 브라우저로 받아 대조한다.
- 링크(OA CC BY-NC-ND 4.0, IEEE 자동 차단): https://ieeexplore.ieee.org/document/11014074/

### 📄 `tama2025_physics_guided_residual_bed_topography_wacv` [선례]
**Learning subglacial bed topography from sparse radar with physics-guided residuals**. Tama, B. A., Wang, J., Janeja, V. & Cham, M. (2026, *Proc. IEEE/CVF WACV 2026*, 5447–5456; arXiv 2025). doi:10.1109/WACV61042.2026.00528 · arXiv:2511.14473
- **무엇** [본문]: BedMachine 사전값에 대한 정규화 두께 잔차를 DeepLabV3+ 로 예측하고 질량 보존·흐름 방향 총변동·비음수 등 물리 항으로 정칙화한다. 버퍼를 둔 블록 홀드아웃으로 그린란드 2개 소지역에서 RMSE 3.05–10.54 m 를 보고한다.
- **우리와의 차별/활용**: "물리 기준선 + 잔차" 구조와 누설 방지 블록 홀드아웃의 인접 분야 선례다(NOVELTY 2절). 라벨 수 축과 같은 라벨 재보정 기준선은 없다. 원고는 arXiv 대신 WACV 게재본을 인용할 수 있다. 파일은 arXiv 판이다.
- 파일: `references/08_recent_alt_2022_2026/tama2025_physics_guided_residual_bed_topography_wacv.pdf`

### 📄 `tian2026_noahpy_differentiable_permafrost_gmd` [방법]
**NoahPy: a differentiable Noah land surface model for simulating permafrost thermo-hydrology**. Tian, W., Yu, H., Zhao, S. et al. (2026, *Geosci. Model Dev.* 19, 57–72). doi:10.5194/gmd-19-57-2026
- **무엇** [본문]: Noah 지표 모형의 열·수분 방정식을 순환 신경망 구조로 다시 구현해 미분 가능하게 만든다. 원 모형 재현 NSE 는 0.99 이상이고, 영구동토 지점 1곳 보정 결과 토양 온도 NSE 0.9 이상이다.
- **우리와의 차별/활용**: 미분가능 물리 모형의 영구동토 선례다. 지역 전이 평가가 없어 위협 없음. 고찰의 향후 과제 인용 후보다.
- 파일: `references/08_recent_alt_2022_2026/tian2026_noahpy_differentiable_permafrost_gmd.pdf`

### 🔗 `gou2026_differentiable_permafrost_thermal_compgeo` [방법]
**δHT4P: a differentiable physical modeling framework for thermal evolution of permafrost**. Gou, L. & Likos, W. J. (2026, *Comput. Geotech.* 196, 108132). doi:10.1016/j.compgeo.2026.108132
- **무엇** [제목]: 영구동토 열 진화의 미분가능 물리 모형 틀이다.
- **우리와의 차별/활용**: NoahPy 와 같은 범주다. 신규성 관계 [미확인].
- 링크(구독 필요): https://doi.org/10.1016/j.compgeo.2026.108132

### 🔗 `zhao2025_hybrid_shaw_noah_rf_active_layer_wrr` [선례]
**A hybrid modeling approach for improved simulation of thermal-hydrological dynamics in active layer on the Qinghai-Tibet Plateau**. Zhao, Y., Nan, Z., Ji, H. et al. (2025, *Water Resour. Res.* 61, e2025WR040288). doi:10.1029/2025WR040288
- **무엇** [초록]: 랜덤 포레스트로 보정한 Noah 모의값을 SHAW 모형의 하부 경계조건으로 써서 자료가 적은 지역의 활동층 온도·수분을 모의한다. 고원 영구동토 7지점 시험 자료에서 토양 온도 NSE 0.81(Noah 0.69)이다.
- **우리와의 차별/활용**: ML 보정과 물리 모형 결합의 선례다. 지역 전이와 라벨 수 축은 없다.
- 링크(OA CC BY-NC 4.0, Wiley 자동 차단): https://doi.org/10.1029/2025WR040288

### 📄 `uxa2026_analytical_statistical_alt_mapt_tc` [방법]
**Simple analytical–statistical models (ASMs) for mean annual permafrost table temperature and active-layer thickness estimates**. Uxa, T., Hrbáček, F. & Kňažková, M. (2026, *The Cryosphere* 20, 97–112). doi:10.5194/tc-20-97-2026
- **무엇** [본문]: 활동층 안 두 깊이의 융해·동결 지수만으로 영구동토 상면 온도와 ALT 를 추정하는 해석·통계 모형 2종을 제시한다. 주요 영구동토 지역 55지점에서 평균 오차가 0.05 °C, 9 % 미만이다.
- **우리와의 차별/활용**: 물리 기준선 후보지만 활동층 안 지온 관측이 필요해 미관측 셀에는 쓸 수 없다. 기준선 선택(Stefan, Kudryavtsev)의 근거 문단에 인용한다.
- 파일: `references/08_recent_alt_2022_2026/uxa2026_analytical_statistical_alt_mapt_tc.pdf`

### 📄 `mccormick2026_analytical_active_layer_thaw_subsidence_eartharxiv` [방법]
**Analytical prediction of active-layer thaw and subsidence under seasonal thermal forcing: application to Svalbard permafrost**. McCormick, L. & Schmidt, D. (2026, *EarthArXiv* preprint). doi:10.31223/X5WJ5W (PDF 내부 제목: "From an exact thaw-consolidation solution to active-layer prediction: asymptotic limits, uncertainty, and application to an Arctic permafrost site")
- **무엇** [본문]: Lunardini 의 융해·압밀 정확해에서 Stefan 수가 작을 때와 클 때의 근사식을 유도하고(오차 약 2 %, 1.3 %) 민감도를 정량한다. 계절 평균 지표 온도를 넣으면 계절 말 ALT 를 약 8 % 과대 추정하고 융해 궤적을 약 2주 앞당긴다.
- **우리와의 차별/활용**: 위협 없음. Stefan 계열 기준선의 구조적 편향을 설명하는 근거다.
- 파일: `references/08_recent_alt_2022_2026/mccormick2026_analytical_active_layer_thaw_subsidence_eartharxiv.pdf`

### 🔗 `peng2026_physics_guided_ml_air_temperature_warm_permafrost_accr` [맥락]
**Physics-guided machine learning approach for reconstructing air temperature in warm permafrost on the Qinghai‒Xizang Plateau**. Peng, C.-Y., Luo, D., Sheng, Y. et al. (2026, *Adv. Clim. Change Res.* 17, 740–754). doi:10.1016/j.accre.2026.04.013
- **무엇** [제목]: 물리 유도 ML 로 따뜻한 영구동토 지역의 기온을 재구성한다.
- **우리와의 차별/활용**: 강제 자료 재구성이며 ALT 전이가 아니다. 위협 없음으로 추정. ACCR 는 OA 학술지로 알려져 있으나 이번에 확인하지 못했다 [미확인].
- 링크(OA 여부 [미확인]): https://doi.org/10.1016/j.accre.2026.04.013

### 📄 `yu2026_knowledge_guided_freeze_thaw_essdd` [선례]
**A knowledge-guided daily multi-layer soil freeze-thaw dataset for the Northern Hemisphere during 1950–2025**. Yu, H., Wu, M., Yin, D. et al. (2026, *Earth Syst. Sci. Data Discuss.*). doi:10.5194/essd-2026-683
- **무엇** [본문]: 토양 온도 모의값에서 동결·융해 관계를 먼저 학습하고 현장 관측으로 제약하는 지식 유도 신경망(FT-KGML)으로 북반구 0.1°, 1950–2025년, 깊이 10·30·50 cm 의 일별 동결·융해 상태를 만든다. 독립 관측소 일별 분류 정확도는 86.1, 89.7, 90.4 % 다.
- **우리와의 차별/활용**: 물리 모의 사전학습 후 관측 제약의 영구동토 선례다. ALT 와 라벨 수 축은 없다.
- 파일: `references/08_recent_alt_2022_2026/yu2026_knowledge_guided_freeze_thaw_essdd.pdf`

### 📄 `liu2025_surface_thermal_offsets_three_poles_npjclim` [맥락]
**Divergent controls on surface and thermal offsets in permafrost across the three poles**. Liu, J., Luo, D., Wu, Q. et al. (2025, *npj Clim. Atmos. Sci.* 8, 354). doi:10.1038/s41612-025-01235-1
- **무엇** [본문]: 삼극 117지점에서 지표 오프셋(SO)은 대규모 기후가, 열 오프셋(TO)은 국지 기질이 지배함을 보인다. SO 는 북극 3.1 °C, 제3극 3.2 °C, 남극 1.0 °C 다.
- **우리와의 차별/활용**: 물리 매개변수의 지역 차 근거다(N7 보조).
- 파일: `references/08_recent_alt_2022_2026/liu2025_surface_thermal_offsets_three_poles_npjclim.pdf`

### 📄 `zhao2025_west_kunlun_permafrost_thermal_state_egusphere` [맥락]
**The thermal state of permafrost under climate change on the Qinghai-Tibet Plateau from 1980 to 2022: a case study of the West Kunlun**. Zhao, J., Zhao, L., Sun, Z., Hu, G., Zou, D. et al. (2025, *EGUsphere* preprint). doi:10.5194/egusphere-2024-3956 (원 제목에 "in under" 중복 표현이 있다)
- **무엇** [본문]: ML 로 만든 1 km 월별 지표 온도를 강제 자료로 이동 격자 영구동토 모형(MVPM)을 돌려 서쿤룬 지역(약 5.6만 km²)의 지온과 ALT 를 모의한다. 지온 ±0.25 °C, ALT ±0.25 m 정확도를 보고한다.
- **우리와의 차별/활용**: 물리 모형과 ML 강제 자료의 결합 사례다. 위협 없음.
- 파일: `references/08_recent_alt_2022_2026/zhao2025_west_kunlun_permafrost_thermal_state_egusphere.pdf`

**(3) 전이·라벨 희소 조건 (인접 분야 포함)**

### 📄 `feng2023_differentiable_hydrology_ungauged_hess` [선례]
**The suitability of differentiable, physics-informed machine learning hydrologic models for ungauged regions and climate change impact assessment**. Feng, D., Beck, H., Lawson, K. & Shen, C. (2023, *Hydrol. Earth Syst. Sci.* 27, 2357–2373). doi:10.5194/hess-27-2357-2023
- **무엇** [본문]: 신경망이 HBV 매개변수를 예측하는 미분가능 모형(δ)을 LSTM 과 비교한다. 무작위 미계측 유역(PUB)에서는 비슷하거나 낫고, 지역 홀드아웃(PUR)에서는 δ 모형이 일별 지표와 평균·고유량 추세에서 LSTM 보다 낫다.
- **우리와의 차별/활용**: 하이브리드 모형의 지역 홀드아웃 평가 선례다. 대상 라벨 수가 0 으로 고정되어 있다는 점이 우리와 다르다(NOVELTY 3절, 원고 인용).
- 파일: `references/08_recent_alt_2022_2026/feng2023_differentiable_hydrology_ungauged_hess.pdf`

### 📄 `du2026_qtp_active_layer_moisture_90m_essdd` [자료]
**A first 90 m resolution active layer moisture dataset across the Qinghai–Tibet Plateau permafrost region**. Du, E., Wu, T., Dai, L. et al. (2026, *Earth Syst. Sci. Data Discuss.*). doi:10.5194/essd-2026-330
- **무엇** [본문]: 2009–2024년 현장 표본 342개와 원격탐사·지형·토양 변수로 활동층 평균 체적 수분을 90 m 로 만든다. 무작위 5-fold R² 0.62–0.63 에 비해 그룹 기반 공간 교차검증 R² 는 0.30–0.38 로 낮고, 유역 하나 제외 검증과 AOA 마스크를 함께 제공한다.
- **우리와의 차별/활용**: 무작위 분할의 낙관 편향을 같은 논문 안에서 보고한 영구동토 사례다(N8 보조). 이 자료 계열의 GPR·현장 ALT 점(Zenodo 21999366)이 우리 티베트 라벨 출처이며, 원고 표의 "article DOI [verify]" 를 이 DOI 로 채울 수 있다. 최종 게재 여부는 [미확인]이다.
- 파일: `references/08_recent_alt_2022_2026/du2026_qtp_active_layer_moisture_90m_essdd.pdf`

**(4) 관측망 설계·대표성**

### 📄 `pallandt2022_panarctic_ec_network_representativeness_bg` [방법]
**Representativeness assessment of the pan-Arctic eddy covariance site network and optimized future enhancements**. Pallandt, M. M. T. A., Kumar, J., Mauritz, M. et al. (2022, *Biogeosciences* 19, 559–583). doi:10.5194/bg-19-559-2022
- **무엇** [본문]: 생기후·토양 변수 18개로 북극 와상관 관측망의 대표성 지표(ER1, ER4)를 정의한다. 영역의 절반은 탑 1개 이상으로 대표되지만 신뢰할 외삽이 가능한 곳은 3분의 1 이다. 새 지점 15개를 더하면 대표성이 20 % 오른다.
- **우리와의 차별/활용**: 관측망 설계의 방법 선례다(대상은 탄소 플럭스). 관측 위치 알고리즘 절에서 환경 공간 대표성 지표의 출처로 인용한다.
- 파일: `references/08_recent_alt_2022_2026/pallandt2022_panarctic_ec_network_representativeness_bg.pdf`

### 📄 `betti2025_mapping_on_a_budget_spatial_sampling_arxiv` [방법]
**Mapping on a budget: optimizing spatial data collection for ML**. Betti, L., Sanni, F., Sogoyou, G. et al. (2025, arXiv:2509.03749; 저자 7명 이상, 전체 목록은 PDF 확인). arXiv:2509.03749
- **무엇** [본문 1쪽]: 위성 영상 ML 에서 지점마다 비용이 다르고 예산이 제한될 때 학습 자료 수집 위치를 최적화하는 문제를 정식화하고 방법을 제시한다.
- **우리와의 차별/활용**: 관측 위치 선택(추가 탐사) 설계의 일반 선례다. 영구동토 적용은 없다.
- 파일: `references/08_recent_alt_2022_2026/betti2025_mapping_on_a_budget_spatial_sampling_arxiv.pdf`

### 📄 `streletskiy2022_gtnp_measurement_guidelines_zenodo` [자료]
**Measurement recommendations and guidelines for the Global Terrestrial Network for Permafrost (GTN-P)**. Streletskiy, D., Noetzli, J., Smith, S. L., Vieira, G., Schoeneich, P., Hrbáček, F. & Irrgang, A. M. (2022, Zenodo). doi:10.5281/zenodo.5973079
- **무엇** [본문 1–3쪽]: GTN-P 의 시추공 지온(TSP)과 활동층(CALM) 측정 권고와 지침이다. 문서 스스로 WMO Guide No. 8 개정판의 영구동토 지침으로 대체될 임시 문서라고 밝힌다.
- **우리와의 차별/활용**: 라벨 정의(탐침, 지온 유도 ALT)의 표준 인용이다.
- 파일: `references/08_recent_alt_2022_2026/streletskiy2022_gtnp_measurement_guidelines_zenodo.pdf`

### 📄 `streletskiy2026_calm_long_term_alt_cee` [맥락]
**Long-term monitoring of active layer thickness confirms global permafrost degradation**. Streletskiy, D. A., Nyland, K. E., Shiklomanov, N. I. et al. (2026, *Commun. Earth Environ.* 7, 671). doi:10.1038/s43247-026-03824-1
- **무엇** [본문]: 북극, 남극, 산악 영구동토 156지점의 2000–2024년 ALT 관측을 종합한다. 유의한 증가는 북극 55 %, 남극 38 %, 유럽 산악·아시아 고지대 90 % 이상 지점에서 나타난다. 북극 변화는 융해 도일 증가가 가장 크게 설명하고, 다음이 강우 증가다. CALM 지점 분포가 공간적으로 고르지 않다고 적는다.
- **우리와의 차별/활용**: 서론 동기, 지점 내 √DDT 계수 비교(NOVELTY 10절), 관측망 불균등의 근거다. 원고 인용 문헌이다.
- 파일: `references/08_recent_alt_2022_2026/streletskiy2026_calm_long_term_alt_cee.pdf`

### 📄 `brown2025_beyond_magt_monitoring_metrics_egusphere` [방법]
**Beyond MAGT: learning more from permafrost thermal monitoring data with additional metrics**. Brown, N. & Gruber, S. (2025, *EGUsphere* preprint). doi:10.5194/egusphere-2025-2658
- **무엇** [본문]: 120년 모의 70여 개로 지온 지표를 평가해 영구동토 상면 높이, 연교차 0 깊이, 열 적분, MAGT, MAGST 의 5개 지표를 권고한다. 10–20 m 사이 센서 깊이에 따라 MAGT 추세가 10년 관측 기간의 절반에서 0.23 °C/10년 이상 다르다.
- **우리와의 차별/활용**: 관측 설계(센서 배치)의 영구동토 선례다.
- 파일: `references/08_recent_alt_2022_2026/brown2025_beyond_magt_monitoring_metrics_egusphere.pdf`

### 📄 `tregubov2024_talik_monitoring_network_gpr_urbsci` [맥락]
**Substantiation of the monitoring network of talik zones in urbanized permafrost areas based on GPR profiling data (Anadyr, Chukotka)**. Tregubov, O. D. & Uyagansky, K. K. (2024, *Urban Sci.* 8, 94). doi:10.3390/urbansci8030094
- **무엇** [본문]: 아나디리 시의 지온과 GPR 탐사로 탈릭 경계를 그리고, 위험 구역 20곳의 경계·중심 관측정 35개와 GPR 통제 측선 12개로 감시망을 설계한다.
- **우리와의 차별/활용**: 국지 규모 감시망 설계 사례다.
- 파일: `references/08_recent_alt_2022_2026/tregubov2024_talik_monitoring_network_gpr_urbsci.pdf`

### 📄 `zhang2024_lateral_heat_flow_permafrost_modeling_scirep` [맥락]
**Impacts of lateral conductive heat flow on ground temperature and implications for permafrost modeling**. Zhang, Y., Hong, G. & Bonney, M. T. (2024, *Sci. Rep.* 14, 31595). doi:10.1038/s41598-024-78901-6
- **무엇** [본문]: 수평 전도 열흐름이 지온에 주는 영향을 평형·과도 조건에서 계산하고, 캐나다 북동부 광산 지역 시추공 191개의 깊이별 지온 차이를 설명한다. 격자가 작으면 1차원 모형 오차가 5 °C 에 이를 수 있고, 결과를 관측 지점 선정 지침으로 쓸 수 있다고 적는다.
- **우리와의 차별/활용**: 1차원 물리식의 한계와 관측 위치 선정 근거다. Sci Rep 영구동토 모형 논문의 형식 참고 자료이기도 하다.
- 파일: `references/08_recent_alt_2022_2026/zhang2024_lateral_heat_flow_permafrost_modeling_scirep.pdf`

**(5) ALT 지도 제품·자료 논문**

### 📄 `wei2026_nh_alt_1km_2000_2024_essdd` [벤치마크]
**A 1 km resolution dataset of Northern Hemisphere permafrost active layer thickness (2000–2024)**. Wei, Y., Wu, Z., Wang, J. et al. (2026, *Earth Syst. Sci. Data Discuss.*). doi:10.5194/essd-2026-447 · 자료 doi:10.5281/zenodo.21667583
- **무엇** [본문]: 연 ALT 관측 2196건으로 앙상블 ML(5종 중 CatBoost·LightGBM 결합)을 학습해 북반구 1 km 연별 ALT(2000–2024)를 만든다. 지점 하나 제외 교차검증 R² 0.76, RMSE 60.48 cm 이고, 관측·예측 Sen 기울기 상관은 0.73 이다. 북반구 평균 ALT 125.7 cm 는 CCI(67.9 cm)와 Peng et al. 2024 ML 제품(162.9 cm)의 중간이다.
- **우리와의 차별/활용**: 경쟁 제품이자 커버리지 비교 대상이다(N9). 학습 표가 CALM 지점이라 우리 채점 셀과 겹칠 수 있으므로 표본 중복 표시가 필요하다. 라벨 수 축·지역 홀드아웃·물리 기준선 비교가 없어 위협 낮음. 제품 간 차이가 수십 cm 라는 점은 기존 제품 비교 SI 의 근거다.
- 파일: `references/08_recent_alt_2022_2026/wei2026_nh_alt_1km_2000_2024_essdd.pdf`

### 📄 `liu2024_widespread_alt_deepening_2003_2020_erl` [벤치마크]
**Widespread deepening of the active layer in northern permafrost regions from 2003 to 2020**. Liu, Z., Kimball, J. S., Ballantyne, A. P. et al. (2024, *Environ. Res. Lett.* 19, 014020; 온라인 2023). doi:10.1088/1748-9326/ad0f73 · 격자 자료 doi:10.5281/zenodo.10070609
- **무엇** [본문]: 현장 ALT 2966 지점-연으로 랜덤 포레스트를 학습해 북부 영구동토 지역 1 km 연별 ALT(2003–2020)를 만든다. 무작위 10-fold 교차검증 상위 10개 모형 앙상블이 RMSE 21.6 cm, R² 0.97 이다. 지역의 약 65 % 가 깊어지는 추세(평균 0.11 cm/년)다.
- **우리와의 차별/활용**: 기존 ALT 제품 비교(XG) 후보다. R² 0.97 은 무작위 검증 수치이므로 Wei 2026 의 LOSO RMSE 60.5 cm 와 대비해 무작위 분할의 낙관 편향 사례로 함께 적는다. 파일은 White Rose 기관 저장소 사본이다.
- 파일: `references/08_recent_alt_2022_2026/liu2024_widespread_alt_deepening_2003_2020_erl.pdf`

### 🔗 `li2022_nh_alt_2000_2018_stefan_jgra` [벤치마크]
**Active layer thickness in the Northern Hemisphere: changes from 2000 to 2018 and future simulations**. Li, C., Wei, Y., Liu, Y. et al. (2022, *J. Geophys. Res. Atmos.* 127, e2022JD036785). doi:10.1029/2022JD036785
- **무엇** [초록]: 관측 자료와 ERA5-Land 기온으로 Stefan 모형을 구동해 북반구 1 km ALT(2000–2018)를 모의하고 CMIP6 로 미래를 투영한다. 평균 ALT 는 127.19 cm 에서 145.37 cm 로 연 0.65 cm 증가한다.
- **우리와의 차별/활용**: 프로젝트 문서의 "Li C. 2022"(첫 저자 Chuanhua Li)다. 현지 관측으로 E 를 정하는 관행의 출처다(NOVELTY 4절, N5 조건 문단). E 지도 공개 여부와 관련 Dryad 자료(doi:10.5061/dryad.f4qrfj6zh)의 내용은 [미확인]이다.
- 링크(구독 필요): https://doi.org/10.1029/2022JD036785

### 🔗 `li2022_nh_permafrost_extent_alt_1969_2018_stoten` [맥락]
**Changes in permafrost extent and active layer thickness in the Northern Hemisphere from 1969 to 2018**. Li, G., Zhang, M., Pei, W. et al. (2022, *Sci. Total Environ.* 804, 150182). doi:10.1016/j.scitotenv.2021.150182
- **무엇** [제목]: 북반구 영구동토 범위와 ALT 의 1969–2018년 변화를 추정한다.
- **우리와의 차별/활용**: 첫 저자가 Guanji Li 로, Li C. 2022(위)와 다른 논문이다. 혼동하지 않는다. 신규성 관계 [미확인].
- 링크(구독 필요): https://doi.org/10.1016/j.scitotenv.2021.150182

### 🔗 `peng2023_alt_permafrost_area_projections_ef` [벤치마크]
**Active layer thickness and permafrost area projections for the 21st century**. Peng, X., Zhang, T., Frauenfeld, O. W. et al. (2023, *Earth's Future* 11, e2023EF003573). doi:10.1029/2023EF003573
- **무엇** [제목, Wei 2026 참고문헌 기준]: 21세기 ALT 와 영구동토 면적을 투영한다. 방법은 [미확인]이다. Wei 2026 이 비교한 "Peng et al. 2024" 1 km ML 제품은 이 계열로 보이나 DOI 는 [미확인]이다.
- **우리와의 차별/활용**: 기존 제품 비교 후보다. 그림 모범 사례 조사(09 절)에서도 후보였으나 그림은 검토하지 못했다.
- 링크(OA, Wiley 자동 차단): https://doi.org/10.1029/2023EF003573

### 🔗 `liu2025_reanalysis_permafrost_alt_asl` [맥락]
**Near-surface permafrost extent and active layer thickness characterized by reanalysis/assimilation data**. Liu, Z., Guo, D., Hua, W. et al. (2025, *Atmos. Sci. Lett.* 26, e1289). doi:10.1002/asl.1289
- **무엇** [초록]: 재분석·동화 자료 7종(CFSR, MERRA-2, ERA5, ERA5-Land, GLDAS 3종)의 영구동토 범위·ALT 표현력을 비교한다. 대부분 표현력이 제한적이고 GLDAS-CLSMv20 이 가장 낫다.
- **우리와의 차별/활용**: ERA5-Land 강제 자료 편향 논의의 근거다(계수 차이의 원인 분리 한계, NOVELTY 6절).
- 링크(OA CC BY 4.0, Wiley 자동 차단): https://doi.org/10.1002/asl.1289

### 📄 `guo2026_ensemble_thaw_conditions_essdd` [맥락]
**An ensemble dataset of permafrost thaw conditions for northern high latitudes from open satellite data**. Guo, D., Wang, C. & Zang, S. (2026, *Earth Syst. Sci. Data Discuss.*). doi:10.5194/essd-2026-299 · 자료 doi:10.5281/zenodo.19148960
- **무엇** [본문]: 공개 영구동토 제품 3종을 앙상블해 영구동토 비율과 MAGT 를 만들고, 위성 자료 16종과 XGBoost 로 융해 취약 지수(PTI)를 예측한다(전체 정확도 91.8 %). 시추공 26곳과의 Spearman r 은 0.69 다.
- **우리와의 차별/활용**: ALT 제품이 아니어서 위협 없음.
- 파일: `references/08_recent_alt_2022_2026/guo2026_ensemble_thaw_conditions_essdd.pdf`

### 📄 `zou2025_qtp_permafrost_temperature_15m_essd` [자료]
**Permafrost temperature baseline at 15 m depth on the Qinghai–Tibetan Plateau (2010–2019)**. Zou, D., Zhao, L., Hu, G. et al. (2025, *Earth Syst. Sci. Data* 17, 1731–1742). doi:10.5194/essd-17-1731-2025
- **무엇** [본문]: 시추공 231개의 15 m 지온으로 SVR 을 학습해 약 1 km 격자 MAGT15m 를 만든다(R² 0.48). 고원 평균은 −1.85 ± 1.58 °C 다.
- **우리와의 차별/활용**: 위협 없음. 티베트 지온 자료의 출처 후보다.
- 파일: `references/08_recent_alt_2022_2026/zou2025_qtp_permafrost_temperature_15m_essd.pdf`

### 📄 `talucci2025_firealt_paired_burned_essd` [자료]
**Permafrost–wildfire interactions: active layer thickness estimates for paired burned and unburned sites in northern high latitudes**. Talucci, A. C., Loranty, M. M., Holloway, J. E. et al. (2025, *Earth Syst. Sci. Data* 17, 2887–2909). doi:10.5194/essd-17-2887-2025
- **무엇** [본문]: 기여자 18명의 융해 깊이 52,466건을 모아 수정 Stefan 식으로 계절 말 ALT 48,669건을 추정한다(9446 구획, 화재·비화재 쌍 157개).
- **우리와의 차별/활용**: 우리 FireALT 라벨(원고 Methods 의 Talucci 2024 자료, doi:10.18739/A2RN3092P)의 자료 논문이다. 계절 중 측정을 Stefan 식으로 계절 말로 환산한 라벨이므로 라벨 정의 혼합 민감도 논의에 쓴다. 그림 모범 사례 조사(09 절)에서 Fig 2(봄·가을 단면 삽화)를 개념 패널 참고로 골랐다.
- 파일: `references/08_recent_alt_2022_2026/talucci2025_firealt_paired_burned_essd.pdf`

### 📄 `zhu2024_alaska_tundra_vegetation_active_layer_db_essd` [자료]
**A synthesized field survey database of vegetation and active-layer properties for the Alaskan tundra (1972–2020)**. Zhu, X., Chen, D., Kogure, M. et al. (2024, *Earth Syst. Sci. Data* 16, 3687–3703). doi:10.5194/essd-16-3687-2024
- **무엇** [본문]: 알래스카 툰드라의 식생·활동층 현장 조사 자료를 통합하고 화재 이력을 붙인 데이터베이스다. 구획 크기는 1 m × 1 m 부터 다양하다.
- **우리와의 차별/활용**: 알래스카 추가 라벨 출처 후보다.
- 파일: `references/08_recent_alt_2022_2026/zhu2024_alaska_tundra_vegetation_active_layer_db_essd.pdf`

### 📄 `chang2024_deformation_alt_nonlinear_npjclim` [맥락]
**Unraveling the non-linear relationship between seasonal deformation and permafrost active layer thickness**. Chang, T., Yi, Y., Jiang, H. et al. (2024, *npj Clim. Atmos. Sci.* 7, 308). doi:10.1038/s41612-024-00866-0
- **무엇** [본문]: 칭하이-티베트 고원에서는 ALT 와 계절 지표 변형의 상관이 음(r = −0.53)이어서 배수가 나쁜 북극 토양의 양의 관계와 반대다. 식생이 성기고 토양이 건조할수록 더 음이 된다.
- **우리와의 차별/활용**: InSAR 기반 ALT 공변량의 지역 의존성 근거다. KOPRI 정윤택 박사의 InSAR 변위 연구(10 절)와 협의할 때 함께 본다.
- 파일: `references/08_recent_alt_2022_2026/chang2024_deformation_alt_nonlinear_npjclim.pdf`

**(6) 리뷰**

### 🔗 `greene2026_computational_methods_permafrost_survey_peps` [맥락]
**Advanced computational methods for predicting permafrost conditions: a survey**. Greene, T., Kaabouch, N. & Pasch, T. (2026, *Prog. Earth Planet. Sci.* 13, 55). doi:10.1186/s40645-026-00828-5
- **무엇** [제목]: 영구동토 상태 예측의 계산 방법 리뷰다. 공저자 Pasch 는 Ahajjam 2025 의 공저자다.
- **우리와의 차별/활용**: Ahajjam 2025 의 검증 방식을 요약했을 가능성이 있어 함께 확인한다 [미확인].
- 링크(OA, Springer 인증 리다이렉트로 자동 차단): https://doi.org/10.1186/s40645-026-00828-5

### 🔗 `bartsch2023_permafrost_monitoring_from_space_survgeophys` [맥락]
**Permafrost monitoring from space**. Bartsch, A., Strozzi, T. & Nitze, I. (2023, *Surv. Geophys.* 44, 1579–1613). doi:10.1007/s10712-023-09770-3
- **무엇** [제목]: 위성 원격탐사 기반 영구동토 감시의 리뷰다.
- **우리와의 차별/활용**: 서론 맥락과 InSAR·광학 공변량 후보 검토용이다.
- 링크(OA, Springer 자동 차단): https://doi.org/10.1007/s10712-023-09770-3

### 🔗 `zhao2024_qtp_permafrost_review_ppp` [맥락]
**Investigation, monitoring, and simulation of permafrost on the Qinghai-Tibet Plateau: a review**. Zhao, L., Hu, G., Liu, G. et al. (2024, *Permafr. Periglac. Process.* 35, 412–422). doi:10.1002/ppp.2227
- **무엇** [제목]: 칭하이-티베트 고원 영구동토 조사·감시·모의의 리뷰다.
- **우리와의 차별/활용**: 티베트 독립 지역 자료 확보 경로 확인용이다.
- 링크(OA, Wiley 자동 차단): https://doi.org/10.1002/ppp.2227

## 09 · 그림 모범 사례 (상위 학술지 영구동토·평가 설계 논문)  (10/17 PDF)

폴더: `references/09_figure_exemplars/` · 폴더 안내: `references/09_figure_exemplars/README.md` · 그림별 분석: `docs/research/2026-10-04/alt_figure_exemplars.md`

선정 기준: 그림을 PDF 쪽 렌더링으로 직접 확인한 뒤, 우리 Fig 1–7(자료 지도, 라벨 수 곡선, 물리 개념, 조건별 성능, 관측 배치, 불확실성 지도, 워크플로)에 옮길 요소가 있는 그림만 골랐다. 공통 규칙은 다섯 가지다. (1) 개념도는 상자와 문장 대신 실제 좌표·단면·도형 위 직접 라벨로 그린다. (2) 범례 상자 대신 선 끝·점 무리 옆 직접 라벨을 쓴다. (3) 성능 곡선에 기준선(영 모형, 물리 모형, 외부 기준)을 함께 그려 교차점이 결론이 되게 한다. (4) 불확실성은 정의를 밝힌 오차 막대·음영 띠로 보인다. (5) 소형 다중 그림은 행·열 머리만으로 읽히게 하고 범례를 한 번만 둔다.

관계 표기: [그림] 그림 설계 모범 사례, [맥락] 내용 참고(그림은 따르지 않음), [그림·반면교사] 피할 설계의 예.

다른 절에 둔 그림 분석 대상: `biskaborn2019_gtnp_warming`(07 절, Fig 1·3·4·7), `meyer2022_globalmaps_aoa`(05 절), `aalto2018_circumarctic_ground_temp_alt`(12 절, Fig 3 방법별 구간 폭), `talucci2025_firealt_paired_burned_essd`·`wei2026_nh_alt_1km_2000_2024_essdd`(08 절), `linnenbrink2024_knndm`(05 절, Fig 3·5). 비교 기준(모범 사례 아님): `gautam2025_alaska_alt`(01 절, 연속 ALT 에 범주 색), `ran2022_panarctic`(01 절), `obu2019_ttop_nh`(02 절).

### 📄 `hjort2018_arctic_infrastructure_risk` [그림]
**Degrading permafrost puts Arctic infrastructure at risk by mid-century**. Hjort, J., Karjalainen, O., Aalto, J., Westermann, S. et al. (2018, *Nat. Commun.* 9, 5147). doi:10.1038/s41467-018-07557-4
- **무엇**: 통계 모형 앙상블(GLM·GAM·RF·GBM)로 현재·미래 영구동토 분포를 예측하고 기반시설 위험을 평가한다. Fig 1 은 MAGT 시추공 점(6계급 발산형, 검은 테두리)과 영구동토 범주 면을 한 지도에 겹치고 확대도 2개를 축척 막대와 함께 잇는다. Fig 2 는 2 × 2 점·범위 막대 그림에서 범주를 하나 건너 회색 띠로 칠한다. Fig 3 은 기반시설을 색이 아니라 선 종류와 점 크기로 구분한다.
- **우리와의 차별/활용**: Fig 1(관측 지점 + 영구동토 구역 지도, 확대도 연결과 축척), Fig 4·Fig 6(범주별 점·오차 막대와 회색 띠 묶음)에 쓴다. CC BY-SA.
- 파일: `references/09_figure_exemplars/hjort2018_arctic_infrastructure_risk.pdf`

### 📄 `karjalainen2019_circumpolar_geohazard_maps` [그림]
**Circumpolar permafrost maps and geohazard indices for near-future infrastructure risk assessments**. Karjalainen, O., Aalto, J., Luoto, M., Westermann, S. et al. (2019, *Sci. Data* 6, 190037). doi:10.1038/sdata.2019.37
- **무엇**: 30초 해상도 지리 자료로 지온과 ALT 를 통계 모형으로 예측하고 지반 위험 지수를 만든다(RCP 2.6, 4.5, 8.5, 2041–2060·2061–2080). Fig 2 는 MAGT 지점(검은 삼각형, n = 797)과 ALT 지점(흰 원, n = 303)을 색이 아니라 모양과 명도로 구분한다. Fig 3 은 4행 × 2열 소형 다중 지도를 행·열 머리와 공유 범례 하나로 읽게 한다. Fig 1 은 상자 6개에 문장을 채운 흐름도로 반면교사다.
- **우리와의 차별/활용**: Fig 1(자료 종류를 모양으로 구분), Fig 6 SI 지도판에 쓴다. 내용 면에서는 ALT 거리 제외(500 km) 검증의 선례로 원고 서론에 인용한다(NOVELTY 8절). CC BY-SA.
- 파일: `references/09_figure_exemplars/karjalainen2019_circumpolar_geohazard_maps.pdf`

### 📄 `nitzbon2020_ice_rich_permafrost_siberia` [그림]
**Fast response of cold ice-rich permafrost in northeast Siberia to a warming climate**. Nitzbon, J., Westermann, S., Langer, M., Martin, L. C. P. et al. (2020, *Nat. Commun.* 11, 2201). doi:10.1038/s41467-020-15725-8
- **무엇**: 물리 모형 결과를 개념 단면도, 소형 다중 그림, 모형 설정도로 나누어 보인다. Fig 1 은 지형 단면 블록도 위에 단위 이름을 직접 쓰고 분석 제외 단위만 회색 글자로 구분한다. Fig 3 은 4행 × 3열에서 깊이 1 m 당 길이를 행 사이에 같게 맞추고, 12패널 가운데 하나에만 "읽는 법" 화살표 주석을 단다. Fig 5 는 문장 없이 기둥·화살표·치수선으로 모형 구조를 설명한다.
- **우리와의 차별/활용**: Fig 3(개념 단면과 결과 소형 다중 그림), Fig 7(워크플로를 상자 대신 "읽는 법" 주석 하나로)에 쓴다. CC BY.
- 파일: `references/09_figure_exemplars/nitzbon2020_ice_rich_permafrost_siberia.pdf`

### 📄 `ploton2020_spatial_validation` [그림]
**Spatial validation reveals poor predictive performance of large-scale ecological mapping models**. Ploton, P., Mortier, F., Réjou-Méchain, M., Barbier, N. et al. (2020, *Nat. Commun.* 11, 4540). doi:10.1038/s41467-020-18321-y
- **무엇**: 중앙아프리카 산림 조사 자료(수목 1,180만 그루)로 대규모 지상부 생물량 지도의 랜덤 포레스트 절차를 재현한다. 비공간 검증은 변동의 절반 이상을 설명한다고 나오지만 공간 자기상관을 고려한 검증에서는 예측력이 거의 0 이다. Fig 5 는 버퍼 반경에 따른 R²·RMSPE 곡선에 영 모형 기준선을 함께 그려 약 100 km 부근의 교차점이 결론이 되게 한다. Fig 3 은 세 단계 확대 도식으로 공간 fold 와 버퍼를 보인다.
- **우리와의 차별/활용**: 무작위 분할 과대평가(보조 주장)의 대표 선례로 원고에 인용한다(NOVELTY 9절). 그림은 Fig 2(곡선과 기준선의 교차), Fig 5(거리를 기대 성능으로 바꾼 지도), Fig 1c(평가 설계 확대 도식)에 쓴다. CC BY.
- 파일: `references/09_figure_exemplars/ploton2020_spatial_validation.pdf`

### 📄 `tsai2021_parameter_learning_scaling` [그림]
**From calibration to parameter learning: harnessing the scaling effects of big data in geoscientific modeling**. Tsai, W.-P., Feng, D., Pan, M., Beck, H. E. et al. (2021, *Nat. Commun.* 12, 5988). doi:10.1038/s41467-021-26107-z
- **무엇**: 미분 가능 매개변수 학습(dPL)과 기존 보정(SCE-UA)을 비교한다. Fig 7b 는 학습에 쓴 유역 비율에 따른 중앙 KGE 곡선에 외부 기준값을 점선과 별표로 표시해 "기준에 도달하는 자료 비율"을 바로 읽게 한다. Fig 1(색 평행사변형 흐름도)과 Fig 7a(y 축 4개)는 반면교사다.
- **우리와의 차별/활용**: 영구동토 문헌에 라벨 수 곡선의 상위 학술지 예가 드물어 Fig 2·Fig 7 의 대체 참고로 둔다. 다축 구성은 쓰지 않는다. CC BY.
- 파일: `references/09_figure_exemplars/tsai2021_parameter_learning_scaling.pdf`

### 📄 `burke2020_cmip6_permafrost_physics` [그림]
**Evaluating permafrost physics in the Coupled Model Intercomparison Project 6 (CMIP6) models and their sensitivity to climate change**. Burke, E. J., Zhang, Y. & Krinner, G. (2020, *The Cryosphere* 14, 3155–3174). doi:10.5194/tc-14-3155-2020
- **무엇**: CMIP6 모형의 영구동토 물리를 관측과 비교한다. Fig 1 은 온도·깊이 축 위에 연 최대·최소·평균 곡선과 ALT 구간, Dzaa 를 직접 표시한 개념도다. Fig 9 는 MAAT 구간별로 모형과 CALM 관측 상자그림을 나란히 둔 5행 × 4열 소형 다중 그림이다(범례를 모든 칸에 반복하는 점은 약점).
- **우리와의 차별/활용**: Fig 3(Stefan 융해 전선 개념을 좌표 위 도식으로), Fig 4(기후·계수 오차 구간별 ML 이득 분포)에 쓴다. CC BY.
- 파일: `references/09_figure_exemplars/burke2020_cmip6_permafrost_physics.pdf`

### 📄 `langer2024_cryogridlite_ensemble` [그림]
**The evolution of Arctic permafrost over the last 3 centuries from ensemble simulations with the CryoGridLite permafrost model**. Langer, M., Nitzbon, J., Groenke, B., Assmann, L.-M. et al. (2024, *The Cryosphere* 18, 363–385). doi:10.5194/tc-18-363-2024
- **무엇**: CryoGridLite 앙상블로 지난 3세기 북극 영구동토를 모의한다. Fig 3 은 순차형 색의 모의 ALT 지도 위에 CALM 지점의 모형 − 관측 차이를 0 중심 발산형 원으로 겹치고, 산점도에 앙상블 5–95 백분위 오차 막대, 1:1 선, ±20 % 선을 둔다. Fig 1 은 앙상블에서 바꾼 경계만 자홍색으로 칠한 토양 기둥 개념도다.
- **우리와의 차별/활용**: Fig 6(예측 지도 위 관측 잔차 원, 커버리지 산점도)에 가장 가깝다. Fig 3(보정하는 계수만 강조색), Fig 7 분해 패널에도 쓴다. CC BY.
- 파일: `references/09_figure_exemplars/langer2024_cryogridlite_ensemble.pdf`

### 📄 `nitzbon2024_no_permafrost_tipping_point_preprint` [그림]
**No respite from permafrost-thaw impacts in the absence of a global tipping point**. Nitzbon, J., Schneider von Deimling, T., Aliyeva, M., Chadburn, S. E. et al. (2024, *Nat. Clim. Change* 14, 573–585). doi:10.1038/s41558-024-02011-4 · 파일은 EarthArXiv 프리프린트 v1(doi:10.31223/x55x08, CC BY-NC-SA)
- **무엇**: 전 지구 영구동토 티핑 포인트 논의를 검토한다. Fig 3 은 위 행에 상자 없는 블록도 3개(라벨을 지표·단면 곡선을 따라 직접 씀), 아래 행에 같은 색 체계의 깊이·시간 결과 패널 4개를 짝지운다. Fig 1 은 결과 곡선과 근거 관계를 한 그림 안 삽입 그림으로 겹친다.
- **우리와의 차별/활용**: Fig 7(워크플로와 분해를 "개념 위, 결과 아래" 짝으로), Fig 1(지역 구분 범례를 행렬로), Fig 6(중심선, 범위 음영)에 쓴다. 게재본 그림과 다를 수 있다 [미확인].
- 파일: `references/09_figure_exemplars/nitzbon2024_no_permafrost_tipping_point_preprint.pdf`

### 📄 `shen2023_differentiable_modelling_review` [그림]
**Differentiable modelling to unify machine learning and physical models for geosciences**. Shen, C., Appling, A., Gentine, P., Bandai, T. et al. (2023, *Nat. Rev. Earth Environ.* 4, 552–567). doi:10.1038/s43017-023-00450-9 · 파일은 OSTI 수락 원고
- **무엇**: 미분가능 모형으로 ML 과 물리 모형을 결합하는 틀을 리뷰한다. Fig 2 는 비용 함수 지형 위에 ML, 미분가능 지구과학, 과정 기반 모형의 탐색 공간을 닫힌 곡선으로 겹치고 경로 화살표 2개로 결합 방식을 보이는 상자 없는 개념도다. Fig 1 은 파란 상자 중심 도식으로 반면교사다.
- **우리와의 차별/활용**: Fig 3(물리 기준선과 잔차 ML 의 위치를 함수 공간 도식으로)과 발표 개념 슬라이드에 쓴다. 게재본 그림은 출판사 재작도본일 수 있다 [미확인].
- 파일: `references/09_figure_exemplars/shen2023_differentiable_modelling_review.pdf`

### 📄 `clayton2021_alt_soil_water` [맥락]
**Active layer thickness as a function of soil water content**. Clayton, L. K., Schaefer, K. M., Battaglia, M. J. et al. (2021, *Environ. Res. Lett.* 16, 055028). doi:10.1088/1748-9326/abfa4c · 파일은 OSTI/LANL 수락 원고
- **무엇**: 알래스카 현장 자료로 ALT 와 체적 함수율의 관계(잠열 가설과 열전도 가설)를 검토한다.
- **우리와의 차별/활용**: Stefan 형 해석과 직접 관련되어 본문 인용 후보다. 그림은 반면교사다(2차원 밀도에 jet 색표, 색 채운 이름 상자 범례, 패널마다 회귀식 반복).
- 파일: `references/09_figure_exemplars/clayton2021_alt_soil_water.pdf`

### 🔗 `natali2019_winter_co2_loss_permafrost_ncc` [그림]
**Large loss of CO2 in winter observed across the northern permafrost region**. Natali, S. M., Watts, J. D., Rogers, B. M., Potter, S. et al. (2019, *Nat. Clim. Change* 9, 852–857). doi:10.1038/s41558-019-0592-8
- **무엇**: 겨울철 CO2 플럭스 관측을 종합해 북부 영구동토 지역의 겨울 탄소 손실을 추정한다. 저자 원고(PMC8781060) 그림 이미지만 확인했다. Fig 1 은 관측 지점 지도와 생물군계·영구동토 구역별 바이올린을 한 그림에 두고, Fig 3 은 기준기 큰 지도와 미래 소형 지도 2 × 2 가 범례 하나를 공유한다(음이 아닌 플럭스에 발산형 색을 쓴 점은 약점).
- **우리와의 차별/활용**: Fig 1(지도와 지역별 라벨 분포), Fig 6 지도판(기준 큰 지도와 조건별 소형 지도)에 쓴다. 게재본 그림과 다를 수 있다 [미확인].
- 링크(OA 저자 원고, 출판사·Europe PMC 자동 차단): https://europepmc.org/articles/PMC8781060

### 🔗 `hjort2022_permafrost_degradation_infrastructure_nree` [그림]
**Impacts of permafrost degradation on infrastructure**. Hjort, J., Streletskiy, D., Doré, G. et al. (2022, *Nat. Rev. Earth Environ.* 3, 24–38). doi:10.1038/s43017-021-00247-8
- **무엇** [제목]: 영구동토 열화가 기반시설에 주는 영향을 리뷰한다. 그림은 검토하지 못했다.
- **우리와의 차별/활용**: 서론 맥락과 그림 추가 검토 후보다. 저자 목록은 Crossref 기준(6인)이다.
- 링크(OA 기관 저장소, 봇 확인 페이지로 자동 차단): http://hdl.handle.net/10138/344541

### 🔗 `smith2022_changing_thermal_state_permafrost_nree` [그림]
**The changing thermal state of permafrost**. Smith, S. L., O'Neill, H. B., Isaksen, K., Noetzli, J. et al. (2022, *Nat. Rev. Earth Environ.* 3, 10–23). doi:10.1038/s43017-021-00240-1
- **무엇** [제목]: 영구동토 열 상태 변화의 리뷰다(저자 5인). 그림은 검토하지 못했다.
- **우리와의 차별/활용**: 서론 배경 인용 후보다. 과제 후보 "Smith et al. 2022 (Permafrost & Periglac. Process.)" 와 같은 논문인지는 [미확인]이다.
- 링크(구독 필요): https://doi.org/10.1038/s43017-021-00240-1

### 🔗 `natali2021_permafrost_carbon_feedbacks_pnas` [그림]
**Permafrost carbon feedbacks threaten global climate goals**. Natali, S. M., Holdren, J. P., Rogers, B. M. et al. (2021, *PNAS* 118, e2100163118). doi:10.1073/pnas.2100163118
- **무엇** [제목]: 영구동토 탄소 되먹임과 전 지구 기후 목표의 관계를 다룬다. 그림은 검토하지 못했다.
- **우리와의 차별/활용**: 서론 동기 문장 후보다. 저자 목록은 Crossref 기준(7인)이다.
- 링크(OA, Europe PMC 자동 차단): https://europepmc.org/articles/PMC8166174

### 🔗 `chen2023_pdo2_alt_soil_moisture_insar_polsar_ess` [그림]
**Permafrost Dynamics Observatory (PDO): 2. Joint retrieval of permafrost active layer thickness and soil moisture from L-band InSAR and P-band PolSAR**. Chen, R. H., Michaelides, R. J., Zhao, Y., Huang, L. et al. (2023, *Earth Space Sci.* 10, e2022EA002453). doi:10.1029/2022EA002453
- **무엇** [제목]: L밴드 InSAR 와 P밴드 PolSAR 로 ALT 와 토양 수분을 함께 산출한다(ReSALT 계열). 그림은 검토하지 못했다.
- **우리와의 차별/활용**: 원격탐사 ALT 산출의 맥락이고 정윤택 박사 협의(InSAR·PolSAR) 때 참고한다. ALT 산출 제품이므로 공변량 규칙상 예측 변수로 쓰지 않는다.
- 링크(OA, Wiley 자동 차단): https://doi.org/10.1029/2022EA002453

### 🔗 `mishra2021_permafrost_soc_stocks_sciadv` [그림·반면교사]
**Spatial heterogeneity and environmental predictors of permafrost region soil organic carbon stocks**. Mishra, U., Hugelius, G., Shelef, E., Yang, Y. et al. (2021, *Sci. Adv.* 7, eaaz5236). doi:10.1126/sciadv.aaz5236
- **무엇**: 영구동토 지역 토양 유기탄소 저장량의 공간 이질성과 환경 예측 변수를 분석한다. 그림 검토 결과 무지개형 지도와 이중 y 축 막대를 써서 모범 사례에서 제외했다.
- **우리와의 차별/활용**: 피할 설계의 예다(순차형 색, 축 하나로 대체).
- 링크(OA, 출판사·Europe PMC 자동 차단): https://europepmc.org/articles/PMC7904252

### 🔗 `hugelius2020_peatland_carbon_nitrogen_pnas` [그림·반면교사]
**Large stocks of peatland carbon and nitrogen are vulnerable to permafrost thaw**. Hugelius, G., Loisel, J., Chadburn, S., Jackson, R. B. et al. (2020, *PNAS* 117, 20438–20446). doi:10.1073/pnas.1916387117
- **무엇**: 북부 이탄지 탄소·질소 저장량과 영구동토 융해 취약성을 추정한다. 그림 검토 결과 원판 지도 겹침과 고채도 다색상을 써서 모범 사례에서 제외했다.
- **우리와의 차별/활용**: 피할 설계의 예다(겹치지 않는 격자 배치, 순차형 색으로 대체).
- 링크(OA, 출판사·Europe PMC 자동 차단): https://europepmc.org/articles/PMC7456150

## 10 · KOPRI 협업 연구자 (김승희·정윤택)  (36/63 PDF)

폴더: `references/10_kopri_collaborators/` (하위 `seung_hee_kim/`, `seung_hee_kim/abstracts/`, `yoon_taek_jung/`, `_kopri_center_reports/`) · 신원 판정 근거와 협의 사항: `references/10_kopri_collaborators/README.md`

신원: 두 사람 모두 KOPRI 원격탐사빙권정보센터 소속이다(KOPRI 직원검색, 2026-10-04). 동일인 판정 신뢰도는 김승희 높음(ORCID 0000-0002-6434-5388), 정윤택 높음(ORCID 0000-0002-6240-2975)이다. 정윤택은 KOPRI 소속 출판물이 아직 없어 첫 협의 때 본인 확인을 권한다. 동명이인(Chapman University 의 Seung Hee Kim, 국립수산과학원 Seung-Hee Kim, KOPRI 한승희 Seunghee Han 등)은 병합하지 않았다. 공개된 직업 정보(소속, ORCID, 논문, 학위논문 서지)만 적었다.

관계 표기: [협업자] 공동 작업 예정 연구자의 저작, [자료] 센터 과제 보고서(협업자 비저자 또는 참여연구원 등재). 2차 분석의 KPDC 접점은 이 절 끝의 표에 둔다.

우리 연구와의 접점 요약: 정윤택 박사는 중앙 야쿠티아 영구동토의 InSAR 계절 변위를 Stefan 해 기반 구간 선형 모형으로 해석하고(RSE 2023), 물리 모형(RVoG) 모의 자료로 딥러닝 역산기를 학습했다(박사학위논문). 이는 Stefan 계수의 수분 항을 관측으로 제약하는 공변량 경로이자, 물리 출력을 학습 신호로 쓰는 우리 설계와 방법상 가깝다. 연구 지역은 우리 Syrdakh 자료(`pohl2026_syrdakh_observatory_essd`, 12 절)와 같은 중앙 야쿠티아다. 김승희 박사는 SAR·DEM 처리와 XGBoost·SHAP 예측을 다뤘고, 참여연구원으로 등재된 센터 과제가 콘슬 Site 1·2 무인기 초분광·라이다 자료와 KOPRI-KPDC-00001470(콘슬 Pleiades 50 cm 영상·DSM)을 만들었다. 김승희 박사의 영구동토·ALT 논문은 확인되지 않았다.

다른 절에 둔 관련 항목: `chang2024_deformation_alt_nonlinear_npjclim`(08 절, InSAR 변형과 ALT 의 지역 의존 관계), `chen2023_pdo2_alt_soil_moisture_insar_polsar_ess`(09 절), `whitcomb2023_pband_polsar_alt_alaska_erl`(08 절).

**10.1 김승희 (Seung Hee Kim)**

### 🔗 `kim2026_terra_nova_fast_ice_xgboost_shap_nzjgg` [협업자]
**Quantifying the relative impact of atmospheric variables and polynya dynamics on fast ice in Terra Nova Bay, Antarctica**. Kim, S. H., Choi, C., Lee, S., Han, H. & Kim, S. (2026, *N. Z. J. Geol. Geophys.* 69(1), e70026). doi:10.1002/jgo2.70026
- **무엇** [README 기록]: Terra Nova Bay 정착빙 면적을 XGBoost 와 SHAP 으로 예측·해석한다. 1–5개월 지연 기상 변수를 넣으면 검증 R 0.57, 빼면 0.38 이다.
- **우리와의 차별/활용**: 트리 부스팅 + 설명 기법, 선행 기간 기상 변수 입력이라는 점에서 우리 CatBoost 잔차 모형과 TDD 입력 구조가 같은 계열이다. 협의 때 방법 용어를 맞추는 데 쓴다.
- 링크(OA CC BY-NC 4.0, Wiley 자동 차단): https://doi.org/10.1002/jgo2.70026

### 🔗 `kim2026_victoria_land_research_review_nzjgg` [협업자]
**Past, present, and future of the Antarctic continent: research activities in Victoria Land, Antarctica**. Kim, D., Han, H., Cui, X., Naeher, S., Kim, S. H., Jovane, L. & Ahn, J. (2026, *N. Z. J. Geol. Geophys.* 69(3), e70071). doi:10.1002/jgo2.70071
- **무엇** [제목]: 남극 빅토리아랜드 연구 활동을 정리한다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 링크(구독 필요): https://doi.org/10.1002/jgo2.70071

### 🔗 `eayrs2024_ml_sea_ice_applications_bams` [협업자]
**Advances in machine learning techniques can assist across a variety of stages in sea ice applications**. Eayrs, C. et al. (공저자 30명 중 Kim, S. H.) (2024, *Bull. Am. Meteorol. Soc.* 105(3), E527–E531). doi:10.1175/BAMS-D-23-0332.1
- **무엇** [제목]: 해빙 응용의 여러 단계에서 ML 기법의 쓰임을 정리한 학회 보고다.
- **우리와의 차별/활용**: 김승희 박사의 ML 관련 이력 확인용이다.
- 링크(무료 열람, 자동 차단): https://doi.org/10.1175/BAMS-D-23-0332.1

### 📄 `lee2024_antarctic_ice_tongue_area_sentinel1_gee` [협업자]
**Google Earth Engine 의 Sentinel-1 SAR 를 활용한 남극 빙설면적 변화 모니터링 (Assessment of Antarctic ice tongue areas using Sentinel-1 SAR on Google Earth Engine)**. Lee, N.-M., Kim, S. H. & Kim, H.-C. (2024, *대한원격탐사학회지* 40(3), 285–293). doi:10.7780/kjrs.2024.40.3.5
- **무엇** [초록]: GEE 에서 Sentinel-1 SAR 영상을 Otsu 이진화하고 월 평균으로 오탐을 줄여 Campbell 빙설(CGT)과 Drygalski 빙설(DIT)의 면적 변화를 추적한다. CGT 면적은 2016-01 에서 2024-01 까지 약 26 % 줄었고(주로 분리 사건), DIT 는 전체적으로 소폭 늘었다. Sentinel-2 광학 영상으로 검증한다.
- **우리와의 차별/활용**: GEE 기반 SAR 시계열 처리 경험의 예다.
- 파일: `references/10_kopri_collaborators/seung_hee_kim/lee2024_antarctic_ice_tongue_area_sentinel1_gee.pdf`

### 🔗 `kwon2023_passive_microwave_snow_density_grain_tgrs` [협업자]
**Sensitivity of passive microwave satellite observations to snow density and grain size over Arctic sea ice**. Kwon, Y.-J., Kim, H.-C., Kim, J.-M., Park, J.-W. & Kim, S. H. (2023, *IEEE Trans. Geosci. Remote Sens.* 61, 1–10). doi:10.1109/TGRS.2023.3322401
- **무엇** [제목]: 북극 해빙 위 적설 밀도·입경에 대한 수동 마이크로파 관측의 민감도를 분석한다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 링크(구독 필요): https://doi.org/10.1109/TGRS.2023.3322401

### 🔗 `kim2023_ice_shelf_hydrostatic_flexure_igarss` [협업자]
**Stability assessment of Antarctic ice shelves using hydrostatic flexure**. Kim, S. H. (2023, *IGARSS 2023*, 199–202). doi:10.1109/IGARSS52108.2023.10281976
- **무엇** [제목]: 정수압 휨으로 남극 빙붕 안정성을 평가한다. 아래 KOPRI 보고서(PE22320)와 같은 주제다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 링크(구독 필요): https://doi.org/10.1109/IGARSS52108.2023.10281976

### 📄 `jeong2023_russell_glacier_uav_lidar_dataset` [협업자]
**그린란드 러셀빙하 소형무인항공기 라이다 센서 기반 정밀 지형 데이터셋 (Small unmanned aerial vehicle LiDAR-based high spatial resolution topographic dataset in Russell Glacier, Greenland)**. Jeong, Y., Lee, S., Kim, S. H. & Kim, H.-C. (2023, *GEO DATA* 5(1), 1–7). doi:10.22761/GD.2023.0006
- **무엇** [제목·1쪽]: 그린란드 러셀빙하에서 소형 무인기 라이다로 정밀 지형 자료를 만든 자료 논문이다(저자 소속 KOPRI 원격탐사빙권정보센터).
- **우리와의 차별/활용**: 콘슬 무인기 라이다 자료(센터 과제)의 처리 방식과 정확도를 이해하는 데 쓴다.
- 파일: `references/10_kopri_collaborators/seung_hee_kim/jeong2023_russell_glacier_uav_lidar_dataset.pdf`

### 📄 `lee2023_greenland_glacier_fixedwing_uav_gis` [협업자]
**고정익 무인항공기를 이용한 그린란드 지형정보 데이터 구축 (Establishment of geographic information data of Greenland glacier using fixed-wing unmanned aerial vehicle)**. Lee, S., Kim, S. H. & Kim, H.-C. (2023, *GEO DATA* 5(1), 34–39). doi:10.22761/GD.2023.0007
- **무엇** [제목·1쪽]: 고정익 무인기로 그린란드 빙하 지형정보 자료를 구축한 자료 논문이다.
- **우리와의 차별/활용**: 무인기 지형 자료의 구축 절차 참고용이다.
- 파일: `references/10_kopri_collaborators/seung_hee_kim/lee2023_greenland_glacier_fixedwing_uav_gis.pdf`

### 🔗 `han2022_campbell_glacier_tongue_decadal_giscience` [협업자]
**Decadal changes of Campbell Glacier Tongue in East Antarctica from 2010 to 2020 and implications of ice pinning conditions analyzed by optical and SAR datasets**. Han, H., Kim, S. H. & Kim, S. (2022, *GIScience Remote Sens.* 59, 705–721). doi:10.1080/15481603.2022.2055380
- **무엇** [제목]: 광학·SAR 자료로 Campbell 빙설의 2010–2020년 변화와 고정(pinning) 조건을 분석한다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 링크(OA CC BY-NC 4.0, 자동 차단): https://doi.org/10.1080/15481603.2022.2055380

### 📄 `kim2022_landfast_ice_jangbogo_insar_coherence` [협업자]
**Sentinel-1 영상레이더 간섭 긴밀도 영상의 레이어 병합을 활용한 남극 장보고 과학기지 주변 정착해빙 탐지**. Kim, S. H. & Han, H. (2022, *지질공학* 32(2), 271–280). doi:10.9720/kseg.2022.2.271
- **무엇** [초록]: 시간 기선 6·12·18일 Sentinel-1 간섭 긴밀도 영상을 겹쳐 장보고기지 주변 정착해빙을 탐지하고 2017-07 부터 2018-06 까지 정착빙 지도 50장을 만든다. 신생·평탄 해빙의 낮은 후방산란과 미세 이동이 한계다.
- **우리와의 차별/활용**: 간섭 긴밀도 처리 기술은 콘슬 주변 InSAR 공변량 산출에 그대로 쓰인다.
- 파일: `references/10_kopri_collaborators/seung_hee_kim/kim2022_landfast_ice_jangbogo_insar_coherence.pdf`

### 📄 `park2021_kompsat5_cosmo_sea_ice_drift` [협업자]
**Feasibility study on estimation of sea ice drift from KOMPSAT-5 and COSMO-SkyMed SAR images**. Park, J.-W., Kim, H.-C., Korosov, A., Demchev, D., Zecchetto, S., Kim, S. H., Kwon, Y.-J., Han, H. & Hyun, C.-U. (2021, *Remote Sens.* 13, 4038). doi:10.3390/rs13204038
- **무엇** [초록]: MOSAiC 원정 자료로 KOMPSAT-5 와 COSMO-SkyMed X밴드 SAR 의 해빙 이동 추정 가능성을 7개월 이상의 영상쌍·부이 대응 자료로 평가하고 두 센서 교차 결합을 시험한다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 파일: `references/10_kopri_collaborators/seung_hee_kim/park2021_kompsat5_cosmo_sea_ice_drift.pdf`

### 📄 `kwon2021_amsr2_6ghz_min_tb_polar_oceans` [협업자]
**Spatial and temporal variability of minimum brightness temperature at the 6.925 GHz band of AMSR2 for the Arctic and Antarctic oceans**. Kwon, Y.-J., Hong, S., Park, J.-W., Kim, S. H., Kim, J.-M. & Kim, H.-C. (2021, *Remote Sens.* 13, 2122). doi:10.3390/rs13112122
- **무엇** [초록]: AMSR2 2012–2020년 자료로 극지 해수의 6.925 GHz 최소 밝기온도를 추정하고, 해수면 온도·풍속·수증기로 매개변수화한 복사 전달 모형 값과 비교한다. 연 평균은 일정하나 계절 변동은 북극과 남극이 다르다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 파일: `references/10_kopri_collaborators/seung_hee_kim/kwon2021_amsr2_6ghz_min_tb_polar_oceans.pdf`

### 📄 `han2020_summer_arctic_sea_ice_xband_dualpol_roughness` [협업자]
**Surface roughness signatures of summer arctic snow-covered sea ice in X-band dual-polarimetric SAR**. Han, H. et al. (Kim, S. H. 포함 9명) (2020, *GIScience Remote Sens.* 57, 650–669). doi:10.1080/15481603.2020.1767857
- **무엇** [초록]: 2017·2018년 8월 척치해 주변 빙역의 적설 일년빙에서 지상 레이저 스캐너로 4 cm DSM 을 만들어 RMS 높이를 구하고, X밴드 이중 편파(HH, VV) SAR 신호와 표면 거칠기의 관계를 분석한다.
- **우리와의 차별/활용**: 현장 정밀 지형과 SAR 신호를 대응시키는 검증 설계의 예다. 파일은 KOPRI 저장소 공개본이다.
- 파일: `references/10_kopri_collaborators/seung_hee_kim/han2020_summer_arctic_sea_ice_xband_dualpol_roughness.pdf`

### 📄 `kim2020_multiyear_sea_ice_backscatter_beaufort` [협업자]
**Evolution of backscattering coefficients of drifting multi-year sea ice during end of melting and onset of freeze-up in the western Beaufort Sea**. Kim, S. H. et al. (11명) (2020, *Remote Sens.* 12, 1378). doi:10.3390/rs12091378
- **무엇** [초록]: 2019-08 현장에서 설치한 GPS 추적기 중심 1 km × 1 km 영역을 Sentinel-1 영상 24장(17일)으로 추적해 융빙 말·결빙 초 다년빙의 후방산란을 분석한다. 입사각 의존도는 HH −0.24 dB/deg, HV −0.10 dB/deg 이고, HH 정규화 후방산란은 하루 0.15 dB 씩 증가했다.
- **우리와의 차별/활용**: 현장 관측과 위성 시계열을 묶는 설계 경험의 예다.
- 파일: `references/10_kopri_collaborators/seung_hee_kim/kim2020_multiyear_sea_ice_backscatter_beaufort.pdf`

### 📄 `han2019_larsenc_iceberg_a68` [협업자]
**Changes in a giant iceberg created from the collapse of the Larsen C Ice Shelf, Antarctic Peninsula, derived from Sentinel-1 and CryoSat-2 data**. Han, H., Lee, S., Kim, J.-I., Kim, S. H. & Kim, H.-C. (2019, *Remote Sens.* 11, 404). doi:10.3390/rs11040404
- **무엇** [초록]: 2017-07 Larsen C 빙붕에서 떨어진 A68A 빙산의 면적, 표류 속도, 회전, 건현을 Sentinel-1 SAR 와 CryoSat-2 고도계로 1.5년간 추적한다. 면적 감소는 2 % 에 그쳤다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 파일: `references/10_kopri_collaborators/seung_hee_kim/han2019_larsenc_iceberg_a68.pdf`

### 🔗 `kang2019_submarine_groundwater_airborne_tir_hydrolprocess` [협업자]
**Quantitative estimation of submarine groundwater discharge using airborne thermal infrared data acquired at two different tidal heights**. Kang, K. et al. (Kim, S. H. 포함 10명) (2019, *Hydrol. Process.* 33, 1089–1100). doi:10.1002/hyp.13387
- **무엇** [제목]: 두 조위의 항공 열적외선 자료로 해저 지하수 유출을 정량 추정한다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 링크(구독 필요): https://doi.org/10.1002/hyp.13387

### 📄 `kim2018_thwaites_ice_rumple_degradation` [협업자]
**Progressive degradation of an ice rumple in the Thwaites Ice Shelf, Antarctica, as observed from high-resolution digital elevation models**. Kim, S. H., Kim, D. & Kim, H.-C. (2018, *Remote Sens.* 10, 1236). doi:10.3390/rs10081236
- **무엇** [초록]: 2011–2013년 TanDEM-X 로 Thwaites 빙붕의 고해상도 DEM 과 변형 지도를 만들어 아이스 럼플의 고도 변화를 추적한다. 럼플 열화는 빙붕이 얇아져 해저면 접촉을 잃은 결과일 수 있고, 점탄성 변형 모형으로 해석한다.
- **우리와의 차별/활용**: DEM 기반 변화 탐지와 물리 모형 해석을 결합한 예다.
- 파일: `references/10_kopri_collaborators/seung_hee_kim/kim2018_thwaites_ice_rumple_degradation.pdf`

### 📄 `kim2018_campbell_glacier_grounding_line_dem` [협업자]
**고해상도 DEM 을 활용한 로스해 Campbell 빙하의 지반접지선 추정**. 김승희, 김덕진 & 김현철 (2018, *대한원격탐사학회지* 34(3), 545–552). doi:10.7780/kjrs.2018.34.3.9
- **무엇** [초록]: 2013·2016년 TanDEM-X·TerraSAR-X 단일 패스 간섭 자료와 15일 이내 CryoSat-2 고도계 자료로 고해상도 DEM 을 만들어 Campbell 빙하의 접지선을 추정한다. 3년 동안 접지선 위치의 유의한 변화는 없었다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 파일: `references/10_kopri_collaborators/seung_hee_kim/kim2018_campbell_glacier_grounding_line_dem.pdf`

### 📄 `kim2017_tandemx_cryosat2_dem_thwaites_grounding_line` [협업자]
**Combined usage of TanDEM-X and CryoSat-2 for generating a high resolution digital elevation model of fast moving ice stream and its application in grounding line estimation**. Kim, S. H. & Kim, D. (2017, *Remote Sens.* 9, 176). doi:10.3390/rs9020176
- **무엇** [초록]: Thwaites 빙하 가장자리의 지형을 TanDEM-X 간섭으로 추출하고, 간섭 DEM 의 절대 고도 오프셋을 CryoSat-2 고도와의 선형 최소제곱으로 보정해 접지선 추정에 쓴다.
- **우리와의 차별/활용**: DEM 생성·보정 기술은 콘슬 주변 지형 공변량 처리에 참고한다.
- 파일: `references/10_kopri_collaborators/seung_hee_kim/kim2017_tandemx_cryosat2_dem_thwaites_grounding_line.pdf`

### 🔗 `kim2017_intertidal_flat_airborne_sar_tandemx_igarss` [협업자]
**Intertidal flat topographies measured by long-baseline airborne SAR and TanDEM-X**. Kim, D., Choi, C., Jung, J., Kang, K., Kim, S. H. & Hwang, J.-H. (2017, *IGARSS 2017*, 2987–2990). doi:10.1109/IGARSS.2017.8127627
- **무엇** [제목]: 장기선 항공 SAR 와 TanDEM-X 로 갯벌 지형을 측정한다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 링크(구독 필요): https://doi.org/10.1109/IGARSS.2017.8127627

### 🔗 `xu2016_tidal_flat_waterline_seasonal_ecss` [협업자]
**Estimation of seasonal topographic variation in tidal flats using waterline method: a case study in Gomso and Hampyeong Bay, South Korea**. Xu, Z., Kim, D., Kim, S. H., Cho, Y.-K. & Lee, S.-G. (2016, *Estuar. Coast. Shelf Sci.* 183, 213–220). doi:10.1016/j.ecss.2016.10.026
- **무엇** [제목]: 수선법으로 곰소만·함평만 갯벌의 계절 지형 변화를 추정한다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 링크(Crossref VoR 라이선스 CC BY-NC-ND 4.0, 자동 차단): https://doi.org/10.1016/j.ecss.2016.10.026

### 🔗 `lee2016_submarine_groundwater_jeju_tir_hydrolprocess` [협업자]
**Submarine groundwater discharge revealed by aerial thermal infrared imagery: a case study on Jeju Island, Korea**. Lee, E., Kang, K., Hyun, S. P., Lee, K.-Y., Yoon, H. & Kim, S. H. (2016, *Hydrol. Process.* 30, 3494–3506). doi:10.1002/hyp.10868
- **무엇** [제목]: 항공 열적외선 영상으로 제주도 해저 지하수 유출을 탐지한다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 링크(구독 필요): https://doi.org/10.1002/hyp.10868

### 🔗 `kang2016_doppler_tropical_cyclone_scansar_tgrs` [협업자]
**Doppler velocity characteristics during tropical cyclones observed using ScanSAR raw data**. Kang, K., Kim, D., Kim, S. H. & Moon, W. M. (2016, *IEEE Trans. Geosci. Remote Sens.* 54, 2343–2355). doi:10.1109/TGRS.2015.2499443
- **무엇** [제목]: ScanSAR 원시 자료로 열대저기압 해면의 도플러 속도 특성을 분석한다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 링크(구독 필요): https://doi.org/10.1109/TGRS.2015.2499443

### 📄 `hwang2016_fmcw_radar_multifrequency` [협업자]
**다중 대역폭을 갖는 FMCW 레이더 설계 및 제작**. Hwang, J., Kim, S. H., Kang, K. & Kim, D. (2016, *한국전자파학회논문지* 27(4), 377–387). doi:10.5515/KJKIEES.2016.27.4.377
- **무엇** [초록]: 300 MHz 와 500 MHz 대역폭 톱니파를 조합한 X밴드 FMCW 영상 레이더를 설계·제작하고 성능을 시험한다.
- **우리와의 차별/활용**: 연구 이력 파악용이다(레이더 하드웨어 경험).
- 파일: `references/10_kopri_collaborators/seung_hee_kim/hwang2016_fmcw_radar_multifrequency.pdf`

### 🔗 `kim2016_intertidal_flat_longbaseline_insar_igarss` [협업자]
**Measurements of intertidal flat topography using a long-baseline airborne interferometric SAR**. Kim, D. et al. (Kim, S. H. 포함 6명) (2016, *IGARSS 2016*, 7655–7658). doi:10.1109/IGARSS.2016.7730996
- **무엇** [제목]: 장기선 항공 간섭 SAR 로 갯벌 지형을 측정한다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 링크(구독 필요): https://doi.org/10.1109/IGARSS.2016.7730996

### 🔗 `kim2016_airborne_sar_interferometry_tidal_flat_ursi` [협업자]
**A long baseline airborne SAR interferometry for tidal flat mapping**. Kim, D. et al. (Kim, S. H. 포함 6명) (2016, *URSI AP-RASC 2016*, 350–352). doi:10.1109/URSIAP-RASC.2016.7601339
- **무엇** [제목]: 장기선 항공 SAR 간섭으로 갯벌 지도를 만든다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 링크(구독 필요): https://doi.org/10.1109/URSIAP-RASC.2016.7601339

### 📄 `kim2015_thwaites_ice_shelf_disintegration` [협업자]
**Disintegration and acceleration of Thwaites Ice Shelf on the Amundsen Sea revealed from remote sensing measurements**. Kim, J.-W., Kim, D., Kim, S. H., Ha, H. K. & Lee, S. H. (2015, *GIScience Remote Sens.* 52, 498–509). doi:10.1080/15481603.2015.1041766
- **무엇** [제목]: 원격탐사 관측으로 아문센해 Thwaites 빙붕의 붕괴와 흐름 가속을 보인다.
- **우리와의 차별/활용**: 연구 이력 파악용이다. 파일은 KOPRI 저장소 공개본이다.
- 파일: `references/10_kopri_collaborators/seung_hee_kim/kim2015_thwaites_ice_shelf_disintegration.pdf`

### 📄 `kim2015_airborne_remote_sensing_coastal` [협업자]
**Development of a cost-effective airborne remote sensing system for coastal monitoring**. Kim, D., Jung, J., Kang, K., Kim, S. H., Xu, Z., Hensley, S., Swan, A. & Duersch, M. (2015, *Sensors* 15, 25366–25384). doi:10.3390/s151025366
- **무엇** [초록]: SAR 와 열적외선 센서를 실은 저비용 항공 원격탐사 시스템을 구축하고, 갯벌 지형, 연안 표층 해류, 해수면 온도 관측을 위한 보정 기법과 지구물리 모형 알고리즘을 개발한다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 파일: `references/10_kopri_collaborators/seung_hee_kim/kim2015_airborne_remote_sensing_coastal.pdf`

### 📄 `kang2014_coastal_sst_ground_tir_sensor` [협업자]
**지상용 열적외선 센서의 항공기 탑재를 통한 연안 해수면 온도 추출**. Kang, K., Kim, D., Kim, S. H., Cho, Y.-K. & Lee, S.-H. (2014, *대한원격탐사학회지* 30(6), 797–807). doi:10.7780/kjrs.2014.30.6.10
- **무엇** [초록]: 지상용 열적외선 센서를 항공기에 실어 해안선이 복잡한 한반도 연안의 고해상도 해수면 온도 추출 가능성을 검증한다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 파일: `references/10_kopri_collaborators/seung_hee_kim/kang2014_coastal_sst_ground_tir_sensor.pdf`

### 📄 `xu2013_yoobudo_tidal_flat_topography_scattering` [협업자]
**원격탐사자료를 이용한 인공구조물 건설에 의한 군산 유부도 조간대의 지형변화 및 표면특성 연구**. Xu, Z., Kim, D. & Kim, S. H. (2013, *대한원격탐사학회지* 29(1), 57–68). doi:10.7780/kjrs.2013.29.1.6
- **무엇** [초록]: 1998–2012년 Landsat TM/ETM+ 수선 추출로 유부도 주변 지형 변화를 구하고, RADARSAT-2 완전 편파 자료에 Freeman-Durden 분해를 적용해 퇴적면 산란 특성을 분석한다. 퇴적 면적은 4.5 km² 이상이다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 파일: `references/10_kopri_collaborators/seung_hee_kim/xu2013_yoobudo_tidal_flat_topography_scattering.pdf`

### 🔗 `xu2013_sea_ice_thickness_laser_icebreaker_jcre` [협업자]
**Extraction of sea ice thickness using a laser rangefinder mounted on an icebreaker**. Xu, Z., Kim, D. & Kim, S. H. (2013, *J. Cold Reg. Eng.* 27, 183–195). doi:10.1061/(ASCE)CR.1943-5495.0000062
- **무엇** [제목]: 쇄빙선에 실은 레이저 거리계로 해빙 두께를 추출한다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 링크(구독 필요): https://doi.org/10.1061/(ASCE)CR.1943-5495.0000062

### 📄 `kim2013_airborne_marine_meteorology_system` [협업자]
**연안 해양기상(해상풍, 수온) 관측을 위한 항공 원격탐사 시스템 개발**. Kim, D., Cho, Y.-K., Kang, K., Kim, J.-W. & Kim, S.-H. (2013, *바다* 18(1), 32–39). doi:10.7850/jkso.2013.18.1.32
- **무엇** [1쪽 요약]: SAR 센서와 열적외선 센서, GPS·IMU·온습도계로 항공 원격탐사 시스템을 구축해 연안 해상풍과 수온 관측에 쓴다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 파일: `references/10_kopri_collaborators/seung_hee_kim/kim2013_airborne_marine_meteorology_system.pdf`

### 📄 `kim2012_iceberg_detection_radarsat2_west_antarctica` [협업자]
**고해상도 다중편파 RADARSAT-2 SAR 자료를 이용한 서남극해의 빙산 탐지**. Kim, J.-W., Kim, D., Kim, S.-H., Hwang, B.-J. & Yackel, J. (2012, *대한원격탐사학회지* 28(1), 21–28). doi:10.7780/kjrs.2012.28.1.021
- **무엇** [초록]: 서남극 Wilkinson 빙하 주변에서 C밴드 완전 편파 RADARSAT-2 자료에 Freeman-Durden 분해, H/A/α 분해, Wishart 무감독 분류를 적용해 빙산을 탐지한다. [1−H][1−A] 결합 매개변수가 해빙과 구별되지 않던 빙산의 분리 가능성을 보였다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 파일: `references/10_kopri_collaborators/seung_hee_kim/kim2012_iceberg_detection_radarsat2_west_antarctica.pdf`

### 🔗 `kim2011_iceberg_detection_radarsat2_apsar` [협업자]
**Iceberg detection using full-polarimetric RADARSAT-2 SAR data in west Antarctica**. Kim, J.-W. et al. (2011, *3rd APSAR*). DOI [미확인]
- **무엇** [제목]: 위 2012년 논문과 같은 주제의 학술대회 발표다(Google Scholar 프로필 기재).
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 링크(구독 필요, DOI [미확인]): 서지만 기록

### 🔗 `becker2009_srtm30_plus_marine_geodesy` [협업자]
**Global bathymetry and elevation data at 30 arc seconds resolution: SRTM30_PLUS**. Becker, J. J. et al. (Kim, S.-H. 포함 18명) (2009, *Mar. Geod.* 32, 355–371). doi:10.1080/01490410903297766
- **무엇** [제목]: 30초 해상도 전 지구 수심·고도 자료 SRTM30_PLUS 를 기술한다.
- **우리와의 차별/활용**: 귀속은 추정이다(Google Scholar 프로필 기재, UCSD 학부 기간과 Scripps 공저). 본인 확인 전까지 이력 목록에서 따로 표시한다.
- 링크(구독 필요): https://doi.org/10.1080/01490410903297766

### 📄 `kim2023_kopri_report_ice_shelf_stability_deflection` [협업자]
**휨 정도를 활용한 남극 빙붕 안정성 평가 방안 연구 (Stability assessment of Antarctic ice shelves using deflection)**. 김승희 (2023, KOPRI 연구보고서 PE22320, 과제 기간 2022-03-01–2022-12-31). KOPRI 저장소 201206/14515
- **무엇** [제목]: 빙붕의 휨 정도로 안정성을 평가하는 방안을 연구한 기관 고유 과제 보고서다(35쪽). 한글 글꼴 인코딩 때문에 텍스트 추출이 깨진다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 파일: `references/10_kopri_collaborators/seung_hee_kim/kim2023_kopri_report_ice_shelf_stability_deflection.pdf`

학술대회 초록 9편(KOPRI 저장소, 각 2–4쪽)은 아래에 같은 형식으로 짧게 둔다. 모두 [협업자]이고 해빙·빙붕 주제다.

### 📄 `kim2020_abs_melt_pond_fraction_sar` [협업자]
**영상레이더 활용 북극 해빙 융빙호 분포 비율 추정 연구**. 김승희, 김현철 (2020, 대한원격탐사학회 학술대회). KOPRI 저장소 201206/12259
- **무엇** [제목]: SAR 로 북극 해빙 융빙호 비율을 추정한다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 파일: `references/10_kopri_collaborators/seung_hee_kim/abstracts/kim2020_abs_melt_pond_fraction_sar.pdf`

### 📄 `kim2020_abs_incidence_angle_multiyear_ice_beaufort_miz` [협업자]
**Incidence angle dependence and evolution of drifting multi-year sea ice during the onset of freeze-up in the marginal ice zone of the western Beaufort Sea**. Kim, S. H. et al. (2020, AGU 2020 초록 제출본). KOPRI 저장소 201206/12268
- **무엇** [제목]: 위 Remote Sens. 2020 논문과 같은 자료의 학회 초록이다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 파일: `references/10_kopri_collaborators/seung_hee_kim/abstracts/kim2020_abs_incidence_angle_multiyear_ice_beaufort_miz.pdf`

### 📄 `kim2020_proc_incidence_angle_arctic_sea_ice_sar` [협업자]
**영상레이더 활용 북극 해빙 관측을 위한 입사각 특성 연구**. 김승희 외 (2020, 대한조선학회 극지기술연구회 하계 발표회). KOPRI 저장소 201206/12350
- **무엇** [제목]: SAR 해빙 관측의 입사각 특성을 분석한다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 파일: `references/10_kopri_collaborators/seung_hee_kim/abstracts/kim2020_proc_incidence_angle_arctic_sea_ice_sar.pdf`

### 📄 `kim2020_proc_sentinel1_multiyear_ice_freezeup` [협업자]
**융빙기와 결빙기 간 다년생 북극 해빙 Sentinel-1 C밴드 SAR 신호 특성 연구**. 김승희 외 (2020, 학술대회 발표). KOPRI 저장소 201206/12353
- **무엇** [제목]: 융빙기와 결빙기 사이 다년빙의 Sentinel-1 신호 변화를 분석한다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 파일: `references/10_kopri_collaborators/seung_hee_kim/abstracts/kim2020_proc_sentinel1_multiyear_ice_freezeup.pdf`

### 📄 `kim2019_proc_flexural_strength_depolarization_sentinel1` [협업자]
**Estimation of flexural strength of Arctic sea ice in the marginal ice zone using the depolarization ratio of Sentinel-1**. Kim, S. H., Kim, H.-C., Han, H. & Park, J.-W. (2019, AGU Fall Meeting 2019, C51D-1324). KOPRI 저장소 201206/12492
- **무엇** [제목]: Sentinel-1 편파 해소비로 주변 빙역 해빙의 굽힘강도를 추정한다.
- **우리와의 차별/활용**: 연구 이력 파악용이다. 관련 KPDC 자료(해빙 굽힘강도)가 같은 이름으로 등록되어 있다.
- 파일: `references/10_kopri_collaborators/seung_hee_kim/abstracts/kim2019_proc_flexural_strength_depolarization_sentinel1.pdf`

### 📄 `kim2019_proc_flexural_strength_polsar` [협업자]
**Flexural strength of Arctic sea ice using spaceborne polarimetric SAR data**. Kim, S. H. et al. (2019, 학술대회 발표). KOPRI 저장소 201206/12690
- **무엇** [제목]: 위성 편광 SAR 로 북극 해빙 굽힘강도를 추정한다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 파일: `references/10_kopri_collaborators/seung_hee_kim/abstracts/kim2019_proc_flexural_strength_polsar.pdf`

### 📄 `kim2019_proc_ice_rumple_thinning` [협업자]
**Estimation of ice shelf thinning derived from surface depression of an ice rumple**. Kim, S. H., Kim, D. & Kim, H.-C. (2019, 구두 발표 초록집 발췌). KOPRI 저장소 201206/12499
- **무엇** [제목]: 아이스 럼플 표면 함몰로 빙붕 박화를 추정한다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 파일: `references/10_kopri_collaborators/seung_hee_kim/abstracts/kim2019_proc_ice_rumple_thinning.pdf`

### 📄 `han2019_proc_iceberg_a68_monitoring` [협업자]
**Monitoring iceberg A68 calved from the Larsen C Ice Shelf using satellite remote sensing**. Han, H., Lee, S., Kim, J.-I., Kim, S. H. & Kim, H.-C. (2019, 초록집 발췌). KOPRI 저장소 201206/12477
- **무엇** [제목]: 위성 원격탐사로 A68 빙산을 감시한다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 파일: `references/10_kopri_collaborators/seung_hee_kim/abstracts/han2019_proc_iceberg_a68_monitoring.pdf`

### 📄 `han2019_proc_iceberg_a68_initial_evolution` [협업자]
**인공위성 관측으로 분석된 거대 빙산 A-68 의 초기 진화**. 한향선 외 (2019, 한국지구과학연합회). KOPRI 저장소 201206/12471
- **무엇** [제목]: 위성 관측으로 A-68 빙산의 초기 진화를 분석한다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 파일: `references/10_kopri_collaborators/seung_hee_kim/abstracts/han2019_proc_iceberg_a68_initial_evolution.pdf`

**10.2 정윤택 (Yoon Taek Jung)**

### 🔗 `jung2025_phd_thesis_polinsar_boreal_deformation_sejong` [협업자]
**시계열 편파/간섭 SAR 기법을 이용한 한대림 지역 지반 변형 모니터링 (Monitoring ground deformation in boreal forest using time-series polarimetric SAR interferometry)**. 정윤택 (2025, 박사학위논문, 세종대학교 대학원 지구정보공학과, 지도교수 박상은; 학위 수여 2026-02). UCI I804:11042-200000962018
- **무엇** [공개 초록]: 동결기에는 C밴드 Sentinel-1 시계열 InSAR 로 융기를 추정하고 시계열 패턴으로 활동층 수문 특성을 간접 추정한다. 해빙기에는 간섭 긴밀도가 낮아 기존 InSAR 가 침하 추정에 실패하므로, L밴드 ALOS-2 다중 편파 자료와 시간 변화를 고려한 RVoG 모형 모의 자료로 학습한 딥러닝 역산(Physics-informed deep learning inversion)으로 지표 위상을 복원한다. 연구 지역은 야쿠츠크 인근 중앙 저지대(연속 영구동토, 침엽수림)다.
- **우리와의 차별/활용**: 물리 모형 출력을 학습 신호로 쓰는 구조가 우리 물리 증강·잔차 ML 설계와 가깝다. 공동 원고의 방법 용어를 맞출 때 먼저 읽는다.
- 링크(웹 열람, dCollection 뷰어): https://sejong.dcollection.net/srch/srchDetail/200000962018

### 📄 `lee2024_central_yakutia_landsat_landcover` [협업자]
**Monitoring long-term land cover change in central Yakutia using sparse time series Landsat data**. Lee, Y., Kim, S.-Y., Jung, Y. T. & Park, S.-E. (2024, *Remote Sens.* 16, 1868). doi:10.3390/rs16111868
- **무엇** [초록]: 시간 간격이 성긴 Landsat 시계열에 새 분류 후 변화 분석 방식을 적용해 중앙 야쿠티아 한대 생태계의 서로 다른 시공간 규모 경관 변화를 관측한다. KOPRI 과제 PE21900 지원 논문이다.
- **우리와의 차별/활용**: 중앙 야쿠티아 토지피복·열카르스트 변화 자료로 Syrdakh 지역 공변량 해석에 참고한다.
- 파일: `references/10_kopri_collaborators/yoon_taek_jung/lee2024_central_yakutia_landsat_landcover.pdf`

### 🔗 `jung2023_yakutia_seasonal_deformation_insar_rse` [협업자]
**Observation of spatial and temporal patterns of seasonal ground deformation in central Yakutia using time series InSAR data in the freezing season**. Jung, Y. T., Park, S.-E. & Kim, H.-C. (2023, *Remote Sens. Environ.* 293, 113615). doi:10.1016/j.rse.2023.113615
- **무엇** [초록, KOPRI 저장소 메타데이터]: 동결기 Sentinel-1 SBAS-InSAR 변위 시계열을 Stefan 해에 근거한 3구간 선형 모형으로 나누고, 구간별 기울기를 활동층 상·중·하부의 수문 특성으로 해석한다.
- **우리와의 차별/활용**: 우리 물리 기준선(Stefan 식, ALT ∝ √TDD)과 같은 물리를 관측 변위 해석에 쓴다. ALT 로 학습하지 않은 원시 계절 변위는 공변량 규칙에 걸리지 않으므로 협의의 핵심 문헌이다. 저자 원고 공유를 요청한다.
- 링크(구독 필요): https://doi.org/10.1016/j.rse.2023.113615

### 🔗 `jung2023_cryogenic_process_polarimetry_interferometry_igarss` [협업자]
**Observation of cryogenic process using polarimetry and interferometry**. Jung, Y. T., Lee, Y., Kim, M. & Park, S.-E. (2023, *IGARSS 2023*, 229–232). doi:10.1109/IGARSS52108.2023.10282761
- **무엇** [제목]: 편파와 간섭 SAR 로 동결 과정을 관측한다.
- **우리와의 차별/활용**: 동결기 PolSAR 신호 해석의 학회판이다.
- 링크(구독 필요): https://doi.org/10.1109/IGARSS52108.2023.10282761

### 🔗 `lee2023_boreal_permafrost_landsat_trends_igarss` [협업자]
**Monitoring long-term change trends in boreal permafrost area using time series Landsat**. Lee, Y., Jung, Y. T., Kim, S.-Y. & Park, S.-E. (2023, *IGARSS 2023*, 233–236). doi:10.1109/IGARSS52108.2023.10282420
- **무엇** [제목]: 시계열 Landsat 으로 한대 영구동토 지역의 장기 변화 경향을 감시한다.
- **우리와의 차별/활용**: 위 Remote Sens. 2024 논문의 학회판이다.
- 링크(구독 필요): https://doi.org/10.1109/IGARSS52108.2023.10282420

### 🔗 `park2022_yakutia_permafrost_optical_polsar_rse` [협업자]
**Monitoring permafrost changes in central Yakutia using optical and polarimetric SAR data**. Park, S.-E., Jung, Y. T. & Kim, H.-C. (2022, *Remote Sens. Environ.* 274, 112989). doi:10.1016/j.rse.2022.112989
- **무엇** [제목]: 광학·편광 SAR 자료로 중앙 야쿠티아 영구동토 변화를 감시한다. KOPRI 저장소(201206/16163)에 원격탐사빙권정보센터 결과물로 등록되어 있다.
- **우리와의 차별/활용**: 열카르스트 변화와 SAR 산란 특성의 관계를 이해하는 데 쓴다. 저자 원고 공유를 요청한다.
- 링크(구독 필요): https://doi.org/10.1016/j.rse.2022.112989

### 🔗 `jung2022_frost_heave_thaw_settlement_insar_egu` [협업자]
**Monitoring frost heave and thaw settlement of permafrost using timeseries InSAR measurements**. Jung, Y. T., Lee, Y. & Park, S.-E. (2022, EGU General Assembly 2022, EGU22-11674). doi:10.5194/egusphere-egu22-11674
- **무엇** [제목]: 시계열 InSAR 로 영구동토의 동상과 융해 침하를 감시한다.
- **우리와의 차별/활용**: RSE 2023 과 박사학위논문 사이의 중간 결과다.
- 링크(웹 열람, 초록 페이지): https://doi.org/10.5194/egusphere-egu22-11674

### 🔗 `lee2022_postfire_optical_sar_egu` [협업자]
**Spatiotemporal post-fire change analysis using optical and SAR imagery**. Lee, Y., Oh, J., Kim, S. Y., Jung, Y. T. & Park, S.-E. (2022, EGU22-12301). doi:10.5194/egusphere-egu22-12301
- **무엇** [제목]: 광학·SAR 영상으로 화재 후 시공간 변화를 분석한다.
- **우리와의 차별/활용**: 우리 화재-ALT 자료(`talucci2025_firealt_paired_burned_essd`, 08 절)와 주제가 겹친다.
- 링크(웹 열람, 초록 페이지): https://doi.org/10.5194/egusphere-egu22-12301

### 🔗 `jung2021_thermokarst_optical_sar_yakutia_igarss` [협업자]
**Combined use of optical and SAR data for thermokarst terrain: a case study in central Yakutia**. Jung, Y. T., Lee, Y., Kim, M. & Park, S.-E. (2021, *IGARSS 2021*, 5589–5591). doi:10.1109/IGARSS47720.2021.9554270
- **무엇** [제목]: 광학과 SAR 자료를 함께 써서 중앙 야쿠티아 열카르스트 지형을 분석한다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 링크(구독 필요): https://doi.org/10.1109/IGARSS47720.2021.9554270

### 🔗 `park2020_kumamoto_damage_indicators_igarss` [협업자]
**Analysis of single-pol and quad-pol damage indicators for extraction of building damages caused by 2016 Kumamoto earthquake**. Park, S.-E., Lee, Y., Kim, M. & Jung, Y. T. (2020, *IGARSS 2020*, 3877–3879). doi:10.1109/IGARSS39084.2020.9324134
- **무엇** [제목]: 단일·완전 편파 피해 지표로 2016년 구마모토 지진 건물 피해를 추출한다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 링크(구독 필요): https://doi.org/10.1109/IGARSS39084.2020.9324134

### 📄 `park2020_polsar_earthquake_building_damage` [협업자]
**Detection of earthquake-induced building damages using polarimetric SAR data**. Park, S.-E. & Jung, Y. T. (2020, *Remote Sens.* 12, 137). doi:10.3390/rs12010137
- **무엇** [초록]: 2016년 구마모토 지진 전후 PALSAR-2 편파 자료로 편파 산란 전력 변화, 행렬 비유사도 등 변화 탐지 방식을 비교해 지반 진동에 의한 건물 피해 탐지 가능성을 평가한다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 파일: `references/10_kopri_collaborators/yoon_taek_jung/park2020_polsar_earthquake_building_damage.pdf`

### 🔗 `park2019_earthquake_damage_polsar_igarss` [협업자]
**Detection of earthquake-induced damages using polarimetric SAR remote sensing**. Park, S.-E., Jung, Y.-T. & Cho, K. (2019, *IGARSS 2019*, 5027–5029). doi:10.1109/IGARSS.2019.8898807
- **무엇** [제목]: 편파 SAR 로 지진 피해를 탐지한다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 링크(구독 필요): https://doi.org/10.1109/IGARSS.2019.8898807

### 📄 `park2019_water_cloud_model_vegetation` [협업자]
**Theoretical evaluation of water cloud model vegetation parameters**. Park, S.-E., Jung, Y. T., Cho, J.-H., Moon, H. & Han, S.-h. (2019, *Remote Sens.* 11, 894). doi:10.3390/rs11080894
- **무엇** [초록]: Water Cloud Model 의 원형과 개선형을 이론 산란 모형 예측과 비교해 최적 식생 기술 변수를 제시하고, 산란체 형태·크기·방향 분포에 따른 A·B 매개변수의 회귀 관계를 분석한다.
- **우리와의 차별/활용**: 식생 피복 아래 SAR 신호 해석의 물리 모형 배경이다.
- 파일: `references/10_kopri_collaborators/yoon_taek_jung/park2019_water_cloud_model_vegetation.pdf`

### 📄 `jung2018_polsar_calibration_comparison` [협업자]
**Comparative analysis of polarimetric SAR calibration methods**. Jung, Y. T. & Park, S.-E. (2018, *Remote Sens.* 10, 2060). doi:10.3390/rs10122060
- **무엇** [초록]: Van Zyl, Quegan, Villa 편파 보정 방법을 합성 자료와 ALOS PALSAR 완전 편파 자료로 비교해 시스템 매개변수와 지표 특성에 따른 보정 성능을 평가한다.
- **우리와의 차별/활용**: PolSAR 자료 전처리 품질을 판단하는 기준 문헌이다.
- 파일: `references/10_kopri_collaborators/yoon_taek_jung/jung2018_polsar_calibration_comparison.pdf`

### 📄 `jung2018_palsar2_quadpol_flood_detection` [협업자]
**Evaluation of polarimetric parameters for flood detection using PALSAR-2 quad-pol data**. Jung, Y. T., Park, S.-E., Baek, C.-S. & Kim, D.-H. (2018, *대한원격탐사학회지* 34(1), 117–126). doi:10.7780/kjrs.2018.34.1.8
- **무엇** [초록]: 일본 조소시 홍수 전후 L밴드 PALSAR-2 완전 편파 자료로 침수 지역 판별에 유용한 편파 매개변수를 평가한다. HH 강도, Shannon 엔트로피, 모형 분해 표면 산란 성분이 유용했고, Shannon 엔트로피 무감독 변화 탐지가 가장 좋았다.
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 파일: `references/10_kopri_collaborators/yoon_taek_jung/jung2018_palsar2_quadpol_flood_detection.pdf`

### 🔗 `jung2018_polsar_land_monitoring_ieice` [협업자]
**On the application of polarimetric SAR remote sensing to land surface environmental monitoring**. Jung, Y.-T. et al. (2018, *IEICE Technical Report*). 상세 서지 [미확인]
- **무엇** [제목]: 편파 SAR 의 지표 환경 감시 응용을 다룬다(Google Scholar 기재).
- **우리와의 차별/활용**: 연구 이력 파악용이다.
- 링크(구독 필요, 상세 서지 [미확인]): 서지만 기록

귀속 제외: 자원환경지질 2019 산사태 탐지 논문(정윤택 저자 아님), MorphEst 2021(Yoonho Jung), 2004년 한국수자원학회 발표(귀속 근거 없음).

**10.3 원격탐사빙권정보센터 과제 보고서 (콘슬 접점 자료)**

### 📄 `kimhc2023_kopri_report_arctic_cryosphere_remote_sensing_phase1` [자료]
**북극 빙권변화 정량 분석을 위한 원격탐사 연구**. 김현철(연구책임자) (2023-02, KOPRI 연구보고서, 과제 번호 2022-00341). KOPRI 저장소 201206/14551
- **무엇** [본문 일부]: 참여연구원에 김승희가 포함된다. 3장에 콘슬 Site 1·2 무인기 비행 경로와 Site 1 라이다 점군·RGB·초분광 모자이크 그림이 있고, 알래스카 배로 인근 열카르스트 호수 분류도 있다(317쪽).
- **우리와의 차별/활용**: 콘슬 사례 분석의 미세 지형 공변량 후보(같은 공변량 셀 안에 ALT 34–96 cm 가 공존하는 문제)의 1차 기록이다.
- 파일: `references/10_kopri_collaborators/_kopri_center_reports/kimhc2023_kopri_report_arctic_cryosphere_remote_sensing_phase1.pdf`

### 📄 `kimhc2025_kopri_report_arctic_cryosphere_remote_sensing` [자료]
**북극 빙권변화 정량 분석을 위한 원격탐사 연구**. 김현철(연구책임자) (2025-12-31, KOPRI 연구보고서 BSPE25040-130-12). KOPRI 저장소 201206/16908
- **무엇** [본문 일부]: 참여연구원에 김승희가 포함되고 정윤택은 없다. "3-2-4. 동토지역 초분광/라이다 자료 구축" 에서 콘슬을 KOPRI 대표 동토 모니터링 사이트(스노우펜스, NDVI, CO2 측정)로 소개하고, 초분광·라이다·RGB 동시 획득과 PPK 후처리(보정 후 위치 오차 0.02–0.05 m)를 기술한다. 성과 목록에 KOPRI-KPDC-00001470 이 있다(507쪽).
- **우리와의 차별/활용**: 콘슬 무인기 자료의 정확도와 이용 조건을 협의하기 전에 읽는다. 김승희 박사가 콘슬 무인기 작업에 직접 참여했는지는 [미확인]이다.
- 파일: `references/10_kopri_collaborators/_kopri_center_reports/kimhc2025_kopri_report_arctic_cryosphere_remote_sensing.pdf`

**10.4 KPDC 식별자 (우리 사용 자료와 센터 산출물)**

| KPDC 식별자 | 제목 (KPDC 표기) | 비고 |
|---|---|---|
| KOPRI-KPDC-00002125 | Permafrost core samples in Council, Alaska, USA, 2022 | 우리 사용. 스노우펜스 6개 × 처리 3종, 코어 18개 |
| KOPRI-KPDC-00002707 | Soil temperature data from active layer of Council, Alaska permafrost site in 2024 | 우리 사용 |
| KOPRI-KPDC-00002955 | Soil temperature data from active layer of Council, Alaska permafrost site in 2025 | 우리 사용 |
| KOPRI-KPDC-00001470 | Multispectral satellite image and digital surface model for Council, AK | 센터 산출물(Pleiades 50 cm 영상·DSM), 우리 미사용 |

세 식별자(00002125, 00002707, 00002955)의 공식 제목은 원고 자료 가용성 절의 [확인 필요] 항목을 채우는 데 쓴다. 김승희 이름으로 검색되는 KPDC 해빙·빙하 자료 12건의 목록은 폴더 README 3.4절에 있다.

## 11 · 검증 설계·통계·ML 방법  (22/31 PDF)

폴더: `references/11_validation_and_methods/` · 원 조사 기록: `references/_index_parts/library_manuscript_refs.md` (원고 `docs/MANUSCRIPT_DRAFT_METHODS_INTRO_2026-09-30.md` 의 References 표 67항목과 NOVELTY 문서 인용 문헌을 대조한 목록)

범위: 공간 검증과 적용 범위, 물리 유도 ML 과 미관측 대상 전이(타 분야 포함), 불확실성 구간과 채점, 표 형식 학습기와 생성 모형, 다중 비교와 동등성 검정. OA 상태 표기는 OpenAlex 분류(gold, hybrid, bronze, green, closed, 2026-10-04 조회)를 따른다.

다른 절에 둔 원고 인용 방법 문헌: `ploton2020_spatial_validation`(09 절), `meyer2021_aoa`·`meyer2022_globalmaps_aoa`·`linnenbrink2024_knndm`·`omalley2026_incontext_subsurface_temp`(05 절), `willard2020_physics_ml_survey`(06 절, 게재본 Willard et al. 2022 ACM Comput. Surv. 55(4), 66, doi:10.1145/3514228), `read2019_pgdl_lake_temp`(03 절), `feng2023_differentiable_hydrology_ungauged_hess`·`tama2025_physics_guided_residual_bed_topography_wacv`(08 절).

**11.1 공간 검증과 적용 범위**

### 🔗 `wadoux2021_spatial_cv_map_accuracy_ecolmodel` [방법]
**Spatial cross-validation is not the right way to evaluate map accuracy**. Wadoux, A. M. J.-C., Heuvelink, G. B. M., de Bruin, S. & Brus, D. J. (2021, *Ecol. Model.* 457, 109692). doi:10.1016/j.ecolmodel.2021.109692
- **무엇** [검색 결과 요지, 원문 미대조]: 표준 교차검증과 공간 교차검증 모두 지도 정확도를 편향 추정할 수 있다고 주장한다. 모형에 의존하지 않는 비편향 정확도 평가는 확률 표본과 설계 기반 추론으로 얻는다고 결론짓는다.
- **우리와의 차별/활용**: 블록 절반 분할 채점이 "지도 전체 정확도"가 아니라 "제외 지역 채점 블록의 예측 오차"임을 명시하는 근거다.
- 링크(구독 필요, WUR 저장소 기록에 파일 없음): https://doi.org/10.1016/j.ecolmodel.2021.109692

### 🔗 `roberts2017_cv_structured_data` [방법]
**Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure**. Roberts, D. R. et al. (2017, *Ecography* 40, 913–929; 온라인 2016). doi:10.1111/ecog.02881
- **무엇**: 구조가 있는 자료에서 무작위 교차검증이 예측 오차를 과소추정하고 비인과 예측 변수의 과적합을 허용함을 보인다. 블록 교차검증을 권하되, 블록이 예측 변수 범위를 제한해 외삽을 유발할 수 있음을 함께 지적한다.
- **우리와의 차별/활용**: 원고 Methods 에서 라벨 블록과 채점 블록 분리의 근거로 인용된다.
- 링크(출판사 무료 열람, Wiley 403 자동 차단): https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/ecog.02881

### 📄 `valavi2019_blockcv` [방법]
**blockCV: an R package for generating spatially or environmentally separated folds for k-fold cross-validation of species distribution models**. Valavi, R., Elith, J., Lahoz-Monfort, J. J. & Guillera-Arroita, G. (2019, *Methods Ecol. Evol.* 10, 225–232; 온라인 2018). doi:10.1111/2041-210X.13107
- **무엇**: 공간 또는 환경으로 분리된 k-겹을 만드는 R 패키지를 제시한다. 공변량의 공간 자기상관 거리 측정, 블록 생성, 겹 시각화 기능을 포함한다.
- **우리와의 차별/활용**: 0.5° 블록 분할의 도구 선례다(원고 Methods 인용 후보). 파일은 bioRxiv 사전 공개본(doi:10.1101/357798 v1, CC BY-NC-ND)이라 게재본과 문구가 다를 수 있다.
- 파일: `references/11_validation_and_methods/valavi2019_blockcv.pdf`

**11.2 물리 유도 ML 과 미관측 대상 전이**

### 📄 `karpatne2017_theory_guided_data_science` [맥락]
**Theory-guided data science: a new paradigm for scientific discovery from data**. Karpatne, A. et al. (2017, *IEEE Trans. Knowl. Data Eng.* 29, 2318–2331). doi:10.1109/TKDE.2017.2720168
- **무엇**: 과학 지식을 데이터 과학 모형에 넣는 이론 유도 데이터 과학(TGDS)을 개념화하고 연구 주제를 분류한다. 과학적 일관성을 일반화 가능한 모형의 필수 요소로 둔다.
- **우리와의 차별/활용**: 물리 증강 ML 용어의 출발점으로 서론 첫 인용 후보다. 파일은 arXiv 판(arXiv:1612.08544)이다.
- 파일: `references/11_validation_and_methods/karpatne2017_theory_guided_data_science.pdf`

### 📄 `jia2021_pgml_lake_temperature` [선례]
**Physics-guided machine learning for scientific discovery: an application in simulating lake temperature profiles**. Jia, X. et al. (2021, *ACM/IMS Trans. Data Sci.* 2(3), 20). doi:10.1145/3447814
- **무엇**: 물리 모형(General Lake Model) 출력과 순환 신경망을 결합한 PGRNN 을 제안한다. 물리 모의값 사전학습과 에너지 보존 손실로 관측이 적을 때도 물리 모형보다 정확하고 물리적으로 일관된 수온 예측을 얻는다.
- **우리와의 차별/활용**: 두 기준선(원천 물리, 같은 라벨로 재보정한 물리)과 비교하는 보고 형식의 직접 선례다(원고 Methods "Following Read et al. (2019) and Jia et al. (2021)"). 파일은 arXiv 판(arXiv:2001.11086)이다.
- 파일: `references/11_validation_and_methods/jia2021_pgml_lake_temperature.pdf`

### 📄 `willard2021_meta_transfer_unmonitored_lakes` [선례]
**Predicting water temperature dynamics of unmonitored lakes with meta-transfer learning**. Willard, J. D. et al. (2021, *Water Resour. Res.* 57, e2021WR029579). doi:10.1029/2021WR029579
- **무엇**: 관측 호수 145곳의 원천 모형 가운데 대상 호수에 맞는 모형을 메타 모형으로 골라 미관측 호수 305곳의 수온을 예측한다. 미보정 과정 모형의 중앙 RMSE 2.52 °C 에 대해 PGDL 메타 전이 앙상블은 1.88 °C 였다.
- **우리와의 차별/활용**: 미관측 대상에서 라벨 수(1–50)에 따른 평가의 인접 분야 선례다(원고 서론, NOVELTY 3절). 파일은 arXiv 판(arXiv:2011.05369)이다.
- 파일: `references/11_validation_and_methods/willard2021_meta_transfer_unmonitored_lakes.pdf`

### 🔗 `pool2019_limited_discharge_regionalization` [선례]
**Value of a limited number of discharge observations for improving regionalization: a large-sample study across the United States**. Pool, S., Viviroli, D. & Seibert, J. (2019, *Water Resour. Res.* 55, 363–377; 온라인 2018). doi:10.1029/2018WR023855
- **무엇**: 미계측 유역에서 단기 관측 3–24개로 다수 공여 유역의 매개변수 예측에 가중을 주는 정보 지역화를 제안한다. 미국 579개 유역 leave-one-out 시험에서 표본 연도 10개 중 8개 연도에 최대 94 % 유역의 지역화가 개선되었다.
- **우리와의 차별/활용**: 라벨 수 축 평가의 직접 선례다(원고 서론, NOVELTY 8절).
- 링크(OA bronze·ZORA 수락본, 자동 차단): https://www.zora.uzh.ch/id/eprint/169290/1/Pool_et_al-2019-Water_Resources_Research.pdf

### 🔗 `oudin2008_regionalization_913_catchments_wrr` [선례]
**Spatial proximity, physical similarity, regression and ungaged catchments: a comparison of regionalization approaches based on 913 French catchments**. Oudin, L., Andréassian, V., Perrin, C., Michel, C. & Le Moine, N. (2008, *Water Resour. Res.* 44, W03413; 논문 번호 [미확인]). doi:10.1029/2007WR006240
- **무엇**: 프랑스 913개 유역에서 회귀, 공간 근접, 물리 유사성 기반 매개변수 지역화를 비교한다. 공간 근접이 가장 낫고, 회귀 방식은 전국 단일 중앙값 매개변수와 거의 같은 수준이다.
- **우리와의 차별/활용**: 공변량 회귀로 만든 계수 지도가 지역 간에 실패한다는 결과(N7)의 수문학 선례다.
- 링크(구독 필요): https://doi.org/10.1029/2007WR006240

### 📄 `staudinger2025_limited_discharge_models` [선례]
**How well do process-based and data-driven hydrological models learn from limited discharge data?** Staudinger, M. et al. (2025, *Hydrol. Earth Syst. Sci.* 29, 5005–5029). doi:10.5194/hess-29-5005-2025
- **무엇**: 독일 중규모 유역 3곳에서 훈련 자료의 수와 선택 방식을 바꿔 과정 기반 모형 3종과 자료 기반 모형 4종의 학습 거동을 비교한다. 결합·조건부 엔트로피 같은 정보 측도로 평가한다.
- **우리와의 차별/활용**: "자료가 적으면 과정 모형이 유리하다"는 통념을 시험한 선례다(NOVELTY 9절).
- 파일: `references/11_validation_and_methods/staudinger2025_limited_discharge_models.pdf`

### 🔗 `portes2026_ml_spatial_transferability` [선례]
**When machine learning extrapolates in space: how local data shape spatial transferability**. Portes, C., Ienco, D. & Gabriel, E. (2026, *Spat. Stat.* 74, 101009). doi:10.1016/j.spasta.2026.101009
- **무엇** [초록]: 프랑스 Xylella fastidiosa 감시 자료(2015–2023)로 공간 외삽 조건의 예측 성능을 분석한다. 미관측 지역에 현지 자료를 단계적으로 넣는 공간 전이 평가로 외삽에서 내삽으로의 비선형 전이를 보인다.
- **우리와의 차별/활용**: "미관측 지역에 현지 자료를 단계 투입하는 평가 최초"를 쓸 수 없게 하는 일반 선례다. 원고 서론의 "spatial prediction in general" 인용이며, 설계 자체의 일반적 신규성을 주장하지 않는 근거다.
- 링크(OA hybrid CC BY·HAL 사전본 hal-05697915, 봇 확인 페이지로 자동 차단): https://doi.org/10.1016/j.spasta.2026.101009

### 🔗 `steyerberg2004_validation_updating_logistic_statmed` [방법]
**Validation and updating of predictive logistic regression models: a study on sample size and shrinkage**. Steyerberg, E. W. et al. (2004, *Stat. Med.* 23, 2567–2586). doi:10.1002/sim.1844
- **무엇**: 다른 기관에서 개발한 예측 모형을 표본 크기별로 검증하고 갱신한다(GUSTO-I 자료). 절편·기울기 재보정 같은 간소한 갱신이 계수 재추정보다 낫고, 구조적 수정은 큰 표본에서만 권한다.
- **우리와의 차별/활용**: 적은 라벨의 재보정이 오차를 키울 수 있다는 원고 Methods 문장과, 원천 계수로 수축하는 P1 재보정의 근거다.
- 링크(구독 필요): https://doi.org/10.1002/sim.1844

### 📄 `tahvildari2026_physics_hybrid_pore_pressure` [선례]
**A physics-informed hybrid ML framework for pore pressure and fracture gradient prediction in carbonate reservoirs**. Tahvildari, S. P., Shojaei, S. & Masihi, M. (2026, *Sci. Rep.* 16, 8925; 논문 번호는 NOVELTY 기록, OpenAlex 미등재). doi:10.1038/s41598-026-41773-z
- **무엇**: 고전 경험식 출력에 적응 보정층과 불확실성 모듈을 결합한 기울기 부스팅 하이브리드로 공극압과 파쇄 압력 구배를 예측한다. 시추공 6곳에서 보정된 고전 모형 RMSE 0.7–1.1 MPa 에 대해 하이브리드는 0.45 MPa 였다.
- **우리와의 차별/활용**: 보정된 물리식과 비교하는 형식의 Sci Rep 선례다(NOVELTY 10절). 영구동토 문헌은 아니다.
- 파일: `references/11_validation_and_methods/tahvildari2026_physics_hybrid_pore_pressure.pdf`

**11.3 불확실성 구간과 채점**

### 📄 `dunn2023_hierarchical_conformal` [방법]
**Distribution-free prediction sets for two-layer hierarchical models**. Dunn, R., Wasserman, L. & Ramdas, A. (2023, *J. Am. Stat. Assoc.* 118, 2491–2502; 온라인 2022). doi:10.1080/01621459.2022.2060112
- **무엇**: 집단마다 분포가 달라 교환 가능성이 깨지는 2층 계층 자료의 conformal 예측 집합을 다룬다. CDF pooling, 단일·반복 부분표본 방식을 제안하고, 커버리지 보장이 필요하면 반복 부분표본을 권한다.
- **우리와의 차별/활용**: 지역을 집단으로 둔 계층 conformal 구간(원고 Methods, N9)의 직접 근거다. 파일은 arXiv 판(arXiv:1809.07441)이다.
- 파일: `references/11_validation_and_methods/dunn2023_hierarchical_conformal.pdf`

### 📄 `romano2019_conformalized_quantile_regression` [방법]
**Conformalized quantile regression**. Romano, Y., Patterson, E. & Candès, E. J. (2019, *NeurIPS* 32, 3538–3548). arXiv:1905.03222
- **무엇**: conformal 예측과 분위 회귀를 결합해 이분산에 적응하는 구간을 만든다. 유한 표본 커버리지를 보장하고 다른 conformal 방법보다 짧은 구간을 보인다.
- **우리와의 차별/활용**: 분위 기반 구간 보정의 표준 참고문헌이다. 현재 원고 초안에는 인용이 없다.
- 파일: `references/11_validation_and_methods/romano2019_conformalized_quantile_regression.pdf`

### 📄 `gneiting2007_proper_scoring_rules` [방법]
**Strictly proper scoring rules, prediction, and estimation**. Gneiting, T. & Raftery, A. E. (2007, *J. Am. Stat. Assoc.* 102, 359–378). doi:10.1198/016214506000001437
- **무엇**: 확률 예보 채점 규칙의 적정성 이론을 일반 확률 공간에서 정리하고 CRPS, 분위 점수, 구간 점수 등을 예로 든다.
- **우리와의 차별/활용**: 원고 Methods 의 CRPS, pinball 손실, 구간 점수 계산 근거다. 파일은 저자(워싱턴대 Raftery) 누리집 공개본이다.
- 파일: `references/11_validation_and_methods/gneiting2007_proper_scoring_rules.pdf`

### 📄 `kakhani2024_conformal_soc` [선례]
**Uncertainty quantification of soil organic carbon estimation from remote sensing data with conformal prediction**. Kakhani, N. et al. (2024, *Remote Sens.* 16, 438). doi:10.3390/rs16030438
- **무엇**: 유럽 LUCAS 토양 자료에서 원격탐사 공변량과 랜덤 포레스트에 conformal prediction 을 결합해 토양 유기탄소 예측 구간을 만들고 기존 불확실성 정량화 방법과 비교한다.
- **우리와의 차별/활용**: 환경 매핑의 conformal 커버리지 선례다(N9 인접).
- 파일: `references/11_validation_and_methods/kakhani2024_conformal_soc.pdf`

**11.4 표 형식 학습기와 생성 모형**

### 📄 `hollmann2025_tabpfn_nature` [방법]
**Accurate predictions on small data with a tabular foundation model**. Hollmann, N. et al. (2025, *Nature* 637, 319–326). doi:10.1038/s41586-024-08328-6
- **무엇**: 수백만 개 합성 자료로 사전학습한 표 형식 기반 모형 TabPFN 을 제시한다. 표본 1만 개 이하 자료에서 기존 방법보다 정확하다.
- **우리와의 차별/활용**: TabPFN v2 실행(LGT, LGF)의 출처다(원고 Methods).
- 파일: `references/11_validation_and_methods/hollmann2025_tabpfn_nature.pdf`

### 📄 `qu2025_tabicl` [방법]
**TabICL: a tabular foundation model for in-context learning on large data**. Qu, J., Holzmüller, D., Varoquaux, G. & Le Morvan, M. (2025, arXiv:2502.05564; 학회 게재 [미확인]). arXiv:2502.05564
- **무엇**: 열-행 어텐션으로 고정 차원 행 임베딩을 만든 뒤 트랜스포머로 문맥 내 학습을 하는 표 형식 분류 모형이다. TALENT 분류 자료 200개에서 TabPFNv2 와 대등하고 최대 10배 빠르다.
- **우리와의 차별/활용**: TabICL 계열의 원 논문이다. 원고는 v2(아래)를 인용한다.
- 파일: `references/11_validation_and_methods/qu2025_tabicl.pdf`

### 📄 `qu2026_tabiclv2` [방법]
**TabICLv2: a better, faster, scalable, and open tabular foundation model**. Qu, J., Holzmüller, D., Varoquaux, G. & Le Morvan, M. (2026, arXiv:2602.11139). arXiv:2602.11139
- **무엇**: 합성 자료 생성기, 확장 가능한 softmax 어텐션, Muon 최적화기를 도입한 회귀·분류 기반 모형이다. 조정 없이 TabArena 와 TALENT 에서 RealTabPFN-2.5 를 넘는다고 보고한다.
- **우리와의 차별/활용**: 원고 Methods 의 TabICL v2(tabicl 2.0.2) 출처다. ICML 2026 게재 여부는 [미확인]이다.
- 파일: `references/11_validation_and_methods/qu2026_tabiclv2.pdf`

### 📄 `prokhorenkova2018_catboost` [방법]
**CatBoost: unbiased boosting with categorical features**. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V. & Gulin, A. (2018, *NeurIPS* 31, 6639–6649). arXiv:1706.09516
- **무엇**: 순서 부스팅과 범주형 특징 처리로 기존 기울기 부스팅의 목표 누설에 따른 예측 이동을 줄인다.
- **우리와의 차별/활용**: 방법 축 기본 학습기 CatBoost 의 출처다.
- 파일: `references/11_validation_and_methods/prokhorenkova2018_catboost.pdf`

### 📄 `gorishniy2021_ft_transformer` [방법]
**Revisiting deep learning models for tabular data**. Gorishniy, Y., Rubachev, I., Khrulkov, V. & Babenko, A. (2021, *NeurIPS* 34). arXiv:2106.11959
- **무엇**: 표 형식 DL 구조를 같은 규약으로 비교하고 ResNet 형 기준선과 FT-Transformer 를 제시한다. GBDT 와 비교해 보편적으로 우월한 방법은 없다고 결론짓는다.
- **우리와의 차별/활용**: 축소 FT-Transformer 형 학습기의 출처다.
- 파일: `references/11_validation_and_methods/gorishniy2021_ft_transformer.pdf`

### 📄 `gorishniy2025_tabm` [방법]
**TabM: advancing tabular deep learning with parameter-efficient ensembling**. Gorishniy, Y., Kotelnikov, A. & Babenko, A. (2025, *ICLR*; arXiv 2024). arXiv:2410.24210
- **무엇**: 하나의 모형이 매개변수를 공유하는 다수 MLP 앙상블을 모방하는 TabM 을 제안하고, 대규모 평가에서 MLP 계열이 어텐션·검색 기반 구조보다 실용적임을 보인다.
- **우리와의 차별/활용**: 원고의 다중 헤드 MLP(코드명 `tabm`)가 TabM 과 다른 구조라는 명시 문장의 근거다.
- 파일: `references/11_validation_and_methods/gorishniy2025_tabm.pdf`

### 📄 `holzmuller2024_realmlp_better_by_default` [방법]
**Better by default: strong pre-tuned MLPs and boosted trees on tabular data**. Holzmüller, D., Grinsztajn, L. & Steinwart, I. (2024, *NeurIPS* 37). arXiv:2407.04491
- **무엇**: 개선 MLP 인 RealMLP 와, GBDT·RealMLP 의 메타 조정 기본값을 제시한다. 메타 훈련 자료 118개로 조정하고 별도 자료 90개에서 검증한다.
- **우리와의 차별/활용**: RealMLP(pytabkit 1.7.3 기본값) 학습기의 출처다.
- 파일: `references/11_validation_and_methods/holzmuller2024_realmlp_better_by_default.pdf`

### 📄 `bergstra2012_random_search` [방법]
**Random search for hyper-parameter optimization**. Bergstra, J. & Bengio, Y. (2012, *J. Mach. Learn. Res.* 13, 281–305). DOI 없음(JMLR)
- **무엇**: 하이퍼파라미터 최적화에서 무작위 탐색이 격자 탐색보다 효율적임을 경험과 이론으로 보인다.
- **우리와의 차별/활용**: LGF 조정 실험의 무작위 탐색(최대 34개 설정) 근거다.
- 파일: `references/11_validation_and_methods/bergstra2012_random_search.pdf`

### 📄 `cawley2010_model_selection_overfitting` [방법]
**On over-fitting in model selection and subsequent selection bias in performance evaluation**. Cawley, G. C. & Talbot, N. L. C. (2010, *J. Mach. Learn. Res.* 11, 2079–2107). DOI 없음(JMLR)
- **무엇**: 모형 선택 기준 추정량의 분산이 선택 단계의 과적합을 일으키고, 그 영향이 학습 알고리즘 간 성능 차이와 비슷한 크기일 수 있음을 보인다.
- **우리와의 차별/활용**: 기본값보다 겹별로 일관되게 나을 때만 비기본 설정을 고르는 규칙의 근거다.
- 파일: `references/11_validation_and_methods/cawley2010_model_selection_overfitting.pdf`

### 📄 `durkan2019_neural_spline_flows` [방법]
**Neural spline flows**. Durkan, C., Bekasov, A., Murray, I. & Papamakarios, G. (2019, *NeurIPS* 32). arXiv:1906.04032
- **무엇**: 단조 유리 2차 스플라인 변환으로 결합·자기회귀 흐름의 유연성을 높이고 해석적 역변환을 유지한다.
- **우리와의 차별/활용**: 조건부 스플라인 정규화 흐름 학습기의 출처다.
- 파일: `references/11_validation_and_methods/durkan2019_neural_spline_flows.pdf`

### 📄 `ho2020_ddpm` [방법]
**Denoising diffusion probabilistic models**. Ho, J., Jain, A. & Abbeel, P. (2020, *NeurIPS* 33, 6840–6851). arXiv:2006.11239
- **무엇**: 확산 확률 모형을 노이즈 제거 점수 정합과 연결된 가중 변분 하한으로 학습해 고품질 이미지 생성을 보인다(요약은 PDF 초록 기준, OpenAlex 초록 필드는 오염됨).
- **우리와의 차별/활용**: 200단계 노이즈 제거 확산 학습기의 출처다.
- 파일: `references/11_validation_and_methods/ho2020_ddpm.pdf`

### 📄 `lipman2023_flow_matching` [방법]
**Flow matching for generative modeling**. Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M. & Le, M. (2023, *ICLR*; arXiv 2022). arXiv:2210.02747
- **무엇**: 고정된 조건부 확률 경로의 벡터장을 회귀해 연속 정규화 흐름을 시뮬레이션 없이 학습한다. 확산 경로와 최적 수송 경로를 포함한다.
- **우리와의 차별/활용**: 조건부 flow matching 학습기의 출처다.
- 파일: `references/11_validation_and_methods/lipman2023_flow_matching.pdf`

**11.5 다중 비교와 동등성 검정**

### 🔗 `hartung2001_random_effects_meta_analysis_statmed` [방법]
**A refined method for the meta-analysis of controlled clinical trials with binary outcome**. Hartung, J. & Knapp, G. (2001, *Stat. Med.* 20, 3875–3889). doi:10.1002/sim.1009
- **무엇**: 랜덤효과 메타분석에서 처리 효과 분산의 개선 추정량에 기반한 검정을 제안한다. 모의실험에서 명목 유의수준을 기존 검정보다 잘 지킨다.
- **우리와의 차별/활용**: 지역 간 Δ 의 Hartung-Knapp 랜덤효과 평균(LGX)의 근거다.
- 링크(구독 필요): https://doi.org/10.1002/sim.1009

### 🔗 `holm1979_sequential_multiple_test_sjs` [방법]
**A simple sequentially rejective multiple test procedure**. Holm, S. (1979, *Scand. J. Stat.* 6, 65–70). doi:10.2307/4615733 (JSTOR)
- **무엇**: 가설을 하나씩 기각하는 순차 기각형 다중 검정으로, 참 가설의 어떤 조합에서도 1종 오류율을 통제한다.
- **우리와의 차별/활용**: 사전 지정 대비 10개의 Holm 보정 근거다. 원고 표에 DOI 를 추가할 수 있다.
- 링크(구독 필요): https://doi.org/10.2307/4615733

### 🔗 `schuirmann1987_tost_equivalence_jpb` [방법]
**A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability**. Schuirmann, D. J. (1987, *J. Pharmacokinet. Biopharm.* 15, 657–680). doi:10.1007/BF01068419
- **무엇**: 생물학적 동등성 판정에서 두 단측 검정(TOST)과 검정력 접근을 비교한다. 단측 α = 0.05 의 TOST 가 대부분 경우 검정력 접근보다 균일하게 우월하다.
- **우리와의 차별/활용**: ±δ 동등성 판정 규칙의 근거다. OpenAlex 가 Zenodo 사본(1232484)을 공공 영역으로 표기하나 권리 근거를 확인할 수 없어 받지 않았다.
- 링크(구독 필요): https://doi.org/10.1007/BF01068419

## 12 · 원고 참고문헌 (영구동토 과학·자료·응용)  (11/17 PDF)

폴더: `references/12_manuscript_refs/` · 원 조사 기록: `references/_index_parts/library_manuscript_refs.md`

범위: Stefan 해와 ALT 지역 매핑의 고전, ALT·지온 지도와 기존 제품, 영구동토 물리-ML 결합 선례, 공변량 자료원, 관측 자료 논문. 원고 인용 문헌 가운데 다른 폴더에 PDF 가 있는 항목은 아래 위치표로 찾는다.

원고 인용 문헌 위치표(다른 절에 둔 항목):

| 원고 인용 | 키 | 절 |
|---|---|---|
| Karjalainen et al. 2019 | `karjalainen2019_circumpolar_geohazard_maps` | 09 |
| Ran et al. 2022a (ESSD) | `ran2022_panarctic` | 01 |
| Gautam et al. 2025 | `gautam2025_alaska_alt` | 01 |
| Liu Z. et al. 2024 | `liu2024_widespread_alt_deepening_2003_2020_erl` | 08 |
| Wei et al. 2026 | `wei2026_nh_alt_1km_2000_2024_essdd` | 08 |
| Li C. et al. 2022 | `li2022_nh_alt_2000_2018_stefan_jgra` (링크) | 08 |
| Shen et al. 2023 | `shen2023_qtp_permafrost_alt_1980_2020_stoten` (링크) | 08 |
| Du J. et al. 2026 | `du2026_alt_heterogeneity_arctic_foothills_tc` | 08 |
| Du Q. et al. 2026 | `du2026_seasonal_precip_alt_retrospective_buildings` | 08 |
| Du E. et al. 2026 | `du2026_qtp_active_layer_moisture_90m_essdd` | 08 |
| Ahajjam et al. 2025 | `ahajjam2025_multihorizon_alt_circumarctic_jgrmlc` (링크) | 08 |
| Streletskiy et al. 2026 | `streletskiy2026_calm_long_term_alt_cee` | 08 |
| Pilyugina et al. 2025 / 2023 | `pilyugina2025_piml_permafrost_stability_ieeeaccess` (링크) / `pilyugina2023_pinn_permafrost_risk` | 08 / 06 |
| Wang G. et al. 2025 | `wang2025_alt_stefan_catboost_et_qtp_rs` | 08 |
| Zhang C. et al. 2024 | `zhang2024_climate_permafrost_model_efactor_ml_erl` (링크) | 08 |
| Herrington et al. 2026 | `herrington2026_ml_reanalysis_soil_temperature_taac` (링크) | 08 |
| Gay et al. 2026 | `gay2026_zero_curtain_ai_eo` | 14 |
| Huang S. et al. 2025 (EGUsphere) | `piml2025_ne_china_permafrost_extent` | 06 |
| Liu Y. et al. 2023 | `liu2023_pilstm_permafrost_gipl2` (링크, 첫 저자는 Yibo Liu, 상단 경고 참조) | 06 |
| Biskaborn et al. 2019 | `biskaborn2019_gtnp_warming` | 07 |
| Talucci et al. 2025 | `talucci2025_firealt_paired_burned_essd` | 08 |
| Westermann et al. 2024 (CCI) | `esa_cci_permafrost_v4` (DOI 불일치, 상단 경고 참조) | 01 |

**12.1 Stefan 해와 ALT 지역 매핑의 고전**

### 🔗 `stefan1891_eisbildung` [방법]
**Über die Theorie der Eisbildung, insbesondere über die Eisbildung im Polarmeere**. Stefan, J. (1891, *Ann. Phys.* 278(2), 269–286). doi:10.1002/andp.18912780206
- **무엇**: 결빙 경계가 이동하는 열전도 문제를 다루고 극해 얼음 두께 성장의 이론을 제시한 원 논문이다. 얼음 두께가 누적 냉각량의 제곱근에 비례한다는 해의 출발점이다.
- **우리와의 차별/활용**: 원고 Stefan 식의 원전이다. 원고 표의 [verify] 는 이번 조회로 권·쪽·DOI 가 확인되었다.
- 링크(공공 영역 Zenodo 2042312, 이 서버에서 403): https://zenodo.org/record/2042312

### 🔗 `nelson1997_kuparuk_alt_regional_aar` [선례]
**Estimating active-layer thickness over a large region: Kuparuk River basin, Alaska, U.S.A.** Nelson, F. E. et al. (1997, *Arct. Alp. Res.* 29, 367–378). doi:10.1080/00040851.1997.12003258
- **무엇**: 알래스카 Kuparuk 유역 26,278 km² 에서 탐침, 온도 기록계, GIS 를 결합해 1995년 여름 주 단위 ALT 지도를 만든다. 대표 1 km 단위에서 예측값이 측정 평균과 약 6 cm 이내다.
- **우리와의 차별/활용**: Stefan 계수 E 를 현지 측정으로 보정하는 관행의 출처다(원고 서론 첫 문단).
- 링크(구독 필요): https://doi.org/10.1080/00040851.1997.12003258

### 🔗 `shiklomanov2002_kuparuk_13yr_alt_mapping_ppp` [선례]
**Active-layer mapping at regional scales: a 13-year spatial time series for the Kuparuk region, north-central Alaska**. Shiklomanov, N. I. & Nelson, F. E. (2002, *Permafr. Periglac. Process.* 13, 219–230). doi:10.1002/ppp.425
- **무엇**: 기온을 강제력으로, 피복별 edaphic 계수를 국지 반응으로 둔 Stefan 해로 Kuparuk 지역의 13년 연별 ALT 지도를 만든다. 연간 기후 변동이 ALT 를 크게 바꾼다.
- **우리와의 차별/활용**: 피복별 E 로 지역 ALT 를 그리는 방식의 선례이고, 계수 지도 실험(N7) 서술의 배경이다.
- 링크(구독 필요): https://doi.org/10.1002/ppp.425

### 🔗 `riseborough2008_permafrost_modelling_advances_ppp` [맥락]
**Recent advances in permafrost modelling**. Riseborough, D., Shiklomanov, N., Etzelmüller, B., Gruber, S. & Marchenko, S. (2008, *Permafr. Periglac. Process.* 19, 137–156). doi:10.1002/ppp.615
- **무엇**: 2003년 이후 영구동토 모델링의 진전을 시간·열·공간 기준으로 분류해 검토한다. 아격자 변동성 매개변수화를 남은 과제로 든다.
- **우리와의 차별/활용**: Stefan, Kudryavtsev 같은 단순 해석 모형의 위치를 설명하는 리뷰다.
- 링크(구독 필요): https://doi.org/10.1002/ppp.615

**12.2 ALT·지온 통계 지도와 계수 전이**

### 📄 `aalto2018_circumarctic_ground_temp_alt` [선례]
**Statistical forecasting of current and future circum-Arctic ground temperatures and active layer thickness**. Aalto, J., Karjalainen, O., Hjort, J. & Luoto, M. (2018, *Geophys. Res. Lett.* 45, 4889–4898). doi:10.1029/2018GL078007
- **무엇**: 다중 통계 기법 앙상블로 환북극 MAGT 와 ALT 를 예측하고 거리 블록 교차검증으로 전이성을 평가한다. 현재 영구동토가 유지될 조건의 면적을 15.1 ± 2.8 × 10⁶ km² 로 추정한다.
- **우리와의 차별/활용**: ALT 거리 블록 검증의 선례다(원고 서론, NOVELTY 8절). 그림 조사(09 절)에서는 Fig 3(방법별 구간 폭을 모양 부호로)을 Fig 6 참고로 골랐다. 파일은 헬싱키대 HELDA 저장소 사본이다.
- 파일: `references/12_manuscript_refs/aalto2018_circumarctic_ground_temp_alt.pdf`

### 📄 `peng2018_nh_alt_changes` [선례]
**Spatiotemporal changes in active layer thickness under contemporary and projected climate in the Northern Hemisphere**. Peng, X., Zhang, T., Frauenfeld, O. W., Wang, K., Luo, D., Cao, B., Su, H., Jin, H. & Wu, Q. (2018, *J. Clim.* 31, 251–266; 온라인 2017). doi:10.1175/JCLI-D-16-0721.1
- **무엇**: 현장 관측, 융해 지수 기반 Stefan 해, E 계수로 북반구 1971–2000년 ALT 기후값과 1850–2100년 변화를 계산한다. 현장 ALT 는 대체로 40–320 cm 다.
- **우리와의 차별/활용**: 현지 관측으로 E 를 정하는 관행의 출처다. NOVELTY 의 "Peng 2018" 을 이 논문으로 특정한 것은 이번 작업의 판단이므로 원문 대조가 필요하다.
- 파일: `references/12_manuscript_refs/peng2018_nh_alt_changes.pdf`

### 📄 `ran2022b_third_pole_infrastructure` [선례]
**Permafrost degradation increases risk and large future costs of infrastructure on the Third Pole**. Ran, Y. et al. (2022, *Commun. Earth Environ.* 3, 238). doi:10.1038/s43247-022-00568-6
- **무엇**: 자료 기반 투영, 다중 위험 지수, 수명 교체 모형을 결합해 청장고원 기반시설의 영구동토 열화 비용을 추정한다. SSP245 에서 2090년까지 약 63.1억 달러의 추가 비용이 필요하다.
- **우리와의 차별/활용**: E 를 공변량에서 ML 로 추정한 선례다(원고 서론). ALT 오차는 보고하지 않았다(NOVELTY 2절). 1차 `ran2022_panarctic`(ESSD)와 다른 논문이다.
- 파일: `references/12_manuscript_refs/ran2022b_third_pole_infrastructure.pdf`

### 📄 `hasler2015_mountain_permafrost_bc` [선례]
**The influence of surface characteristics, topography and continentality on mountain permafrost in British Columbia**. Hasler, A., Geertsema, M., Foord, V., Gruber, S. & Noetzli, J. (2015, *The Cryosphere* 9, 1025–1038). doi:10.5194/tc-9-1025-2015
- **무엇**: 브리티시컬럼비아 현장 7곳에서 지표·열 offset 을 측정한다. 대기후 조건이 과정의 효과를 바꾸므로 지표·지형별 offset 은 지역 간에 옮기기 어렵다고 결론짓는다.
- **우리와의 차별/활용**: 경험 계수가 지역을 넘어 옮겨지지 않는다는 결과(N7)의 영구동토 선례다. NOVELTY 에서 요약 도구 경유로 적은 수치는 이 PDF 로 대조한다.
- 파일: `references/12_manuscript_refs/hasler2015_mountain_permafrost_bc.pdf`

### 📄 `garibaldi2026_ttop_parameter_importance` [선례]
**Determining TTOP model parameter importance and overall performance across northern Canada**. Garibaldi, M. C., Bonnaventure, P. P., Way, R. G. et al. (2026, *The Cryosphere* 20, 2375–2392). doi:10.5194/tc-20-2375-2026
- **무엇** [초록]: 캐나다 북부 330지점의 기온·지온 자료로 TTOP 매개변수의 중요도를 지점 하나 제외 교차검증과 랜덤 포레스트로 평가한다. 동결기 n-factor 와 동결 도일이 성능을 좌우하고, 토지피복 기반 매개변수는 지점 간에 전이되지 않는다.
- **우리와의 차별/활용**: "경험 계수가 지역을 넘어 옮겨지지 않는다"(N7)의 직접 선례다. 우리 기여는 이를 ALT 오차 단위와 라벨 수 축으로 수치화한 것이다.
- 파일: `references/12_manuscript_refs/garibaldi2026_ttop_parameter_importance.pdf`

### 📄 `parsekian2021_alt_airborne_sar_validation` [맥락]
**Validation of permafrost active layer estimates from airborne SAR observations**. Parsekian, A. D. et al. (2021, *Remote Sens.* 13, 2876). doi:10.3390/rs13152876
- **무엇**: 알래스카 3개 지역에서 항공 SAR 유도 ALT 를 보정 GPR 자료로 검증한다. 79 % 지점에서 불확실성 범위 안에서 일치했고, 평균 불확실성은 GPR ALT 0.14 m, SAR ALT 0.19 m 다.
- **우리와의 차별/활용**: 라벨 측정 불확실성(0.14 m)과 잔차 ML 이득의 크기를 비교하는 근거다(NOVELTY 9절).
- 파일: `references/12_manuscript_refs/parsekian2021_alt_airborne_sar_validation.pdf`

### 📄 `nitze2021_rts_deep_learning` [선례]
**Developing and testing a deep learning approach for mapping retrogressive thaw slumps**. Nitze, I., Heidler, K., Barth, S. & Grosse, G. (2021, *Remote Sens.* 13, 4294). doi:10.3390/rs13214294
- **무엇**: PlanetScope, ArcticDEM 으로 후퇴성 융해 사태(RTS)를 분할하는 딥러닝을 캐나다·러시아 6개 지역의 지역 교차검증으로 평가한다. 4개 지역은 maxIoU 0.39–0.58 이었고 2개 지역에서는 실패했다.
- **우리와의 차별/활용**: 영구동토 ML 의 지역 홀드아웃 선례다. 신규성 문장의 범위를 "ALT 또는 MAGT 매핑"으로 한정하는 근거다.
- 파일: `references/12_manuscript_refs/nitze2021_rts_deep_learning.pdf`

### 📄 `li2026_tunnel_freezing_depth_xgboost` [선례]
**Freezing depth prediction of surrounding rock in seasonally frozen tunnels based on bayes-optimized XGBoost**. Li, J. & Jia, J. (2026, *Sci. Rep.* 16; 논문 번호 [미확인]). doi:10.1038/s41598-026-49818-z
- **무엇**: 계절 동결 터널 주변 암반의 수열 결합 모형과 라틴 초입방 표본 수치 모의로 동결 깊이 자료를 만들고, 베이즈 최적화 XGBoost 로 예측한다.
- **우리와의 차별/활용**: 수치 모의 라벨로 학습한 선례다(NOVELTY 6절 표). 영구동토 ALT 문헌은 아니다.
- 파일: `references/12_manuscript_refs/li2026_tunnel_freezing_depth_xgboost.pdf`

**12.3 공변량 자료원과 관측 자료 논문**

### 📄 `munozsabater2021_era5_land` [자료]
**ERA5-Land: a state-of-the-art global reanalysis dataset for land applications**. Muñoz-Sabater, J. et al. (2021, *Earth Syst. Sci. Data* 13, 4349–4383). doi:10.5194/essd-13-4349-2021
- **무엇**: ERA5 의 지표 성분을 9 km 해상도로 다시 계산한 ERA5-Land 의 구성과 검증을 기술한다. 근지표 상태량에 고도 보정을 적용한다.
- **우리와의 차별/활용**: TDD, FDD, 토양 온도, 적설 수당량 공변량의 출처다(원고 Methods).
- 파일: `references/12_manuscript_refs/munozsabater2021_era5_land.pdf`

### 📄 `poggio2021_soilgrids2` [자료]
**SoilGrids 2.0: producing soil information for the globe with quantified spatial uncertainty**. Poggio, L. et al. (2021, *SOIL* 7, 217–240). doi:10.5194/soil-7-217-2021
- **무엇**: 약 24만 지점 토양 관측과 400개 이상의 공변량으로 250 m 전 지구 토양 물성 지도와 공간 불확실성을 만든다. 고위도 관측 부족을 한계로 든다.
- **우리와의 차별/활용**: 토양 공변량 9개의 출처다. 고위도 관측 부족은 토양 공변량의 정보 한계 서술에 쓴다.
- 파일: `references/12_manuscript_refs/poggio2021_soilgrids2.pdf`

### 🔗 `pekel2016_global_surface_water` [자료]
**High-resolution mapping of global surface water and its long-term changes**. Pekel, J.-F., Cottam, A., Gorelick, N. & Belward, A. S. (2016, *Nature* 540, 418–422). doi:10.1038/nature20584
- **무엇**: Landsat 영상 300만 장 이상으로 1984–2015년 30 m 월별 지표수 변화를 정량화한다. 영구 지표수 약 9만 km² 가 사라지고 18.4만 km² 가 새로 생겼다.
- **우리와의 차별/활용**: 지도 마스크에 쓴 JRC 지표수 출현 빈도층의 출처다. 원고 표의 [verify] 는 이번 조회로 권·쪽·DOI 가 확인되었다.
- 링크(GEOMAR OceanRep 수락본, 봇 확인 페이지로 자동 차단): http://oceanrep.geomar.de/35170/7/Pekel.pdf

### 🔗 `akerman2008_subarctic_sweden_active_layers_ppp` [자료]
**Thawing permafrost and thicker active layers in sub-arctic Sweden**. Åkerman, H. J. & Johansson, M. (2008, *Permafr. Periglac. Process.* 19, 279–292). doi:10.1002/ppp.626
- **무엇**: 스웨덴 Torneträsk 지역 9개 지점(최대 29년 격자 측정)의 ALT 가 연 0.7–1.3 cm 증가했고 최근 10년에 가속했다.
- **우리와의 차별/활용**: Abisko 하위 지점 CALM 누리집 자료의 출처 논문이다. 재배포 허가가 확인되지 않은 자료로 원고에 기록되어 있다.
- 링크(구독 필요): https://doi.org/10.1002/ppp.626

### 📄 `pohl2026_syrdakh_observatory_essd` [자료]
**Thermo-hydrological river valley observatory in Yedoma permafrost from 2012 through 2022 in Syrdakh, Central Yakutia**. Pohl, E. et al. (2026, *Earth Syst. Sci. Data* 18, 3525–3557). doi:10.5194/essd-18-3525-2026
- **무엇**: 중앙 야쿠티아 Syrdakh 에서 열카르스트 호수를 잇는 하천 단면에 지중 온도 사슬을 설치해 2012–2022년 열·수문 관측 자료를 제공한다.
- **우리와의 차별/활용**: Syrdakh 융해 깊이 라벨(Zenodo 19890671)의 자료 논문이다. 원고 표의 서지 제목은 Zenodo 기록 제목이므로 게재 논문 제목(위)으로 바꾼다. 정윤택 박사 연구 지역(중앙 야쿠티아, 10 절)과 겹친다.
- 파일: `references/12_manuscript_refs/pohl2026_syrdakh_observatory_essd.pdf`

**12.4 원고 자료 인용 (데이터셋·단행본, PDF 대상 아님, 25건)**

항목 수 집계에 넣지 않는다. 요약은 `data/processed/ext_labels/*_meta.json` 의 `name` 필드와 원고 본문을 따른다.

| 인용 | DOI 또는 위치 | 내용 | 관련성 |
|---|---|---|---|
| Åkerman, 1998 | 없음 (CAPS v1.0 CD-ROM, NSIDC; CALM 누리집) | Abisko S2, Kapp Linné S1 하위 지점 탐침 원자료 | 확인적 풀 밖 라벨. 재배포 허가 미확인 |
| Boike et al., 2024 | 10.1594/PANGAEA.971586 | T-MOSAiC 2022 myThaw 표준 규약 융해 깊이 | 추가 지역 라벨 |
| Du et al., 2026 (Zenodo) | 10.5281/zenodo.21999366 | 청장고원 활동층 수분 코드 묶음의 GPR·인력 시추 ALT 조사점 | 티베트 라벨. 논문은 `du2026_qtp_active_layer_moisture_90m_essdd` |
| Fu, 2025 | 10.6084/m9.figshare.29206613.v1 | 청장고원 지온과 ALT 2001–2020 (CC BY 4.0) | 티베트 라벨 |
| Grosse, 2007 | 10.1594/PANGAEA.611409 | Cape Mamontov Klyk 주변 활동층 자료 | 러시아 라벨 |
| Hammar et al., 2025 | 10.1594/PANGAEA.974461 | 2023 myThaw 융해 깊이 | 추가 지역 라벨 |
| Jorgenson and Kanevskiy, 2025 | 10.18739/A27P8TG0G | Alaska Permafrost Soils Inventory and Thermokarst Monitoring Database 2024 Update (CUSP v1.1 경유) | 알래스카 보조 라벨 |
| Kudryavtsev et al., 1974 | 없음 (모스크바대 출판부 단행본, CRREL Draft Translation 606, 1977) | Kudryavtsev 해석해의 원전 | 공개본 [미확인], 원고 표 [verify] 유지 |
| Liang et al., 2023 | 10.1594/PANGAEA.961876 | 시베리아 북동부 낙엽송 둔덕 융해 깊이 | 러시아 라벨 |
| Makarieva et al., 2017 | 10.1594/PANGAEA.881754 | 콜리마 물수지 관측소 융해 깊이·적설 시계열 1954–1997 | 러시아 라벨 |
| Martin et al., 2023 | 10.1594/PANGAEA.956039 | T-MOSAiC 2021 myThaw | 추가 지역 라벨 |
| Moore et al., 2025 | 10.3334/ORNLDAAC/2369 | ABoVE 토양 수분·ALT 2005–2024, 버전 2 | 주 지역 라벨(알래스카·캐나다). DataCite 연도 2026 표기 차이 |
| Nixon, 2003 | 10.7265/7m84-k262 | NSIDC GGD353 캐나다 활동층 관측(매켄지 계곡 융해관) | 캐나다 라벨 |
| Palmtag et al., 2022 | 10.17043/palmtag-2022-pedon-1 | 북부 영구동토 지역 토양 단면 자료 | 토양 보조 자료 |
| Petrone et al., 2016 | 10.1594/PANGAEA.845258 | 서그린란드 GPR·지형·식생 기반 퇴적층·ALT 모형 | 그린란드 라벨(CUSP 경유) |
| Pohl et al., 2026 (Zenodo) | 10.5281/zenodo.19890671 | Syrdakh 열·수문 관측소 융해 깊이 2012–2018 | 러시아 라벨. 논문은 `pohl2026_syrdakh_observatory_essd` |
| Sannel, 2020 | 10.17043/sannel-2020-temperature-1 | Tavvavuoma 이탄지 지온, 융해 깊이, 적설 | 스웨덴 라벨 |
| Scheer et al., 2024 | 10.1594/PANGAEA.964306 | 일루리사트 ALT 탐침 측정 2020–2021 | 그린란드 라벨 |
| Streletskiy et al., 2025 | 10.1594/PANGAEA.972777 | GTN-P CALM 35년 ALT 집계 | 주 라벨 출처. 저자 목록 [verify] 유지 |
| Talucci et al., 2024 | 10.18739/A2RN3092P | FireALT 자료 원본(Arctic Data Center) | 논문은 `talucci2025_firealt_paired_burned_essd` |
| Veremeeva et al., 2025 | 10.1594/PANGAEA.973813 | ALLena 레나 델타 융해 깊이 1998–2022 | 주 지역(레나) 라벨 |
| Walker et al., 2009 | 10.1594/PANGAEA.842711 | 야말 측선 융해 깊이 2007–2008 | 러시아 라벨 |
| Westermann et al., 2024 | 10.5285/d34330ce3f604e368c06d76de1987ce5 | ESA CCI Permafrost ALT v4.0 (CEDA) | 모형 기반 공변량. 제품 문서는 `esa_cci_permafrost_v4` |
| Yang and Qiu, 2026 | 10.5281/zenodo.18150789 | 티베트 고원 ALT·MAGT 현장 관측 집계표 | 기술 통계용, 분석 미사용 |
| Zastruzny et al., 2024 | 10.1594/PANGAEA.967139 | 디스코섬 사면 동결면 깊이 2015 | 그린란드 라벨 |

원고 References 표에 반영할 서지 확인 결과(원 조사 기록 5절 요약): Stefan 1891 과 Pekel 2016 의 [verify] 는 해소되었다. Du et al. 2026 article DOI 는 10.5194/essd-2026-330 으로 채울 수 있다(사전본, 게재 확정 [미확인]). Holm 1979 에 DOI 10.2307/4615733 을 추가할 수 있다. Tama et al. 2025 는 WACV 2026 게재본(doi:10.1109/WACV61042.2026.00528)으로 인용할 수 있다. Pohl et al. 2026 은 게재 논문 제목으로 바꾼다. NOVELTY 8절의 "Adjei 2026"(환경 매핑의 conformal 커버리지)은 서지를 특정하지 못했다 [미확인].

## 13 · 그림·슬라이드 설계 규칙  (18/23 PDF)

폴더: `references/13_figure_and_slide_design/` · 폴더 안내: `references/13_figure_and_slide_design/README.md` · 규칙 정리와 인용문: `docs/research/2026-10-04/figure_slide_standards.md`

관계 표기는 모두 [방법](그림·슬라이드 제작 규칙의 근거)이다. 수집처는 출판사 공개 쪽(PLOS, Nature Communications, Copernicus, Frontiers, TOS), arXiv, 저자 소속 기관 사이트(Penn State, writing.engr.psu.edu)다. 원고 그림의 품질 기준은 09 절(상위 학술지 그림 모범 사례)에서, 제작 규칙의 근거는 이 절에서 가져온다.

**13.1 그림 일반**

### 📄 `rougier2014_ten_simple_rules_better_figures` [방법]
**Ten simple rules for better figures**. Rougier, N. P., Droettboom, M. & Bourne, P. E. (2014, *PLoS Comput. Biol.* 10, e1003833). doi:10.1371/journal.pcbi.1003833
- **무엇**: 그림 설계 10원칙(청중, 메시지, 매체, 캡션, 기본값 불신, 색, 왜곡 금지, chartjunk 배제 등)을 제시한다.
- **우리와의 차별/활용**: 그림 점검표의 출발점이다. 기본값을 그대로 쓰지 않는다는 원칙을 matplotlib 스타일 파일에 반영한다.
- 파일: `references/13_figure_and_slide_design/rougier2014_ten_simple_rules_better_figures.pdf`

### 📄 `weissgerber2015_beyond_bar_line_graphs` [방법]
**Beyond bar and line graphs: time for a new data presentation paradigm**. Weissgerber, T. L., Milic, N. M., Winham, S. J. & Garovic, V. D. (2015, *PLoS Biol.* 13, e1002128). doi:10.1371/journal.pbio.1002128
- **무엇**: 소표본 연속 자료를 평균 막대로 요약하면 분포가 가려진다고 보이고 개별 점 표시를 권한다.
- **우리와의 차별/활용**: 지역별 Δ 처럼 지역 수가 적은 결과를 막대 대신 점·구간으로 그리는 근거다.
- 파일: `references/13_figure_and_slide_design/weissgerber2015_beyond_bar_line_graphs.pdf`

### 📄 `jambor2021_image_based_figures` [방법]
**Creating clear and informative image-based figures for scientific publications**. Jambor, H. et al. (2021, *PLoS Biol.* 19, e3001161). doi:10.1371/journal.pbio.3001161
- **무엇**: 영상·지도 그림의 축척 막대, 주석, 색각 이상 대응 지침을 제시한다.
- **우리와의 차별/활용**: 지도 패널(Fig 1, Fig 6)의 축척·삽입도·주석 규칙의 근거다.
- 파일: `references/13_figure_and_slide_design/jambor2021_image_based_figures.pdf`

### 🔗 `midway2020_principles_effective_data_visualization` [방법]
**Principles of effective data visualization**. Midway, S. R. (2020, *Patterns* 1, 100141). doi:10.1016/j.patter.2020.100141
- **무엇** [제목]: 자료 시각화의 일반 원칙을 정리한 논문이다.
- **우리와의 차별/활용**: Rougier 2014 다음에 읽는 그림 일반 문헌이다.
- 링크(OA, 자동 차단): https://doi.org/10.1016/j.patter.2020.100141

### 🔗 `kelleher2011_ten_guidelines_data_visualization` [방법]
**Ten guidelines for effective data visualization in scientific publications**. Kelleher, C. & Wagener, T. (2011, *Environ. Model. Softw.* 26, 822–827). doi:10.1016/j.envsoft.2010.12.006
- **무엇** [제목]: 과학 논문 자료 시각화의 10개 지침이다.
- **우리와의 차별/활용**: 환경 모델링 분야 그림 지침의 보조 문헌이다.
- 링크(구독 필요): https://doi.org/10.1016/j.envsoft.2010.12.006

**13.2 색**

### 📄 `crameri2020_misuse_of_colour` [방법]
**The misuse of colour in science communication**. Crameri, F., Shephard, G. E. & Heron, P. J. (2020, *Nat. Commun.* 11, 5444). doi:10.1038/s41467-020-19160-7
- **무엇**: 지각 균일 색지도의 필요성, 색지도 선택 흐름도, 점검 4항목을 제시한다.
- **우리와의 차별/활용**: 부호 있는 오차·잔차에는 0 중심 발산형, 비음수 ALT 에는 지각 균일 순차형을 쓰는 규칙의 근거다.
- 파일: `references/13_figure_and_slide_design/crameri2020_misuse_of_colour.pdf`

### 📄 `thyng2016_true_colors_oceanography_colormaps` [방법]
**True colors of oceanography: guidelines for effective and accurate colormap selection**. Thyng, K. M., Greene, C. A., Hetland, R. D., Zimmerle, H. M. & DiMarco, S. F. (2016, *Oceanography* 29(3), 9–13). doi:10.5670/oceanog.2016.66
- **무엇**: 순차·발산·순환 색지도의 분류와 cmocean 색지도를 제시한다.
- **우리와의 차별/활용**: 지구과학 그림의 색지도 선택 기준이다.
- 파일: `references/13_figure_and_slide_design/thyng2016_true_colors_oceanography_colormaps.pdf`

### 📄 `stoelzle2021_rainbow_colormap_hydrology` [방법]
**Rainbow color map distorts and misleads research in hydrology: guidance for better visualizations and science communication**. Stoelzle, M. & Stein, L. (2021, *Hydrol. Earth Syst. Sci.* 25, 4549–4565). doi:10.5194/hess-25-4549-2021
- **무엇**: 지구과학 문헌의 rainbow 색지도 사용 실태를 조사하고 역할별 색 점검표를 제시한다.
- **우리와의 차별/활용**: rainbow·jet 금지 규칙의 지구과학 근거다.
- 파일: `references/13_figure_and_slide_design/stoelzle2021_rainbow_colormap_hydrology.pdf`

### 📄 `kovesi2015_good_colour_maps` [방법]
**Good colour maps: how to design them**. Kovesi, P. (2015, arXiv:1509.03700). arXiv:1509.03700
- **무엇**: 지각 명도 증분의 균일성을 기준으로 색지도를 설계하고 시험 영상으로 결함을 찾는 방법을 제시한다.
- **우리와의 차별/활용**: 색지도 검증 절차의 근거다.
- 파일: `references/13_figure_and_slide_design/kovesi2015_good_colour_maps.pdf`

### 📄 `nunez2018_cividis_cvd_colormaps` [방법]
**Optimizing colormaps with consideration for color vision deficiency to enable accurate interpretation of scientific data**. Nuñez, J. R., Anderton, C. R. & Renslow, R. S. (2018, *PLoS ONE* 13, e0199239). doi:10.1371/journal.pone.0199239
- **무엇**: 색각 이상에 최적화한 색지도 cividis 를 제시한다.
- **우리와의 차별/활용**: 색각 이상 대응 순차형 색지도의 선택지다.
- 파일: `references/13_figure_and_slide_design/nunez2018_cividis_cvd_colormaps.pdf`

### 📄 `crameri2018_staglab_scientific_visualisation` [방법]
**Geodynamic diagnostics, scientific visualisation and StagLab 3.0**. Crameri, F. (2018, *Geosci. Model Dev.* 11, 2541–2562). doi:10.5194/gmd-11-2541-2018
- **무엇**: oslo·davos·broc 등 Scientific colour maps 를 처음 공개하고 rainbow 색지도의 시각 오차(최대 7.5 %)를 보인다.
- **우리와의 차별/활용**: Scientific colour maps 를 쓸 때의 원 출처 인용이다.
- 파일: `references/13_figure_and_slide_design/crameri2018_staglab_scientific_visualisation.pdf`

### 🔗 `crameri2018_scientific_colour_maps_zenodo` [방법]
**Scientific colour maps**. Crameri, F. (2018, Zenodo). doi:10.5281/zenodo.1243862
- **무엇**: Scientific colour maps 소프트웨어 보관본이다.
- **우리와의 차별/활용**: 색지도를 그림에 쓰면 이 DOI 를 함께 인용한다.
- 링크(소프트웨어 보관본): https://doi.org/10.5281/zenodo.1243862

### 🔗 `light2004_end_of_rainbow_eos` [방법]
**The end of the rainbow? Color schemes for improved data graphics**. Light, A. & Bartlein, P. J. (2004, *Eos* 85(40), 385–391). doi:10.1029/2004EO400002
- **무엇** [제목]: rainbow 색표의 문제와 대안 색 체계를 제시한 초기 지구과학 문헌이다.
- **우리와의 차별/활용**: rainbow 금지 근거의 역사적 출처다.
- 링크(무료 공개, 자동 차단): https://doi.org/10.1029/2004EO400002

### 🔗 `wong2011_points_of_view_color_blindness` [방법]
**Points of view: Color blindness**. Wong, B. (2011, *Nat. Methods* 8, 441). doi:10.1038/nmeth.1618
- **무엇** [제목]: 색각 이상을 고려한 색 선택 지침과 색 조합을 제시한 짧은 칼럼이다.
- **우리와의 차별/활용**: 범주형 색(지역, 학습기) 선택의 보조 근거다.
- 링크(구독 필요): https://doi.org/10.1038/nmeth.1618

**13.3 슬라이드**

### 📄 `alley2005_sentence_headlines_visual_evidence` [방법]
**Rethinking the design of presentation slides: a case for sentence headlines and visual evidence**. Alley, M. & Neeley, K. A. (2005, *Tech. Commun.* 52(4), 417–426). DOI 없음
- **무엇**: assertion-evidence 슬라이드 지침표(헤드라인 28 pt, 본문 18–24 pt, 헤드라인 2줄 제한)를 제시한다.
- **우리와의 차별/활용**: 발표 자료의 "주장 헤드라인 + 그림 증거" 구조의 근거다. 상자형 글머리 슬라이드를 피하는 이유를 설명한다.
- 파일: `references/13_figure_and_slide_design/alley2005_sentence_headlines_visual_evidence.pdf`

### 📄 `alley2006_headline_design_audience_retention` [방법]
**How the design of headlines in presentation slides affects audience retention**. Alley, M., Schreiber, M., Ramsdell, K. & Muffo, J. (2006, *Tech. Commun.* 53(2), 225–234). DOI 없음
- **무엇**: 대형 지질학 강의 4개 반에서 같은 내용을 구 헤드라인 슬라이드와 문장 헤드라인 슬라이드로 가르쳐 헤드라인 내용 회상 정답률을 비교한다. 평균 정답률은 구 헤드라인 69 %, 문장 헤드라인 79 % 다(p < 0.001).
- **우리와의 차별/활용**: 헤드라인 형식 선택의 실험 근거다. 사용자 문체 규칙(명사형 제목)과 함께 지키는 방식은 `figure_slide_standards.md` 의 판단(헤드라인은 과장 없는 평서문 한 문장, 그림 안 소제목과 축 이름은 명사형)을 따른다.
- 파일: `references/13_figure_and_slide_design/alley2006_headline_design_audience_retention.pdf`

### 📄 `garner2013_assertion_evidence_comprehension` [방법]
**How the design of presentation slides affects audience comprehension: a case for the assertion–evidence approach**. Garner, J. K. & Alley, M. P. (2013, *Int. J. Eng. Educ.* 29(6), 1564–1579). DOI 없음
- **무엇**: 110명 실험으로 assertion-evidence 슬라이드의 이해도·오개념·인지 부하 효과를 측정한다.
- **우리와의 차별/활용**: 슬라이드 구조 선택의 실험 근거다.
- 파일: `references/13_figure_and_slide_design/garner2013_assertion_evidence_comprehension.pdf`

### 📄 `naegle2021_ten_simple_rules_presentation_slides` [방법]
**Ten simple rules for effective presentation slides**. Naegle, K. M. (2021, *PLoS Comput. Biol.* 17, e1009554). doi:10.1371/journal.pcbi.1009554
- **무엇**: 슬라이드 10원칙(1장 1생각, 1장 1분, 헤드라인에 결론 등)을 제시한다.
- **우리와의 차별/활용**: 덱 분량과 슬라이드당 메시지 수를 정하는 기준이다.
- 파일: `references/13_figure_and_slide_design/naegle2021_ten_simple_rules_presentation_slides.pdf`

### 📄 `bourne2007_ten_simple_rules_oral_presentations` [방법]
**Ten simple rules for making good oral presentations**. Bourne, P. E. (2007, *PLoS Comput. Biol.* 3, e77). doi:10.1371/journal.pcbi.0030077
- **무엇**: 구두 발표 10원칙을 제시하고 1분당 시각자료 1개 이하를 권한다.
- **우리와의 차별/활용**: 발표 시간 대비 슬라이드 수 결정에 쓴다.
- 파일: `references/13_figure_and_slide_design/bourne2007_ten_simple_rules_oral_presentations.pdf`

### 📄 `lortie2017_ten_simple_rules_short_presentations` [방법]
**Ten simple rules for short and swift presentations**. Lortie, C. J. (2017, *PLoS Comput. Biol.* 13, e1005373). doi:10.1371/journal.pcbi.1005373
- **무엇**: 짧은 발표의 10원칙으로, 논문 그림을 그대로 붙이지 않기와 기성 템플릿 지양을 포함한다.
- **우리와의 차별/활용**: 논문 그림을 슬라이드용으로 다시 그리는 규칙의 근거다.
- 파일: `references/13_figure_and_slide_design/lortie2017_ten_simple_rules_short_presentations.pdf`

### 📄 `garner2009_powerpoint_vs_assertion_evidence` [방법]
**Common use of PowerPoint versus the assertion–evidence structure: a cognitive psychology perspective**. Garner, J. K., Alley, M., Gaudelli, A. F. & Zappe, S. E. (2009, *Tech. Commun.* 56(4), 331–345). DOI 없음
- **무엇**: 기본형 PowerPoint(주제 제목 + 글머리)의 한계를 다매체 학습 원칙으로 분석한다.
- **우리와의 차별/활용**: 글머리 위주 슬라이드를 쓰지 않는 이유의 인지 심리 근거다(심화).
- 파일: `references/13_figure_and_slide_design/garner2009_powerpoint_vs_assertion_evidence.pdf`

### 📄 `kosslyn2012_powerpoint_flaws_psychological_analysis` [방법]
**PowerPoint presentation flaws and failures: a psychological analysis**. Kosslyn, S. M., Kievit, R. A., Russell, A. G. & Shephard, J. M. (2012, *Front. Psychol.* 3, 230). doi:10.3389/fpsyg.2012.00230
- **무엇**: 슬라이드 결함을 인지 원칙 8개로 분석하고 작업기억 한계(약 4단위)를 근거로 든다.
- **우리와의 차별/활용**: 슬라이드당 글머리 4개 이하 규칙의 근거다.
- 파일: `references/13_figure_and_slide_design/kosslyn2012_powerpoint_flaws_psychological_analysis.pdf`

### 📄 `garner2016_slide_structure_presenter_understanding` [방법]
**Slide structure can influence the presenter's understanding of the presentation's content**. Garner, J. K. & Alley, M. P. (2016, *Int. J. Eng. Educ.* 32(1A), 39–54). DOI 없음
- **무엇**: 120명 실험으로 슬라이드 구조가 작성자 본인의 내용 이해에도 영향을 준다는 결과를 보인다.
- **우리와의 차별/활용**: 발표 준비 과정에서 주장 헤드라인을 먼저 쓰는 이유의 근거다(심화).
- 파일: `references/13_figure_and_slide_design/garner2016_slide_structure_presenter_understanding.pdf`

## 14 · Scientific Reports 원고 형식 예시  (8/8 PDF)

폴더: `references/14_scirep_exemplars/` · 폴더 안내: `references/14_scirep_exemplars/README.md` · 형식 분석과 문장 틀: `docs/research/2026-10-04/scirep_manuscript_exemplars.md`

선정 기준: 2022–2026년 Sci Rep 게재, 지구과학·원격탐사·환경 ML, 영구동토·빙권·토양·수문·공간 검증 주제, 오픈액세스(OpenAlex `best_oa_location` 확인). 관계 표기 [형식]은 원고 구성·문장 형식의 참고 대상이라는 뜻이다. 이 논문들의 그림 품질은 중간 수준이므로 그림 기준은 09 절에서 가져온다.

다른 절에 둔 Sci Rep 예시: `gautam2025_alaska_alt`(01 절, 가장 가까운 선행 연구, 무작위 70/30 분할과 전체 자료 조율을 저자가 스스로 적음), `singh2024_conformal_eo`(05 절, 불확실성 용어 정의와 코드 가용성 절), `zhang2024_lateral_heat_flow_permafrost_modeling_scirep`(08 절), `tahvildari2026_physics_hybrid_pore_pressure`(11 절), `li2026_tunnel_freezing_depth_xgboost`(12 절).

### 📄 `jones2024_postfire_permafrost_stabilization` [형식]
**Post-fire stabilization of thaw-affected permafrost terrain in northern Alaska**. Jones, B. M. et al. (2024, *Sci. Rep.* 14, 8499). doi:10.1038/s41598-024-58998-5
- **무엇**: 북알래스카 산불 뒤 융해 영향 영구동토 지형의 안정화를 LiDAR 와 시추 자료로 분석한다.
- **우리와의 차별/활용**: 수치 중심 서술의 모범이다. 초록과 결과 문장 대부분에 단위가 붙은 수치가 있고, 'framework'·'comprehensive' 를 한 번도 쓰지 않는다. 초록 문장 구성을 먼저 따른다. CC BY.
- 파일: `references/14_scirep_exemplars/jones2024_postfire_permafrost_stabilization.pdf`

### 📄 `hall2026_rts_expansion_alaska` [형식]
**Predicting retrogressive thaw slump expansion across northern Alaska**. Hall, E. C., Chipman, M. L. & Lara, M. J. (2026, *Sci. Rep.*; 권·논문 번호 [미확인]). doi:10.1038/s41598-026-70171-8
- **무엇**: 북알래스카 후퇴성 융해 사태(RTS) 확장을 GLMM 통계 모형으로 예측하고 투영한다.
- **우리와의 차별/활용**: 방법의 누설 방지·외삽 처리 서술과, 고찰 끝에 한계마다 해석에 미치는 영향을 함께 적은 한계 문단의 예다. 파일은 Article in Press 판(편집 전)이므로 최종본이 나오면 교체한다. CC BY-NC-ND.
- 파일: `references/14_scirep_exemplars/hall2026_rts_expansion_alaska.pdf`

### 📄 `fisher2023_open_water_evaporation` [형식]
**Remotely sensed terrestrial open water evaporation**. Fisher, J. B. et al. (2023, *Sci. Rep.* 13, 8174). doi:10.1038/s41598-023-34921-2
- **무엇**: 개방 수면 증발을 과정 모델과 ML 11종으로 비교하는 벤치마크다.
- **우리와의 차별/활용**: 물리 모델 대 ML 서술의 직접 참고다(벤치마크 논리, 오차 원천 열거). 결론부의 일반론 문장은 따르지 않는다. CC BY.
- 파일: `references/14_scirep_exemplars/fisher2023_open_water_evaporation.pdf`

### 📄 `khanal2023_soc_stocks_nepal` [형식]
**Mapping soil organic carbon stocks in Nepal's forests**. Khanal, S., Nolan, R. H., Medlyn, B. E. & Boer, M. M. (2023, *Sci. Rep.* 13, 8090). doi:10.1038/s41598-023-34247-z
- **무엇**: 네팔 산림 토양 유기탄소 저장량을 분위 회귀 포레스트(QRF)로 지도화한다.
- **우리와의 차별/활용**: 공간 CV 대 무작위 CV, AOA, 전 지구 제품 비교를 보고하는 방식의 예다. 우리 보조 주장(무작위 분할 과대평가)과 기존 제품 비교 SI 의 서술 형식으로 쓴다. CC BY.
- 파일: `references/14_scirep_exemplars/khanal2023_soc_stocks_nepal.pdf`

### 📄 `isaksen2022_barents_warming` [형식]
**Exceptional warming over the Barents area**. Isaksen, K. et al. (2022, *Sci. Rep.* 12, 9371). doi:10.1038/s41598-022-13568-5
- **무엇**: 바렌츠해 지역 기온 상승을 관측과 재분석으로 분석한다.
- **우리와의 차별/활용**: 서론 끝의 질문 번호 목록, 질문별로 구성한 고찰, 서론 끝 결과 요약 문단의 예다. 코드 가용성 절을 따로 둔 예이기도 하다. CC BY.
- 파일: `references/14_scirep_exemplars/isaksen2022_barents_warming.pdf`

### 📄 `hatami2022_temperature_snow_freeze_thaw` [형식]
**Compound changes in temperature and snow depth lead to asymmetric and nonlinear responses in landscape freeze–thaw**. Hatami, S. & Nazemi, A. (2022, *Sci. Rep.* 12, 2196). doi:10.1038/s41598-022-06320-6
- **무엇**: 기온과 적설 깊이의 복합 변화가 경관 동결·융해에 비대칭·비선형 반응을 낳음을 보인다.
- **우리와의 차별/활용**: 주장형 논문 제목과 검정 명시 방식의 예다. 결과 소제목은 명사구로 쓴다는 점(예시 10편 공통)과 구분해서 본다. CC BY.
- 파일: `references/14_scirep_exemplars/hatami2022_temperature_snow_freeze_thaw.pdf`

### 📄 `feeney2022_soil_map_comparison_soc` [형식]
**Multiple soil map comparison highlights challenges for predicting topsoil organic carbon concentration at national scale**. Feeney, C. J. et al. (2022, *Sci. Rep.* 12, 1379). doi:10.1038/s41598-022-05476-5
- **무엇**: 국가 규모 표토 유기탄소 지도 8종을 비교·검증한다.
- **우리와의 차별/활용**: 지도 비교와 범위 한정 문장의 예다. 기존 ALT 제품 비교(Wei 2026, Liu Z. 2024, CCI) 서술에 쓴다. CC BY.
- 파일: `references/14_scirep_exemplars/feeney2022_soil_map_comparison_soc.pdf`

### 📄 `gay2026_zero_curtain_ai_eo` [형식]
**Resolving circumarctic zero-curtain phenomena with AI-integrated earth observations**. Gay, B. A., Miner, K. R., Rietze, N., Poulter, B., Pastick, N. J. & Miller, C. E. (2026, *Sci. Rep.* 16, 28715). doi:10.1038/s41598-026-61719-9
- **무엇** [본문]: 현장 관측 6271만 건과 원격탐사 33억 건을 통합한 물리 정보 전이 학습 틀(GeoCryoAI)로 zero-curtain 후보를 30 m 로 추정한다. 후보 탐지 정확도 96.4 %, 지리 분리 지점 교차검증에서 성능 저하 2 % 미만이다.
- **우리와의 차별/활용**: 영구동토 + 물리 정보 DL 의 Sci Rep 선례이며 원고 서론에 인용한다(물리 규칙 라벨). 대상이 ALT 가 아니어서 위협 낮음. 문체는 피할 예다('framework' 1,000단어당 4.1회, 'comprehensive' 1.9회). CC BY-NC-ND.
- 파일: `references/14_scirep_exemplars/gay2026_zero_curtain_ai_eo.pdf`
