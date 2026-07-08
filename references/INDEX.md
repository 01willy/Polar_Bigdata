# References — 문헌 인벤토리 (Polar_Bigdata)

> 총 **49편**(web-검증 완료, 2026-07-06 문헌 워크플로 57 에이전트). 오픈액세스 PDF **36편** 카테고리 폴더에 다운로드, 나머지는 링크전용(페이월/자동차단).
> PDF는 저작권상 git 제외(`references/**/*.pdf`), 본 INDEX(.md)만 추적. 종합 분석·모델 로드맵·T1/T2 판정은 `gpt/handoff/20260706_1717-lit-review-forecasting-4d-tracks.md`.

**범례**: 📄=PDF 로컬 다운로드됨 · 🔗=링크전용(페이월/자동차단) · 관계=[벤치마크|경쟁자|선례|방법|맥락|off-direction]

## ⚠️ 인용 전 수정할 메타데이터 오류 (검증에서 적발)
- **Ran2022 공저자 일부 날조** → 정정: Ran Y., Li X., Cheng G., Che J., Aalto J., Karjalainen O., Hjort J., Luoto M., Jin H., Obu J., Hori M., Yu Q., Chang X.
- **`rahaman_northslope_ai_temps`(Earth Sci Inf 2024) 저자 오귀속** → 실제 저자 = **Chance, Ahajjam, Putkonen, Pasch** (MERRA-2 기반 GBDT/RF/SVR). Rahaman2025(arXiv:2510.06258)와 **다른 그룹**.
- **`ieki2025_neuralkriging_japan`** = 실제 저자 **Ieki et al.**(과거 노트의 'Suzuki 2025'는 오귀속).

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
- **무엇**: Physics-informed ML that regularizes a data-driven model with the heat equation, trained over permafrost monitoring data + climate projections to forecast permafrost thaw-degradation risk over a decadal horizon with improved numerical stability. Physics-constrained temporal risk forecasting precedent.
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
