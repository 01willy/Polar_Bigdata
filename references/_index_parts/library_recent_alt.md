# 최근 ALT·영구동토 ML 문헌 목록 (2022–2026)

- **작성일**: 2026-10-04
- **폴더**: `references/08_recent_alt_2022_2026/` (PDF 30편, 약 206 MB, 저작권상 git 제외)
- **범위**: 영구동토 활동층 두께(ALT)와 지온 예측의 ML, 물리 결합 ML, 새 지역 전이, 라벨 희소 조건, 관측망 설계, ALT 지도 제품. 2022–2026년 출판물이 중심이고, 인접 분야는 신규성 문장에 직접 걸리는 것만 넣었다.
- **조사 경로**: OpenAlex 검색 18건(첫 묶음), Crossref 서지 검색 약 40건, arXiv 검색 페이지 5건, 웹 검색 12건. OpenAlex는 첫 묶음 이후 공유 IP의 일일 무료 한도가 소진되어(재설정 UTC 자정) 이후 OA 위치는 출판사 URL 규칙과 Crossref 라이선스 필드로 확인했다. 이메일을 보내는 서비스(Unpaywall 등)는 쓰지 않았다.
- **내려받기 원칙**: 출판사 OA 페이지, Copernicus, arXiv, EarthArXiv, Zenodo, MDPI 파일 서버만 썼다. 섀도 라이브러리는 쓰지 않았다. 모든 파일은 `file`과 `pdfinfo`로 PDF 여부와 쪽수를 확인했다.
- **확인 수준 표기**: [본문] PDF 본문을 읽음, [초록] Crossref 또는 출판사 초록만 확인, [제목] 서지만 확인, [미확인] 확인하지 못한 사실.

## 범례

| 표기 | 뜻 |
|---|---|
| 📄 08 | 이번 작업에서 `08_recent_alt_2022_2026/`에 내려받음 |
| 📄 타 폴더 | 같은 시간대의 다른 작업이 이미 다른 폴더에 저장함. 중복을 피해 그 경로를 적음 |
| 🔓 차단 | 출판사 OA(라이선스 확인)이지만 자동 내려받기가 봇 차단(Cloudflare, Radware, Springer 인증 리다이렉트)으로 실패. 브라우저에서 직접 받을 수 있음 |
| 🔒 구독 필요 | 구독 문헌. DOI와 착지 URL만 기록 |
| 📚 기존 | 2026-07-06 인벤토리(`references/INDEX.md`, 49편)에 이미 있음 |

---

## 1. 신규성 문장과의 관계 요약

우리 신규성 문장: **학습에서 제외한 지역의 ALT 오차를, 대상 지역 라벨 수의 함수로, 같은 라벨로 재보정한 물리 기준선 대비 보고한다.** 세 요소(지역 홀드아웃, 라벨 수 축, 재보정 물리 기준선)를 함께 갖춘 ALT·MAGT 문헌은 이번 조사에서도 확인되지 않았다. 요소별로 겹치는 문헌은 아래와 같다.

| 문헌 | 대상 | 지역·지점 홀드아웃 | 대상 라벨 수 축 | 재보정 물리 기준선 | 판정 |
|---|---|---|---|---|---|
| O'Malley 2026 [본문] 📚 | 지중 온도(지열, 0–수 km) | 있음. 미국 학습, 앨버타·호주·영국 적용 | 있음. 문맥 관측 1, 2, 5, 10, 20, 40개(표 S2) | 없음. 물리 정보 ML 모형(Stanford Thermal Model)은 미국 안에서만 비교했고, 저자들은 이 모형을 다른 지역에 적용할 수 없다고 밝힌다. 전이 지역의 비교 대상은 universal kriging과 Transparent Earth다 | **설계 선례. 위협 중간.** 문장 범위를 "영구동토 ALT"와 "재보정 물리 기준선"으로 한정해야 한다 |
| Ahajjam 2025 [초록] | ALT, CALM 115지점 | [미확인] | 초록에는 없음 | 초록에는 없음(비교 대상은 ML 기준 모델) | **위협 미확인.** 투고 전 본문 대조 필수 |
| Wei 2026 [본문] | ALT 1 km 제품 | 지점 하나 제외(LOSO) | 없음 | 없음(CCI 제품과 지도 비교만) | 위협 낮음. 경쟁 제품으로 인용 |
| Du Q. 2026 [본문] | ALT, QTEC 시추공 54개 | 공간 블록 × 시간 홀드아웃 | 없음 | 없음 | 위협 낮음. 작은 효과 크기 보고 방식의 선례 |
| Gay 2026 [본문] | zero-curtain(ALT 아님) | 지리 분리 지점 CV | 없음 | 없음 | 위협 낮음 |
| Tama 2025 [본문] | 빙하 기반암 지형 | 버퍼를 둔 블록 홀드아웃 | 없음 | 물리 사전값 위 잔차(재보정 아님) | 위협 낮음. 잔차 구조와 홀드아웃 선례 |
| Feng 2023 [본문] | 유량 | 지역 홀드아웃(PUR) | 없음(대상 라벨 0) | 미분가능 물리 모형 | 위협 낮음. 인접 분야 |
| Garibaldi 2026 [초록] | TTOP 지온 | 지점 하나 제외 | 없음 | 매개변수 지점 간 비전이 보고 | 음성 결과(계수 비전이)의 직접 선례 |
| Portes 2026 [초록] | 일반 공간 ML | 관측 밖 외삽 | 현지 자료 단계 투입 | 없음 | 일반 설계 선례. 이미 원고 인용 목록에 있음 |
| Liu Z. 2024, Wang G. 2025 [본문] | ALT | 없음(무작위 10-fold) | 없음 | 없음 | 위협 없음. 무작위 분할의 낙관 편향 사례 |

조치 사항:
1. Ahajjam 2025의 검증 방식(지점 무작위 분할인지 지점·지역 홀드아웃인지)을 투고 전 본문으로 확인한다. 같은 연구진의 사전 공개본(Wilcox 2026, ESSOAr)도 자동 내려받기가 막혀 있다. 브라우저에서 받으면 된다.
2. O'Malley 2026 표 S2를 이번에 PDF 본문에서 직접 확인했다(앨버타 MAE: 관측 1개 3.65 °C, 5개 2.50 °C, 20개 2.22 °C, 40개 2.19 °C). 서론에서 "라벨 수 축 평가 자체는 인접 분야에 선례가 있다"로 인용한다.
3. 신규성 문장의 한정어("영구동토 ALT", "같은 라벨로 재보정한 물리 기준선 대비")를 유지한다.

---

## 2. ALT 예측 ML (직접 비교 대상)

### `ahajjam2025_multihorizon_alt_circumarctic_jgrmlc` 🔓 차단
- **서지**: Ahajjam, A., Wilcox, A., Soaper, M., Chance, R. & Pasch, T. Multi-horizon active layer thickness prediction across circum-Arctic permafrost regions using geospatial machine learning. *J. Geophys. Res. Mach. Learn. Comput.* 2, e2025JH000969 (2025). doi:10.1029/2025JH000969
- **OA**: CC BY 4.0(Crossref 라이선스). Wiley 사이트가 자동 내려받기를 차단했다. 착지 URL https://doi.org/10.1029/2025JH000969
- **요약** [초록]: CALM 115지점의 연 최대 ALT를 예측한다. 지리공간 특징 80개에서 다단계 선택으로 변수를 고르고, CatBoost·Extra Trees·Bagging 가중 앙상블을 예측 시차(당해, +1, +2, +5년)마다 따로 학습한다. 사전 공개본 초록은 R² 0.8 이상, RMSE 25 cm 미만을 보고한다.
- **신규성 관계**: **위협 미확인.** 초록에는 대상 지역 라벨 수 축과 물리 기준선이 없다. "다양한 북극 환경으로의 일반화"를 주장하므로 지점·지역 홀드아웃이 있다면 N1 서술의 "지역 홀드아웃 ALT 연구 없음" 부분을 고쳐야 한다.

### `wilcox2026_circumarctic_alt_geospatial_ml_essoar` 🔓 차단
- **서지**: Wilcox, A., Ahajjam, A., Soaper, M., Chance, R. & Pasch, T. A data-driven framework for active layer thickness prediction across circum-Arctic permafrost regions using geospatial machine learning. *ESS Open Archive* (2026). doi:10.22541/essoar.177306896.67279112/v1
- **OA**: CC BY 4.0. ESSOAr가 자동 내려받기를 차단(403)했다. 착지 URL https://doi.org/10.22541/essoar.177306896.67279112/v1
- **요약** [초록]: Ahajjam 2025와 같은 틀의 사전 공개본이다. 시차 4종, 다단계 변수 선택, 앙상블, CALM 115지점, R² 0.8 이상과 RMSE 25 cm 미만을 보고한다.
- **신규성 관계**: Ahajjam 2025와 같다. 본문 확인 경로로 쓴다.

### `ahajjam2026_alt_trajectory_causal_calm_egusphere` 📄 08
- **서지**: Ahajjam, A., Soaper, M., Gupta, U., Wilcox, A., Caparó Bellido, A., Weaver, S., Parker, S., Kidanu, S. & Pasch, T. Beyond permafrost observation: long-term ALT trajectory classification and causal inference across the circum-Arctic CALM network. *EGUsphere* preprint (2026). doi:10.5194/egusphere-2026-2639
- **OA**: CC BY 4.0, 08 폴더
- **요약** [본문]: CALM 129지점(1990–2024)의 ALT 시계열을 방향과 속도에 따라 6개 범주(ALDI)로 나누고, 지점 고정효과와 1차 차분 추정으로 범주별 동인을 추정한다. 두꺼워지는 지점과 얇아지는 지점의 비는 3.5 : 1이다. 열 강제는 두꺼워지는 지점에서만 지배적이다.
- **신규성 관계**: 예측이나 전이 평가가 아니다. 위협 없음. Ahajjam 연구진의 CALM 자료 처리 흐름을 확인하는 용도다.

### `wang2025_alt_stefan_catboost_et_qtp_rs` 📄 08 (같은 PDF가 `12_manuscript_refs/wang2025_alt_qtp_rs_ml.pdf`에도 있음)
- **서지**: Wang, G. et al. Simulation of active layer thickness based on multi-source remote sensing data and integrated machine learning models: a case study of the Qinghai-Tibet Plateau. *Remote Sens.* 17, 2006 (2025). doi:10.3390/rs17122006
- **OA**: CC BY(MDPI)
- **요약** [본문]: Stefan 식 결과를 입력 특징으로 넣고 CatBoost와 Extra Trees를 블렌딩한 SCE 모형으로 칭하이-티베트 고원 ALT를 추정한다. 10-fold 교차검증 MAE 20.7 cm, RMSE 32.7 cm, R² 0.873이다. 1958–2022년을 역산해 1998년 전후 증가율 0.25 → 1.26 cm/년을 보고한다.
- **신규성 관계**: 물리 출력을 입력으로 쓰는 구조의 선례다. 무작위 10-fold이고 지역 홀드아웃, 라벨 수 축, Stefan 단독 오차가 없다. 위협 없음.

### `zhang2024_climate_permafrost_model_efactor_ml_erl` 🔓 차단
- **서지**: Zhang, C., Douglas, T. A., Brodylo, D., Bosche, L. V. & Jorgenson, M. T. Combining a climate-permafrost model with fine resolution remote sensor products to quantify active-layer thickness at local scales. *Environ. Res. Lett.* 19, 044030 (2024). doi:10.1088/1748-9326/ad31dc
- **OA**: CC BY 4.0. IOP 사이트가 캡차로 막았다. 착지 URL https://iopscience.iop.org/article/10.1088/1748-9326/ad31dc
- **요약** [초록]: 기후-영구동토 모델의 토양 계수(edaphic factor)를 고해상도 초분광·라이다·위성 자료로 ML 추정해 ALT를 국지 규모로 산출한다. 인테리어 알래스카 실험지 2곳의 2014–2022년 현장 관측에서 ALT 분산의 60 % 이상을 설명한다.
- **신규성 관계**: E의 ML 추정 선례(국지 규모)다. 지역 간 전이와 라벨 수 축은 없다. 위협 없음.

### `du2026_seasonal_precip_alt_retrospective_buildings` 📄 08 (같은 PDF가 `12_manuscript_refs/du2026_qtec_alt_seasonal_precip.pdf`에도 있음)
- **서지**: Du, Q., Wang, F., Li, G. et al. Seasonal precipitation provides modest incremental information for retrospective estimation of active-layer thickness at monitored sites along the Qinghai–Tibet Engineering Corridor. *Buildings* 16, 3023 (2026). doi:10.3390/buildings16153023
- **OA**: CC BY(MDPI)
- **요약** [본문]: 칭하이-티베트 공학 회랑 시추공 54개의 연 ALT(2001–2020)에 지점 고정효과, 2차 시간 추세, 전년 ALT, 기온 지연을 넣은 뒤 강수의 추가 가치를 시험한다. 9개 보류 연도(2012–2020)의 롤링 원점 평가에서 RMSE가 2.05 % 줄었고, 엄격한 공간 블록 × 시간 홀드아웃에서는 전이되는 이득이 없었다.
- **신규성 관계**: 공간 블록 × 시간 홀드아웃과 작은 효과 크기 보고 방식이 우리 음성 결과 서술과 같은 유형이다. 라벨 수 축과 물리 기준선은 없다. 위협 없음.

### `du2026_alt_heterogeneity_arctic_foothills_tc` 📄 08
- **서지**: Du, J., Endsley, K. A., Bakian Dogaheh, K. et al. Assessing spatial heterogeneity of active layer thickness over Arctic-foothills tundra, North Slope Alaska. *The Cryosphere* 20, 4277–4291 (2026). doi:10.5194/tc-20-4277-2026
- **OA**: CC BY 4.0
- **요약** [본문]: 노스슬로프 구릉 툰드라의 90 m × 90 m 표본구 4개에서 탐침 ALT를 약 1700회 측정하고 드론·항공 자료와 랜덤 포레스트로 0.1 m 해상도 ALT 지도(5 km × 5 km)를 만든다. 해상도를 0.1 m에서 1000 m로 낮출수록 오차가 단계적으로 커지고, 10 m 해상도에서는 지형 요인의 기여가 약 65 %다.
- **신규성 관계**: 위협 없음. 1 km 셀 라벨의 대표성 한계와 격자 안 이질성 논의(WF 3차 "격자 안 상세도 없음")의 근거로 인용한다.

### `brodylo2024_alt_multiscale_interior_alaska_erl` 🔓 차단
- **서지**: Brodylo, D., Douglas, T. A. & Zhang, C. Quantification of active layer depth at multiple scales in Interior Alaska permafrost. *Environ. Res. Lett.* 19, 034013 (2024). doi:10.1088/1748-9326/ad264b
- **OA**: CC BY 4.0, IOP 캡차. 착지 URL https://iopscience.iop.org/article/10.1088/1748-9326/ad264b
- **요약** [초록]: 현장(1 m²) ALT를 항공 초분광·라이다로 1 km²까지 올리고, 다시 위성 자료로 100 km² 규모로 올리는 단계적 상향 규모화 틀이다.
- **신규성 관계**: 위협 없음. 점 라벨과 격자 규모 불일치의 근거.

### `hantson2025_scaling_arctic_features_ald_ere` 🔓 차단
- **서지**: Hantson, W., Yang, D., Serbin, S. P. et al. Scaling Arctic landscape and permafrost features improves active layer depth modeling. *Environ. Res. Ecol.* 4, 015001 (2025). doi:10.1088/2752-664X/ad9f6c
- **OA**: CC BY 4.0, IOP 캡차. 착지 URL https://iopscience.iop.org/article/10.1088/2752-664X/ad9f6c
- **요약** [초록]: 지상, 무인기, 항공(AVIRIS-NG), 위성(Sentinel-2) 자료를 함께 써서 활동층 깊이의 미세 이질성과 관측 규모의 관계를 분석한다. 경관 규모 평균 ALD는 항공·중해상도 위성 자료로 포착된다.
- **신규성 관계**: 위협 없음. 규모 의존성 논의용.

### `whitcomb2023_pband_polsar_alt_alaska_erl` 🔓 차단
- **서지**: Whitcomb, J., Chen, R., Clewley, D. et al. Maps of active layer thickness in northern Alaska by upscaling P-band polarimetric synthetic aperture radar retrievals. *Environ. Res. Lett.* 19, 014046 (2024; 온라인 2023). doi:10.1088/1748-9326/ad127f
- **OA**: CC BY, IOP 캡차. 착지 URL https://iopscience.iop.org/article/10.1088/1748-9326/ad127f
- **요약** [제목, 프로젝트 문서 기록]: P밴드 편광 SAR(AirMOSS) ALT 산출값을 북알래스카로 상향 규모화한 지도다.
- **신규성 관계**: 위협 없음. 원격탐사 ALT 산출의 맥락.

### `lin2026_regional_drivers_alt_site_scale_egusphere` 📄 08
- **서지**: Lin, Y., Zhang, B., Suo, H. et al. Regional variations in drivers of active layer thickness: a site-scale analysis across Northern Hemisphere permafrost. *EGUsphere* preprint (2026). doi:10.5194/egusphere-2026-841
- **OA**: CC BY 4.0
- **요약** [본문]: ALT 관측 지점 785개를 환북극(CAP), 아환북극(SCAP), 칭하이-티베트(QTP)로 나눈다. 평균 ALT는 84.9, 200, 224 cm이고, 5년 이상 기록이 있는 291지점의 60 %가 증가 추세다. PLS 경로 모형에서는 토양 특성이 기온보다 영향이 크고, 이 경향은 환북극과 고원에서 뚜렷하다. 같은 저자의 SSRN 판(doi:10.2139/ssrn.6044956)이 있다.
- **신규성 관계**: 예측 평가가 아니다. 위협 없음. 785지점 편집 목록은 라벨 출처 점검에 쓸 수 있다(자료는 CALM, GTN-P 등 공개 출처).

### `xiao2025_gpr_augmented_ml_mountain_permafrost_rs` 📄 08
- **서지**: Xiao, Y., Liu, G., Hu, G. et al. Mapping mountain permafrost via GPR-augmented machine learning in the northeastern Qinghai–Tibet Plateau. *Remote Sens.* 17, 2015 (2025). doi:10.3390/rs17122015
- **OA**: CC BY(MDPI)
- **요약** [본문]: 시추공, 토양 단면, GPR 횡단선 128개, 고지대 경험점 22개로 만든 존재·부재 표본 1037개로 분류기 13종을 비교한다(5-fold × 40회 반복). LightGBM·CatBoost가 F1 0.98 이상이다. GPR과 고지대 표본을 더하면 대표성이 낮은 지형에서 일반화가 좋아진다.
- **신규성 관계**: 위협 없음. "대표성이 낮은 곳에 관측을 더하면 일반화가 좋아진다"는 관측 설계 논의의 사례로 인용할 수 있다. 단 분류 문제이고 무작위 교차검증이다.

### `lin2025_miso_alaska_soil_permafrost_mapping_arxiv` 📄 08
- **서지**: Lin, Y., Chen, T., Brungard, C. et al. Fine-scale soil mapping in Alaska with multimodal machine learning. arXiv:2506.17302 (2025). SIGSPATIAL 2025 투고(arXiv 주석 기준)
- **OA**: arXiv
- **요약** [초록]: 지리공간 기반 모델 특징, 암묵 신경 표현, 대조 학습을 결합한 MISO로 알래스카 전역의 지표 근처 영구동토 존재와 토양 분류 지도를 만든다. 공간 교차검증과 영구동토대·MLRA별 분석에서 랜덤 포레스트보다 미관측 위치 일반화가 좋다.
- **신규성 관계**: ALT 회귀가 아니다. 위협 없음. 기반 모델 특징을 쓴 영구동토 지도의 공간 CV 사례.

### `kang2026_qtp_depth_resolved_permafrost_dl_climdyn` 🔒 구독 필요
- **서지**: Kang, X., Li, Z., Wan, J. et al. Real-time, depth-resolved permafrost thermal monitoring across the Qinghai-Tibet Plateau enabled by bayesian-optimized deep learning. *Clim. Dyn.* 64, 339 (2026). doi:10.1007/s00382-026-08304-y
- **OA**: [미확인]. Crossref 라이선스는 TDM 조항만 있다. 착지 URL https://doi.org/10.1007/s00382-026-08304-y
- **요약** [제목]: 베이즈 최적화 딥러닝으로 칭하이-티베트 고원의 깊이별 지온을 추정한다.
- **신규성 관계**: [미확인]. 지온 대상이고 ALT가 아니다. 초록 확인 후 분류한다.

### `zhang2026_qtp_permafrost_cmip6_ml_jclim` 🔒 구독 필요
- **서지**: Zhang, T., Zhang, T., Zhou, X. et al. Long-term evolution of permafrost across the Qinghai–Tibet Plateau: perspectives from multimodel ensembles and machine learning. *J. Clim.* 39, 2385–2400 (2026). doi:10.1175/JCLI-D-25-0473.1
- **요약** [초록]: CMIP6 자료와 SVR, CatBoost, LightGBM, DNN으로 칭하이-티베트 고원의 영구동토 범위와 최대 계절 동결 깊이를 예측하고 SSP 4종으로 2100년까지 투영한다. DNN이 R² 0.961로 가장 높다.
- **신규성 관계**: 위협 없음. 범위·동결 깊이 대상, 전이 평가 없음.

### `jiang2026_qtp_ml_reconstruction_multiscale_gpc` 🔒 구독 필요
- **서지**: Jiang, P., Ding, K., Ni, J. et al. Multi-scale pattern analysis of permafrost dynamics on the Qinghai–Tibet Plateau based on machine-learning reconstruction. *Glob. Planet. Change* 259, 105341 (2026). doi:10.1016/j.gloplacha.2026.105341
- **요약** [제목]: ML 재구성 자료로 칭하이-티베트 고원 영구동토 변화의 다중 규모 양상을 분석한다.
- **신규성 관계**: [미확인]. 맥락 인용 후보.

### `ding2025_qtp_ml_climate_memory_jhydrol` 🔒 구독 필요
- **서지**: Ding, K., Jiang, P., Ni, J. et al. Machine learning uncovers a multi-year climate memory in permafrost degradation on the Qinghai–Tibet Plateau: the critical roles of precipitation and [제목 이하 생략, 원문 확인 필요]. *J. Hydrol.* 663, 134272 (2025). doi:10.1016/j.jhydrol.2025.134272
- **요약** [제목]: ML로 영구동토 퇴화의 다년 기후 기억(강수 등)을 분석한다.
- **신규성 관계**: [미확인]. 위협 낮음으로 추정.

### `shen2023_qtp_permafrost_alt_1980_2020_stoten` 🔒 구독 필요
- **서지**: Shen, T., Jiang, P., Ju, Q. et al. Changes in permafrost spatial distribution and active layer thickness from 1980 to 2020 on the Tibet Plateau. *Sci. Total Environ.* 859, 160381 (2023). doi:10.1016/j.scitotenv.2022.160381
- **요약** [제목, 프로젝트 문서의 검색 요약 수준]: 1980–2020년 티베트 고원의 영구동토 분포와 ALT 변화를 추정한다. NOVELTY 문서는 "ALT에서 Stefan 우위 결론"으로 기록했으나 본문은 확인하지 못했다.
- **신규성 관계**: N6(라벨 0 직접 ML의 열세)의 방향 선례로 인용 예정. 인용 전 본문 대조 필요.

### `herrington2026_ml_reanalysis_soil_temperature_taac` 🔓 차단
- **서지**: Herrington, T. C., Erler, A. R. & Fletcher, C. G. Application of machine learning to improve reanalysis soil temperatures over the extratropical northern hemisphere. *Theor. Appl. Climatol.* 157, 419 (2026). doi:10.1007/s00704-026-06338-0
- **OA**: 2026-10-04 첫 OpenAlex 질의에서 출판사 PDF 위치가 표시되었다. Springer가 자동 내려받기를 막았다. 착지 URL https://doi.org/10.1007/s00704-026-06338-0
- **요약** [제목, 프로젝트 문서의 초록 수준]: 재분석 토양 온도를 ML로 보정하고 단순 보정 기준선과 비교한다.
- **신규성 관계**: N2(재보정 기준선)의 인접 선례. 위협 낮음.

### `kriuk2026_panarctic_hybrid_risk_m2garss` 🔒 구독 필요 (📚 arXiv판 있음)
- **서지**: Kriuk, B. Hybrid physics-ML framework for pan-Arctic permafrost infrastructure risk at record 2.9-million observation scale. *2026 IEEE M2GARSS*, 214–218 (2026). doi:10.1109/M2GARSS67833.2026.11582706
- **요약**: arXiv:2510.02189(📚 `04_spatiotemporal_4d/kriuk2025_panarctic_hybrid_risk.pdf`)의 학회판이다. 대상은 영구동토 비율과 기반시설 위험이다.
- **신규성 관계**: 위협 없음. 기존 인벤토리 판단과 같다.

---

## 3. 물리 결합·하이브리드·미분가능 모형

### `pilyugina2025_piml_permafrost_stability_ieeeaccess` 🔓 차단 (📚 arXiv판 있음)
- **서지**: Pilyugina, P., Chernikov, T., Smirnova, M. et al. A physics-informed machine learning framework for permafrost stability assessment. *IEEE Access* 13, 96423–96433 (2025). doi:10.1109/ACCESS.2025.3573072
- **OA**: CC BY-NC-ND 4.0. IEEE 사이트가 자동 요청에 502를 반환했다. 착지 URL https://ieeexplore.ieee.org/document/11014074/
- **요약**: 2023년 사전 공개본(📚 `06_physics_ml/pilyugina2023_pinn_permafrost_risk.pdf`)의 게재판이다. 사전 공개본 본문 기준으로 Kudryavtsev 모형 출력을 CatBoost 입력으로 넣어 ALT와 MAGT를 직접 예측하고, 시간 분할과 무작위 5-fold로 검증한다(NOVELTY 문서 10절).
- **신규성 관계**: 물리 출력 입력형 선례. 지역 홀드아웃과 라벨 수 축이 없다. 게재판에서 검증이 바뀌었는지는 [미확인]. 브라우저로 받아 대조한다.

### `tama2025_physics_guided_residual_bed_topography_wacv` 📄 08 (같은 논문이 `11_validation_and_methods/tama2025_physics_guided_residuals_bed.pdf`에도 있음)
- **서지**: Tama, B. A., Wang, J., Janeja, V. & Cham, M. Learning subglacial bed topography from sparse radar with physics-guided residuals. *Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV 2026)*, 5447–5456 (2026). arXiv:2511.14473. CVF 공개본 https://openaccess.thecvf.com/content/WACV2026/papers/Tama_Learning_Subglacial_Bed_Topography_from_Sparse_Radar_with_Physics-Guided_Residuals_WACV_2026_paper.pdf
- **OA**: arXiv판을 받았다. 쪽수는 웹 검색 결과 기준이다.
- **요약** [본문]: BedMachine 사전값에 대한 정규화 두께 잔차를 DeepLabV3+로 예측하고, 질량 보존·흐름 방향 총변동·비음수 등 물리 항으로 정칙화한다. 버퍼를 둔 블록 홀드아웃으로 그린란드 2개 소지역에서 RMSE 3.05–10.54 m를 보고한다.
- **신규성 관계**: "물리 기준선 + 잔차"와 "누설 방지 블록 홀드아웃"의 구성 선례(빙하). 라벨 수 축과 같은 라벨 재보정 기준선은 없다. 위협 낮음.

### `tian2026_noahpy_differentiable_permafrost_gmd` 📄 08
- **서지**: Tian, W., Yu, H., Zhao, S. et al. NoahPy: a differentiable Noah land surface model for simulating permafrost thermo-hydrology. *Geosci. Model Dev.* 19, 57–72 (2026). doi:10.5194/gmd-19-57-2026
- **OA**: CC BY 4.0
- **요약** [본문]: Noah 지표 모형의 열·수분 방정식을 순환 신경망 구조로 다시 구현해 미분 가능하게 만든다. 원 모형 재현 NSE는 0.99 이상이고, 영구동토 지점 1곳에서 보정한 결과 토양 온도 NSE 0.9 이상이다. Adam 기반 보정이 기존 보정보다 빠르고 안정적이다.
- **신규성 관계**: 미분가능 물리 모형의 영구동토 선례. 지역 전이 평가는 없다. 위협 없음. 토의의 "향후 과제" 인용 후보.

### `gou2026_differentiable_permafrost_thermal_compgeo` 🔒 구독 필요
- **서지**: Gou, L. & Likos, W. J. δHT4P: a differentiable physical modeling framework for thermal evolution of permafrost. *Comput. Geotech.* 196, 108132 (2026). doi:10.1016/j.compgeo.2026.108132
- **요약** [제목]: 영구동토 열 진화의 미분가능 물리 모형 틀이다.
- **신규성 관계**: [미확인]. NoahPy와 같은 범주.

### `zhao2025_hybrid_shaw_noah_rf_active_layer_wrr` 🔓 차단
- **서지**: Zhao, Y., Nan, Z., Ji, H. et al. A hybrid modeling approach for improved simulation of thermal-hydrological dynamics in active layer on the Qinghai-Tibet Plateau. *Water Resour. Res.* 61, e2025WR040288 (2025). doi:10.1029/2025WR040288
- **OA**: CC BY-NC 4.0. Wiley 차단. 착지 URL https://doi.org/10.1029/2025WR040288
- **요약** [초록]: 랜덤 포레스트로 보정한 Noah 모의값을 SHAW 모형의 하부 경계조건으로 써서 자료가 적은 지역에서 활동층 온도·수분을 모의한다. 고원 영구동토 7지점의 시험 자료에서 토양 온도 NSE 0.81(Noah 0.69)이다.
- **신규성 관계**: ML 보정 + 물리 모형 결합의 선례. 지역 전이·라벨 수 축 없음. 위협 없음.

### `uxa2026_analytical_statistical_alt_mapt_tc` 📄 08
- **서지**: Uxa, T., Hrbáček, F. & Kňažková, M. Simple analytical–statistical models (ASMs) for mean annual permafrost table temperature and active-layer thickness estimates. *The Cryosphere* 20, 97–112 (2026). doi:10.5194/tc-20-97-2026
- **OA**: CC BY 4.0
- **요약** [본문]: 활동층 안 두 깊이의 융해·동결 지수만으로 영구동토 상면 온도와 ALT를 추정하는 해석·통계 모형 2종을 제시한다. 주요 영구동토 지역 55지점에서 평균 오차가 0.05 °C, 9 % 미만이고, 저자들은 다른 해석·통계 모형과 같거나 낫다고 보고한다.
- **신규성 관계**: 물리 기준선 후보지만 활동층 안 지온 관측이 필요해 미관측 셀에는 쓸 수 없다. 우리 기준선 선택(Stefan, Kudryavtsev)의 근거 문단에 인용한다.

### `mccormick2026_analytical_active_layer_thaw_subsidence_eartharxiv` 📄 08
- **서지**: McCormick, L. & Schmidt, D. Analytical prediction of active-layer thaw and subsidence under seasonal thermal forcing: application to Svalbard permafrost. *EarthArXiv* preprint (2026). doi:10.31223/X5WJ5W (PDF 내부 제목: "From an exact thaw-consolidation solution to active-layer prediction: asymptotic limits, uncertainty, and application to an Arctic permafrost site")
- **OA**: EarthArXiv
- **요약** [본문]: Lunardini의 융해·압밀 정확해에서 Stefan 수가 작을 때와 클 때의 근사식을 유도하고(오차 약 2 %, 1.3 %) 민감도를 정량한다. 계절 평균 지표 온도를 넣으면 계절 말 ALT를 약 8 % 과대 추정하고 융해 궤적을 약 2주 앞당긴다.
- **신규성 관계**: 위협 없음. Stefan 계열 기준선의 구조적 편향을 설명하는 근거.

### `gay2026_zero_curtain_ai_eo` 📄 타 폴더 (`14_scirep_exemplars/gay2026_zero_curtain_ai_eo.pdf`)
- **서지**: Gay, B. A., Miner, K. R., Rietze, N., Poulter, B., Pastick, N. J. & Miller, C. E. Resolving circumarctic zero-curtain phenomena with AI-integrated earth observations. *Sci. Rep.* 16, 28715 (2026). doi:10.1038/s41598-026-61719-9
- **OA**: CC BY(Nature). 08 폴더에도 받았으나 먼저 저장된 14 폴더 사본을 남기고 지웠다.
- **요약** [본문]: 현장 관측 6271만 건과 원격탐사 33억 건을 통합한 물리 정보 전이 학습 틀(GeoCryoAI)로 zero-curtain 후보를 30 m로 추정한다. 후보 탐지 정확도 96.4 %, 지리 분리 지점 교차검증에서 성능 저하 2 % 미만이다.
- **신규성 관계**: 물리 규칙 라벨과 물리 손실의 선례(대상은 ALT가 아님). 위협 낮음. Sci Rep 영구동토 ML 논문의 형식 참고 자료다.

### `peng2026_physics_guided_ml_air_temperature_warm_permafrost_accr` 🔒 구독 필요 [OA 여부 미확인]
- **서지**: Peng, C.-Y., Luo, D., Sheng, Y. et al. Physics-guided machine learning approach for reconstructing air temperature in warm permafrost on the Qinghai‒Xizang Plateau. *Adv. Clim. Change Res.* 17, 740–754 (2026). doi:10.1016/j.accre.2026.04.013
- **OA**: ACCR는 OA 학술지로 알려져 있으나 이번에 확인하지 못했다. 착지 URL https://doi.org/10.1016/j.accre.2026.04.013
- **요약** [제목]: 물리 유도 ML로 따뜻한 영구동토 지역의 기온을 재구성한다.
- **신규성 관계**: 강제 자료 재구성이며 ALT 전이가 아니다. 위협 없음으로 추정.

### `yu2026_knowledge_guided_freeze_thaw_essdd` 📄 08
- **서지**: Yu, H., Wu, M., Yin, D. et al. A knowledge-guided daily multi-layer soil freeze-thaw dataset for the Northern Hemisphere during 1950–2025. *Earth Syst. Sci. Data Discuss.* (2026). doi:10.5194/essd-2026-683
- **OA**: CC BY 4.0
- **요약** [본문]: 토양 온도 모의값에서 동결·융해 관계를 먼저 학습하고 현장 관측으로 제약하는 지식 유도 신경망(FT-KGML)으로 북반구 0.1°, 1950–2025년, 깊이 10·30·50 cm의 일별 동결·융해 상태를 만든다. 독립 관측소 일별 분류 정확도는 86.1, 89.7, 90.4 %다.
- **신규성 관계**: 물리 모의 사전학습 후 관측 제약의 영구동토 선례. ALT·라벨 수 축 없음. 위협 없음.

### `garibaldi2026_ttop_parameter_importance` 📄 타 폴더 (`12_manuscript_refs/garibaldi2026_ttop_parameter_importance.pdf`)
- **서지**: Garibaldi, M. C., Bonnaventure, P. P., Way, R. G. et al. Determining TTOP model parameter importance and overall performance across northern Canada. *The Cryosphere* 20, 2375–2392 (2026). doi:10.5194/tc-20-2375-2026
- **OA**: CC BY 4.0. 08 폴더 사본은 중복이라 지웠다.
- **요약** [초록]: 캐나다 북부 330지점의 기온·지온 자료로 TTOP 매개변수의 중요도를 지점 하나 제외 교차검증과 랜덤 포레스트로 평가한다. 동결기 n-factor와 동결 도일이 성능을 좌우하고, 토지피복 기반 매개변수는 지점 간에 전이되지 않는다.
- **신규성 관계**: "경험 계수가 지역을 넘어 옮겨지지 않는다"(N7)의 직접 선례. 우리 기여는 이를 ALT 오차 단위와 라벨 수 축으로 수치화한 것이다.

### `liu2025_surface_thermal_offsets_three_poles_npjclim` 📄 08
- **서지**: Liu, J., Luo, D., Wu, Q. et al. Divergent controls on surface and thermal offsets in permafrost across the three poles. *npj Clim. Atmos. Sci.* 8, 354 (2025). doi:10.1038/s41612-025-01235-1
- **OA**: CC BY
- **요약** [본문]: 삼극 117지점에서 지표 오프셋(SO)은 대규모 기후, 열 오프셋(TO)은 국지 기질이 지배함을 보인다. SO는 북극 3.1 °C, 제3극 3.2 °C, 남극 1.0 °C다.
- **신규성 관계**: 물리 매개변수의 지역 차 근거(N7 보조). 위협 없음.

### `zhao2025_west_kunlun_permafrost_thermal_state_egusphere` 📄 08
- **서지**: Zhao, J., Zhao, L., Sun, Z., Hu, G., Zou, D. et al. The thermal state of permafrost under climate change on the Qinghai-Tibet Plateau from 1980 to 2022: a case study of the West Kunlun. *EGUsphere* preprint (2025). doi:10.5194/egusphere-2024-3956 (원 제목에 "in under"라는 중복 표현이 있다)
- **OA**: CC BY 4.0
- **요약** [본문]: ML로 만든 1 km 월별 지표 온도를 강제 자료로 이동 격자 영구동토 모형(MVPM)을 돌려 서쿤룬 지역(약 5.6만 km²)의 지온과 ALT를 모의한다. 지온 ±0.25 °C, ALT ±0.25 m 정확도를 보고한다.
- **신규성 관계**: 물리 모형 + ML 강제 자료 결합. 위협 없음.

---

## 4. 전이·라벨 희소 조건 (인접 분야 포함)

### `omalley2026_incontext_subsurface_temp` 📚 기존 (`05_uq_transfer/`)
- **서지**: O'Malley, D. et al. In-context learning enables continental-scale subsurface temperature prediction from sparse local observations. arXiv:2605.16665 (2026)
- **이번 확인** [본문]: 미국에서 학습한 트랜스포머를 재학습 없이 앨버타·호주·영국에 적용하고, 문맥 관측 수를 1–40개로 바꾼 MAE를 보고한다(표 S2: 앨버타 3.65 → 2.19 °C, 호주 7.50 → 6.21 °C, 영국 6.86 → 5.34 °C). 물리 정보 ML 모형인 Stanford Thermal Model은 미국 안에서만 비교했고(MAE 6.4 °C), 저자들은 이 모형을 다른 지역에 적용할 수 없다고 밝힌다. 전이 지역의 비교 대상은 universal kriging과 Transparent Earth다.
- **신규성 관계**: **설계 선례, 위협 중간.** 1절 조치 2 참조.

### `feng2023_differentiable_hydrology_ungauged_hess` 📄 08 (같은 PDF가 `11_validation_and_methods/feng2023_differentiable_hydrology_ungauged.pdf`에도 있음)
- **서지**: Feng, D., Beck, H., Lawson, K. & Shen, C. The suitability of differentiable, physics-informed machine learning hydrologic models for ungauged regions and climate change impact assessment. *Hydrol. Earth Syst. Sci.* 27, 2357–2373 (2023). doi:10.5194/hess-27-2357-2023
- **OA**: CC BY 4.0
- **요약** [본문]: 신경망이 HBV 매개변수를 예측하는 미분가능 모형(δ)을 LSTM과 비교한다. 무작위 미계측 유역(PUB)에서는 비슷하거나 낫고, 지역 홀드아웃(PUR)에서는 δ 모형이 일별 지표와 평균·고유량 추세에서 LSTM보다 낫다.
- **신규성 관계**: 하이브리드 모형의 지역 홀드아웃 평가 선례(대상 라벨 0 고정). 이미 원고 인용 목록에 있다.

### `portes2026_ml_extrapolation_local_data_spatstat` 🔒 구독 필요
- **서지**: Portes, C., Ienco, D. & Gabriel, E. When machine learning extrapolates in space: how local data shape spatial transferability. *Spat. Stat.* 74, 101009 (2026). doi:10.1016/j.spasta.2026.101009
- **요약** [초록, 프로젝트 문서 기록]: 관측 영역 밖으로 외삽할 때 현지 자료를 단계적으로 더하는 효과를 평가한다.
- **신규성 관계**: "미관측 지역에 현지 자료를 단계 투입하는 평가 최초"를 쓸 수 없게 하는 일반 선례. 이미 원고 인용 목록에 있다.

### `du2026_qtp_active_layer_moisture_90m_essdd` 📄 08
- **서지**: Du, E., Wu, T., Dai, L. et al. A first 90 m resolution active layer moisture dataset across the Qinghai–Tibet Plateau permafrost region. *Earth Syst. Sci. Data Discuss.* (2026). doi:10.5194/essd-2026-330
- **OA**: CC BY 4.0
- **요약** [본문]: 2009–2024년 현장 표본 342개와 원격탐사·지형·토양 변수로 활동층 평균 체적 수분을 90 m로 만든다. 무작위 5-fold R² 0.62–0.63에 비해 그룹 기반 공간 교차검증 R² 0.30–0.38로 낮고, 유역 하나 제외 검증과 AOA 마스크를 함께 제공한다.
- **신규성 관계**: 무작위 분할의 낙관 편향을 같은 논문에서 보고한 영구동토 사례(N8 보조). 이 자료 계열의 GPR·현장 ALT 점은 우리 티베트 라벨 출처이기도 하다(원고 Methods의 Du et al. 2026 Zenodo 코드).

---

## 5. 관측망 설계·대표성

### `pallandt2022_panarctic_ec_network_representativeness_bg` 📄 08
- **서지**: Pallandt, M. M. T. A., Kumar, J., Mauritz, M. et al. Representativeness assessment of the pan-Arctic eddy covariance site network and optimized future enhancements. *Biogeosciences* 19, 559–583 (2022). doi:10.5194/bg-19-559-2022
- **OA**: CC BY 4.0
- **요약** [본문]: 생기후·토양 변수 18개로 북극 와상관 관측망의 대표성 지표(ER1, ER4)를 정의한다. 영역의 절반은 탑 1개 이상으로 대표되지만 신뢰할 외삽이 가능한 곳은 3분의 1이다. 새 지점 15개를 더하면 대표성이 20 % 오른다.
- **신규성 관계**: 관측망 설계의 방법 선례(대상은 탄소 플럭스). 관측 위치 알고리즘 절에서 환경 공간 대표성 지표의 출처로 인용할 수 있다.

### `betti2025_mapping_on_a_budget_spatial_sampling_arxiv` 📄 08
- **서지**: Betti, L., Sanni, F., Sogoyou, G. et al. (PDF 1쪽 저자 7명 이상, 전체 목록은 PDF 확인) Mapping on a budget: optimizing spatial data collection for ML. arXiv:2509.03749 (2025)
- **OA**: arXiv
- **요약** [본문 1쪽]: 위성 영상 ML에서 비용이 다른 지점과 예산 제약 아래 학습 자료 수집 위치를 최적화하는 문제를 처음 정식화하고 방법을 제시한다.
- **신규성 관계**: 관측 위치 선택(추가 탐사) 설계의 일반 선례. 영구동토 적용은 없다. 위협 없음.

### `streletskiy2022_gtnp_measurement_guidelines_zenodo` 📄 08
- **서지**: Streletskiy, D., Noetzli, J., Smith, S. L., Vieira, G., Schoeneich, P., Hrbáček, F. & Irrgang, A. M. Measurement recommendations and guidelines for the Global Terrestrial Network for Permafrost (GTN-P). Zenodo (2022). doi:10.5281/zenodo.5973079
- **OA**: Zenodo 공개
- **요약** [본문 1–3쪽]: GTN-P의 시추공 지온(TSP)과 활동층(CALM) 측정 권고와 지침이다. 문서 스스로 WMO Guide No. 8 개정판의 영구동토 모범 지침으로 대체될 임시 문서라고 밝힌다.
- **신규성 관계**: 라벨 정의(탐침, 지온 유도)의 표준 인용.

### `streletskiy2026_calm_long_term_alt_cee` 📄 08 (같은 PDF가 `12_manuscript_refs/streletskiy2026_calm_alt_degradation.pdf`에도 있음)
- **서지**: Streletskiy, D. A., Nyland, K. E., Shiklomanov, N. I. et al. Long-term monitoring of active layer thickness confirms global permafrost degradation. *Commun. Earth Environ.* 7, 671 (2026). doi:10.1038/s43247-026-03824-1
- **OA**: CC BY
- **요약** [본문]: 북극, 남극, 산악 영구동토 156지점의 2000–2024년 ALT 관측을 종합한다. 유의한 증가는 북극 55 %, 남극 38 %, 유럽 산악·아시아 고지대 90 % 이상 지점에서 나타난다. 북극 변화는 융해 도일 증가가 가장 크게 설명한다. CALM 지점 분포가 공간적으로 고르지 않다고 적는다.
- **신규성 관계**: 위협 없음. 서론 동기, √DDT 관계, 관측망 불균등의 근거.

### `brown2025_beyond_magt_monitoring_metrics_egusphere` 📄 08
- **서지**: Brown, N. & Gruber, S. Beyond MAGT: learning more from permafrost thermal monitoring data with additional metrics. *EGUsphere* preprint (2025). doi:10.5194/egusphere-2025-2658
- **OA**: CC BY 4.0
- **요약** [본문]: 120년 모의 70여 개로 지온 지표를 평가해 영구동토 상면 높이, 연교차 0 깊이, 열 적분, MAGT, MAGST의 5개 지표를 권고한다. 10–20 m 사이 센서 깊이에 따라 MAGT 추세가 10년 관측 기간의 절반에서 0.23 °C/10년 이상 다르다.
- **신규성 관계**: 관측 설계(센서 배치)의 영구동토 선례. 위협 없음.

### `tregubov2024_talik_monitoring_network_gpr_urbsci` 📄 08
- **서지**: Tregubov, O. D. & Uyagansky, K. K. Substantiation of the monitoring network of talik zones in urbanized permafrost areas based on GPR profiling data (Anadyr, Chukotka). *Urban Sci.* 8, 94 (2024). doi:10.3390/urbansci8030094
- **OA**: CC BY(MDPI)
- **요약** [본문]: 아나디리 시의 지온과 GPR 탐사로 탈릭 경계를 그리고, 위험 구역 20곳의 경계·중심 관측정 35개와 GPR 통제 측선 12개로 감시망을 설계한다.
- **신규성 관계**: 국지 규모 감시망 설계 사례. 위협 없음.

### `zhang2024_lateral_heat_flow_permafrost_modeling_scirep` 📄 08
- **서지**: Zhang, Y., Hong, G. & Bonney, M. T. Impacts of lateral conductive heat flow on ground temperature and implications for permafrost modeling. *Sci. Rep.* 14, 31595 (2024). doi:10.1038/s41598-024-78901-6
- **OA**: CC BY
- **요약** [본문]: 수평 전도 열흐름이 지온에 주는 영향을 평형·과도 조건에서 계산하고 캐나다 북동부 광산 지역 시추공 191개의 깊이별 지온 차이를 설명한다. 격자가 작으면 1차원 모형 오차가 5 °C에 이를 수 있고, 결과는 관측 지점 선정 지침으로 쓸 수 있다고 적는다.
- **신규성 관계**: 1차원 물리식의 한계와 관측 위치 선정 근거. Sci Rep 영구동토 모형 논문의 형식 참고.

(관측 설계 관련으로 2절의 `xiao2025`(대표성 낮은 지형 표본 추가 효과)와 4절의 `du2026_qtp_active_layer_moisture`(AOA 마스크)도 함께 본다.)

---

## 6. ALT 지도 제품·자료

### `wei2026_nh_alt_1km_2000_2024_essdd` 📄 08
- **서지**: Wei, Y., Wu, Z., Wang, J. et al. A 1 km resolution dataset of Northern Hemisphere permafrost active layer thickness (2000–2024). *Earth Syst. Sci. Data Discuss.* (2026). doi:10.5194/essd-2026-447. 자료 doi:10.5281/zenodo.21667583, 코드 https://zenodo.org/records/21835259
- **OA**: CC BY 4.0
- **요약** [본문]: 연 ALT 관측 2196건으로 앙상블 ML(5종 중 CatBoost·LightGBM 결합)을 학습해 북반구 1 km 연별 ALT(2000–2024)를 만든다. 지점 하나 제외 교차검증 R² 0.76, RMSE 60.48 cm이고, 관측·예측 Sen 기울기 상관은 0.73이다. CCI(Westermann 2024, 평균 67.9 cm)와 Peng et al. 2024 ML 제품(평균 162.9 cm)의 중간(125.7 cm)이다.
- **신규성 관계**: 경쟁 제품. 학습 표가 CALM 지점이라 우리 채점 셀과 겹칠 수 있다(EXPERIMENT_PLAN_LG 기록). 라벨 수 축·지역 홀드아웃·물리 기준선 비교가 없으므로 위협 낮음. 제품 간 차이가 수십 cm라는 점은 우리 기존 제품 비교 SI의 근거다.

### `liu2024_widespread_alt_deepening_2003_2020_erl` 📄 08 (같은 PDF가 `12_manuscript_refs/liu2024_active_layer_deepening_2003_2020.pdf`에도 있음)
- **서지**: Liu, Z., Kimball, J. S., Ballantyne, A. et al. Widespread deepening of the active layer in northern permafrost regions from 2003 to 2020. *Environ. Res. Lett.* 19, 014020 (2024; 온라인 2023). doi:10.1088/1748-9326/ad0f73. 격자 자료 doi:10.5281/zenodo.10070609
- **OA**: CC BY 4.0. IOP는 캡차로 막혀 White Rose 기관 저장소 사본(https://eprints.whiterose.ac.uk/id/eprint/206431/)을 받았다.
- **요약** [본문]: 현장 ALT 2966 지점-연으로 랜덤 포레스트를 학습해 북부 영구동토 지역 1 km 연별 ALT(2003–2020)를 만든다. 무작위 10-fold 교차검증 상위 10개 모형 앙상블이 RMSE 21.6 cm, R² 0.97이다. 지역의 약 65 %가 깊어지는 추세(평균 0.11 cm/년)다.
- **신규성 관계**: 무작위 분할 성능이 공간 분리 성능보다 크게 좋게 나오는 사례(Wei 2026 LOSO RMSE 60.5 cm와 대비). 위협 없음.

### `li2022_nh_alt_2000_2018_stefan_jgra` 🔒 구독 필요
- **서지**: Li, C., Wei, Y., Liu, Y. et al. Active layer thickness in the Northern Hemisphere: changes from 2000 to 2018 and future simulations. *J. Geophys. Res. Atmos.* 127, e2022JD036785 (2022). doi:10.1029/2022JD036785. 관련 자료 Dryad doi:10.5061/dryad.f4qrfj6zh [내용 미확인]
- **요약** [초록]: 영구동토 관측 자료와 ERA5-Land 기온으로 Stefan 모형을 구동해 북반구 1 km ALT(2000–2018)를 모의하고 CMIP6로 미래를 투영한다. 평균 ALT는 127.19 → 145.37 cm(0.65 cm/년)로 증가한다.
- **신규성 관계**: 프로젝트 문서의 "Li C. 2022"가 이 논문이다(첫 저자 Chuanhua Li). 현지 관측으로 E를 정하는 관행의 출처(N5 조건 문단). 위협 없음.

### `li2022_nh_permafrost_extent_alt_1969_2018_stoten` 🔒 구독 필요
- **서지**: Li, G., Zhang, M., Pei, W. et al. Changes in permafrost extent and active layer thickness in the Northern Hemisphere from 1969 to 2018. *Sci. Total Environ.* 804, 150182 (2022). doi:10.1016/j.scitotenv.2021.150182
- **요약** [제목]: 북반구 영구동토 범위와 ALT의 1969–2018년 변화를 추정한다.
- **신규성 관계**: [미확인]. Li C. 2022와 저자가 다르다(첫 저자 Guanji Li). 혼동하지 않는다.

### `peng2018_nh_alt_changes` 📄 타 폴더 (`12_manuscript_refs/peng2018_nh_alt_changes.pdf`)
- **서지**: Peng, X., Zhang, T., Frauenfeld, O. W., Wang, K., Luo, D., Cao, B., Su, H., Jin, H. & Wu, Q. Spatiotemporal changes in active layer thickness under contemporary and projected climate in the Northern Hemisphere. *J. Clim.* 31, 251–266 (2018). doi:10.1175/JCLI-D-16-0721.1
- **요약** [검색 결과 수준]: 융해 지수와 토양 계수(edaphic factor)로 북반구 ALT를 추정하고 미래 기후에서 투영한다.
- **신규성 관계**: 범위(2022–2026) 밖이지만 지정 항목이라 넣었다. 현지 관측으로 E를 정하는 관행의 출처.

### `peng2023_alt_permafrost_area_projections_ef` 🔓 차단
- **서지**: Peng, X., Zhang, T., Frauenfeld, O. W. et al. Active layer thickness and permafrost area projections for the 21st century. *Earth's Future* 11, e2023EF003573 (2023). doi:10.1029/2023EF003573
- **OA**: Earth's Future는 OA 학술지다. Wiley 차단. 착지 URL https://doi.org/10.1029/2023EF003573
- **요약** [제목, Wei 2026 참고문헌 기준]: 21세기 ALT와 영구동토 면적을 투영한다. 공저자에 Hjort, Aalto, Karjalainen, Luoto가 있다. 방법은 [미확인]. Wei 2026이 비교한 "Peng et al. 2024" 1 km ML 제품(Peng, X., Jin, H. & Zhao, G. 북반구 1850–2100 ALT 자료, 2024)은 이 계열로 보이나 DOI는 [미확인]이다.
- **신규성 관계**: 기존 제품 비교 후보. 위협 없음.

### `liu2025_reanalysis_permafrost_alt_asl` 🔓 차단
- **서지**: Liu, Z., Guo, D., Hua, W. et al. Near-surface permafrost extent and active layer thickness characterized by reanalysis/assimilation data. *Atmos. Sci. Lett.* 26, e1289 (2025). doi:10.1002/asl.1289
- **OA**: CC BY 4.0. Wiley 차단. 착지 URL https://doi.org/10.1002/asl.1289
- **요약** [초록]: 재분석·동화 자료 7종(CFSR, MERRA-2, ERA5, ERA5-Land, GLDAS 3종)의 영구동토 범위·ALT 표현력을 비교한다. 대부분 표현력이 제한적이고 GLDAS-CLSMv20이 가장 낫다.
- **신규성 관계**: ERA5-Land 강제 자료 편향 논의의 근거(계수 차이의 원인 분리 한계, NOVELTY 6절). 위협 없음.

### `guo2026_ensemble_thaw_conditions_essdd` 📄 08
- **서지**: Guo, D., Wang, C. & Zang, S. An ensemble dataset of permafrost thaw conditions for northern high latitudes from open satellite data. *Earth Syst. Sci. Data Discuss.* (2026). doi:10.5194/essd-2026-299. 자료 doi:10.5281/zenodo.19148960
- **OA**: CC BY 4.0
- **요약** [본문]: 공개 영구동토 제품 3종을 앙상블해 영구동토 비율과 MAGT를 만들고, 위성 자료 16종과 XGBoost로 융해 취약 지수(PTI)를 예측한다(전체 정확도 91.8 %). 시추공 26곳과 Spearman r 0.69다.
- **신규성 관계**: ALT 제품이 아니다. 위협 없음.

### `zou2025_qtp_permafrost_temperature_15m_essd` 📄 08
- **서지**: Zou, D., Zhao, L., Hu, G. et al. Permafrost temperature baseline at 15 m depth on the Qinghai–Tibetan Plateau (2010–2019). *Earth Syst. Sci. Data* 17, 1731–1742 (2025). doi:10.5194/essd-17-1731-2025
- **OA**: CC BY 4.0
- **요약** [본문]: 시추공 231개의 15 m 지온으로 SVR을 학습해 약 1 km 격자 MAGT15m를 만든다(R² 0.48). 고원 평균 −1.85 ± 1.58 °C다.
- **신규성 관계**: 위협 없음. 티베트 지온 자료의 출처 후보.

### `talucci2025_firealt_paired_burned_essd` 📄 08
- **서지**: Talucci, A. C., Loranty, M. M., Holloway, J. E. et al. Permafrost–wildfire interactions: active layer thickness estimates for paired burned and unburned sites in northern high latitudes. *Earth Syst. Sci. Data* 17, 2887–2909 (2025). doi:10.5194/essd-17-2887-2025
- **OA**: CC BY 4.0
- **요약** [본문]: 기여자 18명의 융해 깊이 52,466건을 모아 수정 Stefan 식으로 계절 말 ALT 48,669건을 추정한다(9446 구획, 화재·비화재 쌍 157개).
- **신규성 관계**: 우리 라벨 출처(원고 Methods의 Talucci 2024 자료)의 자료 논문. 계절 중 측정을 Stefan 식으로 계절 말로 환산한 라벨이므로 라벨 정의 혼합 민감도 논의에 쓴다.

### `zhu2024_alaska_tundra_vegetation_active_layer_db_essd` 📄 08
- **서지**: Zhu, X., Chen, D., Kogure, M. et al. A synthesized field survey database of vegetation and active-layer properties for the Alaskan tundra (1972–2020). *Earth Syst. Sci. Data* 16, 3687–3703 (2024). doi:10.5194/essd-16-3687-2024
- **OA**: CC BY 4.0
- **요약** [본문]: 알래스카 툰드라의 식생·활동층 현장 조사 자료를 통합하고 화재 이력을 붙인 데이터베이스다. 구획 크기는 1 m × 1 m부터 다양하다.
- **신규성 관계**: 알래스카 추가 라벨 출처 후보. 위협 없음.

### `chang2024_deformation_alt_nonlinear_npjclim` 📄 08
- **서지**: Chang, T., Yi, Y., Jiang, H. et al. Unraveling the non-linear relationship between seasonal deformation and permafrost active layer thickness. *npj Clim. Atmos. Sci.* 7, 308 (2024). doi:10.1038/s41612-024-00866-0
- **OA**: CC BY
- **요약** [본문]: 칭하이-티베트 고원에서는 ALT와 계절 지표 변형의 상관이 음(r = −0.53)이어서 배수가 나쁜 북극 토양의 양의 관계와 반대다. 식생이 성기고 토양이 건조할수록 더 음이 된다.
- **신규성 관계**: InSAR 기반 ALT 공변량의 지역 의존성 근거. SAR를 예측 변수로 쓰지 않은 이유의 보조 근거. 위협 없음.

---

## 7. 리뷰

### `greene2026_computational_methods_permafrost_survey_peps` 🔓 차단
- **서지**: Greene, T., Kaabouch, N. & Pasch, T. Advanced computational methods for predicting permafrost conditions: a survey. *Prog. Earth Planet. Sci.* 13, 55 (2026). doi:10.1186/s40645-026-00828-5
- **OA**: SpringerOpen(OA 학술지). Springer 인증 리다이렉트로 자동 내려받기 실패. 착지 URL https://doi.org/10.1186/s40645-026-00828-5
- **요약** [제목]: 영구동토 상태 예측의 계산 방법 리뷰다. 공저자 Pasch는 Ahajjam 2025의 공저자다.
- **신규성 관계**: [미확인]. Ahajjam 2025의 검증 방식을 요약했을 가능성이 있어 함께 확인한다.

### `bartsch2023_permafrost_monitoring_from_space_survgeophys` 🔓 차단
- **서지**: Bartsch, A., Strozzi, T. & Nitze, I. Permafrost monitoring from space. *Surv. Geophys.* 44, 1579–1613 (2023). doi:10.1007/s10712-023-09770-3
- **OA**: OpenAlex 첫 질의에서 OA PDF 위치 표시. Springer 차단. 착지 URL https://doi.org/10.1007/s10712-023-09770-3
- **요약** [제목]: 위성 원격탐사 기반 영구동토 감시의 리뷰다.
- **신규성 관계**: 맥락. 위협 없음.

### `zhao2024_qtp_permafrost_review_ppp` 🔓 차단
- **서지**: Zhao, L., Hu, G., Liu, G. et al. Investigation, monitoring, and simulation of permafrost on the Qinghai-Tibet Plateau: a review. *Permafr. Periglac. Process.* 35, 412–422 (2024). doi:10.1002/ppp.2227
- **OA**: OpenAlex 첫 질의에서 OA PDF 위치 표시. Wiley 차단. 착지 URL https://doi.org/10.1002/ppp.2227
- **요약** [제목]: 칭하이-티베트 고원 영구동토 조사·감시·모의의 리뷰다.
- **신규성 관계**: 티베트 독립 지역 자료 확보 경로 확인용.

### `memis2025_ml_permafrost_degradation_review` 📚 기존 (`05_uq_transfer/koven_review2025_ml_permafrost.pdf`)
- **서지**: Memiş, M. A., Keskin, I., Demir, S. & Ulus Memiş, Ş. Machine learning-based prediction of permafrost degradation and its implications on geotechnical infrastructure: a comprehensive review. *AI Civ. Eng.* 4, 28 (2025). doi:10.1007/s43503-025-00080-8
- **확인**: 기존 파일명의 "koven"은 오귀속이다. PDF 1쪽과 Crossref 모두 첫 저자가 Memiş다. INDEX.md 수정이나 파일명 변경은 하지 않았다(작업 범위 밖).

---

## 8. 지정 항목 대조표

| 지정 항목 | 확인 결과 | 위치 |
|---|---|---|
| Pilyugina et al. 2025 IEEE Access | doi:10.1109/ACCESS.2025.3573072, OA지만 자동 차단 | 3절. 2023 arXiv판 📚 |
| Wang G. et al. 2025 | Remote Sens. 17, 2006 | 📄 08 |
| O'Malley et al. 2026 arXiv:2605.16665 | 📚 기존. 표 S2 이번에 본문 확인 | 4절 |
| Ahajjam et al. 2025 JGR MLC | doi:10.1029/2025JH000969, OA지만 자동 차단. 검증 방식 [미확인] | 2절 |
| Tama et al. 2025 WACV | WACV 2026 게재, arXiv:2511.14473 | 📄 08 |
| Wei et al. 2026 ESSD Discussions | doi:10.5194/essd-2026-447 | 📄 08 |
| Liu Z. et al. 2024 | ERL 19, 014020 | 📄 08 |
| Zhang C. et al. 2024 | ERL 19, 044030, OA지만 자동 차단 | 2절 |
| Du Q. 2026 | Buildings 16, 3023 | 📄 08 |
| Li C. 2022 | JGR Atmos. 127, e2022JD036785, 구독 필요 | 6절 |
| Peng 2018 | J. Clim. 31, 251–266 | 📄 타 폴더 |
| Shen 2023 | STOTEN 859, 160381, 구독 필요 | 2절 |
| Kriuk 2025 | 📚 기존(arXiv). 학회판 M2GARSS 2026 추가 기록 | 2절 |
| Koven review 2025 | 📚 기존 파일이 실제로는 Memiş et al. 2025 | 7절 |

## 9. 중복과 정리 메모

- 같은 시간대의 다른 작업(`11_validation_and_methods`, `12_manuscript_refs`, `14_scirep_exemplars`)과 6편이 겹친다. 08 폴더 저장이 늦은 3편(Gay 2026, Garibaldi 2026, Ran 2022 CEE)은 08 사본을 지웠다. 08 저장이 먼저인 6편(Wang G. 2025, Du Q. 2026, Liu Z. 2024, Streletskiy 2026, Feng 2023, Tama 2025)은 양쪽에 남아 있으므로 INDEX 통합 때 한쪽으로 정리한다.
- Ran, Y. et al. Permafrost degradation increases risk and large future costs of infrastructure on the Third Pole. *Commun. Earth Environ.* 3, 238 (2022), doi:10.1038/s43247-022-00568-6은 `12_manuscript_refs/ran2022b_third_pole_infrastructure.pdf`에 있다(E의 ML 추정 선례, NOVELTY 2절).
- 기존 인벤토리의 `02_alt_dl_mapping/yin2024_alt_upscaling_airborne_gee.pdf`는 PDF 1쪽 기준 저자가 Merchant, M. A. & McBlane, L.이다. INDEX.md의 "et al. (IntechOpen chapter)" 표기와 파일명을 통합 때 고친다.
