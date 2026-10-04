# 14 · Scientific Reports 예시 논문 (원고 형식 학습용)

작성일: 2026-10-04. 분석 문서: `docs/research/2026-10-04/scirep_manuscript_exemplars.md`.

선정 기준: 2022–2026년 Scientific Reports 게재, 지구과학·원격탐사·환경 ML 분야, 영구동토·빙권·토양·수문·공간 검증 주제. 모두 오픈액세스 원문이다(출판사 PDF, OpenAlex `best_oa_location` 확인). 섀도 라이브러리는 쓰지 않았다.

PDF 는 저작권상 git 에서 제외된다(`references/**/*.pdf`). 이 README 만 추적 대상이다.

## 이 폴더의 파일 (8편)

| 파일 | 서지 | DOI | 라이선스 | 비고 |
|---|---|---|---|---|
| `jones2024_postfire_permafrost_stabilization.pdf` | Jones, B. M. et al. Post-fire stabilization of thaw-affected permafrost terrain in northern Alaska. Sci. Rep. 14, 8499 (2024) | 10.1038/s41598-024-58998-5 | CC BY | 영구동토. 초록·결과 문장의 수치 밀도가 높다 |
| `hall2026_rts_expansion_alaska.pdf` | Hall, E. C., Chipman, M. L. & Lara, M. J. Predicting retrogressive thaw slump expansion across northern Alaska. Sci. Rep. (2026) | 10.1038/s41598-026-70171-8 | CC BY-NC-ND | Article in Press 판(최종 편집본 아님). 권·논문 번호 [미확인]. 한계 문단과 누설 방지 서술의 예 |
| `gay2026_zero_curtain_ai_eo.pdf` | Gay, B. A. et al. Resolving circumarctic zero-curtain phenomena with AI-integrated earth observations. Sci. Rep. 16, 28715 (2026) | 10.1038/s41598-026-61719-9 | CC BY-NC-ND | 영구동토 + 물리 정보 딥러닝. 문체는 피할 예(과장어 빈도 최고) |
| `isaksen2022_barents_warming.pdf` | Isaksen, K. et al. Exceptional warming over the Barents area. Sci. Rep. 12, 9371 (2022) | 10.1038/s41598-022-13568-5 | CC BY | 빙권 기후. 서론의 질문 목록과 고찰의 질문별 구성 |
| `hatami2022_temperature_snow_freeze_thaw.pdf` | Hatami, S. & Nazemi, A. Compound changes in temperature and snow depth lead to asymmetric and nonlinear responses in landscape freeze–thaw. Sci. Rep. 12, 2196 (2022) | 10.1038/s41598-022-06320-6 | CC BY | 동결·융해. 주장형 제목의 예 |
| `feeney2022_soil_map_comparison_soc.pdf` | Feeney, C. J. et al. Multiple soil map comparison highlights challenges for predicting topsoil organic carbon concentration at national scale. Sci. Rep. 12, 1379 (2022) | 10.1038/s41598-022-05476-5 | CC BY | 토양 지도 비교·검증. 범위 한정 문장의 예 |
| `khanal2023_soc_stocks_nepal.pdf` | Khanal, S., Nolan, R. H., Medlyn, B. E. & Boer, M. M. Mapping soil organic carbon stocks in Nepal's forests. Sci. Rep. 13, 8090 (2023) | 10.1038/s41598-023-34247-z | CC BY | 공간 CV 대 무작위 CV, AOA, 전지구 제품 비교 |
| `fisher2023_open_water_evaporation.pdf` | Fisher, J. B. et al. Remotely sensed terrestrial open water evaporation. Sci. Rep. 13, 8174 (2023) | 10.1038/s41598-023-34921-2 | CC BY | 수문. 과정 모델 대 ML 11종 벤치마크(물리 대 ML 서술의 직접 참고) |

## 이미 다른 폴더에 있는 예시 논문 (중복 다운로드하지 않음)

| 파일 | 서지 | DOI |
|---|---|---|
| `../00_core10/gautam2025_alaska_alt.pdf` (`../01_benchmark/` 에도 있음) | Gautam, S., Mishra, U., Scott, S. N. & Lara, M. J. Machine learning and process-based modeling of spatiotemporal changes in active layer thickness across Alaska. Sci. Rep. 15, 42420 (2025) | 10.1038/s41598-025-26586-w |
| `../05_uq_transfer/singh2024_conformal_eo.pdf` | Singh, G. et al. Uncertainty quantification for probabilistic machine learning in earth observation using conformal prediction. Sci. Rep. 14, 16166 (2024) | 10.1038/s41598-024-65954-w |

## 읽는 순서 제안

1. Jones 2024: 초록과 결과의 수치 서술.
2. Hall 2026: 방법의 누설 방지·외삽 처리, 고찰 끝의 한계 문단.
3. Fisher 2023: 과정 모델과 ML 벤치마크의 논리, 오차 원천 열거.
4. Khanal 2023: 공간 CV·AOA·외부 제품 비교의 보고 방식.
5. Isaksen 2022: 서론의 목적·질문 목록, 질문별 고찰.
6. Gautam 2025: 가장 가까운 선행 연구. 무작위 70/30 분할과 전체 자료 조율(저자 스스로 누설 가능성을 적음)을 확인한다.
7. Gay 2026: 피해야 할 문체의 예로 읽는다.
