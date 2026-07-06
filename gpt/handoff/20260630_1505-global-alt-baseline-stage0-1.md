# Handoff: 전 지구 ALT 예측 — 데이터 구축 + Stage 0/1 baseline
**Project**: Polar_Bigdata (영구동토 well 온도 → 3D ALT/열구조 예측)
**Date**: 2026-06-30 15:05
**Session focus**: 프로젝트를 처음부터 구축 — 목표 확정, 전 지구 데이터 무계정 취득, ALT 예측 평가골격(Stage 0) + 물리피처 baseline(Stage 1)까지.
**Author**: Claude Opus 4.8 + willy010313

---

## 1. TL;DR
- **확정 목표**: 전 지구 다지역 borehole 온도 + CALM 활성층 + 공변량 → 공간인식 DL로 **ALT(활성층 두께) 주 출력** + 얕은 3D 열구조 + 불확실성, **알래스카 학습 → 타 지대 전이(transfer) 검증**. (PINN은 선택, 순수 DL로 시작 가능.)
- **데이터(무계정)**: GTN-P 전 지구 borehole 온도 450 데이터셋/397 borehole + ALT 106; PANGAEA CALM **259 사이트/12국/3,604 site-year** ALT; WorldClim 기후·고도.
- **Stage 0 핵심**: 무작위 CV는 공간 자기상관으로 **4배 과대평가**(IDW RMSE 무작위 28→LORO 111cm). → 헤드라인 지표는 **공간블록 + LORO(지역전이)** 만 사용.
- **Stage 1**: 도일(degree-day) 물리피처 추가로 개선(LORO GBM 103.9→96.9cm). **GBM이 전이에 RF보다 안정**(RF 과적합).
- **블로커/갈림길**: 무계정 거친 공변량의 정확도 한계 도달(LORO ~97cm). 다음 큰 지렛대 = **계정 필요 데이터(InSAR/ERA5월별/MODIS)**.

## 2. Context
- 신규 프로젝트(이전 핸드오프 없음). 초기 아이디어: well 온도 + active layer로 3D 영구동토 모델(tri-mesh) 보간.
- 세션 중 범위를 "알래스카 파일럿(3D 온도장+0°C 메시)"에서 → **"전 지구 ALT 예측 + 전이 검증"** 으로 사용자 합의하에 재정의(아래 §5).

## 3. What we did
- **데이터 파이프라인 구축** (`src/polar/`): `acquire.py`(GTN-P list+bulk, combined=true 폴백), `covariates.py`(WorldClim/USGS/SoilGrids), `alt_dataset.py`(PANGAEA CALM 파싱), `geo.py`(EPSG:3413), `preprocess.py`, `interpolate.py`/`model.py`(알래스카 3D 크리깅·RF, 초기 단계 산출). scripts/01~06.
- **전 지구 GTN-P 다운로드**: PT 450 데이터셋(397 borehole; Russia·US·Svalbard·Canada·Sweden·Switzerland·Austria), ALT 106. 933MB. `data/raw/gtnp/`.
- **전 지구 ALT 데이터셋**: `data/processed/alt_global.csv` (3,604행, 14열: site/lat/lon/year/alt_cm + WorldClim 5).
- **Stage 0** (`src/polar/alt_model.py:run`): 4 CV(무작위/site-disjoint/공간블록500km/LORO) × baseline(지역평균/IDW/Kriging/GBM). 베리오그램 range(skgstat)로 블록크기. → `cv_leakage_table.csv`, `cv_splits.json`, `variogram_range.json`.
- **Stage 1** (`run_stage1`): 도일피처(TDD/FDD/√TDD Stefan/대륙성/적설 proxy; bio 사인곡선 근사 — 월별 WorldClim 서버 불통 대응) + GBM/RF nested 비교 + 변수중요도. → `stage1_results.csv`, `stage1_feature_importance.csv`, `alt_features.csv`.
- **시각화 14종** (`outputs/figures/`): 개념도·전지구 ALT 커버리지·CV 누설·Stage1 등. **참고문헌 6편 PDF** `references/`.
- 모두 CPU. GPU 미사용(공간 DL 단계부터 6–9번 예정).

## 4. Key numbers (this session)
| Method | CV / Domain | Metric | Value | Source |
|---|---|---|---|---|
| IDW | 무작위 K-fold | ALT RMSE | **28 cm** (누설 과대평가) | `data/processed/cv_leakage_table.csv` |
| IDW | LORO(전이) | ALT RMSE | 111 cm | 〃 |
| GBM(공변량5) | 공간블록 | ALT RMSE | 83.9 cm | `data/processed/stage1_results.csv` |
| GBM +도일피처 | 공간블록 | ALT RMSE | 77.6 cm | 〃 |
| RF +도일피처 | 공간블록 | ALT RMSE | 70.2 cm | 〃 |
| GBM +도일피처 | LORO(전이) | ALT RMSE | **96.9 cm** | 〃 |
| RF +도일피처 | LORO(전이) | ALT RMSE | 108.6 cm (과적합) | 〃 |
| 변수중요도 | RF 전체 | wc_elev | 0.67 (지배) | `stage1_feature_importance.csv` |
| 데이터 규모 | CALM 전지구 | 사이트/행 | 259 / 3,604 | `data/processed/alt_global.csv` |

ALT 분포(전지구) 5/50/95% = 34/64/427 cm → 절대 RMSE가 높은 이유(범위 광범위 + 거친 공변량).

## 5. Decisions made
- **주 출력 = ALT 지도**(+ 얕은 3D 열구조 보조), 깊은 base는 데이터 부족이라 물리 외삽+큰 불확실성으로만. — 깊은 borehole 소수(전지구 ≥50m ~36개).
- **다지역 전이(Alaska→타지대) 프레이밍** 채택(GIPL2/CCI 전면 능가가 아님). — 의의=관측기반 독립방법+전이+교정 UQ+메시. (기존중복: GIPL2·CCI·InterPIGNN·Ran2022.)
- **헤드라인 CV = 공간블록 + LORO**, 무작위 CV 금지. — 누설 4배 과대평가 입증됨.
- **전이엔 GBM > RF** (RF 과적합). 무계정 데이터로 진행, 계정 필요 데이터는 사용자가 추후.

## 6. Open questions / blockers
- **정확도 한계(LORO ~97cm)**: 무계정 WorldClim(18km climatology)이 거칠어 한계. 해결 증거 = InSAR/ERA5월별/MODIS 추가 시 LORO RMSE 감소 여부.
- **고도(elev) 중요도 0.67 지배**: 지역 대리값으로 과적합 우려 → region-invariant 피처/공간블록으로 완화 필요.
- **WorldClim 월별 서버(geodata.ucdavis.edu) 불통**: 진짜 도일은 ERA5 월별(CDS 계정) 권장.

## 7. Next steps (prioritized)
1. **[방향 결정]** A. 계정 데이터(NASA Earthdata·Copernicus CDS, 무료) 확보 → InSAR/ERA5/MODIS 연동 ★최대 향상 / B. 공간 DL(Stage 2~3, patch-CNN/neural field) / C. 지역 집중. — owner: user 결정 → Claude 실행.
2. **Stage 2** (`docs/PLAN_ALT_pipeline.md`): WorldClim 9×9 패치 + nodata 마스크 텐서화, patch-aggregated GBDT로 "공간 컨텍스트 가치" 검증. — Claude, 데이터 있음(즉시).
3. **Stage 4 전이 정량화**: LORO degradation factor + 외삽비율↔성능저하 상관(few-shot 곡선). — 주 novelty.
4. **인프라**: `git init` + `.claude/project.yaml` 설정(자동 cleanup/handoff 적용), 디스크 모니터(/home 98%).

## 8. Pointers
- 계획: `docs/PLAN.md`(§0 확정목표), `docs/PLAN_ALT_pipeline.md`(Stage 0~7 상세), `docs/GLOSSARY.md`(용어), `docs/SESSION_LOG.md`.
- 핵심 데이터: `data/processed/alt_global.csv`(전지구 ALT+공변량), `alt_features.csv`(도일피처).
- 핵심 코드: `src/polar/alt_model.py`(Stage 0/1), `covariates.py`(공변량), `alt_dataset.py`(CALM).
- 결과: `data/processed/{cv_leakage_table,stage1_results,stage1_feature_importance}.csv`; `outputs/figures/` 14종(특히 11_cv_leakage, 12_stage1_features, 10_global_alt_coverage).
- 참고문헌: `references/`(Ran2022·GIPL2·Gautam2025·InterPIGNN·Groenke·GTN-P PDF).
- Active jobs: none. Ckpt: none(아직 DL 학습 전).

## 9. Caveats for GPT
- **GIPL2/CCI를 "전면 능가"하는 게 목표가 아님** — 관측기반 보간 + 전이 + UQ가 차별점.
- **무작위 CV 수치(예 28cm) 인용 금지** — 누설로 과대평가. 항상 공간블록/LORO 수치 사용.
- **도일(TDD/FDD)은 현재 bio 사인곡선 근사값**(월별 실측 아님). ERA5 확보 시 정밀화 예정.
- 절대 RMSE가 높은 건(~70~110cm) 전지구 ALT 범위(34~427cm)+거친 공변량 탓 — 모델 결함이 아니라 데이터 한계.
- **깊은 3D base mesh는 보류**(데이터 부족). 현재 타깃은 ALT + 얕은(0~20m) 구조.
- git 미초기화 — 이 핸드오프는 커밋 안 됨(저장소 없음).
