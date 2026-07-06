# 영구동토 ALT + 얕은 3D 열구조 예측 — 통합 실행 계획

> 3개 독립 설계안(MVP-GBM / 공간인식-DL / Rigor-평가)을 우리 실제 파일에 맞춰 합성한 세부 plan.
> 상위 목표는 [PLAN.md §0](PLAN.md) 참조. (2026-06-29 수립)

**검증된 사실(설계 반영):**
- `alt_global.csv` 모델 5열(`wc_bio1,wc_bio4,wc_bio7,wc_bio12,wc_elev`)·`alt_cm` NaN 0개. (NaN은 `calm_id`/`area` 메타뿐 → 지역키는 `country` 사용, `area` 금지.)
- 공변량 이동 확인: non-AK의 25%가 AK `wc_bio1`/`wc_elev` 범위 밖, ALT 17% 외삽.
- `xgboost` 미설치(→ sklearn `HistGradientBoostingRegressor` fallback), `skgstat`·`pykrige`·`torch+CUDA` 설치됨.
- WorldClim 래스터는 이미 보유(`data/raw/covariates/worldclim/`) → 패치 추출 가능.

**합성 원칙:**
1. **평가가 모델보다 먼저** — 누설-안전 CV 골격을 Stage 0에서 고정.
2. **GBM이 의무 baseline** — 공간 DL은 공간블록+LORO 두 CV에서 부트스트랩 CI 비중첩으로 GBM을 이긴 뒤에만 채택.
3. **주 novelty = 전이(LORO) + 교정된 셀별 불확실성 + 얕은 3D 구조** ("CCI/GIPL2 전면 능가"가 아님).
4. ALT=전 지구 주 출력, 열구조(0~20m)=알래스카 35-borehole 보조 스코프.

---

## (1) 단계별 로드맵

### Stage 0 — 누설 진단 & 평가 골격 고정 (즉시)
- **목표:** 무작위 CV가 부정직하고 LORO가 진짜 전이 측정임을 한 표로 입증 + baseline 바닥선 고정.
- **입력:** `data/processed/alt_global.csv`. 입력 5열 + `site`(그룹)·`country`(지역)·`lat,lon`·타깃 `log1p(alt_cm)`.
- **모델:** GBM(sklearn HistGBM) + baseline 사다리: 지역평균 / IDW / Ordinary Kriging(pykrige) / GBM.
- **CV 4분할:** 무작위 K-fold → leave-one-site-out → 공간블록(≥variogram range) → leave-one-region-out(country). RMSE 단조 증가 = 누설 증거.
- **블록 크기:** `skgstat`로 ALT 잔차 베리오그램 range 추정.
- **산출물:** `src/polar/alt_model.py`, `alt_model_table.csv`, `cv_leakage_table.csv`, `variogram_range.json`, `cv_splits.json`.

### Stage 1 — 강 baseline 확정
- 피처 엔지니어링(bio만으로): Stefan thaw/freeze degree-day proxy, 대륙성 `bio7/bio4`, 적설 proxy.
- 모델: point-wise RF(Ran2022 복제) / HistGBM·(설치 시)LightGBM·XGBoost, nested CV(outer=공간블록·LORO).
- 산출물: `alt_baselines.py`, `baseline_cv_results.csv`, `outputs/figs/baseline_skill_by_region.png`.

### Stage 2 — 공간 컨텍스트 입력 준비
- 9×9 WorldClim 패치(`tifffile`) + `-32768` nodata 마스크 채널 + EPSG:3413 좌표(`geo.py`).
- 게이트 검증: patch-aggregated GBDT가 단일픽셀 GBDT를 이기는지 → "공간 컨텍스트 가치" vs "DL 가치" 분리.
- 산출물: `alt_patches.npz`, `alt_features.parquet`.

### Stage 3 — 공간인식 DL (분포 내)
- 경량(<수십만 파라미터) patch-CNN(64d) + Fourier 좌표 + 물리 MLP → fusion → heteroscedastic ALT 헤드(μ,logσ²) on log1p.
- Ablation: coord-MLP → +패치 → +Fourier → full. 유의성=블록 paired 부트스트랩.
- 산출물: `nn/field_model.py`, `12_train_field.py`, `alt_field_cv.csv`, gap-free 전지구 ALT 맵.

### Stage 4 — 전이 검증 [주 novelty]
- `country` LORO. Alaska→Switzerland/Mongolia 산악=외삽 극단.
- 산출물: `loro_cv_scores.csv`(degradation = LORO/블록 RMSE), `transfer_degradation_vs_covshift.png`, few-shot 곡선(N=0,1,3,5,10).

### Stage 5 — 불확실성 정량화 & 교정 [공동 novelty]
- deep ensemble(5–10) + heteroscedastic / GBM quantile·NGBoost / split+Mondrian(region) conformal.
- 지표: PICP(90%), MPIW, CRPS, reliability, region conditional coverage. 전이서 불확실성이 정직히 넓어지는가.
- 산출물: `alt_mean.tif`, `alt_uncertainty.tif`, `calibration_reliability.png`, `per_region_coverage.csv`.

### Stage 6 — 얕은 3D 열구조 + 멀티태스크 (Alaska 보조)
- 공유 trunk + 2헤드(ALT, thermal), joint loss + ALT↔0°C crossing 물리 일관성. single/multi/pretrain ablation.
- 평가: leave-one-borehole-out, 기존 RF **1.53°C** 능가 목표.
- 산출물: `nn/multitask.py`, `outputs/meshes/shallow_thermal_volume.vti`.

### Stage 7 — 외부 벤치마킹 & 포지셔닝
- CCI ALT 동일 held-out 비교(CEDA 계정 후), 문헌(Ran2022/Gautam2025) 환산·한계 표, 3축 기여 그림, model card.

---

## (2) 평가 설계

| CV | 목적 | 단위 |
|---|---|---|
| 무작위 K-fold | 누설 진단(과대평가 폭로) | row |
| leave-one-site-out | 누설 진단 | site |
| **공간블록** | 분포 내 일반화 | ≥range km 블록(EPSG:3413) |
| **LORO** | **전이=주 novelty** | country |
| leave-future-years | 시간 일반화 | site 초기→후기 연도 |
| leave-one-borehole-out | 열구조 헤드 | borehole |

- 헤드라인=공간블록+LORO(무작위 아님). nested CV로 하이퍼파라미터.
- 지표: RMSE/MAE/bias(cm, expm1 후) region별; degradation=LORO/블록; UQ=PICP/MPIW/CRPS/reliability; 유의성=블록 paired 부트스트랩.
- 벤치마크 사다리: 지역평균 < IDW < Kriging < GBM < 공간DL. 외부: CCI/GIPL2/문헌.

---

## (3) 위험과 대응 (요약)
- **공간 자기상관 누설**(CALM 군집) → 헤드라인 공간블록+LORO, 누설표 공개.
- **지역 공변량 외삽**(non-AK 25% 범위 밖) → LORO 정직 측정, 외삽↔저하 상관, few-shot, UQ 확대.
- **희소 정답(259) DL 과적합** → GBM 의무 baseline, 용량제한·증강, 못 이기면 GBM 채택.
- **시간 자기상관**(평균 13.9년/site) → GroupKFold by site.
- **열구조 Alaska-only**(35 borehole) → 보조 스코프, 전지구 3D 미주장.
- **xgboost 미설치** → HistGBM fallback. **GPU**: 6–9번, 타 사용자 점유 시 보류.

---

## (4) 지금 당장: Stage 0 첫 단계
`src/polar/alt_model.py` 생성:
1. `alt_global.csv` 로드 → 5열 NaN-free 단언, `y=log1p(alt_cm)`, `site`/`country` 키 → `alt_model_table.csv`.
2. baseline 사다리(지역평균/IDW/Kriging/GBM)를 4 CV(무작위/LOSO/공간블록/LORO)로 평가 → `cv_leakage_table.csv`.
3. `skgstat`로 잔차 베리오그램 range → `variogram_range.json` + `cv_splits.json`.

재사용: `model.py:lobo_cv` 패턴, `geo.py:to_xy`(공간블록), `covariates.py`(WorldClim).
