# P0·P1 실험 결과 (2026-07-14)

계획: [EXPERIMENT_PLAN_2026-07-14.md](EXPERIMENT_PLAN_2026-07-14.md). 모든 수치는 `data/processed/*.csv` 근거.

## P0. 데이터 인벤토리 + 6모델 예측·오차 지도

- 인벤토리 세계지도(지역×데이터 매트릭스): `outputs/maps/data_inventory_world.{png,pdf}`,
  생성 `scripts/4_visualization/map_data_inventory_world.py`.
- 6모델(+앙상블) 예측·오차 지도: `outputs/maps/tournament_pred_maps.{png,pdf}`,
  `tournament_error_maps.{png,pdf}`, 생성 `scripts/4_visualization/map_tournament_error_maps.py`.
- 위치 동등가중 재채점(N=14,348 위치, 점→위치 groupby, σ(관측)=18.5cm), 기존 점단위와 순위 동일:

| 모델 | RMSE(cm) | skill |
|---|---|---|
| GBM | 16.1 | +0.13 |
| Diffusion | 16.2 | +0.13 |
| 앙상블(GBM+FT-T) | 16.2 | +0.12 |
| Flow | 17.3 | +0.07 |
| MLP | 17.7 | +0.05 |
| FT-Transformer | 17.7 | +0.04 |
| TabM | 20.5 | −0.10 |

## P1-a/b. 다지역 통합 셀 데이터셋 v2

`data/processed/dl_dataset_cell_v2.csv` 17,423행 × 42열(기존 14,348행 값 불변, QC 컬럼 5종 보존).
메타 `dl_dataset_cell_v2_meta.json`(known_issues 3건 문서화).

- **Lena_RU 3,037셀**(ALLena PANGAEA 973813, 8-9월 최대 융해깊이, QF2 제외): 중앙 37.3cm, IQR 30-48.
- **GTNPenv 37셀**(GTN-P 시추공 t_max 0°C 교차 유도, 내부 0.01° 클러스터 병합·기존 중복 제거): RU 15·SJ 10·US 9·CH 2·AQ 1, 중앙 145cm.
- **QTP_CN 1셀**(TPDC Wudaoliang, 2014=280.0·2019=316.2cm, censored 연도 제외, censor_flag=1): 298.1cm.
- 공변량: DEM 신규 31타일 전량 확보(지형 6종 100%), ERA5 8종(레나 삼각주 수역 마스크는 0.5° 폴백), CCI 2종.
  InSAR/PolSAR는 알래스카 한정이라 신규 행 NaN 유지 + `insar_miss` 신설(polsar_valid=0).
- 교차검증: e5_maat 레나 −10.8°C·QTP −5.6°C, dem_elev 레나 26m·QTP 4,613m, cci_alt 레나 33cm·QTP 77cm(라벨과 정합).

## P1-c. 전 공변량(25)·전 지역 6모델 재비교

하네스 `scripts/3_deep_learning/unified_tournament_cell.py` (셀 단위, log1p 타깃, 공간블록 0.5° 6-fold + LORO(test≥100),
torch·GBM 모두 fold별 train 중앙값 대체 + 결측 플래그 입력, GPU 6, 272초).
산출: `unified_tournament_{results,perregion,predictions}.csv`, `unified_feature_comparison.csv`, `unified_vs_alaska_results.csv`.

### 총괄 (pooled)

| CV | 1위 | 2위 | GBM | 비고 |
|---|---|---|---|---|
| 공간블록 | GBM 19.33(skill 4.5%) | 앙상블 20.09 | 19.33 | 통합셋은 보간+전이 혼합이라 v1 알래스카 17.2와 직접 비교 불가 |
| LORO | 앙상블 19.19(0.2%) | FT-T 19.37 | 20.82 | 전이에서 DL(FTT)이 GBM 우세 |

### LORO 지역별 (핵심)

| test 지역 | n | 최선 모델 | RMSE(cm) | skill | GBM |
|---|---|---|---|---|---|
| ABoVE_AK | 13,542 | 앙상블 15.03 / FT-T 15.12 | 15.0 | +13.5% | 17.57(−1.1%) |
| ABoVE_CA | 726 | TabM 27.10 / MLP 27.16 | 27.1 | +10.2% | 29.40(+2.6%) |
| Lena_RU | 3,037 | Diffusion 25.25 | 25.2 | −19.7% | 29.76(−41.1%) |

- 알래스카 전이(train 비알래스카 3.8k셀)에서 FT-T/앙상블이 GBM 대비 2.5cm 우세. 소표본·이질 train에서 DL 이점 첫 확인.
- 레나 전이는 전 모델 skill 음수(bias +12~+22cm, 알래스카의 깊은 ALT로 과대예측). 전이 병목 지속.

### 결측 라우팅 아티팩트 (진단, 레나 전이 GBM)

| train | 결측 입력 | RMSE(cm) |
|---|---|---|
| 알래스카특화 | NaN 네이티브 | 21.37 |
| 알래스카특화 | 중앙값 대체 | 22.11 |
| 통합(레나 제외) | 중앙값 대체 | 29.76 |
| 통합(레나 제외) | NaN 네이티브 | **50.56** |

NaN 네이티브 GBM은 train의 NaN-InSAR 표본이 GTNPenv(고ALT) 37셀뿐이라 "InSAR 결측=깊은 ALT" 라우팅을 학습,
레나(전면 결측)를 고ALT로 오예측. 다지역 통합에서는 중앙값 대체+플래그가 필수. 단 대체 방식은 알래스카
in-domain을 오염(아래) → 모달리티 드롭아웃 등 후속 필요.

### 통합 vs 알래스카특화 (GBM FULL, 중앙값 대체)

| test | 특화 | 통합 | 판정 |
|---|---|---|---|
| Alaska 공간블록 | 18.08(+2.4%) | 20.93(−12.9%) | 특화 우세(레나 3천 셀의 대체 InSAR 값이 매니폴드 오염) |
| Lena_RU 전이 | 22.11 | 29.76 | 특화 우세(GTNPenv 결측 패턴 간섭) |

주의: NaN 네이티브 입력의 예비 실행(수정 전)에서는 Alaska 18.71 vs 17.72로 통합이 우세했다.
**통합 학습의 득실은 결측 모달리티 처리 방식에 따라 뒤집힌다** · P1 확정 결론은 처리 방식 명시 필수.

### 공변량 기여 (GBM, 통합셋)

| CV | M3 기후+지형(14) | M4 +InSAR(20) | FULL(25) |
|---|---|---|---|
| 공간블록 | 25.49 | 19.19 | 19.33 |
| LORO | 22.44 | 20.91 | 20.82 |

## 판정·후속

1. P1 게이트("개선 시 채택"): 다지역 통합 학습은 **현 결측 처리에서는 미채택**(알래스카 성능 저하).
   단 (a) LORO 전이에서 DL 우세, (b) 결측 아티팩트 정량화, (c) 레나 스케일 전이 실측(25-30cm)은 채택.
2. 후속: 모달리티 드롭아웃/마스크 임베딩(결측을 학습 신호로), CCI prior 단독 대비 레나 벤치마크,
   P2(Stefan 물리 base)로 전이 bias(+12~22cm) 교정.
3. 지도·차트: `charts_unified_tournament.py`(예정) → 덱 반영.
