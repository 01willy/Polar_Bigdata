# 보고서 그림 계획 (main.tex 10쪽판)

갱신: 2026-07-28. 감사 결과 반영. 공통 규칙: **그림 내부 제목·부제·주장형 문장 전부 제거**(캡션이 대신함),
300dpi PNG + 벡터 PDF 동시 출력, 냉색 규약(oslo_r/vik/acton/broc), em-dash 금지, `use_polar()` 스타일,
축소 게재 시 판독 가능한 폰트.

## 게재 목록 (절 → 그림, ✎=수정 재렌더 ★=신규)

| # | 절 | 파일 | 작업 |
|---|----|------|------|
| 1 | §2 데이터 | maps/data_inventory_world | ✎ 제목 2줄 제거, 빨강·주황 주석 → 청회색, 매트릭스 폰트 확대 |
| 2 | §2 데이터 | figures/s7_kpdc/s7_council_map | ✎ 제목 제거, 우측 패널 좌표 문자열 잘림 수정 |
| 3 | §3 방법 | figures/s0_schema/spatial_block_folds_map | ✎ 제목·em-dash 제거 |
| 4 | §4.1 | figures/s1_baseline/alaska_obs_vs_pred_maps | ✎ 상단 설명문 제거, 확대박스 주석 폰트 확대, 빨간 박스 → 진청 |
| 5 | §4.1 | figures/06_deep_learning/alt_info_bottleneck | ✎ 주장형 제목·부제·하단 각주 제거 |
| 6 | §4.2 | figures/02_evaluation/transfer_loro_summary | ★ 전 접근법 LORO 종합 가로막대(지구통계 IDW·OK·RK / 순수 ML CatBoost·MLP·FT-T / 물리 Stefan / 정교화 잔차·source-aware·mixture), Stefan 21.26 기준선, in-domain 병기. 근거 s11_comparison_table.csv + e1_kriging_results.csv + s1_baseline_results.csv |
| 7 | §4.2 | figures/s2_physics/physics_loro_bars | ✎ em-dash 제목 제거 |
| 8 | §4.2 | figures/e1_kriging/e1_variogram_regions | ★ CV 문맥별 적합 변동도(range·sill·nugget) 비교. 근거 e1_kriging_variograms.csv (AK 652km vs LORO 3233·7116·9763km) |
| 9 | §4.3 | figures/s3_aug/aug_response_curves | ✎ 제목 2줄 제거, 패널 (a)-(d) 라벨, y축 라벨 간결화 |
| 10 | §4.4 | figures/s11_uq/s11_calibration_curve | ✎ 제목 제거, raw 곡선 대비 상향 |
| 11 | §4.4 | figures/s11_uq/s11_uq_maps | ✎ 제목 제거, 점 크기 확대 |
| 12 | §4.4 | maps/alt_aoa_mask | ✎ 제목·em-dash 제거 재렌더 |
| 13 | §4.5 | figures/s10_shallow3d/s10_depth_slices | ✎ 3줄 제목 제거, 컬러바·눈금 폰트 확대 |
| 14 | §4.5 | figures/s10_shallow3d/s10_fence_sections | ✎ 4줄 제목 제거 |
| 15 | §4.5 | figures/s9_timelapse/alt_year_panels_v2 | ✎ 제목·하단 각주 제거, 마커 확대, dpi 300 |
| 16 | §4.5 | figures/e2_seasonal_dt/e2_dt_curves | ✎ 제목 제거, dpi 300, 주황 점선 → 규약색 |
| 17 | §4.5 | figures/e2_seasonal_dt/e2_rmse_bars | ✎ 제목 제거, dpi 300 |
| 18 | §4.6 | figures/s7_kpdc/s7_section_16layer_A | ✎ 제목-범례 겹침 수정, 제목 제거 |
| 19 | §4.6 | figures/s7_kpdc/s7_alt_comparison | ✎ 제목 제거, 범례 위치 조정 |
| 20 | §4/§2 | maps/local_demo_restyled | ✎ 제목 제거, 출처 각주 캡션 이동, 스케일바 검토 |
| 21 | §5 논의 | figures/11_physics_ml/03_Ex_byregion_boxplot | ✎ 300dpi 재렌더, 구어체 제목 제거. 근거 w3_physics_ml_Ex_diag.csv |

기존 게재 중 제외: e1_rmse_bars(#6 종합 그림으로 대체).

## 표 신설 (그림 아님, tex에서 조판)

- 물리식 5종 성능(in-domain·LORO): s2_physics_results.csv
- KPDC 관측방식별 ALT 스펙트럼 + 구간검열 집계: s7_kpdc_meta.json, s7_council_alt_derived.csv
- 다축 검증 요약: s11_multiaxis_validation.csv (기존 래스터 표 이미지 대체)
- 전이 표에 FT-Transformer 행 추가(22.5cm, "최상 신경망도 앵커 미달")

## 재렌더 주의

- e2_seasonal_dt.py·s7_kpdc_validation.py·aoa_transfer.py는 분석+그림 겸용 스크립트.
  그림만 재생성하는 경로(기존 결과 CSV 로드)를 확인하고, 없으면 figs-only 분기를 추가한다. 학습 재실행 금지.
- 데이터 로직 변경 금지. 그림 코드만 수정.
- GPU 불필요(전부 matplotlib·CPU).
