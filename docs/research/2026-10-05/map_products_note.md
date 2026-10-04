# ALT 1 km 지도 산출 기록(알래스카, 레나 델타)

작성 2026-10-05. 지시: 보고 덱과 논문용 고해상 ALT 지도 2종. 판정 없는 서술 산출이다.

## 1. 해상도에 관한 정직성 문장(그림 설명문에도 같은 내용)

- 1 km 격자는 표시 해상도이다. 기후 8열과 √TDD 는 ERA5-Land 0.1° 격자(알래스카 약 4–6 × 11 km, 레나 약 3 × 11 km)에서, 토양 9열은 SoilGrids 약 5 km 창에서 온다. 1 km 단위로 변하는 입력은 지형 6열(Copernicus DEM 30 m 창)과 CCI ALT 뿐이다.
- 잔차 ML 보정은 주로 기후 격자 사이에서 작동한다(C8: WF9 에서 알래스카 지역 내 R1 이득은 ERA5 격자 사이 성분 −1.03 cm, 격자 안 성분 −0.01 cm).
- 지도에서 확인한 값: 0.1° 셀 평균이 1 km 지도 분산의 97 %(알래스카)와 61 %(레나)를 설명한다. 0.1° 셀 안 1 km 값의 표준편차는 1.8 cm(알래스카, 전체 9.5 cm), 1.1 cm(레나, 전체 1.8 cm)이다.

## 2. 알래스카(새 산출)

| 항목 | 값 |
|---|---|
| 영역 | 59–71.5°N, 168–141°W, Natural Earth 10m 육지, PFR ≥ 50 % |
| 셀 | 상자 1,739,845 · 육지 1,303,263 · 영역 899,633 · 표시 858,517(회색 41,116: 수체 13,256, CCI 결측 18,643, 토양 도일 결측 9,217) |
| DEM | 필요 타일 256(기존 144, 새로 받음 112), data/raw/dem +2.4 GB |
| SoilGrids | 기존 namerica 창 + 새 창 map_alaska_v1_0(168.3–166.0°W, 셀 5,325) |
| 방법 | WF6 R1(교차검증 λ). E0 1.5046(원천 3,860셀), E_ls 1.6167, E_n 1.6166, λ 0.5(겹 5개 모두 0.5) |
| 구간 | 상수 폭 분할 등각 ±20.15 cm(블록 5겹 교차검증 잔차, 겹 안 λ·E_n 재추정). 하한 0 절단 5,384셀 |
| 교차검증 RMSE | R1 13.80 cm, P1 14.30 cm(블록 5겹, 셀 가중) |
| 예측 요약(표시 셀) | R1 중앙값 55.7 cm(1–99 % 36.8–74.8), 보정량 중앙값 −1.4 cm(1–99 % −9.1–5.5) |
| 외삽 | 표시 셀의 73.2 % 가 공변량 1개 이상 범위 밖(범주 0/1/2/3+: 229,553/158,897/113,373/356,694). 주된 열 dem_rough, dem_slope, dem_tpi |
| 시간 | 격자 grid 6 s, tiles 411 s(1차)+4 s, dem 348 s, e5 203 s, cci 73 s, sg 26 s, qa 약 60 s, mask 46 s, merge 56 s. 예측 48 s |

## 3. 레나(기존 h49 산출 + 보조)

- 지도 a 는 h49 패널 (c)(R1 라벨 전량, λ 0.25, E_n 1.4346)이다. 보정량은 m_R1_all − m_P1_all 이다.
- 구간 패널은 h49 의 표지 없는 원천 Stefan(P0) 구간(q 0.707, 로그 비, 폭 ∝ 예측)이다. 지도 a 방법의 구간은 레나 라벨 블록 교차검증의 상수 폭 등각 ±25.39 cm(RMSE R1 20.77, P1 21.48 cm)이며 설명문에만 적었다.
- 외삽 패널의 기준 행은 최종 방법의 학습 행(원천 셀 ∪ 레나 라벨)이다. 표시 셀의 84.3 % 가 범위 밖 열을 가진다(주로 e5_tcold, e5_maat, e5_fdd). h49 의 원천 기준으로는 모든 표시 셀이 범주 3 이다.

## 4. 주의

- PFR 문턱: 지시의 'PFR ≥ 0.5' 를 50 %로 읽었다(파일 단위 %). 10 % 문턱이면 1,173,828셀, 0.5 % 문턱이면 1,229,208셀이다(build_map_grid_alaska_v1.py 의 PFR_MIN 한 줄).
- DEM QA: 알래스카 v3 13,606행 재추출에서 DEM 6열의 11.2–11.5 %(ABoVE_AK 1,525–1,562행)가 v3 값과 다르다(고도 차 중앙값 0.04 m, 최대 4.6 m). 다른 19열과 토양 도일은 허용 차 안이다. 그 행들의 v3 DEM 은 셀 중심이 아닌 좌표에서 계산된 것으로 보인다. 격자는 셀 중심 규칙을 쓴다.
- 수체 판정은 GSW 70N·80N 타일이 덮는 60°N 이북만이다(GSW 결측 179셀).
- 상수 폭 구간은 라벨과 비슷한 셀에 대한 평균 포함률이다. 외삽 셀에서는 보장되지 않는다.

## 5. 파일

- 스크립트: scripts/1_data_prep/build_map_grid_alaska_v1.py, scripts/2_evaluation/map_alaska_v1.py, scripts/2_evaluation/map_lena_final_extras_v1.py, scripts/4_visualization/paper_v3/maps_alt_v3.py
- 자료: data/processed/map_alaska/{alaska_grid_x25_v1.csv.gz, alaska_grid_x25_v1_meta.json, alaska_pred_v1.csv.gz, alaska_pred_v1_meta.json, alaska_cv_oof_v1.csv}, data/processed/map_lena/{lena_final_extras_v1.csv.gz, lena_final_extras_v1_meta.json, lena_cv_oof_v1.csv}
- 그림: outputs/figures/paper/v3_restructure/maps/{Alaska_ALT_map_v3, Lena_ALT_map_v3}.{pdf,svg,png} 와 _legend.md, deck/assets/paper_report/maps/*_slide.png
- 명세: figures/figure_spec_maps_v3.json
