# XK·XL 실행 기록(2026-10-05)

등록: docs/EXPERIMENT_PLAN_FINAL_BATCH_ADDENDUM_XK_XL_2026-10-05.md. 두 묶음 모두 판정어가 없는 서술 분석이다. 로컬 CPU(CatBoost 4 스레드, nice 10).

## XK_support_scale

- 스크립트: scripts/2_evaluation/xk_support_scale_v1.py(블록 밖 예측과 집계), scripts/4_visualization/paper_v3/xk_support_scale_fig.py(그림).
- 예측: map_alaska_v1.nested_oof(0.5° 블록 5겹, 씨앗 키 (지역, r, split 0, n −1, draw 0), 겹 안 λ 블록 교차검증, E_n 재추정). 겹별 λ: 알래스카 0.5×5, 레나 0.5·0.25·0.5·0.5·1.0, 캐나다 1.0·1.0·0.25·1.0·1.0. 적합 시간 23.6 s(지역 합), 전체 36.8 s, 재집계 12.9 s.
- 제품 값: XG 추출 표의 value_static_cm(CCI v5 1997–2023, Wei 2000–2024, Yi·Kimball 2001–2015)와 x25 의 cci_alt(CCI v4). 라벨 셀을 모두 덮으므로 새로 표집하지 않았다.
- 격자: floor((좌표 + 0.05)/g), g = 0.05·0.1·0.25°(0.1° = ERA5-Land 셀). 라벨 셀 3개 미만 격자 제외. 블록 재표집 1,000회(셀 가중, 블록 등가중).
- 결과(셀 집합 models = P1·R1·CCI v4 유한 셀, 마스크 없음, 셀 가중 RMSE cm, 1 km / 0.05° / 0.1° / 0.25°)

| 지역 | P1 | R1 | P1 − R1 [95 % CI] |
|---|---|---|---|
| 알래스카(13,606셀) | 14.30 / 12.94 / 12.76 / 13.16 | 13.80 / 11.27 / 11.22 / 11.55 | 0.50 [0.18, 1.10] / 1.67 [0.60, 2.69] / 1.54 [0.50, 2.54] / 1.61 [0.54, 2.72] |
| 레나(2,958셀) | 21.02 / 24.66 / 26.38 / 15.85 | 21.01 / 24.97 / 26.75 / 15.07 | 0.01 / −0.30 / −0.38 / 0.78, 모두 CI 가 0 포함 |
| 캐나다(747셀) | 28.84 / 20.29 / 20.16 / 15.85 | 26.22 / 19.18 / 18.80 / 14.89 | 2.62 [0.00, 4.42] / 1.11 / 1.35 / 0.96 |

- 고정 해석 규칙의 적용: (1) '모든 방법의 RMSE 가 크기와 함께 준다'는 성립하지 않는다(CCI v4 는 알래스카 19.9 → 23.1 cm, 레나는 비단조). 그래서 그 문장을 쓰지 않는다. (2) R1 − P1 차이는 알래스카에서 0.05° 가 가장 크고(1.67 cm) 0.1°·0.25° 도 비슷하며(1.54, 1.61 cm) 1 km 가 가장 작다(0.50 cm). 0.05° 는 규칙의 두 갈래(0.1° 이상, 1 km)에 없으므로 이 사실만 쓴다. 캐나다는 1 km 에서 가장 크다(2.62 cm). 레나는 모든 크기에서 CI 가 0 을 포함한다.
- 제품 비교 문장은 '같은 격자 크기에서 RMSE 가 a cm 작았다/컸다' 형식만 쓴다(표: xk_support_scale_v1.csv, 열 d_rmse_vs_r1_*).
- 주의: 레나의 common 집합은 Wei 값이 있는 1,342셀뿐이다. 그래서 결과 열람 뒤 models 집합(P1·R1·CCI v4)을 더했다(구현 결정으로 기록). 0.05° 이상 격자는 단위 수(알래스카 56–105, 레나 22–56, 캐나다 19–37)와 블록 수(8–36)가 적어 CI 가 넓다. Wei 5 km 마스크는 알래스카 4,474셀을 뺀다.
- 산출: data/processed/xbatch/XK_support_scale/{xk_oof_cells_v1.csv, xk_support_scale_v1.csv, xk_meta.json}, 그림 outputs/figures/paper/v3_restructure/maps/XK_support_scale.{pdf,svg,png}, _legend.md, _source.csv, 덱 deck/assets/paper_report/maps/XK_support_scale_slide.png.

## XL_map_products

- 스크립트: scripts/2_evaluation/xl_map_products_v1.py(제품 표집, 요약 표), scripts/4_visualization/paper_v3/xl_map_products_fig.py(그림).
- 표집: XG 의 extract_cci·extract_tif·extract_nc_latlon 을 1 km 셀 중심에 적용(최근접 화소, 기간 평균, zip md5 재계산 생략). 시간: 레나 15 s, 알래스카 47 s. 유효 셀(알래스카 899,633 중): CCI v5 877,684, Wei 875,199, Aalto 868,973, Yi·Kimball 749,923.
- 요약(표시 셀 평균 cm, 우리와의 피어슨 r): 알래스카 우리 56.1, Stefan 57.5(0.96), CCI v5 105.5(0.46), Wei 70.8(−0.15), Aalto 78.8(−0.10), Yi·Kimball 105.2(0.34). 레나 우리 42.2, Stefan 41.1(0.36), CCI v5 41.2(−0.23), Wei 58.6(0.05), Aalto 55.2(−0.02).
- 표시 해상도 시연: 0.1° 셀 평균이 1 km 지도 분산의 97 %(알래스카), 61 %(레나)를 설명한다.
- 차이가 큰 지역(이름만, 원인 단정 없음): 알래스카 내륙(CCI v5, Yi·Kimball, 50 cm 이상), 약 61–63°N·141–150°W(Wei), 약 62–64°N·148°W 서쪽(Aalto 보다 우리가 깊음). 레나 북·동부 델타(CCI v5 보다 깊음), 대부분 영역(Wei, Aalto 보다 얕음).
- 주의: 차이 색 범위는 지침 규칙(|차| 99 백분위)으로 알래스카 ±215 cm 이다. 그래서 Stefan 차이(±10 cm)는 거의 흰색으로 보인다. 지도에는 정확도 주장을 붙이지 않았다.
- 산출: data/processed/xbatch/XL_map_products/{xl_alaska_products_v1.csv.gz, xl_lena_products_v1.csv.gz, xl_summary_v1.csv, xl_meta.json}, 그림 XL_Alaska_product_differences, XL_Lena_product_differences, XL_display_resolution(.pdf, .svg, .png, _legend.md), 덱 자산 *_slide.png.

## 그림 개정 2(2026-10-05 검토 반영)

- XL 제품 비교: 두 행(위: 잔차 ML·제품 ALT 한 색 범위(합동 2–98 백분위), 회색 글로 영역 평균과 r; 아래: 제품 − 잔차 ML 한 발산 범위(|차| 합동 98 백분위), Stefan − 잔차 ML 은 자체 ±10 cm). 부호는 개정 1 과 반대(제품 − 우리)다.
- XL 표시 해상도: 원 1 km 셀 메시(육각형 모자이크 제거), 같은 크기 정사각 패널 4개, 지역마다 색 막대 하나.
- XK: 같은 간격의 범주 x(1 km, 0.05°, 0.1°, 0.25°)와 눈금 아래 격자 수, 위 행 RMSE(로그, 공유), 아래 행 Stefan − 잔차 ML 차와 CI(셀 가중·블록 등가중), 0 선. Wei 마스크 판은 원천 표에만.
