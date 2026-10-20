# 그림 변수 통제 감사 (원고 v13, 2026-10-20)

대상: `paper/manuscript/en_v13` 의 Fig. 1–7 과 Supplementary Fig. S1–S10. 기준: 비교 대상 패널은 같은 설정을 보여야 하고, 설정이 다르면 그 차이가 의도된 것이고 캡션에 적혀 있어야 한다.

근거 자료
- 그림 스크립트: `scripts/4_visualization/paper_v13/fig1_data.py` … `fig7_maps.py`, `si_s01_stefan.py` … `si_s10_site_series.py`, 공용 모듈 `c13_common.py`, `c13_maps.py`, `fig34_common13.py`, `si13_common.py`
- 패널 원천 자료: `outputs/figures/paper/v13/source_data/Fig*.json`, `outputs/figures/paper/v13_si/SI_figures.json`
- 캡션: `sections/methods.tex` (Fig. 1, 2), `sections/results.tex` (Fig. 3–7), `si.tex` (S1–S10)
- 라벨·제품 표: `data/processed/v5/labels_cell_year_v5.csv` (in_forcing == 1), `products_cell_year_v5.csv`, 실행 폴더 `data/processed/v5/runs/*`, 지도 `data/processed/v5/maps/*`

주 설정(main)의 정의: 기후 입력 year_obs(라벨 연도의 TDD 를 관측일까지 누적), 공변량 F8TS(기후 8 + 지형 6 + 토양 9 = 23), CatBoost, 잔차 가중 λ = 0.5, 씨앗 3개(0, 1, 2) 평균, 0.5° 공간 블록 5겹, 실행 폴더 `runs/core_v6`.

행 번호는 2026-10-20 12:50 시점 파일 기준이다(`fig4_labels.py`, `fig5_products.py`, `fig7_maps.py`, `results.tex` 는 감사 중 갱신된 판을 다시 읽었다).

---

## 1. 패널별 설정 표

약어: AK = Alaska, LE = Lena Delta, CA = Canada(NW). 라벨 연도 '전량' = AK 2010–2024 (15,540), LE 2010–2022 (2,436), CA 2010–2024 (1,196). 기후 입력 year_obs = 관측일까지 누적 TDD, year = 라벨 연도 전체 계절 TDD, clim = 2011–2024 평균. 'OOF' = 블록 교차검증 out-of-fold 예측.

| 패널 | 지역 | 라벨 연도 | 기후 입력 | 검증 설계 | 공변량 | 학습기·λ | 씨앗 | 부분집합 규칙 | 지도 연도·기간 | 색 범위·색지도 / 축 범위 | 표시 보간 | 자료 원천(run) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1a | AK·LE·CA | 전량 | – | – | – | – | – | in_forcing == 1, loc_id 고유 수(13,253 / 2,206 / 730) | 영구동토 확률 1997–2021 평균 | oslo_r 0–0.30 구간, 0–100 % | bilinear 1,500 px | labels_cell_year_v5, cci_pfr_mean_1997_2021.nc |
| 1b | AK·LE·CA | 2010–2024 | – | – | – | – | – | in_forcing == 1 | – | y 축 지역별 독립(0–8,000 / 0–800 / 0–500) | – | labels_cell_year_v5 |
| 1c | 3지역·입력·제품 6종 | 축 1997–2025 | ERA5-Land 월별 2010–2024 | – | – | – | – | '사용 연도' = 제품값이 있는 라벨 셀-연도의 연도 범위(Wei 2010–2024, Liu 2010–2020, CCI v4 2010–2021, CCI v5 2010–2023, YK 2010–2015) | – | – | – | products_cell_year_v5(_meta), forcing_meta.json |
| 2a | LE(곡선: 셀 14813, 2013년, 관측일 8/17), AK(Stefan anchor, 15,540) | 전량 | 곡선에 year_obs·year·clim 세 변형 모두 표시 | – | – | E 원점 최소제곱 | – | 곡선 셀: 가장 춥고 더운 해 모두 관측·라벨 연도 최다·TDD_obs/전체 비 최소 | – | y 범위 고정(JSON) | – | forcing_year.csv, maps/Alaska/meta.json |
| 2b | AK | 훈련 전량 | year(2014 전체 계절) | 없음(전량 적합 지도 모형) | F8TS(썸네일은 7/23 변수) | CatBoost λ 0.5 | 0,1,2 | 각 입력 썸네일 2–98 % 개별 스트레치 | 2014 | ALT oslo_r 35–85 cm, 잔차 vik ±25 cm | 0.05°×0.02° 상자 평균 | maps/Alaska/pred_2014, meta.json |
| 2c | LE(폴드), AK(아노말리) | LE 2010–2022; year·block×year 는 2012–2021만 채점(≥20 셀-연도, 17행 제외) | year_obs | random·block·year·block×year·region | – | – | – | make_folds(k 5, seed 0), core_v6 preds 와 대조 | 아노말리 2012, 2019 | vik ±15 cm | 상자 평균 | testbed_v5, maps/Alaska |
| 3a | AK·LE·CA | 전량 | clim vs year_obs | block | F8TS | P1, R1 λ 0.5, D0 | 3 | 전량 | – | y 폭 8 cm 동일, 오프셋 지역별 | – | core_v6 summary_metrics, deltas_forcing |
| 3b | AK·LE·CA | random·block·region 전량; year·block×year 는 ≥20 셀-연도 연도만(LE 2012–2021: 2,419 / 2,374; CA 2012–2024: 1,164 / 1,135; AK block×year 15,421) | year_obs | 5 설계 | F8TS | P0, P1, R1 λ 0.5, D0 | 3 | 설계별 부분집합 | – | y 지역별 자동; 비지역 4 설계 최대의 1.3배 초과값은 상단에 표기(LE Direct ML region 39.3) | – | core_v6 summary_metrics |
| 3c | AK·LE·CA | 전량 | year_obs | block | F8, F8T, F8TS, F8TS_M, F8TS_X, F8TS_ALL | R1 λ 0.5 − P1 | 3 | 블록 부트스트랩 2,000 | – | x 공통 | – | featsets_v6(F8, F8T, F8TS_X) + modis_v6(F8TS, F8TS_M, F8TS_ALL) |
| 3d | AK·LE·CA | 전량 | year_obs(단, 3분위 공변량 'TDD' 는 전체 계절 TDD) | block OOF | F8TS | R1 λ 0.5 − P1 | 3 | 3분위별 ≥30 셀-연도 · ≥5 블록, 2,000 재표집 | – | vik ±7 cm 공통 | – | outputs/figures/paper/v7/source_data/F4e_gain_conditions.csv ← core_v6 preds |
| 4a | AK·LE·CA | 전량 풀 | year_obs | block, 훈련 폴드를 n 으로 서브샘플 | F8TS | R1 λ 0.5, D0 − P1(n) | 3(= 추출 3회, 씨앗마다 추출) | n: AK 25–2,500·all, LE 25–1,000·all, CA 25–500·all(CA 1,000 은 n_test < 0.95·all 규칙으로 탈락); 1,000 재표집 | – | y 지역별 자동 | – | ladder_v6(nMsgU) |
| 4b | AK·LE·CA | 원천 2지역 전량 + 표적 n | year_obs | region hold-out(표적 블록 폴드) | F8TS | P0, P1n, D0S, R1S λ 0.5 | 3 | n 0–500; 1.3배 초과값 상단 표기(LE Direct ML n=0 39.5) | – | y 지역별 자동 | – | ladder_v6 |
| 4c | 3지역 + 외부 8 소지역 | 외부 2010–2024 | – | – | – | – | – | 외부 73 셀, 816 셀-연도 | – | – | – | v9_ext/labels_cell_year_v5 |
| 4d | 외부 7 소지역 풀(그린란드 제외) | 외부 2010–2020 | year_obs | 소지역 블록별 채점, n 0/5/10/25/50/all | F8TS | R1S λ 0.5, P1n, D0S + 제품 기준선 | 3(씨앗 평균) | Wei·Liu·CCI v5 값이 모두 있는 474 셀-연도 | – | y 12–66 고정 | – | extra_v9 preds_EXTFT(dfuMW) |
| 4e | 외부 6 소지역 | 2010–2020 | year_obs | 동일 | F8TS | R1S 0·5 라벨, CCI v5 | 3 | FT_SUBS 고정 6 소지역(≥30 셀-연도; 캄차카 16·그린란드 제외) | – | x log 8–220 | – | extra_v9 |
| 5a | AK·LE·CA | AK 2010–2020, LE 2013–2020, CA 2010–2019 | year_obs | block OOF | F8TS | P1, R1 λ 0.5, D0 + 제품 5 | 3 | Wei·Liu·CCI v4 공통 셀-연도(14,326 / 461 / 532); Wei 훈련지 25 km 밖(4,229 / 449 / 489); CCI v5·Aalto 는 값 있는 곳 | – | x 10–45 + 압축 구간 45–130 | – | core_v6 common_cells_eval.csv |
| 5b | LE | 2013–2020 | year_obs | block OOF | F8TS | R1 λ 0.5; Wei; Liu | 3 | 5a 공통 461 | – | 0–120 고정 | – | core_v6 preds + load_v5 |
| 5c | AK·LE | 5a 공통 | year_obs | block OOF | F8TS | R1 + 제품 4 | 3 | 관측 ALT 4 계급; 계급 < 10 셀-연도 미표시(MIN_BIN); CCI v5 는 값 있는 곳; 이상값 1개 가장자리 표기 | – | y 지역별 자동 | – | 동일 |
| 5d | AK·LE·CA | AK 2010–2022, LE 2013–2017·2020, CA 2013–2024 | year_obs | block OOF | F8TS | R1, P1, Wei, Liu | 3 | 연도별로 R1·P1·Wei 값 있는 셀-연도(CCI v4 불요); Liu 는 ≥30 남을 때만; 연도 ≥30; 거리 마스크 없음 | – | y 0–50 / 0–90 / 0–60; n 띠 log 공통 | – | outputs/figures/paper/v7/source_data/F5_yearwise_rmse.csv ← core_v6 preds |
| 6a–c | AK | 훈련 전량 | year(2014 전체 계절) | 없음(지도 모형) | F8TS | R1 λ 0.5 | 3 | AOA 밖 페이드(우리 지도만) | 2014 | oslo_r 35–85 cm(우리 지도 1–99 %, 5 cm 단계); Wei 13.6 %, Liu 20.5 % 화소가 상한 초과 | 우리 지도 1 km → 250 m, 제품 bilinear 재투영 | maps/Alaska/pred_2014; raw wei/liu ALT_2014.tif |
| 6d | AK | 2014 | year_obs(OOF) | block OOF | F8TS | R1 λ 0.5 | 3 | 0.009° 셀 집계: 관측 101 셀 → 공통 95(우리 지도·Wei·Liu 값); 25 km 마스크 없음(85/95 가 ≤25 km) | 2014 | y −55…45 cm, 초과값 표기 | – | core_v6 preds, products_cell_year_v5 |
| 6e–g | LE | 훈련 전량 | year(2016) | 없음 | F8TS | R1 λ 0.5 | 3 | 동일 | 2016 | oslo_r 30–60 cm(10 cm 단계); Wei 24.4 %, Liu 41.3 % 상한 초과 | 동일 | maps/Lena/pred_2016 |
| 6h | LE | 2016 | year_obs(OOF) | block OOF | F8TS | R1 | 3 | 관측 35 셀 → 공통 14 | 2016 | y −20…55 cm | – | 동일 |
| 7a | AK | 훈련 전량 | year(연도별 전체 계절) | 없음 | F8TS | R1 | 3 | AOA 페이드; 관측 0.009° 셀 321 개 표시 | 2011–2024 평균 | oslo_r 40–90 cm | 250 m bilinear, ≤2 셀 틈 메움, 음영 0.30; 창은 보간 없음 | maps/Alaska/mean_2011_2024 |
| 7b | LE | 동일 | 동일 | 동일 | 동일 | 동일 | 3 | 관측 셀 143 개 | 2011–2024 평균 | oslo_r 40–60 cm | 동일 | maps/Lena/mean_2011_2024 |
| 7c | AK·LE | – | year | – | – | – | – | 셀별 OLS 기울기 | 2011–2024 | broc ±10 cm/10 yr 공통 | 없음(4배 격자) | maps/*/pred_2011…2024 |
| 7d | AK·LE | 관측 ≥50 셀-연도 연도(AK 2011–2022, LE 2012–2021) | – | – | – | – | – | 1.5 km 내 최근접 지도 셀, 예측 평균에 중심 맞춤 | 2011–2024 | y AK −15…10, LE −20…50 cm | – | maps + labels |
| 7e | AK | – | year | – | – | – | – | – | 2011–2024 각 연도 | vik ±15 cm 공통 | 없음(4배) | maps/Alaska/pred_* |
| S1a | Samoylov 2015 | 탐침 일별 | 일별 누적 TDD | – | – | E 원점 최소제곱 | – | ALLena 탐침 전체 | – | x 6–10월 | – | PANGAEA 973813, ERA5-Land |
| S1b | Samoylov 2015, Kurungnakh 2020 | – | 관측일 TDD | – | – | 사이트별 E | – | – | – | – | – | 동일 |
| S1c | AK(밀도)·LE·CA | ≥4년 관측 셀(AK 1,808·LE 134·CA 219 셀-연도) | year_obs | – | – | 지역별 원점 회귀 | – | 창 밖 점 제외(x −8…6, y ±45: AK 45, LE 1 제외; 회귀는 전체) | – | 공통 | – | testbed_v5 |
| S1d | AK·LE·CA | 전량 | year_obs | block(폴드별 E 평균) | – | P0, P1 | – | – | – | – | – | core_v6 meta P1 json |
| S2a | AK·LE·CA | 전량 | year_obs | block | F8TS | D0, F1, D1, R1 (CatBoost) − P1 | 3 | 2,000 재표집 | – | a·b 공통 y | – | core_v6 deltas_vs_ref |
| S2b | AK·LE·CA | 전량 | year_obs | block | F8TS | R1·D0 × 학습기 5종 | 3 | 동일 | – | 공통 y | – | core_v6, rf_v6, nn_v6 |
| S3a–b | AK·LE·CA | 전량 | year_obs | 없음(전량 적합) | F8TS | R1 잔차 학습기, D0 | – | SHAP exact | – | a 0–100 %, b 공통 x | – | extra_v9 shap_* (gxnyhb) |
| S4a | AK·LE·CA | 전량 풀 | year_obs | block, n 서브샘플 | F8TS | D0, D1, F1, R1 − P1(n) | 3 | 4a 와 같은 탈락 규칙 | – | y 지역별 자동 | – | ladder_v6 |
| S4b | AK·LE·CA | 원천 + 표적 n | year_obs | region hold-out | F8TS | P0, P1n, D0S, D1S, R1S, R1T | 3 | n 0–500 | – | y 지역별 자동 | – | ladder_v6 |
| S5a | 3지역 + 외부 8 소지역 | 2010–2024 | – | – | – | – | – | – | – | – | – | v9_ext |
| S5b | 외부 8 소지역 + 전체 | 2010–2024 | year_obs | 외부 무라벨(R1), 국지 E 재보정(R1loc) | F8TS | R1, R1loc, Wei, Liu, CCI v5 | 3 | 'in_range' 부분집합(훈련 공변량 범위 안) + 계열별 값 있는 행(W Siberia R1 n 218, Liu 153) | – | x log 4–200 | – | extra_v9 ext_metrics(gxnyhb) |
| S5c | 외부 6 소지역 | 2010–2020 | year_obs | 4e 와 동일 | F8TS | R1S 0·5, CCI v5 | 3 | Fig. 4e 와 동일 행·동일 값 | – | x log 4–200 | – | extra_v9 preds_EXTFT |
| S6a | 3지역 + Wei 훈련지 | 전량 | – | – | – | – | – | – | – | – | – | ALT_variables.xlsx, labels |
| S6b | AK·LE·CA | 전량 | – | – | – | – | – | 라벨 셀별 최근접 Wei 훈련지 거리 | – | x log 0.5–2,000 | – | xg_leak_cells_v1.csv |
| S6c | AK | – | year(2014) | – | – | – | – | 10번째 셀마다 표본, 위도대 × 고도(≤600 / >600 m) | 2014 | y −15…140 고정 | – | v9_qa/map_band_alaska2014_by_elev.csv |
| S7 | AK·LE·CA | AK 2010–2024 (15,532), LE 2013–2021 (814), CA 2010–2024 (1,191) | year_obs | block | F8TS | Praw, Pcal, RP λ 0.5, P1, R1 | 3 | Wei 값 있는 셀-연도 | – | x 15–32 공통 | – | extra_v9 pa_metrics |
| S8a–b | AK | 훈련 전량 | year | – | F8TS | R1 | 3 | 50 km 창 2개 | 2011–2024 평균 | oslo_r 40–90 | 창: 보간 없음 | maps/Alaska/mean_2011_2024 |
| S8c | AK | – | clim(AOA 는 2011–2024 평균 공변량) | – | – | – | – | – | 정적 | 2색 | 없음 | 동일(aoa_in) |
| S8d–e | LE | 훈련 전량 | year(2016) | – | F8TS | R1 | 3 | 24 km 창 | 2016 | oslo_r 30–60(extend neither) | d 보간, e 없음 | maps/Lena/pred_2016 |
| S9a | AK | 훈련 전량 | year | – | F8TS | R1 − P1 | 3 | AOA 페이드 | 2011–2024 평균 | vik ±15 공통(AK 대칭 15, LE 10) | 250 m | maps/Alaska/mean_2011_2024 |
| S9b | LE | 훈련 전량 | year(2016) | – | F8TS | R1 − P1 | 3 | 동일 | 2016 | 동일 | 250 m | maps/Lena/pred_2016 |
| S10a | AK loc 14268(71.32°N 156.60°W), 관측 15년 | 2010–2024 | year_obs(관측일)·year(계절 말) 둘 다 | block OOF(별도 재적합) | F8TS | R1, P1; Wei·Liu 2010–2020 | 3 | 관측 연도 최다 셀 | – | y 자동 | – | site_series_year_obs.csv.gz(v5_site_series.py) |
| S10b | LE loc 14813(71.78°N 129.41°E), 관측 8년 | 2010–2024 | 동일 | 동일 | 동일 | R1, P1(제품값 없음) | 3 | 동일 | – | y 자동 | – | 동일 |

---

## 2. 지적 사항

번호 앞의 괄호는 요청된 범주(1 설정 불일치, 2 축·색 범위, 3 명명, 4 캡션–스크립트 불일치, 5 주 설정 밖 자료)이다.

### 2.1 즉시 수정이 필요한 항목

**F1 [4/5] SI PDF 가 v12 그림을 싣는다(원본은 수정됨, 재빌드 미완).**
감사 시점(12:50) 의 `si.tex:10` `\sifig` 매크로는 `v13_si/#1.pdf` 존재 여부를 검사한 뒤 `\includegraphics` 경로로는 `v12_si/#1.pdf` 를 썼다. `build/si.log`(12:14) 에 10개 파일 모두 `v12_si/` 경로로 적혀 있고, `v12_si` 와 `v13_si` 의 PDF 10개는 md5 가 모두 다르다. 12:55 판 `si.tex:10` 은 `v13_si` 로 고쳐졌으나 `si.pdf`(12:14) 는 아직 v12 판 보충 그림이다. `build.sh si` 재실행이 필요하다.

**F2 [3/4] '셀'의 단위가 그림마다 다르다.**
`methods.tex:5` "All observations were assigned to 1 km cells", Fig. 1 캡션(`methods.tex:9`) "number of 1 km cells with ALT labels", Fig. 2 캡션(`methods.tex:45`) "Each label is one 1 km cell in one year", Table 1 과 Fig. 3–5·S1–S7 의 모든 '셀-연도' 수는 `loc_id` 단위이다. 그런데 `scripts/1_data_prep/build_v5_labels_cell_year.py:6, 53` 에서 `loc_id` 는 위경도를 소수 4자리(약 11 m)로 반올림한 지점이다. 실측: 고유 `loc_id` 간 최근접 거리 중앙값 AK 5 m, LE 12 m, CA 13 m; 0.5 km 안에 이웃이 있는 비율 98–99 %. Alaska 13,253 `loc_id` 는 0.009° 격자로 321 셀, 15,540 셀-연도는 910 셀-연도(한 셀-연도에 최대 701 지점)이다. 반면 Fig. 6 d·h(`fig6_product_maps.py:86–87, 97`) 와 Fig. 7 a·b(`fig7_maps.py:146–147`, `v7_f6_maps.py:135–140`, `CELL = 0.009`) 는 실제 1 km 셀로 집계한다(공통 셀 95 / 14, 표시된 관측 셀 321 / 143). 같은 낱말 '셀'이 본문·Fig. 1·2·3·4·5 에서는 약 10 m 지점, Fig. 6·7·S8 에서는 1 km 셀을 뜻한다. Fig. 1a 의 "13,253 cells" 와 Fig. 7a 의 관측 셀 321 개가 한 원고 안에 공존한다.

**F3 [4] v13 패널 재배치 뒤 남은 참조 오류.**
감사 시점(12:50) 기준으로 아래 네 곳이 옛 패널 글자를 가리켰다. 12:55 판에서 `si.tex:73`(→ 4d,e), `si.tex:75`(→ 5d), `si.tex:127`, `results.tex:43`(→ d) 은 고쳐졌음을 확인했다.
- (수정됨) `si.tex:73` Table S2 "External regions (Fig.~4c …)" → Fig. 4d.
- (수정됨) `si.tex:75` Table S2 "Year-wise comparison (Fig.~5b)" → Fig. 5d.
- (수정됨) `si.tex:127` S5 캡션 "enters only the pooled value of Fig.~4c" → Fig. 4d.
- (수정됨) `results.tex:43` 연도별 비교 문장 `Fig.~\ref{fig:products}b` → d.
- (미수정) 스크립트 머리말: `fig7_maps.py:7–9` 는 d = 아노말리 지도, e = 시계열이라 적었으나 코드(`:222`, `:240`) 와 캡션(`results.tex:50`) 은 d = 시계열, e = 아노말리 지도. `fig5_products.py:8` 의 "b RMSE per observation year" 는 지금 d.

**F4 [4] Fig. 4e 에 없는 소지역을 본문이 Fig. 4e 근거로 인용한다.**
`results.tex:28` 은 캄차카(\nftKaRszero → \nftKaRsfive) 를 "(Fig.~\ref{fig:labels}e)" 와 함께 인용하지만, `fig4_labels.py:297` 의 행은 `v9_f10_extra.py:153` 의 `FT_SUBS` 6개(캄차카·그린란드 제외) 뿐이다(JSON e.values 6 소지역). S5 캡션(`si.tex:127`)도 캄차카는 풀 값에만 들어간다고 적는다.

**F5 [1/4] Fig. 3b 의 다섯 설계는 같은 셀-연도 집합이 아니다.**
캡션(`results.tex:9`) "All panels use the cell-year labels of 2010–2024". year·block×year 설계는 ≥20 셀-연도 연도만 채점한다(JSON `label_years_by_region_and_split`): LE 2012–2021, 2,419 / 2,374 (전량 2,436); CA 2012–2024, 1,164 / 1,135 (전량 1,196); AK block×year 15,421. 설계 간 RMSE 비교에 집합 차이가 섞여 있는데 캡션이 말하지 않는다(방법 절에는 ≥20 규칙이 있다).

**F6 [1/4] Fig. 5d 의 셀-연도 집합이 5a–c 와 다르다.**
`fig5_products.py:289–291`(d) 는 연도별로 R1·P1·Wei 값이 있는 셀-연도이고 Liu 는 ≥30 이 남을 때만 넣으며 CCI v4 를 요구하지 않는다(AK 2021–2022, CA 2021–2024 포함). 5a–c(`:283–286`, `:197`) 는 Wei·Liu·CCI v4 공통 셀-연도(2010–2020) 이다. 예: LE 2017 은 5a 기준 16 셀-연도, 5d 기준 109. 캡션(`results.tex:35`) "on the same cell-years of each year" 는 집합 정의가 없어 5a 의 공통 집합으로 읽힌다.

**F7 [2/4] Fig. 6 의 행 간 설정 차이가 캡션에 없다.**
- 색 범위: AK 35–85 cm(5 cm 단계), LE 30–60 cm(10 cm 단계) (`fig6_product_maps.py:49–52, 142`). 캡션은 "on a common colour range" 라고만 적어 두 행이 같은 범위로 읽힌다.
- 범위 규칙: 우리 지도의 1–99 백분위수이므로 제품 화소가 상한에서 포화한다(JSON `share_of_drawn_pixels_beyond_range`: AK Wei 13.6 %, Liu 20.5 %; LE Wei 24.4 %, Liu 41.3 %). 제품 지도 b·c·f·g 가 실제보다 균질하게 보인다.
- 오차 축: AK −55…45 cm, LE −20…55 cm (`:50–52`, `:195`), 비대칭이고 행마다 다르다.
- AOA 페이드는 우리 지도에만 적용(`:160` vs `:162`) 되지만 범례(`:288`) 는 그림 전체 규칙처럼 적혀 있다.
- 지도 a·e 는 전량 적합 모형 + 전체 계절 TDD, 점 d·h 는 OOF + 관측일 TDD 이다. 캡션은 "spatial-block out-of-fold" 만 말하고 기후 입력 차이는 Table S2 에만 있다.

**F8 [1] S8·S9 가 Alaska 는 다년 평균, Lena 는 2016 단일 연도를 나란히 둔다.**
`si_s08_map_detail.py:43–53`, `si_s09_correction_maps.py:33–35`; 캡션 `si.tex:130–131` 은 차이를 적지만 이유가 없다. `maps/Lena/mean_2011_2024.csv.gz` 에 `p1`, `r1` 열이 있어 Lena 평균 보정 지도도 바로 만들 수 있다(Fig. 7b 도 이미 평균). S8d(Lena 2016 ALT) 는 Fig. 6e 를 크게 그린 것이다.

**F9 [4] S5b 의 부분집합과 S5c 의 중복.**
`si_s05_external.py:137` 정의: b 는 'in_range'(훈련 공변량 범위 안 지점) 부분집합이고 계열별로 값 있는 행에서 채점한다. 캡션(`si.tex:127`) 은 후자만 적고 in_range 를 적지 않는다(W Siberia: b 의 R1 n 218, c 의 n 153). S5c 는 Fig. 4e 와 소지역 6개·값이 완전히 같다(JSON `values_c` == Fig4 `e.values`).

### 2.2 수정 권고 항목

**F10 [2] 패널 간 축·색 범위가 다른데 캡션이 말하지 않는 곳.**
| 그림 | 범위 | 판단 |
|---|---|---|
| 1b (`fig1_data.py:268`) | y 0–8,000 / 0–800 / 0–500 | 100배 차이라 공통 축은 불가. "independent y axes" 를 캡션에 적는다. |
| 3a (`fig3_accuracy.py:145, 155–156`) | y 폭 8 cm 동일, 오프셋 지역별 | 변화량 비교용 설계. 캡션에 "same y span, offsets differ" 를 적는다. |
| 3b (`:173–175`), 4a (`fig4_labels.py:139–140`), 4b (`:162`), 5c (`fig5_products.py:244–245`), S4, S10 | 지역별 자동 | 값 수준이 지역마다 달라 공통 축은 세부를 숨긴다. 캡션에 한 문장. |
| 5d (`fig5_products.py:140–143`) | y 0–50 / 0–90 / 0–60 | 공통 0–90 은 AK(10–25 cm) 를 압축. 유지하되 명시. |
| 6 d·h | −55…45 / −20…55 | 공통 대칭 ±60 cm 로 통일 가능(초과값 표기 규칙 이미 있음). |
| 7a vs 7b (`fig7_maps.py:139, 171`) | 40–90 / 40–60 cm | 지역별 범위는 타당하나 "colour ranges differ between regions" 명시. S8 a–b(40–90) vs d–e(30–60) 도 같음. |
| 7d (`:85–87`) | AK −15…10, LE −20…50 | LE 2020 관측 아노말리 46.9 cm(n 143) 가 축을 정한다. 공통 −20…20 에 2020 값을 가장자리 표기(다른 그림과 같은 규칙)로 두면 AK 의 변동이 보인다. |
| 공통 범위가 이미 적용된 곳 | 3c x, 3d ±7, 5a x, 5b 0–120, 4d 12–66, 7c ±10, 7e ±15, S2, S7, S9 ±15 | 문제 없음. |

**F11 [3] 명명 불일치(그림 글자 vs 캡션·본문).**
- 지역: 그림 "Canada (NW)" (`c13_common.py:59`, `fig34_common13.py:64`, `si13_common.py:52`) vs 캡션 "Canada", 방법 절 "north-western Canada"; Fig. 2c "Training: Alaska, Canada".
- 제품: 그림 "Wei 2026", "Liu 2024", "Aalto 2018", "Yi–Kimball 2019 (Alaska)" (`c13_common.py:54–56`, `fig1_data.py:106`) vs 캡션·Table S1 "Wei et al.", "Liu et al.", "Aalto et al.", "Yi and Kimball".
- 방법: 그림 "Stefan (recalibrated)", "Stefan (source coeff.)", "Physics + residual ML", "Direct ML" (`fig34_common13.py:75–80`, `c13_common.py:51`) vs 캡션 "recalibrated Stefan model", "source-coefficient Stefan model", "physics-residual model", "direct ML". Fig. 6 눈금 "Physics + residual ML (block CV)" (`fig6_product_maps.py:53`) vs 머리글 "Physics + residual ML" (`:55`). Fig. 4d 범례 "Stefan (recalibrated)" (`fig4_labels.py:285`) 는 외부 라벨 n 으로 재보정한 P1n 인데 S4b 는 "Stefan (recalibrated on target labels)" (`si_s04_label_ladder.py:35`), JSON 은 "Stefan (recalibrated on n)".
- 기후 입력: Fig. 3a 눈금 "2011–2024 mean / Year-specific" (`fig3_accuracy.py:41`), Fig. 2 범례 "To observation day (model input) / Full season (yearly maps) / Mean 2011–2024" (`fig2_workflow.py:488`), S10 범례 "observation day / end of season" (`si_s10_site_series.py:81`), 캡션 "mean climate input / year-specific climate input / full-season". 같은 세 변형에 네 가지 표현.
- TDD: Fig. 3d 행 이름 "TDD" (`fig3_accuracy.py:47`) 는 전체 계절 TDD 인데 모형 입력은 TDD_obs. 단위 "°C d" (`fig2_workflow.py:485, 510`) vs "°C·d" (`si_s01_stefan.py:38, 83`) vs 본문 "(°C d)^−1/2".
- 라벨 수: Fig. 4e "0 labels / 5 labels" (`fig4_labels.py:317`) vs S5c "0 local labels / 5 local labels" (`si_s05_external.py:127`).
- 관측 셀: Fig. 7 범례 "Observed cell"(1 km 셀) vs Fig. 1a "cells"(지점). F2 와 연결.

**F12 [4] Fig. 4 캡션이 적지 않는 스크립트 규칙.**
- a: n 범주가 지역마다 다르다(AK 25–2,500, LE 25–1,000, CA 25–500; CA 1,000 은 `fig4_labels.py:64–74` 의 n_test < 0.95·all 규칙으로 탈락, JSON `dropped_categories`). "three draws" 는 `v6_label_ladder.py:68–70` 의 `tr_full.sample(n, random_state=s)` 로 씨앗마다 한 번 추출이므로 맞지만, 추출과 학습기 씨앗이 함께 바뀐다(분산이 합쳐짐). b 도 같은 방식(`:99–102`) 인데 캡션에는 추출 언급이 없다.
- b: 1.3배 초과값 상단 표기 규칙(`:160–176`; LE Direct ML n=0 39.5 cm) 이 캡션에 없다(Fig. 3b 캡션에는 있다).
- d: 면 제목 "All external sites (pooled)" (`:283`) 이지만 그린란드가 빠진 7 소지역(JSON `subregions`). 범례 "Stefan (recalibrated)" 는 외부 n 라벨로 재보정한 값.

**F13 [4] Fig. 5c 캡션 누락.** 계급 < 10 셀-연도 미표시 규칙(`fig5_products.py:236`, 현재는 탈락 계급 없음) 과 "CCI v5 는 값 있는 곳"(AK 13,048 / 14,326) 이 캡션(`results.tex:35`) 에 없다.

**F14 [4] Fig. 1c '사용 연도' 정의.** `fig1_data.py:118–120, 363` 은 제품값이 있는 라벨 셀-연도의 연도 범위(커버리지) 를 '사용 연도'로 그린다(Wei 2010–2024, CCI v4 2010–2021, CCI v5 2010–2023). 실제 비교(Fig. 5a) 는 2010–2020 공통 집합이고 Wei 2021–2024 는 Fig. 5d 에서만, CCI v4 2021·CCI v5 2021–2023 은 어디서도 채점되지 않는다. 캡션 "years used for comparison at the label cells" 는 과대 표현이다.

**F15 [5] 주 실행 폴더 밖 자료의 출처가 캡션에 없다.**
- Fig. 3c: F8·F8T·F8TS_X 는 `featsets_v6`, F8TS·F8TS_M·F8TS_ALL 은 `modis_v6` (`fig3_accuracy.py:105–114`). F8TS 의 Δ 는 core_v6 와 동일(−0.15 / −3.59 / −3.74 cm), 신뢰구간은 실행 간 둘째 자리에서 다르다(AK core_v6 −0.569…0.119, featsets_v6 −0.558…0.114). Table S4 에 실행이 있으나 캡션에는 없다.
- Fig. 3d (`:33`) 와 Fig. 5d (`fig5_products.py:31`) 는 `outputs/figures/paper/v7/source_data/` 의 CSV 를 읽는다(각각 `v9_f4e_gain_conditions.py`, `v7_f5_products_years.py` 가 core_v6 preds 로 계산). 원천이 이전 판 그림 폴더에 있다.
- S10: `v5_site_series.py` 의 별도 3씨앗 재적합(core_v6 OOF 파일이 아님). 캡션 "spatial-block out-of-fold" 는 맞지만 실행이 다르다.
- 지도(Fig. 2b, 6, 7, S8, S9): 전량 적합·전체 계절 TDD 로 주 설정과 기후 입력이 다르다. 방법 절·Table S2 에는 있으나 Fig. 6 캡션에는 없다(F7).

**F16 [4] S1c 의 창 밖 점.** `si_s01_stefan.py:48, 128` 은 x −8…6, y ±45 밖 점을 그리지 않는다(AK 1,808 중 45, LE 134 중 1; 회귀는 전체 사용). 캡션(`si.tex:123`) 에 없다.

**F17 [4] S4 캡션 "with all methods".** b 에는 R1T(표적 라벨만) 와 D1S 가 있고 F1(physics-input ML) 은 없다(`si_s04_label_ladder.py:35–37`).

**F18 [3] 보충 스크립트 머리말·JSON 의 옛 번호.** `si_s04_label_ladder.py:1, 13` ("Supplementary Fig. S10", 실행 경로 `si_s10_label_ladder.py`), `si_s05_external.py:1, 13` ("S4"), `si_s06…si_s10` 도 한 번호씩 어긋남; `SI_figures.json` 의 `script`·`legend_draft` 필드도 옛 파일명·번호. 독자에게는 보이지 않으나 출처 추적을 흐린다.

**F19 [4] `si.tex:123–132` 라벨(수정됨).** 감사 시점에는 S1·S2 가 모두 `fig:s1`, 이후 S3 = `fig:s2` … S10 = `fig:s9` 였다. 12:55 판은 `fig:s1`…`fig:s10` 으로 바로잡혀 있다.

### 2.3 확인하여 문제 없다고 판단한 항목
- Fig. 2 캡션의 수치: 곡선 셀 2013년(JSON 14813, 8/17), Stefan anchor 15,540, 지도 2014, 아노말리 2012·2019, 채점 제외 연도 규칙(17행) 모두 스크립트와 일치.
- Fig. 3d 해칭 규칙(≥30 셀-연도·≥5 블록, 2,000 재표집) 과 색 범위 ±7 cm(자료 최대 |Δ| 6.2 cm) 일치.
- Fig. 4d 474 셀-연도·2010–2020, 4e 6 소지역 ≥30, S7 의 연도·n, Fig. 5b n 461, Fig. 6 의 95 / 14 셀·85 셀 ≤25 km, Fig. 7d 연도 범위(AK 2011–2022, LE 2012–2021) 는 매크로와 일치.
- 공통 범위가 필요한 곳에 공통 범위를 쓴 패널: Fig. 3c·3d, 4d, 5a·5b, 7c·7e, S2, S7, S9.
- 실행 메타: `ladder_v6`, `extra_v9`, `maps/*/meta.json` 모두 λ 0.5, CatBoost, 씨앗 0·1·2, F8TS(23 변수) 로 주 설정과 같다.

---

## 3. Fig. 6 의 연도 선택(Alaska 2014, Lena 2016) 검토

Fig. 6 의 단위는 0.009° 셀이므로 '가장 잘 관측된 해'를 두 단위로 센다(in_forcing == 1; 제품값은 `products_cell_year_v5.csv`, Liu 는 2020 까지).

| 연도 | AK 라벨 지점 | AK 관측 셀 | AK Wei·Liu 모두 있는 셀 | LE 라벨 지점 | LE 관측 셀 | LE Wei·Liu 모두 있는 셀 | CA 라벨 지점 |
|---|---|---|---|---|---|---|---|
| 2013 | 1,343 | 137 | 135 | 241 | 19 | 6 | 121 |
| 2014 | 7,861 | 101 | 100 | 93 | 23 | 12 | 8 |
| 2015 | 171 | 44 | 44 | 471 | 16 | 6 | 4 |
| 2016 | 2,237 | 82 | 82 | 681 | 35 | 15 | 4 |
| 2017 | 877 | 81 | 80 | 139 | 12 | 2 | 150 |
| 2018 | 1,526 | 103 | 103 | 64 | 23 | 6 | 150 |
| 2019 | 450 | 51 | 50 | 95 | 28 | 7 | 65 |
| 2020 | 149 | 22 | 22 | 143 | 27 | 5 | 0 |

판단
- 캐나다는 지도가 없으므로(`maps/` 에 Alaska·Lena 만) Fig. 6 은 두 지역만 해당한다.
- 현재 근거 "best-observed year" 는 라벨 지점 수 기준에서만 성립한다. 셀 기준으로 Alaska 는 2013(137 셀) 이 2014(101) 보다 많다.
- 단일 공통 연도는 2016 으로 가능하다: Alaska 2016 은 지점 2,237(2위), 관측 셀 82, Wei·Liu 공통 셀 82; Lena 2016 은 지점·셀·공통 셀 모두 최다(681 / 35 / 15). 2014 를 공통 연도로 쓰면 Lena 공통 셀이 12 로 줄고, 2017 은 Lena 공통 셀 2 로 불가. Liu 가 2020 에 끝나므로 2021 이후는 불가.
- 권고: 두 행 모두 2016 으로 통일하고 캡션에 "2016, the year with the most observed cells in the Lena Delta and the second most in Alaska" 한 문장을 적는다. Alaska 2014 를 유지하려면 "the year with the most label points (7,861), which fall in 101 1 km cells" 로 단위를 밝혀야 한다(F2). 2016 으로 바꾸면 S6c(Alaska 2014 vs 제품), S8d·S9b(Lena 2016) 와 본문 `results.tex:39` 의 수치 매크로(\nfeAk*, \nfeLe*, \nmapAkFootMtn*) 를 함께 갱신해야 한다.

---

## 4. 수정 목록(우선순위)

| 순위 | 파일 · 위치 | 변경 내용 |
|---|---|---|
| P0-1 | `paper/manuscript/en_v13/si.tex:10` `\sifig` (12:55 판에서 `v13_si` 로 수정됨) | `build.sh si` 재실행; `build/si.log` 에서 `v12_si` 가 사라졌는지와 `si.pdf` 시각이 12:55 이후인지 확인 (F1) |
| P0-2 | `sections/methods.tex:5, 9, 45`, `sections/table1.tex`, Fig. 1a (`fig1_data.py:206`), Fig. 7 범례 (`fig7_maps.py:166`) | 라벨 단위를 "site(point)-year" 또는 "location-year" 로 고쳐 부르고, 1 km 셀 수(AK 321, LE 143, CA 71)와 셀-연도 수(910 / 244 / 144) 를 Table 1 또는 Table S2 에 병기; Fig. 7·S8 범례는 "Observed 1 km cell"; 혹은 라벨을 실제로 1 km 셀로 집계하는 재설계를 검토 (F2) |
| P0-3 | `fig7_maps.py:7–9`, `fig5_products.py:8` (tex 쪽 네 곳은 12:55 판에서 수정됨) | 스크립트 머리말의 패널 글자를 코드·캡션과 맞춤 (F3) |
| P0-4 | `results.tex:28` 또는 `fig4_labels.py:draw_e` | 캄차카(16)·그린란드를 4e 에 "n < 30" 표식(열린 기호·회색)으로 추가하거나, 본문의 캄차카 수치를 S5b 또는 수치 표 인용으로 바꿈 (F4) |
| P0-5 | `results.tex:9` Fig. 3 캡션 | b 에 "year hold-out and block-by-year score only years with ≥ 20 cell-years (Lena Delta 2012–2021, 2,419 and 2,374 cell-years; Canada 2012–2024, 1,164 and 1,135)"; a 에 "same y span, offsets differ"; d 의 TDD 를 "full-season TDD" 로 (F5, F10, F11) |
| P0-6 | `results.tex:35` Fig. 5 캡션 d | "cell-years of each year with a value of Wei et al. (and Liu et al. where ≥ 30 remain; ESA CCI v4 not required), a larger set than the common cell-years of a–c" 와 "y ranges differ between regions"; c 에 MIN_BIN·CCI v5 규칙 (F6, F13) |
| P0-7 | `results.tex:41` Fig. 6 캡션; `fig6_product_maps.py:49–52` | 캡션에 "colour range = 1st–99th percentile of our map in each row (35–85 cm Alaska, 30–60 cm Lena Delta); product pixels above the range are drawn in the top colour (13.6 % / 20.5 % of Wei et al. / Liu et al. pixels in Alaska, 24.4 % / 41.3 % in the Lena Delta); fading outside the area of applicability applies to our map only; maps use the full-season TDD of the year and the full-data model, d and h use out-of-fold predictions with TDD to the observation day". `err_lim` 을 두 행 공통 ±60 cm 로 (F7, F10) |
| P0-8 | `fig6_product_maps.py:49–52` `ROWS[...]["year"]`; 연동 `results.tex:39`, `shared_v11/numbers_v11.tex` 재생성 | 두 행 모두 2016 으로 통일(또는 2014 유지 시 캡션에 지점·셀 수 명시); 3절 참조 (F2, F7) |
| P0-9 | `si_s09_correction_maps.py:35` `ROWS[1]["src"]` → `"mean_2011_2024"`, label "Lena Delta, mean 2011–2024"; `si_s08_map_detail.py:51–53` | S9b 를 Lena 평균 보정 지도로, S8d 를 Lena 평균(또는 d 삭제·e 만 유지) 으로 바꾸고 캡션 `si.tex:130–131` 갱신 (F8) |
| P0-10 | `si.tex:127` S5 캡션; `si_s05_external.py:99–118` | b 에 "sites inside the covariate range of the training data (in_range)" 추가; c 는 삭제하고 "as Fig. 4e" 로 대체하거나 8 소지역 전부로 확장 (F9) |
| P1-1 | `results.tex:24` Fig. 4 캡션 | a 에 "n up to 2,500 (Alaska), 1,000 (Lena Delta), 500 (Canada), limited by the training-fold size; each draw uses its own learner seed"; b 에 초과값 표기 규칙; d 에 "seven sub-regions (Greenland has no rows with all three products)" 와 "Stefan recalibrated on the n external labels" (F12) |
| P1-2 | `results.tex:50` Fig. 7 캡션; `fig7_maps.py:85–87` | a·b "colour ranges differ between regions"; d 는 공통 −20…20 cm 에 2020 값 가장자리 표기, 또는 "y ranges differ" 명시 (F10) |
| P1-3 | `methods.tex:9` Fig. 1 캡션; `fig1_data.py:118–120` | c 의 막대를 "years with a product value at the label cells" 로 바꾸거나, 실제 채점 연도(Fig. 5a 2010–2020, Fig. 5d Wei 2010–2024) 만 칠함; b 에 "independent y axes" (F14, F10) |
| P1-4 | `c13_common.py:51–59`, `fig34_common13.py:64, 75–80`, `si13_common.py:45–52` | 그림 글자와 캡션 용어 통일: 지역 "Canada" 또는 양쪽 "NW Canada"; 제품 "Wei et al. (2026)" 형식 또는 캡션을 "Wei 2026" 형식으로; 방법 명칭 한 벌; 기후 입력 세 변형의 표현 한 벌("TDD to observation day / full-season TDD / 2011–2024 mean"); 단위 "°C d" 로 통일 (F11) |
| P1-5 | `results.tex:9` Fig. 3c 캡션, `si.tex:132` S10 캡션 | 실행 출처 한 구절("sensitivity sets from runs CJsgU and CECkU, Supplementary Table S4"; "predictions from a separate three-seed block cross-validation fit") (F15) |
| P1-6 | `scripts/4_visualization/v9_f4e_gain_conditions.py`, `v7_f5_products_years.py` 출력 경로 | `F4e_gain_conditions.csv`, `F5_yearwise_rmse.csv` 를 `outputs/figures/paper/v13/source_data/` 로 재생성하고 `fig3_accuracy.py:33`, `fig5_products.py:31` 경로 갱신 (F15) |
| P1-7 | `si.tex:123, 126` | S1c 에 "points outside the axis window are not drawn (45 of 1,808 Alaska cell-years); fits use all"; S4b 에 포함 방법 명시 (F16, F17) |
| P2-1 | `si.tex:123–132` | (12:55 판에서 완료) 라벨 `fig:s1`…`fig:s10` (F19) |
| P2-2 | `si_s04_label_ladder.py:1, 13` 등 SI 스크립트 머리말, `SI_figures.json` 의 `script`·`legend_draft` | 현재 번호·파일명으로 (F18) |

재생성 순서: P0-8·P0-9 를 먼저 결정한 뒤 Fig. 6, S6c, S8, S9 를 다시 그리고 `v13_numbers.py` 로 매크로를 재생성한 다음 P0-1 의 경로를 고쳐 `main`·`si` 를 다시 빌드한다.
