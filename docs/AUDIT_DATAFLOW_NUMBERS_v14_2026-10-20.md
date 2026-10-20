# 자료 흐름 · 수치 · 실험 설계 감사 (원고 en_v13 v13.2, 2026-10-20 16:30 기준)

대상: `paper/manuscript/en_v13/` (sections/*.tex, si.tex, main.tex), 매크로 `shared_v10/numbers_v{6,9,10}.tex` · `shared_v11/numbers_v11.tex` · `shared_v13/numbers_v13.tex`, 생성기 `scripts/2_evaluation/v{6,9,10,11,13}_numbers.py`, 시험지 `src/polar/testbed_v5.py`, 실행 하네스 `scripts/3_deep_learning/{v5_compare,v6_label_ladder,v9_extra,v5_map_predict}.py`, 자료 구축 `scripts/1_data_prep/{build_v5_labels_cell_year,build_v5_forcing_year,build_v5_products_cell_year,build_v9_external}.py`, 집계 `scripts/2_evaluation/{v5_aggregate,v6_label_ladder_agg,v9_extra_agg,v9_extft_agg,v5_products_eval,v5_common_cells,v5_yearwise,v5_site_series}.py`, 그림 `scripts/4_visualization/paper_v13/*.py` 와 원천 JSON `outputs/figures/paper/v13/source_data/*.json`, `v13_si/SI_figures.json`, 실행 결과 `data/processed/v5/runs/{core_v6,ladder_v6,extra_v9,featsets_v6,modis_v6,rf_v6,nn_v6}`, 지도 `data/processed/v5/maps/<Region>/`.

방법: 파일은 읽기만 했다. 수치 검증은 CSV · JSON 에 대한 Python 집계(nice 10, 4 스레드, GPU 미사용, 학습 없음)로 했다. 이전 감사 3건(`docs/AUDIT_FIGURE_CONTROL_2026-10-20.md`, `AUDIT_TEXT_NUMBERS_2026-10-20.md`, `AUDIT_EXPERIMENT_SETTINGS_2026-10-20.md`)과 `docs/FIGURE_AUDIT_v12_2026-10-18.md` 12절을 먼저 읽고, 거기서 확정된 항목은 조치 적용 여부만 확인했다(4절). 행 번호는 16:30 시점 파일 기준이다.

등급: **P0** 투고 전 반드시 수정(사실과 다른 진술 또는 본문과 방법의 모순), **P1** 수정 권고(낡은 그림 · 캡션 정의 오류 · 재빌드 필요), **P2** 비고(정밀도 · 재현성 · 표현).

---

## 1. 발견 사항 요약

| # | 등급 | 위치 | 요지 |
|---|---|---|---|
| F1 | P0 | `results.tex:39`, `results.tex:41`(Fig. 6 제목), `ko_v13/sections/results.tex:39, 41` | "2013 is the only year with observations in all three regions" 는 사실이 아니다. 세 지역 모두 라벨이 있는 해는 12개(2010–2019, 2021–2022)이고, 세 지도(우리 · Wei · Liu)가 모두 값을 가진 1 km 관측 셀이 세 지역에 다 있는 해도 2013–2019 로 7개다. |
| F2 | P0 | `methods.tex:45`(Fig. 2 캡션 a) | "the yearly maps use the full-season sum" 은 v13.2 지도(9월 1일 TDD, `maps/*/meta.json ref_doy 244`)와 방법 절(`methods.tex:22, 72`)에 어긋난다. 그림 자체는 "To 1 Sep (maps)" 를 그린다(`fig2_workflow.py:492, 649`). 국문 캡션은 이미 고쳐져 있다(`ko_v13/sections/methods.tex:45`). |
| F3 | P1 | `en_v13/si.pdf`(15:01), `deliverables/14_paper_v13_2026-10-18/supplementary_en.pdf`(md5 동일) | 보충 PDF 가 S8(15:22:15) · S9(15:22:52) 그림 파일보다 먼저 빌드되었다. `build.sh si` 와 `tools/package_paper_v13.sh` 재실행이 필요하다. |
| F4 | P1 | `results.tex:9`(Fig. 3 캡션) | "the year and block-by-year designs score only years with at least 20 site-years" 이나 블록×연도 겹의 최소 행 수는 10 이다(`testbed_v5.py:200`, 알래스카 최소 겹 10행). 방법 절(`methods.tex:48`)은 10 으로 맞게 적었다. |
| F5 | P1 | `si.tex:127`(S5 캡션 b), `si_s05_external.py:6, 137` | "sites inside the covariate range of the training data" 는 정의가 틀렸다. `in_range` 는 공변량이 아니라 라벨 범위(셀 평균 ALT ≤ 300 cm)다(`v9_extra_agg.py:54`). |
| F6 | P1 | `si.tex:128`(S6 캡션 c), `si_s06_training_sites.py:229, 258`, `data/processed/v9_qa/map_band_alaska2014_by_elev.csv`(10-15 17:58) | S6c 는 10-15 의 전체 계절 TDD 알래스카 2014 지도(현재 `maps_fullseason_2026-10-14/`)에서 계산한 표를 그린다. 현재 `maps/Alaska/pred_2014.csv.gz`(10-20 13:51, 9월 1일 TDD)와 다르다. 본문 수치(`\nmapAkFootMtn*` +18.7/+52.9, 2013)는 v13 Fig. 6 값으로 덮어써 S6c(2014, +21.9/+55.2)와 어긋난다. |
| F7 | P1 | `si.tex:132`(S10 캡션), `v5_site_series.py:3, 38, 46` | S10 의 "end of season" 선은 관측일 TDD 로 학습한 모델에 그 해 전체 계절 TDD 를 넣은 외삽이다. 지도는 같은 문제 때문에 9월 1일 TDD 로 바꿨는데(`FIGURE_AUDIT_v12` 12절) S10 만 전체 계절 입력이 남았고 캡션은 이를 적지 않는다. |
| F8 | P2 | `methods.tex:53` | "1000 times for ... the covariate-set ... sensitivity runs": 그림 3c 의 F8TS · F8TS_M · F8TS_ALL 은 modis_v6(B = 2000, `run_v6b.sh`), F8 · F8T · F8TS_X 는 featsets_v6(B = 1000, `run_v6.sh:42`)이다. 한 패널에 두 B 가 섞인다. |
| F9 | P2 | `methods.tex:22` | "interquartile range of observation days 224--243" 은 지역별 사분위의 바깥 경계(알래스카 224–230, 레나 229–241, 캐나다 234–243)이고, 전체 라벨의 IQR 은 224–234 이다. "the full-season sum would exceed it by 20--30\%" 는 실측 중앙값 37 % · 23 % · 12 %(관측일 TDD 비 0.73 · 0.81 · 0.89 의 역수)와 맞지 않는다. |
| F10 | P2 | `methods.tex:22, 72`, `results.tex:48`, `v5_map_predict.py:95–102`, `testbed_v5.py:113` | "1 September (day 244)": 코드는 모든 해에 244일을 쓰므로 윤년(2012 · 2016 · 2020 · 2024)에는 8월 31일이다. |
| F11 | P2 | `testbed_v5.py:118` | 관측일 보간의 달 번호를 평년 달 시작표로만 찾아 윤년의 '달 끝 하루'가 다음 달로 분류된다. 영향: 윤년 행 3,735 중 64행에서 1 °C d 초과, 최대 8.6 °C d(≤ 1.1 %). 결과에 영향 없음. |
| F12 | P2 | `methods.tex:43`, `v6_label_ladder.py:100–101` | "All learned models were fitted with three random seeds": 전이 사다리 n = 0 은 씨앗 0 하나만 쓴다(`ladder_metrics.csv` n_label 0 의 `rmse_seed_sd` NaN). |
| F13 | P2 | `methods.tex:5`, `build_v5_labels_cell_year.py:113–114` | ALLena 는 위치·연도별 최댓값을 라벨로, 관측일은 평균 doy 로 둔다. 한 해에 여러 날 관측된 레나 지점-연도 384개(15.8 %, 그중 8·9월 걸침 19)는 라벨 날짜(최댓값 측정일)와 TDD 누적일(평균일)이 다르다. 본문은 "retaining the mean day of observation" 이라 적어 규칙 자체는 맞게 서술했다. |
| F14 | P2 | `front.tex:18` | 자료 가용성에 "yearly 1 km maps of Alaska and the Lena Delta" 만 있다. 캐나다 지도(`maps/Canada`, 1,151,169 셀)가 추가되었다. |
| F15 | P2 | `table1.tex:30`, `methods.tex:64` | Yi and Kimball 행의 역할이 "Comparison" 이나 v13 에서는 어느 그림 · 표에서도 채점하지 않는다(Fig. 5 `ROWS` 는 제품 5종, `fig5_products.py:36–41`; 초록도 "five published products"). |
| F16 | P2 | `si.tex:7`, `ko_v13/si.tex:7` | 보충은 `numbers_v13.tex` 를 읽지 않는다. 지금은 v13 이 덮어쓰는 매크로 84개(본문에 쓰이는 40개 포함)가 보충 본문에 쓰이지 않아 값 차이가 없지만(3.2절), 보충 캡션에 `\nfe*` · `\nanobs*` · `\nmapAk*` 를 쓰는 순간 v9 · v11 의 옛 값(2014 · 2016 지도)이 조용히 인쇄된다. |
| F17 | P2 | `scripts/3_deep_learning/v5_map_predict.py:3–4, 8, 21, 23`, `scripts/4_visualization/paper_v13/fig2_workflow.py:12–13`, `scripts/2_evaluation/v13_numbers.py:1–2`, `outputs/figures/paper/v13/source_data/Fig6_product_maps.json` `why_this_year`, `fig6_product_maps.py:2, 55, 367` | 낡은 주석 · 기본값: 지도 스크립트 기본 `--run core_cb --forcing year`(실제 실행은 core_v6 · year_obs · `--ref-doy 244`), Fig. 2 머리말 "filled circle = full-season TDD (maps)", v13_numbers 머리말, Fig. 6 의 "the only year with labels in all three regions". |
| F18 | P2 | `docs/numbers_v9.json`, `shared_v10/numbers_v9.tex:\nfe*`, `docs/numbers_v11.json`, `shared_v11/numbers_v11.tex`, `outputs/figures/paper/v7/source_data/F8_product_maps.json`, `F7_series_significance.json` | 전체 계절 지도(2014 · 2016)에서 만든 매크로 원천이 남아 있다. 본문에서는 모두 v13 이 덮어쓰므로 인쇄값은 현행이다. 원천 정리 또는 재생성 권고. |

이전 감사에서 지적된 항목 가운데 v13.2 에서 조치가 적용되었음을 확인한 것은 4절에 적었다.

---

## 2. 발견 사항 상세

### F1 [P0] "2013 is the only year with observations in all three regions"

- 본문: `results.tex:39` "Maps of one common year show these differences in space (Fig. 6); 2013 is the only year with observations in all three regions." `results.tex:41` Fig. 6 제목 "... for 2013, the only year with observations in all three regions." 국문 `ko_v13/sections/results.tex:39, 41` "세 지역 모두에 관측이 있는 유일한 해인 2013년". 원천 JSON `Fig6_product_maps.json` `why_this_year` 와 `fig6_product_maps.py:2` 도 같은 문장.
- 자료(`labels_cell_year_v5.csv`, in_forcing == 1): 세 지역 모두 라벨이 있는 해 = 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022(12개). 연도별 라벨 수(알래스카 / 캐나다 / 레나): 2013 1,343 / 121 / 241; 2014 7,861 / 8 / 93; 2016 2,237 / 4 / 681; 2017 877 / 150 / 139; 2018 1,526 / 150 / 64; 2019 450 / 65 / 95.
- Fig. 6 의 단위(1 km 셀, Wei · Liu 값이 모두 있는 관측 셀; `fig6_product_maps.py:101–118`)로 세어도 2013 은 유일하지 않다: 2013 135 / 34 / 6, 2014 100 / 6 / 12, 2015 44 / 3 / 6, 2016 82 / 3 / 14, 2017 80 / 7 / 2, 2018 103 / 7 / 6, 2019 50 / 1 / 7(알래스카 / 캐나다 / 레나). 2013 은 알래스카 · 캐나다에서 셀 수가 가장 많고 레나에서는 2016(14) · 2014(12) · 2019(7) 다음이다(6 셀, 2015 · 2018 과 같음).
- 수정: `results.tex:39` 를 "2013, the year with the most observed 1 km cells with product values in Alaska and Canada (NW) (135 and 34) among the years observed in all three regions" 로, Fig. 6 제목(`results.tex:41`)을 "... for 2013, a year observed in all three regions" 로 고친다. 국문 두 곳도 같이 고친다. `fig6_product_maps.py:2, 367` 의 문구와 JSON `why_this_year` 도 갱신한다(수치 변화 없음).

### F2 [P0] Fig. 2 캡션 a 의 "yearly maps use the full-season sum"

- 본문: `methods.tex:45` "... the model input is the sum accumulated to the observation day (TDD$_{obs}$) and the yearly maps use the full-season sum."
- 방법 절: `methods.tex:22` "Yearly maps use the TDD accumulated to 1 September (day 244)"; `methods.tex:72` "with the TDD accumulated to 1 September". 지도 메타 `maps/{Alaska,Lena,Canada}/meta.json` `ref_doy: 244`, `ref_doy_info.mean_fraction_of_full_season` 0.917 / 0.918 / 0.903. 그림 스크립트 `fig2_workflow.py:492` 는 "To 1 Sep (maps)" 표지를 그리고 `:649` 는 "maps use the TDD accumulated to 1 September in place of TDD_obs" 라 기록한다. JSON `Fig2_workflow.json` `a.curve.tdd_to_1_sep_degC_day` 577.83(전체 계절 628.76, 관측일 471.79).
- 국문 `ko_v13/sections/methods.tex:45` 는 "연도별 지도는 9월 1일까지 누적한 값을 쓴다" 로 이미 고쳐져 있다. 영문만 낡았다.
- 수정: `methods.tex:45` 의 "the yearly maps use the full-season sum" 을 "the yearly maps use the sum accumulated to 1 September" 로 바꾼다. `fig2_workflow.py:12–13` 머리말("filled circle = full-season TDD (maps)")도 갱신한다.

### F3 [P1] 보충 PDF 가 S8 · S9 최종 파일보다 오래되었다

- 시각: `paper/manuscript/en_v13/si.pdf` 15:01:15(빌드 로그 `build/si.log` 15:01:15). `outputs/figures/paper/v13_si/FigS8_map_detail.pdf` 15:22:17, `FigS9_correction_maps.pdf` 15:22:53(PDF 내부 CreationDate 20261020152215 / 152252). `SI_figures.json`(15:22) 의 S8 · S9 `outputs.bytes` 는 현재 파일과 같다.
- 납품: `deliverables/14_paper_v13_2026-10-18/supplementary_en.pdf` 의 md5 가 `si.pdf` 와 같다(1df863fc…). 본문 PDF 는 15:23 빌드로 Fig. 6(15:20) · Fig. 7(15:21) 이후다.
- 영향: 12절 기록대로 15:20 이후의 재렌더는 해안선 · 경위선 단순화(`c13_maps.coast(tol_m)`, `graticule` 120점)로 보이며 수치는 바뀌지 않았다. 그래도 보충 PDF 에 들어간 S8 · S9 는 파일 목록의 S8 · S9 와 다른 판이다.
- 수정: `cd paper/manuscript/en_v13 && ./build.sh si`(국문도 동일) 뒤 `tools/package_paper_v13.sh` 재실행. 재빌드 뒤 `build/si.log` 시각이 15:22 이후인지 확인한다.

### F4 [P1] Fig. 3 캡션의 블록×연도 최소 행 수

- 본문: `results.tex:9` "the year and block-by-year designs score only years with at least 20 site-years."
- 코드: `testbed_v5.py:189` `folds_year(d, min_test=20)`; `:200` `folds_block_year(d, k=5, seed=0, min_test=10)`. 예측 파일 확인: 알래스카 블록×연도 겹 44개의 최소 행 수 10, 레나 12, 캐나다 21(`core_v6/preds/*__block_year__year_obs__F8TS__P1__-__lam-1.csv.gz`). 방법 절 `methods.tex:48` 은 "(folds with at least 10 site-years)" 로 맞다.
- 수정: 캡션을 "the year hold-out scores only years with at least 20 site-years and the block-by-year design only folds with at least 10" 로 고친다.

### F5 [P1] S5 캡션 b 의 "covariate range"

- 본문: `si.tex:127` "(b) RMSE by sub-region, at the sites inside the covariate range of the training data, ...". `si_s05_external.py:6, 137` 도 같은 문구.
- 코드: `v9_extra_agg.py:54` `cell_mean = w.groupby("loc_id").y.transform("mean"); w["in_range"] = cell_mean <= 300.0  # 학습 라벨 범위(0–300 cm) 안 셀`. 즉 라벨(관측 ALT)의 셀 평균이 300 cm 이하인 셀이다. 제외되는 것은 스발바르 N5(셀 평균 594 cm, 11행) 하나이며, 방법 절 `methods.tex:61` 은 이를 "One Svalbard borehole with a mean ALT of about 6 m, outside the training range" 로 맞게 적었다.
- 수정: 캡션을 "at the sites whose mean observed ALT lies within the range of the training labels (one Svalbard borehole of about 6 m excluded, as in Methods)" 로 고친다.

### F6 [P1] S6c 가 전체 계절 2014 지도를 그린다

- 본문: `si.tex:128` "(c) Mean difference between each product and our Alaska map of 2014 by latitude band, for cells at or below and above 600 m elevation (every tenth 1 km cell)."
- 원천: `si_s06_training_sites.py:229` 가 읽는 `data/processed/v9_qa/map_band_alaska2014_by_elev.csv` 는 10-15 17:58 에 `v9_qa_checks.py:58` 이 당시의 `maps/Alaska/pred_2014.csv.gz`(전체 계절 TDD, 지금은 `maps_fullseason_2026-10-14/Alaska/`)에서 만든 표다. 현재 `maps/Alaska/pred_2014.csv.gz` 는 10-20 13:51 의 9월 1일 TDD 지도다. 그림 파일 `FigS6_training_sites.pdf` 는 10-18 21:37 판이다.
- 본문 수치: `results.tex:39` 의 산지 차이(`\nmapAkFootMtnWei` +18.7, `\nmapAkFootMtnLiu` +52.9)는 `v13_numbers.py:90–95` 가 Fig. 6 JSON(2013, 9월 1일 지도)에서 다시 정의한 값이다. S6c(2014 전체 계절 지도)의 같은 띠 값은 +21.9 / +55.2(`numbers_v9.tex`)다. 본문은 Fig. 6 만 인용하므로 모순은 아니지만, S6c 는 더 이상 존재하지 않는 지도의 그림이다.
- 수정(택일): (a) `v9_qa_checks.py` 3절(지도 띠 비교)을 현재 지도로 다시 돌려 `map_band_alaska2014_by_elev.csv` 를 갱신하고 S6 를 다시 그린다. (b) S6c 를 Fig. 6 JSON 의 `band_differences`(2013) 로 바꾸어 본문 수치와 같은 표를 그린다(권장: 본문 · 보충이 한 원천). 어느 쪽이든 캡션에 "TDD to 1 September" 를 적는다.

### F7 [P1] S10 의 "end of season" 은 전체 계절 TDD 외삽이다

- 본문: `si.tex:132` "the physics-residual model at the observation day and at the end of the season, the recalibrated Stefan model at the end of the season".
- 코드: `v5_site_series.py:3` "관측 없는 해는 연간 TDD 로 끝-계절값"; `:38` `p1_eos = E * g.e5_sqrt_tdd.values`(격자 `forcing_year.csv` 의 1–12월 합), `:46` `r1_eos = p1_eos + lam * r_res`; 학습은 `with_forcing(df, 'year_obs')`(`:25`). 즉 관측일 TDD 로 적합한 E 와 잔차 학습기에 전체 계절 TDD 를 넣는다. 이는 `AUDIT_EXPERIMENT_SETTINGS` F1 이 지적한 공변량 이동(관측일 대비 전체 계절 TDD 비 1.37 · 1.23 · 1.13)이며, 지도는 그 때문에 9월 1일 TDD 로 바꿨다(TDD244/TDDobs 중앙값 1.23 · 1.11 · 1.06).
- 수정(택일): (a) `v5_site_series.py` 에 `--ref-doy 244` 를 더해(`v5_map_predict.py:90–105` 와 같은 보간) 재실행하고 S10 의 선을 "1 September" 로 바꾼다(로컬 CPU, 학습 3씨앗 × 5겹 × 3지역). (b) 캡션에 "end of season = full-season TDD of the year, which exceeds the training range of the observation-day input (see Methods); the maps use the TDD to 1 September" 를 적는다.

### F8 [P2] 그림 3c 의 부트스트랩 반복 수

- 본문: `methods.tex:53` "$B=2000$ (1000 times for the label-number experiments and for the covariate-set and learner sensitivity runs)".
- 실행: `run_v6.sh:41` core_v6 2000, `:42` featsets_v6 1000, `:43` confirm/blk 1000, `:44` rf_v6 1000; `run_v6b.sh` modis_v6 2000(`AUDIT_EXPERIMENT_SETTINGS` 1절). `fig3_accuracy.py:105–114` 는 F8TS · F8TS_M · F8TS_ALL 을 modis_v6(2000), F8 · F8T · F8TS_X 를 featsets_v6(1000)에서 가져온다(JSON `c.row_source`).
- 수정: 문장을 "1000 for the label-number experiments, the learner sensitivity runs and the covariate sets F8, F8T and F8TS_X; 2000 for the MODIS covariate sets" 로 하거나, featsets_v6 를 `v5_aggregate.py --nboot 2000` 으로 재집계한다(평가만, 수 분).

### F9 [P2] 관측일 사분위와 전체 계절 초과율

- 본문: `methods.tex:22` "(interquartile range of observation days 224--243), whereas the full-season sum would exceed it by 20--30\%".
- 자료(in_forcing == 1, `obs_doy` 가 있는 지점-연도): 알래스카 중앙값 224, IQR 224–230(14,929행; CALM 611행은 날짜 없음); 레나 234, 229–241; 캐나다 237, 234–243(1,131행; CALM 65행 날짜 없음). 전체 IQR 224–234. 관측일 TDD / 전체 계절 TDD 중앙값 0.73 · 0.81 · 0.89(`\ntddfrac*` 와 같음) → 전체 계절이 관측일 값을 넘는 비율 중앙값 37 % · 23 % · 12 %. 9월 1일 TDD / 전체 계절 0.91 · 0.88 · 0.89.
- 수정: "regional interquartile ranges of the observation day span days 224--243" 로 바꾸고, 초과율은 "by 12--37\% (median per region)" 또는 매크로(`\ntddfrac*`)로 쓴다.

### F10 [P2] 윤년의 244일

- `v5_map_predict.py:96–102` 와 `testbed_v5.py:113, 118–122` 는 모든 해에 244일을 쓴다. 윤년(2012 · 2016 · 2020 · 2024)의 244일은 8월 31일이다. 본문 `methods.tex:22, 72`, `results.tex:48` 의 "1 September" 는 평년에만 맞다.
- 수정: 첫 언급에 "(day of year 244, 1 September in non-leap years and 31 August in leap years)" 를 덧붙인다.

### F11 [P2] 관측일 보간의 달 번호(윤년)

- `testbed_v5.py:118` 은 달 번호를 평년 누적일 `[0, 31, 59, 90, ...]` 로만 찾고, `:117, 120` 의 달 시작일 · 일수는 윤년을 반영한다. 윤년에는 각 달의 마지막 날(예: 8월 30일 = 243일)이 다음 달로 분류되어 `frac` 이 0 으로 잘리고 그 달 말까지의 누적값이 쓰인다.
- 크기(19,172행 정확 계산과 비교): 평년 오차 0; 윤년 3,735행 중 64행에서 1 °C d 초과, 최대 8.6 °C d, 상대 오차 최대 1.1 %. 결과에 영향 없으나, 재현 코드를 공개하기 전에 `np.searchsorted(st[i], doy, side="right") - 1` 형태로 행별 윤년 표를 쓰면 없어진다.

### F12 [P2] 전이 사다리 n = 0 의 씨앗 수

- `v6_label_ladder.py:100–101` `if n == 0 and s != seeds[0]: continue`. `ladder_metrics.csv` TL n_label 0 행의 `rmse_seed_sd` 는 NaN(씨앗 1개). 그림 4b 와 `\ntlrmse*zero` 매크로(레나 D0S 39.5, R1S 23.8, 알래스카 R1S 17.8, 캐나다 D0S 24.2 · R1S 24.3)는 씨앗 0 단독 값이다. 외부 적응(EXTFT) n = 0 은 세 씨앗을 쓴다(`v9_extra.py:135–137`).
- 수정: `methods.tex:43` 또는 그림 4 캡션에 "(n = 0 of the transfer experiment: one seed)" 를 적거나 씨앗 3개로 재실행한다(원천 전량 학습 2회 추가, Rescale).

### F13 [P2] ALLena 라벨 날짜

- `build_v5_labels_cell_year.py:113–114`: 위치·연도별 `alt_cm = max`, `obs_doy = mean`. 레나 지점-연도 2,436 중 `n_obs > 1` 이 384(15.8 %), 그중 8월과 9월에 걸친 것 19. 이 행에서 라벨(최댓값, 대개 마지막 관측)과 TDD 누적일(평균일)이 어긋난다. ABoVE 는 평균값 · 평균일이라 일관된다.
- 본문 `methods.tex:5` 는 규칙을 그대로 적었으므로 오류는 아니다. 보충 S1 에 한 줄 비고를 권한다.

### F14 [P2] 자료 가용성의 지도 범위

- `front.tex:18` "the yearly 1 km maps of Alaska and the Lena Delta (2011--2024) will be deposited". 캐나다 지도가 추가되었다(`maps/Canada/meta.json`, 1,151,169 셀, `results.tex:48`).
- 수정: "of Alaska, the Lena Delta and north-western Canada".

### F15 [P2] Yi and Kimball 의 역할

- `table1.tex:30` "Yi and Kimball ... 2010--2015 (Alaska) & Comparison". `fig5_products.py:36–41` 의 제품 행은 Wei · Liu · CCI v4 · CCI v5 · Aalto 5종이고 `v10_si_tables.py:59–61` 의 표 S5 도 같다. `front.tex:11` "five published products". Yi and Kimball 값은 추출(`products_cell_year_v5_meta.json` yk 2001–2015)만 되고 채점되지 않는다.
- 수정: 표 1 역할을 "Not scored (period only, Fig. 1c)" 로 적거나 행을 뺀다. `methods.tex:64` 의 "Six products were sampled" 는 유지 가능하다.

### F16 [P2] 보충이 numbers_v13 을 읽지 않는다

- `main.tex:30` 은 `shared_v13/numbers_v13.tex` 를 읽고, `si.tex:7`(국문 `ko_v13/si.tex:7`)은 v6 · v9 · v10 · v11 만 읽는다. `numbers_v13.tex:33–56, 68–127` 은 `\providecommand + \renewcommand` 로 84개 매크로를 정의 또는 덮어쓴다(`\nfe*`, `\nmapAk*`, `\nanobs*`, `\nncellmap*`, `\naoa*`); 그중 본문에 쓰이는 것이 40개다.
- 현재 영향 없음: 보충 본문 · 표에 쓰인 매크로 32개(`\nqc*`, `\next*`, `\nyrs*`, `\ndistwei*`)는 모두 v13 이 덮어쓰지 않는 것이다(3.2절 표). 그러나 보충 캡션에 `\nfeAkN` 같은 매크로를 쓰면 v11 의 95(2014) 가 인쇄된다.
- 수정: `si.tex:7` 끝에 `\IfFileExists{../shared_v13/numbers_v13.tex}{\input{../shared_v13/numbers_v13}}{}` 를 더한다(국문 동일).

### F17 [P2] 낡은 주석 · 기본값

- `v5_map_predict.py:21` `--run` 기본 core_cb, `:23` `--forcing` 기본 year, `:3–4, 8` "core_cb" 주석. 실제 실행은 `--run data/processed/v5/runs/core_v6 --forcing year_obs --ref-doy 244`(`maps/*/meta.json run · forcing · ref_doy`). 기본값으로 다시 돌리면 다른 지도가 나온다.
- `fig2_workflow.py:12–13` 머리말 "filled circle = full-season TDD (maps)"; `v13_numbers.py:1–2` 머리말(Fig 5 b · c 만 적음); `fig6_product_maps.py:2, 55, 367`, `Fig6_product_maps.json` `why_this_year`(F1).
- 수정: 기본값을 실행값으로 바꾸고 머리말을 갱신한다. Code availability(`methods.tex:83`)에 지도 명령을 적는다.

### F18 [P2] 전체 계절 지도에서 만든 매크로 원천

- `v9_numbers.py:69–77` 은 `outputs/figures/paper/v7/source_data/F8_product_maps.json`(10-15, 2014 · 2016 전체 계절 지도)에서 `\nfe{Ak,Le}{Ours,Oof,Wei,Liu}` 를, `:56–59` 는 `v9_qa/map_band_alaska2014_by_elev.csv` 에서 `\nmapAk*` 를 만든다. `v10_numbers.py:87–94` 는 `F7_series_significance.json`(10-15) 에서 `\nanobs*` 를 만든다. `v11_numbers.py:16, 31` 은 `paper/v11/source_data` 의 Fig 6 · 7 JSON(2014 · 2016) 을 읽는다.
- 본문에 쓰인 해당 매크로는 모두 `numbers_v13.tex` 가 마지막에 덮어써 현행 값이다(3.2절 표의 "main != si" 40건이 그 목록). `docs/numbers_v9.json` · `numbers_v11.json` 은 옛 값으로 남아 있다.
- 수정: v9 · v10 · v11 생성기에서 지도 유래 블록을 v13 JSON 으로 옮기거나 제거해, 덮어쓰기에 의존하지 않게 한다.

---

## 3. 질문별 검증 결과

### 3.1 연도별 입력 · 출력 분리와 모델 입력 관리

**라벨과 기후 입력의 연도 · 관측일 정합**
- `testbed_v5.py:54–55` `load_v5` 는 `forcing_year.csv` 를 `(loc_id, year)` 로 라벨에 붙인다(라벨 자신의 해). `:109–123` `with_forcing('year_obs')` 는 그 해의 월말 누적 `tdd_cum_1..12` 를 `obs_doy` 까지 선형 보간해 `e5_tdd` · `e5_sqrt_tdd` 를 바꾼다. 날짜 없는 CALM(알래스카 611행, 캐나다 65행)은 244일(`:113`). 라벨 19,172행 모두 `usable_mask`(`:150–152`) 통과.
- `build_v5_forcing_year.py:116–131`: 나머지 7개 기후 변수는 달력년(maat · twarm · stl1 · tdd_soil), 전겨울 10–4월(fdd · tcold), 수문년 10–9월(swe)이며 본문 `methods.tex:22` · 표 S3 와 같다. 관측일 이후의 달이 포함되는 변수(maat · twarm · stl1)는 본문이 "computed for each cell and year" 로 적어 규칙은 명시되었다.
- 모든 실행이 같은 적재 경로를 쓴다: `v5_compare.py:51, 65`, `v6_label_ladder.py:35–36`, `v9_extra.py:41–42`, `v5_map_predict.py:43–44`, `v5_products_eval.py:34`, `v5_common_cells.py:18`.

**분할의 누설 여부(예측 파일 재검증)**
- 예측 파일 열: `region, scheme, fold, forcing, featset, method, learner, lam, seed, cell_year_id, loc_id, year, block, y, pred`(`v5_compare.py:98–101`). 사다리 · 외부 파일도 `fold, n, n_label, seed, year, block`(+ `subregion`) 을 가진다.
- 블록 겹: 라벨에서 `folds_block(k 5, seed 0)` 를 다시 만들어 `core_v6` P1 예측의 `fold` 와 비교, 불일치 0행(15,540 / 2,436 / 1,196). 블록이 두 겹에 걸친 경우 0. 겹 크기 알래스카 4,797 · 2,619 · 2,618 · 2,618 · 2,888(행 수 균형 배정, `testbed_v5.py:164–176`).
- 연도 홀드아웃: 겹 `y<yyyy>` 의 행은 모두 그 해이고 학습은 `y != yy`(`:196`). 채점 연도 알래스카 15개 전부(최소 겹 29행), 레나 2012–2021(2,419행, 17행 미채점), 캐나다 2012–2024 중 9개(1,164행).
- 블록×연도: 겹마다 연도 하나, 블록 부분이 블록 겹과 일치, 학습은 `(f != i) & (y != yy)`(`:208`). 최소 겹 10 · 12 · 21행.
- 지역 홀드아웃: 시험 = 지역 전량(15,540 / 2,436 / 1,196), 학습 = 다른 두 지역에서 시험 지역 셀로부터 100 km 안 제외(`:212–224`). 전이 사다리는 완충 없음(`v6_label_ladder.py:91`, 본문 `methods.tex:61` 명시).

**연도별 지도**
- `maps/{Alaska,Lena,Canada}/meta.json`: `years 2011–2024`, `forcing year_obs`, `run core_v6`, `ref_doy 244`, `ref_doy_info.note` "e5_tdd and e5_sqrt_tdd of the grid replaced by the TDD accumulated to the reference day; other climate inputs unchanged", 생성 13:16 / 13:51 / 14:32. 격자 강제력 `grid/*_forcing_year.csv`(10-20 13:08–13:42)에 `tdd_cum_1..12` 열이 있다. `v5_map_predict.py:108–116` 은 해마다 그 해의 격자 강제력 행을 붙여 예측한다. E = 1.949 / 1.679 / 1.746 는 라벨 전량 적합값과 일치. 구간 원천 `*__block__year_obs__F8TS__R1__catboost__lam0.5.csv.gz`(`:59–64`), AOA 는 2011–2024 평균 입력(ref-doy 반영, `:123–126`).
- 옛 지도는 `maps_fullseason_2026-10-14/`(Alaska · Lena 만)에 있고, v13 그림 · 매크로 가운데 이를 읽는 곳은 없다. 다만 10-15 에 그 지도에서 파생된 표를 읽는 곳이 S6c(F6)와 v9 · v10 · v11 생성기(F18)다.

**제품 값의 연도**
- `build_v5_products_cell_year.py:88–90` 은 (loc_id, product, year, value_cm) 긴 표를 만들고, `testbed_v5.py:87–99` `product_wide` 가 라벨의 (loc_id, year) 로 같은 해 값을 붙인다(`prod_<p>`), 다년 평균은 `prod_<p>_mean`. 기후값(clim) 판에서만 평균으로 바꾼다(`:128–131`). 제품 비교 · 공통 셀 · 연도별 · Fig. 6(`fig6_product_maps.py:103–104`) · 외부(`v9_extra.py:118–119, 133`) 모두 관측 연도 값을 쓴다.

**데이터 관리 비고**
- 표 생성 시각: 라벨 `labels_cell_year_v5_meta.json` 10-14 14:31, 강제력 `forcing_meta.json` 10-12 14:37(셀 17,029, fail 0), 제품 메타 10-10, 외부 10-15 17:41. 라벨이 강제력보다 뒤에 다시 만들어졌지만 셀 집합(static_v5 17,029)이 같아 19,172행 전부 강제력이 있다(usable 19,172).
- 핵심 실행 `core_v6` 결과 표(10-14 16:43–17:24) 는 라벨 표(10-14 14:31) 이후다. 사다리 10-15, 외부 10-15/16.

### 3.2 본문 · 그림 · 실험 수치 정합

**매크로 추적 결과(스크립트 집계)**
- 본문 7개 절 + si.tex + si_tables_en.tex 에 쓰인 매크로 263개, 미정의 0. 여러 파일에 정의된 것 40개는 모두 `numbers_v13.tex` 의 덮어쓰기이며 본문(main)에서는 v13 값이 유효하다. 보충에 쓰인 32개는 모든 파일에서 같은 값이다(F16).
- 생성 순서와 시각: 지도 13:16–14:32 → `numbers_v10.tex` 14:59:13 → `numbers_v6.tex` 14:59:21(= `en_v6/numbers.tex`, diff 없음) → `numbers_v13.tex` 14:59:23 → `main.pdf` 15:23. `numbers_v9.tex`(10-16 14:05, = `outputs/report_v9/numbers_v9.tex`) 와 `numbers_v11.tex`(10-16 18:42)는 지도와 무관한 값만 본문에 쓰이거나 v13 이 덮어쓴다. git diff 로 본 v6 · v10 의 변경은 `−0.0`/`+0.0` → `0.0` 표기와 캐나다 지도 매크로(`\naoaCa` 43, `\nhwCa` 41.3, `\nncellmapCa`, `\nanminCa` …), 새 지도의 `\nanmin/max*` 뿐이다.

**본문 수치 → 원천 표 → 그림 원천 JSON 대조(모두 일치)**

| 본문 위치 · 매크로 | 원천 | 그림 원천 | 판정 |
|---|---|---|---|
| `results.tex:5` `\ndoymed*` 224/234/237, `\ntddfrac*` 0.73/0.81/0.89 | `v10_numbers.py:64–72`, 라벨 · `with_forcing` | Fig. 2a JSON tdd_obs 471.8/628.8 | 재계산 일치 |
| `results.tex:7, 15` `\ndforc*`, `\nrmse*blk*`, `\ndref*blk*` | core_v6 `deltas_forcing.csv` · `summary_metrics.csv` · `deltas_vs_ref.csv`(year_obs, F8TS, block, λ 0.5) | Fig3 JSON a · b `plotted_rmse_cm` | 일치 |
| `results.tex:12` `\nrmseLeregDz` 39.3, `\nrmseLeregPz` 21.5, `\noptim*` | 같은 표, region · random | Fig3 b `printed_numbers` 39.3 | 일치 |
| `results.tex:15` `\ndfeat*` | modis_v6(F8TS · M · ALL) / featsets_v6(F8 · F8T · F8TS_X) | Fig3 c `row_source` | 일치(B 혼재는 F8) |
| `results.tex:19` `\ngc*`, `\ndtddLewarm` | `v7/source_data/F4e_gain_conditions.csv`(core_v6 preds, 2000회, ≥30 · ≥5 블록), core_v6 `conditions_tdd.csv` | Fig3 d 같은 CSV | 일치 |
| `results.tex:22` `\nll*` | ladder_v6 `ladder_deltas.csv`(LL, ref P1, 1000회) | Fig4 a | 일치(R1 n 25: −0.85 [−1.87, +0.15] …) |
| `results.tex:26` `\ntlrmse*` | `ladder_metrics.csv` TL `rmse`(씨앗 평균 예측) | Fig4 b values(레나 D0S 0: 39.48, R1S 25: 19.9) | 일치(n = 0 단일 씨앗은 F12) |
| `results.tex:28` `\nft*`, `\nftd*`, `\nftScored*`, `\nftdCciRows` | `v7/source_data/F10_extft.csv`(= extft 재집계, 474행), `extft_deltas.csv`(2000회), preds_EXTFT | Fig4 d · e(474행; 4e 6 소지역) | 일치(캄차카 미표시 명시됨) |
| `results.tex:31` `\ncc*all`, `\nnprodLe*mask`, `\ndprodLe*Romask` | core_v6 `common_cells_eval.csv` mask 0, `products_eval_mask25.csv`(vs wei n 802, vs cci5y n 2,352) | Fig5 a(461 / 449) | 일치, 행 집합 본문 명시됨 |
| `results.tex:33` `\nrLe*`, `\nbias*`, `\nthinN*`, `\nccLeN` | `v13_numbers.py:9–20` ← Fig5 JSON b · c | 같은 JSON | 일치(b n 461 = common_cells 461) |
| `results.tex:37` `\ndistwei*`, `\nbiasLeNorth*`, `\ntlvp*` | `v9_qa/qa_summary.json`, `lat_band_bias.csv`, `transfer_vs_products.csv`(지도 무관) | S6 b | 일치 |
| `results.tex:39, 41` `\nfe*`, `\nmapAkFootMtn*` | `v13_numbers.py:70–96` ← Fig6 JSON(2013, 9월 1일 지도, OOF) | 같은 JSON(126 / 6 / 33 셀, 59 / 2 / 6 within 25 km) | 일치(S6c 와의 차이는 F6) |
| `results.tex:43` `\nyw*`, `\nrdelta*` | `v7/source_data/F5_yearwise_rmse.csv`(mask 0, shown, ≥30), core_v6 `yearwise/multiyear_changes.csv` | Fig5 d years_shown | 일치 |
| `results.tex:45` `\npadLe*` | extra_v9 `pa_deltas.csv` | S7 | 일치 |
| `results.tex:48` `\nanmin/max*`, `\nanobs*`, `\ntrend*`, `\naoa*`, `\nhw*`, `\nncellmap*` | v10(지도 pred 파일 평균), v13(Fig7 JSON d · e · a_b_c), v6(meta), v10(meta) | Fig7 JSON: 알래스카 pred 2011 −2.818 · 2019 +5.984, 레나 2016 −8.91 · 2020 +8.955, 캐나다 2020 −3.817 · 2023 +6.356; obs 연도 12 / 10 / 7, r 0.13 / 0.73 / 0.64; trend +2.07 / +6.68 / +1.54, 양 78 / 100 / 91 % | 일치 |
| `methods.tex:5`, 표 1 `\nncy*`, `\nncell*`, `\nnblk*`, `\nnkm*`, `\nnmulti*`, `\ntopyr*` | v6(라벨), v13(0.009° 셀 규칙 `v13_numbers.py:58–60` = `v7_f6_maps.py:138–144`) | Fig7 observed_cells_drawn 321 / 143 / 71 | 일치 |
| `methods.tex:61`, 표 1, S3 `\nextcy` 816, `\nextcells` 73, `\nftScoredRows` 805 / 72 / 49, `\nextRawSy` 857 | `v9_ext/build_meta.json`, preds_EXTFT(스발바르 N5 11행 제외) | Fig4 d `all_external_cell_years` 805 | 일치 |
| 표 S4 · S5 | `v10_si_tables.py` ← core_v6 `summary_metrics.csv` · `common_cells_eval.csv`(10-14) | Fig3 b, Fig5 a | 일치 |

**정의와 계산이 다른 곳**: F1(연도 선택 근거), F2(Fig. 2 캡션), F4(블록×연도 최소 행), F5(in_range 정의), F8(B), F9(IQR · 초과율), F12(n = 0 씨앗). 그 밖에 본문 수치와 Methods 정의가 어긋나는 곳은 찾지 못했다.

### 3.3 실험 설계

| 항목 | 코드 | 본문 | 판정 |
|---|---|---|---|
| 블록 겹 | 0.5° 블록 `floor(lat/0.5)·1e5 + floor(lon/0.5)`(`build_v5_labels_cell_year.py:128`), 행 수 균형 탐욕 배정(`testbed_v5.py:164–176`), k 5, 씨앗 0(`v5_compare.py:27`, `v6_label_ladder.py:62, 92`, `v9_extra.py:68`) | `methods.tex:48` | 일치, 예측 파일로 재검증 |
| 짝지은 블록 부트스트랩 | `testbed_v5.py:415–432`(다항 블록 가중, 같은 행), core_v6 · 제품 · 외부 2000(`run_v6.sh:41`, `ingest_v6.sh:28–29`, `v9_extft_agg.py:12`), 사다리 1000(`v6_label_ladder_agg.py:11, 19–26` 블록 재추출) | `methods.tex:53` B = 2000 / 1000 | 일치(공변량 집합 혼재는 F8) |
| 라벨 수 사다리 | 학습 겹에서 n 행 무작위 추출 `tr_full.sample(n, random_state=s)`(`v6_label_ladder.py:70`), 같은 n 으로 `fit_phys(tr)`(`:77`), 시험 겹 전체 채점, 추출 3회 = 씨앗 0 · 1 · 2, 채점은 세 추출의 예측 평균(`v6_label_ladder_agg.py:35`) | `methods.tex:61`, Fig. 4 캡션 "three draws whose predictions are averaged before scoring" | 일치 |
| 전이 | 원천 = 다른 두 지역 전량(완충 없음, `:91`), 표적 n 은 표적 블록 겹의 학습 겹에서 추출(`:102`), `En` 은 n ≥ 3 이면 표적 n 으로(`:105`), 잔차 학습 앵커 = 원천 행 E0 · 표적 행 En(`:118–119`), 시험 앵커 En | `methods.tex:61` | 일치 |
| 외부 CALM | 2010–2024, 알래스카 · 캐나다 · 스위스 · 카자흐스탄 제외(`build_v9_external.py:33`), 검열값 1 · 주석 3 제거(`:34–39`), v4 셀 1 km 안(`:47`), 연구 라벨 100 km 안 제외(`:51–53`, 사모일로프 · 비코프스키 3곳), 셀·연도 평균(`:55`), 0.5° 블록(`:61`); 소지역별 블록 하나씩 제외, 그린란드는 셀(블록 1개, 셀 2개; `v9_extra.py:140`), n ∈ 0·5·10·25·50·all 을 같은 소지역의 다른 블록에서(`:142, 157`), 셀 평균 ALT ≤ 300 cm(`:129`; 스발바르 N5 11행 제외 → 805 / 72 / 49); 소지역별 행 수 W 시베리아 218, NE 시베리아 244, C 시베리아 81, 추코트카 75, 유럽 러시아 73, 스발바르 65, 그린란드 30, 캄차카 19 | `methods.tex:61`, S3, 표 1, Fig. 4 캡션 | 일치 |
| 제품 누설 마스크 | Wei 학습 지점(보충표 ALT_variables.xlsx 좌표)까지의 최근접 거리 `xg_leak_cells_v1.csv`(product == wei, 17,643 셀), 라벨 19,172행 모두 거리 있음, 25 km 이하 제외(`v5_common_cells.py:19–21, 37`, `v5_products_eval.py:38–43`) | `methods.tex:64` | 일치 |
| 공통 지점-연도 | Wei · Liu · CCI v4 가 관측 연도 값을 모두 가진 행(`v5_common_cells.py:38`, `fig5_products.py:197`은 R1 OOF 유한 조건 추가, n 동일 단언 `:276–279`) | `methods.tex:64`, Fig. 5 캡션 | 일치 |
| Fig. 6 공통 셀 | 0.009° 셀(`fig6_product_maps.py:101–102`)에서 우리 지도 · Wei · Liu 값이 모두 있는 관측 셀(`:118`); 우리 계열 = `core_v6` 블록 CV OOF R1(씨앗 평균 뒤 셀 평균, `:105–108, 112`), 지도값은 `our_map_in_sample_not_shown` 으로만 기록(`:331`) | Fig. 6 캡션 `results.tex:41` | 일치(연도 선택 근거는 F1) |
| 지역 홀드아웃의 P0 ≡ P1 | `v5_compare.py:94` | Fig. 3 캡션 | 일치 |

---

## 4. 이전 감사 지적의 조치 확인

적용됨(코드 · 원고에서 확인): 그림 5b → 5d 참조(`results.tex:43`), 표 S2 · S5 캡션의 4c · 5b → 4d,e · 5d(`si.tex:73, 75, 127`), 초록 · 결론의 17.8 cm 통일(`\ndrefLeregDzabs`), L31 의 마스크 행 집합 명시(802 / 2,352), 캄차카 "not shown in Fig. 4e", `\sifig` 경로 v13_si, 라벨 `fig:s1`…`fig:s10`, 지점-연도 용어와 1 km 셀 수 병기(`methods.tex:5`, 표 1), 블록×연도 10(방법 절), R1cv 의 "with the full-season input", 세 추출 평균(Fig. 4 캡션), 전이 사다리 완충 없음, 지역 홀드아웃 P0 = P1(Fig. 3 캡션), Fig. 3c 실행 출처, S1c 창 밖 점, S10 별도 적합, Fig. 7d y 범위 상이, S5 b in_range 언급(단 정의 오류 F5), 지도 9월 1일 TDD 로 재생성(F1 of settings audit), Fig. 6 2013 × 3지역 · 한 색 범위 · 오차 축 ±60, S8 · S9 3지역, 캐나다 지도 매크로(`\naoaCa`, `\nhwCa`, `\nncellmapCa`), 자료 가용성 외 지도 관련 문구.

미적용 또는 부분 적용: featsets_v6 · rf_v6 · nn_v6 의 B = 1000 재집계(문장으로 대체, F8), S6c 의 2014 전체 계절 지도(F6), S10 의 전체 계절 외삽(F7), 지도 스크립트 기본값 · 주석(F17), 1 km 재집계(사용자 결정 보류, 본문은 지점 단위로 서술).

---

## 5. 일치로 확인한 항목(요약)

- 모든 실행이 라벨 표 19,172행(15,540 / 2,436 / 1,196), in_forcing == 1, 같은 `load_v5 → with_forcing('year_obs') → usable_mask` 경로를 쓴다. 예측 파일의 `year` · `block` · `fold` 가 설계와 일치하고, 블록 겹은 재계산과 0행 불일치, 연도 · 블록×연도 · 지역 겹의 시험 집합이 설계대로다.
- 지도 메타(ref_doy 244, 2011–2024, year_obs, core_v6) 와 Methods · 결과 · 표 S2 · Fig. 2b · Fig. 6 · Fig. 7 · S8 · S9 의 서술이 일치한다. 지도 유래 매크로(`\nanmin/max*`, `\nhw*`, `\naoa*`, `\nncellmap*`, `\ntrend*`, `\nanobs*`, `\nfe*`, `\nmapAkFootMtn*`)는 모두 새 지도 이후(14:59)에 생성되었거나 v13 이 덮어쓴다.
- 제품 값은 관측 연도 값이며, 공통 지점-연도(461 / 449 레나, 14,326 / 4,229 알래스카, 532 / 489 캐나다)와 표 S5 · Fig. 5 · 본문 매크로가 같은 표에서 나온다.
- Fig. 3 · 4 · 5 · 7 JSON 의 인쇄 · 기록 수치와 본문 매크로 사이에 반올림을 넘는 차이가 없다(3.2절 표).
- `main.pdf`(15:23) 는 원고 절(14:06–14:54) · 매크로(14:59) · 본문 그림(13:05–15:21) 이후 빌드이고, 본문 · 보충 빌드 로그에 미정의 매크로 0, 그림 경로는 모두 v13 · v13_si 이다. 납품 본문 PDF 는 `main.pdf` 와 md5 가 같다.
- 외부 CALM 의 QC · 완충 · 소지역 · 제외 규칙과 수(857 → 816 → 805), 누설 마스크의 거리 표 커버리지(19,172행 전부), 사다리의 추출 · 재보정 · 앵커 규칙, 부트스트랩 구현(짝지은 블록 재추출)이 본문 서술과 일치한다.
- 이전 감사에서 수정된 본문 참조 · 캡션 · 용어가 현재 파일에 반영되어 있다(4절).
