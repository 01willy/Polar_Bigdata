# 그림 제작 계약 v7 (모든 그림 스크립트가 따른다)

## 공통 코드
- 스타일: `import polar.style_v5 as S` (`src/polar/style_v5.py`). 시작에 `S.setup(scale=...)`, 그림은 `fig = S.figure(w_mm, h_mm)`, 패널 문자 `S.panel_letter(ax, "a")`, 추가 축 묶기 `S.register("a", cax)`, 묶음 크롭 `S.register("ab", *S.PANELS["a"], *S.PANELS["b"])`, 저장 `S.export(fig, "F1_study_area", outdir=ROOT/"outputs/figures/paper/v7")`. `--slide` 인자면 `S.setup(scale=1.69)` 로 슬라이드판(outdir/slide/<stem>.png 와 패널 크롭 `<stem>__<letters>.png`)이 나온다. 예시 코드: `scripts/4_visualization/v6_figs.py`(f1, f2), `v6_maps.py`(drape, coast, scalebar, draw).
- 색·이름: `S.METHOD_HEX[m]`, `S.METHOD_LS[m]`, `S.REGION_HEX[r]`, `S.REGION_EN[r]`, `S.PRODUCT_HEX`, `S.NEUTRAL`, 지도 `scripts/4_visualization/paper_v3/maps_alt_v3.py` 의 `MV.CM_ALT`(oslo_r), `MV.CM_DIFF`(broc), `MV.pct_range`, `MV.sym_max`, `MV.colorbar`, `MV.ne_layers`.
- 글꼴: 라틴 FreeSans(토큰), 논문 7 pt(최소 5 pt), 패널 문자 8 pt 굵게. 선 0.6–1.2 pt. 위·오른쪽 spine 없음.
- 수치는 `docs/numbers_v6.json` 또는 결과 CSV 에서 읽는다. 손으로 적지 않는다.
- 실행: CPU 만. `OMP_NUM_THREADS=2 nice -n 10 python3 <script> [--fig N] [--slide]`. GPU·soffice 사용 금지.
- 크기: 논문 폭 180 mm(`S.W2`), 높이 ≤ 230 mm. 패널 간 wspace ≥ 0.35.

## 금지
- 그림 안 제목(suptitle, set_title 로 문장), 해석 문장, 수치 나열 텍스트, 범례 상자 테두리(frameon=False), 글씨 담은 사각형, 아이콘, 그라데이션, 그림자, 곡선·사선 화살표, 무지개 색표, 내부 코드명(core_v6, F8TS 같은 태그는 라벨에 쓰지 않는다. 방법 이름은 영어 서술형: "Stefan, recalibrated", "direct ML", "physics + residual ML").

## 자료 위치
- 라벨 `data/processed/v5/labels_cell_year_v5.csv`(in_forcing==1; loc_id, region, lat, lon, block, year, alt_cm, obs_doy, obs_months, source). 강제력 `data/processed/v5/forcing_year.csv`(loc_id, year, e5_tdd, e5_sqrt_tdd, tdd_cum_1..12 ...), `forcing_clim.csv`.
- 제품 `data/processed/v5/products_cell_year_v5.csv`(loc_id, product[wei,liu,cci4y,cci5y,yk,aalto], year, value_cm). 또는 `polar.testbed_v5.load_v5(PROC)` 의 prod_* 열.
- 결과 표(관측일 정합, 주 실행) `data/processed/v5/runs/core_v6/`: summary_metrics.csv(region, scheme[random,block,year,block_year,region], forcing[year_obs,year,clim], featset, method, learner, lam, rmse, bias, r, r_anom ...), deltas_vs_ref.csv(ref=P1; delta, lo, hi), deltas_forcing.csv(pair 'year_obs:clim'), conditions_tdd.csv(tdd_bin 0/1/2), common_cells_eval.csv(region, mask_km 0/25, series, rmse, n), products_eval_mask25.csv(series 'ours:R1:0.5', definition 'vs wei (year-matched)', delta_vs_product, delta_lo, delta_hi), yearwise/{yearwise_metrics.csv(series 'ours:R1:0.5:blockCV', year, bias, n), multiyear_changes.csv}. 공변량 사다리 `runs/modis_v6/deltas_vs_ref.csv`(featset F8,F8T,F8TS,F8TS_M,F8TS_ALL ...; 'featsets_v6' 에 F8,F8T,F8TS_X,F8TS_L), 학습기 `runs/rf_v6`, `runs/nn_v6`(learner rf/mlp/tabm/ftt). 주 설정: featset F8TS, learner catboost, lam 0.5(R1), forcing year_obs, scheme block.
- OOF 예측 `runs/core_v6/preds/<Region>__<scheme>__<forcing>__<featset>__<method>__<learner>__lam<λ>.csv.gz`(cell_year_id, loc_id, year, block, y, pred, seed).
- 지점 시계열 `data/processed/v5/site_series/site_series_year_obs.csv.gz`(loc_id, region, year, observed, y, p1_eos, r1_eos, r1_obs, d0_eos ...).
- 지도 `data/processed/v5/maps/<Alaska|Lena>/pred_<year>.csv.gz`(cell_id, lat, lon, year, e5_tdd, p1, r1), `mean_2011_2024.csv.gz`(cell_id, p1, r1, lat, lon, aoa_di, aoa_in), `meta.json`. 격자 정적 `data/processed/map_alaska/alaska_grid_x25_v1.csv.gz`(cell_id, lat, lon, dem_elev, dem_slope, sg_soc_0_5, sg_clay_5_15, pfr_mean, gray, land ...), 레나 `map_lena/lena_grid_x25_v1.csv.gz`. 음영 `map_<key>/hillshade_v1/shade_250m.tif`(EPSG 투영, `v6_maps.drape` 참조). 격자 연별 강제력 `data/processed/v5/grid/<key>_forcing_year.csv`(알래스카 3.4 GB: 필요한 열·연도만 chunk 로 읽는다).
- 영구동토 확률 `data/processed/cci_pfr_mean_1997_2021.nc`. ERA5-Land 월별 `data/raw/era5land/nh_monthly_2010-2024.nc`.

## 산출
- `outputs/figures/paper/v7/<stem>.{pdf,svg,png}` + `slide/<stem>.png` + `slide/<stem>__<panels>.png`.
- 스크립트: `scripts/4_visualization/v7_<name>.py`, 각 그림 함수 하나, `--fig`·`--slide` 인자.
- 완료 전 자체 점검: PNG 를 열어 글자 겹침·잘림·빈 패널·범례 가림을 확인하고 고친다. 결과 요약에 남은 결함을 적는다.
