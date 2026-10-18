# 2026-10-14 결정 사항과 작업 기록: 관측일 정합, 그림·덱·원고 전면 재작성, Rescale 재실행

작성 2026-10-14. 사용자 요청 16항목(10-14)에 대한 결정과 조치. 결과 수치의 정본은 `data/processed/v5/runs/<run>/` 표이며, 원고·보고서·덱의 수치는 `scripts/2_evaluation/v6_numbers.py` 가 만드는 매크로(`paper/manuscript/en_v6/numbers.tex`, `docs/numbers_v6.json`)에서만 가져온다.

## 1. 관측 월(일) 정합의 방향 (요청 2)

**결정**: 주 강제력을 '관측일까지 누적한 융해 도일'(`year_obs`)로 바꾼다. 그 해 전체 누적(`year`)은 보조, 2011–2024 평균(`clim`)은 기존 방식의 대조군이다.

근거와 심사자 질문에 대한 답
- 왜 월·일 단위 입력을 쓰는가: 같은 셀-연도에서 9월 관측은 8월 관측보다 약 10 cm 깊다(10-12 보강 분석). 전체 연도 누적 입력은 이 차이를 지점 특성으로 오인한다. 관측일까지의 누적 비(관측일/전체)는 알래스카 중앙값 0.73, 레나 0.81, 캐나다 0.89 다.
- 왜 월별 출력(월별 지도)을 내지 않는가: 라벨이 계절 말(8–10월) 한 번뿐이라 월별 출력은 검증할 수 없다. 지도는 '그 해 전체 누적'으로 계절 말 ALT 를 내고, 월별 누적 TDD(`tdd_cum_1..12`)는 입력 표에 두어 필요하면 계절 안 깊이를 낼 수 있게 한다(그림 2a 의 계절 안 Stefan 관계가 근거).
- 구현: `src/polar/testbed_v5.py` `with_forcing(df, "year_obs")` 가 월 누적값을 관측 DOY 로 선형 보간한다. 관측일이 없는 CALM 지점은 DOY 244(8월 31일). 격자 지도는 전체 연도 누적(`year`)을 쓰되 학습은 관측일 정합으로 한다(`v5_map_predict.py --forcing year_obs`).

## 2. 연도별 실험의 논리 점검 (요청 15)

- 입력이 연도로 분리되는가: 라벨 행마다 `year` 가 있고 강제력은 `forcing_year.csv` 의 (loc_id, year) 행을 합친다. 정적 공변량은 연도 공통. 제품은 라벨 연도의 값(`v5_products_cell_year`). 점검 `data/processed/v5/audit_logic_v5.md` 13/13 통과(10-12). core_v6 도착 뒤 `scripts/2_evaluation/v5_audit_logic.py` 를 다시 돌린다.
- 연도 홀드아웃: `folds_year` 는 관측 연도 하나를 통째로 채점한다. 블록×연도는 블록 겹과 연도 겹을 함께 뺀다.
- 지도의 연도: `maps/<Region>/pred_<year>.csv.gz` 는 그 해 격자 강제력으로만 예측하고 제목에 'ERA5-Land <year>' 를 붙인다(그림 6).
- 빠진 조합과 추가 실험: (a) 관측일 정합으로 전 실험 재실행(core_v6, featsets_v6, confirm·blk025·blk100·rf_v6, nn_v6) — Rescale CJsgU(CPU)·RkmTeb(GPU) 완료·반영. (b) MODIS 2010–2024(AppEEARS 재요청 6건 완료, `modis_cell_year_v5.csv` 260,760 행)로 사다리 재실행 — Rescale CECkU 완료·반영. (c) 지점 전 연도 시계열(`v5_site_series.py`, 관측일·연도 정합) — 완료(그림 5). (d) 지도 재생성(관측일 정합 학습) — 완료(그림 6). 논리 점검 `audit_logic_core_v6.md` 13/13 통과. 결과 표는 `docs/RESULTS_v6_2026-10-14.md`.

## 3. 결론 문장 (요청 13·14, core_v6 관측일 정합 기준, 수치는 docs/RESULTS_v6_2026-10-14.md)

- 레나 델타: 모든 연도·모든 제품보다 잔차 결합이 낫다(공통 셀 21.1 cm 대 Wei 34.6, Liu 33.0 cm; 짝지은 차이 Wei −12.5 [−15.6, −5.8]). 지지. 단, 다년 셀 230쌍의 연도 간 변화는 Wei(0.55)가 잔차 결합(0.29)보다 잘 따라간다.
- 알래스카: 전체 공통 셀에서 잔차 결합 17.1 cm 대 Wei 17.9, Liu 18.8 cm. 제품 학습 지점 25 km 밖에서는 15.3 cm 대 Wei 20.8, Liu 15.2 cm(Wei 대비 −5.2 [−12.0, +1.1], 구간이 0 을 포함). '누설을 뺀 뒤 Wei 보다 낫고 Liu 와 같다(경향)'로 쓴다.
- 캐나다: 잔차 결합 25.8 cm, 직접 학습 24.1 cm 대 Wei 27.0 cm(짝지은 차이 Wei −0.8 [−9.4, +1.3]). '같은 수준'.
- 결합의 조건: 지형·토양 입력이 있어야 하고(기후만: 알래스카 +5.9), 이득은 각 지역의 따뜻한 3분위에서 크다(레나 −6.2, 캐나다 중간 −4.3 cm). 10-12 의 '추운 셀' 판정은 관측일 정합에서 뒤집혔다.
- 전체: 물리 잔차 결합이 세 지역 모두에서 가장 안전한 선택(지역 안 레나 −3.6·캐나다 −3.7·알래스카 −0.2 cm, 새 지역에서 물리식 수준). 직접 학습은 새 지역에서 레나 +17.8 cm 로 무너진다.

## 4. 산출물 (요청 3–12, 16)

| 항목 | 위치 | 상태 |
|---|---|---|
| 논문 그림 F1–F6(300/600 dpi, pdf·svg·png) | `outputs/figures/paper/v6/` | F1·F2·F3·F4·F6 완성(F3 d·F5 는 지점 시계열 도착 뒤) |
| 슬라이드판·패널 크롭 | `outputs/figures/paper/v6/slide/<F>__<panel>.png` | `src/polar/style_v5.py` 패널 등록(`panel_letter`·`register`)과 `_crop_panels` |
| 방법 도식 5종(시간 정의·흐름·구조·유사라벨/잔차·평가) | `deck/assets/v6/method/V1–V5_*_slide.png` | `scripts/4_visualization/v6_method_figs.py`, 글자 QA 통과 |
| 덱 29쪽(초심자 대상, 내부 용어 없음) | `deck/deck_spec_v6.json`, `deck/build_v6_report.py` → `deck/render/v6_report.{pptx,pdf}` | R07·R08 대기 표지 |
| 국문 보고서 | `outputs/report_v6/report_v6.tex` → `.pdf` | 8쪽, 매크로 수치 |
| 영문 원고(Sci Rep) | `paper/manuscript/en_v6/` (`main.tex`, `sections/*.tex`, `numbers.tex`, `refs_v6.bib`) → `main.pdf` 20쪽 | 초록 200단어 안, [DECISION] 저자·저장소·사사 |

그림 규칙: `design/style_tokens_v4.json`(FreeSans 7 pt, 한글 Pretendard), 지도는 음영 드레이프(`v6_maps.py`), 적용 범위 밖은 흐리게. AI 티가 나던 개념도는 실자료 그림(그림 2a: 사모일로프 2015·쿠룽나흐 2020 계절 안 녹는 깊이)으로 대체했다.
