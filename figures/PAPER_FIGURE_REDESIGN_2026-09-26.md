# 논문 그림 재설계 스펙(Sci Rep 규격, 상위 저널 수준)

작성 2026-09-26, 개정 2026-09-26(심사 지적 필수 33건·선택 17건 반영, 파일 끝 '개정 이력'). 근거: 저널 관행 조사(Nature 계열 12편 41장), 현행 그림 비평(Fig2 원형·h3 8장·h2 4장), 스타일 시스템 초안(색각 시뮬레이션 포함), h4 공용 통계 규약(`src/polar/h4_common.py`). 기준 문서: `docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md` §3·§5·§7·§8. 이 문서는 스펙만 담고 코드·자료·계획서는 수정하지 않는다. 계획서에 반영할 사항은 §2.0 '계획서 갱신 대상'에 모은다.

자료 접두 규약: 완료 실험은 `data/processed/h3/`, `h2/`, `m1/`의 확정 열 이름을 쓴다. 진행 중 실험(A2·B1–B4·C1·C2·D1)은 `data/processed/h4/` 접두(`a2_`, `b1_`, …)를 쓴다. 본 실행 산출 파일명은 계획서 §8을 따르되, 스모크와 이름이 다르면 스모크 이름의 `_smoke` 제거형을 우선 가정한다(예: `b2_pooling.csv`, `b3_summary.csv`·`b3_evar.csv`, `c1_tests.csv`, `c2_coverage.csv`, `d1_summary.csv`). h4 실험의 지역 안 CI는 각 스크립트가 저장한 `<tag>_blocksse.npz`(`h4_common.save_stores`)에서 계산한다. 요약 CSV에 블록 CI 열(`blk_d_phys`, `blk_d_phys_lo`, `blk_d_phys_hi`, `split_win`, `rep_win`)이 없으면 `scripts/2_evaluation/h4_analysis.py`가 npz에서 재집계해 같은 이름으로 덧붙인다(재집계 항목 신설 필요, §6 단계 2).

---

## 0. 요약(현행 대비 변경점)

1. 생성 원천을 `paper_figs.py` 하나로 고정하고 h2·h3 스크립트는 보고서용으로만 남긴다. 스타일은 `src/polar/paperstyle.py`(신설 제안) 한 곳에서 정의한다. 방법 색 7종 + 참조 검정을 색각 시뮬레이션과 흰 바탕 대비(선·마커 ≥ 3:1)로 확정하고, 지역은 산점 패널에서만 마커로 구분한다.
2. 그림 문법을 저널형으로 바꾼다. 그림 안 제목·설명 문장·수치 열·matplotlib 표를 없애고, 패널 문자 굵은 소문자, 범례 그림당 1개, 물리식 기준선(0선)을 모든 Δ 패널에서 같은 실선으로 반복한다. 빈 마커는 한 그림 안에서 한 의미만 갖는다(§1.1).
3. 오차 표현을 통일한다. h4 자료의 지역 안 CI는 블록 부트스트랩(`h4_common.boot_delta_blocks`)만 그림에 그리고, (분할, 반복) 행 재표집 CI(`ci_rep`)는 Supplementary Table로 보낸다. 모든 CI에 출처 종류(`ci_kind`)를 붙이고, 종류가 다른 CI가 한 그림에 들어가면 캡션에 명시한다(§1.6).
4. 최소 라벨 수 n*는 사전 고정 주 정의(블록 CI 상한 < 0, 분할 승률 ≥ 2/3, 반복 승률 ≥ 0.75, `h4_common.min_n`)로만 쓴다. 미달성은 '> n_max (max tested)'로 표기하고 대입값을 그리지 않는다. |log E비|와의 관계는 달성/미달성 이진 서술과 `achieved_test`·`kendall_censored`로 보고한다.
5. 그림 역할을 재배치한다. Fig 3 = 라벨 예산 계단과 부분 풀링(H25·B2), Fig 4 = 교락 제거 최소 라벨 수(A2), Fig 5 = 라벨 배치(H29·B3), Fig 6 = 라벨 0 지역(H18–H23·H30·C1·C2), Fig 7 = 배포 절차와 확인적 검증(워크플로·B1·D1). 판정 전 가설(F1–F11)의 주장 문장은 '[F# 결과 확정 후 기재]' 자리표로 둔다.
6. 지도 3장(Fig 1a, Fig 5a·b)을 EPSG:3413 계열 극 입체 투영 + Natural Earth 해안선 + 위도선 + 축척 막대 + 면적 비례 원으로 재작성한다. Fig 1a 배경은 ESA CCI PFR 다년 평균 2단계 회색이다.
7. 인쇄 규격을 강제한다. 폭 180 mm 정확(bbox tight 후 ±1 mm), 높이 ≤ 200 mm, 글꼴 Arial → Liberation Sans 대체(DejaVu 임베드 금지), 최소 글자 6.5 pt. 패널 배치는 `axes_mm()` 절대 좌표로 하고, 슬롯 폭 = 좌측 라벨 여백 + 축 상자 폭으로 계산한다. 캡션은 그림별 단어 예산(§5.1)을 지킨다.

---

## 1. 스타일 시스템

### 1.1 방법 색(전 그림 고정)

| 키 | 방법 | HEX | L* | 흰 바탕 대비 | 선 | 마커 | 용도 |
|---|---|---|---|---|---|---|---|
| `phys` | 물리식 앵커(E0 고정, n = 0) | #4d4d4d | 33 | 8.5:1 | Δ 패널 0선 실선 0.7 pt, 절대 RMSE 패널 점선 `:` 0.7 pt | 없음 | 기준선·0선 |
| `ref_allA` | 전량 A 라벨 참조(E_own 기지 또는 전량 라벨 잔차 ML) | #000000 | 0 | 21:1 | 파선 `(0, (4, 2))` 0.7 pt | 없음 | 전량 라벨 참조(상한 아님) |
| `oracle_branch` | 사후 최선 분기(B1·D1 전용) | #000000 | 0 | 21:1 | 없음 | `*` | 사후 참조, Fig 7b·c만 |
| `direct` | 직접 ML(물리 없음) | #6b7280 | 48 | 4.8:1 | 실선 | `x` | Fig 1d, Fig 2a, Fig 6a |
| `refit` | E 재적합(대표 라벨) | #2b5c8f | 38 | 6.9:1 | 실선(무작위 라벨은 파선) | `o` | Fig 3, Fig 5 |
| `augment` | 물리 의사라벨 증강 | #2e6b2e | 40 | 6.4:1 | 일점쇄선 `-.` | `^` | Fig 2a, Fig 3a–d |
| `residual` | 물리 앵커 + 잔차 ML | #9a7bc9 | 57 | 3.5:1 | 실선 | `D` | 주 결과 |
| `tfm` | 표형 파운데이션 모델(잔차 예측기) | #ad921a | 61 | 3.0:1 | 실선 | `s` | S7 |
| `cci` | CCI 결합 | #568f72 | 55 | 3.8:1 | 실선 | `v` | Fig 6a·c |

- 대비 기준: 선·마커는 흰 바탕 대비 ≥ 3:1. 원안 cci #7fc4bd(L* 75)는 2.0:1, tfm #c99a4a(L* 67)는 2.6:1로 미달이어서 교체했다.
- 색각 근거: Machado 2009 deut·prot severity 1.0 시뮬레이션 후 CIEDE2000(재검증 스크립트는 세션 scratchpad `cvd.py`, 구현 단계 1에서 `src/polar/cvd.py`로 이관). 심사가 예시한 #3d8b84(L* 53, 4.0:1)는 시뮬레이션에서 direct 회색과 ΔE00 4.6이라 Fig 6a(direct·residual·cci 공존)에서 구분되지 않아 기각했다. #568f72의 시뮬레이션 ΔE00은 residual 28.5, direct 16.0, augment 17.6, phys 20.6, refit 32.7, tfm 16.1이다. 교체 후 전 쌍 최소는 14.1(direct·refit, Fig 2a 공존)이며 §5.1의 판정 기준 12를 넘는다. 원안의 "최소 15.1" 문구는 재검증 값으로 대체한다.
- 그레이스케일 구분: refit(L* 38)과 augment(L* 40)는 명도가 같으므로 augment는 일점쇄선을 전 그림에서 쓴다(Fig 3a–d 공존).
- 계획서 §7.1의 색 지정(#1f77b4·#2ca25f·#7b3294·#e08214·#2a9d8f)은 위 표로 대체한다. 계획서 문구와 `paper_figs.py COL`, `figure_spec.json` 스타일 문구를 함께 갱신해야 한다(구현 단계 1).
- 이름 규칙: 범례·캡션에서 전량 라벨 결과는 "all A-block labels (reference)"로 쓰고 "oracle"로 쓰지 않는다. 현 Fig2 원형에서 전량 A 라벨 잔차 ML은 Canada(RMSE 32.3 대 물리식 30.0 cm)·Russia E(35.0 대 32.3 cm)에서 물리식보다 나쁘다(`h3/h25_curve.csv` scope allA, S3 λ 0.25 α 1의 분할 평균 Δ도 Canada +2.3, Russia E +0.2 cm). 상한이 아니다. "oracle"은 사후 최선 분기(`oracle_branch`)에만 쓴다.

**채움·선종 의미 규칙(전 그림)**

| 부호 | 의미(유일) | 예외 |
|---|---|---|
| 빈 마커(`mfc="white"`) | 블록 등가중 채점(셀 가중 = 채운 마커와 짝) | Fig 4a–d만: 해당 n에서 최소 n 세 조건 불충족(Fig 4는 블록 등가중 채점을 그리지 않는다). 예외는 범례 첫 항목에 적는다 |
| CI 막대 선종(실선/파선) | 같은 방법의 두 변형(λ 0.25/0.5, 정보 없음/공변량만, n = 3/10, n = 10/40) | 그림마다 범례에 변형 이름 명시 |
| 곡선 선종 | 같은 방법 안의 처리(E 처리, 무작위/대표 라벨) | augment 일점쇄선은 방법 고유 |
| ▲(축 안 '미달성' 띠 위) | 최소 n 미달성(절단). 지역 마커를 쓰는 패널에는 두지 않는다 | 없음 |
| `>`/`<` 채운 마커(계열 색) | 가로 Δ 축(포레스트)의 범위 밖 값(off-scale) | 없음 |
| ↑ 열린 머리 화살표(계열 색, 축 위 끝 2.6 mm) + 수치 6.5 pt | 세로 Δ 축의 위 범위 밖 값(off-scale, 아래 범위 밖은 ↓). ▲ 와 겹치지 않게 채운 삼각형을 쓰지 않는다 | 없음 |
| 세로 틱 `|` 4 pt | 덤벨의 최악 대상 값 | 없음 |

### 1.2 지역 마커(색 아님)

| 상위 지역 | 마커 | 하위 지역 | 라벨 |
|---|---|---|---|
| Alaska(AL-1…6) | `o` | 크기 0.85배(채움 유지) | "AL-3" 6.5 pt |
| Canada(CA-1…3) | `s` | 동일 | |
| Lena(LE-1·2) | `D` | 동일 | |
| Russia W | `^` | 없음 | |
| Russia E | `v` | 없음 | |
| Russia C·Greenland | `P`·`*` | 없음 | 점 추정만, 캡션에 n |
| Mongolia/Central Asia(D1) | `h` | 없음 | 외부 홀드아웃 |

- 지역 마커는 대상 점이 9개 이상인 산점·스트립 패널(Fig 5c·d의 대상별 반투명 점)에만 쓴다. 점이 8개 이하인 패널(Fig 1d, Fig 2b)과 포레스트는 지역을 y축 라벨로만 표시한다. 3 pt에서 `s`·`D`·`h` 구분이 어렵기 때문이다.
- 하위 지역은 빈 마커를 쓰지 않는다(빈 마커 = 블록 등가중, §1.1). 크기 0.85배로만 구분한다.
- 포레스트 정렬은 |log(E_own/E0)| 오름차순, 라벨 형식 "Canada (0.00)"(E비 절대 로그). 지역별 채점 셀 수는 Supplementary Table ST1로 보내고 캡션에는 표 번호만 쓴다.
- 수준 오차 지역과 구조 오차 지역은 축 배경 띠(#f2f2f2)와 축 우측 6.5 pt 브래킷 텍스트("level", "structure")로 구분한다. 같은 패널에 n > n_max 음영(#f2f2f2)이 있으면(Fig 3a–d) 축 배경 대신 축 상자 위 3 mm 머리 띠를 칠한다(구조 = 채움 없음, 수준 = #f2f2f2, 중간 = 회색 빗금 #c8c8c8, Fig 4 와 같은 빗금).

### 1.3 연속 색

| 양 | 컬러맵 | 규칙 |
|---|---|---|
| ΔRMSE·블록 라벨 가치(부호) | `cmc.broc`(0 중심, 파랑 = 음수 = 개선, 갈색 = 악화) | `TwoSlopeNorm(vcenter=0)`. vmax = 같은 색막대를 쓰는 지도 패널의 블록 값 |값| 99 백분위를 5 cm 단위로 올림. 선·점 Δ 패널은 이 규칙에 포함하지 않는다 |
| 라벨 밀도·n | 면적 비례 원(색 아님) | s = clip(0.6·n, 4, 120) pt², 채움 refit 파랑 alpha 0.45 |
| ALT·오차 폭(비부호) | `cmc.oslo_r`, `cmc.acton` | 순차, 균일 |
| 결측·해양·AOA 밖 | #e9ecef, #B8BEC6 + 해칭 `///` | 색만으로 구분하지 않음 |
| 영구동토 배경(Fig 1a) | 회색 2단계 | 연속(PFR ≥ 90 %) #d0d0d0, 불연속(50–90 %) #e6e6e6, 그 외 육지 #f4f4f4 |

- 발산 색 원은 0 부근이 거의 흰색이라 육지 #f4f4f4 위에서 사라진다. 모든 색 원에 0.3 pt #808080 테두리를 준다.

### 1.4 rcParams와 규격

| 항목 | 값 |
|---|---|
| 글꼴 | `["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]`, `pdf.fonttype 42`, `svg.fonttype none` |
| 글자 | 기본 7.5 pt, 축 라벨 7.5, 눈금 7, 범례 6.5, 패널 안 주석 6.5, 패널 문자 9 굵게 |
| 폭·높이 | 2단 180 mm, 1단 88 mm, 높이 ≤ 200 mm |
| 축·선 | 축선 0.6 pt, 눈금 0.5 pt·길이 2.5 pt·바깥, 위·오른쪽 spine 제거, 격자 없음 |
| 데이터 선·마커 | 주 1.0 pt, 기준선 0.7 pt, 마커 3.0 pt(테두리 0.6 pt), 개별 점 1.8 pt alpha 0.35 |
| 오차 | 95 % 부트스트랩 CI, 캡 없음, 0.8 pt(포레스트 1.2 pt 둥근 끝), 띠 alpha 0.18 |
| 범례 | 그림당 1개, 테두리 없음, `handlelength 1.8`, 항목 ≤ 8 |
| 기타 | `axes.unicode_minus True`(저장 후 PDF 텍스트에서 U+2212 확인), `mathtext.default regular`, 저장 PNG 600 dpi + PDF, `savefig.pad_inches 0.02` |
| 지도 | `NorthPolarStereo(central_longitude=lon0, true_scale_latitude=70)`, NE 50 m 해안선 0.4 pt #808080, 육지 #f4f4f4, 위도 5°·경도 10°(범북극 10°·30°) 0.4 pt #999999, 축척 막대 200 km(범북극 1,000 km) 좌하, 삽도 0.22 |

### 1.5 헬퍼 시그니처(`src/polar/paperstyle.py`, 신설 제안)

```python
MM = 1 / 25.4; W2, W1, HMAX = 180 * MM, 88 * MM, 200 * MM
PAPER_RC: dict                       # §1.4
METHOD: dict[str, dict]              # §1.1 (color, ls, lw, marker)
REGION_MARKER: dict[str, str]        # §1.2
DELTA_ZERO = dict(color="#4d4d4d", lw=0.7, ls="-")
AOA_MASK = dict(facecolor="#B8BEC6", hatch="///", edgecolor="#808080", lw=0)

def use_paper() -> None
def paper_figure(width_mm=180, height_mm=None) -> Figure          # 빈 그림, 축은 axes_mm으로만 추가
def axes_mm(fig, x_mm, y_mm, w_mm, h_mm, **kw) -> Axes            # 좌하 원점 절대 좌표(축 상자만, 눈금 라벨 제외)
def slot_mm(fig, x_mm, y_mm, label_mm, w_axis_mm, h_mm, **kw) -> Axes   # 슬롯 = 좌측 라벨 여백 + 축 상자
def label_panels(axes, letters="abcdefgh", dx_mm=-3.0, dy_mm=1.5) -> None
def style_of(method, variant=None) -> dict
def plot_curve(ax, x, y, lo=None, hi=None, method="residual", ls=None, band=False,
               filled_mask=None, label=None) -> Line2D                # filled_mask: n별 채움(Fig 4a–d)
def forest(ax, labels, est, lo, hi, method, offset=0.0, ls="-", offscale=None,
           sharey_with=None, short_labels=None) -> None
def censor_band(ax, x, n_max, names, range_limited=None) -> None     # '미달성' 띠 + ▲ + n_max 6.5 pt
def zero_line(ax, axis="x", better_text=True) -> None
def mark_delta_axis(ax, which="x"|"y") -> None                       # ax.set_gid("delta_x"|"delta_y")
def shared_diverging_norm(map_arrays, step_cm=5.0) -> TwoSlopeNorm   # 같은 색막대 지도 패널만
def paper_map_ax(fig, rect_mm, extent, lon0=None, grid_lat=5, grid_lon=10, scalebar_km=200, inset=True) -> GeoAxes
def density_circles(ax, lon, lat, n, k=0.6, smin=4, smax=120, legend_n=(10, 100, 1000), edge="#808080") -> list
def one_legend(fig, handles, rect_mm, ncol=None) -> None
def region_label(name, abslogE) -> str          # "Canada (0.00)"
def save_paper(fig, name, caption, spec_id, sources, ci_kinds) -> dict   # PDF+PNG, CAPTIONS.md, figure_spec, source data, qa_check
def qa_check(fig, pdf_path) -> dict   # min_font_pt, fonts_embedded, n_text_overlaps, width_mm, height_mm,
                                       # has_title_in_axes, delta_axes_have_zero(gid 기반), unicode_minus_ok, caption_words
```

- `forest`·`plot_curve`는 Δ 축에 `mark_delta_axis`를 자동 호출한다. `qa_check`의 `delta_axes_have_zero`는 gid가 `delta_x`/`delta_y`인 축에서 0선(`DELTA_ZERO`) 존재를 검사한다.
- `paper_figure(ncols, nrows, wspace, …)` 비율 배치는 폐기한다. 모든 그림은 §2 각 절의 mm 표를 그대로 `slot_mm`에 옮긴다.
- 인접 포레스트가 같은 행 목록을 가지면 `sharey_with`로 y 눈금 라벨을 공유하고 오른쪽 패널의 라벨 여백을 2 mm로 줄인다. 행 목록이 다르면 `short_labels`(약칭)를 쓰고 캡션에 약칭 대응표를 둔다.

### 1.6 Δ·CI·최소 n 표기 규칙

- Δ = 방법 RMSE − 물리식 RMSE(cm), 음수 = 개선. 축 라벨 "ΔRMSE vs physics (cm)". 다른 기준을 쓰면 축 라벨에 기준을 쓴다("ΔRMSE vs random labels (cm)"). 상대 Δ는 "ΔRMSE vs physics (% of physics RMSE)"로 쓴다(Fig 7c만).
- 0선은 `DELTA_ZERO`로 항상 그리고 축 끝에 6.5 pt "better ←"(가로) 또는 "better ↓"(세로)를 1회.
- 포레스트는 가로(Δ = x), 곡선은 세로(Δ = y). 한 그림 안에서 섞지 않는다.

**CI 종류(`ci_kind`)와 출처**

| `ci_kind` | 정의 | 함수 | 적용 자료 | 그림 표기 |
|---|---|---|---|---|
| `block` | 분할 안 채점 블록 B개 복원 재표집 1,000회. 재표집 인덱스를 방법 A·B와 모든 반복·seed에 공유(짝지음), 반복 RMSE = √(ΣSSE/Σ셀), 반복·seed 평균 후 A − B, 분할별 부트스트랩 분포를 같은 번호끼리 평균해 결합 | `h4_common.boot_delta_blocks` | h4 실험(A2·B1–B4·C1·C2·D1), 지역 안 Δ | 그림의 주 CI(띠·막대) |
| `rep_row` | (분할, 반복) 행 Δ의 평균을 행 재표집한 백분위 부트스트랩 1,000회 | `m1_stats.boot_delta`, `h4_common.rep_boot_ci` | h2·h3·m1 기존 자료. h4 자료에서는 보조 열 `ci_rep_lo/hi` | h4 자료에서는 그리지 않고 Supplementary Table ST3로. 기존 자료는 그리되 캡션에 명시 |
| `strat_AB4` | AB4 지역 평균의 층화 블록 부트스트랩 | `m1_stats.strat` | 지역 평균 행(AB4 MEAN) | 평균 행의 CI |

- `rep_row` CI는 반복 수가 늘수록 좁아지고 채점 블록 변동을 반영하지 않는다(감사 지적). h4 자료에서 이 CI를 주 CI로 쓰지 않는다.
- 각 그림의 '통계 표기'에 CI 출처 표(패널, `ci_kind`, 자료 파일)를 둔다. 한 패널에 두 종류가 섞이면(예: Fig 1d 지역 `rep_row` + 평균 `strat_AB4`, Fig 6a h2 `rep_row`·`strat_AB4` 대 Fig 6c C1 `block`) 캡션 문장 1개로 적는다. 같은 띠·막대 문법으로 종류가 다른 CI를 한 패널 안에서 비교 가능한 것처럼 그리지 않는다.
- 다중 비교는 Holm(`m1_stats.holm`), 유의 표기는 별표 대신 Supplementary Table ST2의 p 목록. 캡션에는 "Holm-adjusted p values in Supplementary Table 2"만 쓴다.
- 셀 가중·블록 등가중 두 채점을 함께 보일 때는 채운 마커 = 셀 가중, 빈 마커 = 블록 등가중으로 전 그림 통일한다(Fig 2d, Fig 6c 커버리지 보조).

**최소 n·손익분기·회복률(사전 고정)**

- 최소 n 주 정의 n*: 블록 부트스트랩 95 % CI 상한 < 0, 분할 승률(분할별 평균 Δ < 0 비율) ≥ 2/3, 반복 승률 ≥ 0.75를 모두 만족하는 최소 n(`h4_common.min_n`). 캡션 문구(영문): "n* is the smallest n at which the block-bootstrap 95 % CI upper bound of ΔRMSE is below 0, at least two of three splits improve, and at least 75 % of repeats improve (pre-specified)."
- 미달성은 `censored=True`와 n_max를 기록하고 표·그림에 "> n_max (max tested)"로만 쓴다. 대입값(예: H27 `be_n` = 2 × n_max의 20·320·640)은 표·그림에 수치로 쓰지 않는다.
- 검사 범위 부족: n_max < 40(F1의 판정 경계 n ≤ 40을 검사할 수 없음)인 미달성 대상은 "range-limited"로 따로 표시한다(Russia E n_max 10 등).
- 순위 통계는 `h4_common.spearman_censored`·`kendall_censored`(절단 = 최대 순위 동률)로만 계산한다. 달성 여부와 |log E비|의 관계는 `h4_common.achieved_test`(Mann-Whitney U 정확 p, 완전 분리 여부)로 보고한다. 완전 분리이면 캡션에 "complete separation"을 쓰고 로지스틱 계수를 보고하지 않는다.
- 회복률 = (물리식 − 방법)/(물리식 − 전량 라벨), 분모 ≤ 0이면 NaN(`h4_common.recovery`). 회복률 규칙 손익분기(H27)는 보조 정의이며 축 라벨 "break-even n (recovery rule, H27)"으로만 쓴다. 주 정의 축 라벨은 "n* (CI rule, A2)"이다.
- 절단(축 밖): |Δ| > 축 범위는 계열 색 채운 `>`/`<` 마커를 축 끝에 두고 범례에 "off-scale"을 넣는다.

---

## 2. 본문 그림 Fig 1–7

### 2.0 절 ↔ 그림 대응과 배정 변경

**원고 절 대응(확정)**

| 원고 절 | 그림 | 실험 | 비고 |
|---|---|---|---|
| 절 1 라벨 있는 지역 | Fig 1(문제 정의), Fig 2(레시피·분해) | M1, A1, H28 | |
| 절 2 라벨 희소 지역 | Fig 3(라벨 예산 계단·부분 풀링), Fig 4(교락 제거 최소 n), Fig 5(라벨 배치) | H25·B2, A2, H29·B3 | A2는 절 2로 확정(희소 라벨의 최소 수 문제) |
| 절 3 라벨 0 지역 | Fig 6 | H18–H23, H30, C1, C2 | |
| 결론 | Fig 7(배포 절차 + B1·D1 검증), Table 1 | 종합, B1, D1 | |

**계획서 갱신 대상(이 문서는 수정하지 않음, 계획서 담당이 반영)**

1. §1 표: "잔차 학습이 물리식을 넘는 최소 라벨 수와 그 조건(A2)" 행을 절 1에서 절 2로 옮긴다.
2. §1 표 절 2 첫 행: "손익분기 n은 |log(E_own/E0)|가 결정(ρ −0.84)"를 "H27에서 |log E비| ≳ 0.2인 대상(4/14)만 검사 범위 안에서 물리식을 넘었다(Mann-Whitney 정확 p = 0.002, 완전 분리). 절단 10/14를 포함한 ρ는 달성 여부의 점이연 상관에 가까워 주 통계로 쓰지 않는다"로 바꾼다.
3. §1 표 절 2 프로토콜 행: "라벨 수에 관계없이 물리식 이하를 보장"을 "[F4 결과 확정 후 기재]"로 바꾼다.
4. §7.2 표: Fig 3·Fig 4 행의 제목·패널·자료를 이 문서 §2.3·§2.4로 교체(순서 교환), Fig 5 행에서 d(프로토콜)를 제거, Fig 6 행 d에서 D1을 제거, Fig 7 행에 b(B1 덤벨)·c(D1 상대 Δ)를 추가, Table 1 행에서 "손익분기 n"을 제거하고 "권장 절차 분기"를 추가.
5. §7.4: 절 대응 문장은 유지하되 "A2는 절 2"를 덧붙인다.
6. §7.1 색 지정과 §9 "손익분기 정의" 문장은 이 문서 §1.1·§1.6으로 대체.

**그림 배정 변경(계획서 §7.2 대비)**

| 그림 | §7.2 | 개정 | 근거 |
|---|---|---|---|
| Fig 1 | a 지도, b 산점, c 도식, d 전이 오차 | b를 지역별 z = log ALT − log √TDD 스트립으로 바꾸고 산점은 S1로. d는 AB4 4지역 + 평균만 | '지역마다 E가 다르다'를 직접 인코딩, 알래스카 과밀 해소, Fig 6a와 중복 축소 |
| Fig 2 | a 사다리, b 지역별 Δ, c 분해, d 중첩 대 사전 지정 | 유지. λ 구분을 빈 마커에서 CI 선종으로 | 빈 마커 의미 과부하 해소 |
| Fig 3 | A2 최소 n | H25 라벨 예산 계단(a–d) + B2 부분 풀링(e) | A2가 H25 S3의 교락을 제거하는 후속 실험이므로 H25를 먼저 둔다 |
| Fig 4 | H25 계단 + 손익분기 + B2 | A2 교락 제거 최소 n(a–d 곡선, e 달성/미달성 이진 패널, f 분산 − 집중) | 필요 라벨 수 패널을 하나로 합치고 주 정의(A2)만 본문에 둔다. H27 손익분기는 S6 |
| Fig 5 | 지도 + 규칙 분산 + 규칙 RMSE + 프로토콜 | 라벨 배치 한 주장으로 축소(a·b 지도, c 분산 비, d 규칙 Δ). 프로토콜은 Fig 7b | 한 그림 한 주장 |
| Fig 6 | a 기각, b CCI, c conformal, d 배포 + D1 | a 기각, b 배포 규칙, c CCI, d conformal. D1은 Fig 7c | D1 척도(물리식 RMSE 270.6 cm)가 다른 Δ 패널과 한 자릿수 이상 다르고, D1은 배포 절차(F11)의 검증이다 |
| Fig 7 | 워크플로 | a 결정 나무(간선 직교 경로) + b B1 프로토콜 덤벨 + c D1 외부 홀드아웃 상대 Δ | 절차와 그 확인적 검증을 한 그림에 |

### 2.1 Fig 1. 문제 정의: 지역별 E 차이와 전이 오차

**주장 한 문장**: 지역마다 Stefan 계수 E가 다르고, 이 차이가 라벨 없는 전이에서 직접 ML을 물리식 아래로 떨어뜨린다.

| 패널 | 내용 | x | y | 색 | 마커·크기 | 오차 |
|---|---|---|---|---|---|---|
| a | 범북극 지도: 라벨 셀 밀도(블록 단위 원)와 지역 번호 상자 1–8 | EPSG:3413 | | 원 채움 refit 파랑 alpha 0.45, 배경 CCI PFR 다년 평균 2단계 회색(§1.3) | 면적 비례 원 s ∝ 블록 셀 수, 지역 상자 0.5 pt #4d4d4d | 없음 |
| b | 지역별 z = log ALT − log √TDD 스트립(행 = 지역, 셀 단위 반투명 점 + 상자 IQR + 평균 z 점) | log E = z (눈금은 E 값 표기) | 지역(정렬 |log E비|) | 점 #6b7280 alpha 0.2(1.8 pt), 평균 검정 | 평균 z 점 3.0 pt, E0 세로선 `phys` 0.7 pt | 평균 z의 95 % CI(셀 재표집 아닌 블록 재표집 1,000회) |
| c | 세 조건 도식(정보 없음·공변량만·라벨 있음)과 채점 블록 A/B | 격자 3열 × 2행 | | 상자 tint: 조건 = #f4f4f4, 자료 = refit 파랑 tint, 채점 = phys 회색 | 화살표 1종 | 없음 |
| d | 정보 없음 전이 오차: 직접 ML − 물리식, AB4 4지역 + AB4 평균 1행 | ΔRMSE vs physics (cm) | 지역(y 라벨) | direct 회색 | `x`, 평균 행 3.6 pt | 지역 `rep_row`, 평균 `strat_AB4` |

- b의 E 산출 규칙: E_region = exp(mean z)(`h4_common.offset_z`, 로그 공간 원점 통과 최소제곱 log ALT = log E + log √TDD와 동일). 이것은 `offset_mle_prior`의 지역 평균 z̄_j와 같은 양이다. 선형 공간 원점 통과 최소제곱(기울기 = Σ ALT·√TDD / Σ TDD)은 값이 다르므로 S1 산점에서 민감도로만 보고한다.
- S1 산점(ALT 대 √TDD)의 지역별 선은 "원점 통과 최소제곱(기울기 = E)"으로 쓰고 절편을 두지 않는다. 알래스카는 무작위 500점 서브샘플, 선은 전체 셀로 적합한다.

**자료**

| 패널 | 파일 | 열 | 필터·가공 |
|---|---|---|---|
| a | `data/processed/fidelity_base_v3.csv` + `data/raw/cci_pfr/ESACCI-PERMAFROST-L4-PFR-*-fv04.0.nc` | `lat, lon, region, block`(열 이름 확인 필요, `h4/c1_smoke_cci_layers_cells.csv`와 동일 스키마 가정) | 블록별 평균 위경도·셀 수. PFR 1997–2021(25개 연도 파일) 화소 평균 후 ≥ 90 %·50–90 % 두 단계. 1997–2002는 ERA5 MODIS LST 편향 보정 산출, 2003–2021은 MODIS LST CryoGrid 산출로 산출 방식이 다르다(캡션 명시) |
| b | `fidelity_base_v3.csv` + `h2/h21_physics_rules.csv` | ALT 라벨·√TDD 열, `region, E_air_own, n, n_blocks, E_ratio_vs_AK` | 지역 6 + 알래스카, `offset_z(df, y_col, s_col)` |
| c | 없음(도식) | | |
| d | `m1/m1_sc_summary.csv` | `cond="noinfo", family_tag∈{direct}, target, rmse, n, n_blocks` + 물리식 행(`anchor`, `family_tag="analytic"`) | Δ·CI는 `h4/a1_recipe_table.csv`에 noinfo 직접 ML − 물리식 행이 없으면 `h4_analysis.py`에 재집계 항목 추가(`m1_model_shard*.csv`, `m1_stats.summarize_delta`) |

**통계 표기**: CI 출처 표(b `block`(셀 재표집 대신 블록 재표집), d 지역 `rep_row`, 평균 `strat_AB4`). 캡션에 Δ 정의, 두 CI 종류가 d에 공존함을 1문장, 지역별 채점 셀 수는 ST1. b의 지역별 E와 95 % CI는 ST1 열로. 배경 출처 문장: "Permafrost zones: ESA CCI Permafrost PFR v4.0, mean of 1997–2021, continuous ≥ 90 %, discontinuous 50–90 %."

**레이아웃(180 × 128 mm, `slot_mm` 좌하 원점)**

| 행 | 높이 | 슬롯(좌측 라벨 여백 + 축 상자 폭) | 합계 |
|---|---|---|---|
| 행 1 | 80 mm(축 72 + x 라벨 8) | a 0 + 98(지도, 눈금 라벨 없음), 간격 4, b 20(지역 이름) + 56 | 98 + 4 + 76 + 우 2 = 180 |
| 행 2 | 44 mm(축 32 + x 라벨 12) | c 0 + 98, 간격 4, d 20 + 56 | 180 |
| 기타 | 범례 a 안 좌하(원 크기 10·100·1,000 셀 + 배경 2단계), 패널 문자 여백 상단 4 mm | | 높이 80 + 44 + 4 = 128 |

**저널 관행**: P9 Fig 1·2(극 입체, 형태 = 출처, 영구동토 배경), P4 Fig 1a(지역 번호 상자 + 범례 표 n), P3 Fig 1(첫 그림에 결론 지도), P2 Fig 1(도식 상자 색 = 구성 요소 의미, 화살표 1종).

**현행 재사용·합침·폐기**: h2_fig01(c) E 계수 포레스트 → b 스트립으로 대체. h2_fig01(a) 묶음 막대 → d 포레스트로 대체(폐기). h2_map01·map02 → 폐기(a로 대체). h3_fig00 → 폐기.

**예상 함정**: (1) 지역 상자 번호와 Table 1 순번을 일치시킨다. (2) a 원의 최대 크기가 알래스카(13,542셀)에 맞춰지면 러시아(31셀)는 4 pt² 하한에 붙으므로 캡션에 하한을 명시한다. (3) PFR 연도 파일 25개를 한 번에 올리면 메모리 부담이 크다(공유 서버 OOM 이력). 연도별로 읽어 누적 평균하고 결과 래스터를 `data/processed/cci_pfr_mean_1997_2021.nc`로 캐시한다. (4) d 러시아 W 직접 ML Δ가 +10 cm 이상이면 축 범위를 [−4, 8]로 두고 off-scale 마커를 쓴다.

### 2.2 Fig 2. 라벨 있는 지역: 레시피와 이득 분해

**주장 한 문장**: 대조군 사다리의 매 단이 유의하고, 사전 지정 레시피(Stefan + CatBoost λ 0.25)의 이득은 E 재적합(수준)과 잔차(구조) 두 성분으로 분해된다.

| 패널 | 내용 | x | y | 색 | 마커 | 오차 |
|---|---|---|---|---|---|---|
| a | 대조군 사다리 5단, AB4 평균 Δ | ΔRMSE (cm), 기준 = 축 라벨 명시 | 단(영문 약칭, `LADDER_LABEL`) | 단별 방법 색(direct·refit·augment·residual) | 방법 마커 | `strat_AB4` |
| b | 지역별 Δ(레시피 λ 0.25·0.5) | ΔRMSE vs physics (cm) | 지역 6 + 평균(y 라벨) | residual 보라 | `D` 채움, λ 0.25 CI 실선·λ 0.5 CI 파선, 세로 오프셋 ±0.18 | 지역 `rep_row`, 평균 `strat_AB4` |
| c | 이득 분해: 수준 성분 대 총 이득(덤벨), 구조 성분 = 선분 | ΔRMSE vs physics (cm) | 15 대상(정렬 |log E비|, 수준·구조 띠) | 수준 = refit 파랑 `o`, 총 = residual 보라 `D`, 선분 색 = 구조 성분 부호(개선 #4d4d4d, 악화 #b0b0b0) 0.8 pt | | 없음(분할 1–3, "point estimates"). `a1` blocksse가 생기면 수준·총 이득에 `block` CI 추가 |
| d | 중첩 선택 대 사전 지정 대 전량 라벨 참조: 셀 가중·블록 등가중 두 채점 | 선택 방식(범주 4) | ΔRMSE vs physics (cm) | residual 보라(중첩·사전), `ref_allA` 검정 | 셀 가중 채움·블록 등가중 빈 마커(§1.6 전역 규칙), 두 점을 얇은 선으로 연결 | 두 채점 각각 `strat_AB4` |

**자료**

| 패널 | 파일 | 열 | 필터 |
|---|---|---|---|
| a | `h4/a1_ladder.csv` | `label, delta_rmse, ci_lo, ci_hi, p_holm, n_regions_ci` | 현재 파일의 기준은 증강(`Stefan 유사라벨 − none/const/const_t/shuffle/tddlin`)이므로 축 라벨 "ΔRMSE, augmentation minus control (cm)"로 두거나, A1 재집계로 물리식 기준 행을 추가한다(우선). 앵커+잔차 단은 `h4/a1_recipe_table.csv` `H="H13", target="REGION_SUMMARY_AB4"` |
| b | `h4/a1_recipe_table.csv` | `H="H13", cond="labels", target, delta, ci_lo, ci_hi, n_cells, n_blocks, label`(λ 0.25/0.5) | REGION_SUMMARY_AB4 행을 평균으로 |
| c | `h4/a1_decomp.csv` | `target, rmse_phys, level_shrunk, struct_on_Eshrunk, total_shrunk_resid, abslogE, n_splits` | 15행 전부. 수준 띠: abslogE ≥ 0.15 |
| d | `h3/h28_tests.csv` | `test∈{H28_nested_cb_vs_stefan, H28ref_prespec25_vs_stefan, H28ref_prespec50_vs_stefan, H28ref_oracle_vs_stefan}, target="MEAN[Lena,Canada,Russia_W,Russia_E]", delta, ci_lo, ci_hi, delta_blockeq, ci_lo_beq, ci_hi_beq, block_majority` | `H28ref_oracle`의 범례·축 라벨은 "all A-block labels (reference)" |

`LADDER_LABEL`(a1_ladder `label` → 영문 행 라벨):

| 원문 `label` | 행 라벨 |
|---|---|
| `covonly: Stefan 유사라벨 − none (직접 catboost_lo, r=10)` | vs no pseudo-labels |
| `covonly: Stefan 유사라벨 − const (직접 catboost_lo, r=10)` | vs constant |
| `covonly: Stefan 유사라벨 − const_t (직접 catboost_lo, r=10)` | vs constant + TDD |
| `covonly: Stefan 유사라벨 − shuffle (직접 catboost_lo, r=10)` | vs shuffled |
| `covonly: Stefan 유사라벨 − tddlin (직접 catboost_lo, r=10)` | vs linear TDD |

로더는 사전에 없는 `label`이 나오면 실패한다(한국어 문자열이 그림에 새지 않게).

**통계 표기**: CI 출처 표(a·d `strat_AB4`, b 지역 `rep_row`·평균 `strat_AB4`, c 없음). 이 그림은 h4 블록 CI를 쓰지 않는 기존 자료 그림이며 캡션에 1문장으로 적는다. Holm p(가족 = 사다리 5 대비, 레시피 2 대비)는 ST2. d 수치(셀 가중 대 블록 등가중)는 ST4. c는 분할 수(1–3)와 CI 부재를 캡션에 쓴다.

**레이아웃(180 × 136 mm)**

| 행 | 높이 | 슬롯(좌측 라벨 여백 + 축 상자 폭) | 합계 |
|---|---|---|---|
| 행 1 | 46 mm(축 34 = 7행 × 4.8 + x 라벨 12) | a 26(단 약칭) + 50, 간격 없음, b 24(지역 라벨) + 78 | 76 + 102 + 우 2 = 180 |
| 범례 | 7 mm | 가로 1줄(방법 색 4 + λ 선종 2 + 채점 채움/빈 2 = 8항목) | |
| 행 2 | 79 mm(축 66 = 15행 × 4.4 + x 라벨 13) | c 24(대상 라벨) + 76, 간격 2, d 14 + 62(축 높이 40, 상단 정렬) | 100 + 2 + 76 + 우 2 = 180 |
| 기타 | 패널 문자 상단 4 mm | | 46 + 7 + 79 + 4 = 136 |

**저널 관행**: P3 Fig 2·3(두 방법 색 + 기준선 전 패널 반복), P9 Fig 4(포레스트 계수 ± CI, 0선), P4 Fig 3d(막대 = 평균, 개별 값 = 점), P6 Fig 1a(결론 패널을 a에).

**현행 재사용·합침·폐기**: h3_fig05(a) → d로 합침(문자열 열 제거). h3_fig05(b) 표 → Table 1·ST4로 이동. h2_fig01(b) 수준/구조 해칭 분해 → c 덤벨로 대체.

**예상 함정**: (1) a의 기준이 행마다 다르면 h2_fig08의 결함이 재발한다. 물리식 기준 행을 재집계하지 못하면 a를 "증강 − 대조군" 단일 기준으로 통일하고 앵커+잔차 단은 b로만 보고한다. (2) c 러시아 W(−12.4)와 CA-3(+5.1) 때문에 x범위가 [−13, 7]이 되어 ±1 cm 대상이 뭉친다. x범위 [−8, 6] + 러시아 W off-scale 마커. (3) b 러시아 C·그린란드는 CI가 없으므로 점만 두고 캡션에 쓴다(빈 마커 금지). (4) Fig 7 태그 "−0.7 [−1.0, −0.3]"는 이 그림 b의 평균 행에서 오며 `workflow_tags.json`에서만 읽는다(§2.7).

### 2.3 Fig 3. 라벨 예산 계단과 부분 풀링(H25·B2)

**주장 한 문장**: 라벨 수가 늘 때 E 재적합·증강·잔차 ML의 이득은 지역 오차 유형(수준·구조)과 라벨 배치 규칙에 따라 다르게 변하며(H25, 확정), 적응형 부분 풀링 추정기는 고정 κ=10 수축을 넘지 못한다(F5 기각).

| 패널 | 내용 | x | y | 색 | 선·마커 | 오차 |
|---|---|---|---|---|---|---|
| a–d | 주 4지역 계단(Lena·Canada = 구조, Russia W·Russia E = 수준 또는 범위 제한): E 재적합(k-중심 실선·무작위 파선), 증강, 잔차 ML λ 0.25, 전량 A 라벨 참조 | Target labels n(로그, 3–320 네 패널 공유, 눈금 3·10·30·100·300) | ΔRMSE vs physics (cm), 네 패널 공유 symlog(linthresh 2 cm, 눈금 −10·−5·−2·−1·0·1·2·5) | refit 파랑, augment 초록, residual 보라, `ref_allA` 검정 파선 + 분할 범위 수염(축 상자 우측 바깥 1.5 mm, `d_phys_splits`) | 방법 마커 모두 채움 | `block`(h25b 재실행). 잔차 ML만: k-중심 = 채운 띠, 무작위 = 파선 테두리(범례 두 항목) |
| e | 부분 풀링: 대상 층화 평균 Δ 대 n, 추정기 4종(κ=10 수축·오프셋 MLE·PPI++·혼합효과 부스팅), 앵커만(variant E), 무작위 라벨. n 3·5·10 = 14 대상, 20–160 = 12, 320 = 9, 대상 집합이 바뀌는 곳에서 선을 끊는다 | Target labels n(로그, 3–320) | ΔRMSE vs physics (cm), 선형 −2 ~ 3 | refit 파랑 단일(E 추정 방법 계열) | 선종 + 마커(κ10 파선 `o`, 오프셋 MLE 일점쇄선 `s`(Fig 4 선종), PPI++ 점선 `P`), 혼합효과 부스팅 = 축 위 ↑ + 수치(off-scale). 추정기 표지는 e 축 안 4줄 | `block`(대상별 부트스트랩 분포를 번호끼리 평균) |

- e 설계 변경(2026-09-26 본 실행 후): 원안의 `b2_pooling.csv` 구조 4대상 n = 10 포레스트를 14 대상 층화 평균 Δ 대 n 선 그림으로 대체했다. a–d 에서 레나의 무작위·k-중심 부호 차이가 보이므로, e 는 같은 무작위 라벨 조건의 모집단 수준 비교(전 n)를 보여야 F5 판정(고정 κ=10 대비 적응형 추정기)을 읽을 수 있다. 구조 4대상 n = 10 값은 `h4/b2_pooling.csv` 에 남는다.
- a–d 라벨 풀 한계: Russia W·Russia E는 n_max 10이다. 곡선 끝에 세로 틱(0.6 pt, 3 mm)을 두고 n > n_max 구간을 #f2f2f2 음영으로 칠해 절단 원인을 그림에서 읽게 한다.
- a–d CI 결정: (ii) 띠 제거를 채택한다. (i) H25 주 4지역을 blocksse 저장 경로로 재실행하면 `block` 띠로 교체할 수 있으며(`h25_label_budget.py --blocksse`, A2 본 실행 이후 자원이 있을 때), 교체 전까지 `rep_row` CI는 ST3에만 둔다. 이 그림 안에서 e(`block` 막대)와 a–d(CI 없음)는 문법이 겹치지 않는다.

**자료**

| 패널 | 파일 | 열 | 필터 |
|---|---|---|---|
| a–d | `h3/h25_curve.csv` | `target, stage∈{S1, S2, S3}, rule∈{random, kmedoid}, scope="n", lam∈{0, 0.25}, alpha=1, n, d_phys_mean, n_runs, n_splits`; 참조 = `scope="allA", stage="S3", lam=0.25, alpha=1`의 분할 평균 | `d_phys_lo/hi`(`rep_row`)는 그림에 쓰지 않고 ST3로 |
| e | `h4/b2_regional.csv` | `est, n, n_targets, targets, delta, ci_lo, ci_hi, n_targets_improved` | `vs=phys, region_set=ALL, variant=E, rule=random, scope=n`, `est∈{shrink_k10, offset_mle, ppi, mixed_boost}`. `offset_mle30`·`offset_ls`(민감도)는 S6 |

**통계 표기**: CI 출처 표(a–d 없음, e `block`). 캡션에 추출 수(3 분할 × 20 추출, 잔차 5 × 2 seed), 전량 A 라벨 참조의 정의("all A-block labels (reference), not an upper bound"), symlog 축 명시("y axis symmetric-log, linear within ±2 cm"). 약칭 대응(κ10 = shrinkage κ = 10, MLE = offset maximum likelihood, ME = mixed effects, PPI = PPI++)은 캡션 1문장.

**레이아웃(180 × 118 mm)**

| 행 | 높이 | 슬롯(좌측 라벨 여백 + 축 상자 폭) | 합계 |
|---|---|---|---|
| 행 1 | 54 mm(축 40 + x 라벨 10 + 패널 위 조건 라벨 4) | a 14 + 36, b 4 + 36, c 4 + 36, d 4 + 36(b–d는 y 공유, 눈금 라벨 숨김) | 50 + 40 × 3 + 우 10(수준·구조 브래킷) = 180 |
| 행 2 | 60 mm(축 48 = 16행 × 3 + x 라벨 12) | e 24(추정법 약칭 10 + 대상 브래킷 14) + 100, 간격 4, 범례 영역 50(세로 1열, 재적합 k-중심·무작위, 증강, 잔차, 참조, 물리식 0선 = 6항목) | 124 + 4 + 50 + 우 2 = 180 |
| 기타 | 패널 문자 상단 4 mm | | 54 + 60 + 4 = 118 |

**저널 관행**: P2 Fig 2(상대 점수 축 + 기준선, 결정론 실선·확률 점선), P3 Fig 4(눈금 라벨에 n 병기), P9 Fig 4(포레스트).

**현행 재사용·합침·폐기**: `outputs/figures/paper/Fig2_label_budget` a–d → 살림(Δ 전환, 폭 191 → 180 mm, hspace 0.95 → `slot_mm` 절대 배치, Liberation Sans, f 안 범례 제거, 물리식 = 0선 실선, 참조 = 검정 파선, c·d x축 3–10 → 3–320 공유). 현 원형의 눈금 하이픈 '−12'(unicode_minus 미적용)와 물리식·참조 기준선이 같은 점선인 결함을 고친다. 현 원형 c 패널의 'n* = 3'(`breakeven_n_rec50`) 주석은 제거한다(회복률 규칙이며 주 정의가 아니다). Fig2 원형 e·f(n = 3·10 Δ 포레스트) → S4로 이동. h3_fig02(c)·h3_fig03(a)(손익분기 산점) → S6으로 이동. h3_fig01·02(a)(b) → 폐기.

**예상 함정**: (1) symlog 선형 구간(±2 cm) 경계에서 기울기가 꺾여 보인다. 캡션에 명시하고 눈금 1·2를 반드시 둔다. (2) e에서 PPI++ CI가 넓어(라벨 10개) x범위를 지배하면 [−4, 8] + off-scale 마커. (3) B2 미완료 시 e는 κ10 행만 채우고 `_draft` 저장. (4) 본 실행에서 B2 블록 SSE가 저장되지 않으면 e를 `rep_row`로 그리지 말고 `_draft`에 머문다.

### 2.4 Fig 4. 교락 제거 최소 라벨 수(A2)

**주장 한 문장**: E 처리와 라벨 배치를 분리해 잔차 학습의 최소 라벨 수 n*를 판정한 결과는 [F1–F3 결과 확정 후 기재]. 선행 H27(회복률 규칙, S6)에서는 |log E비| ≳ 0.2인 대상만 검사 범위 안에서 물리식을 넘었다.

| 패널 | 내용 | x | y | 색 | 선·마커 | 오차 |
|---|---|---|---|---|---|---|
| a–d | 대표 4대상 n 곡선(AL-3·CA-2 = 수준, Lena·Canada = 구조), 라벨 배치 all_blocks 고정 | Target labels n(로그, 3–320 공유) | ΔRMSE vs physics (cm), Fig 3a–d와 같은 symlog 축 | residual 보라 1색 + `ref_allA` 검정 | E0 고정 실선, κ=10 수축 파선, 오프셋 최대우도 일점쇄선, 모두 `D`. n별 채움 = 최소 n 세 조건 충족, 빈 = 불충족(이 그림의 유일한 빈 마커 의미). E_own 고정 = 검정 파선(마커 없음) | E0 고정만 `block` 띠 |
| e | 달성/미달성 이진 패널: E0 고정·all_blocks·λ 0.25, 14 대상 | |log(E_own/E0)| | 위 영역: n*(로그, 채운 `D`, 대상 이름 6.5 pt). 축 끊김 아래 영역: 'not reached' 띠(높이 8 mm, 공통 y), ▲ #4d4d4d + "> n_max" 6.5 pt, range-limited는 ▲ #B8BEC6 + "(range-limited)" | residual 보라(달성) | 지역 마커 쓰지 않음 | 없음(판정 결과) |
| f | 라벨 배치 효과: Δ(all_blocks) − Δ(concentrated), n = 10·40, 14 대상 | ΔRMSE, spread minus concentrated (cm) | 대상(y 라벨, 정렬 |log E비|) | residual 보라 | `D` 채움, n = 10 CI 실선·n = 40 CI 파선, 세로 오프셋 ±0.18 | `block` |

- a–d 대상 선택: 수준 대표는 n_max가 160–320인 AL-3·CA-2로 한다. Russia W는 라벨 풀 n_max 10이라 n* 판정 범위가 짧고 Fig 3에 이미 있으므로 제외한다.
- e 설계 근거: 미달성 대상의 n_max가 10·20·160·320으로 달라 "1.15 × n_max 화살표"는 가짜 순서를 만든다. 미달성은 공통 y의 띠에 두고 n_max를 글자로만 적는다. 생존형 계단(x = n 로그, y = 누적 달성 대상 비율, 수준·구조 두 계단, 절단은 n_max에서 계단 끝 틱)은 주장이 |log E비| 축에 있으므로 S6 보조로 둔다(자료는 `a2_minn`의 `n_star, censored, n_max` 세 열).
- 다른 E 처리(κ10·오프셋 MLE·E_own 고정)의 n*는 e에 겹치지 않고 ST5와 S12(전체 대상 행렬)로 보낸다.

**자료**

| 패널 | 파일 | 열 | 필터 |
|---|---|---|---|
| a–d | `h4/a2_curve.csv` | `target, e_treat∈{E0_fixed, shrink_k10, offset_mle, E_own_fixed}, spread, scope, lam, stage, n, d_phys_mean, blk_d_phys, blk_d_phys_lo, blk_d_phys_hi, split_win, rep_win, n_runs, n_splits` | `spread="all_blocks", scope="n", stage="resid", lam=0.25`. 대상 4. 블록 CI·승률 열이 없으면 `h4_analysis.py`가 `a2_blocksse.npz`에서 재집계(현 스모크 `a2_smoke_*`는 blocksse가 없고 `d_phys_lo/hi`는 `rep_row`). n별 채움 = `min_n` 세 조건을 그 n 한 점에 적용한 불리언 |
| e | `h4/a2_minn.csv` + `h4/a2_targets.csv` | `target, parent, e_treat, spread, lam, stage, n_star, censored, n_max`(주 정의, `min_n` 산출) + `logE_ratio_own`(없으면 `h3/h27_loo_S3.csv abs_logE`) | `e_treat="E0_fixed", spread="all_blocks", stage="resid", lam=0.25`. `lam=0` 행은 `stage="E_only"`(잔차 없음)이므로 제외. `target="Alaska_f*"` 제외(14 대상) |
| f | `h4/a2_spread_tests.csv`(`h4_analysis.py` 신설) | `target, n∈{10, 40}, e_treat="E0_fixed", delta, ci_lo, ci_hi, split_win, rep_win, ci_rep_lo, ci_rep_hi, ci_kind="block"` | `boot_delta_blocks(key_A = spread all_blocks, key_B = spread concentrated)`, 같은 (split, rep, seed) 짝 |

**로더 규칙(스모크·구형 파일 호환)**: `a2_smoke_minn.csv`에는 `censored`·`n_star` 열이 없고 미달성을 `minn_ci = −1` 센티널로 기록한다(예: AL-3 E0_fixed resid all_blocks `minn_ci` −1, `n_max` 20). 구형 파일을 읽을 때 `censored = minn_ci < 0`, 표시값 = n_max로 변환한다. 단 `minn_ci`는 CI 상한 단일 조건(`rep_row`)이므로 `_draft` 그림에만 쓰고, 본 그림은 `min_n` 세 조건 산출(`n_star`)만 쓴다.

**통계 표기**: CI 출처 표(a–d `block`, f `block`). 캡션에 n* 주 정의(§1.6 영문 문장 그대로), 반복 수(분할 3 × 추출 10 × seed 2), E_own 고정 = "E_own known (reference), not an upper bound". e 통계 문장: "Targets reaching n* within the tested range versus not: Mann-Whitney U exact p = [값] (h4_common.achieved_test), complete separation [yes/no]; censoring-aware Kendall τ_b between |log E ratio| and n* = [값] (censored targets tied at the maximum rank)." Spearman ρ는 쓰지 않는다. 판정 가설(F1–F3)은 본문에.

**레이아웃(180 × 132 mm)**

| 행 | 높이 | 슬롯(좌측 라벨 여백 + 축 상자 폭) | 합계 |
|---|---|---|---|
| 행 1 | 54 mm | Fig 3 행 1과 동일(a 14 + 36, b–d 4 + 36, 우 10) | 180 |
| 범례 | 7 mm | 가로 1줄(E 처리 3 선종, 참조, 채움 = 세 조건 충족, 빈 = 불충족, 물리식 0선 = 7항목) | |
| 행 2 | 67 mm(축 55 + x 라벨 12) | e 14(로그 n 눈금) + 80(위 영역 44 mm, 끊김 3 mm, 미달성 띠 8 mm), 간격 2, f 22(대상 이름) + 60 | 94 + 2 + 82 + 우 2 = 180 |
| 기타 | 패널 문자 상단 4 mm | | 54 + 7 + 67 + 4 = 132 |

**저널 관행**: P3 Fig 3·4(조건 × 방법 격자, 패널 위 조건 라벨 "AL-3 (0.30)"), P2 Fig 4a(앙상블 얇은 선 + 굵은 평균), P6 Fig 1a(같은 방법 변형을 선종으로).

**현행 재사용·합침·폐기**: h3_fig01(4패널 골격)·h3_fig01b → a–d 골격만 재사용, α=10 계열 폐기(S6). h3_fig03(c) → 폐기(LOO 실패는 본문 문장). 원 Fig 3e와 원 Fig 4e(두 정의의 필요 라벨 수 산점)는 이 그림 e 하나로 합치고 H27은 S6으로 옮긴다.

**예상 함정**: (1) A2 미완료 시 스모크(`a2_smoke_*.csv`, n ∈ {3, 16, 20, 326})로 골격을 만들되 그림 안에 "smoke"를 남기지 않고 파일명 `_draft`로 저장한다. (2) e에서 미달성 대상 이름이 띠 안에서 겹치면(|log E비| 0.003–0.07 구간에 5대상) 이름을 띠 아래 두 줄로 엇갈려 배치하고, 그래도 겹치면 이름 대신 Table 1 번호를 쓴다. (3) 달성 대상이 3 미만이면 `achieved_test`의 p는 정보가 없다. 그 경우 캡션에 개수만 쓴다. (4) E_own 고정은 참조이며 상한이 아니다(구조 지역에서 물리식보다 나쁠 수 있음).

### 2.5 Fig 5. 라벨 배치(H29·B3)

**주장 한 문장**: 라벨 배치가 수보다 중요하다. 단일 블록 라벨은 대상의 56–62 %에서 해로웠고(H29, 확정), 선택 규칙의 E 추정 분산 감소 효과는 [F7 결과 확정 후 기재].

| 패널 | 내용 | x | y | 색 | 마커 | 오차 |
|---|---|---|---|---|---|---|
| a | 캐나다 블록 라벨 가치 지도(S3 기준) | EPSG:3413 lon0 = −120 | | 원 채움 = `broc` 0 중심(a·b 공통 vmax), 해로운 블록(value_S3 > 0) 해칭 `///` | 면적 비례 원 s ∝ n_cells, 테두리 0.3 pt #808080, k-중심 선호 블록(kmedoid_freq ≥ 0.5)은 테두리 검정 0.8 pt | 없음(색 = 분할 평균 가치) |
| b | 레나 블록 라벨 가치 지도 | EPSG:3413 lon0 = 127 | | a와 동일, 색막대 공유 1개 | 동일 | |
| c | 선택 규칙 5종의 E 추정 분산 비(무작위 대비), n = 3 | 규칙(약칭 rand·k-ctr·clus-w·D-opt·act-PPI) | Variance ratio vs random(로그) | refit 파랑 | 대상 14개 반투명 지역 마커 1.8 pt, 중앙값 채운 3.6 pt | 없음(개별 점이 분산) |
| d | 규칙별 B블록 Δ 대 무작위, n = 10, S3 | ΔRMSE vs random labels (cm) | 규칙(5행, c와 같은 약칭) | residual 보라 | 평균 채움 `D`, 대상 14개 반투명 지역 마커 | 평균 `strat_AB4` 또는 14 대상 평균의 `block`(아래) |

**자료**

| 패널 | 파일 | 열 | 필터·집계 |
|---|---|---|---|
| a·b | `h3/h29_block_value.csv` | `target, split, block, n_cells, lat, lon, value_S3, kmedoid_freq, E_block_minus_E0` | 행 단위가 (target, split, block)이므로 `groupby(["target", "block"])`로 `value_S3` 평균, `n_cells` 평균, `kmedoid_freq` 평균, `lat, lon` 첫 값. 결과 Canada 61블록·Lena 33블록. 유의성 CI 열이 없으므로 해칭은 부호만(캡션 명시). `h3/h29_summary.csv` `frac_harm_S3`를 캡션 |
| c | `h4/b3_evar.csv` | `target, rule, n=3, vtot_ratio, mse_tot_ratio, var_rep_ratio, rep_degenerate` | 주 = `vtot_ratio`(V_tot = V_boot + V_between, 무작위 대비). `rep_degenerate=True` 대상은 점을 회색 처리하고 캡션. 판정 F7은 `h4/b3_f7.csv` `n_pass, median_ratio` |
| d | `h4/b3_summary.csv` + `h4/b3_region.csv` | `target, rule, stage="S3", n=10, blk_d_rand, blk_d_rand_lo, blk_d_rand_hi, blk_majority_rand`; 평균 행 `b3_region` MEAN 계열 `delta, ci_lo, ci_hi, ref="random"` | 대상 점은 `blk_d_rand`(블록 부트스트랩 점 추정). `d_rand_lo/hi`(`rep_row`)는 ST3 |

**통계 표기**: CI 출처 표(d 대상 `block`, 평균 `strat_AB4`). 캡션에 색막대 단위(cm, "block value = ΔRMSE when the block's labels are added"), 축척 기준 위도(70°N), 규칙 약칭 대응 1문장, a·b 해칭은 부호 기준이며 유의 표시가 아님.

**레이아웃(180 × 128 mm)**

| 행 | 높이 | 슬롯(좌측 라벨 여백 + 축 상자 폭) | 합계 |
|---|---|---|---|
| 행 1 | 66 mm | a 0 + 84, 간격 2, b 0 + 84, 색막대 2.5 + 눈금 라벨 7.5 | 84 + 2 + 84 + 10 = 180 |
| 범례 | 6 mm | 가로 1줄(원 크기 10·100·1,000, k-중심 테두리, 해칭, 지역 마커 요약) | |
| 행 2 | 52 mm(축 40 + x 라벨 12) | c 12(로그 눈금) + 70, 간격 4, d 20(규칙 약칭) + 72 | 82 + 4 + 92 + 우 2 = 180 |
| 기타 | 패널 문자 상단 4 mm | | 66 + 6 + 52 + 4 = 128 |

**저널 관행**: P4 Fig 4(지역 확대 지도 + 국가명), P6 Fig 4(부호 지도 발산 색), P6 Fig 2a·b(해칭), P4 Fig 3d(막대 = 평균, 개별 모델 = 점).

**현행 재사용·합침·폐기**: h3_fig04 → a·b로 재작성(등장방형·색막대 2개·크기 범례 2개 폐기). h2_fig07b(규칙 − 무작위 Δ 요약) → d 골격 재사용. h2_fig07 10패널 → S5. h2_map03 → 폐기. 원 Fig 5e(프로토콜 덤벨) → Fig 7b로 이동.

**예상 함정**: (1) 캐나다 블록 61개 중 CA-1(6셀, 75.5°N)이 본체에서 멀어 extent가 커진다. 본체 extent는 60–70°N, 130–115°W로 잡고 CA-1은 삽도 위치만. (2) 레나 블록 33개가 72°N 부근 100 km 안에 몰려 원이 겹친다. 원 최대 s를 80 pt²로 낮추고 겹침 순서는 n 오름차순. (3) vmax는 a·b 두 지도 블록 값에서만 정한다(다른 Δ 패널 값을 넣으면 색이 옅어진다).

### 2.6 Fig 6. 라벨 0 지역

**주장 한 문장**: 라벨 없는 지역에서 H18–H23 우회 기법은 물리식을 유의하게 넘지 못했고 배포 규칙은 물리식과 동급이었다(확정). CCI 결합과 계층 conformal 구간의 판정은 [F9·F10 결과 확정 후 기재].

| 패널 | 내용 | x | y | 색 | 마커 | 오차 |
|---|---|---|---|---|---|---|
| a | 우회 기법 기각 포레스트(H18 LST 강제, H18x 토양, H19 공변량 축소 x14·x16, H21 계수 대여, H22 IRM·V-REx·DANN, H23 MAML, H20 CCI blockE), 조건 2종 | ΔRMSE vs physics (cm) | 기법(10행, 약칭, 가족 구분선 정보/학습목표) | direct 회색(공변량·계수), residual 보라(학습목표·MAML), cci(H20) | 채움, 정보 없음 CI 실선·공변량만 CI 파선, 세로 오프셋 | `strat_AB4`(h2), H21(±3) off-scale |
| b | 배포 규칙별 AB4 평균·최악 RMSE(덤벨) | RMSE (cm), [25, 50] | 규칙 7(약칭) | 규칙별 방법 색 | 평균 채움, 최악 = 세로 틱 `|`, 선 연결 | 없음(Δ CI는 ST2) |
| c | CCI 다층 결합 4수준 대 물리식, AB4 평균 + 지역 | ΔRMSE vs physics (cm) | 수준(4행) | cci | 평균 `v` 채움, 지역은 y 오프셋 반투명 점(지역 라벨 없이 캡션) | 평균 `strat_AB4` 또는 `block`(c1_tests `ci_scope`에 따름) |
| d | 계층 conformal 커버리지 대 폭, LORO 라벨 0 | Empirical coverage(90 % 목표) | Interval width (cm) | 참조(CQR·앵커 풀링) = direct 회색, 계층 = refit 파랑 | 모두 `o` 채움 + 점 옆 6.5 pt 약칭(CQR-ak, CQR-iw, anc-ak, anc-iw, pool, cdf, blk, sub, wconf). 방법 마커(`^`·`D`·`s`)는 재사용하지 않는다 | 두 축 95 % CI 막대, 목표 0.90 세로선 + [0.85, 0.95] 회색 띠 |

**자료**

| 패널 | 파일 | 열 | 필터 |
|---|---|---|---|
| a | `h2/h_tests_all.csv` | `is_mean=True, confirmatory=True, family, source, test, cond∈{noinfo, covonly}, delta, ci_lo, ci_hi, p_holm` + `h2/h23_tests.csv`(MAML) | H20·H21은 `h_tests_main.csv`. MAML 기준이 `delta_vs_shrink`면 `h23_summary.csv` `d_phys`로 재집계 |
| b | `h2/h_deploy_gating.csv` + `h3/h30_deploy_worst.csv` | `rule, mean_AB4, worst_AB4, ml_frac_mean`; `target, rule, rmse, aoa_frac, n` | 규칙 7 |
| c | `h4/c1_tests.csv` | `method∈{stefan_cci, cci_uncw, cci_blocksmooth, gate_gtd_pfr}, ref="stefan", cond="noinfo", target, ci_scope, delta, ci_lo, ci_hi, split_win, rep_win, ci_rep_lo, ci_rep_hi, p_holm, ci_hi_neg` | 스모크 스키마가 이미 `block` 규약(`ci_lo/hi` = 블록, `ci_rep_*` = 보조)을 따른다. `cci_uncw_srcvar`는 민감도(S8) |
| d | `h4/c2_coverage.csv` | `test="label0", scope="all(noinfo)", method∈{ref_cqr_ak, ref_cqr_iw, ref_anchor_ak, ref_anchor_iw, pooled, hier2_cdf, hier2_blk, hier2_sub}, coverage, coverage_lo, coverage_hi, width_cm, width_cm_lo, width_cm_hi, interval_score, coverage_cellw`; `test="labeled", n=3, method="wconf"` | `hier2_exact`(`is_inf=True`, 폭 ∞)는 제외하고 캡션 |

**통계 표기**: CI 출처 표(a `strat_AB4`(h2 기존), b 없음, c C1 `block`·평균 `strat_AB4`, d C2 커버리지 CI). 캡션 1문장: "Panel a uses stratified bootstrap intervals from the earlier experiments, panel c uses the block bootstrap (Methods)." Holm 가족(정보 4, 학습목표 3, CCI 4)과 p 목록은 ST2. d 커버리지 정의(셀 가중이 주, 블록 등가중은 ST6).

**레이아웃(180 × 132 mm)**

| 행 | 높이 | 슬롯(좌측 라벨 여백 + 축 상자 폭) | 합계 |
|---|---|---|---|
| 행 1 | 62 mm(축 50 = 10행 × 5 + x 라벨 12) | a 28(기법 약칭) + 60, 간격 2, b 24(규칙 약칭) + 64(축 높이 35, 상단 정렬) | 88 + 2 + 88 + 우 2 = 180 |
| 범례 | 7 mm | 가로 1줄(방법 색 3, 조건 선종 2, 목표 띠 = 6항목) | |
| 행 2 | 59 mm(축 47 + x 라벨 12) | c 18(수준 약칭) + 66, 간격 4, d 12(폭 눈금) + 78 | 84 + 4 + 90 + 우 2 = 180 |
| 기타 | 패널 문자 상단 4 mm | | 62 + 7 + 59 + 4 = 132 |

a·b는 행 목록이 달라 y 눈금을 공유하지 않는다. 기법·규칙 이름은 약칭(예: "H18 LST", "H22 IRM")으로 두고 캡션에 대응표 1문장을 둔다.

**저널 관행**: P9 Fig 4(포레스트), P6 Fig 1b·c(개별 점 + 오차 막대 + 목표선, 소형 통계는 캡션), P2 Fig 2(기준선 반복).

**현행 재사용·합침·폐기**: h2_fig08 → a 골격 재사용(기준선별 구분선, 영문 약칭, 수치 열 제거, 조건은 CI 선종). h3_fig07(a) → b 덤벨. h3_fig06 히트맵 → 폐기(ST 표). h2_fig02·04·05·06 → 폐기(a로 흡수). h2_map02 → 폐기. 원 Fig 6e(D1) → Fig 7c로 이동.

**예상 함정**: (1) a의 H23 MAML은 기준이 수축이라 물리식 기준 행과 섞이면 축 의미가 갈린다. 재집계하거나 구분선 아래 "vs shrink"를 축에 명시한다. (2) d의 `hier2_exact` 무한 폭 제거를 캡션에 쓰지 않으면 누락으로 읽힌다. (3) b 절대 RMSE 축에서 규칙 간 차이 1–3 cm가 작게 보이므로 x범위를 [25, 50]으로 자르고 캡션에 명시. (4) d 약칭 라벨 9개가 겹치면 `adjustText` 대신 수동 오프셋 표(`C2_LABEL_OFFSET`)를 스펙으로 고정한다.

### 2.7 Fig 7. 배포 절차와 확인적 검증(워크플로·B1·D1)

**주장 한 문장**: 새 지역의 라벨 수와 대표점 3개의 E비로 절차가 정해지며, 2단계 프로토콜의 확인적 검정(B1)과 외부 홀드아웃(D1) 결과는 [F4·F11 결과 확정 후 기재].

**a. 결정 나무(격자 5열 × 5행, 셀 34 × 16 mm)**

| id | 종류 | 텍스트 | 격자(열, 행), 폭(열 수) | 스타일 |
|---|---|---|---|---|
| N0 | 시작 | New region: covariates x25, TDD | (0, 2), 1 | 둥근 상자 #f4f4f4 |
| Q1 | 결정 | Labels available? | (1, 2), 1 | 마름모 흰 바탕 0.6 pt |
| P0 | 잎 | Physics anchor E0 (± CCI) + hierarchical conformal 90 % interval + AOA mask | (2, 0), 3 | 상자 phys 회색 tint |
| S3 | 절차 | Sample 3 cells (k-centre); r̂ = |log(Ê₃/E0)| | (2, 2), 1 | 상자 흰 바탕 0.6 pt |
| Q2 | 결정 | r̂ ≥ τ? | (3, 2), 1 | 마름모 |
| P1 | 잎 | E re-fit (offset MLE / partial pooling), stop | (4, 1), 1 | 상자 refit 파랑 tint |
| P2 | 잎 | Keep E0, spread labels over blocks, residual ML (λ 0.25) | (4, 3), 1 | 상자 residual 보라 tint |
| P3 | 잎 | Pre-specified recipe: Stefan + CatBoost residual λ 0.25 | (2, 4), 3 | 상자 residual 보라 tint |

- 원안의 Q3(블록 ≥ 8)·O1·O2는 제거하고, 각 잎 상자 하단에 보고 형식 배지(6.5 pt, 회색 테두리 0.4 pt): "95 % CI if blocks ≥ 8, else point + interval"를 둔다.
- 원안 Q2(절차와 결정을 한 마름모)는 S3(절차 상자)와 Q2(결정 마름모)로 나눴다.

**간선(직교 경로, 격자 좌표 waypoint)**

| 간선 | 라벨 | waypoint(열, 행) | 관통 검사 |
|---|---|---|---|
| N0 → Q1 | | (0, 2) → (1, 2) | 없음 |
| Q1 → P0 | n = 0 | (1, 2) 위 꼭짓점 → (1, 0) → (2, 0) P0 왼쪽 | (1, 1) 빈 칸 |
| Q1 → S3 | n = 3–10 | (1, 2) → (2, 2) | 없음 |
| Q1 → P3 | n ≥ tens | (1, 2) 아래 꼭짓점 → (1, 4) → (2, 4) P3 왼쪽 | (1, 3) 빈 칸 |
| S3 → Q2 | | (2, 2) → (3, 2) | 없음 |
| Q2 → P1 | yes | (3, 2) 위 꼭짓점 → (3, 1) → (4, 1) | (3, 1) 빈 칸 |
| Q2 → P2 | no | (3, 2) 아래 꼭짓점 → (3, 3) → (4, 3) | (3, 3) 빈 칸 |

- P0(행 0)과 P3(행 4)은 열 2–4를 차지하고 P1(행 1)·P2(행 3)은 열 4만 차지하므로 어느 간선도 다른 노드를 지나지 않는다. `fig7_workflow()`는 간선 선분과 모든 노드 상자의 교차를 검사해 교차가 있으면 실패한다.
- 화살표 1종(0.6 pt, 머리 3 pt). 분기 라벨 6.5 pt. τ는 B1 확정 전까지 기호 "τ"로 두고 캡션에서 정의한다.

**태그와 `workflow_tags.json`**: 각 잎 상자 우하단 회색 6.5 pt 태그. 모든 태그 수치(Fig 2 태그 "−0.7 [−1.0, −0.3]" 포함)는 `outputs/figures/paper/workflow_tags.json`에서만 읽는다. 블록 CI 재집계 후 수치가 바뀔 수 있으므로 스펙·캡션·코드에 수치를 직접 쓰지 않는다. 근거 패널은 Δ가 실제로 그려진 패널이어야 한다.

```json
[
  {"leaf_id": "P0", "fig_ref": "Fig. 6a",   "source_csv": "data/processed/h2/h_tests_all.csv",
   "filter": "is_mean & confirmatory", "stat": "delta_range", "lo": null, "hi": null,
   "text": "Fig. 6a · Δ {lo} to {hi} cm (n.s.)"},
  {"leaf_id": "P1", "fig_ref": "Fig. 3a–d", "source_csv": "data/processed/h3/h25_curve.csv",
   "filter": "stage=='S1' & rule=='kmedoid' & scope=='n' & n<=10 & target in LEVEL", "stat": "d_phys_mean_range", "lo": null, "hi": null,
   "text": "Fig. 3a–d · Δ {lo} to {hi} cm"},
  {"leaf_id": "P2", "fig_ref": "Fig. 4a–d", "source_csv": "data/processed/h4/a2_curve.csv",
   "filter": "e_treat=='E0_fixed' & spread=='all_blocks' & stage=='resid' & lam==0.25 & n<=10", "stat": "blk_d_phys_range", "lo": null, "hi": null,
   "text": "Fig. 4a–d · Δ {lo} to {hi} cm"},
  {"leaf_id": "P3", "fig_ref": "Fig. 2b",   "source_csv": "data/processed/h4/a1_recipe_table.csv",
   "filter": "H=='H13' & target=='REGION_SUMMARY_AB4' & lam==0.25", "stat": "delta_ci", "lo": null, "hi": null,
   "text": "Fig. 2b · Δ {est} [{lo}, {hi}] cm"},
  {"leaf_id": "PROTOCOL", "fig_ref": "Fig. 7b", "source_csv": "data/processed/h4/b1_protocol.csv",
   "filter": "method=='protocol@{tau_sel}'", "stat": "mean_d_by_n", "lo": null, "hi": null,
   "text": "Fig. 7b · mean Δ {n3} (n = 3), {n10} (n = 10) cm"}
]
```

필수 키는 `leaf_id, fig_ref, source_csv, filter, stat, lo, hi`이고 `est`·`text`는 선택이다. `paper_figs.py::build_workflow_tags()`가 원천 CSV를 필터해 `lo`·`hi`(와 `est`)를 채우고, `fig7_workflow()`와 `table1_summary()`가 같은 JSON을 읽는다(수치 단일 관리).

**b. B1 2단계 프로토콜 덤벨**

| 항목 | 값 |
|---|---|
| x | ΔRMSE vs physics (cm), [−2, 6], 14 대상 평균(채움)과 최악 대상(세로 틱 `|`) |
| y | 방법 8행(약칭): E0+res (n = 0), always E0+res, always shrink+res(현행 S3), shrink only, refit only, protocol@τ_sel, explore-gate@τ_sel, oracle branch |
| 부호 | 방법 색(residual 포함 = 보라, 수축·재적합만 = 파랑, oracle branch = 검정 `*`), n = 3 CI 없는 덤벨 실선·n = 10 파선, 세로 오프셋 ±0.18 |
| 보조 | 행 우측 폭 8 mm 소형 비율 막대(n_worse/14, 회색 #b0b0b0, 눈금 없음, 0 및 14 끝 틱) |
| off-scale | refit only n = 3 최악 +37.0 cm, n = 10 최악 +10.2 cm는 축 끝 채운 `>` |

**c. D1 외부 홀드아웃(몽골·중앙아시아 46셀/21블록) 상대 Δ**

| 항목 | 값 |
|---|---|
| x | ΔRMSE vs physics (% of physics RMSE) = `d_phys_mean / rmse_phys × 100`(독립 축, 캡션 명시) |
| y | 방법 × n: resid_src_only (n = 0), e0fix_resid, shrink_only, refit_only, shrink_resid_S3, protocol@τ_sel(n = 3·10) + oracle(참조 검정 `*`) |
| 오차 | `block`(`d1_blocksse.npz`에서 재집계), 사전 예측 구간(`d1_prediction.json`)을 회색 띠(zorder 0) |
| 게재 조건 | 아래 함정 (4)의 편향 점검 통과 시에만 본문. 불통과 시 S로 이동하고 c 자리는 비운 채 b를 180 mm로 확장 |

**자료**

| 패널 | 파일 | 열 | 필터 |
|---|---|---|---|
| a | `workflow_tags.json` | 위 스키마 | |
| b | `h4/b1_protocol.csv` | `method, n, n_targets, n_worse, n_worse_ci, mean_d, worst_d, worst_target, gap_to_oracle` | 실제 `method` 이름: `E0_resid_src`(n = 0), `always_E0_resid`, `always_shrink_resid`, `shrink_only`, `refit_only`, `protocol@0.1`·`@0.15`·`@0.2`, `explore_gate_agree@0.1`·`@0.15`·`@0.2`, `oracle_branch`, `oracle_any`. 본문은 τ_sel 1개, 나머지 τ와 `oracle_any`는 ST4 |
| b τ | `h4/b1_meta.json` 키 `tau_sel_loto`(선택값)와 `h4/b1_tau_loto.csv`(`target, tau_sel`) | | 현재 `b1_meta.json` 키는 `taus`(후보 목록)뿐이고 선택 결과 열·키가 없다. `h32_two_stage_protocol.py` 또는 `h4_analysis.py`가 LOTO 선택을 기록해야 한다. 키가 없으면 `fig7_workflow()`는 실패하고 `protocol@0.15`로 대체하지 않는다 |
| c | `h4/d1_summary.csv` + `h4/d1_prediction.json` + `h4/d1_blocksse.npz` | `target, method, n, tau, rmse_mean, rmse_phys, d_phys_mean, blk_d_phys, blk_d_phys_lo, blk_d_phys_hi, split_win, rep_win, bias_mean, n_eval_mean, ci_flag` | `tau = −1.0`은 τ 비적용 방법 행(프로토콜 외 전부)이며 τ 필터에서 제외하지 말고 그대로 쓴다. 프로토콜은 `tau == τ_sel` 행만. `refit_allA`·`shrink_allA`(n = 23 전량)는 ST7. 현 `d_phys_lo/hi`는 `rep_row`이므로 `ci_rep_*`로 개명 |

**통계 표기**: CI 출처 표(b 없음, n_worse_ci는 대상별 `rep_row` CI 기반 개수로 ST4, c `block`). 캡션에 F4 판정 규칙(악화 대상 수, 오라클 대비 평균 Δ 차 ≤ 0.5 cm), F11 사전 예측 대 관측 문장, c 독립 축 문장. n_worse 목록·최악 대상 이름은 ST4.

**레이아웃(180 × 150 mm)**

| 행 | 높이 | 슬롯 | 합계 |
|---|---|---|---|
| 행 1 | 84 mm | a: 좌 5 + 격자 5열 × 34 = 170 + 우 5 | 180 |
| 범례 | 6 mm | b·c 공용 1줄(방법 색 3, n 선종 2, oracle, 예측 띠 = 7항목) | |
| 행 2 | 56 mm(축 44 + x 라벨 12) | b 30(방법 약칭) + 48 + 비율 막대 8, 간격 2, c 26(방법·n 약칭) + 64 | 86 + 2 + 90 + 우 2 = 180 |
| 기타 | 패널 문자 상단 4 mm | | 84 + 6 + 56 + 4 = 150 |

**저널 관행**: P7 Fig 1(단계 번호 + 의미 색 + 최종 산출), P2 Fig 1(상자 색 = 구성 요소, 화살표 1종), P9 Fig 4(포레스트).

**현행 재사용·합침·폐기**: h3_fig00 → 폐기(수동 좌표·한국어 문장 상자). h3_fig07(b) → c 형식 참고.

**예상 함정**: (1) 상자 텍스트 3줄 초과 시 6.5 pt에서 34 mm 폭을 넘는다. `textwrap` 폭 28자, 잎 P0·P3은 3열 폭이라 한 줄 80자까지. `qa_check` 겹침 검사로 확인. (2) b에서 현재 관측값은 사전 주장과 반대 방향이다. `protocol@0.15`는 n = 10에서 악화 8/14로 `shrink_only` n = 3(3/14)보다 많고, 오라클 분기(1/14·4/14)와의 평균 Δ 격차가 0.77 cm로 F4 기준 0.5 cm를 넘는다. 주장 문장·태그는 F4 판정 전까지 자리표로 둔다. (3) 기대 Δ 태그는 확정 실험 값 외의 추정치를 쓰지 않는다. (4) D1 편향 점검: 물리식 RMSE 270.6 cm, `e0fix_resid`의 `bias_mean` ≈ −196 cm, `resid_src_only` −223 cm로 예측이 관측보다 약 2 m 낮다. 게재 전 (i) 라벨 정의(ALT 대 계절 동결 깊이·최대 해빙 깊이), (ii) 단위(cm 대 m), (iii) 셀의 영구동토 존재 여부(CCI PFR < 50 % 셀 비율), (iv) CALM 원자료 연도 범위를 확인하고 결과를 `docs/EXPERIMENT_LOG.md`에 적는다. (5) D1 Δ(−14 ~ −141 cm)는 다른 Δ 패널(±12 cm)과 공통 축·공통 발산 범위를 적용할 수 없으므로 상대 Δ 독립 축으로만 그린다.

---

## 3. Table 1 설계(대상별 요약, 2026-09-26 구현본)

구현: `scripts/4_visualization/paper/table1.py`(`paper_figs.py --only table1`). 산출 `outputs/figures/paper/Table1_region_summary.{csv,tex,pdf,png}`. 표 본체는 LaTeX booktabs, PNG/PDF 는 180 mm 미리보기.

| 열 | 내용 | 출처 파일·열 | 표기 |
|---|---|---|---|
| Target (No.) | 대상 이름 + Fig 1a 지역 번호(`fig1.REGION_NO`, 알래스카 1, 레나 2, 캐나다 3, 러시아 W 4, 러시아 E 5, 러시아 C 6, 그린란드 7). 하위 지역은 상위 지역 번호 | `h3/h25b_targets.csv` `parent` | Russia W (4), AL-3 (1) |
| Cells / blocks | 대상 전체 라벨 셀·블록 | `h3/h25b_targets.csv` `n_cells, n_blocks`; Russia C·Greenland 는 `fidelity_base_v3.csv` F4_direct | 750 / 39 |
| E_own/E0 (\|ln\|) | 최소제곱 자체 계수 대 원천 계수(100 km 버퍼 제외) | `h3/h25b_targets.csv` `E_own, E0` | 1.66 (0.51) |
| 오차 유형(행 묶음) | 수준 \|ln\| ≥ 0.20, 중간 0.15–0.20, 구조 < 0.15 | 위와 같음 | 묶음 머리 |
| Residual ΔRMSE, n = 0 | 원천 잔차만(E0 고정, λ 0.25), 블록 CI | `h4/a2_curve.csv` scope n0 | −0.95 [−1.18, −0.57] |
| 90 % interval coverage | 계층 conformal(hier2_cdf) LORO 커버리지, 블록 CI | `h4/c2_coverage.csv` label0, scope all | 0.75 [0.59, 0.88] |
| Width (cm) | 같은 행의 평균 구간 폭 | `h4/c2_coverage.csv` `width_cm` | 71 |
| n* Shrunk E (H25): k-medoids·random | 세 조건 주 정의(S3, λ 0.25) | `h3/h25b_breakeven.csv` = `h3/h27c_targets.csv` main | 3, 또는 "> 320" |
| n* E0 fixed (A2) | 세 조건 주 정의(E0 고정, all_blocks, λ 0.25) | `h4/a2_minn.csv` | 320, 또는 "> 160" |
| Recipe ΔRMSE | 사전 지정 레시피 λ 0.25, 라벨 전량(AB4 + 층화 평균) | `h4/a1_recipe_table.csv` H13 = `m1/m1_sc_tests.csv` | −0.68 [−1.01, −0.27] |

- 행: H25b 분석 대상 14(오차 유형 묶음 안 \|ln\| 내림차순), 라벨 0 평가만 있는 러시아 C·그린란드, 요약 2행(주 4지역 평균, n* 달성 수). CA-1 은 A 블록 라벨 3개라 제외.
- 초안(주 지역 7 + 외부 홀드아웃 + 평균, 하위 지역은 ST3)에서 바꾼 근거: F1–F11 의 분석 단위가 14 대상이고, 주 지역 행만 두면 알래스카(하위 지역으로만 채점)의 n* 열이 빈다.
- 초안 열의 이동: 외부 홀드아웃(8) → Fig 7c·ST7(라벨 정의가 달라 같은 열 체계로 비교 불가). 'Worst deploy RMSE' → Fig 7b(최악 대상 틱)·ST4a. 'Workflow branch' → Fig 7a 잎 태그(`workflow_tags.json`, 표와 이중 관리 방지). H27 손익분기 → ST6a.
- 미달성 n* 는 "> n_max" 만 쓴다(대입값 금지). 블록 < 8 인 러시아 C·그린란드는 CI 자리를 "[–]" + 각주.
- 극단값 각주: AL-6 라벨 0 잔차 Δ +6.02 [4.95, 8.74] cm 는 물리식 RMSE 가 14 대상 중 최저(10.4 cm)이고 3분할 모두 악화(+3.8 ~ +7.1 cm)임을 각주로 적는다(문장은 자료에서 생성).
- 구간 폭 반올림: 평균 행 80.5 cm → 81 cm(사실 목록 width_ab4 81). 같은 MEAN6 폭을 쓰는 다른 표시 항목도 81 cm 로 적는다.

---

## 4. 부록 그림 S1–S12와 Supplementary Table

| 번호 | 내용 | 자료 | 형식 |
|---|---|---|---|
| S1 | 자료·공변량 목록과 처리 흐름 + ALT 대 √TDD 산점(지역별 원점 통과 최소제곱선, 기울기 = E, 선형·로그 공간 두 적합 비교) | `fidelity_base_v3_meta.json`, `covariates_ext_v1.csv`, `fidelity_base_v3.csv` | 표 + 흐름도 1행 + 산점 |
| S2 | 채점 규약 도식: A/B 블록, 블록 부트스트랩(분할 안 블록 재표집·방법 공통 인덱스), 층화 블록 부트스트랩, 세 채점 | 없음 | Fig 1c 확장판, 180 × 80 mm |
| S3 | 하위 지역 14개 정의 지도(위치 키 a–n) + 표(셀·블록·E비) | `h3/subregions.csv`, `h27_loo_S3.csv` | 극 입체 지도 3장 + 표 |
| S4 | H25 하위 지역 10개 계단 + Fig2 원형 e·f(n = 3·10 Δ 포레스트) | `h25_curve.csv` | Fig 3a–d 문법(띠 없음), 2 × 5 격자 |
| S5 | H18–H24 상세 표와 관측 설계 곡선(h2_fig07 재작성) | `h_tests_all.csv`, `h24_curve.csv`, `h3_hurts.csv` | 표 + 2 × 5 곡선 |
| S6 | 필요 라벨 수 보조: (a) H27 손익분기 n(recovery rule) 대 |log E비|, 미달성은 Fig 4e와 같은 띠 문법, (b) A2 생존형 계단(수준·구조), (c) α·λ·κ 민감도, (d) n_theory 대 경험 n(F6) | `h27_loo_S3.csv`, `a2_minn.csv`, `h25_curve.csv`, `b2_theory.csv`, `b2_theory_corr.csv` | 4패널. (a) 캡션 통계: 달성 4/14, Mann-Whitney U = 40, 정확 p = 0.002, 완전 분리, Kendall τ_b −0.64(p = 0.004), 값은 `h4_analysis.py` 재산출로 확정 |
| S7 | TFM few-shot(B4): TFM 대 CatBoost 잔차 Δ, n = 3·10·40 | `h4/b4_summary.csv`, `b4_tests.csv` | 포레스트, tfm 대 residual |
| S8 | CCI 편향 전파 + `cci_uncw_srcvar` 민감도 | `h4/c1_cells.csv` | 산점(1:1 점선) + 지도 1 |
| S9 | 라벨 정의·다년 평균 민감도(D3) | `m1/label_year_sensitivity_cells.csv` | 짝지은 Δ 포레스트 |
| S10 | 약라벨 다중 충실도(C4, 수행 시) | InSAR 표 1건 | 표만 |
| S11 | 소프트웨어·시드·실행 시간 | `*_meta.json` `elapsed_s`, 패키지 버전 | 표 |
| S12 | A2 전체 대상 행렬: E 처리 3종별 패널(행 = 14 대상 |log E비| 정렬·수준/구조 띠, 열 = n 로그 격자, 셀 색 = `blk_d_phys`, broc 0 중심 공통 vmax), 최소 n 세 조건 충족 경계를 셀 테두리 계단선으로 | `h4/a2_curve.csv` | 3패널 행렬 |

| 표 | 내용 |
|---|---|
| ST1 | 지역·대상별 채점 셀·블록 수, 지역별 E와 95 % CI(Fig 1b) |
| ST2 | 그림별 Holm 가족과 조정 p 목록 |
| ST3 | h4 자료의 `ci_rep`(행 재표집) 대 `block` CI 대조, H25 `rep_row` CI |
| ST4 | B1 전 방법·τ의 n_worse, n_worse_ci, 최악 대상, 오라클 격차. Fig 2d 두 채점 수치 |
| ST5 | A2 E 처리·배치별 n* 전체(절단 표기) |
| ST6 | H27 손익분기 n(회복률 규칙, 절단 표기), C2 블록 등가중 커버리지 |
| ST7 | D1 절대 RMSE·편향·전량 라벨 참조 행 |

공통: 본문 규칙 그대로(180 mm, 7–8 pt, 패널 문자, 범례 1개). 부록도 결론형 제목을 그림 안에 두지 않는다.

---

## 5. QA 절차와 완료 기준

### 5.1 절차(그림 1장당)

1. `save_paper()` 저장 직후 `qa_check()` 자동 실행. 실패 항목이 있으면 저장 파일명에 `_FAIL` 접미를 붙이고 CAPTIONS·figure_spec 갱신을 건너뛴다.
2. `pdffonts` 확인: 임베드 글꼴이 Arial 또는 LiberationSans(Type 42)만. DejaVu 출현 시 실패. PDF 텍스트 추출에서 음수 눈금이 U+2212인지 확인(하이픈 '-' 출현 시 실패).
3. PNG를 180 mm 폭 100 %로 렌더한 확인용 시트(`outputs/figures/paper/_qa/<name>_100pct.png`, 300 dpi)를 만들고 `visual-reviewer`, `scientific-figure-reviewer` 두 검토 에이전트에 각 1회, 수정 후 1회 더 검토.
4. 색각·대비 재검: 같은 패널 안 색 쌍을 `cvd.py`(Machado deut·prot)로 ΔE00 ≥ 12, 선·마커 흰 바탕 대비 ≥ 3:1 확인. 방법 색이 §1.1 표와 다르면 실패.
5. 캡션 점검(그림별 단어 예산, 총 ≤ 350단어):

| 블록 | 예산(단어) | 내용 |
|---|---|---|
| 정의 | 80 | 그림 이름 첫 문장, Δ 정의·기준, n*·참조·약칭 정의 |
| 통계 방법 | 80 | CI 종류(`ci_kind`)·횟수·짝지음 단위, 반복 수, 종류 혼재 문장, 판정 규칙 |
| 패널 설명 | 150 | 패널별 1–2문장, 독립 축·symlog·절단·off-scale 명시 |
| 자료 경로 | 40 | 원천 CSV 경로, Supplementary Table 번호 |

   Holm p 목록, 지역별 채점 셀 수, n_worse 목록, 세 채점 수치 등 열거형 수치는 캡션에 넣지 않고 Supplementary Table(§4 ST1–ST7)로 보내며 캡션에는 표 번호만 쓴다. `qa_check`의 `caption_words`가 블록별 단어 수를 세어 초과 시 실패한다.
6. 데이터 대조: 그림에 찍힌 값 3개 이상을 원천 CSV에서 다시 읽어 캡션·Table 1·`workflow_tags.json`과 일치 확인(`qa_values.py`, 대조 로그 `_qa/<name>_values.txt`).

### 5.2 완료 기준(모두 충족 시 완료)

| 항목 | 기준 |
|---|---|
| 규격 | 폭 180.0 ± 1.0 mm(1단 88.0 ± 1.0), 높이 ≤ 200 mm, 최소 글자 ≥ 6.5 pt, 축 라벨 ≥ 7 pt, `slot_mm` 합계가 §2 mm 표와 일치 |
| 글꼴 | Type 42, Arial 또는 Liberation Sans만, 음수 눈금 U+2212 |
| 충돌 | 텍스트 바운딩 박스 교차 0건, 범례 대 데이터 겹침 0건, Fig 7 간선·노드 교차 0건 |
| 문법 | 그림 안 제목·설명 문장·수치 열·matplotlib 표 없음, 패널 ≤ 6, 범례 1개 ≤ 8항목, 빈 마커 의미 그림당 1개, 패널 문자 소문자 굵게 |
| 색·부호 | §1.1 색 일치, 대비 ≥ 3:1, 지역 마커는 산점 패널만, Δ 지도 0 중심 공통 범위(같은 색막대 지도만), 무지개·붉은 계열 없음, AOA 회색 + 해칭 |
| Δ·통계 | 모든 Δ 축(gid `delta_*`)에 0선 + 기준·단위, h4 자료는 `block` CI만 그림에, `ci_kind` 캡션 명시, 절단은 ▲ 띠·off-scale은 채운 `>`, 대입값 미표기 |
| 주장 | 판정 전 가설(F1–F11)을 근거로 한 주장 문장·태그는 자리표 상태 |
| 지도 | EPSG:3413 계열, 해안선, 위도선, 축척 막대, 삽도, 면적 비례 원 + 3단계 범례, 원 테두리 |
| 기록 | CAPTIONS.md·figure_spec.json paper_* 항목 자동 갱신, 자료 경로 캡션 기재, `_qa/` 로그 존재, 그림별 Source Data(`outputs/figures/paper/source_data/<Fig>_<panel>.csv`, 그림에 찍힌 값 그대로; Nature Communications 병행 투고 대비) |
| 검토 | 두 검토 에이전트 2회차에서 잔여 지적 0(또는 문서화된 수용 근거) |

---

## 6. 구현 순서(`paper_figs.py` 함수 목록)

| 순서 | 단계 | 함수·파일 | 의존 |
|---|---|---|---|
| 1 | 스타일 모듈 신설: rcParams·METHOD(§1.1 개정 색)·REGION_MARKER·헬퍼(§1.5, `axes_mm`·`slot_mm`·`censor_band`·`mark_delta_axis` 포함), `src/polar/cvd.py`, 계획서 §7.1 색 문구·`paper_figs.py COL`·`figure_spec.json` 스타일 문구 동시 갱신 | `src/polar/paperstyle.py` | 없음 |
| 2 | 블록 CI 재집계: `h4_analysis.py`에 npz → `blk_d_phys(_lo/_hi)`, `split_win`, `rep_win`, `ci_rep_lo/hi`, `ci_kind` 덧붙이기(a2·b2·b3·d1), `a2_minn`의 `min_n` 세 조건 산출, `a2_spread_tests.csv`, `achieved_test`·`kendall_censored` 결과, B1 τ LOTO 선택 기록 | `scripts/2_evaluation/h4_analysis.py` | h4 실험 |
| 3 | 자료 로더 공통화(필터·정렬·라벨 형식·`LADDER_LABEL`·구형 `minn_ci` 변환·h29 집계) | `load_h25_curve, load_recipe_table, load_tests_mean, load_subregions, load_a2_minn, load_h29_blocks, order_by_abslogE` | 1 |
| 4 | Fig 3 재작성(기존 H25 자료, 현 원형 전환) | `fig3_label_budget()`(현 `fig2_label_budget` 개명) | 1·3 |
| 5 | Fig 2(a1·h28, 기존 자료) | `fig2_recipe_decomp()` | 3 |
| 6 | Fig 6 a·b(h_tests_all·h_deploy_gating·h30), c·d는 C1·C2 자료 | `fig6_label0()` | 3 |
| 7 | 지도 헬퍼 검증(cartopy, NE 50 m 캐시, scalebar)과 PFR 평균 캐시 후 Fig 1 | `fig1_problem()`, `schematic_conditions(ax)`, `pfr_mean_cache()` | 1 |
| 8 | Fig 5 a·b 지도 + c·d(B3) | `fig5_label_placement()` | 7 |
| 9 | Fig 4(A2, 본 실행 전 `_draft`) | `fig4_min_labels()` | 2·3 |
| 10 | 태그 JSON 생성과 Fig 7(결정 나무 + B1 + D1) | `build_workflow_tags()`, `fig7_workflow()` | 2 |
| 11 | Table 1 생성기 | `table1_summary()` → CSV + LaTeX | 10 |
| 12 | 실험 완료 순으로 자료 교체: A2 → Fig 4·Table 1; B2 → Fig 3e; B3 → Fig 5 c·d; B1·D1 → Fig 7b·c; C1·C2 → Fig 6 c·d; 태그 JSON 재생성 | 각 `fig*()` 재실행, `_draft` 접미 제거 | 실험 |
| 13 | 부록 S1–S12, ST1–ST7 | `figS1_data … figS12_a2_matrix`, `tableS1 … tableS7` | 4–12 |
| 14 | QA 일괄 실행과 검토 2회 | `python paper_figs.py --qa`(`_qa/` 시트·값 대조·캡션 단어) | 13 |

- `REG = {"fig1": fig1_problem, …, "table1": table1_summary, "S1": figS1_data, …}`. 실행 인자 `--only`, `--draft`(스모크 자료 허용), `--qa`.
- 보고서용 `h2_figs.py`·`h3_figs.py`는 변경하지 않는다. 논문 그림은 위 함수만 원천으로 쓴다.

## 남은 가정·위험

- `fidelity_base_v3.csv`의 위경도·지역·블록 열 이름은 확인하지 않았다(`c1_smoke_cci_layers_cells.csv` 스키마와 같다고 가정).
- h4 요약 CSV의 블록 CI 열은 현재 C1(`c1_tests`)·B3(`b3_summary`의 `blk_*`)만 갖고 있다. A2·B2는 스모크에 blocksse가 없고, D1은 npz가 있으나 요약 CSV에 열이 없다. `h4_analysis.py` 재집계 항목(§6 단계 2)이 없으면 Fig 3e·4·7c는 `_draft`에 머문다.
- H25에는 블록 SSE가 없어 Fig 3a–d는 CI 없이 그린다. 재실행 여부는 A2 본 실행 이후 자원으로 결정한다.
- B1 τ LOTO 선택 결과가 산출 파일에 없다(`b1_meta.json` 키 `taus`만 존재). 기록 전까지 Fig 7b·c는 완성할 수 없다.
- D1 물리식 편향(약 −2 m)의 원인(라벨 정의·단위·영구동토 부재 셀)을 확인하기 전에는 Fig 7c 게재를 확정하지 않는다.
- 색각 검증은 시뮬레이션이며 실제 관찰자 검증이 아니다. 전 쌍 최소 ΔE00 14.1(direct·refit)은 판정 기준 12를 넘지만 여유가 작다.
- H27 S6 통계(Mann-Whitney 정확 p 0.002, Kendall τ_b −0.64)는 이 문서 작성 시 `h4_common` 함수로 계산한 참고값이며 최종 값은 `h4_analysis.py` 산출로 확정한다.

---

## 개정 이력

### 2026-09-26 개정(심사 지적 반영)

필수 지적

| # | 지적 | 반영 |
|---|---|---|
| 1 | §1.6 CI 규칙이 반복 부트스트랩 | §1.6 CI 종류 표 신설. h4 자료 주 CI = `boot_delta_blocks`(`block`), 기존 방식은 `ci_rep` 보조 열로 ST3에만. 그림에는 `block`만 |
| 2 | a2_curve·b2_pooling·d1_summary 열 이름 | §2.4·§2.3·§2.7 자료 표를 `blk_d_phys(_lo/_hi)`·`split_win`·`rep_win`으로 교체, 스모크 `d_phys_lo/hi`는 `ci_rep_*`로 개명. 열 부재 시 `h4_analysis.py`가 `<tag>_blocksse.npz`에서 재집계(머리말·§6 단계 2) |
| 3 | h25 곡선 CI 처리 | (ii) 채택: Fig 3a–d(구 Fig 4a–d) 띠 제거, 캡션 "no block-level CI available". (i) 재실행은 전환 경로로 §2.3에 기록. 같은 그림 안 CI 문법 혼재 없음 |
| 4 | ρ −0.84 주장 삭제 | 주장 문장을 "|log E비| ≳ 0.2인 대상만 검사 범위 안에서 물리식을 넘었다"로 바꾸고(§2.4, 계획서 갱신 대상 2) 통계를 `achieved_test`·`kendall_censored`로. H27 참고값(달성 4/14, U = 40, 정확 p = 0.002, 완전 분리, τ_b −0.64)을 S6에 |
| 5 | 이진 표현 재설계 | Fig 4e: 위 영역 달성 n*(채운 `D`), 축 끊김 아래 'not reached' 띠(▲ + n_max 6.5 pt). 1.15 × n_max 화살표 폐기. Russia E 등 n_max < 40은 range-limited 회색 ▲ |
| 6 | n* 정의 전체 기재 | §1.6 영문 캡션 문장, Table 1 n* 열 정의를 세 조건으로. Fig 4a–d n별 채움/빈으로 조건 충족 표시 |
| 7 | Fig 3e·4e 중복 | 한 패널(Fig 4e)로 합치고 주 정의(A2)만. H27은 S6(a). Table 1의 H27 열은 ST6으로, 절단 표기 "> n_max (max tested)" 통일, 대입값 금지 |
| 8 | Fig 3·4 역할·순서 | Fig 3 = H25 계단 + B2, Fig 4 = A2로 교환. §2.0 배정 표 갱신, 계획서 §7.2·§7.4 갱신 대상 4·5에 기록(계획서는 이 과업 범위 밖이라 직접 수정하지 않음) |
| 9 | 절 흐름 불일치 | §2.0 절 ↔ 그림 대응표 신설, A2를 절 2로 확정, 계획서 §1 표를 갱신 대상 1로 기록 |
| 10 | Fig 5 주장 결과 중립 | 프로토콜 주장은 Fig 7로 옮기고 "[F4 결과 확정 후 기재]" 자리표. 관측 수치(8/14 대 3/14, 격차 0.77 cm)는 §2.7 함정 (2)에 기록. 태그도 JSON 자리표 |
| 11 | Fig 5 한 그림 한 주장 | Fig 5를 라벨 배치 4패널로 축소, 프로토콜 덤벨은 Fig 7b |
| 12 | Fig 6e 축 설계 | D1을 Fig 7c로 옮기고 상대 Δ(% of physics RMSE) 독립 축. 편향 −196 cm 점검 항목을 §2.7 함정 (4)에 |
| 13 | oracle 명명 | §1.1 `oracle` → `ref_allA` "all A-block labels (reference)", 용도 "전량 라벨 참조(상한 아님)", 근거 수치 병기. oracle은 `oracle_branch`만 |
| 14 | 캡션 350단어 | §5.1 5단계에 블록별 예산(80·80·150·40)과 ST 이관 규칙, `caption_words` 검사 |
| 15 | 폭 예산 | 모든 그림 mm 표를 "좌측 라벨 여백 + 축 상자" 슬롯으로 재작성. Fig 6 행 1: 88 + 2 + 88 + 2 = 180. 약칭 + 캡션 대응표, 같은 행 목록이면 `sharey_with` |
| 16 | Fig 7 태그 근거 | 태그 근거를 Δ가 그려진 패널(Fig 3a–d, Fig 4a–d, Fig 2b, Fig 6a, Fig 7b)로 교체, Fig 2 태그 포함 전 수치를 `workflow_tags.json`에서만 |
| 17 | CI 규칙 충돌 | §1.6 `ci_kind` ∈ {block, rep_row, strat_AB4}, 각 그림 통계 표기에 CI 출처 표, 혼재 패널(Fig 1d, Fig 2b, Fig 6a 대 c) 캡션 문장 |
| 18 | 최소 n 정의 불일치 | Fig 4 캡션·Table 1·Fig 7 태그를 세 조건 정의로. 순위 통계는 `spearman_censored`·`kendall_censored`만(§1.6), 본문 캡션에는 Kendall과 `achieved_test` |
| 19 | a2_minn 열 가정 | §2.4 로더 규칙: `censored = minn_ci < 0`, 표시값 = n_max, `stage="resid", lam=0.25` 필터, `lam=0`은 `stage="E_only"`, Alaska_f* 제외. 구형 `minn_ci`는 `_draft`만 |
| 20 | 두 y축이 다른 양 | 축 라벨 "n* (CI rule, A2)"와 "break-even n (recovery rule, H27)" 구분, H27은 S6로 분리. 현 원형 'n* = 3' 주석 제거(§2.3 재사용 항) |
| 21 | 빈 마커 과부하 | §1.1 채움·선종 의미 표: 빈 마커 = 블록 등가중(전역), Fig 4a–d만 예외(조건 불충족). λ·조건·n 변형은 CI 선종, 무작위 라벨은 곡선 파선, 최악은 세로 틱, 절단은 ▲ 띠 |
| 22 | 절단 표현 | 축 끊김 + 'not reached' 범주 띠 + ▲(§2.4 e). 생존형 계단은 주장이 |log E비| 축에 있으므로 S6(b) 보조로(자료 `n_star, censored, n_max`) |
| 23 | Russia E 은닉 | (i) 채택: 네 패널 공통 symlog(linthresh 2 cm, 눈금 −10·−5·−2·−1·0·1·2·5), x 로그 3–320 공유, n_max 세로 틱 + 초과 구간 음영 |
| 24 | mm 레이아웃 | 각 그림 mm 표를 슬롯 형식으로 재작성하고 높이 합계 검산(Fig 2 136 mm: 15행 × 4.4 mm). §1.5에 `axes_mm`·`slot_mm` 추가, `wspace` 비율 배치 폐기 |
| 25 | Fig 7 간선 관통 | Q2를 S3(절차) + Q2(결정)로 분리, Q3·O1·O2 제거 후 잎 배지로 흡수, 간선 직교 waypoint 표와 교차 검사 |
| 26 | Fig 7 태그 근거·스키마 | §2.7에 `workflow_tags.json` 스키마(`leaf_id, fig_ref, source_csv, filter, stat, lo, hi`)와 예시 |
| 27 | Fig 6e 척도·필터 | 상대 Δ 독립 축(§2.7 c), `tau = −1.0` 행 처리 규칙, 실제 method 이름(`e0fix_resid`, `resid_src_only`, `shrink_resid_S3`, `refit_allA`, `shrink_allA`, `oracle`) 반영 |
| 28 | Fig 5e 필터 | Fig 7b 자료 표에 실제 method 이름 전체, τ 선택 출처 `b1_meta.json` 키 `tau_sel_loto`·`b1_tau_loto.csv`(현재 부재, 기록 필요)와 대체 금지 규칙 |
| 29 | cci 대비 | #7fc4bd(2.0:1) → #568f72(L* 55, 3.8:1). 예시 #3d8b84는 CVD에서 direct와 ΔE00 4.6이라 기각. residual과 ΔE00 28.5. tfm #c99a4a(2.6:1)도 #ad921a(3.0:1)로 교체 |
| 30 | Fig 6c 마커 충돌 | conformal 패널(현 Fig 6d)은 `o` 채움 + 색 2종 + 6.5 pt 약칭 라벨. 방법 마커 재사용 금지 |
| 31 | Fig 5a·b 집계·테두리·vmax | `groupby(target, block)` 평균 규칙, 원 테두리 0.3 pt #808080, vmax는 같은 색막대 지도 패널로 한정(§1.3) |
| 32 | Fig 1a 배경 | ESA CCI PFR v4.0 1997–2021 평균 2단계 회색(≥ 90 %, 50–90 %), 산출 방식 변경(2003) 캡션, 캐시 파일 |
| 33 | Fig 1b 적합 형식 | E = exp(mean z)(`offset_z`, 로그 공간 원점 통과 최소제곱)로 명시, S1 산점 선은 "원점 통과 최소제곱(기울기 = E)", 선형 공간 적합은 민감도 |

선택 지적

| # | 반영 |
|---|---|
| 1 | 채택: Fig 3a–d·4a–d x 3–320 공유, n_max 초과 음영 |
| 2 | 조건부 채택: a1 blocksse가 생기면 Fig 2c에 `block` CI(현재 파일 없음) |
| 3 | 채택: 채움 = 셀 가중, 빈 = 블록 등가중 전역 규칙(§1.1·§1.6), Fig 4a–d만 예외 |
| 4 | 채택: augment 일점쇄선 전역 |
| 5 | 채택: Fig 1d를 AB4 4지역 + 평균으로 축소 |
| 6 | 채택: Table 1 'Workflow branch' 열, Fig 7 태그와 Table 1이 같은 JSON |
| 7 | 채택: §5.2 기록에 그림별 Source Data |
| 8 | 부분 채택: 본문 Fig 4a–d는 유지(필수 6 요구), 전체 대상 행렬은 S12 |
| 9 | 채택: Fig 1b = z 스트립, 산점은 S1 |
| 10 | 미채택: 결정 나무 + 잎별 Δ 분포는 폭(100 + 70 mm)에서 노드 텍스트가 6.5 pt로 들어가지 않는다. 대신 Fig 7b·c에 B1·D1 분포 요약 |
| 11 | 채택: Fig 2c 선분 색 = 구조 성분 부호 두 단계 회색 |
| 12 | 채택: Fig 7b 행 우측 n_worse/14 비율 막대 8 mm |
| 13 | 채택(4와 동일) |
| 14 | 채택: 지역 마커는 9점 이상 산점 패널만(§1.2) |
| 15 | 채택: `mark_delta_axis`(gid)와 `qa_check` 연동(§1.5) |
| 16 | 채택: Fig 3 재사용 항과 §5.1 2단계에 unicode_minus 확인, 191 mm 규격 초과 수정 |
| 17 | 채택: §2.2 `LADDER_LABEL` 대응표, 미등록 라벨은 실패 |
