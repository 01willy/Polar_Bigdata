# 겹침·화살표·바탕 지도 점검 기록(2026-10-05, v4)

범위: 덱이 쓰는 PNG(`deck/assets/paper_report/*.png`, `slide_panels/`, `maps/`, `method/`)와 논문 그림(Fig 1–7, Table 1, `maps/`, `si/`).
점검 방식: PNG 를 100 %(600 dpi 논문, 300 dpi 슬라이드)로 잘라 보고, 스크립트의 글자 겹침·캔버스 밖·글자 크기 점검을 함께 돌렸다.
고친 항목만 적는다. '이상 없음'은 본 뒤에 고칠 것이 없었다는 뜻이다.

## 0. 공통 변경(바탕 지도 토큰 `color.map_base`)

- `scripts/4_visualization/paper_v3/style.py`: `BASEMAP` 이 `color.map_base.paper` 를 읽고, `set_basemap(medium)` 이 `use_v3(medium)` 안에서 매체(논문·슬라이드)에 맞게 바꾼다. `buffer_line` 추가. 영구동토 구역 열쇠에 'Permafrost zone' 머리를 적는다(`PF_KEY_LABELS`).
- `maps_alt_v3.py`: 육지·해안·경위선을 모듈 상수가 아니라 그릴 때 `S.BASEMAP` 에서 읽는다(매체 전환이 반영되도록). 위치 삽도의 육지에도 해안선을 그린다.
- `fig1.py`: a·c·d·티베트 삽도에 해안선(Natural Earth 50 m / 10 m, 토큰 폭)을 더하고, 완충 등치선을 `buffer_line` 색으로. 열쇠를 3줄(크기, 'Permafrost zone', 구역 견본)로.
- `fig6.py`, `si_figs.py`(S8–S12 의 target_maps): 본 지도·삽도에 해안선, 열쇠에 'Permafrost zone' 머리.
- `fig5.py`, `fig7.py`: 육지 다각형에 해안 테두리(토큰 폭), 위치 삽도의 육지 색을 토큰 색으로.
- 위도·경도 라벨의 띄어쓰기: FreeSans 의 ° 자간 때문에 '60° N' 이 갈라져 보여 '60°N', '170°W' 로 통일(fig1, fig5, fig6, fig7; maps_alt_v3·XL 은 Cartopy 기본 형식이라 원래 붙어 있음).

## 1. 슬라이드 자산(`deck/assets/paper_report/`)

| 파일 | 바탕 지도 | 고친 것 |
|---|---|---|
| `slide_panels/Fig1_a_slide.png` | 적용(slide 토큰) | 위도 라벨 3개와 1000 km 축척 막대의 자리 탐색에 '육지 위 비용'을 더해 바다 위(그린란드 동쪽·노르웨이해)로 옮김. 열쇠 2줄 → 3줄('Permafrost zone' 머리). 위도 라벨 '60°N' 형식. 지역 원은 지역 색 + 같은 색 60 % 테두리(0.8 pt) 유지 |
| `slide_panels/Fig6_a_slide.png` | 적용 | 70° N·80° N 표지가 스발바르·섬 위에 놓이던 것을 그린란드 동쪽 바다로 제한(경선 후보에 벌점). 축척 막대를 바다(−42°, 60° N)로. 열쇠에 'Permafrost zone' 머리. 해안선 추가 |
| `slide_panels/Fig6_b_slide.png` | 적용 | 해안선 추가(글자 없음) |
| `slide_panels/Fig6_cbar_slide.png` | 해당 없음 | 다시 그림(변경 없음). 글자 12·14 pt |
| `slide_panels/Fig7_b_slide.png`, `Fig7_b_small_slide.png` | 적용 | 육지에 해안 테두리, 위치 삽도 육지 색 토큰. 경위도 눈금 '170°W', '70°N' 형식 |
| `slide_panels/Fig5_c_slide.png` | 적용 | 논문 기하로 그리되 바탕 지도만 slide 토큰(`S.set_basemap("slide")`)으로 바꿔 자름. 해안 테두리 추가 |
| `slide_panels/Alaska_ALT_map_v3_c_slide.png` | 적용 | 육지 #DEDEDE, 해안 #8A8A8A 0.8 pt(maps_alt_v3 토큰). 그 밖 변경 없음 |
| `slide_panels/Fig1_b_slide.png`, `Fig1_e_slide.png`, `Fig7_c_slide.png`, `Fig7_e_slide.png` | 지도 없음 | 100 % 점검: 겹침·잘림·화살표·11 pt 미만 글자 없음. 변경 없음 |
| `maps/Alaska_ALT_map_v3_slide.png`, `maps/Lena_ALT_map_v3_slide.png` | 적용 | 육지·해안·경위선 토큰. 위치 삽도 해안선. 그 밖 변경 없음 |
| `maps/XL_Alaska_product_differences_slide.png`, `maps/XL_Lena_product_differences_slide.png`, `maps/XL_display_resolution_slide.png` | 적용 | maps_alt_v3 바탕 함수 공유라 토큰 자동 반영. 변경 없음 |
| `maps/XK_support_scale_slide.png` | 지도 없음 | 다시 그림. 변경 없음 |
| `maps/*_label_sequence_slide.png`, `.gif` | 다른 담당(label_sequence_fig.py, 17:13–17:16 생성 중) | 손대지 않음. 같은 `S.BASEMAP` 을 읽으므로 다시 돌리면 새 토큰이 적용된다 |
| `*.png`(덱 차트 18개) | 지도 없음 | 100 % 점검: 글자 겹침·잘림 없음, 화살촉 없음. `S26_workflow_path.png` 의 '다음 관측'·'진단' 행 머리는 캔버스 왼쪽 끝에 붙어 있으나 잘리지 않았음(덱 배치 상자 안). 변경 없음 |
| `method/*.png` | 다른 담당(method_figs_v4.py) | 축소 시트로만 보았다. 바탕 지도는 `_C["neutral"]["land"]`(#F2F2F2) 를 직접 읽으므로 새 토큰을 받지 않는다. 방법 그림 담당에게 `color.map_base.slide` 적용을 넘긴다 |

점검 결과(스크립트): 슬라이드 패널 글자 크기 집합 {12, 13, 14} pt, 겹침 0, 캔버스 밖 0.

## 2. 논문 그림(`outputs/figures/paper/v3_restructure/`)

모두 다시 그렸다(PDF·PNG 600 dpi 같은 이름). pdffonts: FreeSans 계열 TrueType 만, Type 3 없음. 글자 7 pt(패널 문자 8 pt).

| 파일 | 바탕 지도 | 고친 것 |
|---|---|---|
| `Fig1` | 적용(paper 토큰) a·c·d·티베트 삽도 | a: 열쇠 3줄('Permafrost zone' 머리), 위도 라벨 '60°N'. c: 완충 등치선 `buffer_line` 색. d: 블록 테두리를 경위선 회색 → 흰색(진해진 육지에서 라벨 블록 경계가 사라졌음), 'Scoring blocks' 지시선이 이웃 채점 블록 위를 지나던 것을 같은 종류 블록 가로지름 벌점(300)으로 막고 글자 자리 후보 2개 추가. e: 'kNNDM' 라벨이 'Region holdout' 두 줄 글상자와 맞닿던 것을 점 아래 가운데로 옮김(겹침 0) |
| `Fig2`, `Fig3`, `Table1` | 지도 없음 | 100 % 점검: 겹침·잘림·화살표 없음. 변경 없음 |
| `Fig4` | 지도 없음 | b: 'Central Russia' 직접 라벨이 이웃 하위 지역의 CI 막대와 겹쳐 왼쪽 아래로 옮기고 지시선(화살촉 없음, INK_AUX 0.6 pt)을 더함. a·c·d 변경 없음 |
| `Fig5` | 적용 a–d | 육지에 해안 테두리, 위치 삽도 육지 토큰 색. a 의 열쇠('Candidate cells', 크기 열쇠)가 허드슨만 해안선 위에 놓여 읽기 어려워 열쇠 뒤에 흰 바탕(alpha 0.85, 테두리 없음, gid `key_backing`; 글씨 상자 점검 예외)을 깔았다. 위도 라벨 '70°N' |
| `Fig6` | 적용 a·b + 삽도 | 해안선, 열쇠 'Permafrost zone' 머리. 위도 라벨 경선 −10° → −22°(그린란드 동해안 위 → 그린란드해), 축척 막대 (12° E, 70° N) → (−2°, 66° N)(아이슬란드·노르웨이 해안과 겹쳐 보이던 것을 바다로). '60°N' 형식 |
| `Fig7` | 적용 b + 삽도 | 육지 해안 테두리, 삽도 육지 토큰 색, 눈금 '170°W'·'70°N'. 그 밖 변경 없음 |
| `maps/Alaska_ALT_map_v3`, `maps/Lena_ALT_map_v3` | 적용 | 육지·해안·경위선 토큰, 위치 삽도 해안선. 변경 없음 |
| `maps/XL_*` (3) | 적용 | 토큰 자동 반영. 변경 없음 |
| `maps/XK_support_scale` | 지도 없음 | d–f 의 'Stefan − residual ML' 차가 검정·회색이었던 것을 잔차 ML 색(#D55E00, 셀 가중 채움·블록 등가중 빈 원)으로, 제품 선 색을 토큰 `color.products`(CCI #9C7A3C, Wei #3E8E91, YK #6B8E23)로. 설명문은 'filled/open' 표현이라 그대로 |
| `si/FigS7` | 적용 a–d | 육지 토큰 색(해안은 이미 있음). 변경 없음 |
| `si/FigS8`, `S9`, `S10`, `S12` | 적용(Fig 6 의 지도 함수 공유) | Fig 6 과 같은 위도 라벨·축척 자리 이동, 해안선, 'Permafrost zone' 머리. 그 밖 변경 없음 |
| `si/FigS11` | 적용 a·b | 육지 토큰 색. 변경 없음 |
| `si/FigS14`, `S15` | S14 적용(maps_alt_v3 바탕) / S15 지도 없음 | 변경 없음 |

점검 결과(스크립트): Fig 1·4·5·6·7 audit 실패 0, 글자 겹침 0, 캔버스 밖 0, 수치 대조 불일치 0(Fig 1 value checks, Fig 5 F-13, Fig 6 plotted numbers 206, Fig 7 values). SI S7–S12 겹침 0.

## 3. 남은 위험

- `method/*.png` 는 방법 그림 담당의 스크립트가 `color.neutral.land`(#F2F2F2) 를 직접 읽는다. `color.map_base.slide` 로 바꾸려면 그쪽 스크립트를 고쳐야 한다.
- `maps/*_label_sequence_*` 는 같은 시간대에 다른 담당이 만들고 있어 손대지 않았다. 다시 돌리면 새 바탕 토큰이 자동 적용된다.
- Alaska 1 km 지도 PDF(2.5 MB)와 XL Alaska(2.6 MB)는 이전부터 2 MB 를 넘었다(래스터 지도). 이번 변경과 무관하다.
- FreeSans 의 '°' 는 글리프 폭이 넓어 '60°N' 도 '60° N' 처럼 보일 수 있다(글꼴 특성).
