# 슬라이드용 패널(2026-10-05)

논문 그림의 그리기 함수와 자료를 그대로 쓰고 글자(v4: FreeSans, 슬라이드 크기에서 12 pt 이상), 선(v4 lines.slide), 표지 크기만 슬라이드 배치에 맞춘 재렌더다. 새 자료와 새 주장은 없다. 만든 스크립트: `scripts/4_visualization/paper_v3/slide_panels.py`.
방식 S = 슬라이드 배치 상자 크기로 패널을 단독으로 다시 그림, P = 논문 배치에 큰 글자로 다시 그린 뒤 덱과 같은 상자로 자름.
v4 토큰(2026-10-05 오후, design/style_tokens_v4.json): FreeSans Regular(한글 Pretendard Regular), 직접 라벨 13 pt, 눈금 12 pt, 축 이름 14 pt, 축 1.0 pt, 자료 2.2 pt, 방법마다 색(제안 방법 주홍 #D55E00 하나만 강조), 화살촉·삼각 표지 없음. 지도 색표는 그대로.

| 덱 쪽 | 대체하는 자르기 | 새 파일 | 크기(in, 300 dpi) | 방식 | 비고 |
|---|---|---|---|---|---|
| 1 (title) | `crops/f1a.png on the title slide` | `Alaska_ALT_map_v3_c_slide.png` | 5.50 × 5.00 | S | Alaska residual ML ALT (panel c of Alaska_ALT_map_v3), no graticule, thin coastline; text sizes [12.0, 14.0] pt |
| 5, 24 (also 1, 2, 37 if wanted) | `crops/f1a.png` | `Fig1_a_slide.png` | 5.02 × 5.20 | S | Lena Delta zoom rectangle and connector to c and d omitted (c, d are not on these slides); text sizes [13.0] pt |
| 5 | `crops/f1b.png` | `Fig1_b_slide.png` | 4.05 × 5.20 | S | x ticks 0.1, 1, 10 (paper also 0.3, 3, 30); text sizes [12.0, 13.0, 14.0] pt |
| 8 | `deck/assets/paper_report/S08_validation_ladder.png (old five-region chart)` | `Fig1_e_slide.png` | 7.50 × 4.60 | S | Fig 1e data and encodings (XH 24 rows + region holdout 6 rows); stage and method names as direct labels; region symbols alpha 0.5; one-line note on the open symbol; text sizes [12.0, 13.0, 14.0] pt |
| 36 | `crops/f5c.png` | `Fig5_c_slide.png` | 3.39 × 3.41 | P | map only, no text inside the deck crop box; same framing as the crop; paper v4 line widths x 2.14 at slide size |
| 11 | `crops/f6a.png` | `Fig6_a_slide.png` | 4.26 × 3.91 | S | circle 3.96 mm (paper 3.0), map circle 86 mm; column head left to the deck; key, inset names, latitude labels and scale bars only in a (as in the paper); text sizes [13.0] pt |
| 11 | `crops/f6b.png` | `Fig6_b_slide.png` | 4.26 × 3.91 | S | circle 3.96 mm (paper 3.0), map circle 86 mm; column head left to the deck; no text in this panel (as in the paper) |
| 11 | `crops/f6_cbar.png` | `Fig6_cbar_slide.png` | 9.95 × 0.79 | S | bar 180 mm centred at 5.21 in from the left edge (midpoint of the two map images at COL[0] and COL[3]); extend min, ±40 cm as in the paper; v4: direction in the label, no arrow; text sizes [12.0, 14.0] pt |
| 21 | `crops/f7b.png` | `Fig7_b_slide.png` | 5.40 × 4.29 | S | same blocks, colour scale ±5 cm; text sizes [12.0, 13.0, 14.0] pt |
| 36 | `crops/f7b.png` | `Fig7_b_small_slide.png` | 3.80 × 3.02 | S | same blocks, colour scale ±5 cm; text sizes [12.0, 13.0] pt |
| 21 | `crops/f7c.png` | `Fig7_c_slide.png` | 4.92 × 4.80 | S | weighting key under the panel; block-equal bar 1.6 mm below the cell-weighted bar (paper 0.8 mm); text sizes [12.0, 13.0, 14.0] pt |
| new slide (XC workflow contrast; no deck page yet) | `none (Fig 7e was a placeholder)` | `Fig7_e_slide.png` | 6.00 × 4.00 | S | Fig 7e data and encodings (XC-1 n 40, 160; XC-2 n 10, 40, 160; regions worse than recalibrated Stefan by > 0.5 cm as faint region symbols); weighting key and region key above the panel; text sizes [12.0, 13.0] pt |

배치 상자(deck/build_paper_report.py): 1쪽 표지 그림 3.80 × 3.80 안(새 표지 지도는 요청 크기 5.5 × 5.0); 5쪽 f1a 높이 5.20, f1b 높이 5.20; 8쪽 검증 사다리는 요청 크기 7.5 × 4.6(현재 S08 차트는 12.0 × 5.1 전폭); 11쪽 f6a·f6b 높이 약 3.91, f6_cbar 폭 9.95 (띠를 COL[0] 에 두면 막대 가운데가 두 지도 그림 사이 가운데); 21쪽 f7b 폭 5.40, f7c 높이 4.80; 24쪽 f1a 높이 5.20; 36쪽 f7b·f5c 3.80 × 3.40 안. 파일 크기는 그 상자에 맞췄다.

Fig 6 지도는 처음에 P 방식으로 만들었으나 큰 글자가 덱 자르기 상자 밖으로 나가 잘려(b 위 열 머리 조각, 삽도 이름, 컬러바 이름) S 방식으로 바꿨다. 지도 원 지름 86 mm(덱 자르기의 92 mm 보다 7 % 작음), 대상 원 지름 3.96 mm(논문 3.0 mm × 1.32).
