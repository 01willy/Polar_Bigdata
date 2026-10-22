# 논문 v15.1: 그림 지적 11항 반영 · 4차 제안 15안 · 저자 메타 (2026-10-22 15:00)

## 요약
- 사용자 그림 지적(Fig 2c 범례, Fig 3 메시지 · 배치, Fig 4a 범례 · 4c 방어, Fig 5 격자 · 색, Fig 6 레나 값 · 문자, Fig 7 라벨 · d · e · a–c, P3_G 제외, 3차 제안 반려)을 모두 반영하고 묶음별 새 제안 3개씩 15안을 실자료로 렌더링하였다.
- 색표 `color.paper_v15`(제품 · 지역 채도 낮춤) 로 본문 7장 · 보충 11장(S11 신설) 재렌더링. Fig 7d 는 '지도 편차 대 TDD 편차' 산점(r 0.92 / 0.96 / 1.00) 으로 교체, 관측 셀 편차 비교는 제거(S10 안내).
- 빌드 영문 26쪽 · 국문 22쪽 · 보충 22쪽, 오류 0; 본문 4,485단어, 초록 200단어, 압축본 1.1 MB. 납품 `deliverables/` 갱신.
- 저자: 극지연 'Center of Remote Sensing and GIS, Korea Polar Research Institute, Incheon 21990', 교신 yc.cho@snu.ac.kr, 서울대 08826.

## 정본
- 조치표 · 제안 · 글자/색 점검: `docs/FIGURE_PROPOSALS_v15_2_2026-10-22.md`
- 감사 16절: `docs/FIGURE_AUDIT_v12_2026-10-18.md`; 점검표 말미 12항: `docs/REVIEW_RESPONSE_v15_2026-10-21.md`
- 사례집(에이전트, 30편 · 40여 그림): `docs/research/2026-10-22/figure_exemplars_round4.md`
- 원고 `paper/manuscript/{en,ko}_v15/`, 그림 `outputs/figures/paper/v15/` · `v15_si/` · `v15_proposals/`(3차 반려안은 `_rejected_round3/`), 스크립트 `scripts/4_visualization/paper_v15/`(fig3_accuracy, fig4_conditions, fig5_products_v15, fig6_product_maps, fig7_maps, si_s11_obs_pred_density, proposals_v15_2a · _2b)

## 수치(제안 렌더에서 확인)
- 반복 관측 지점(5년 이상): 알래스카 181, 레나 25, 캐나다 7.
- 블록별 물리 잔차 − Stefan RMSE: 이득 블록 49/73 · 12/16 · 21/26, 지점-연도 가중 −0.2 / −2.3 / −3.4 cm.
- 공통 지점-연도에서 우리 지도가 제품보다 관측에 가까운 셀: 레나 82 % (Wei) · 85 % (Liu), 알래스카 52 · 50 %, 캐나다 53 · 42 %.
- 우리 지도 − 제품(Fig 6 연도): Wei −8.2 / −17.9 / −27.2 cm, Liu −21.3 / −19.2 / −34.0 cm (알래스카 / 레나 / 캐나다).

## 다음
1. 사용자의 본문 · 방향 피드백(그림 작업 뒤 주기로 함)과 4차 제안 선택.
2. 채택안의 본문 규격화(지도형 P3_2 · P45_2 · P67_3 는 step 1–2 재렌더) → `tools/finish_paper_v15.sh`.
3. 저자 메타 잔여: 네 저자의 영문 표기 · ORCID(선택), 기여 문장, 감사의 글 · 과제 번호, 자료 · 코드 DOI, 공저자 승인.
