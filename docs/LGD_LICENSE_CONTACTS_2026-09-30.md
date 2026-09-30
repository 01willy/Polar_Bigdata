# LGD 약관 미확인 자료원과 허락 요청 대상 (2026-09-30)

근거: LG 개정 15 (m)·(r). 허락 기록 파일 `data/processed/lgd/lgd_license_permission.json` 에 아래 네 자료원이 모두 '허락'으로 적혀야 LGD 주 판정이 전체 판이 된다. 하나라도 없으면 약관 확인분 판이 주 판정이고, 이때 새 얕은 지역 NAtlantic 은 적격 규칙(채점 블록 합집합 6 < 8)으로 주 판정 풀에서 빠진다.

| 자료원 | 셀 수(대상) | 현재 상태 | 요청할 내용 | 연락처(메타 기록) |
|---|---|---|---|---|
| calm_web_subsites | NAtlantic 13 | CALM 누리집에 CC 표기·재배포 조항이 없다. 인용문만 지정 | 논문(Sci Rep, 개방형)에서 셀 평균 ALT 파생값의 사용과, 파생 표(셀 단위)의 공개 저장소 기탁 허락 | CALM 누리집 https://www2.gwu.edu/~calm/data/north.htm 의 사이트 담당자(Abisko 등 사이트별 인용문의 저자) |
| cusp_v1_1 | NAtlantic 6, 캐나다 확충 5 | 저장소 LICENSE 는 저작권 고지만 있고 이용 허락 조항이 없다(원 출처는 CC0, CC BY-NC-SA 3.0 혼재) | 같은 내용. 원 출처별 약관을 따라도 되는지 | https://github.com/jonschwenk/cusp (Triad National Security, LLC 저작권, 관리자 J. Schwenk) |
| nsidc_ggd353_thawtube | 캐나다 확충 33(융해관) | 'Please consult prior to use in publication' 문구. 이용 허락 표시 없음 | 논문 사용 사전 협의(요구 문구 그대로), 파생 표 기탁 허락 | Geological Survey of Canada(자료 책임 기관), NSIDC GGD353 v6, doi:10.7265/7m84-k262 |
| ru_yamal_walker | 러시아 W 1(Kharasavey-1 보고서 값 10행) | PANGAEA 표는 CC BY 3.0, 보고서 PDF 값은 약관 표기 없음 | 보고서 값의 사용 허락. 또는 이 1셀을 약관 확인분 판처럼 빼고 전체 판을 정의하는 방안 검토 | Walker, D. A. 외(2009), PANGAEA.842711 관련 보고서 |

- 연락은 사용자가 한다. 연락 시각과 답은 LG 개정 15 (m)3 아래에 적는다.
- 허락 기록 파일의 형식은 `scripts/2_evaluation/h52_lgd_pool.py` 의 허락 확인 함수가 정한다(자료원마다 granted 여부, 날짜, 근거).
- 투고 전까지 답이 없으면 원고는 약관 확인분 판을 주 판정으로 쓰고, 전체 판은 SI 에 '약관 확인 전 자료 포함' 표지와 함께 둔다(LG 개정 15 (m)2). 이 선택은 결과를 보고 바꾸지 않는다.
