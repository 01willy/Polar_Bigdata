# v7 덱 쪽 목록(리뷰용)

갱신 2026-10-15: F4 패널 e(모델 구조) 추가로 R03 의 크롭이 cde → cd 로 바뀌고 R03b(e·f) 쪽이 추가됐다(deck_spec_v8 R03·R03b 와 같음). 쪽 PNG(deck/render/*_pages)는 build_v7.py 재실행 뒤 갱신된다.

- page-01: S01 | 해마다 달라지는 영구동토 활동층 두께의 1 km 예측 | assets None | box False
- page-02: N01 | 활동층 두께와 관측 현황 | assets ['F1_study_area__a.png', 'F1_study_area__e.png'] | box False
- page-03: N02 | 입력의 시간 정의 | assets ['F2_problem_time__ab.png'] | box False
- page-04: N03 | 목표 · 기존 한계 · 개선 · 기대 효과 | assets ['G1_goal_slide.png'] | box True
- page-05: N04 | 물리 기준: Stefan 관계 | assets ['F2_problem_time__cd.png'] | box False
- page-06: N05 | 세 지역의 관측 셀 | assets ['F1_study_area__b.png', 'F1_study_area__c.png', 'F1_study_area__d.png'] | box False
- page-07: N06 | 관측 연도와 자료 기간 | assets ['F1_study_area__e.png', 'F1_study_area__f.png'] | box False
- page-08: M01 | 방법 구조 | assets ['F3_method__ab.png'] | box False
- page-09: M02 | 평가 설계 | assets ['F3_method__cd.png'] | box False
- page-10: R01 | 그 해 기후 입력의 효과 | assets ['F4_results_core__a.png'] | box False
- page-11: R02 | 검증 분할에 따른 오차 | assets ['F4_results_core__b.png'] | box False
- page-12: R03 | 잔차 결합의 이득과 조건 | assets ['F4_results_core__cd.png'] (deck-native deck/F4_results_core__deck_R03.png: c 공변량 사다리 · d TDD 3분위) | box False
- page-13: R03b | 물리 정보의 위치별 이득과 학습기 | assets ['F4_results_core__ef.png'] (deck-native deck/F4_results_core__deck_R03b.png: e 모델 구조 직접 ML · 물리 입력 ML · 물리 유사라벨 · 물리 + 잔차 ML − 재보정 Stefan, f 학습기 5종) | box False
- page-14: R04 | 기존 제품과 같은 셀 · 같은 해 비교 | assets ['F5_products_years__ab.png'] | box False
- page-15: R05 | 연도별 평가 | assets ['F5_products_years__cd.png'] | box False
- page-16: R06 | 장기 관측 지점의 연도별 시계열 | assets ['F5_products_years__ef.png'] | box False
- page-17: R07 | 연도를 명시한 알래스카 1 km 지도 | assets ['F6_maps__abcd.png'] | box False
- page-18: R08 | 1 km 상세와 적용 가능 영역 | assets ['F6_maps__d.png', 'F6_maps__e.png'] | box False
- page-19: R09 | 레나 델타 2016년 지도 | assets ['F6_maps__fg.png'] | box False
- page-20: R10 | 해마다 다른 지도 | assets ['F7_series_significance__ab.png'] | box False
- page-21: C01 | 결론 | assets ['F7_series_significance__c.png', 'F7_series_significance__d.png'] | box True
- page-22: C02 | 의의와 활용 | assets ['F1_study_area__a.png', 'F7_series_significance__a.png'] | box True
- page-23: S99 | 참고문헌 | assets None | box False
