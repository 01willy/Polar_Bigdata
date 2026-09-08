# 논문화 작업 계획 — Scientific Reports 투고

**작성**: 2026-08-31 (사용자 확정: 투고지 = Scientific Reports)
**전신**: 저널 평가는 `gpt/handoff/20260831_1105-final-deck-audit-journal.md` §5·§7, 메모리 `paper-journal-strategy` 참조.
**원고 기반**: `outputs/report/main.tex` (13쪽 국문 보고서) → 영문 논문으로 재구성.

## 0. 투고 전략 요지

- **지면**: Scientific Reports (IF 4.9, impact-neutral 심사 = 기술적 건전성만 평가, 음성 결과 명시 수용).
- **프레이밍**: "딥러닝 연구"가 아니라 **physics-guided machine learning / 희소 관측 하 공간 일반화 벤치마크**로 서술.
  제목 후보: *Cross-regional generalization of active-layer thickness models under sparse observations: evaluating physics-generated pseudo-labeling*.
- **직접 비교 대상**: Gautam et al. 2025 (Sci Rep, 알래스카 RF vs Stefan). 같은 지면·같은 지역이므로 서론에서 차별점을 선제 명시해야 자기중복 심사를 통과한다.
- **본문 기여 3개로 스코프 재단**:
  - C1. Stefan 유사라벨 증강 + 상수 라벨 대조로 물리 정보 순가치 분리(+ 부정확 물리의 해악 정량화)
  - C2. LORO 전이에서 구조 정교화 3종 반증 + 편향·산포 분해 + 위성 제품 앵커 결합(4/4 조합 동일 방향, **비유의함을 정직 서술**)
  - C3. CQR 보정 예측구간(커버리지 검증) + AOA 동봉 지도
- **보충자료로 강등**: 3D 온도장(R² 0.47), 연별 지도(anomaly 스킬 0), KPDC 현장 대조(사례 보고).
- **버릴 수사**: "정보원 다양성이 지렛대" 단정(CI 0 포함), "시간·깊이 확장" 헤드라인, 13.33 cm의 SOTA 함의.

## 1. P0 — 통계·실험 보완 (계산만, GPU 불필요, 수일)

| # | 작업 | 내용 | 재료 |
|---|---|---|---|
| 1 | **winner's curse 해소** (W2, 필수) | 185조합×λ 탐색을 중첩 선택(내부 블록 CV로 구성 선택 → 외부에서만 채점)으로 재평가. 또는 등가중 앵커+λ=0.25 고정을 **사전 지정 주 추정치**로 교체(24.11→23.27)하고 탐색 결과는 부차로 강등 | `data/processed/s12_hybrid_transfer_shard0-3.csv` |
| 2 | **대조군 사다리** (W6, 필수) | 셔플 Stefan(순서 섞은 물리값)·TDD 선형식(제곱근 없이) 조건 추가 → "물리 정보 기여 vs 1-파라미터 저복잡도 기여" 분리. (옵션) 대상 지역 CCI 평균 상수 | `scripts/3_deep_learning/aug_*` 파이프라인 재실행 |
| 3 | interval score·조건부 커버리지 (W12) | 구간 폭 53.6 cm의 정보량을 proper scoring rule로 보고 | `data/processed/s11_conformal_oof.csv` |
| 4 | 라벨 정의 민감도 (W8, 권장) | 탐침/융해깊이 정의 통일 부분표본으로 결합 이득 재확인 | `data/processed/dl_dataset_cell_v2.csv` |

## 2. 원고 작업 (P0와 병행)

1. **관련연구 재배치** (W5, 필수 — 미이행 시 novelty 기각 위험): 물리 모형 출력을 학습 신호로 쓰는 계보 인용 — Read et al. 2019(WRR, PGDL 합성라벨 사전학습), Jia et al. 2021(PGRNN), Willard et al. 2022(ACM CSUR), Liu et al. 2023(CRST, PI-LSTM) + 자기훈련(Lee 2013)·지식증류(Hinton et al. 2015). 차별점은 **"라벨 없는 지역으로의 공간 표적 증강 + 순가치 대조 설계 + 음성 물리 정량화"**로 좁혀 서술.
2. **Gautam 2025 대비 절**: 차별점 = 공간블록 CV(vs 무작위 분할), LORO 전이 벤치마크, 증강 통제 설계, 커버리지 검증 UQ. 서론에 명시.
3. **용어 교정**: CCI는 "독립 관측"이 아니라 **"산출 구조가 다른 준독립 추정"**(CryoGrid 모델 산출물, ERA5 강제력 일부 공유). r=10 증강이 실측 없는 셀의 반복 복제(가중 조절)임을 방법에 명시.
4. **비유의 결과의 정직 서술**: 앵커 결합 개선은 "4/4 조합 동일 방향의 경향, 95% CI [-2.22, +2.76]" 그대로 보고. Sci Rep은 impact-neutral이라 이 서술이 결격이 아님.
5. **영문화 + 형식**: Sci Rep 형식(분량 제한 느슨). 데이터·코드 공개 성명 필수 준비 — GitHub 공개 + Zenodo DOI 권장.
6. **데이터 가용성 절의 KPDC 식별자 정확 표기**: 지온 프로파일 = KOPRI-KPDC-**00002707**(2024)·**00002955**(2025), 코어 = 00002125. (덱·보고서는 00002125만 표기 — 논문에서는 반드시 구분)

## 3. Sci Rep 실무 확인 사항

- **APC $2,850** — 소속기관의 Springer Nature 한국(KESLI) 협정 적용 여부를 도서관에 확인(협정 페이지에 Sci Rep 포함 여부 미명시, 기관별 상이).
- Nature Portfolio 통계 보고 기준(효과크기·CI·정확 p값 요구) 원문 확인.
- 심사 기준: 기술적 건전성만. 음성 결과·재현 연구 명시 수용(guide-to-referees).

## 4. P1 — 선택(검정력 강화, Sci Rep 필수 아님 / 후속 상위지 재도전용)

- 전이 지역 2곳 이상 추가(스칸디나비아·캐나다 군도 CALM 클러스터) → 블록 수 확충 후 CI 재산정. CI가 0을 벗어나면 PPP/TC 상향 재고 가능.
- GIPL2·Obu 2019·Ran 2022 제품과 동일 셀 head-to-head(산출물 우위의 외부 준거).
- 지역 적응형 계수 E(소량 라벨 재적합 시나리오) — 개선폭 자체를 키움.

## 5. 비차단 정합 이월 (원고 작성 중 함께 처리)

- `report_overview_figure.py`의 n_grid=892,865 하드코딩 → `report_alt_map_hires.py`가 격자 셀 수를 meta JSON에 기록하게 하고 그 값을 읽도록.
- `figures/figure_spec.json`에 재생성 그림 4종(report_overview, transfer_loro_summary, s12_hybrid_summary, alt_annual_fields) 스펙 등록.
- `submission/`(예선 동결본)은 수정 금지. 논문 저장소 공개본은 현행 `scripts/`에서 재구성.

## 6. 진행 순서 제안

1주차: P0-1(winner's curse) + P0-2(대조군 사다리) → 결과에 따라 초록·결론의 주 추정치 확정 → 2주차: 원고 골격(영문) + 관련연구 + Gautam 대비 → 3주차: 그림 영문판 재생성·보충자료 편성·데이터/코드 공개 준비 → 내부 검토 후 투고.
