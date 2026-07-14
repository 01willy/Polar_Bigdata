# Handoff: 중간보고 PPT 전면 재구성 + 연구방향·데이터구조 정밀 검증

**Project**: Polar Bigdata — Permafrost ALT map + shallow 3D thermal (DL)
**Date**: 2026-07-14 11:39
**Session focus**: 중간보고 PPT를 EMP·Digital Rock 논문 수준으로 재구성하고, "학습시간이 왜 짧나·데이터를 제대로 쓰나"라는 사용자 의문을 문헌 20편+와 파이프라인 실측으로 정밀 검증해 다음 세션(GPU 6,7,8,9) 실험계획 P0~P5를 확정.
**Author**: Claude Opus 4.7 (1M) + 01willy

---

## 1. TL;DR
- **핵심 검증**: 우리 학습데이터 6.6MB tabular(14,348셀×36피처)는 ALT ML 관행 범위·상위권. 병목은 데이터 부피가 아니라 **라벨 희소성 + 공변량 정보량**(문헌·분산분해 정합). 학습 초~분은 정상(GBM). "진짜 이미지 딥러닝"(원자료 CNN·파운데이션)은 별도 세계이며 우리는 미실행.
- **자료구조 발견**: ALT 라벨(94% 알래스카)과 지중온도 라벨(9개국 260사이트, 스위스·러시아 72%)은 **별개 관측망**이라 학습 지역이 어긋나 있음. 알래스카만 둘 다 겹침(40중 35). 통합 열쇠 = 지중온도 0°C 통과깊이로 ALT 라벨 유도.
- **파이프라인 실측**: 3D=GBM 조건장(기후+깊이만 입력, 신경장은 폐기), 페이지7 ALT지도=ERA5 8개만 사용, patch-CNN 기시행(17.2≈GBM 17.7). InSAR/PolSAR는 알래스카만 존재.
- **PPT v3**: 흰 표지·Pretendard·EMP톤 그래프 6종·2.5D 단면(3D 대체)·데이터썸네일 아키텍처·지표+상세설명 밀도. `deck/build_midreport.py` 18슬라이드.
- **다음 세션 확정**: GPU 6,7,8,9로 P0(인벤토리·오차지도)→P1(전지역·전공변량 통합 ALT 재학습)→P2(Stefan+DL 잔차)→P3(3D 전공변량+연속성)→P4(임베딩)→P5(이미지 diffusion 트랙).

## 2. Context
- 직전 핸드오프: `gpt/handoff/20260713_1238-cell-reanalysis-tlite-gate-deck-v2.md`(셀 재분석·T-lite 게이트·덱 v2).
- 이번 세션 동기: 사용자가 발표덱 품질(디자인·3D)에 강한 불만 → 전면 재구성. 이어 "학습시간 37초~2.7분이 이상하다, 22GB인데 6.6MB만 쓰나, 데이터 제대로 학습되나"라는 근본 의문 제기 → 정밀 검증 필요.
- 대회 맥락: `docs/CONTEST_PLAN_2026.md`(예선 07-31), 두 트랙(연구 지속 + 대회 스냅샷).

## 3. What we did
- **연구방향 검증(멀티에이전트 문헌조사 20편+ + 파이프라인 실측)**
  - Files: `docs/EXPERIMENT_PLAN_2026-07-14.md`(공변량 인벤토리·라벨분포·Q&A·자료구조 발견)
  - Result: 6.6MB=관행 확인(Gautam 68사이트, Ran ~1000점 < 우리 14,348). 3D는 GBM 조건장·기후+깊이만. patch-CNN 17.2≈GBM 17.7. ALT/지중온도 별개망 확인.
- **중간보고 PPT v3 재구성**
  - Files: `deck/build_midreport.py`, `deck/mk_midreport_figs.py`(그래프6), `deck/mk_architecture_fig.py`, `deck/mk_cross_section.py`, `deck/report_lib.py`(Pretendard), `scripts/4_visualization/render_thermal3d_pyvista.py`
  - Result: 18슬라이드, 렌더 `deck/render/permafrost_midreport.pdf`. 폰트 근본수정(Pretendard 미인식→Noto 대체가 자간깨짐 원인). 3D→2.5D 단면. 페이지카운터 버그 수정.
- **데이터 확보(백그라운드)**: ALLena 시베리아 9,186점(`data/raw/allena/`), TPDC QTEC 티베트(`data/raw/tpdc_qtec_zenodo5009871/`). ds2332 기보유 확인, SMALT=우리 22만점과 동일(중복), FireALT 서버장애.
- **문서**: EXPERIMENT_ROADMAP(E1~E7), CONTEST_PLAN_2026 v2, SESSION_HANDOFF 갱신(GPU 6,7,8,9).

## 4. Key numbers (this session — 전부 기존 감사완료 CSV·실측 근거)
| Method | Domain/Case | Metric | Value | Source |
|---|---|---|---|---|
| 앙상블 | 공간블록/LORO | RMSE | 16.95 cm | data/processed/model_tournament_results.csv |
| 위경도 대조군 | 셀 LORO | skill | 14.7% | data/processed/alt_ablation_cell_results.csv |
| +InSAR(물리최고) | 셀 LORO | skill | 12.7% | 동일 |
| ERA5 실측 공변량 | 전이(LORO) | RMSE | 108.5→87.3 cm | data/processed/stage2_era5_rescore.csv |
| patch-CNN vs GBM | 공간블록 | RMSE | 17.2 vs 17.7 | data/processed/dl_cnn_results.csv |
| 신경장 vs GBM | 사이트분리 | RMSE(°C) | 2.17 vs 1.33 | data/processed/b1_neural_field_results.csv |
| 학습데이터 | 셀 tabular | 크기 | 6.6MB(14,348×36) | data/processed/dl_dataset_cell.csv |
| 원자료 | raw | 크기 | 22GB(PolSAR7+ReSALT6.9+DEM4.3…) | data/raw/ |
| 지중온도 라벨 | 3D | 사이트 | 260(9개국) | data/processed/ground_temp_all.csv |

## 5. Decisions made
- **PPT 3D → 2.5D 도시**: 실척 3D는 깊이90m/수평950km라 왜곡·회색면 문제 → 위도-깊이 단면+0°C 등온선+깊이슬라이스로 교체(사용자 지시).
- **덱 폰트 Pretendard 확정**: ~/.fonts 설치 확인, LibreOffice fontconfig 렌더. report_lib SERIF=Pretendard ExtraBold, SANS=SemiBold.
- **실험 우선순위 확정**: 정보량 확충(P1·P4) > 물리결합(P2) > 다지역(P3) > 이미지CNN(P5 트랙). GPU를 "큰 모델"이 아니라 "정보+물리"에 투자.
- **ALT↔지중온도 통합 방침**: 지중온도 0°C 깊이로 ALT 라벨 유도해 다지역 확장.

## 6. Open questions / blockers
- **다지역 통합 학습이 알래스카 특화보다 나은가**: 툰드라 물리 유사성 가정. P1에서 통합 vs 특화 정직 비교로 해소.
- **3D 스위스(알프스) 편중 학습이 알래스카 툰드라에 전이되나**: P3에서 검증.
- **이미지 조건 diffusion/flow가 라벨 희소(14,348)로 학습 가능한가**: P5에서 파운데이션 임베딩·패치축소·Stefan 조건으로 우회 시도. 안 되면 음성 결과.
- **전 공변량 ALT 지도가 ERA5만 지도보다 나은가**: P0/P1에서 재생성해 비교.

## 7. Next steps (prioritized) — GPU 6,7,8,9
1. **P0 데이터 인벤토리 세계지도 + 6모델별 ALT 예측·오차 지도** — owner: Claude, 즉시(CPU). 재료 `model_tournament_predictions.csv`. Q5·Q9 시각화.
2. **P1 전 공변량+전 지역 통합 ALT 재학습** — owner: Claude, GPU. ALLena·QTEC 파싱→지중온도 0°C로 ALT 유도→DEM+InSAR+PolSAR+CCI 전부 투입→6모델 재비교. 게이트: 공간블록·LORO 개선.
3. **P2 Stefan 물리 base + DL 잔차** — 2단계(물리 사전학습→시추공 파인튜닝). 근거 PI-LSTM 27~69%↓.
4. **P3 3D 전 공변량+연속성 DL → 매끄러운 0°C 등온면 tri-mesh** — 등온면 끊김 해결.
5. **P4 AlphaEarth 임베딩 공변량 추가** — 게이트: GBM 이겨야 채택.
6. **P5(트랙) 이미지 조건 diffusion/flow** — 고위험, 과적합 게이트.
- 모든 결과는 전문 mapping·시각화 후 PPT(`deck/build_midreport.py`) 반영. 선행연구 모식도·DL 모델 슬라이드 추가.

## 8. Pointers
- 실험계획: `docs/EXPERIMENT_PLAN_2026-07-14.md` (공변량 24축·라벨분포·P0~P5·자료구조 발견)
- 로드맵: `docs/EXPERIMENT_ROADMAP.md`(E1~E7) · 대회: `docs/CONTEST_PLAN_2026.md`(v2)
- 현재상태: `SESSION_HANDOFF.md`(갱신됨, GPU 6,7,8,9)
- PPT: `deck/build_midreport.py` → `deck/render/permafrost_midreport.pdf`, 그래프 `deck/mk_*.py`
- 신규 데이터: `data/raw/allena/`(시베리아), `data/raw/tpdc_qtec_zenodo5009871/`(티베트)
- Active jobs: none
- 직전 핸드오프: `gpt/handoff/20260713_1238-cell-reanalysis-tlite-gate-deck-v2.md`

## 9. Caveats for GPT
- **"17cm 물리하한"·"12.97 SOTA 돌파" 폐기**: apparent floor=공변량 정보병목(비가역 하한 ~7cm). RMSE 옆 R²·skill·ALT범위·CV종류 병기 필수.
- **within-Alaska 17cm ≠ 전이 LORO 87.3cm**: 다른 과제. 섞지 말 것.
- **3D 엔진=GBM 조건장**(신경장 폐기 유지). ALT(계절 최대융해)≠MAGT(연평균장).
- **학습데이터=6.6MB tabular**(22GB는 전처리 입력). 짧은 학습시간은 정상.
- **ALT(알래스카)와 지중온도(9개국)는 별개 관측망** — 같은 지역에 둘 다 있다고 가정 금지. 알래스카만 대부분 겹침.
- **InSAR/PolSAR는 알래스카만** 존재 → 다지역 예측 시 결측.
- GPU 최신 지시 **6,7,8,9**(2026-07-14). 사용 전 nvidia-smi.
