# Handoff: 잔여 단계 S6~S11 완주 + E1·E2(피드백 반영) + 예선 제출물 조립

**Project**: Polar Bigdata — Permafrost ALT map + shallow 3D thermal (DL)
**Date**: 2026-07-28 10:07
**Session focus**: 멀티충실도 로드맵 잔여 6단계(S6~S11) 완주, 사용자 피드백 2건(co-kriging 비교·시계열 예측)을 E1·E2로 실증, 사용자 지적 9건 반영해 그림·보고서 전면 재작업 후 예선 제출물(LaTeX 논문·발표덱·Word) 조립.
**Author**: Claude Opus 4.7 (1M) + 01willy

---

## 1. TL;DR (≤5 bullets)
- **잔여 단계 완주**: S7(KPDC 검증)·S9(timelapse)·S11(종합 UQ)·E1(co-kriging) 채택, S6(source-aware)·S8(mixture) negative, S10(얕은3D) 재현, E2(계절내 D(t)) partial. 누설 pytest 16개 통과 유지.
- **핵심 재확인**: 전이(LORO 매크로)에서 **어떤 증강·구조·mixture·정통 공간보간도 Stefan 물리 앵커(21~22cm)를 넘지 못함**. in-domain 최저 13.33cm(S4 앵커+잔차). variogram 지역차 = covariate shift의 공간통계판(E1).
- **시계열 질문 규명**: 연도 간 최대 ALT 외삽은 예측 신호 없음(within-site corr 0.06)이 문헌 정합. 계절 내 융해진행 D(t)는 예측 가능 구조이나 2년 소표본서 persistence(17.5cm)를 물리·GRU 미달(E2 partial). 시간 성능은 방법이 아니라 문제정의·데이터밀도 의존.
- **사용자 지적 9건 반영 재작업**: ALT 관측 밀도 문제(데이터 초집중 규명+배로 250m 확대 inset 4,381셀), 고해상 자산 편입, 얕은3D 논문형(깊이슬라이스/fence/트럼펫), 강점·차별성·의의·한계극복 중심 서술.
- **제출물**: LaTeX 논문 `docs/report/main.pdf`(주), 발표덱 `deck/render/permafrost_summary.pdf`(10장), Word `outputs/CONTEST_REPORT_2026.docx`(보조). 블라인드·수치 CSV대조·문체 규약 통과. push 완료(`f3806ce`).

## 2. Context
- 직전 핸드오프: [20260724_1327-s0-s3-multifidelity-pipeline.md](20260724_1327-s0-s3-multifidelity-pipeline.md) — S0~S3 구현, S4·S5 negative, S3 증강비율 버그 수정. 다음으로 S6~S11 제시.
- 본 세션: (a) 잔여 S6~S11을 병렬 구현·실행·적대검증·표적수정. (b) 사용자 피드백 2건 반영 — 인터폴레이션(co-kriging) 비교=E1, timelapse에 따른 ALT 계산=E2 계절내 D(t)로 재정의. (c) 예선 마감(2026-07-31) 대비 제출물 조립. 사용자 지적 9건(md 아닌 논문형·AI틱 문체·저해상 그림·분석나열·논문구조·구린 3D·고해상 자료·ALT 점 표시·LaTeX)에 대응해 그림·보고서 전면 재작업.
- GPU: 세션 중 3,4,5→6,9→7로 여러 번 재배정(타 사용자 점유 회피). E2는 6,9→7 이전 재실행.

## 3. What we did

### S6 source-aware multi-fidelity (A5) — negative
- **Action**: 공유 인코더+source별(b_s, logσ_s)+Gaussian NLL. Stefan·CCI 2 clean 소스(overlap 100%·99.5%).
- **Result**: LORO 게이트 sa_fusion 21.53·sa_z 21.84cm로 baseline 최고(naive pooling 21.11·Stefan앵커 21.26) 미달. σ헤드 cov90 0.996은 폭 발산(473cm) 무정보. 정확도·UQ 게이트 모두 미달 → negative. 부산물: in-domain sa_z 14.35(pooling의 in-domain 파괴 회복), b_stefan≈0 정합·b_cci 과소추정(식별성 한계). 보고서엔 비교표 1행+소스신뢰도 진단으로만.
- **Files**: `scripts/3_deep_learning/s6_source_aware_mf.py`, `src/polar/source_aware.py`, `s6_source_aware_{results,meta}`.

### S7 KPDC 콘슬 검증 — 채택 (대회 KPDC 주활용)
- **Action**: KPDC Council 일별 다층 지중온도 19 프로파일(2023-2025)에서 표층연결 0°C 등온선 ALT 유도.
- **Result**: 2025 비검열 부분집합 조건부 중앙값 136.4cm(n=5, CI 75.6-153.1). 38시즌 중 23 우측검열(구간검열 정량표). Stefan 59.6·S1모델 53.8cm는 사이트 관측 스펙트럼(ABoVE 35.0↔코어 81.3↔유도 121-153) 내부이나 16층 유도 대비 과소 → 점검증 대표성 잡음 지배. 단일E Stefan 콘슬 약 1.7배 과대(35 vs 59.6)=E(x) 동기. **알래스카 in-domain, 전이 근거 아님**.
- **Files**: `scripts/1_data_prep/s7_parse_kpdc_council.py`, `scripts/3_deep_learning/s7_kpdc_validation.py`, `s7_kpdc_{results,meta}`, `s7_council_{daily_temp,alt_derived}.csv`.

### S8 mixture-of-physics — negative(진단)
- **Result**: 물리5종+공변량 게이트 mixture가 단일 Stefan 못 이김(LORO 29.22 vs 22.24). Alaska LORO fold 파탄(36.48cm, 비-Alaska 학습 게이트가 Alaska 61%를 상방편향 p5_λ 배정). oracle 하한 15.48cm(모델 아님). 전이서 전문가 선택 불가능성 정량 증거. `s8_mixture_{results,meta}`.

### S9 timelapse — 채택
- **Result**: 2010-2024 연별 ALT 프레임 15장 재렌더(mask_ocean·고정색축 40-75cm·oslo_r)+adaptive palette GIF+anomaly(broc) GIF. temporal holdout RMSE 14.97cm[14.72,15.22]·R²0.338. 연 anomaly corr +0.059(예측불가, 각주). 시간정합=연도별 ERA5×연도별 ALT. `s9_timelapse_meta.json`.

### S10 얕은 3D — 재현 + 논문형 재작성
- **Result**: `shallow3d_alaska.py` 재실행이 기존과 바이트단위 동일(RMSE 2.66°C·R²0.4688·0°C→ALT r0.282). 온도장만 검증 성립, 유도 ALT r0.28이나 R²<0(절대정합 미성립). **사용자 지적6 반영**: 3D warp 대신 논문형 재작성 — `s10_depth_slices`(0.5/1/1.5/2m 깊이슬라이스 4패널, vik 0°C중심), `s10_fence_sections`(경도/위도-깊이 단면), `s10_profiles_trumpet`(대표4지점 T(z)). 3D warp는 발표 보조로 강등.

### S11 종합 비교표 + conformal UQ — 채택(헤드라인)
- **Result**: quantile CatBoost raw cov90 44.6%(seed평균)→train 블록내부 CQR 93.4%[88.9,96.6] 보정(폭 14.7→53.6cm). 16행 증강방식 비교표(전 수치 CSV 재계산): LORO서 어떤 증강·구조도 Stefan 앵커 유의 초과 못함, in-domain 최저 S4 13.33. 다축 검증표(공간블록·LORO·temporal·누설pytest16). transductive 라벨링(pool_mlp·고정증강·S5 pretrain 3행). `s11_comparison_table.csv`·`s11_conformal_{results,meta}`.

### E1 co-kriging/RK baseline — 채택 (피드백: 인터폴레이션 비교)
- **Action**: 정통 공간보간(OK·IDW·RK)을 GBM·Stefan과 동일 fold로 정식 비교. pykrige/gstools 사용.
- **Result**: in-domain OK 15.77·IDW 15.80 ≈ GBM 17.74(경쟁), 전이 붕괴(OK 29.40·IDW 50.92·RK 36.30·GBM 41.37). **Stefan 앵커만 양축 최선(14.46/21.26)**. kriging 전이붕괴=variogram(range≈899km·sill 442cm²) 지역차=covariate shift 공간통계판. OK 레나 22.3은 실력 아닌 mean 회귀 아티팩트(range 밖). 정정: STEF in-domain √TDD 셀 오정렬 버그(14.39→14.46), RK 앙상블 아티팩트(15.25→seed평균 17.23).
- **Files**: `scripts/3_deep_learning/e1_kriging_baseline.py`, `e1_kriging_comparison_table.csv`.

### E2 계절내 융해진행 D(t) — partial (피드백: 시계열 예측)
- **Action**: "timelapse에 따라 ALT 계산"을 연도간 최대ALT 외삽(예측불가)이 아니라 계절내 D(t) 예측으로 재정의. KPDC 콘슬 일별 8-16층 온도(19프로파일·2년)에서 0°C 등온선 계절곡선.
- **Result**: 계절내 D(t)는 예측 가능한 계절 구조(Stefan deepening 구간 R² 0.31~0.57, 연도간 corr0.06과 대조). common-support·profile(물리위치) leave-out·calib 제외 통제 후 7일 persistence(17.5cm) > Stefan(40.8)·GRU(46.6). EOS(최대깊이 도달일)는 단조 물리식 원리적 불가, GRU만 내부peak 예측(정확도 우위 아님). 문헌 정합(알래스카 Nature2025 랜덤분할서도 RF R²0.84→0.24 붕괴·Stefan 유지).
- **Files**: `scripts/3_deep_learning/e2_seasonal_dt.py`, `e2_seasonal_dt_summary.csv`.

### 시계열 게이트 미통과 규명(문헌 대조)
- 기존 T-lite(GRU/TCN) temporal holdout 미통과는 방법 실패 아니라 "정적 공변량+연1회 CALM으로 미래연도 외삽"이 예측신호 없는 문제(within-site corr 0.06)임을 문헌 대조로 확정. 순수 시계열DL 성공사례는 forcing 주어짐/조밀 지중온도/물리 사전학습(PI-LSTM) 세팅. 우리 경로=계절내 재정의(E2).

### 사용자 지적 9건 반영 그림·보고서 재작업
- **지적8(ALT 점 표시)**: 버그 아님. 13,606셀이 0.5°격자 83개 지역 초집중(배로 71.25°N 3,429셀·육쿤 1,703·북사면 1,374). 배로 250m 격자 확대 inset(4,381셀)+scatter 전환으로 "수천 관측 실재+공간대표성 한계" 동시 표현. `alaska_obs_vs_pred_maps.png`.
- **지적3·7(고해상)**: local_demo(PolSAR 250m 3패널)·alt_surface_northslope·magt_alaska_2m_20m·field3d_reeval·tournament dpi300+PDF 편입. deploy 2종(Diffusion 검은 아티팩트) 제외.
- **지적1·9(형식)**: LaTeX(xelatex+xeCJK) 논문형 채택. Word 병행(pandoc).
- **지적2·4·5(문체·관점·구조)**: 논문 구조(초록·서론·데이터·방법·결과·논의·결론·참고문헌), 논의를 강점·차별성(GIPL2/CCI/InterPIGNN 대비)·의의 3층위·한계극복 방안 중심으로. 과한 정직·수사·AI틱·em-dash 제거.
- **버그 수정**: xeCJK CJKspace=false(중국어·일본어 기본)가 한국어 단어 공백 압축 → CJKspace=true.

## 4. Key numbers (this session)

| Method | Domain/Case | Metric | Value | Source artifact |
|---|---|---|---|---|
| E1 OK(kriging) | 알래스카 in-domain / LORO | RMSE | 15.77 / 29.40 cm | `e1_kriging_comparison_table.csv` |
| E1 IDW | in-domain / LORO | RMSE | 15.80 / 50.92 cm | `e1_kriging_comparison_table.csv` |
| E1 GBM | in-domain / LORO | RMSE | 17.74 / 41.37 cm | `e1_kriging_comparison_table.csv` |
| E1 Stefan 앵커 | in-domain / LORO | RMSE | 14.46 / 21.26 cm | `e1_kriging_comparison_table.csv` |
| S6 sa_fusion / sa_z | LORO 게이트 | RMSE | 21.53 / 21.84 cm | `s6_source_aware_results.csv` |
| S8 mix_logit | LORO 게이트 / Alaska fold | RMSE | 29.22 / 36.48 cm | `s8_mixture_results.csv` |
| S11 conformal | 알래스카 in-domain cov90 | raw→CQR | 44.6→93.4% [88.9,96.6] | `s11_conformal_results.csv` |
| S4 앵커+잔차 | 알래스카 in-domain(최저) | RMSE | 13.33 cm | `s11_comparison_table.csv` |
| S9 timelapse | temporal holdout | RMSE / R² | 14.97 / 0.338 | `s9_timelapse_meta.json` |
| S10 shallow3d | 사이트블록 온도장 | R² / RMSE | 0.4688 / 2.66°C | `s10_shallow3d_meta.json` |
| E2 persistence / Stefan / GRU | 계절내 D(t) common-support | RMSE | 17.5 / 40.8 / 46.6 cm | `e2_seasonal_dt_summary.csv` |
| S7 콘슬 유도 ALT | 2025 비검열(n=5) | 중앙값 | 136.4 cm (CI 75.6-153.1) | `s7_kpdc_meta.json` |
| Mloc 위경도 대조 | LORO | skill | 0.147 (물리조합 능가) | `alt_ablation_cell_results.csv` |

## 5. Decisions made
- **S6·S8 negative 확정**: 구조 정교화(source-aware·mixture)는 전이서 Stefan 앵커 못 넘음. 보고서엔 비교표 행+진단으로만, 헤드라인 금지.
- **E1 co-kriging 채택**: 정통 공간보간 vs 물리+ML 비교표로 방법론 완결. 결론=보간 in-domain 경쟁·전이 붕괴, 물리 양축 최선.
- **E2 partial**: 계절내 D(t) 예측가능 구조 실증하되 2년 소표본서 persistence 못 넘음. "시간 성능은 문제정의·데이터밀도 의존" 서사.
- **얕은3D 논문형 전환**: 3D warp(희소 회랑 4.7%) 대신 깊이슬라이스·fence·트럼펫. 논문 관례.
- **제출물 형식 = LaTeX 논문(주) + 발표덱 10장 + Word(보조)**. 강점·차별성·의의·한계극복 중심.

## 6. Open questions / blockers
- **S6 식별성 한계**: b_cci가 경험 bias(-4.5~-6.0)를 과소추정(-0.4~-0.6). 인코더가 bias 흡수. b_s prior 또는 인코더-bias 직교화가 후속 과제(진단용).
- **E2 소표본**: 콘슬 2년·9지점뿐. 실측 기상 forcing(현재 지온유래 √cumTDD 대체)·조밀 지중온도 확충 시 개선 여지. persistence가 강baseline인 게 계절내 D(t)의 느린 변화 반영.
- **점검증 대표성 한계**: 관측 소수 사이트 초집중(83개 0.5°셀). 다음 지렛대=면적검증(InSAR 30m·다중프로브). 절대 RMSE 경쟁 대신 방법론 차별로 포지셔닝.

## 7. Next steps (prioritized)
1. **보고서·그림·결과 수정 계속** — owner: Claude+user, 다음 세션. 사용자 예고. 참고문헌 완전서지(저자·권·페이지·DOI), (규정이 물리쪽수 요구시) 단단 조판·그림 확대로 10쪽화, PDF 9.1MB 용량제한시 다운샘플.
2. **감사 minor 표기 정정** — owner: Claude, ~0.5h. S11 44.6%(seed평균 명시)·S2 21.26(fold-safe)/22.24(중앙값E) 구분·E1 상세표 md 이월·S7 CI 통계량 구분(유도136.4 CI[75.6,153.1] vs ABoVE 35.0 CI[31.0,39.0]).
3. **면적검증 착수(선택)** — owner: Claude, InSAR 30m ReSALT 면적평균 기준으로 검증 프로토콜 전환. 점검증 대표성 한계 극복 지렛대.
4. **KPDC 추가 활용(선택)** — 쿠가록 화재/비화재·5분 프로파일(깊이메타 확보시)·토양물성 2016 Thaw depth. 대회 KPDC 심화.

## 8. Pointers
- Authoritative: `docs/EXPERIMENT_LOG.md`(최상단 2026-07-28·07-27), `SESSION_HANDOFF.md`(현재 상태표), `docs/CONTEST_REPORT_2026.md`(분석보고서), `docs/report/main.tex`(LaTeX 논문), `docs/RESEARCH_PLAN_multifidelity_2026-07-22.md`(계획).
- 제출물: `docs/report/main.pdf`(주, gitignore), `deck/render/permafrost_summary.pdf`(gitignore), `outputs/CONTEST_REPORT_2026.docx`(gitignore). 전부 tex/md/build_summary.py에서 재생성.
- Active jobs: none. GPU 세션 종료 시 유휴.
- 결과 CSV: `data/processed/{s6_source_aware,s7_kpdc,s8_mixture,s9_timelapse,s10_shallow3d,s11_comparison_table,s11_conformal,e1_kriging_comparison_table,e2_seasonal_dt_summary}*`.
- 커밋: `4566c5b`(results) `e2d19ba`(code) `3000bca`(viz) `161fcc3`(deck) `f3806ce`(docs), pushed origin/main.
- 관련 이전 핸드오프: `gpt/handoff/20260724_1327-s0-s3-multifidelity-pipeline.md`, `20260721_1637-skeptical-reverify-transfer-ceiling.md`.

## 9. Caveats for GPT
- **canonical 명칭**: S0-S11은 멀티충실도 로드맵 단계(S0 스키마·S1 baseline·S2 Stefan·S3 증강곡선·S4 잔차·S5 pretrain·S6 source-aware·S7 KPDC·S8 mixture·S9 timelapse·S10 3D·S11 UQ). E1=co-kriging, E2=계절내 D(t)는 이번 세션 신규(사용자 피드백). LORO=매크로 지역(Alaska=ABoVE_AK+US Alaska / Lena / Canada).
- **S6·S8·E2는 negative/partial**: "구조 정교화가 전이 뚫었다"로 읽지 말 것. 전이 해법은 물리 앵커뿐(S2·E1 재확인).
- **E1 co-kriging in-domain 15.8은 GBM과 경쟁**이나 전이서 붕괴. "kriging이 낫다" 금지. OK 레나 22.3은 mean 회귀 아티팩트.
- **E2 계절내 D(t)는 콘슬 in-domain**(전이 아님). persistence가 물리·GRU보다 나은 건 계절내 느린 변화 + 2년 소표본 한계. "시계열DL이 물리 이긴다" 아님.
- **S11 conformal은 알래스카 in-domain만 유효**(전이 커버리지 보장 아님). raw 44.6%는 seed평균 기준.
- **KPDC 콘슬(S7·E2)은 알래스카 in-domain 점검증**. 전이 근거로 쓰지 말 것.
- **제출물 블라인드**: 소속·성명·프로젝트코드명(Polar_Bigdata) 금지. 발표덱 푸터는 "영구동토 ALT"로 익명화됨.
- **17cm은 물리하한 아님**: 정보병목(현재 공변량). "SOTA 12.97 돌파"·"17cm 물리하한"은 과거 폐기된 헤드라인(범위축소·apparent floor 아티팩트).
