# Handoff: Source-aware Multi-fidelity 파이프라인 S0~S3 구현 + 다음 세션 세부 plan

**Project**: Polar Bigdata — Permafrost ALT map + shallow 3D thermal (DL)
**Date**: 2026-07-24 13:27
**Session focus**: GPT 멀티충실도 로드맵을 실제 코드로 구현. S0(스키마·누설방지)~S3(증강 반응곡선) 완주, 시각화 논문형 고도화, 전체 리뷰·must_fix. 다음 세션은 S4~S11 진행.
**Author**: Claude Opus 4.8 (1M) + 01willy

---

## 1. TL;DR (≤5 bullets)
- **S0~S3 파이프라인 구현·검증 완료**(GPU 6번만 사용, 커밋 e6a0ee5 push됨). 모델 무관 공통계층(fidelity·tab_models·physics·preprocessing·geomap·gridding) + 누설방지 pytest 16개 통과.
- **S1 발견**: 입력 표준화 버그 수정 후 신경망(MLP 14.37·TabM 14.40 3-seed 앙상블 OOF)이 CatBoost 15.61 상회(대표성 하한 14cm 도달, 단 CI 0 포함=유의성 미확정). 전이(LORO Lena)는 FT-T 22.5가 GBM 40-57 압도.
- **S2 발견**: 물리 5종 중 기본 Stefan만 정확(in-domain 14.56·LORO 게이트 22.24cm). 정교화 물리(edaphic/TTOP/λ/Ku) 상방편향. **physics-as-feature(A2) 무익~악화 → 미채택**(물리는 앵커로만).
- **S3 발견(핵심)**: 물리 pseudo-label 증강은 bias 큰 전이(Lena)서 RMSE 22→17(+5cm) 개선하나, **이득의 ~85%가 물리 정보 아닌 target 앵커링**(placebo 상수도 +3.95). 물리 순가치 +0.71cm. 부정확 물리(Ku)는 두 지역 악화. Canada는 정확한 Stefan만 유효.
- **다음 세션**: S4(잔차)→S5(pretrain)→S6(source-aware A5)→S7(KPDC 검증)→S8(mixture)→S9(timelapse)→S10(3D 온도장)→S11(UQ 종합). + S3에 FT-T 재확인, 기후 시간 불일치 해결.

---

## 2. Context
- 직전 핸드오프: [20260721_1637-skeptical-reverify-transfer-ceiling.md](20260721_1637-skeptical-reverify-transfer-ceiling.md) — 전이는 모델구조로 못 뚫음, 연구목적=정확도·차별성. 다음으로 pooled·InSAR·물리fine-tune·KPDC 파싱 제시.
- 본 세션: (a) 오늘 추가된 GPT 로드맵 `gpt/ALT_multifidelity_research_roadmap_for_claude_code.md`(멀티충실도 = 자료원별 신뢰도 학습)를 우리 자산에 매핑해 `docs/RESEARCH_PLAN_multifidelity_2026-07-22.md`로 확정. (b) KPDC 신규 26파일 정리(`kpdc/README.md`, gitignore 로컬). (c) S0~S3 실제 구현.
- **연구 재프레이밍**: 증강을 "라벨 개수 늘리기"가 아니라 "자료원별 신뢰 구조 모델링(source-aware multi-fidelity)"으로 승격. 사용자 비전(3종 증강)은 유지, 병합 대신 관측모델 y_s=A_s[z]+b_s+ε_s.

## 3. What we did

### S0 — fidelity 스키마 + 누설방지 pytest + overlap gate
- **Files**: `src/polar/fidelity.py`(공변량 코어34·라벨파생7 제외·매크로 LORO·fold-safe Stefan E), `scripts/1_data_prep/build_fidelity_schema.py`, `tests/test_leakage.py`(16 통과).
- **Result**: overlap gate — direct 대비 Stefan 100%·CCI 99.5% clean(full source-aware 가능), InSAR 82%(알래스카·캐나다만·Lena 0), PolSAR 58%. → A5 clean 소스는 Stefan·CCI 2개뿐(둘 다 이미 feature) 확정.

### S1 — 실측-only 다모델 baseline
- **Files**: `src/polar/tab_models.py`(7모델 통합·lazy CUDA), `scripts/3_deep_learning/s1_baseline_tournament.py`, `src/polar/geomap.py`(지도배경). **GPU 6**.
- **Result**: in-domain(알래스카 공간블록) MLP 14.37·TabM 14.40 > CatBoost 15.61 > XGB 16.16 > LGBM 16.48 > FT-T 18.56. LORO(SHARED, SAR제외) FT-T Lena 22.5 최선 ≫ GBM 40-57.
- **버그 수정 2건**: (1) GPU 오용(CUDA_VISIBLE_DEVICES가 torch 초기화 후 설정돼 물리 0번 사용)→lazy CUDA + import 전 고정 + 6-9 assert. (2) torch 입력 미표준화→TabM 61cm 발산·MLP 저평가→fold-safe z-score로 14.4 정상화.

### S2 — 물리 5종 앙상블 + physics feature
- **Files**: `src/polar/physics.py`(Stefan기본·edaphic·TTOP·Kudryavtsev·λ, 수식·단위 실측검증), `scripts/3_deep_learning/s2_physics_baseline.py`. **GPU 6**.
- **Result**: p1 Stefan in-domain 14.56·LORO 게이트 22.24(비가중평균 Alaska·Lena·Canada). 정교화 물리 상방편향(+32~38cm bias). A2 physics-feature 무익(Δ≈0)~악화 → 미채택. 멤버 상관 0.89-1.00.

### S3 — 증강비율 반응곡선 (엄격 통제)
- **Files**: `scripts/3_deep_learning/s3_augmentation_curve.py`, `scripts/4_visualization/s3_aug_curve_figs.py`. **GPU 6**.
- **Result(catboost, mlp 전이발산 제외)**: Lena Stefan +4.66 ≈ placebo +3.95(물리순가치 +0.71). Canada Stefan +1.18만 유효(placebo -6.7·Ku -20 악화). r≥1 포화. → 증강 이득 대부분 앵커링, 부정확 물리는 해.
- **레나 전이 지도**: 증강 전 RMSE 22.0→Stefan 증강 17.0, bias +16.5→+9.1(`outputs/figures/s3_aug/lena_aug_mapping.png`).

### 전체 리뷰 + must_fix + 시각화 고도화
- 3렌즈 리뷰: critical 0, 헤드라인 무효화 누설 없음. must_fix 처리: LORO 매크로 지리누설 수정(게이트 21.86→22.24), s2 OOF E=1.0 버그(p1 중앙 30.5→47.5), fold-safe 가드 배선, sigma_prior 금지, 서술정정.
- 시각화 논문형 고도화(4렌즈 조사: Ran2022·Whitcomb2024·Obu2019·CCI): `geomap`(hexbin_map·field_map·support_mask·inset_locator·scalebar·mask_ocean), `gridding`(make_grid·interp_obs). S1·레나 재렌더. **바다 삐짐 아티팩트 mask_ocean으로 수정**(관측 좌표는 육지, hexbin 셀 삐짐이었음).

## 4. Key numbers (this session)

| Method | Domain/Case | Metric | Value | Source artifact |
|---|---|---|---|---|
| MLP / TabM | 알래스카 in-domain 공간블록 | RMSE(3seed 앙상블 OOF) | 14.37 / 14.40 | `s1_baseline_results.csv`·`s1_baseline_oof.csv` |
| CatBoost | 알래스카 in-domain | RMSE | 15.61 | `s1_baseline_results.csv` |
| FT-Transformer | LORO Lena 전이 | RMSE | 22.5 | `s1_baseline_results.csv` |
| Stefan(p1) | in-domain / LORO 게이트 | RMSE | 14.56 / 22.24 | `s2_physics_results.csv` |
| physics-as-feature | in-domain / LORO | ΔRMSE | ≈0 / +0.5~1.8(악화) | `s2_physics_results.csv` |
| Stefan pseudo(증강) | Lena 전이 | ΔRMSE 개선 | +4.66 (placebo +3.95) | `s3_aug_curve_results.csv` |
| Stefan pseudo | Lena 지도 재계산 | RMSE 전→후 | 22.0→17.0 | `lena_aug_mapping.png` |
| overlap | direct×Stefan / ×CCI | clean % | 100 / 99.5 | `source_overlap_matrix.csv` |

## 5. Decisions made
- **physics-as-feature(A2) 미채택**: in-domain 무익·LORO 악화. 물리는 앵커(직접 예측)로만. (GPT 로드맵 위험2 실증)
- **S3 증강 해석**: 물리 pseudo 이득 대부분은 target 앵커링(bias 교정), 물리 정보 순가치 작음. 부정확 물리(Ku)는 해. → "증강이 언제 돕고 언제 해치나"의 조건부 규명이 대회 비교분석 핵심.
- **LORO는 매크로 지역 단위**(Alaska=ABoVE_AK+US Alaska): 동일 지점 세부라벨 누설 방지.
- **시각화 인프라 = geomap/gridding 논문형**(hexbin·격자필드·mask_ocean·inset). S0/S2 재렌더는 보고서 단계로 유예.

## 6. Open questions / blockers
- **S3에 FT-Transformer 미포함(설계 미흡)**: S1에서 FT-T가 전이 최선이었는데 S3는 catboost+mlp만. mlp는 전이 발산. → 다음 세션 FT-T로 증강 재확인 필요.
- **기후 시간 불일치**: alt_cm=1974-2024 다년평균, e5_=2015-2020 평년값. 둘 다 정적이라 "정적 climatology 매핑"으로 일관하나 기간 불일치. 시간정합은 S9(연도별 기후+연도별 ALT)에서 해결.
- **S6 source-aware 증분 미실증**: clean 소스 Stefan·CCI 2개뿐이고 둘 다 이미 feature. 구조적 증분이 있는지는 실증 대상(최대 리스크).
- **TabPFN(라이선스 TABPFN_TOKEN 필요)·RealMLP(torchvision 충돌)** 보류 중. 토큰 설정 시 자동 합류.

## 7. Next steps (prioritized) — 다음 세션 세부 plan

**공통 규약(모든 단계)**: GPU 6-9만(assert 배선됨), fold-safe(누설 pytest 통과 전제), ≥3 seed·블록부트스트랩 CI, 결과 CSV+meta 저장, 결과물은 geomap 논문형 시각화(mask_ocean 포함).

1. **S11 (보정 UQ) — 헤드라인, owner: Claude, ~1.5일**. GBM+분위+conformal(기존 56→86% 재활용)로 cov90 85-90% 지도. 생성모델(Diffusion/Flow) 과신 대조. 예측+구간폭+커버리지 곡선. *사용자 지시로 이번엔 S11 제외하고 나머지부터. 아래 2~9 먼저.*
2. **S4 (잔차학습 fallback) — owner: Claude, ~0.5일(S)**. Stefan 앵커+저용량 잔차(e5_+지형, shift-robust), shrinkage λ 스윕. LORO ≤22.24 게이트, 미달 즉시 negative 확정(P2 재확인).
3. **S5 (물리 pretrain→finetune) — owner: Claude, ~1일(M)**. dense Stefan pseudo 사전학습→실측 finetune. `train_pretrain_finetune.py` 확장. LORO에서 S1 baseline 대비.
4. **S3 보강 (FT-T 증강 재확인) — owner: Claude, ~0.5일**. S3 증강에 FT-Transformer 추가(전이 최선 모델). mlp 발산 대체. 물리 순가치 재확정.
5. **S6 (source-aware multi-fidelity A5) — owner: Claude, ~2-3일(L, 최대 리스크)**. 공유 인코더+source별 (b_s, logσ_s) 헤드, Gaussian NLL. Stefan·CCI 2소스 full A5. **Stefan-feature+CCI-feature baseline 대비 증분 실증 필수**(가정 금지). 미달 시 UQ 산출물로만. 신규 `source_aware_mf.py`.
6. **S7 (support/censored + KPDC 검증) — owner: Claude, ~1일(M)**. KPDC 콘슬 8층 지중온도→0°C 등온선 ALT 유도, 얕은3D 검증. 코어 thaw depth(72-88cm)=ALT 라벨. 탐침한계 구간검열 우도. **대회 KPDC 주 활용 충족**(보고서 첫머리 배치+정량표).
7. **S9 (timelapse ALT) — owner: Claude, ~1일**. 연도별 기후(era5land_temporal_covariates.py, 2010-2024)+연도별 ALT로 연별 지도 GIF(고정 색상한계). **기후 시간 불일치 해결**. 연 anomaly 예측불가(corr 0.06)는 정직 각주.
8. **S10 (얕은 3D 온도장) — owner: Claude, ~1-2일(L)**. shallow3d_alaska·field3d로 0-3m T(z) + 0°C 등온면. PyVista 3D. R² 0.47 재현.
9. **S8 (mixture-of-physics) — owner: Claude, ~1일**. 물리 전문가 게이팅. 진단용(어느 물리가 어느 환경서 선택).
10. **S2/S0 시각화 재렌더 + mask_ocean 적용** — 보고서 작성 시. s2_physics_figs·s0_schema_figs에 mask_ocean 추가.

## 8. Pointers
- Authoritative: `docs/RESEARCH_PLAN_multifidelity_2026-07-22.md`(§11 보강 = 지역전략·DL·물리식·시각화·기록), `docs/EXPERIMENT_LOG.md`(S0-S3 상세), `docs/EXPERIMENT_PLAN_2026-07-21.md`(+쉬운설명).
- GPT 로드맵 원본: `gpt/ALT_multifidelity_research_roadmap_for_claude_code.md`.
- 결과 CSV: `data/processed/{s1_baseline,s2_physics,s3_aug_curve}_{results,meta}`, `source_overlap_matrix.csv`, `covariate_availability_by_region.csv`, `fidelity_schema_meta.json`.
- 시각화: `outputs/figures/{s0_schema,s1_baseline,s2_physics,s3_aug}/`(PNG 추적, PDF gitignore).
- 대용량(gitignore, 재생성): `data/processed/fidelity_base.csv`(build_fidelity_schema.py로 재생성), KPDC `kpdc/`(로컬만).
- Active jobs: none. GPU 6-9 유휴.
- 커밋: `07dfdf7`(code) `9d65be8`(docs) `2d600d7`(results) `d533213`(viz) `e6a0ee5`(fix), pushed origin/main.

## 9. Caveats for GPT
- **GPU는 6,7,8,9만**. 과거 CUDA_VISIBLE_DEVICES 미스로 물리 0번(타 사용자) 오용 사고 있었음. 스크립트에 assert+lazy CUDA 배선됨.
- **"신경망 14.4 최선"은 3-seed 앙상블 OOF RMSE**. seed-mean은 MLP 14.66·TabM 15.03. CI 0 포함이라 CatBoost 대비 "일관 우세이나 유의성 미확정". 단정 금지.
- **"Stefan LORO 최선"은 FT-T(22.5)와 동급**(seed 노이즈 범위). "물리 앵커가 전이 하한 확보·GBM 압도"로 표현.
- **physics-as-feature는 미채택**(무익). 물리는 앵커로만.
- **S3 증강 이득 ≠ 물리 정보**: 대부분 target 앵커링. placebo(상수)도 유사 개선. 물리 순가치는 작음(+0.71cm Lena).
- **canonical**: 셀 데이터 = `fidelity_base.csv`(공변량 코어34, gitignore). 물리 = Stefan `E√TDD` fold-safe E. LORO = 매크로 지역(Alaska/Lena/Canada). alt_cm = 정적 다년평균(timelapse 아님).
- **mlp는 전이(covariate shift)서 발산** 경향(S3). 전이엔 FT-T·catboost 사용.
