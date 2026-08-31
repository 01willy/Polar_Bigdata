# Polar_Bigdata 실험 로그 (chronological)

> 세션별 작업 기록. 큐레이션된 마스터 인덱스는 [EXPERIMENTS.md](EXPERIMENTS.md),
> GPT 공유 핸드오프는 [gpt/handoff/](../gpt/handoff/) 참조.

## 2026-07-29 17:35 — S13 통합실험 폐기 + 보고서 수치·그림·본문 정합 전면 감사

실험 1건(폐기) + 보고서 정합 작업. 산출 수치의 신규 생성은 없고, 기존 S1~S12 결과에 대한
검증과 서술 정정이 중심이다.

### S13 통합 세 상황 실험 — 설계 결함으로 폐기
- `scripts/3_deep_learning/s13_unified_settings.py` 신규 작성·실행(4 shard, 5,625행, 유효 4,988).
  목적은 지역 내·공변량만·정보 없음 세 조건을 같은 평가 셀에 정렬해 조건별 최고 방법을
  한 표로 제시하는 것이었다.
- 독립 검증(서브에이전트 2종: 설계 감사 + 주장 재계산)에서 결함 확인.
  1. TASKS에 레나·캐나다의 indomain이 빠져 조건 효과와 지역 난이도가 교락된다.
  2. indomain 학습집합이 transfer의 상위집합이 아니라 표본 크기와 교락된다.
  3. 유사라벨(Stefan·CCI)이 입력 피처 `e5_sqrt_tdd`·`cci_alt`의 결정론적 함수여서
     대상 지역 정보를 주입하지 못한다. anchor==pseudo인 27개 구성은 잔차가 항등 0이다.
  4. 검증 분할 없이 평가블록 B에서 직접 선택했고, 구성 수가 pseudo 144 대 나머지 15로
     9.6배 달라 최소값 비교가 성립하지 않는다.
  5. anchor='none' 분기에 y 유한 마스크가 없어 레나 18개 구성이 소실된다.
  6. `n_ps`를 대상 블록이 아닌 학습원 크기에 비례시켜 r이 지역마다 다른 복제율을 뜻한다.
  7. 결과행의 n이 실제 평가 셀 수가 아니고, block_ci가 NaN을 거르지 않는다.
- 주장 재계산에서도 부호 뒤집힘 1건, 통계적 뒷받침 부족 2건 확인. 유효 독립 블록이
  레나 2.4개·캐나다 3.6개에 불과해 1 cm 내외 차이는 해상도 밖이다.
- **판단: 폐기.** S13이 확인하려던 세 목적 중 두 개(알래스카·완전전이의 증강 사용 여부,
  Stefan 앵커의 실험적 논거)는 기존 실험에 이미 답이 있고(`aug_within_alaska_results.csv`,
  `s6_source_aware_results.csv` fixaug_catboost 21.28, 표 tab:physics·S12 앵커 4종),
  나머지 하나(세 조건 한 축 정렬)는 평가 셀이 달라 원리적으로 불가능하다. 문서화로 대체.
- 보고서에는 S13이 한 줄도 반영된 적이 없어 본문 영향 없음.

### 보고서 수치 감사 — 전 항목 원자료 대조 통과
- 표 5종(tab:models, tab:physics, tab:aug, tab:transfer, tab:s12) 전 항목을
  `data/processed/*` 로 재계산해 대조, 전부 일치(허용 0.02 cm).
- 본문 전용 수치도 대조 통과: 편향 분해(레나 +5.51→-1.52, 캐나다 +2.31→+6.59,
  산포 27.39→24.92), 물리 순가치(레나 1.65, 캐나다 10.20), 전이 증강 21.28.
- 집계 함정 1건 자체 발견: 구성별로 캐나다 mlp가 대량 발산해 '두 지역 평균'이
  레나 단독값(14.33)이 되는 경우가 있다. 지역 수 확인 후 집계하면 표의 21.32가 맞다.

### 본문 서술 정정 4건 (결론 불변)
1. TTOP: `physics.py:142` 가 `p3 = np.where(ttop<0, p2, np.nan)` 이므로 독립 융해깊이 식이
   아니라 edaphic에 영구동토 판정을 씌운 것이다. 레나·캐나다는 전 셀이 판정을 통과해
   edaphic과 값이 같다(그림에서 두 막대가 알래스카에서만 갈라지는 이유). 본문·표·그림
   라벨을 `edaphic + TTOP 판정`으로 통일.
2. 잔차 가중 λ: 지역 내는 단조 감소가 아니라 λ≈0.5~0.75에서 최소 후 상승한다.
   전이는 λ=0부터 단조 증가. 두 목표의 최적 가중이 어긋난다는 결론은 오히려 강화.
3. `fig:s12`(d)가 한 계열만 그려 표 tab:s12와 어긋났다. λ별 최선 포락선으로 교체하니
   두 조건 모두 λ=0.25가 최소(21.32 / 22.92)로 표와 정확히 일치.
4. 본문의 '세 구조'(사전학습·다중충실도·혼합)와 그림의 '구조 정교화 3종'(혼합·
   source-aware·잔차학습)이 다른 집합이었다. 본문에 수치와 함께 명시.

### 그림 재생성
- `s12_figs.py`: 패널(d) 포락선 교체 + 최소점 표식, 지역명 레나델타로 통일, 죽은 루프 제거.
- `s1_baseline_figs.py`: 물리식 단독(14.56)·평균 예측(17.38) 세로 기준선 추가 → 표 tab:models와 일치.
- `s3_aug_curve_figs.py`: 패널 제목 한글화(캐나다/레나델타), 내부 (a)(b) 중복 라벨 제거,
  범례 지역명 한글화. '전이' 표기 제거(이 실험은 공변량만 조건이므로).
- `s2_physics_figs.py`: TTOP 라벨 변경.
- `report_alt_map_hires.py`: 계수 E를 전 지역 적합 1.5710에서 알래스카 실측 최소제곱
  1.6167로, 영구동토 마스크를 TTOP<0 에서 연평균기온<0 으로 변경. 다른 두 지도
  (`report_alt_annual_maps.py`, `report_alt_map_best.py`)와 통일. 영구동토 육지 84.8%.
  기후값 기간만 다르다(2015-2020 대 2010-2024)는 점은 캡션에 명시.

### 보고서 구조
- 표 tab:settings(검증 조건 정의) 신설. 결과 각 절 첫머리에 `조건:` 표기.
- 그림 12개 캡션 전부에 모델·입력 집합·검증축·조건 명시.
- 컴파일: 13쪽, 그림 12, 표 8, 오류 0, 미해결 참조 0. 번호 등장 순서 정상.

### 4축 정합 감사 결과 (세션 종료 직전, workflow wf_60c63c38-968)

감사 4축(수치·프로토콜·근거·문서모순) × 적대적 검증. 지적 57건 중 검증 통과 10건 + 자체
재확인 6건. **결론이 뒤집힌 항목은 없으나 제출 전 수정이 필요한 항목이 있다.**

#### A. 선택적 부분집합 인용 (제출 전 필수)
1. `main.tex:562` 유도 융해깊이 `23.8 cm(n=12)`는 Fairbanks 6개 시추공 제외 부분집합.
   전 표본(n=18)은 40.66 cm. `shallow3d_alaska_results.csv` cv=alt_match 두 행 모두 존재.
   산출 코드 `shallow3d_alaska.py:311`은 n=12 행을 "41 cm를 단독 헤드라인으로 쓰지 않도록
   함께 기록"한 참고행으로 명시하는데 보고서가 그 지시를 반대로 적용했다.
   → 주값을 `40.7 cm(n=18, R^2 -0.29)`로 교체하고 23.8 cm(n=12)는 Fairbanks 제외 민감도로
   병기. **주의**: 메타의 "3 m 관측격자 상한 절단 아티팩트" 서술은 원자료로 뒷받침되지
   않는다(Fairbanks 6행 모두 `alt_obs_near_maxdepth=0`). 이 설명을 보고서로 옮기지 말 것.
2. `main.tex:602` 및 초록 KPDC 상대편향 `약 0.1%`는 Council 2019 제외 n=3 값.
   전체 n=4는 +1.5%(RMSE 1.04, 평균 대비 2.8%). 원자료 메타가 제외를 정당화하나
   보고서는 제외 사실을 밝히지 않는다.
3. `main.tex:604` 심부 유도 `136.4 cm`는 2025 시즌·우측검열 제외·peak_days>=40 인 n=5
   조건부 중앙값(CI 75.6-153.1). 메타가 "콘슬 전체 대표 ALT로 일반화하지 않는다"고
   명시했는데 보고서는 조건 없이 인용.

#### B. 평가 셀 집합 불일치 (수치 수정 필요, 결론 유지)
4. `tab:s12` 행 간 평가 셀이 다르다. `eval_metrics._clean()`이 예측 NaN 쌍을 버리므로
   cci_alt 결측 셀이 CCI 포함 행에서만 자동 탈락한다(레나 3037→2958, 캐나다 742→741).
   같은 셀로 맞추면 개선폭이 정보없음 1.33→0.85, 공변량만 2.05→1.23 cm로 줄어
   약 1.5~1.7배 과대 표기 상태다. 캐나다는 결측이 1셀인데 RMSE가 0.92 cm 움직여,
   보고된 산포 감소 2.48 cm 중 0.95 cm가 이상치 1셀 제외 효과다.
   **결론은 유지**: 짝을 맞춰도 네 조합 모두 오차 감소, |편향| 감소 셋 성립.

#### B'. 수정용 대체값 (종합 에이전트 산출, 재계산 불필요)
- `tab:s12` 물리식 단독 행: `23.72 → 22.91`, `24.60 → 24.11`
  (지역별 공변량만 레나 16.158·캐나다 29.665, 정보없음 레나 21.650·캐나다 26.572).
  이 값은 결과 CSV의 `anchor=stefan_cci_w`(W_SC=1.0, 사실상 CCI 유효셀 한정 Stefan) 행과
  정확히 일치하므로 대체만 하면 된다. `기계학습 단독` 행도 같은 비대칭을 가진다.
- 캐나다 결측 1셀은 lat 52.80·lon -118.117·관측 ALT 250.2 cm·Stefan 예측 56.2 cm의
  극단 이상점이다. 이 한 셀 때문에 캐나다 RMSE가 공변량만 31.29→29.67,
  정보없음 27.49→26.57로 움직인다.
- `main.tex:502,506` 편향·산포: `5.5 → 5.8`, `2.3 → 2.6`, 산포 `27.4 → 26.4`.
- `main.tex:508,512` 개선폭: 공변량만 `2.05 → 1.23`, 정보없음 `1.33 → 0.85`.
- `main.tex:512` 0.62 cm는 짝지은 부트스트랩 평균이고 점추정 델타는 0.85 cm다.
  두 값이 다른 이유(부트스트랩 평균 대 점추정)를 본문에 구분해 적어야 한다.
- `main.tex:602` KPDC 상대편향: `약 0.1% → n=4 기준 약 1.5%`(Council 2019 제외 시 0.13%).
  "현장을 대표함을 확인" 단정을 "소표본 범위에서 큰 계통 편차가 관측되지 않았다"로 완화.
- `main.tex:471` 전이 최적에 "입력 14종" 명시 + 같은 shared25 기준값 21.96 cm 병기.

#### C. 프로토콜 서술 오류
5. `main.tex:198,322` "0.5도 공간블록을 집계 단위이자 교차검증 그룹으로 삼았고" — 0.5도는
   교차검증 그룹일 뿐 집계 단위가 아니다. 집계 단위는 loc_id(고유 좌표)다.
   재확인: 알래스카 고유좌표 13,606 = 셀 수, 블록은 74개. 같은 보고서 136행과도 모순.
6. `tab:transfer` 지역 내 열(FULL 34 + 3-seed 앙상블 OOF)과 전이 열(SHARED_CORE 25 +
   seed별 RMSE 평균)이 공변량 집합과 집계 규약이 동시에 다른데 표에 표기가 없다.
7. `main.tex:471` "잔차를 저용량 부스팅으로 바꾸고 λ=0.25로 낮춘 설정" — 공변량 집합도
   shared25에서 shift14로 바뀐다. 같은 shared25의 catboost_lo·λ=0.25는 21.956이다.
8. 방법 절이 최소제곱 E만 정의한다. 중앙값비 추정량(`physics.fit_E`)은 미정의인 채로
   tab:physics·tab:models 수치의 근거로 쓰인다.
9. `fig:s10` 캡션 "시추공 단위 홀드아웃" — 실제는 사이트 14개를 6 fold로 묶은
   사이트-블록 6-fold다. 명칭이 검증 강도를 낮게 표현한다.
10. 공간블록 수 "지역당 11-30개" — 재확인 결과 레나 20, 캐나다 31이다. 범위는 11-31.
11. `tab:aug`·`fig:augcurve` 절에 공변량 집합 표기가 없다(실제 SHARED_CORE 25).
12. `fig:hires` 캡션이 SoilGrids 사용을 언급하나, 그려진 장은 p1_stefan = E·√TDD로
    토양 물성이 들어가지 않는다. SoilGrids는 다른 물리 멤버와 TTOP 계산에만 쓰인다.
13. `main.tex:471`이 35.45 cm를 `fig:transfer`(a)의 근거로 인용하나, 그 그림은 shift14만
    그리므로 35.45(shared25)는 그림에 없다. shift14의 ridge·λ0.75는 37.29다.
    그림 인용을 빼거나 그림에 shared25 계열을 추가해야 한다.
14. `main.tex:602`가 인용한 `fig:uq`(b)는 ALT 유도 비교 그림이라 √TDD 검증을 보이지 않는다.
    `kpdc_figs.py:82-93`의 산점도를 실제로 산출해 인용하거나 그림 참조를 삭제해야 한다.
15. `main.tex:322` "지점 단위로 채점할 때와 견주면"도 코드와 어긋난다. `restructure_gate.py`는
    세 조건 모두 loc_id 셀평균을 채점하며, 11%·7~8% 이득은 0.5도 집계가 아니라
    1/n_loc 학습 가중과 셀평균 직접 학습에서 나온다.

#### D. 문서·스펙 미갱신
13. `figures/figure_spec.json`이 폐기된 E_global=1.5710·TTOP<0 마스크를 여전히 규정.
    그림 재생성 시 되돌아갈 위험. TTOP 옛 라벨, 증강 실험의 '전이' 오칭도 잔존.
    이번 세션 신규 그림 3종(transfer_loro_summary·s12_hybrid_summary·alt_annual_fields)
    항목 자체가 없다.
14. `SESSION_HANDOFF.md`가 이전 세션 상태(docs/report, 6쪽)에 멈춤. 진입점 문서인데 낡음.
15. `.claude/project.yaml`의 `authoritative_reports_dir: docs/` — 실제는 outputs/report/.
16. `docs/CONTEST_REPORT_2026.md`(구 md 보고서)가 LaTeX 본과 불일치. 조건 다른 수치를
    LORO 열에 배치하는 등. 제출 묶음에서 제외하거나 갱신 필요.
17. `deck/build_summary.py`가 증강 실험을 '전이'로 호칭, 물리 순가치 10.4(실제 10.2).
18. `figure_plan.md` 계획 21항목 중 8개 미게재, 보고서 게재 6개는 계획에 없음.

#### E. 반증되었거나 과장된 지적 (수정 불필요)
- 콘슬 153.1 cm를 신뢰상한으로 본 지적 → 실제 유도 관측값이다(ID21_B 2025). 감사자 오류.
- 알래스카 내부 증강 서술이 원자료와 정반대라는 지적 → **보고서가 맞다.**
  `aug_within_alaska_results.csv`의 aug_helps_sig=True는 누설 미통제 결과이며 이미 기각된
  헤드라인이다. 누설통제 결과(donor 제거 14.22→15.99, real_only 14.96보다 악화)가
  `docs/RESULTS_SUMMARY_2026-07-20.md:11`에 있다. 감사자가 data/processed만 봐서 반대 결론.
  다만 근거가 CSV가 아니라 md에 있어 추적성이 낮다(개선 여지).
- 결합 개선 0.62 cm·CI [-2.22,+2.76]가 미근거라는 지적 → 재현 가능함이 확인됐다.
- Stefan을 2모수 아핀식으로 본 지적 → 절편 적합 구현이 5개 스크립트에 존재한다.
- CONTEST_REPORT md의 캐나다 726셀 내부기준 불일치 지적 → 자료원 기준으로 일관.

#### G. 수정 대상 파일 (종합 에이전트 정리)
- `outputs/report/main.tex` — 198, 322, 471, 482, 502, 506, 508, 512, 517, 523-527, 562, 602행
- `figures/figure_spec.json` — 662, 665, 671행
- `scripts/3_deep_learning/s12_hybrid_transfer.py` — 210-216행(유효 마스크·부트스트랩 NaN)
- `scripts/4_visualization/report_alt_map_hires.py` — 62행 사문 변수 `S2_META` 제거
- `scripts/4_visualization/kpdc_figs.py` — 82-93행 산점도 산출
- 재생성 그림: `fig:s12`, `fig:transfer`(a), `fig:transfersum`, s7_kpdc √TDD 산점도

#### F. 블라인드 위험
19. `main.tex:31,35`에 사용자 계정명이 박힌 폰트 절대경로가 남아 있다. 컴파일 PDF에는
    노출되지 않으나 LaTeX 소스를 함께 제출하면 블라인드가 깨진다.

### 미결
- 분량 13쪽. 대회 안내 '10장 내외' 대비 3쪽 초과. 축소 후보는 그림 5(해상도 결정 요인),
  그림 11(3차원 지중온도장·계절 융해), 4.4절(지중온도 유도 라벨 검증). 사용자 판단 대기.
- 다음 세션에서 최종 수정 예정.

## 2026-07-28 — 예선 제출물 조립 + 사용자 지적 9건 반영 보고서 전면 재작업

실험 아님(문서·그림·제출물). 상세 스냅샷 `SESSION_HANDOFF.md` 최상단.
- **발표덱 10장**: `deck/build_summary.py` 재구성(S6~S11·E1·E2 반영) → `deck/render/permafrost_summary.{pptx,pdf}`. 페이지버그·제목클리핑·블라인드 푸터 익명화·s7 dpi300 수정. visual QA 통과.
- **분석보고서(md→검증)**: `docs/CONTEST_REPORT_2026.md`(10섹션). 핵심 수치 전 항목 CSV 대조 정합, 블라인드·em-dash 0. §2.1 관측분산 실측정정(sd 18.5→20, 범위 1~298). S2 physics-as-feature 로그 부호오류 CSV대조 정정.
- **그림 전면 재작업(사용자 지적)**: (8) ALT 관측 밀도문제=데이터구조 규명(13,606셀이 83개 0.5°지역 초집중)+배로 250m 확대 inset(4,381셀), scatter 전환. (3·7) 고해상 자산 편입(local_demo PolSAR·northslope·magt·field3d·tournament, dpi300+PDF), deploy 2종 제외. (6) 얕은3D 논문형 재작성=`s10_depth_slices`(깊이슬라이스 4패널)·`s10_fence_sections`·`s10_profiles_trumpet`, 3D warp 강등. em-dash·주황색·저해상 다수 교정.
- **LaTeX 논문(주 제출물)**: `docs/report/main.{tex,pdf}`(6쪽 2단, xelatex+xeCJK). 논문구조(초록·서론·데이터·방법·결과·논의·결론·참고문헌), 강점·차별성·의의·한계극복 중심 서술. **xeCJK CJKspace=false 한글공백 버그 수정**(CJKspace=true). Word 병행 `outputs/CONTEST_REPORT_2026.docx`(pandoc).
- **다음 세션**: 결과·보고서·그림 수정 계속. 참고문헌 완전서지, 물리쪽수 조정(규정시), PDF 용량.

## 2026-07-27 — E1 co-kriging baseline + E2 계절내 D(t) (사용자 피드백 반영) + 적대검증·정정

GPU **7**(세션 중 3,4,5→6,9→7 재배정, E2는 6,9→7 이전 재실행). 사용자 피드백 2건 반영: 인터폴레이션(co-kriging) 비교, timelapse에 따른 ALT 계산(시계열). 누설 pytest 16개 통과, git 미커밋.

- **E1 co-kriging/RK vs GBM·Stefan (채택)**: 정통 공간보간을 정식 baseline으로 편입. in-domain 알래스카 공간블록 OK 15.77·IDW 15.80이 공변량 GBM 17.73과 경쟁(사실상 동률~소폭 우세), RK 17.23(seed평균; 앙상블 15.25는 seed간 bias 상쇄 아티팩트라 헤드라인 제외). **물리 Stefan 앵커만 양축 최선(in-domain 14.46·LORO 21.26)**. LORO 전이서 보간 붕괴: OK 29.40·IDW 50.92(레나 84.8 폭주)·RK 36.30·GBM 41.37. **진단**: kriging 전이 붕괴 원인=variogram(range≈899km·sill 442cm²)이 지역마다 다름=covariate shift의 공간통계판. OK 레나 22.3은 실력 아닌 mean 회귀 아티팩트(range 밖→train 전역평균 수렴). 정정: STEF in-domain √TDD 셀 오정렬 버그(bias −2.5 아티팩트→−0.99·RMSE 14.39→14.46), RK 앙상블 아티팩트 서술. 스크립트 `e1_kriging_baseline.py`, 그림 `outputs/figures/e1_kriging/`.
- **E2 계절내 융해진행 D(t) (partial, KPDC 심화)**: "timelapse에 따라 ALT 계산"을 연도간 최대ALT 외삽(S9 corr0.06 예측불가)이 아니라 **계절내 D(t) 예측으로 재정의**. KPDC 콘슬 일별 8-16층 온도(19프로파일·2년·2023-2025)에서 표층연결 0°C 등온선 계절곡선 유도. 물리(Stefan √cumTDD)·시계열 GRU·persistence·static 비교, profile(물리위치) leave-out + 계절 전반→후반 시간분할. **계절내 D(t)는 예측 가능한 계절 구조 보유**(Stefan deepening 구간 R² 0.31~0.57, 연도간과 대조). 단 common-support·물리위치 fold·calib 제외 통제 후 7일 persistence(17.5cm)가 강baseline이라 Stefan(40.8)·GRU(46.6) 미달(2년 소표본·지온유래 forcing 한계). EOS(최대깊이 도달일)는 단조 물리식 원리적 불가(argmax=창끝 고정), **GRU만 내부 peak 예측 가능**(정확도 우위 아님, 예측가능성 자체가 차별점). **결론: 시간 성능은 방법 아닌 문제정의·데이터밀도 의존**. 문헌 정합(알래스카 Nature2025 랜덤분할서도 RF R²0.84→0.24 붕괴·Stefan 유지, 순수 시계열DL 성공사례는 forcing 주어짐/조밀지온/물리 사전학습 세팅). 스크립트 `e2_seasonal_dt.py`, 그림 `outputs/figures/e2_seasonal_dt/`.
- **시계열 게이트 미통과 규명**: 기존 T-lite(GRU/TCN) temporal holdout 미통과는 방법 실패 아니라 "정적 공변량+연1회 CALM으로 미래연도 외삽"이 예측신호 없는 문제(within-site corr 0.06)임을 문헌 대조로 확정. 우리도 가능한 경로=forcing 명시입력·물리 사전학습·조밀 지중온도(E2가 계절내로 재정의한 시도).
- **다음**: 시각화 통일(8서사축 대표그림)·예선 보고서 조립(데이터출처·기법 명시, 블라인드, PPT 10장 내외).

## 2026-07-27 — S6~S11 완주(잔여 6단계) + 적대검증 3렌즈 + 표적 수정

GPU **3,4,5**(사용자 2026-07-27 지시). 잔여 단계 S6·S7·S8·S9·S10·S11을 병렬 구현·실행(S6=GPU3·S8/S10=GPU4/5·S7/S9/S11=CPU), 이어 누설·수치정합·시각화 QA 3렌즈 적대검증, 지적사항 표적 수정·재렌더. 누설 pytest 16개 통과 유지. git 미커밋(cleanup 세션 담당).

- **S7 KPDC 콘슬 검증 (채택, 대회 KPDC 주활용 충족)**: Council(수어드반도) 일별 지중온도 프로파일 19종(배치 A 8·B 11, 깊이 역순 QC교정·재설치 불연속 site 분리)에서 0°C 등온선 ALT 유도. 2025 비검열 부분집합 조건부 중앙값 136.4cm(n=5, CI 75.6-153.1), 38시즌 중 23건 우측검열(구간검열 정량표 확보). Stefan 59.6·S1모델 53-55cm는 사이트 관측 스펙트럼(ABoVE 35↔코어 81↔유도 121-153cm) 내부이나 16층 유도 대비 과소 → **점검증 대표성 잡음 지배 재확인**(알래스카 in-domain, 전이 근거 아님). `s7_kpdc_{results,meta}.csv/json`, `s7_council_{daily_temp,alt_derived}.csv`, 그림 `outputs/figures/s7_kpdc/`.
- **S9 timelapse 마감 (채택)**: 재학습 없이 2010-2024 연별 ALT 프레임 15장을 mask_ocean·고정색축(40-75cm)·oslo_r로 재렌더 + adaptive palette GIF(`timelapse_alt_alaska_v2.gif`). 연 anomaly broc 프레임·GIF 추가(`timelapse_alt_anomaly_v1.gif`, 절대변화 0.7%로 비가시→편차 가시화). 홀드아웃 RMSE 14.97cm[14.72,15.22]·R²0.338, 연 anomaly corr +0.059(예측불가, 각주 명시). 시간정합=연도별 ERA5×연도별 ALT(정적 다년평균과 구분). `s9_timelapse_meta.json`, `outputs/figures/s9_timelapse/`.
- **S10 얕은 3D 재현 (채택)**: `shallow3d_alaska.py` 재실행이 기존 산출과 바이트단위 동일(RMSE 2.66°C·R²0.4688·0°C→ALT r0.282, 문서치 일치). 깊이별 지도·남북 단면(0°C masked contour, 갭 진회색)·0°C 등온면 3D(PyVista) 재렌더. 온도장만 검증 성립, 0°C 유도 ALT는 r0.28이나 R²<0(절대정합 미성립). `s10_shallow3d_meta.json`, `outputs/figures/s10_shallow3d/`, `outputs/volumes_3d/s10_*`.
- **S6 source-aware multi-fidelity A5 (negative)**: 공유 인코더+소스별(b_s,logσ_s)+Gaussian NLL 구조가 LORO 게이트 sa_fusion 21.53·sa_z 21.84cm로 baseline 최고(naive pooling 21.11·Stefan앵커 21.26·고정증강 21.28)를 못 넘음. σ헤드 cov90 0.996은 폭발산(473cm) 무정보 커버리지 → 정확도·UQ 게이트 모두 미달 **negative 확정**. 부산물: in-domain서 sa_z 14.35로 pooling의 in-domain 파괴 회복(Δ+1.80 유의), b_stefan≈0은 실제 bias 정합·b_cci는 과소추정(식별성 한계). **보고서엔 비교표 1행+소스신뢰도 진단으로만 편입**(헤드라인 금지, 계획 문서 권고). `s6_source_aware_{results,meta}`, `outputs/figures/s6_source_aware/`.
- **S8 mixture-of-physics (negative, 진단가치)**: 물리5종+공변량 게이트 mixture가 단일 Stefan 못 이김(LORO 29.22 vs 22.24, in-domain 동률). S2 예측대로. 진단: 게이트 선택은 환경의존적 해석 가능(한랭·저SOC=Stefan, 온난=λ변형)이나 정확한 전문가가 Stefan뿐이라 전환이 오히려 편향전문가로 이동. Alaska LORO fold 파탄(비-Alaska 학습 게이트가 Alaska 61%를 상방편향 p5_λ 배정→36.48cm)=**전이서 전문가 선택 불가능성 정량 증거**. oracle 하한 15.48cm(모델 아님). `s8_mixture_{results,meta}`, `outputs/figures/s8_mixture/`.
- **S11 종합 비교표 + conformal UQ (채택, 헤드라인)**: quantile CatBoost raw cov90 44.6%(과신)→train 블록내부 CQR 93.4%[88.9,96.6] 보정(폭 14.7→53.6cm). 16행 증강방식 비교표(전 수치 CSV 재계산): LORO 게이트서 어떤 증강·구조도 Stefan 앵커(최소제곱 E 21.26) 유의 초과 못함, in-domain 최저는 S4 앵커+잔차 13.33cm. 다축 검증표(공간블록·LORO·temporal·소스leave-out·누설pytest16통과). **transductive 라벨링**: pool_mlp·고정증강·S5 pretrain 3행에 'target 셀 보조관측 사용' regime 열 부여(S5 아티팩트 기각과 기준 일치, 검증 지적 반영). `s11_comparison_table.csv`·`s11_multiaxis_validation.csv`·`s11_conformal_{results,meta}`, `outputs/figures/s11_uq/`.
- **적대검증 결과**: 직접 라벨 누설 0, 핵심 수치 전수 재현. 지적 반영 수정 완료 — 그림 제목 em-dash 제거(문체규칙), S8 환경구간 라벨 겹침(범위구분자 '~'·자릿수통일), S8 지도 overplotting(마커 투명·순서), S6 막대 y축 0시작, S7 지도 컬러바 분위조정·contour 클러터 축소, S10 갭색·masked contour, S9 경도라벨(160/150/140°W)·GIF duration 표기. 수치 표기정정: S7 프로파일 19개·CI[31.0,39.0], S6 b_cci −0.2~−0.9, S8 mix_mlp seed평균 라벨. **S2 physics-as-feature 로그 부호오류 정정**(위 2026-07-23 항목, CSV 대조로 무익→소폭개선이나 앵커 대체 불가).
- **다음**: 대회 예선 보고서 조립(§8 산출물 7종). 대표 시각화 셀렉션·통일 재렌더. S6 inductive 변형 이중보고는 여유 시.

## 2026-07-24 — S5: dense Stefan pseudo 사전학습→실측 finetune → 이득은 transductive 아티팩트

`RESEARCH_PLAN_multifidelity` S5. 물리(Stefan) 유도장을 dense 격자(pretrain_weaklabels 500k, 알래스카·서부캐나다)에서 사전학습한 신경망이 실측 finetune 후 from-scratch 대비 전이를 개선하는지 게이트식 검증. pseudo y=E_train·√TDD(fold-safe E), 격자 test-블록 버퍼 제거, 입력 14(지형+기후) 고정 공정비교, mlp·ftt 3seed. 스크립트 `s5_pretrain_finetune.py`.
- **게이트는 외견상 개선(FT-T scratch 22.47→pretrain 21.56)이나 전이 지식 아님.** 개선 전량이 Alaska fold(scratch 19.17→pretrain 15.22, Δ+3.95)에서 나오는데 이 fold는 **transductive**(격자가 test인 알래스카 공변량을 포함). 깨끗한 비-transductive 사례 **Lena는 Δ+0.05로 사실상 무효**(격자가 Lena를 안 덮음). Canada(transductive)는 오히려 Δ−1.29 악화.
- **결론**: 물리 사전학습 이득 = 격자가 target 공변량 공간을 덮는 transductive 노출 효과이지 전이 가능한 물리 지식이 아니다. S3(증강 이득 대부분이 target 앵커링)·S4(잔차 전이 파탄)·전이 상한 서사와 정합. covariate shift 밖에선 사전학습도 물리 앵커를 못 넘는다.
- **MLP 전이 발산 재확인**: 게이트 scratch 33.72·pretrain 36.96(둘 다 파탄), 전이엔 FT-T만 유효. in-domain AK는 mlp scratch 17.37→pretrain 15.70(Δ+1.66)로 사전학습이 in-domain엔 유효.
- **산출**: `s5_pretrain_results.csv`·`s5_pretrain_meta.json`. 시각화 `outputs/figures/s5_pretrain/`(scratch→pretrain 덤벨·게이트 막대, †=transductive 표기).

## 2026-07-24 — S3 버그 수정: 증강비율 표본상한 제거(포화 결론 정정) + FT-T 재확인

S3 재검토 중 **증강비율 반응곡선을 무력화하던 버그 발견·수정**. `take = min(n_ps, len(pseudo_idx))`가 pseudo 표본 수를 target 풀 크기(Lena 1519·Canada 371)로 상한 처리해, source(알래스카 13606)보다 풀이 작으니 r≥0.25 전 구간이 동일한 풀 전체만 사용 → r=0.25와 r=10 결과 완전 동일. `replace=n_ps>pool`로 오버샘플링을 의도했으나 상한이 무력화. **`take=n_ps`로 수정**(r≥0.25는 항상 replace=True 오버샘플링, r이 pseudo 손실 가중을 실제로 좌우).
- **"r≥1 포화" 결론 정정**: 포화가 아니라 상한 버그 아티팩트였음. 수정 후 catboost Lena stefan은 r=0.25 17.24→r=10 16.54로 **r=10까지 단조 개선**(포화 없음). 증강 이득은 r을 키울수록 계속 증가.
- **FT-T 증강 재확인(S1 전이 최선 모델)**: mlp가 전이 발산해 S3에 빠졌던 것을 FT-T로 보강. (정확한 r-스윕 재산출은 아래 재실행 결과로 갱신.)
- 영향 범위: 기존 S3 net-value(Stefan−placebo, 고정 r 비교)는 유효하나 곡선 형태·포화 서술은 수정판으로 대체. 실험로그·핸드오프의 "r≥1 포화" 문구 정정 대상.

## 2026-07-24 — S4: Stefan 앵커 + 저용량 잔차 shrinkage 게이트 → negative 확정(P2 재확인)

`RESEARCH_PLAN_multifidelity` S4. 재검증 질문: 과거 "잔차학습 무익(48cm)"은 covariate shift 심한 토양 입력 조건의 판정이었다. shift-robust 입력(지형6+기후8)·저용량 모델(ridge<catboost d3<catboost d6)·shrinkage λ∈{0,.25,.5,.75,1}로 재게이트. 예측=E_train·√TDD+λ·g(x), fold-safe E, LORO 매크로(Alaska·Lena·Canada) 비가중평균 게이트. 스크립트 `s4_residual_learning.py`.
- **게이트 판정: negative 확정.** 사후 λ 곡선에서 λ>0 전 구간이 게이트 악화(λ0 21.26 → λ0.25 21.83 이상). 저용량·shift-robust 입력으로도 잔차학습은 물리 앵커를 넘지 못한다.
- **지역 비대칭이 원인(블록부트스트랩 95% CI)**: Lena(+0.39~+1.00)·Canada(+0.91~+3.16)는 유의 개선. 그러나 Alaska fold(잔차를 비알래스카 소표본 3.8천셀에서 학습→13.6천셀 전이)는 −3.4~−35cm 유의 파탄. 게이트(비가중평균)는 Alaska 파탄이 지배. → 잔차 전이는 "라벨 풍부 지역→빈곤 지역" 방향만 소폭 유효, 역방향은 파탄.
- **λ 자동선택 불가 실증**: inner 공간블록 CV는 in-domain 이득만 보고 λ>0 선택 → 게이트 27.7~30.0 붕괴(catboost). ridge만 λ*≈0 선택해 21.2 유지. 타깃 라벨 없이는 shrinkage 조절 자체가 불가능(전이 상한 재확인).
- **in-domain은 반대로 개선**: AK 공간블록 14.46→**13.33**(ridge·shared25·λ0.75, 프로젝트 in-domain 최저. 기존 최선 MLP 14.37 하회). Stefan 앵커+잔차 구조가 in-domain 유효 확인.
- **E 추정기 발견**: 최소제곱 E(fidelity.fit_stefan_E)가 중앙값비 E(physics.fit_E, S2 게이트 22.24) 대비 LORO 앵커 우세(21.26). 물리 앵커 자체도 E 추정으로 ~1cm 개선 여지.
- **산출**: `s4_residual_{results,oof}.csv`·`s4_residual_meta.json`. 시각화 `outputs/figures/s4_residual/`(λ곡선 지역별+게이트·부트스트랩CI·in-domain vs 전이 대비).

## 2026-07-24 — S3: 물리 pseudo-label 증강비율 반응곡선 (엄격 통제) + 시각화 인프라 고도화

`RESEARCH_PLAN_multifidelity` S3. 알래스카 실측 + target(Lena/Canada) 물리 pseudo를 r∈{0,.25,.5,1,2,5,10}로 증강, target 전이 개선 규명. 공간블록 pseudo/test 분리(거리버퍼)·placebo(알래스카 평균 상수) 대조·블록부트스트랩. 스크립트 `s3_augmentation_curve.py`.
- **⚠️ 1차 결론은 표본상한 버그로 무효화**(아래 "S3 버그 수정" 항목·2026-07-24 참조). 원 스크립트가 pseudo 표본을 풀 크기로 상한 처리해 r 스윕이 무력(모든 r 동일). "r≥1 포화·물리 순가치 +0.71cm" 등은 저-r에 눌린 아티팩트였다. **아래는 수정 후 정확한 스윕 결과로 대체**.
- **핵심 발견(catboost·FT-T 일치, mlp는 전이 발산 제외)**: (1) **포화 없음**, r=10까지 단조 개선. (2) **물리 순가치(Stefan−placebo)가 r에 따라 증가**: Lena +0.8→+1.7cm, Canada +9.6→+10.4cm(두 모델 일치). (3) **Canada는 물리가 필수**: placebo가 −7~−8 악화인데 Stefan은 +2~+3 개선 → 앵커링이 아니라 물리 정보가 전이 견인. (4) **Lena는 base 모델 품질에 의존**: catboost(base 21.9, 약함)는 앵커링+소폭 물리로 개선, FT-T(base 14.5, 전이 최강)는 증강이 오히려 소폭 해(순가치는 여전히 placebo보다 나음). (5) **부정확 물리(Ku)는 r 키울수록 더 악화**(−16~−30). → RQ1/RQ3 답: 물리 pseudo 순가치는 정확한 물리·bias 큰 전이·약한 base일수록 크고, 부정확 물리는 양이 늘수록 해.
- **mlp 전이 발산**: 알래스카만 학습→Lena/Canada 극단 covariate shift에서 신경망 발산(base RMSE 3.6만). catboost·FT-T 강건. 시각화·결론서 mlp 제외.
- **산출**: `s3_aug_curve_results{,_ftt}.csv`·`meta.json`. 시각화 `outputs/figures/s3_aug/`(반응곡선 물리vs placebo·물리 순가치, 두 모델 일치).
- 대회 함의: "증강 비교분석표"의 핵심 = 증강이 언제 돕고(정확 물리·bias 큰 전이·약한 base) 언제 해치나(부정확 물리·이미 강한 전이 모델). 단순 병합 무익 결론(P1/P2)을 조건부로 정밀화.

## 2026-07-24 — 전체 리뷰(S0~S2) + must_fix 처리 + 시각화 고도화 착수 → S1 재렌더

시각화 인프라 논문형 고도화(4렌즈 조사: Ran2022·Whitcomb2024·Obu2019·ESA CCI·Crameri2020). `src/polar/geomap.py`(hexbin_map·field_map·support_mask·add_inset_locator·add_scalebar·circular_boundary·to_proj·ALT_LEVELS), `src/polar/gridding.py`(make_grid·interp_obs·grid_predict) 신규. S1 대표그림 재렌더(hexbin 셀통계+범북극 위치 inset+스케일바). S0/S2 재렌더는 보고서 단계로 유예. 이후 각 단계 결과물은 이 인프라로 시각화.

## 2026-07-24 — 전체 리뷰(S0~S2) + must_fix 처리

멀티에이전트 전체 리뷰(3렌즈: 누설·결과정합·물리/시각화). **critical 0건, pytest 통과, 헤드라인 무효화 누설 없음**. 발견된 must_fix 처리:
- **LORO 지리 누설 수정**: `MACRO_REGION` 도입, `loro_splits` 매크로 지역 기준(ABoVE_AK+US Alaska→Alaska). 동일 지점 세부라벨 train/test 분리 제거. LORO 게이트 21.86→**22.24cm**(US Alaska 제외 정정).
- **s2 OOF E=1.0 스케일 버그 수정**: `phys0=physics_ensemble(df, E=E_GLOBAL)`. p1_stefan OOF 중앙 30.5→**47.5cm**(관측 47.9 정합). fold-safe p1(`p1_stefan_calib_ak`) 별도 저장.
- **fold-safe 가드 프로덕션 배선**: `assert_fold_safe_E`를 s2 E 역산 직전 삽입. `prep`을 `src/polar/preprocessing.py`(`fold_prep`)로 공용화(s1/s2 중복 제거). 테스트 11→**16개**(physics.fit_E fold-safe·fold_prep train-only·macro LORO·sigma_prior 금지 추가).
- **sigma_prior_cm 금지**: `LABEL_DERIVED_BANNED`에 추가(alt_sd 파생, S3 σ 역가중 누설 방지).
- **서술 정정**: S1 in-domain은 "3-seed 앙상블 OOF 14.37/14.40, seed-mean 14.66/15.03" 병기. "Stefan 최선"→"**Stefan이 전이 하한 확보·GBM 압도, 최신 DL(FT-T)과 동급**"(seed 노이즈 범위). 멤버 상관 0.89~1.00.
- **다음 순서 확정(사용자)**: S0~S11 전부 완주. 시각화 먼저 고도화(최신 논문형 지도·3D·timelapse). 핵심 S3→S11 먼저, 나머지 순차. GPU 6-9.
- 리뷰 상세: `gpt/handoff` 또는 세션 기록. 남은 low(meta 해시 일부·TTOP 마스크 그림 범례)는 시각화 재생성 시 처리.

## 2026-07-23 — S2: 물리식 5종 앙상블 baseline + physics-as-feature(A2)

`RESEARCH_PLAN_multifidelity` S2. 워크플로 정밀조사(수식·계수·단위 실측검증)로 `src/polar/physics.py` 구현(Stefan 기본·edaphic·TTOP·Kudryavtsev·λ보정). SoilGrids 이미 물리단위 확인(이중변환 금지), bdod×1000·soc/1000·TDD×86400 가드.
- **Part A 물리 baseline**: in-domain **p1 Stefan 14.56cm**(bias -0.99, 정확도 담당) ≪ p4 Ku 25.29 < p3 TTOP 31.10 < p2 edaphic 40.95 < p5 λ 46.30(정교화 물리는 상방편향 +32~38cm). **LORO 게이트(비가중평균 AK·Lena·CA 고정) p1 Stefan 21.86cm 최선**(AK17.5·Lena21.7·CA26.4), p4 Ku 39.66(2위), 나머지 59-69. → **물리 정교화가 전이 개선 안 함**(기본 Stefan이 전 지역 최선), Gautam2025·W3 정합. 게이트 프로토콜 고정(기존 18.24는 다른 지역집합/집계, 비판 지적 반영).
- **Part B physics-as-feature(A2)**: CSV 재계산 기준 in-domain 소폭 개선(catboost 15.55→15.29 Δ−0.26·mlp 14.66→14.58 Δ−0.08·lightgbm 16.78→16.45 Δ−0.33), LORO catboost 지역평균 39.37→37.28(Δ−2.09)도 소폭 개선. **그러나 base·+physics 모두 순수 ML 전이 파탄대(catboost 37~50cm)**로 Stefan 앵커(LORO 21~22cm)를 전혀 대체 못 함 → **A2 미채택**(물리 feature는 파탄한 ML을 미미하게 완화할 뿐, 전이는 앵커=p1 직접예측으로만 유효). *최초 로그의 'Δ+0.04 무익·LORO 악화'는 표기 오류로 정정(2026-07-27 S11 검증서 `s2_physics_results.csv` 대조 확인).*
- **멤버 다양성 한계(적대검증)**: 5종 상관 0.93~1.00(전부 Stefan축). 실질 다양성=수준 오프셋·TTOP 동토마스크(81.5%)·Ku 눈 성분(p1과 비상관 최대). phys_std는 상대 불확실성 지표.
- **산출**: `s2_physics_results.csv`·`s2_physics_oof.csv`·`s2_physics_meta.json`. 시각화 5종 `outputs/figures/s2_physics/`(물리5종 지도·앙상블스프레드·동토마스크·물리별LORO·feature효과, 실제 지도배경).
- 다음: S3(증강비율 반응곡선, 엄격통제) 또는 S6(source-aware). 전체 리뷰 후 결정.

## 2026-07-23 — S1: 실측-only 다모델 baseline (여러 DL 병렬, 표준화 버그 수정) + 지도 시각화 인프라

`RESEARCH_PLAN_multifidelity` S1 완료. 모델군 7개(HistGBM·LightGBM·XGBoost·CatBoost·MLP·FT-T·TabM). 평가 2축: 알래스카 in-domain 공간블록 6-fold(FULL 34), LORO 전이(SHARED 25, SAR 제외). 3-seed.
- **결과(표준화 후, seed평균)**: in-domain **MLP 14.37·TabM 14.40**(신경망 최선, 대표성 하한 14cm 도달) > CatBoost 15.61 > XGBoost 16.16 > LightGBM 16.48 > HistGBM 17.21 > FT-T 18.56. 전이(Lena) **FT-T 22.5**(최선) > TabM 27.3 > MLP 28.8 ≫ GBM류 40-57(covariate shift). → in-domain은 신경망/GBM 접전, 전이는 DL 우세. **기존 "6모델 동률·GBM 우위 16.95" 갱신**(표준화 신경망이 앞섬, 단 CI는 S11).
- **버그 2건 근본 수정**: (1) **GPU 오용** — CUDA_VISIBLE_DEVICES가 torch 초기화 후 설정돼 물리 0번(타 사용자 공유) 사용. `tab_models.py` lazy CUDA(`_dev()`) + `s1` GPU고정을 전 import 앞 + 6/7/8/9 assert + uuid 가드. 검증: PID가 물리6(uuid b840175b) 사용·GPU0 무점유 확인. (2) **torch 입력 미표준화** — 신경망에 raw 스케일 투입해 TabM full 61cm 발산·MLP 저평가. fold-safe z-score 추가 → MLP/TabM 14.4cm로 정상화·최선 등극. grad clip도 추가.
- **신규 모듈**: `src/polar/tab_models.py`(7모델 통합 인터페이스, available_models 자동감지), `src/polar/geomap.py`(cartopy 실제 지도배경 매핑, 알래스카/범북극/레나 프리셋), `scripts/3_deep_learning/s1_baseline_tournament.py`, `scripts/4_visualization/s1_baseline_figs.py`.
- **산출**: `s1_baseline_results.csv`·`s1_baseline_oof.csv`·`s1_baseline_meta.json`. 시각화 4종 `outputs/figures/s1_baseline/`(실제 지도배경 위 관측vs예측·잔차맵·Taylor·모델비교막대, 냉색).
- **보류(자동 스킵)**: TabPFN(라이선스 `TABPFN_TOKEN` 필요), RealMLP/pytabkit(torchvision 충돌). FT-T in-domain 저조는 별도 점검 대상.
- 다음: S2(물리식 5종 앙상블 Stefan/modified/λ보정/TTOP/Ku, LORO 18.24 하한 게이트).

## 2026-07-23 — S0 착수: fidelity 스키마 + 누설방지 pytest + overlap gate + 시각화

세부계획(`docs/RESEARCH_PLAN_multifidelity_2026-07-22.md`) S0 구현·검증 완료. 모델 무관 공통 계층.
- **`src/polar/fidelity.py`**(신규): 공변량 코어34(지형6+기후8+토양9+InSAR5+PolSAR3+CCI2+flag1) 전량 사용, 라벨파생7(alt_sd/min/max·n_obs·n_years·year_min/max) 영구제외. split 3축(0.5°블록 GroupKFold·LORO·leave-source-out), fold-safe Stefan E 역산(assert_fold_safe_E), SHARED_CORE25(SAR 제외 pooled 전이용).
- **`build_fidelity_schema.py`**(신규): `fidelity_base.csv`(17423×45), `source_overlap_matrix.csv`, `fidelity_observations_long.csv`(59184행: F4 17386·CCI 17340·InSAR 14348·PolSAR 10073·GTNPenv 37), `covariate_availability_by_region.csv`, `fidelity_schema_meta.json`.
- **overlap gate 결과**: direct 대비 Stefan 100%·CCI 99.5% clean(full source-aware 가능), InSAR 82.4%(AK100/Lena0/CA100)·PolSAR 57.8%(AK74/나머지0), F3 온도유도 151쌍. → A5 clean 소스는 Stefan·CCI뿐 재확인.
- **`tests/test_leakage.py`**(신규): 누설방지 11테스트 전부 PASS(라벨파생 제외·타깃 제외·SAR 공유코어 배제·블록 GroupKFold 비중첩·폴드 커버리지·LORO 지역분리·leave-source-out·fold-safe Stefan E·가드 자체검증). **이후 모든 게이트 무결성의 전제.**
- **`s0_schema_figs.py`**(신규): overlap 히트맵·0.5°블록 폴드 지도(누설통제 시각확인)·지역×공변량 가용성 막대. 냉색 cmcrameri, PNG300+PDF, `outputs/figures/s0_schema/`.
- 다음: S1(실측-only baseline, 여러 DL 병렬: GBM3·RealMLP·TabM·FT-T·TabPFN, 하나로 단정 금지).

## 2026-07-22 — Source-aware multi-fidelity 세부계획(GPT 로드맵 반영) + 6개 질문 근거분석

사용자 6개 질문(좁은지역 ALT분산·모델고도화·공동학습·input·차별성·불확실성)에 데이터·문헌 근거로 답하고, GPT 멀티충실도 로드맵(`gpt/ALT_multifidelity_...`)을 우리 자산에 매핑한 세부계획 수립. 멀티에이전트 워크플로 2회(차별성 다각도조사 6에이전트, overlap실증+novelty+재검증+종합+비판 5에이전트).

### 데이터 근거 확정
- **좁은지역 ALT분산**: 측정정밀(동일좌표 반복 SD 0.2cm, 측정오차 median 7.7cm), 좁은지역 큰차이는 진짜 미세환경변동(100m 셀내 range 19cm, 같은 100m에 23~226cm 공존). 위치간 분산 86.3% vs 위치내 13.7%. 스케일 의존성(셀내SD 30m 3.7→1km 11.1→10km 13.2cm)이 공간구조 증거. 소수 극단값은 결측코드·site뭉침 품질이슈.
- **모델 비교**: in-domain 앙상블 16.95≈Diffusion 17.09≈GBM 17.24≈Flow 18.31(부트스트랩 동률). LORO Diffusion 23.48·Flow 32.39 > GBM 20.82(생성모델 전이 열세). 정확도는 정보병목 지배.
- **불확실성**: GBM+conformal 56→86%로 충분, 전이선 GBM>diffusion. 생성모델 과신(cov90 74/70.8%), CQLDM식 conformal 후보정 필요.
- **source overlap(GPT 중단기준 판정)**: direct×Stefan 100%·CCI 99.5%(full A5 가능)·InSAR 79.3%(알래스카만)·온도유도 셀 9~17개(paired 107~130). Stefan·CCI는 이미 51 feature라 A5 구조증분은 실증대상.

### 세부계획 → `docs/RESEARCH_PLAN_multifidelity_2026-07-22.md`
증강을 "라벨 개수 늘리기"→"자료원별 신뢰구조 모델링(source-aware multi-fidelity)"로 승격. 사용자 3종 증강 비전 유지하되 병합 아닌 관측모델 y_s=A_s[z]+b_s+ε_s로 분리추정. 단계 S0~S11(실측baseline→Stefan→physics feature→증강비율 반응곡선→source-aware A5→UQ+비교표→...). 9일 현실경로 S0→S1→S2→S3→S6→S11.
- **적대검증 핵심 정정**: S3 초안 동기수치(증강 14.96→14.26)는 **이미 특징복제 누설로 기각된 헤드라인**(donor 제거 15.99). 재동원 금지, 거리버퍼·블록부트스트랩·placebo 셔플대조 게이트 강제. 순서 S2(하한게이트)를 S3 앞으로. 서사축을 S6 정확도가 아니라 S11(UQ)+S2(물리앵커 18.24)+S7(검열 방법론)+S3(반응곡선)에.
- **novelty**: 6요소 조합(source-aware+검열+support+다축OOD+반응곡선)은 ALT 도메인 미발표이나 요소별 first-claim 전부 금지(SCE 2025·Gautam 2025·GeoCryoAI 2025·NCAM 2026·Read 2019·Du 2025 선점). "우리가 아는 한 ALT 최초"로 한정.
- **정직한 경계**: A5 깨끗한 소스는 Stefan·CCI뿐(둘 다 이미 feature), 온도유도 식별 얇음(shallow3d 18쌍 corr 0.28). KPDC는 방법론 앵커로만 방어(보고서 첫머리+정량표 필수). in-domain 정확도는 대표성 하한 14cm에 막힘.

## 2026-07-21 — KPDC 신규 데이터 정리 + 문헌 종합 실험계획(면적검증·pooled·물리주입·KPDC)

멀티에이전트 워크플로(에이전트 12, 툴콜 181)로 KPDC 신규 파일 26종 병렬 파싱 + 문헌 4렌즈 조사 후 실험계획 종합·적대검증.

### KPDC 폴더 정리 (`kpdc/`, gitignore 로컬 전용)
- 64파일을 사이트/데이터종류 2계층으로 재구성: `council/{soil_temp/{active_layer_profile_5min,daily_profile_ID,zl6_shallow_10_40cm,wireless_nodes_2021},soil_moisture,core_alt,aws_met}`, `kougarok/`, `c1_toolik/`, `archive/{zips,duplicates}`. 인벤토리 `kpdc/README.md`.
- **핵심 발견**: (1) ALT 직접 라벨 = `core_alt/AK_core_sample_2022.xlsx` 코어길이 18개(72-88cm, SF1-6×C/H/L). (2) ALT 간접 유도 = ID21-24(1.6m 16층) 최적·ID02-05(0.8m) 차선(0°C 등온선). (3) 완전중복 2종 격리(VWC 2022=온도 2022 md5동일, Avr _vol=_temperature). (4) 공통제약: 좌표·센서깊이 메타 부재 → KPDC 페이지 확보 필요.
- `parse_kpdc_met.py` 경로 갱신(council/aws_met, c1_toolik/aws_met) 후 재실행 검증 완료(`kpdc_station_climate.csv` 재생성).

### 실험계획 (`docs/EXPERIMENT_PLAN_2026-07-21.md` + `_쉬운설명.md`)
연구목적 재정렬(정확도·차별성, 정직성은 도구). 우선순위 E1→E5→E4→E3→E7→E2→E6. 전역통제(≥3seed·74블록 부트스트랩·거리버퍼·셀 통째배정).
- **적대검증이 게이트 정정**: E1 원안(1km≤12cm)은 `grid_support_results.csv`(지지↑→RMSE↑: 점17.04→1km17.49→25km23.05)·`areal_eval_results.csv`(1km 18.82)와 충돌 → **대표성 귀속(Parsekian 오차분해)을 주게이트로**, 셀내SD 9.7-13.3cm가 잔여RMSE ≥60% 설명. E2는 `insar_ablation`(+InSAR 18.79>BASE 17.24)·`field3d_reeval_leakage`(nn 0.05-0.2km) 근거로 footprint 버퍼 강제·R²와 국소RMSE 분리보고·우선순위 하향. E5 blocker(깊이메타) 과장 정정(ID01-24 깊이라벨 있음)·사례연구 프레임. E3 혼합 test 반드시 공간블록(무작위 vs 블록 병기).
- **문헌 근거(웹검증)**: Gautam 2025(RF 시험 22cm·Stefan 18cm, 물리 외삽우세), Uxa 2026(ASM 14-18cm 바닥), Du 2025(스케일 오차예산), Merchant 2024(InSAR 업스케일 R² 0.476), Whitcomb 2023(CALM 11-12cm·P-band 65cm 포화), Parsekian 2021(오차 3분해), PI-LSTM Liu 2023(물리 pretrain+27~69%), Ohmer&Liesch 2026(유사도 층화 pooled), AlphaEarth 2025·Nakata 2026(임베딩).

## 2026-07-14~21 — P0·P1·PPT + P2 3트랙 + 회의적 재검증(Phase1·W2.1·W3) + 알래스카 내부 3트랙 + 연구목적 교정

대형 세션. 회의적 재검증 원칙(모든 헤드라인은 공간블록+LORO·실측 held-out·적대적 검증)을 세워 다수 기존 결론을 교정. 상세는 개별 docs 참조.

### P0·P1 + PPT (`docs/EXPERIMENT_P0_P1_RESULTS_2026-07-14.md`)
- **P0**: 데이터 인벤토리 세계지도(`map_data_inventory_world.py`) + 6모델 예측·오차 지도(`map_tournament_error_maps.py`). 위치가중 GBM 16.1 ≈ Diffusion 16.2 ≈ 앙상블 16.2cm(동률 재확인).
- **P1**: 다지역 통합 셀 v2 조립(`assemble_cell_v2.py`, `parse_allena.py`·`parse_qtec.py`·`derive_alt_gtnp_envelope.py`·`enrich_new_regions.py`). +레나델타 3,037·GTNPenv 37·QTP 1. 하네스 `unified_tournament_cell.py`(전 공변량 25, 공간블록+LORO, GPU). 결과 `unified_tournament_*.csv`.
  - LORO 전이서 DL(FT-T·앙상블) 15.0cm > GBM 17.6(알래스카). 레나 전이 25-30cm 병목. 결측 라우팅 아티팩트(NaN 네이티브 GBM "InSAR 결측=깊은 ALT" 오학습). 통합학습 게이트 미채택.
- **PPT**: 중간보고 21슬라이드(`build_midreport.py`, P0·P1 반영 5b·13b·15b) + 슬림 11p(`build_summary.py`, `mk_summary_figs.py`). 페이지7 MAGT 지도 버그(전지구 시추공이 축 늘림) 수정. RMSE 라벨 정정(공간블록≠전이). 선행연구 통제 과표현 정정.

### P2 3트랙 (`docs/EXPERIMENT_P2_RESULTS_2026-07-14.md`, `p2_{augment,field,stefan}_experiment.py`)
- **핵심**: Stefan 물리(a+E√TDD) LORO 18.2cm ≫ 순수 ML 40.6cm(알래스카 과적합). 잔차학습 무익(REJECT). 물리 우선이 전이에 강함.
- 라벨 증강 미채택(GTNPenv 심부 교란). 3D 기질 전 공변량 ADOPT 잠정(→Phase1서 기각).

### Phase 1 회의적 재검증 (`docs/EXPERIMENT_PHASE1_2026-07-20.md`)
- **증강 "해가 된다" 부분기각**(`aug_backbone_dissect.py`): 증강 자체가 아니라 "심부 GTNPenv 라벨 + 결측 모달리티(신규지역 InSAR/PolSAR 100% 결측=완전 공선)" 결합만 붕괴(레나 22→88cm). 물리·기후만 ML은 면역.
- **3D "지형+CCI 심부 개선" 기각**(`field3d_reeval.py`): site-GKF 누설 착시(72.6% 사이트가 같은 0.5°블록). 누설통제 시 악화(LORO 1.60→1.73°C).
- 평가 프레이밍 정정 `docs/EVAL_FRAMING_NOTE.md`. 계획 재배치 `docs/RESEARCH_PROGRAM_2026-07-17.md`(증강 백본 W1 최상단).

### W2.1 SoilGrids + KPDC (`docs/EXPERIMENT_W21_KPDC_2026-07-20.md`)
- **SoilGrids**(WCS로 취득, VRT 정체 우회, `enrich_soilgrids_wcs.py`·`soil_ablation_gate.py`): 게이트 미채택. 내삽 개선(+5.6%)·전이 붕괴(−63.8%, 레나 62.6cm). 결측 없어도 전이 실패 = **진짜 covariate shift**.
- **KPDC 콘슬**(`parse_kpdc_met.py`·`kpdc_era5_validation.py`): ERA5 √TDD가 실측과 정합(bias ~0.1). 단일 E Stefan 콘슬 1.7배 과대예측(E(x) 동기, 단 in-domain·평균회귀).

### W3 물리결합 엔진 (`docs/EXPERIMENT_W3_2026-07-20.md`, `w3_physics_ml.py`)
- 가설 "토양 E(x)·물리식 형태강제 ML(구조 C)이 전이 회복" **기각**. PHYS_const(상수 E) LORO 18.24cm 여전히 최선. PHYS_soil 19.99·PHYS_nn(미분물리층) 28.5 악화. 모든 모델 레나 skill 음수 → 라벨 없는 OOD 전이는 모델 구조로 못 뚫음.

### 알래스카 내부 3트랙 + 적대적 검증 정정 (`docs/RESULTS_SUMMARY_2026-07-20.md`)
- **증강 × 다중 DL**(`aug_within_alaska.py`): 1차 "4모델 유의 개선"이 **적대적 검증에서 기각**. GBM 개선=test 인접 특징복제 누설(제거 시 14.2→16.0), MLP=seed 운(블록부트스트랩 CI 0 포함). 살아남음: MLP>GBM ≈−0.7cm(3-seed), Stefan 라벨 placebo 대비 정보성. → 증강 개선 미성립, 재실험 조건(거리버퍼·블록부트스트랩·multi-seed·nested) 도출.
- **timelapse**(`timelapse_alaska.py`, GPU 9): 연별 지도는 물리 forcing 최선(연도 홀드아웃 14.97cm). **연도 간 anomaly 예측 불가**(corr 0.06). GIF `outputs/animations/timelapse_alt_alaska.gif`.
- **얕은 3D**(`shallow3d_alaska.py`): 알래스카 0-3m 실측 764행, 필드 2.66°C·R² 0.47, 0°C→ALT r 0.28(심부 0.16 대비 개선, 절대 정합 미완).

### 비판적 검토 + 연구목적 교정
- `docs/CRITICAL_REVIEW_2026-07-20.md`: 점 검증 대표성 잡음(~12cm) 상한, 음성결과 반복, InSAR 스칼라 증류·미활용, KPDC 검증만 사용.
- **연구목적 재확정(사용자)**: "정직함"은 헤드라인 아님. 목적 = 기존 논문 대비 새롭고, 더 많고 적절한 데이터+증강+좋은 DL 비교로 **ALT(2D)·4D(timelapse)·얕은 3D를 정확하게 예측**. 불확실성은 도구.
- **다음 방향**: (9) 전 지역 pooled 학습(전이 아님) + (8) InSAR 30m 제대로 활용 + (11) 물리+ML fine-tune. KPDC는 대회 규칙 충족·검증 보조(과학 엔진은 자체 대형데이터).

### 신규 KPDC(2026-07-20 16:22 추가)
콘슬 8층 토양온도(L1-L8)·VWC·CO2/CH4, 쿠가록 화재/비화재 토양온도·수분, 2016 토양물성(Thaw depth 실측), AWS 2023·2025. 파싱 미착수(다음 세션).


## 2026-07-10 — overnight: 셀 단위 재분석 + T-lite 게이트 + 데이터 확충 + 발표덱 v2

GPT 계획(`gpt/20260709_claude_next_research_plan_dl_alt_3d.md`) P2/P3/P6-C 실행. 코드: `scripts/2_evaluation/overnight_cell_experiments.py`, `scripts/3_deep_learning/tlite_sequence_gate.py`, `scripts/1_data_prep/enrich_cci_cell.py`.

### 셀 단위(location-equal) 다중모달 ablation — 정직한 재분석 (`alt_ablation_cell_results.csv`)
- 기준선 = `dl_dataset_cell.csv`(14,348 셀, 셀평균 ALT). 공간블록·LORO, 표준지표.
- **LORO**: M0 지역평균 21.8 · M1 기후 **16.45(skill 10.8%)** · M3 기후+지형 16.94(지형 추가 악화) · **M4 +InSAR 16.09(skill 12.7%, 물리 최고)** · M5 +PolSAR 16.98 · M9 전체 16.43.
- **위치 대조군(lat+lon 2피처)**: LORO **15.72cm skill 14.7%** — 물리 공변량 조합보다 높음. 위도가 기후 이상을 대리 = **정보 병목의 직접 증거**. (점-단위 옛 ablation의 15% skill는 pseudo-replication 착시였음이 셀 재평가로 확증.)

### 보정 UQ + AOA (셀, `alt_conformal_cell_results.csv`, `alt_aoa_cell_transfer.csv`)
- raw 분위-GBM 90% 커버리지 **56.1%(심한 과신)** → **CQR 보정 85.9%**(폭 50.6cm). 점-단위(71%)보다 raw 과신 심함.
- AOA DI-구간(qcut10→중복제거 6구간): RMSE 저DI 13 → 고DI 30cm. **커버리지는 비단조**(D1 61% → D3 88% 피크 → D6 50%) — 공간 calib/test 분리 + marginal 보장 특성. 정직히 표기.

### T-lite 시계열 DL 게이트 — 정직한 음성 (`tlite_sequence_gate_results.csv`, gate_meta)
- CALM site-year 251사이트·3,345 시퀀스. GRU/TCN vs persistence·climatology·GBM-annual. 검증: site-disjoint 5-fold + temporal holdout(≤2014/≥2015).
- **site-disjoint**: GRU 16.79 < persistence 16.98 < GBM 17.33 (GRU 소폭 최우수). **temporal holdout**: **GBM-annual 15.86 < persistence 17.02 < GRU 19.15 < TCN 23.85** (DL 붕괴).
- **게이트 미통과**(temporal 미충족) → 부록/future work 강등. **정적 tabular ALT는 GBM으로 충분** 재확인. DL은 고차원 EO/SAR·시간축에서 게이트 통과 시만.

### 데이터 확충
- **ESA CCI ALT prior**: 25년 다년평균을 14,348 셀에 추출(`enrich_cci_cell.py`), 전 셀 유효, 관측 셀평균과 **r=0.53**. ablation **M8 +CCI**: 개선 없음/악화(기후와 중복) — 정직한 음성. CCI는 prior/benchmark로만.
- **SoilGrids**: ISRIC VRT vsicurl 원격 읽기 정체(산출 0) → 중단, 다음 세션 재시도(사전 타일 캐시 권장). 계획: `docs/DATA_ACQUISITION_PLAN.md` 갱신.

### 발표덱 v2 (에디토리얼/학술 보고서)
- v1의 "AI틱 라운드카드·테크그라디언트" 결별: 종이 배경 + 세리프 제목(Noto Serif CJK KR) + Pretendard 본문, booktabs 표, 저널형 러닝헤더/푸터, 박스없는 figure-of-merit, 렌더된 수식(skill·Stefan·CQR·DI·분산분해), 번호 캡션. 코드 `deck/report_lib.py`·`deck/build_report.py`(18슬라이드). 배경·동기·선행연구·연구질문 슬라이드 추가.
- 시각+과학 리뷰 반복 반영(수식 여백·표 넘침·여백축소·GRU 게이트 정직 서술·Mloc=위경도·AOA 비단조·CCI 중복). 산출: `deck/render/permafrost_report.{pptx,pdf}`.
- **문체 규율**: 모든 프로젝트 기본 = 정돈된 보고서/논문 톤(메모리 `report-tone-default` 고정, 전역 규칙 `~/.claude/rules/writing-tone.md`).
- **덱 v2 개정(사용자 지적 반영)**: 렌더 텍스트에서 em-dash(—) 전량 제거(불릿 머리·러닝헤더 포함), 전역 규칙에 em-dash 금지 명시. 장식 위젯 절제(finding 컬러 세로바 제거, fom 컬러 규칙선 제거). 그림은 논문 관례로 재작성(도판에 박힌 결론형 굵은 제목 제거, 패널 라벨 (a)/(b), 회귀선 추가). 친절한 예시 추가(skill 계산 예, pseudo-replication 34/96cm 예, 누설 예). 수식 캐시 경로 버그 수정(`assets/eq/`). 시각·과학 리뷰 재반영.
- **미완/다음 세션**: 그림 전면 재구성(concept·지도류 포함) 및 PPT 전면 재구성 검토. 사용자가 GPT와 상의 예정. README·PLAN_FORWARD·EXPERIMENTS 구계획 서술 현행화 필요(교차문서 감사 지적).

## 2026-07-08 — 스레드 R 착수: 데이터 재구조화(㉡집계·㉢가중) + ERA5 다년 확보

### 전략 확정 (재구조화 먼저 vs 다운로드 먼저)
- 확인: ERA5 원본이 디스크에 **2015–2020만** 있는데 **라벨 70%가 2010–2014**(2014=59%). → ㉠ 시간정합은 ERA5 다년 다운로드가 전제.
- 결론: **㉡집계·㉢가중은 무료·즉시**(재구조화 먼저 맞음), **㉠은 targeted ERA5 다운로드 필요**(둘 병렬). 새 모달리티(SoilGrids/Sentinel)는 재구조화 기준선 후.

### ㉡㉢ 집계·가중 (`aggregate_alt_cell.py`, `restructure_gate.py`)
- `dl_dataset_cell.csv`: 225k → **14,348 위치당 1행**. 정답=셀평균 ALT, **alt_sd=셀내 SD(불확실성 라벨, 중앙 2.0cm)**, 위치 동등가중.
- **R3 게이트**(셀평균 ALT 공정채점, 공간블록+LORO): (1)pooled 현재 skill **−0.8%/0.04%** → (2)**+1/n가중 10.9%/7.3%** → (3)cell학습 −1.9%/**8.1%(전이 최고)**.
- **핵심 발견**: 현재 pooled는 위치-동등 채점 시 **거의 평균 수준**(밀집셀 편향으로 점-단위 R²0.2가 부풀려짐). **재가중만으로(무료) 보간 +11%·전이 +7~8% 회복.** 적대검증: 가중치 정합(0.005~1.0, loc합=1)·상위1%셀이 점의 11.9% 확인 → 버그 아님. 절대 skill은 fold 분산 큼(정성 결론). `restructure_gate_results.csv`, `figures/02_evaluation/restructure_gate.png`.

### ㉠ 시간정합 준비 완료
- **ERA5-Land 2010–2024 다운로드 성공**(`era5land_monthly_multiyear.py`, 816MB, 180 monthly steps, t2m/sd/stl1). → 라벨 99.4% 연도정합.

### ㉠ 시간정합 게이트 (`era5land_temporal_covariates.py`, `temporal_gate.py`) — 정직한 혼합/음성
- 연도별 도일/적설/토양온도 파생(`alt_era5_temporal.csv` 2.96M행, 같은 위치 연도별 TDD SD=145 → 연도신호 실재). (위치,연도) 17,800단위, 조인 100%.
- **static vs temporal(그해 기후) GBM**: 보간(공간블록) temporal 8.7%>static 5.7%(+3), **전이(LORO) temporal 5.1%<static 9.9%(−5)**, **per-year holdout temporal 20%<static 28%(−8)**.
- **결론**: 그해 기후를 GBM에 스냅샷으로 넣는 것은 **매핑엔 도움 안 됨**(전이·연도holdout 악화). **ALT 변동은 '그 해 날씨'보다 '위치 고유성질'이 지배**(static이 위치 기후평년 학습 → 강한 baseline). 시간축은 **매핑 지렛대 아님** → 시계열 신호는 **lagged ALT(사이트 지속성)** 로만 유효한데 그건 모니터링 사이트 **예측(T1)** 용(매핑엔 미가용). **T-lite/GRU는 매핑용으로 게이트 탈락**, 예측 응용으로만 별도.

### 스레드 R 종합 (재구조화)
- **채택**: ㉢ 1/n 가중(무료·큰 이득 +11%/+8%) · ㉡ 셀집계+셀내SD(불확실성 라벨·척도정합).
- **탈락(게이트)**: ㉠ 시간정합 climate 스냅샷(매핑엔 혼합/음성).
- **정직한 함의**: "데이터를 올바르게 넣기"의 실질 이득은 **가중/집계**였고, 정확도의 남은 지렛대는 (a)새 모달리티/세밀 해상도 (b)예측(T1)으로의 응용 전환. 모델(DL)이 아님 재확인.

## 2026-07-06 18:40 — P0 실행 + 스레드 A(다중모달 ablation) + 횡단 AOA/UQ

### P0/P1 (기반)
- **`src/polar/eval_metrics.py`** 표준지표(rmse/mae/bias/r2/target_sd/skill_over_mean/coverage/width) — 모든 결과가 RMSE 옆 R²·skill 병기.
- **재채점**(`rescore_results.py`): 토너먼트 R²(앙상블 0.23·GBM 0.20, 전부 skill~12%), 큐레이션 skill **전역 10.4% > 평탄툰드라 7.4%** → "12.97 SOTA 돌파"가 범위축소 아티팩트임 확증. 산출 `model_tournament_results_rescored.csv`, `curated_scope_results_rescored.csv`, `figures/02_evaluation/skill_reframing.png`.
- **apparent-floor 진단**(`diagnose_apparent_floor.py`): 분산분해 within 13.7%/between 86.3%, **비가역하한 ~7.2cm ≪ 현재 16.9cm** = covariate 병목(헤드룸). 산출 `apparent_floor_diagnosis.csv`, `figures/02_evaluation/apparent_floor_diagnosis.png`.
- **`design/`** 디자인 시스템(brand_tokens/layout_rules/visual_qa_checklist).

### 스레드 A — ALT 다중모달 feature ablation (`alt_feature_ablation.py`, GBM 고정, 공간블록+LORO)
- PolSAR/InSAR 데이터셋 행정렬 검증 후 결합(14+PolSAR3+InSAR5). HistGBM NaN 네이티브.
- **핵심(정직-평가 스토리)**: within-domain(공간블록)=**기후(ERA5) 지배**(M2 16.3cm, skill 15%), 지형 추가는 **공간 과적합**으로 악화(M3 19.4); transfer(LORO)=**InSAR 필수**(M4 16.4 최고), 지형만 31.9cm 파탄(skill −65%). **정보원이 보간 vs 전이에서 다름.** per-fold 검증(악화는 fold0 지형=지역대리 과적합)으로 버그 아님 확인.
- 산출 `alt_feature_ablation_results.csv`, `alt_ablation_M6_oof.csv`, `figures/06_deep_learning/alt_feature_ablation.png`. 미취득: SoilGrids/Sentinel/CCI(다음 데이터 확장).

### 횡단 — AOA + Conformal UQ (`aoa_conformal_alt.py`, `aoa_transfer.py`)
- **within-domain CQR**(spatial-block): raw quantile-GBM coverage **71.2% → CQR 보정 89.2%**(목표 90%, width 37.7→53.7cm). 생성모델 과신 교정 실증. `alt_conformal_aoa_results.csv`, `figures/02_evaluation/coverage_calibration.png`, `maps/alt_uncertainty_width.png`.
- **transfer AOA**(LORO, Meyer DI): DI↑ → **RMSE 15.5→27.1cm, coverage 69→51%**; AOA 안(16.9cm/68%) < 밖(21.0cm/62%). ⚠️pseudo-replication이 DI 정규화 깨뜨림 → 참조점 **고유위치 dedup**으로 수정. `alt_aoa_transfer_results.csv`, `maps/alt_aoa_mask.png`.

### 시각화 QA
- scientific-figure-reviewer + visual-reviewer 2에이전트 검토 → 냉색 규약·경도(°W)·주석겹침·음의skill 구분 등 수정. **이관(다음 viz-통합 단계)**: basemap(coastline), dual-axis 분리, fold error-bar, SVG 폰트 감사.

## 2026-07-06 17:20 — 논리 검증 + 문헌 재조사(49편) + 방향 확정(PLAN_FORWARD) + P0 정정

### 검증 결과 (과대표현 정정)
- **"17cm=물리하한" 폐기**: 비가역잡음 ~4cm(분산분해 within 12.3%), 현재 R²≈0.2 = **공변량 정보병목**. pseudo-replication 진단으로 확증(아래).
- **pseudo-replication 발견**: `dl_dataset.csv` 225,421행 → 고유위치 14,348개. 한 위치 내 연도 달라도 **피처 std=0**(정적 climatology), ALT만 SD~14.6cm. 예: 한 셀 2013년 동일피처 ~100점, ALT 34–96cm. → 같은 X→다른 y = 라벨잡음, 모델 평균회귀 강제, apparent floor 생성. CV는 위치그룹핑이라 누설은 없음.
- **"12.97cm=SOTA 돌파" 정정**: skill-over-mean(1−RMSE/자기SD) 전역 10.4% > 평탄툰드라 7.4% → 큐레이션은 설명력을 낮춤. 범위축소 아티팩트. "레짐별 지배 정보원 차이"로 재프레이밍.

### 문헌 재조사
- 워크플로 57에이전트·49편 web-검증, 오픈PDF 36편 `references/0X_*/` 정리 + `references/INDEX.md` + 핵심10편 `references/00_core10/`. 저자 오귀속 3건 적발(Ran2022 공저자·Rahaman↔Chance·Suzuki↔Ieki).
- T1(지점 ALT 예측)=붐빔(Rahaman2025/Luo2022), T2(4D)=열림. 차별성=transfer+UQ+shallow3D(Koven2025 리뷰가 open gap 인증).

### 방향 확정 + P0 착수
- `docs/PLAN_FORWARD.md` 작성: 스레드 A(다중모달 ablation)/B(3D 조건장)/C(진단)/D(T-lite) + 횡단 AOA/UQ. 우선순위=데이터 활용량·규모+기술차별성.
- **P0 정정 반영**: SESSION_HANDOFF·EXPERIMENTS 표현 수정(본 항목), GPT 핸드오프 `gpt/handoff/20260706_1717-lit-review-forecasting-4d-tracks.md`.

## 2026-07-06 11:13 — GitHub 초기화 + cleanup/handoff (토너먼트·floor·큐레이션 아크 정리)

### 이 세션의 작업
- **git 저장소 초기화** (`git init`, main 브랜치). 이전까지 버전관리 없던 상태를 첫 커밋으로 확정.
- **`.gitignore` 작성**: 22GB `data/raw`·대용량 파생(*.parquet/*.npy/dl_dataset*.csv/*_predictions.csv)·모델 체크포인트(*.pt)·저작권 PDF 제외. 코드·문서·작은 결과 CSV·시각화 산출물(그림/지도 28M)은 추적. 스테이징 검증: 157파일 / 28M, 대용량 누출 0.
- **`.claude/project.yaml` 생성**: cleanup/handoff/status 스킬용 프로젝트 설정(gpu_default [4,5], experiment_log, gpt_handoff_dir 등). Polar_Bigdata를 스킬 대상 프로젝트로 등록.
- **문서화**: 이 로그, `SESSION_HANDOFF.md`(롤링 스냅샷), `gpt/handoff/20260706_1113-tournament-floor-curated-demo.md` 생성.

### 이 아크(직전 세션들)에서 확정된 연구 결과 — 커밋되는 상태
| 실험 | 산출 | 핵심 수치 | 근거 CSV |
|---|---|---|---|
| 모델 토너먼트(6종, 6fold+2seed) | GBM≈DL 정보병목 확증, Diffusion 채택 | 앙상블 16.95 ≈ Diffusion 17.09 ≈ GBM 17.24, 전부 부트스트랩 "동률" | `model_tournament_results.csv`, `_significance.csv` |
| point-scale floor 4중 확증 | 17cm=대표성 하한(셀내 SD 11cm) | InSAR +시 17.24→18.79(악화), PolSAR base 38.3/잔차 24(악화) | `insar_ablation_results.csv`, `polsar_residual_results.csv` |
| 정확도-범위 트레이드오프(첫 돌파) | 큐레이션+물리관측이 floor 돌파 | 평탄툰드라 **12.97cm**(SOTA급) → 완만 16.6 → 전역 17.3 | `curated_scope_results.csv` |
| 고정밀 국소 데모 | 북사면 250m ALT 필드 + AoA 마스크 | 3패널(PolSAR/모델/UQ) | `outputs/maps/local_demo_alt_field.png` |
| 시각화 규약 정비 | cmcrameri 냉색 계열 표준화 | oslo/vik/acton/broc | `src/polar/plotstyle.py`, `docs/VISUALIZATION.md` |

### 감사(cleanup Step 2) — ✅ 통과, 블록커 없음
- 헤드라인 수치 전부 CSV 근거와 일치(위 표). 12.97/16.95/17.09/17.24/108.5→87.3/17.24→18.79/38.3→24 교차검증 완료.
- 개념 주의(문서에 반영됨): 연평균 MAGT장 ≠ 계절 최대융해 ALT — "0°C 등온면=ALT" 비교는 오류(과거 1회 범함, 수정됨). SOTA 11-12cm은 좁은·평탄·P-band 직접관측 조건값.

### 재현 메모
- 제외된 대용량 데이터는 `scripts/0_download`·`1_data_prep`로 재생성. 학습셋 조립: `assemble_dl_dataset.py`(→dl_dataset.csv), 물리관측 부착: `insar_ablation.py`/`polsar_residual.py`.
- 큐레이션 실험 재실행: `CUDA_VISIBLE_DEVICES=4,5 python3 scripts/3_deep_learning/curated_local_model.py`.

### 미결(다음 세션)
- 확장 방향 3택 대기: ① 국소 데모 완성도(실측 검증수치+여러 창 일반성) ② 지형 계층 확장(툰드라/산지/삼림 각자 최적화) ③ 3D+전이(CCI 지중온도 1/2/5/10m).
- SoilGrids 다운로드 보류(ISRIC 서버 불가). 서버 복구 시 `scripts/0_download/soilgrids_alaska.py` 재실행.

## 2026-07-14 — 중간보고 PPT 전면 재구성 + 연구방향 검증(데이터규모·학습시간)

### PPT (deck/build_midreport.py v3, 18슬라이드)
- 폰트: Pretendard SemiBold/ExtraBold로 교체(~/.fonts 설치 확인, LibreOffice fontconfig 렌더). 기존 Pretendard 미인식→Noto 대체가 자간 깨짐 원인이었음. report_lib.py 수정.
- 표지: 흰 배경 EMP 톤(청록 accent·룰선), ALT 지도 제거. 줄바꿈 자연화.
- 아키텍처 그림(mk_architecture_fig.py): Digital Rock 논문 톤, 실제 데이터 썸네일(입력)→모델→산출물(출력).
- 3D→2.5D 전환(mk_cross_section.py): PyVista 컷어웨이(SSAA·NaN흰색) 시도 후, 위도-깊이 단면+0°C 등온선+깊이슬라이스 5장의 2.5D로 교체(사용자 지시).
- 그래프 6종 재제작(mk_midreport_figs.py): EMP·Digital Rock 톤(뚜렷한 색·굵은 값라벨·최소격자). bottleneck·sota·era5_transfer·cv_concept·tournament·conformal.
- figure_hero 재설계: 지표 스트립+상세 설명 문단+해석 노트로 밀도↑, 여백·겹침 제거. 페이지 카운터 버그(6/18) 전역카운터로 수정.
- QA: visual-reviewer 2회. 남은 것: 선행연구 이미지/모식도, DL 모델 전용 슬라이드, 실험결과 반영.

### 연구방향 검증 (문헌 20편+ 조사, 파이프라인 실측)
- 학습 데이터 = 6.6MB tabular(14,348셀×36피처), 22GB 아님. 22GB는 전처리(피처증류). 학습시간 초~분(GBM). ALT ML 관행 확인: Gautam2025 68사이트, Ran2022 ~1000점 → 우리 14,348은 상위권. 병목=데이터부피 아닌 라벨희소+공변량정보(분산분해 between 86%).
- 3D = GBM 조건장(vol_thermal_field_alaska.py), 시추공 10,747점 학습, 기후+깊이만 입력. 신경장 폐기(2.2 vs GBM 1.3°C). 0°C 등온면 끊김=GBM 셀독립 예측.
- patch-CNN 기시행: DEM패치+스칼라 17.2 ≈ GBM 17.7(대등). PolSAR7GB·ReSALT7GB SAR는 이미지 아닌 스칼라로만 활용.
- 라벨 지역분포: ALT 94% 알래스카(ABoVE_AK 13,542). 시추공 지중온도는 9개국 260사이트(스위스7741·미국1600·러시아735·서시베리아390 등)로 다지역.
- Stefan+DL 잔차 미실행(PI-LSTM 근거 27~69%↓). 물리 base+DL 보정이 라벨희소·전이열화 처방.

### 데이터 확보
- ALLena(시베리아 레나델타 9,186점), TPDC QTEC(티베트 지온), ds2332(기보유 확인). SMALT=우리 22만점과 동일(중복). FireALT 서버장애·대기.

### 문서
- docs/CONTEST_PLAN_2026.md(v2 두트랙), EXPERIMENT_ROADMAP.md(E1~E7), EXPERIMENT_PLAN_2026-07-14.md(P0~P5·공변량 인벤토리·Q&A). deck/DESIGN_BRIEF_MIDREPORT.md.

### 미결(다음 세션, GPU 6,7,8,9)
- P0: 데이터 인벤토리 세계지도 + 6모델별 ALT 예측·오차 지도(model_tournament_predictions.csv 재료).
- P1: 전 공변량(DEM+InSAR+PolSAR+CCI) + 전 지역(알래스카+시베리아+티베트) 통합 ALT 재학습, 6모델 재비교.
- P2: Stefan 물리 base + DL 잔차. P3: 3D 전공변량+연속성DL. P4: AlphaEarth 임베딩. P5(트랙): 이미지 조건 diffusion/flow.
- 실험 결과는 전문 mapping·시각화 후 PPT 반영.

## 2026-08-31 — 본선 발표덱 최종화(그림 3차 재설계) + 정합 감사 + 발표 문서 3종 + 논문화 전략

### 발표덱 방법부 그림 3차 재설계 (deck/mk_final_figs.py)
- 참조덱(IMAGE_SEUNGWONBAEK, RTM pseudo-MCMC) p3·5·6·8 문법을 분석해 확정 문법 수립. v3(구역 틴트)→v4(외곽 프레임+색 구역)→**v5/v6(평면)** 3차에 걸쳐 재작업. 사용자 반려 사유: 박스 과다·곡선 화살표·번호 틱바가 "AI틱함".
- **확정 문법**: 백색 배경 + 파선 구분선 + 주황 볼드 헤딩 + 텍스트 위계. 박스 허용은 3종뿐(실데이터 이미지 액자, 셰브런(황=학습/녹=평가), 네트워크 연산 블록 #F7E9AE). 표는 북탭스 헤어라인, 판정은 채택(녹 #2F6B33)/제외(적 #96382C) 색 텍스트, **화살표는 수평·수직 직각만**(곡선 rad 금지).
- **금지 목록**(deck_spec_final.json revision_v7~v9 + 메모리 기록): 외곽 프레임, 색 구역 패널, 카드 박스 일반, 판정 태그 박스, 곡선 화살표, 오렌지 번호 틱바, 장거리 파선 우회, 그림 안 pNN 상호참조, 그래프 사후 강조 라벨, "선+주황 라벨+큰 숫자" 통계 카드.
- fig_workflow: 5열 균등 배분 + 행 높이 열 간 정렬(R1~R4)로 화살표 전부 평행. fig_dl_arch: 레인 출력선 전폭 연장, 열2→열3은 "비교" 화살표(입력 아님), 물리열 수직 사슬. fig_aug_design: 2레인 + 직각 배선.
- fig_aug_spatial 재구성: (a)관측·(b)증강 전(21.9)·(c)**증강 전후 변화량 지도**(±30cm) + 공유 컬러바. 전/후 직접 색 비교는 실제 변화 5cm 수준이라 20~120 스케일에서 판독 불가 → 변화량 지도로 대체(원 셀 단위 예측 부재로 스케일 재렌더 불가, 생성 스크립트는 7/24 세션 임시본으로 소실).
- p19·20·21·22 통계 블록을 key-value 헤어라인 표(build_final.py `kv_table`)로 전환, p22 세로 구분선 제거.

### 원본 그림 스크립트 수정·재생성 (수치 불변, 라벨·표기만)
- report_overview_figure/pptx.py: "83개 0.5° 지역"→**"74개 0.5° 블록"**(감사 확정 오류), 개요 캡션 문구.
- report_transfer_summary.py "물리 결합"→"물리 잔차 결합" · s12_figs.py "위성제품"→"위성 제품", 패널 제목을 검증조건 명칭(공변량만/정보 없음)으로 통일 · s4_residual_figs.py "레나"→"레나델타" · s2_physics_figs.py 범례 영문→한글 · s3_aug_curve_figs.py "증강비율"→"증강 비율", 기준선·순가치(+10.2 CatBoost) 주석 추가 · map_data_inventory_world.py InSAR 라벨 리더선 관통 해소.

### 정합 감사 (워크플로 12에이전트, 적대 검증 후 오탐 0)
- 참고문헌 인용 18건 전수 정합. 목록 누락 4건 추가(Gorishniy×2·Prokhorenkova·Riseborough).
- 확정 결함 8건 수정: p4 83→74블록, p6 LORO 세 지역 표기, p13 14.46/14.56 추정량 병기, p15 평가 셀 기준 각주, p5 원자료/집계 구분.
- 초견 이해도 리뷰(3렌즈 29건) + 최종 오탈자/시각 QA(4에이전트 22건) 반영. "석박통합"→"석박사통합", 3D→3차원, 면적평균→면적 평균 등 용어 통일.

### KPDC 콘슬 유도 ALT 적대 재검증 (4에이전트) — 파싱 정당성 확정
- 원자료 독립 재파싱으로 유도값 소수점까지 재현(75.6/121.1/136.4/140.0/153.1, 검열 >80 13건·>160 10건). 깊이 라벨은 여름 단조↓·겨울 역전·심부 진폭 감쇠 3중 물리검사 전부 통과. 원파일 배치 A의 라벨 반전은 실재했으나 s7_parse_kpdc_council.py가 겨울 단조성 rho 검정으로 교정, ID10-14 인터리브 의심은 배제됨.
- **값의 사다리(콘슬)**: 탐침 시즌중반 35.0 → 8월 한정 41.6 → CALM U27 1km 그리드 53–89(최근 88–89) → KOPRI 2016 피트 동결면 58–73 → 코어 하단 104–120(2022) → 심부 16층 시추공 121–153(2024-25). 136.4는 심부 부분표본이지 사이트 대표값 아님(보고서 단서 재확인). ID02 45.3·ID03 61.6은 관측 절단(partial_end) 과소치.
- **주의 2건(미반영, 질의 대응용)**: (1) "탐침 35.0"은 2017-05-30 초여름 측정 20% + CALM U27 1셀이 섞인 셀평균으로 저편향(연최대 기준 41.6이 타당). (2) 지온 프로파일의 KPDC 등록 식별자는 00002707(2024)·00002955(2025) — 덱·보고서는 00002125(코어)만 표기, 병기 검토 필요. 41.6/68.6은 main.tex 미수록이라 슬라이드 미반영.

### 발표 문서 3종 (deck/render/)
- permafrost_final.pptx/pdf 23장 최종본. 발표대본_ALT_ctrl.docx: 덱 정합 + 초견 청중 기준 친절 상세판(용어 최초 정의, 그림 안내 대사, 평이한 비유, 15분 30초 배분). 발표QA대비_ALT_ctrl.docx: 36문항을 **한줄 답/쉽게 설명하면/숫자로/근거** 4단 형식으로 전면 재작성(22→13쪽). 생성기 deck/mk_qa_doc.py + 소스 deck/qa_answers.md를 저장소로 이관(재현성).

### 논문화 가능성 평가 (3에이전트) → 메모리 [[paper-journal-strategy]]
- **판정**: 현 상태 TC/PPP는 reject~major 불투명. 프레이밍 수정 + 저비용 재분석이면 중위권 게재권.
- 살릴 기여 3: C1 유사라벨 순가치 상수대조 설계(선행 미발견) · C2 LORO 구조 정교화 반증(2025 리뷰가 규정한 '전이 미평가' gap 직격) · C3 CQR 커버리지 검증 구간+AOA('UQ 미탑재' gap).
- 치명 약점: W1 앵커 개선 비유의(CI [-2.22,+2.76]) · W2 185조합 winner's curse · W3 전이 지역 2곳 · W5 선행 계보 누락(Read 2019 WRR PGDL, Liu 2023 CRST PI-LSTM).
- **TC 배제 확정**: TC의 ML 게재 사례는 전부 "ML=도구, 빙권 질문=주장" 구조. 방법 조합을 주장으로 낸 tc-2022-9(InSAR+RF 영구동토 매핑)는 리젝 — 우리 프로필과 일치. 공개심사 리젝 기록은 영구 삭제 불가. Uxa et al. 2026(TC)이 이미 ALT RMSE 14.2–18.2cm 게재.
- **결정: 1순위 CRST**(IF 4.9, 구독 경로 APC 0원, Liu 2023 PI-LSTM 선례지) → 2순위 Scientific Reports(IF 4.9, 음성결과 명시 수용, Gautam 2025 동일 지면). 대안 1: Environmental Modelling & Software(IF 5.2, 벤치마크·음성결과·UQ를 본문 기여로 평가, APC $3,400).

### 다음 세션 (P0, 계산 수일)
- 185조합 winner's curse 해소: 중첩 선택 재평가 또는 등가중 앵커(24.11→23.27) 사전 지정을 주 추정치로.
- 대조군 사다리 추가(셔플 Stefan, TDD 선형식)로 "물리 정보 vs 저복잡도" 분리.
- 관련연구 재배치(Read 2019·Jia 2021·Willard 2022·Liu 2023 인용, 차별점을 "공간 표적 증강+순가치 대조+음성 물리"로 좁힘) + 영문화 + 코드·데이터 공개 성명.
- P1(2~6주): 전이 지역 2곳 이상 추가(스칸디나비아·캐나다 군도 CALM)로 CI 재산정 · GIPL2/Obu/Ran head-to-head.

### ✅ 정합 감사 결과와 조치 (2026-08-31, 4축 병렬 감사 → 전건 수정 완료)
**❌→✅ BLOCKER 해소 — s12 그림의 "물리식 단독"이 표 8과 다른 셀 집합이었음**
- `scripts/4_visualization/s12_figs.py:46,85`이 `anchor="stefan"`(전체 셀)을 쓴다. 그림 두 지역 평균 half 23.72 / loro 24.60.
- 반면 main.tex 표 8(tab:s12)·4.7절은 `anchor="stefan_cci_w"`(w_stefan_cci=1.0 → 같은 Stefan 식이나 CCI 유효 셀 한정) 값 **22.91 / 24.11**. 레나델타는 CCI 결측 없어 동일(16.16), 캐나다만 갈림(31.29 vs 29.66).
- main.tex:531 그림 캡션이 "두 패널의 두 지역 평균이 표 8의 두 열이다"라고 명시하므로 **검증 가능한 진술의 직접 위반**. 본문 546행이 "위성 제품 결측 셀 제외, 같은 셀 평가"라고 쓰므로 표가 맞고 그림이 틀림. 편향도 그림 5.51/2.31 vs 본문 5.8/2.6(= stefan_cci_w).
- 원인은 이번 세션 변경이 아니라 기존 anchor 선택. **조치 완료**: s12_figs.py 2곳(46행 series, 85행 (c)패널)을 `anchor="stefan_cci_w"`로 교체 후 그림 재생성 → 검증 결과 half 22.91 / loro 24.11로 표 8과 정확히 일치, 편향도 5.78/2.57(본문 5.8/2.6). 덱 크롭 재생성·리빌드 완료. 부수로 발표덱 p15 면피 각주를 "그림·표 모두 위성 제품이 유효한 같은 셀 집합에서 평가한 값"으로 교체.

**⚠️→✅ 수치 (전건 수정)**
- main.tex 표 8 "기계학습 단독(CatBoost)" 행 27.94/26.61 → 실제 **27.93/26.60**(s12_hybrid_gate.csv 재계산, +0.01 반올림 오류).
- main.tex:588 anomaly 범위 "결정계수 -0.01~+0.00, 상관 0.06~0.10"은 물리식·물리잔차 계열만의 값. s14_annual_results.csv axis=anomaly 22행 실제 범위는 R² -0.7813~+0.0015, 상관 -0.2666~+0.1386. "어느 모형도 지점 평균을 넘지 못한다"는 결론 자체는 전 행 R²≤0.0015로 성립 → 괄호 안 적용 대상 한정 필요.
- `report_overview_figure.py:124` n_grid=892,865 하드코딩이 저장 산출물로 뒷받침되지 않음(main.tex 어디에도 없음). report_alt_map_hires.py가 격자 셀 수를 meta JSON으로 남기게 하고 그 값을 읽도록 변경 권장.
- 발표덱 p20 "구간 폭 약 15–54 cm"(build_final.py:420)는 보정 전 14.7과 보정 후 53.6을 범위처럼 붙인 표기. 셀별 실제 분포는 p2–p98 약 39–76 cm이고 같은 슬라이드 (a) 지도 컬러바(vmin=p2, vmax=p98)와 충돌. main.tex:626은 "14.7에서 53.6"으로 올바름 → 덱만 수정.

**⚠️→✅ 문서·명칭 (전건 수정)**
- **SESSION_HANDOFF.md가 2026-07-27에 정지**(최우선): ①확정 오류 "83개 0.5°지역"(실측 74) 보존 ②존재하지 않는 `docs/report/main.tex`(6쪽)를 주 보고서로 기술(실제 outputs/report/main.tex, 13쪽) ③폐기된 permafrost_summary 10장을 제출덱으로 기술(실제 본선 permafrost_final 23장, 전문에 "본선"·"23장" 0회). README.md:6이 이 파일을 유일 기준으로 지정하므로 다음 세션 오염 경로가 열려 있음.
- main.tex "물리 결합" 9회(474 절제목·493·494 표행·517·520 캡션·351·588 본문) ↔ 그림·덱 "물리 잔차 결합". 그림·덱이 이미 통일됐으므로 **main.tex를 고치는 쪽**. 특히 517행 캡션이 존재하지 않는 막대 이름을 지시.
- main.tex 152행 약어 "(이하 CCI 제품)" ↔ 표·본문 "위성 제품" 혼용(548·555행). 그림·덱은 "위성 제품"으로 통일됨.
- main.tex 표 tab:s12 열머리 "대상 공변량 사용/대상 정보 없음" ↔ tab:settings 정식명 "공변량만/정보 없음"(그림 패널 제목은 이번 세션에 정식명으로 통일됨). fig:s12 캡션 (a)(b) 서술형도 동일.
- main.tex 4.8절 "기후 9종"(= 2.1절 기후 8종의 연별판 + 전겨울 SWE, s14_annual_alt_redesign.py:63-64)이 2.1절 "기후 8종"과 이름 충돌 → 구성 병기 필요.
- `.claude/project.yaml` naming_canonical 낡음: 공변량 "14종=지형6+기후8" → 정본 src/polar/fidelity.py:19-38 기준 **전체 34 / 전이 공통 25 / 지형기후 14 / 광역격자 17**. "물리관측 PolSAR/InSAR 4"도 현행 SAR 8종(InSAR5+PolSAR3)과 불일치. 실험 단계 B0/B0b/B1/B1b/tournament/curated는 main.tex 내 0건 → 현행 S1~S14 체계로 교체 필요.
- `figures/figure_spec.json`: 이번 세션 재생성한 보고서 그림 4종(report_overview, transfer_loro_summary, s12_hybrid_summary, alt_annual_fields) 스펙 미등록. 561행 '증강비율' 구 표기 잔존.
- `scripts/4_visualization/s1_baseline_figs.py:165` 주석에 83개 지역·셀수 3,978 잔존(다른 문서와 불일치).
- `scripts/4_visualization/s12_analyze.py`가 s12_figs.py와 별개로 옛 라벨("위성제품", "설정 B·C") 사용. 산출물(s12_transfer_ranking.png)은 미인용이나 재생성 시 오염 위험.
- deck/ 구버전 빌드 4종(build_deck/report/midreport/summary.py)이 폐기 수치 하드코딩(16.95cm, 56.1→85.9%, 비가역하한 7.16cm 등). build_final.py는 final_lib.py만 import하므로 코드 의존 없음 → **아카이브 보존으로 충분**, 상단에 "재실행 금지" 주석 1행 권장.
- `submission/` 아래 83 잔존은 2026-07-30 예선 제출 동결본이므로 **수정 금지**. 본선 패키지 제작 시 submission/code 복사 대신 현행 scripts/에서 재생성할 것.

**✅ pass 확인분**: 74블록 정정(map_model_gate_meta.json n_blocks=74로 독립 확인) · 캐나다 순가치 +10.2(CSV 시드짝 재계산) · transfer_loro_summary 9개 값 표 7과 완전 일치 · s4_indomain_vs_transfer 수치·캡션 주장 검증 · 신규 CSV 2종(map_model_gate, s14_annual) 모두 본문 인용 근거로 사용 중 · p12 수치 4종 CSV 소수점 재현 · p20 커버리지 93.4%(s11_conformal_results.csv) · 참고문헌 23건 main.tex와 저자·연도·저널·권까지 일치 · main.tex 미수록 수치(41.6·68.6) 덱 소스·렌더 pptx 유입 0건 · 검증조건 3종 명칭 3곳 동일 · main.tex는 이번 세션 미수정(mtime 07-30, diff는 07-29~30 미커밋 잔여분).

**조치 요약(전건 완료)**: s12 앵커 교체(BLOCKER) · main.tex 표8 반올림 27.93/26.60 · anomaly 범위 대상 한정 · "물리 결합"→"물리 잔차 결합" 9건 · "CCI 제품"→"위성 제품" · tab:s12 열머리·fig:s12 캡션을 검증 3조건 정식명으로 · 기후 9종 구성 병기 · 덱 p15 각주 교체·p20 구간 폭 "14.7 → 53.6 cm" · 대본 동일 표현 동기화 · SESSION_HANDOFF 전면 갱신(83→74, outputs/report 경로, 본선덱 23장, P0 로드맵) · project.yaml naming_canonical 갱신(S1~S14 체계, 공변량 34/25/14/17/9) · figure_spec.json·s1_baseline_figs.py 주석·s12_analyze.py 라벨 정리 · deck 구버전 빌드 4종에 아카이브·재실행 금지 배너.
**이월(비차단)**: report_overview_figure.py n_grid=892,865 근거 산출물 부재(→ report_alt_map_hires.py가 meta JSON 기록하도록) · figure_spec.json에 이번 재생성 그림 4종 스펙 미등록 · KPDC 지온 식별자 00002707/00002955 병기 검토. `submission/`(예선 동결본)의 83 잔존은 수정 금지.
