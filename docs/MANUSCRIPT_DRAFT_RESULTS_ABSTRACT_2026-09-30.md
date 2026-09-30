# Manuscript draft: Abstract, Results and Discussion skeleton (2026-09-30)

**작성** 2026-09-30. **과업** `docs/EXECUTION_PLAN_REMAINING_2026-09-30.md` 3.5 의 M2.
**상태** 초안. 결과 수치는 아래 '원천' 의 판정 기록 절에서 옮겼고 다시 계산하거나 해석을 바꾸지 않았다. 원천 절에 없는 값은 `[MISSING: …]`, LGF 결과 자리는 `[RESULT: LGF-…]`, 저자 결정이 필요한 곳은 `[DECISION: …]` 로 표시했다. LGF 창은 2026-10-02 05:01 에 닫힌다.
**원천(결과 정본)** LG 계획서 7.1, 7.1a, 7.2(원고 문장에 주는 결정 1–5 포함), 7.3(주 판정 = 약관 확인분 판), 7.4. LGU 계획서 11.1. WRAPUP '결과 판정 기록(J7)'(AB1–AB10, SC1w–SC3w, L43, AK1w, S-a, S-b)과 1.1 초록 문장 규칙 (a)–(e).
**문장 제약** RESEARCH_FRAME A.1·A.7 의 쓰지 않을 문장, WRAPUP 1.2–1.5·3–5절·11.2, NOVELTY 4절(유지 문장의 조건)과 5절, LGU 10절, LGF 9.2.
**기호와 표시 항목** 기호는 `docs/MANUSCRIPT_DRAFT_METHODS_INTRO_2026-09-30.md`(P0–P3, D0, D1, R0–R3, F1a·F1k·F1n, RM, V1, V2, κ, λ, α)를, 그림 번호와 패널은 `figures/figure_spec.json` 과 `docs/DISPLAY_ITEMS_2026-09-30.md`(Fig 1–7, Table 1)를 따른다. 이 초안은 방법 초안의 `[RESULT: Table 3 result columns …]` 자리를 Fig 7d 표로 채운다.
**열람 상태** 결과 표, 조각, 봉인 폴더, 로그를 열지 않았다. 계획서의 판정 기록 절과 규칙 절만 읽었다.
**표기** Δ(A − B) = RMSE(A) − RMSE(B)(cm)이고 음수가 A 에 유리하다. 대괄호 구간은 셀 가중 95 % CI 다. 판정어는 두 가중(셀 가중, 블록 등가중) 4분 판정(δ 0.5 cm)이다. 재표집 횟수는 LG L1–L8 1,000회(LG 7.1), LG 보조 열·LGX·초록 묶음 10,000회, LGU 1,000회다. 같은 대비의 CI 가 두 원천에서 조금 다르면(재표집 횟수 차이) 가설 판정 문장에는 LG 7.1 값을, 초록 묶음 문장에는 J7 값을 쓴다. 본문 문단의 대괄호 `[가설 id; 원천 절; 그림 패널]` 은 추적용이며 투고 전에 지운다.

---

## Abstract

For active-layer thickness (ALT) in four regions excluded from training, we scored physics-anchored machine learning (ML) against a source-coefficient Stefan model (P0) and its recalibration with n target labels (P1). Ten contrasts of regional means were fixed before results were opened and Holm-adjusted. Without labels, direct ML had larger error than P0 in [k] tested learners (+2.25 cm [1.10, 3.42], gradient boosting). ML with Stefan pseudo-labels had lower error than with shuffled ones (−1.63 cm [−1.98, −1.04]). With 10 labels, P1 had lower error than P0 (−2.45 cm [−3.04, −1.88]). This was significant in one of four regions. Error increased in Canada. Residual ML (R1) showed no established difference from P1. Augmenting R1 with pseudo-labels was equivalent to R1 within 0.5 cm. R1 had lower error than ML with physics-model outputs as inputs (−3.74 cm [−5.23, −2.75]). Hierarchical and constant-width prediction intervals (three regions) showed no established difference in interval score. With all labels, R1 had lower error than P0 (−2.64 cm [−3.48, −1.85]) and a year-matched Stefan model (−1.64 cm). This gain came from one region. It was absent from a three-region mean with Alaska. Against P1, the gain was below 0.5 cm (−0.40 cm [−0.76, −0.22]).

**Word count**: 198 words (`wc -w` on the paragraph above; bracketed intervals count as words; `[k]` counts as one word and is replaced by one number). Sci Rep limit: 200 words, unstructured.

- `[k]` = [RESULT: LGF-F1, LGF-N1] the number of learners whose zero-label D0 − P0 verdict is inferior or equivalent in the four main regions. Recorded so far: 5 learners with inferior verdicts, namely CatBoost with default settings (AB1), CatBoost with 600 iterations and depth 6, random forest and source-tuned CatBoost (L30) and TabPFN v2 (L34 (a), local platform). Add TabICL v2 (LGF-F1) and each tuned neural learner (LGF-N1) only if its verdict is in the first support class of LGF 2.2 ('inferior or equivalent'). A learner in the class 'no evidence of superiority' is not counted.
- AB2 (B:ens − P0, rule (d)) is omitted from the abstract for the word limit. It carries no verdict word elsewhere in the abstract. [DECISION: include AB2 as 'No difference was established between a physics-model ensemble and P0' (10 words) if another sentence is shortened.]
- The sentence 'It was absent from a three-region mean with Alaska' states the auxiliary L8 column of LG 7.1a (three-region mean inferior, +0.77 cm) without a verdict word, because verdict words in the abstract are limited to AB1–AB10 (WRAPUP 1.1). LG 7.2 decision 3 requires this qualifier in the manuscript. [DECISION: keep this wording, or state 'had larger error (+0.77 cm)' if the authors treat the qualifier as part of AB8.]
- Sentence-by-sentence trace: rows A1–A15 of the claim checklist at the end.

---

## Results

### Evaluation overview and data [Table 1; Fig 1]

The main experiment (LG) scored 12 methods in 27 targets over 5 block splits and the label-count grid [LG §1–§3; Fig 1c]. All 117 CPU shards and 231 GPU shards of the main run completed on one platform (cloud, NVIDIA T4) [LG 7.1]. The table of failed fits had no rows [LG 7.1]. The extension experiment (LGX) was aggregated from 4,451 cloud shards (3,860 CPU and 591 GPU) [LG 7.2]. The refitted base methods reproduced the main-run results in 117 of 117 units [LG 7.2]. Splits 4 and 5 of the reduced FT-Transformer-type learner were not completed on the cloud platform [LG 7.2, L26]. The TabPFN v2 grid (LGT) completed 172 of 172 units on the local platform without failures [LG 7.4]. The uncertainty experiment (LGU) completed on the local platform [LGU 11.1]. [RESULT: LGF-F and LGF-N completeness at the window close (units done, failed and partial).]

The label count n is the number of label rows drawn from half A of a target, and a row has a different meaning in different regions [Table 1; WRAPUP 2.1]. Alaska has 13,606 rows at 343 distinct 1 km locations [Table 1; WRAPUP 2.2]. Lena has 3,037 rows at 201 locations, Canada 750 rows at 86 locations, Russia W 31 rows at 29 locations and Russia E 30 rows at 30 locations. Rows in Russia W and Russia E are multi-year means of CALM sites. Rows in Lena, Alaska and Canada are mostly point observations. The full label set is about 15 labels in Russia W and Russia E, several hundred in Canada and about 1,500 in Lena [WRAPUP 2.5]. Comparisons between regions at equal n therefore state the label unit.

Verdicts in the four main regions (Lena, Canada, Russia W and Russia E, mode x) are re-tests in regions used in earlier experiments of this project [WRAPUP 1.2]. Verdicts in the regions of the independent-region experiment (LGD) are independent-region confirmations [WRAPUP 1.2]. Alaska (x) is a reference target and is not averaged with the four main regions. Where recorded, an auxiliary three-region mean of Lena, Canada and Alaska (x) is reported. Among the confirmatory LGX hypotheses, L10, L12 and L29 were blind [LG 7.2]. L15 (n = 0) and L19 contain replication (unblinded) parts, and L30 contains a partly unblinded part [LG 7.2]. LGT results carry the label 'results existed at registration (unopened)' [LG 7.4]. Blinding labels of all hypotheses are listed in Supplementary Table S-R1.

### Error as a function of the label count [Fig 2]

**Direct ML.** In none of the four main regions did D0 have lower error than P0 at any n ≤ 40 under both weightings (0 of 4 regions) [L1; LG 7.1; Fig 2a–d]. In Russia E, D0 had larger error than P0 at every n (+4.3 to +4.5 cm) [L1; Fig 2d]. In Canada at n = 3, the difference was not established [L1; Fig 2b]. There, the cell-weighted interval excluded zero (−2.34 [−3.55, −0.41]) and the block-weighted interval included zero. Hypothesis L1 was supported [L1].

**Augmentation combined with residual learning.** At n = 10, R2 − R1 was −0.13 cm [−0.31, 0.24] in the four-region mean [L2; LG 7.1; Fig 2e]. In the abstract bundle, the same contrast was equivalent within 0.5 cm after Holm adjustment (−0.13 [−0.31, 0.22], adjusted equivalence p 0.002) [AB6; J7; Fig 7d]. At n = 40, R2 had larger error than R1 in Lena and Canada (+0.34 cm), an effect below 0.5 cm [L2; LG 7.1a; Fig 2f]. At n = 160, R2 and R1 were equivalent within 0.5 cm [LG 7.1a; Fig 2f]. The registered outcome for R2 is 'augmentation is for zero labels only' [L2]. With target weights α = 10 and 100, R2 did not have lower error than R1 at any tested n [L18; SI]. With these weights, R2 had larger error than R1 at n = 40 and 160, and some of these effects were below 0.5 cm [L18; SI].

**Stacking.** The stacked method R3 had lower error than R1 at n = 40 (−1.23 [−1.63, −0.50]) and n = 160 (−1.83 [−2.59, −0.40]) in Lena and Canada (partial, 2 of 4 regions) [L2; LG 7.1; SI]. At n = 10, the difference R3 − R1 was not established [L2; LG 7.1a]. The registered verdict for R3 is 'partial' [L2]. [MISSING: R3 − P1* at n = 40 and 160. Without it, no sentence compares R3 with recalibrated physics.]

**Net value of residual learning over recalibration.** At n = 3, the difference R1 − P1 was not established [L4; LG 7.1; Fig 4b]. At n = 10, R1 had lower error than P1 (−0.18 [−0.52, −0.02]; block-weighted −0.29 [−0.52, −0.09]) [L4; LG 7.1; Fig 2e]. This effect is below 0.5 cm [LG 7.1a]. In the abstract bundle, this contrast was significant before correction and not after Holm adjustment (−0.18 [−0.53, −0.01], adjusted p 0.136) [AB5; J7; Fig 7d]. At n = 40, 160 and 320, R1 had lower error than P1 in Lena and Canada (−0.56, −0.64 and −0.83 cm; partial, 2 of 4 regions) [L4; LG 7.1; Fig 2f]. With all labels, R1 had lower error than P1 in the four-region mean (−0.40 [−0.76, −0.23]), an effect below 0.5 cm [L4; LG 7.1, 7.1a; Fig 3c]. By region, R1 had lower error than P1 at every n only in Canada [L4; LG 7.1]. In Russia W and Russia E, the difference was not established at any n. The registered verdict of L4 is 'net value of ML even with sparse labels', with a minimum n of 10 in the four-region mean [L4]. In the three-region mean with Alaska (x), R1 and P1 were equivalent within 0.5 cm at n = 3, 10 and 40 [LG 7.1a]. In that mean, R1 had lower error than P1 from n = 160 [LG 7.1a].

**Stronger recalibrated baselines.** At n = 10, no recalibrated physics baseline had lower error than P1, so the stronger recalibrated baseline P1* equals P1 [L29; LG 7.1a]. At n = 40 and 160, the soil-property Stefan anchor recalibrated with κ = 10 (P1@ed) and the year-matched Stefan anchor recalibrated in the same way (P1@tddm) had lower error than P1 [L29; LG 7.1a]. P1@ed had the lower point estimate and is P1*. Against P1*, the difference R1 − P1* was not established at n = 40 (+1.31 [−0.13, 1.96]) or at n = 160 (+1.30 [0.01, 1.86]) [L29; LG 7.1a]. At n = 40 and 160, we therefore do not state that residual learning improves on recalibrated physics [LG 7.2 decision 2]. [MISSING: P1* at n = 3, at n = 320 and with all labels. LG 7.1a records that R1 − P1@tddm with all labels was not established, but it does not record whether any recalibrated baseline had lower error than P1 with all labels.] In the extended pool with an additional independent region, the L4 result was not maintained [L4e; LG 7.3; section 'Additional independent regions'].

### Use of physics information [Fig 3]

**Pseudo-label controls.** At n = 0, D1 had lower error than D1 with each of the four placebo pseudo-labels [L15; LG 7.2; Fig 3a]. The differences were −1.63 cm for shuffled pseudo-labels, −1.68 cm for a target-level constant and −3.63 cm for a source-level constant. The difference for the linear-TDD placebo was −0.48 cm, an effect below 0.5 cm. At n = 10, D1 had lower error than the shuffled (−1.15 cm), target-level constant (−1.16 cm) and source-level constant (−2.21 cm) placebos [L15; Fig 3a]. At n = 10, D1 and the linear-TDD placebo were equivalent within 0.5 cm (−0.08 cm) [L15]. Seven of eight contrasts showed lower error for D1, and the registered verdict is 'mixed' [L15]. Cell-level physics information contributes to the augmentation gain, and the difference from the linear-TDD placebo is not present at n = 10 [L15; LG 7.2]. In the abstract bundle, D1 − D1@shuffle at n = 0 was −1.63 [−1.98, −1.04] (adjusted p 0.001) [AB3; J7; Fig 7d].

**Residual structure versus physics outputs as inputs.** The residual structure had lower error than the physics-input structure in all four registered contrasts [L10; LG 7.2; Fig 3b]. The contrasts were R0 − F1k at n = 0 (−2.72 [−4.34, −1.73]), R0 − F1a at n = 0 (−2.49 cm), R1 − F1k at n = 10 (−3.74 cm) and R1 − F1n at n = 10 (−2.52 cm). L10 was supported [L10]. In the auxiliary rows, the residual structure also had lower error at n = 3 [LG 7.2 decision 4]. With all labels, R1 had lower error than F1k (−2.93 cm), and R1 − F1n was not established [LG 7.2 decision 4]. At n = 40 and 160 in Lena and Canada, the direction was opposite [LG 7.2 decision 4; Fig 3b]. There, R1 had larger error than F1k (+2.38 [0.52, 3.25] and +2.23 cm) and than F1n (+1.85 [0.73, 2.61] and +1.98 cm) under both weightings. In the three-region mean with Alaska (x), these contrasts were not established at n = 40 and 160 [LG 7.2 decision 4]. The L10 conclusion is therefore restricted to n ≤ 10 and, for F1k, to the full label set [LG 7.2 decision 4]. The auxiliary rows do not change the registered verdict. In the abstract bundle, R1 − F1k at n = 10 was −3.74 [−5.23, −2.75] (adjusted p 0.001) [AB7; J7; Fig 7d].

**Additive versus multiplicative residual.** The multiplicative residual RM had larger error than the additive residual R1 at n = 10 with λ = 0.25 (+0.35 cm, an effect below 0.5 cm) and with λ = 1.0 (+1.09 cm) [L12; LG 7.2; Fig 3b]. With all labels and λ = 0.25, RM had larger error than R1 (+0.25 cm, an effect below 0.5 cm) [L12]. With all labels and λ = 1.0, the difference was not established [L12]. The registered hypothesis of equivalence between RM and R1 was rejected [L12]. The rows with back-transform correction (RMc) are reported in Supplementary Table S-LGX [L12].

**Shuffled target labels.** R1 and R1 with the n target labels permuted among the drawn cells (R1s) were equivalent within 0.5 cm at n = 10 [L17; LG 7.2; Fig 3b]. With all labels, R1 had lower error than R1s (λ = 0.25: −0.60 cm; λ = 1.0: −1.69 cm) [L17]. The registered wording is that the relation between target covariates and residuals contributes with all labels [L17].

**Full label set.** With all labels, R1 had lower error than P0 in the four-region mean (−2.64 [−3.47, −1.90]; block-weighted −2.64 [−3.53, −1.67]) [L8; LG 7.1; Fig 3c]. L8 was supported [L8]. By region, R1 had lower error than P0 in Russia W (−13.98 cm) [L8; Fig 3c]. R1 had larger error than P0 in Canada (+3.38 cm) [L8]. In Lena (−0.78 cm), the difference was not established [L8]. In Russia E (+0.84 cm), only the block-weighted interval excluded zero, on the side of larger error, so the difference was not established under the two-weighting rule [L8]. The improvement of the four-region mean therefore comes from Russia W, where the full label set is about 15 labels [L8; LG 7.1a; WRAPUP 2.5]. In the three-region mean of Lena, Canada and Alaska (x), R1 had larger error than P0 (+0.77 [0.02, 1.28]; block-weighted +1.15 [0.66, 1.63]) [LG 7.1a; Fig 3c]. With all labels, R1 had lower error than P*, the year-matched Stefan model defined under L29 (−1.64 [−2.95, −0.74]; block-weighted −2.23 [−3.17, −1.24]) [L29; LG 7.1a; Fig 3c]. [MISSING: region rows of R1 − P* with all labels.] In the abstract bundle, R1 − P0 with all labels was −2.64 [−3.48, −1.85] (adjusted p 0.001) [AB8; J7; Fig 7d]. R1 − P1 with all labels was −0.40 [−0.76, −0.22] (adjusted p 0.001), an effect below 0.5 cm [AB9; J7; Fig 7d]. Across 50 splits, the share of splits with negative R1 − P0 was at least 0.8 in two of four main regions, so the registered classification is 'region dependent' [L31; LG 7.2; Fig 3c]. [MISSING: the names of the two regions in L31.] No target was flagged as dependent on split composition [L31]. [MISSING: the LGX-N2 error floor (band in Fig 3c), the LGX-N4 region-level inference (Hartung-Knapp interval and sign test) and the WRAPUP 1.6 SD/SE-ratio flags.] Without the region-level inference, no statement about regions in general is made [WRAPUP 11.2].

**Coefficient recalibration and pooling.** At n = 10, P1 had lower error than P0 in the four-region mean (−2.45 [−3.04, −1.88], adjusted p 0.001) [AB4; J7; Fig 3d]. Russia W dominated this mean (−11.96 cm) [AB4]. P1 had larger error than P0 in Canada (+2.26 cm) [AB4]. In Lena and Russia E, the difference was not established [AB4]. The improvement was significant in one of four regions [AB4; J7 note]. AB4 carries a viewing note: pre-result analytic calculations (S-a) had shown values of the same form in some regions [WRAPUP J7]. By the minimum-label-count criterion, P1, P2 and P3 did not reach n* in Lena (up to n = 1,000) or in Canada [LG 7.1; Fig 4a]. In Russia W, P1, P2 and P3 reached n* = 3 [LG 7.1]. [MISSING: Δ of P2 and P3 against P0 by n for Fig 3d.] The covariate-dependent coefficient models V1 and V1r had point estimates above P0 in Canada and CA-3 at n ≥ 40 (+2.9 to +8.1 cm) [L7; LG 7.1; SI]. Some of these rows were below P1, for example Canada V1r at n ≥ 160, but no row met the first condition of L7 (Δ ≤ 0 against P0) [L7]. L7 was rejected [L7]. The log-ratio regression V2 had larger error than P1 (+3.94 cm) and than R1 (+4.12 cm) [L13; LG 7.2; SI]. L13 was not supported [L13].

### Minimum label count [Fig 4]

The minimum label count n* against P0 (cell-random draws, α = 1, reference λ) is shown by target in Fig 4a [LG 7.1 n* summary]. In Lena, R0 reached n* = 40, and P1–P3 did not reach n* up to n = 1,000. In Canada, D0, R0 and R1 (λ = 1.0) reached n* = 3, and P1–P3 did not reach n*. In Russia W, P1–P3, D1 and R1–R3 reached n* = 3. In Russia E and Alaska (x), no method reached n* (Alaska up to n = 1,000). The source-coefficient Stefan model P0 was a stronger baseline than most methods, including the recalibrated model P1 [LG 7.1]. An n* of 3 does not by itself show an effect of the labels, because a method can have lower error than P0 already at n = 0. [MISSING: label net value Δ(n) − Δ(0) for the rows with n* = 3 (Canada D0, R0 and R1; Russia W).] The n* values in Fig 4a are computed against P0 and P1 only [DISPLAY_ITEMS, decision of 13:50]. n* against P* or P1* was not computed.

The minimum n at which R1 had lower error than P1 was 10 in the four-region mean [L4; LG 7.1; Fig 4b]. The additional grid point n = 80 was not run, because its registered condition (a minimum n of 160) did not hold [LG 7.1].

Among the four targets eligible for L3, nested selection of the target weight α gave n* ≤ 40 for R0 in two targets: Lena (n* = 40) and Canada (n* = 10) [L3; LG 7.1; Fig 4c]. AL-2 reached n* = 1,000, and AL-5 did not reach n*. L3 was supported under its registered criterion [L3]. In none of the four targets did nested selection give a smaller n* than α = 1 [LG 7.1, auxiliary column]. In Canada, n* was already 3 at α = 1. The support of L3 is therefore not evidence that a larger α reduced the number of labels needed [LG 7.1]. The auxiliary verdict for R1 was rejected [L3].

[MISSING: counts of targets by four-way verdict for R1 − P0, P1 − P0 and R1 − P1 at each n, separately for independent and sub-regional targets (module `n1_target_summary`, Fig 4d).] Sub-regional targets are reported as counts, and pooled means over sub-regions are given without intervals in the main text [WRAPUP 1.3].

### Additional independent regions [SI; Fig 1a]

No permission record exists for the licence-unverified sources, so the primary LGD verdicts use the licence-verified version [LG 7.3]. In this version, pool PE1 adds the shallow region Russia_C to the four main regions (five regions), and pool PE2 adds the deep region Tibet (six regions) [LG 7.3]. The full version, which also contains the shallow region NAtlantic, is reported in the Supplementary Information with the label 'includes data before licence confirmation' [LG 7.3]. PE1 and PE2 rows carry the label 'cross-environment', because the new regions were fitted on the local platform and the four main regions on the cloud platform [LG 7.3]. Check (i) did not pass for CatBoost [LG 7.3]. The reproduction check (c)2 passed for the main range and did not pass for CatBoost in the extended range, whose status was not assigned [LG 7.3].

**Direct ML.** At n ≤ 40, D0 had lower error than P0 in no region of PE1 (0 of 5) [L1e; LG 7.3]. In PE2, D0 had lower error than P0 in one region (Tibet), which is within the registered threshold of ⌊6/4⌋ = 1 region [L1e]. L1e was supported in P4, PE1 and PE2. The four-region verdict of L1 was maintained in the extended pool (six regions) [L1e; LG 7.3].

**Full label set.** With all labels, R1 had lower error than P0 in PE1 (−3.19 [−4.19, −2.22]; block-weighted [−3.71, −1.53]) [L8e; LG 7.3]. R1 had lower error than P0 in PE2 (−26.78 [−29.09, −24.21]), and Tibet dominates this mean [L8e]. The four-region verdict of L8 was maintained in the extended pool (six regions) [L8e]. [MISSING: the PE2 mean in units of each region's P0 RMSE and the 'scale dependent' label (LG 6B.5).]

**Net value over recalibration.** The four-region verdict of L4 was not maintained in the extended pool [L4e; LG 7.3]. In PE1, R1 − P1 was not established at n = 3 and 10 (5 of 5 regions in the pool) [L4e]. R1 had lower error than P1 at n = 40, 160 and 320 in Lena and Canada (partial, 2 of 5 regions) and with all labels (5 of 5 regions) [L4e]. At n = 10, the four-way verdict was superiority in P4 and equivalence within 0.5 cm in PE1 [LG 7.3]. In the mean over the new regions, R1 − P1 at n = 10 was +0.06 [−0.40, 0.51], and the difference was not established [LG 7.3]. With all labels, R1 had lower error than P1 in this mean (−1.40 [−2.95, −0.19]) [LG 7.3]. [MISSING: which regions enter the 'mean over the new regions' in the licence-verified version.] In PE2, the minimum n was 3, and this result comes from Tibet [L4e]. The L4 sentence 'net value of ML even with sparse labels' therefore applies to the four main regions only [LG 7.3].

**Region-level replication.** In Russia_C (shallow), D0 had lower error than P0 at no n ≤ 40, so criterion (a) of L38 was met [L38; LG 7.3]. In Russia_C, R1 − P0 with all labels was not established, so criterion (b) was not met [L38]. Replication of the full-label result was therefore not confirmed in Russia_C [L38; WRAPUP 11.2]. In Tibet (deep), D0 had lower error than P0 at some n ≤ 40, so criterion (a) was not met [L38]. In Tibet, R1 had lower error than P0 with all labels, so criterion (b) was met [L38]. In the four main regions, criterion (a) held in 4 of 4 regions and criterion (b) in 1 of 4 [L38]. All Tibetan target cells lie above the elevation range of the source cells, and the large P0 error in Tibet matches the prior expectation recorded before the run [LG 7.3; LG 6B.5; Methods]. The Tibetan contrasts are not interpreted as gains of ML. The label definition in Tibet (direct versus temperature-derived labels) did not change the classification (robust) [L39; SI]. The extended versions of Russia W, Russia E, Canada and Russia E without Kytalyk were all weakened (cross-environment, auxiliary) [L40; SI]. Under label-set variants, Tibet was robust for variants (b) and (d)–(g), and Russia_C was robust for variants (c), (d), (f) and (g) and weakened for (b) and (e) [L41; SI]. Under the change of source composition, Tibet was robust and Russia_C weakened [L42; SI]. The independent-region evidence of the primary version therefore rests on one shallow region (Russia_C) and one deep region (Tibet).

### Label placement and proximity [Fig 5]

A placement effect was not established [L43; J7; Fig 5a]. For P1 at n = 10, the difference between block-spread and cell-random draws was not established in the four-region mean (Holm p 1) [L43]. At n = 40, it was not established in Lena and Canada (partial, 2 of 4 regions) [L43]. [MISSING: Δ_L43 and its two-stage intervals by n.]

Component (a) of the proximity test (D0, random versus block design) was −4.51 cm [L23; LG 7.2; Fig 5b]. [MISSING: interval and four-way verdict of L23 (a).] In component (b), the double difference had lower error for D0 (−2.28 cm) and was not established for R1 [L23; Fig 5b]. Component (c), in-block draws versus draws from half A at n = 10, was equivalent within 0.5 cm [L23]. The registered wording for (b) is that the gain under random splits contains a proximity component that does not depend on the label count [L23]. A distance dependence of the residual net value was not confirmed [L24; LG 7.2; Fig 5c]. The recalibration gain did not depend on the distance to the labels [L24].

### Zero-label transfer and prediction intervals [Fig 6]

**Direct ML at n = 0.** At n = 0, D0 had larger error than P0 in the four-region mean (+2.25 [1.10, 3.42], adjusted p 0.023) [AB1; J7; Fig 6a]. With higher-capacity learners, D0 − P0 at n = 0 was +1.94 cm for CatBoost with 600 iterations and depth 6, +2.63 cm for a random forest and +1.59 cm for CatBoost tuned by leave-one-source-region-out validation [L30; LG 7.2; Fig 6a]. All three showed larger error than P0 [L30]. L30 was supported: increasing capacity or tuning by source cross-validation did not make direct ML surpass the physics model in zero-label transfer [L30]. Direct ML with process-model outputs as additional inputs (F1k) had larger error than P0 (+2.62 cm) [L11; LG 7.2; Fig 6a]. With TabPFN v2 on the local platform, D0 had larger error than P0 (+1.02 cm) [L34 (a); LG 7.4; Fig 6a]. CatBoost fitted on the same context also had larger error than P0 (+2.64 cm) [L34 (a)]. [RESULT: LGF-F1, TabICL v2 direct prediction at n = 0 with the 10,000-row context and with the full source set, D0 − P0 and its support class.] [RESULT: LGF-N1, direct prediction at n = 0 of the four source-tuned neural learners (MLP, multi-head MLP, reduced FT-Transformer-type model, RealMLP), D0 − P0 by learner and support class.] With TabPFN v2, the anchored residual R0 and P0 were indistinguishable within the 0.5 cm margin (−0.31 cm) [L34 (b); LG 7.4; SI].

**Physics baselines at n = 0.** At n = 0, the Stefan model with year-matched thawing degree-days and the source coefficient (P0@tddm) had lower error than P0 (−1.00 [−1.35, −0.32]; block-weighted −0.41 [−0.74, −0.08]) [L29; LG 7.2; Fig 6a]. This baseline is P* [L29]. The other baselines were not established against P0 or had larger error than P0 [L29]. They were a soil degree-day Stefan model, a Kudryavtsev-type model, a soil-property Stefan model, the CCI ALT product, an affine Stefan model, an affine CCI model and their ensemble, each with and without scale calibration. In the three-region mean with Alaska (x), P0@tddm − P0 was not established [L29]. For the ensemble B:ens, the four-region difference was −2.73 [−3.33, −0.91], and the four-way verdict was 'not established' (adjusted p 1.0) [AB2; J7; Fig 6a]. The zero-label physics reference is reported as both P0 and P* [LG 7.2 decision 1]. [MISSING: D0 − P* and R0 − P* at n = 0.]

**Prediction intervals without labels.** Generative quantile intervals calibrated by pooling exchangeable source cells (λ = 1.0) under-covered in the four main regions [LGU-B1; LGU 11.1; SI]. Their 90 % coverage was 0.757 [0.672, 0.826] for the normalizing flow and 0.653 [0.564, 0.764] for conditional flow matching. Both values were below 0.85 under both weightings, and LGU-B1 was supported [LGU-B1]. For the hierarchical conformal interval at n = 0, the interval score of the flow normalizer minus its permuted placebo was +3.29 cm [−4.17, 8.16] (block-weighted +0.03 [−6.49, 5.52]) [LGU-B2; Fig 6c]. LGU-B2 was not determined, and we do not state that conditional interval width transfers between regions [LGU-B2; LGU §10]. No normalizer differed from the constant-width normalizer by an established amount, with or without matching of the mean log width [LGU-B4; SI]. The width-allocation gain was therefore not established. The flow normalizer minus the CatBoost quantile normalizer was −7.41 cm [−52.55, 10.00], and equivalence could not be tested [LGU-B3; SI]. [MISSING: coverage and mean width by normalizer for Fig 6b.]

**Prediction intervals with labels.** For the hierarchical stage (iii) minus the calibrated constant-width baseline B4, the interval-score difference was +9.51 cm [−0.75, 22.11] at n = 10 and −2.80 cm [−12.50, 7.89] at n = 40 in the pool of Lena, Canada and Alaska (x) [LGU-A1; LGU 11.1; Fig 6c]. Equivalence could not be tested at either n, because the interval half-widths exceeded the margin of 5 % of the baseline scores (99.0 and 105.5 cm) [LGU-A1]. The 90 % coverage condition was met (0.90) [LGU-A1]. LGU-A1 was not determined [LGU-A1]. A difference in interval score between the hierarchical-model interval and the calibrated constant-width interval was not established [LGU-A1]. The variance components are reported in Supplementary Table S-LGU. In the abstract bundle, AB10 (n = 10) had adjusted p 0.277 [AB10; J7; Fig 7d]. The median prediction of stage (iii) had larger RMSE than P1 at n = 3 (+1.43 [0.68, 1.81]) and at n = 10 (+1.60 [0.57, 1.88]) [LGU-A3; SI].

In the label grid, the R1 90 % interval had coverage between 0.85 and 0.95 in 4 of the 6 cells formed by Lena, Canada and Alaska (x) at n = 40 and 160 [L25; LG 7.2; Fig 6d]. In all 4 of these cells, it was narrower than the R0 interval at n = 0 (c1 = 4, c2 = 4) [L25]. [DECISION: the registered L25 sentence 'labels narrow the interval' conflicts with LGU §10, which lists this sentence as not to be written because width reduction follows from the model structure. This draft uses the descriptive sentences above only.]

**Spatial diagnostics and map.** The effective range of the residuals (log ratio) was 2.8 km in Alaska (37.6 km in the probe layer), 7.9 km in Lena and 8.3 km in Canada [LGU-C1; LGU 11.1; SI]. The median distance from scoring cells to the nearest label was 81.0 km in Alaska, 25.8 km in Lena and 97.0 km in Canada [LGU-C1]. These diagnostics have no confidence intervals and are not evidence that spatial correlation is absent [LGU §10]. The residual-kriging baseline was not run, because no region had an effective range longer than the median scoring distance [LGU 11.1]. Because LGU-B2 was not determined, the Lena map uses the constant normalizer, so interval width is proportional to the predicted ALT [LGU 11.1; WRAPUP 6.6]. The map is in the Supplementary Information, and Fig 6e is not used [LGU 11.1; DISPLAY_ITEMS §2].

### Learners and validation design [SI]

**Learners on the cloud platform.** For R1, the five-split comparison of each learner with CatBoost (L26) gave the following results [L26; LG 7.2; SI]. For the MLP and the multi-head MLP (named `tabm` in the code), a difference from CatBoost was not established. Conditional flow matching had larger error than CatBoost at n = 0 with λ = 1.0. The normalizing flow had larger error than CatBoost at the three tested n with λ = 1.0. The denoising diffusion model, RealMLP and ridge regression had larger error than CatBoost [L26]. [MISSING: n and λ of the L26 rows for the diffusion model, RealMLP and ridge regression.] For the reduced FT-Transformer-type learner, L26 could not be determined because splits 4 and 5 were missing, so its three-split LG contrast is reported [L26; WRAPUP 1.5]. Under the three-split contrast, this learner had larger error than CatBoost in two rows and lower error in none (3 splits) [L5; LG 7.1a]. [MISSING: n and pool of the two FT-Transformer-type rows.] Its pooled contrast at n = 0 (+0.36 cm) was inferior with 1,000 replicates and not established with 10,000 replicates, so this row is labelled resampling-dependent [LG 7.1, 7.1a]. In the four-region pooled contrasts of L5 and L26, no learner had lower error than CatBoost [L5; L26; LG 7.4 note]. In the auxiliary four-way table that includes target rows, conditional flow matching had lower error than CatBoost in one row [LG 7.1a]. [MISSING: target and n of this row.] Shards of the FT-Transformer-type learner continued on the local platform do not enter these verdicts [LG 7.2, L26]. [RESULT: J9 auxiliary verdict of L26 for the FT-Transformer-type learner, cross-environment label.]

**Tabular foundation models on the local platform.** For R1 with TabPFN v2 and with CatBoost on the same context, a difference was not established [L32; LG 7.4; SI]. With λ = 0.25 the two were equivalent within 0.5 cm at every tested n (−0.10 to −0.30 cm) [L32]. With λ = 1.0 the difference was not established at any n (−0.53 to −1.60 cm) [L32]. The verdict is partial at n = 40 and 160 (2 of 4 regions), and it carries the label 'results existed at registration (unopened)' [L32; WRAPUP 1.5]. With TabPFN v2, R1 had lower error than P1 at n = 3 (−0.32 cm) and n = 10 (−0.42 cm) in the four-region mean, both effects below 0.5 cm [L33; LG 7.4]. R1 with TabPFN v2 also had lower error than P1 at n = 40 (−0.62 cm), 160 (−0.93 cm) and 320 (−1.10 cm) in Lena and Canada, and with all labels (−0.80 cm) [L33]. The registered verdict is 'net value of TabPFN residuals even with sparse labels', with a minimum n of 3 (four regions) and 40 (two regions) [L33]. LGT did not compute contrasts against P1*, so at n = 40 and 160 these results are stated against P1 only [LG 7.4]. [MISSING: P1* at n = 3.] With all labels, R1 with TabPFN v2 had lower error than P0 (−3.03 cm) [L35; LG 7.4]. R1 with CatBoost on the same context (−2.76 cm) and direct TabPFN v2 (−3.63 cm) also had lower error than P0 [L35]. [MISSING: LGT contrasts against P* with all labels.] At n = 40 and 160, direct TabPFN v2 had lower error than TabPFN v2 with anchor and residual (−1.79 and −2.45 cm; partial, 2 of 4 regions) [L36; LG 7.4]. At n = 0 and 10, direct TabPFN v2 had larger error (+1.33 and +0.97 cm) [L36]. With all labels, this difference was not established (−0.60 cm) [L36]. Most context variants were robust, and one variant weakened R1 − P1 at n = 40 [L37; SI].

[RESULT: LGF-F2, R1 with TabICL v2 versus CatBoost on the same context.] [RESULT: LGF-F3, TabICL v2 versus TabPFN v2.] [RESULT: LGF-F4, net value of TabICL residuals against P1 and the full-label contrast against P0.] [RESULT: LGF-F5, TabICL residual at n = 0.] [RESULT: LGF-F6, effect of the context limit for TabICL.] [RESULT: LGF-N2, tuned versus default neural learners.] [RESULT: LGF-N3, tuned neural learners versus CatBoost.] [RESULT: LGF-N4, selection table (descriptive).] LGF results are reported in their own tables and are not merged into the L5 and L26 tables [LGF 2.2].

**Validation design.** On the pooled data of five regions, the RMSE of D0 (CatBoost, default settings) increased from random 5-fold cells (22.91 cm) to 0.5° blocks (27.17 cm), to leave-one-cluster-out with a 100 km buffer (30.97 cm) and with a 500 km buffer (31.13 cm) [L20; LG 7.2; SI]. Under region holdout, the RMSE of D0 was 33.25 cm [L20]. The difference between the 500 km design and random cells was +8.22 [5.65, 10.38] cm, and L20 was supported [L20]. From random cells to region holdout, the RMSE of D0 increased by +10.34 [7.29, 12.84] cm (five regions) [L19; LG 7.2]. L19 was not supported, and only the size of this degradation is reported [L19; LG 7.2 decision 5]. The degradation of the Stefan-anchored residual model (RS) minus that of D0 was −8.69 [−10.71, −6.15] cm with λ = 0.25 [L21; LG 7.2]. With λ = 1.0, this difference was not established (−1.12 cm) [L21].

### Deployment scenarios and abstract contrasts [Fig 7]

**Few-label recipe (SC1w).** For the deployment recipe R1 at n = 3 and 10, no region-by-n cell had larger error than P0 [SC1w; J7; Fig 7b]. An error increase within 0.5 cm was confirmed in only 4 of 10 cells: Lena at n = 3, Canada at n = 3 and Russia W at n = 3 and 10 [SC1w]. It was not confirmed for Lena and Canada at n = 10, or for Russia E and Alaska (x) at n = 3 and 10 [SC1w]. With a 1.0 cm margin, 5 of 10 cells were non-inferior [SC1w]. The registered verdict is 'safety not confirmed (non-inferior cells 4/10)' [SC1w].

**Recalibration alone (SC1w-P).** Recalibration alone (P1) increased error relative to P0 in Canada at n = 3 (+0.97 [0.26, 1.30], Holm p 0.021) and at n = 10 (+2.26 [0.71, 2.93], Holm p 0.008) [SC1w-P; J7; Fig 7b]. SC1w-P was rejected [SC1w-P]. This row carries the S-a viewing note [WRAPUP J7].

**Residual learning at 40 labels (SC2w).** At n = 40, R1 had lower error than P1 in Lena and Canada [SC2w; J7; Fig 7b]. In Alaska (x), the difference was not established [SC2w]. The result at n = 160 was the same [SC2w]. Under registered sentence A, residual learning at 40 labels reduced error relative to the κ = 10 shrinkage recalibration (P1) in 2 of 3 regions [SC2w]. Against the stronger recalibrated baseline P1* (P1@ed), the difference was not established at n = 40 and 160 in Lena and Canada [L29; LG 7.1a; J7].

**Sequential stopping rule (SC3w).** At the main threshold τ = 0.05, the sequential stopping rule was inoperative, with a label ratio of 0.91 relative to always using 40 labels [SC3w; J7; Fig 7c]. Its error difference is reported descriptively only [SC3w]. [MISSING: Δ_SC3 and the distribution of stopping stages.] In the descriptive τ sensitivity analysis, τ = 0.1 failed condition (i) [SC3w]. τ = 0.2 met all four conditions with a label ratio of 0.19, but it is not the main verdict [SC3w]. The rule is not included in the deployment guidance [WRAPUP 3.5].

**Label unit and label year.** Under point labels, the few-label recalibration gain was smaller than under 1 km location means (Canada, n = 10, median difference −0.86 cm) [S-a; J7; SI]. The label unit in the deployment guidance is therefore the 1 km location mean [S-a; WRAPUP 2.4]. With single-year labels in Russia W and Russia E, the recalibration gain with 3–10 labels was maintained [S-b; J7; SI].

**Abstract contrast bundle (Fig 7d; fills the result columns of Methods Table 3).** Four-region stratified means unless stated. AB1–AB9 use 10,000 replicates and AB10 uses 1,000 replicates (LGU). Holm adjustment over the ten contrasts, family size 10 [WRAPUP 1.1; J7].

| id | Contrast | n | Δ (cm) [cell-weighted CI] | Four-way verdict | Holm-adjusted p | Abstract rule |
|---|---|---|---|---|---|---|
| AB1 | D0 − P0 | 0 | +2.25 [1.10, 3.42] | inferior | 0.023 | (a) direction |
| AB2 | B:ens − P0 | 0 | −2.73 [−3.33, −0.91] | not established | 1.0 | (d) no difference established |
| AB3 | D1 − D1@shuffle | 0 | −1.63 [−1.98, −1.04] | superior | 0.001 | (a) direction |
| AB4 | P1 − P0 | 10 | −2.45 [−3.04, −1.88] | superior | 0.001 | (a) direction, mean with region count (1 of 4), Canada +2.26 inferior; S-a viewing note |
| AB5 | R1 − P1 | 10 | −0.18 [−0.53, −0.01] | superior | 0.136 | (b) abstract: no difference established; body: significant before correction |
| AB6 | R2 − R1 | 10 | −0.13 [−0.31, 0.22] | equivalent | equivalence 0.002 | (c) equivalent within 0.5 cm |
| AB7 | R1 − F1k | 10 | −3.74 [−5.23, −2.75] | superior | 0.001 | (a) direction; opposite direction at n = 40 and 160 stated in the body |
| AB8 | R1 − P0 | all | −2.64 [−3.48, −1.85] | superior | 0.001 | (a) direction with R1 − P* (−1.64) and the Russia W dominance (−13.98; Canada +3.38 inferior) |
| AB9 | R1 − P1 | all | −0.40 [−0.76, −0.22] | superior | 0.001 | (a) direction, effect below 0.5 cm |
| AB10 | Interval score, stage (iii) − B4 (Lena, Canada, Alaska (x)) | 10 | +9.51 [−0.75, 22.11] | not established | 0.277 | (d) no difference established |

The interval of AB10 is taken from LGU-A1 (LGU 11.1, 1,000 replicates); J7 records its point estimate, verdict and adjusted p. [MISSING: block-weighted intervals of AB1–AB10 and the unadjusted p values (`lgw_bundle.csv`, not quoted in J7).] [MISSING: margin-dependence labels of the relative margin δ_rel (WRAPUP 9.4), pending the h39 rerun after the LGF window.]

### Sensitivity analyses and the Alaskan tests [SI]

**Label definition, anchors and hyperparameters.** Under the probe-only label variant, results were weakened or dependent [L28; LG 7.2; SI]. The conclusions are therefore restricted to source conditions that include GPR-derived labels [L28]. Replacing the anchor gave no gain [L28]. With κ and λ selected by leaving one target out, the results differed from those with the fixed values by more than 0.5 cm [L28]. [MISSING: L28 rows by variant and contrast.] The sign of R0 − P0 in Canada matched that of CatBoost in 4 of 8 learners, so the Canadian R0 result is specific to CatBoost [L27; LG 7.2]. The augmentation effect depended on the pseudo-label ratio, and the convergence condition was not met [L16; LG 7.2]. The reference ratio r = 10 was not changed [L16]. Adding E0·s as an input column (F1a) and D0 were equivalent within 0.5 cm at n = 0, n = 10 and with all labels (|Δ| ≤ 0.15 cm) [L9; LG 7.2]. The direct-ML input variants of L14 were equivalent (D0t) or not established at n = 0 (D0c, D0m) [L14; LG 7.2].

**Within-region validation and the Alaskan tests.** In within-region block cross-validation, R1w had lower error than P1w in Lena (−0.56 cm) and Canada (−1.27 cm) [L22; LG 7.2; SI]. In Alaska, the difference was not established (−0.46 cm; replication, unblinded) [L22]. With λ = 0.5 and 1.0 and with nested selection of λ, R1w had lower error than P1w only in Canada [L22]. L6 could not be determined [L6; LG 7.1]. No combination of method and target weight had lower error than P0 in a majority of the Alaskan sub-regions used as simulated new regions with Alaskan cells in the source (mode i) [L6]. AK1w was supported [AK1w; J7]. Among combinations that had lower error than P0 in Alaskan sub-regions used as simulated new regions with sources inside Alaska, a majority also had lower error than P0 in the same sub-regions when only sources outside Alaska were used (r = 0.57, 68 pairs, 5 targets) [AK1w]. The two sensitivity variants also supported AK1w [AK1w]. At the target level, the majority held in 2 of 5 targets [AK1w]. In mode i, 60.7–77.3 % of the source cells are Alaskan [WRAPUP 5]. L6, AK1w and L22 are reported in one Supplementary Table [WRAPUP 5; DISPLAY_ITEMS §5].

---

## Discussion (skeleton)

**Status.** Skeleton only. Each slot lists the recorded evidence that the paragraph may use. No claim is added beyond the Results.

### D1. Summary of the registered results (slot)

- Zero labels: AB1, L1, L30, L11 and L34 (a) show larger error for direct ML than for P0. [RESULT: LGF-F1, LGF-N1.] A stronger zero-label physics baseline exists (P* = P0@tddm, L29). AB2 is not established.
- Few labels: the mean gain of recalibration at n = 10 (AB4) comes from one region, and Canada has larger error (AB4, SC1w-P). The net value of residual learning at n = 10 is not established after Holm adjustment (AB5), is below 0.5 cm (L4) and is not maintained in the extended pool (L4e).
- Physics information: residual structure has lower error than physics-input structure at n ≤ 10 (L10, AB7), with the opposite direction at n = 40 and 160. The augmentation gain is mixed against placebos (L15, AB3). Augmentation added to R1 is equivalent within 0.5 cm (AB6) and is registered as 'for zero labels only' (L2).
- Full labels: R1 has lower error than P0 and P* in the four-region mean (AB8, L29), the gain comes from Russia W, and the three-region mean with Alaska has larger error (LG 7.1a). Against P1 the effect is below 0.5 cm (AB9). The L8 verdict is maintained in the extended pool (L8e), and replication in Russia_C is not confirmed (L38).
- Uncertainty: both confirmatory hypotheses are not determined (LGU-A1, LGU-B2), and generative quantile intervals under-cover (LGU-B1).

### D2. Deployment table (slot; text table or content of Fig 7a, adds no display item)

| Stage | Labels (unit) | Recorded evidence | Sentence the manuscript may write | Not written |
|---|---|---|---|---|
| T0 | n = 0 | AB1 +2.25 (inferior); L30, L11, L34 (a) larger error; [RESULT: LGF-F1, LGF-N1]; L29: P* = P0@tddm lower error than P0 (−1.00); AB2 not established; LGU-B2 not determined; LGU-B1 coverage 0.757 and 0.653 | Use the Stefan model with the source coefficient and report P0 and P*. Direct ML without a physics anchor had larger error than P0 in the tested learners. Intervals are reported with empirical coverage; with the constant normalizer, width is proportional to the predicted ALT. | Coverage-guaranteed interval or map. Conditional width transfers between regions. Tabular foundation models as a class do not surpass physics. |
| T3 | n = 3 (rows; guidance unit: 1 km location mean, S-a) | SC1w: R1 non-inferior in Lena, Canada and Russia W; not confirmed in Russia E and Alaska (x). SC1w-P: P1 larger error in Canada (+0.97 [0.26, 1.30]). L4: R1 − P1 not established | Recalibration alone increased error in Canada. Non-inferiority of R1 was confirmed in 3 of 5 regions. | Recalibration with 3–10 labels is safe. The recipe did not increase error. |
| T10 | n = 10 | AB4 −2.45 (one of four regions; Canada +2.26); AB5 rule (b); SC1w: non-inferior only in Russia W at n = 10; SC1w-P: Canada +2.26 [0.71, 2.93]; S-b: single-year labels keep the gain (Russia W, E); L43 not established | The mean recalibration gain came from one region. After correction, a gain of residual learning over recalibration was not established. | As T3. Where to measure. |
| T40 | n = 40 (Lena, Canada, Alaska (x)) | SC2w sentence A (2 of 3 regions against P1); R1 − P1* not established; L10 auxiliary: F1k and F1n lower error than R1 at n = 40 and 160; L2: R3 lower error than R1 (2 regions); L36 (TabPFN, local); SC3w inoperative | Residual learning reduced error relative to the κ = 10 recalibration in 2 of 3 regions. Against P1@ed, a difference was not established. | ML exceeds recalibrated physics at n = 40 or 160. |
| All labels | about 15 (Russia W, E) to several hundred or more (Canada, Lena) | AB8 −2.64 and R1 − P* −1.64; Russia W −13.98, Canada +3.38; three-region mean with Alaska +0.77; AB9 −0.40; L8e maintained; L38 Russia_C (b) not confirmed; L31 region dependent | R1 had lower error than P0 and P* in the four-region mean. The gain came from Russia W. The three-region mean with Alaska had larger error. | Unconditional 'ML exceeds the Stefan model'. Statements about regions in general. |
| Stopping and placement | n/a | SC3w inoperative (label ratio 0.91); L43 not established | No stopping rule is recommended. No placement effect was established. | Observation priority. Where to measure. |

[DECISION: WRAPUP 4 allows only method-level placement guidance ('spread labels over several blocks rather than one'). With L43 not established, decide whether this sentence stays, and if so, cite it as a design choice rather than a result.]

### D3. The Stefan coefficient E (pointer)

- Paragraphs D1–D8 are drafted in `docs/MANUSCRIPT_DRAFT_SUPPORT_2026-09-30.md` M3.2.
- Fill-ins available from the recorded sections: D6 `[RESULT: L7 verdict]` = rejected (LG 7.1). D6 `[RESULT: L13 verdict]` = not supported (LG 7.2). D7 `[RESULT: AB4 …]` = superior, adjusted p 0.001, rule (a), Russia W −11.96 cm, Canada +2.26 cm inferior. D7 `[RESULT: AB5, AB9, L4 …]` = AB5 rule (b); AB9 rule (a) with an effect below 0.5 cm; L4 minimum n 10 in four regions, not maintained in PE1 (L4e).
- D3 (terms absorbed by E) may cite L29: at n = 0, the year-matched Stefan model had lower error than P0 in the four main regions (−1.00 [−1.35, −0.32]) and not in the three-region mean with Alaska. [DECISION: whether D3 links this to the year-mismatch term; the data do not separate the terms.]
- D8 is unchanged: E0 is dominated by Alaskan cells (78.0–94.3 % of source cells for the four main targets, Table M3-1).

### D4. Relation to previous work (pointer)

- Positioning follows NOVELTY sections 1 and 3: the contribution is the evaluation design and its registered results, including negative results. No methodological novelty is claimed.
- The retained statements of NOVELTY section 4 need their numbers replaced by recorded values: N6 by AB1, L30 and L15/AB3; N7 by L7 and L13; N5 by AB4 region rows, SC1w-P and L43 (the placement clause stays deleted); N4 by L4, AB5, AB9, L8, P* and L4e; N9 by LGU-B1, LGU-A1, LGU-B2 and L25. [DECISION: earlier-protocol values in NOVELTY section 4 (for example N6 +2.3 cm, N4 −0.68 cm, N9 coverage 0.86) move to the SI register of past results and are not cited in the main text.]
- The word 'curve' can now be used for the label-count curve (NOVELTY N1 condition).

### D5. Limitations (slots)

1. Four main regions, all reused from earlier experiments; two of them have 30–31 labelled rows (Russia W, Russia E). Pooled intervals are conditional on these regions. [MISSING: LGX-N4 region-level inference.]
2. Independent-region evidence in the primary (licence-verified) version rests on one shallow region (Russia_C) and one deep region (Tibet). Tibetan target cells lie above the source elevation range. PE1 and PE2 carry the cross-environment label, and check (i) did not pass for CatBoost. NAtlantic is only in the SI full version.
3. The source coefficient E0 is dominated by Alaskan cells (78.0–94.3 %; MANUSCRIPT_DRAFT_SUPPORT M3 D6).
4. Label definitions: conclusions are restricted to source conditions that include GPR-derived labels (L28). Label years (1990–2024) are not matched to the covariate period (2015–2020), and maps are static.
5. Stronger baselines exist at n = 0 (P*) and at n = 40 and 160 (P1*). n* is computed against P0 and P1 only. P1* at n = 3, 320 and with all labels is not recorded.
6. The equivalence margins (0.5 and 1.0 cm) are conventions fixed before results (WRAPUP 9).
7. Uncertainty: two to four calibration groups, no finite-sample coverage guarantee, under-coverage of generative quantile intervals, both confirmatory LGU hypotheses not determined, map in the SI with width proportional to the prediction.
8. Deployment: few-label safety not confirmed (SC1w), recalibration alone increased error in Canada (SC1w-P), stopping rule inoperative (SC3w), placement effect not established (L43).
9. Learners: the five-split verdict of the reduced FT-Transformer-type learner is not determined; `tabm` is a multi-head MLP and `ftt` a reduced model, not the published architectures (LGF 9.2). [RESULT: LGF limitations after the window.]
10. Registration is internal (git commits, not time-stamped by a third party). Four amendments were committed after the first retrieval, before bundles were unpacked. Several hypotheses contain unblinded parts.
11. Process-model baseline: the CryoGrid-based CCI ALT product enters L29; no site-scale GIPL2 or CryoGrid runs were made (WRAPUP 11.1). Other ALT products were not compared, because their training data may overlap the scoring sites (WRAPUP 10).
12. Sources without confirmed licences are excluded from the primary LGD version and are not redistributed.
13. Russia C and Greenland are reported as point estimates only.

### D6. Sentences not written (internal checklist, not manuscript text)

| Sentence | Source of the rule |
|---|---|
| Unconditional 'ML beats (or exceeds) the Stefan model'; 'ML exceeds physics with sparse labels' | NOVELTY 5; RESEARCH_FRAME A.7 |
| 'ML exceeds recalibrated physics' at n = 40 or 160 | L29; LG 7.2 decision 2; J7 SC2w |
| 'ML exceeds physics' where it improves on P0 but not on P* | L29 |
| 'Recalibration with 3–10 labels is safe'; 'the recipe did not increase error' without non-inferiority | WRAPUP 3.3, 11.2; J7 |
| Validation-design result described as a change of ranking or sign | L19; LG 7.2 decision 5 |
| 'Labels narrow the interval' | LGU 10 (conflicts with L25 wording, see conflict 1) |
| 'Conditional width transfers between regions'; 'the hierarchical model improves intervals' | LGU 10, 11.1 |
| 'Coverage-guaranteed interval or map' | LGU 10; WRAPUP 11.2 |
| 'Generative models reduce point error'; 'diffusion or flow is better than quantile regression'; 'generative and quantile learners are equivalent' | LGU 10 |
| 'There is no spatial correlation' | LGU 10 |
| 'Exceeds within Alaska'; 'a procedure selected in Alaska transfers to new regions' | WRAPUP 5, 11.2 |
| 'Where to measure'; 'observation priority' | WRAPUP 4, 11.2 |
| Region comparison at equal n without label unit; '100 m support labels' | WRAPUP 2.1, 2.5, 11.2 |
| 'AOA mask shows error risk' | WRAPUP 6.8, 11.2 |
| Statements about regions in general without the Hartung-Knapp interval excluding zero | WRAPUP 7.2 (a)2, 11.2 |
| 'Replicated in new regions' unless L38 (a) and (b) both hold in all new regions | WRAPUP 11.2 |
| 'On par' or 'equivalent' because an interval includes zero | WRAPUP 1.5, 11.2 |
| 'Significant' for pooled sub-regional means or for one weighting alone | WRAPUP 1.2, 1.3 |
| Verdict words in the abstract for contrasts outside AB1–AB10 | WRAPUP 1.1 |
| 'Did not exceed' or 'indistinguishable' in L1–L8 result sentences (allowed as hypothesis names and in L19, L30, L34, LGF-F1, LGF-N1 wording) | WRAPUP 1.4 (a) |
| 'Tabular foundation models do not surpass physics' as a class; 'TabICL does not surpass physics' without both contexts; 'networks were sufficiently tuned'; 'TabM' or 'FT-Transformer' as the published models | LGF 9.2 |
| First-ever claims (first physics-ML combination, first region holdout, first label-count evaluation, first physics-derived labels, first two-baseline format) | NOVELTY 5 |
| '72–95 % of the few-label gain is recalibration'; 'the order of ML and physics changes with the validation design' | NOVELTY 5 (N3, N8) |
| Title words 'beat the Stefan equation', 'how many', 'value-of-information' | RESEARCH_FRAME A.1 |

---

## Claim checklist

Every abstract sentence and every Results claim group, with its recorded source and the wording rule applied. 'J7' is the WRAPUP section '결과 판정 기록(J7)'. 'Rule 1.1 (a)–(e)' are the abstract rules of WRAPUP 1.1. 'Rule 1.4 (a)' is the four-sentence rule for LG L1–L8 result sentences.

| ID | Location | Claim (short) | Recorded source | Rule applied |
|---|---|---|---|---|
| A1 | Abstract s1 | Design: held-out regions, P0 and P1 with n target labels | Methods draft; LG §2–§3 | Design statement, no verdict word |
| A2 | Abstract s2 | Ten contrasts fixed before results were opened, Holm-adjusted | WRAPUP 1.1; J7 | No verdict word |
| A3 | Abstract s3 | AB1: D0 larger error than P0 at n = 0, +2.25 [1.10, 3.42]; [k] tested learners | J7 AB1 (inferior, Holm p 0.023); LG 7.2 L30; LG 7.4 L34 (a) | Rule 1.1 (a); learner qualifier without verdict word (1.1 exclusion list); only LGF 2.2 first support class counts in k |
| A4 | Abstract s4 | AB3: Stefan pseudo-labels lower error than shuffled, −1.63 [−1.98, −1.04] | J7 AB3 (superior, 0.001) | Rule 1.1 (a) |
| A5 | Abstract s5 | AB4: P1 lower error than P0 at n = 10, −2.45 [−3.04, −1.88] | J7 AB4 (superior, 0.001) | Rule 1.1 (a) |
| A6 | Abstract s6–s7 | AB4 significant in one of four regions; error increased in Canada | J7 AB4 row (Canada +2.26 inferior) and note below the J7 bundle table | J7 note: mean with region count |
| A7 | Abstract s8 | AB5: no established difference between R1 and P1 | J7 AB5 (superior, Holm p 0.136) | Rule 1.1 (b); body states 'significant before correction' |
| A8 | Abstract s9 | AB6: augmenting R1 equivalent within 0.5 cm | J7 AB6 (equivalent, adjusted equivalence p 0.002) | Rule 1.1 (c) |
| A9 | Abstract s10 | AB7: R1 lower error than physics-input ML, −3.74 [−5.23, −2.75] | J7 AB7 (superior, 0.001) | Rule 1.1 (a); opposite direction at n = 40, 160 stated in the body (J7; LG 7.2 decision 4) |
| A10 | Abstract s11 | AB10: no established difference in interval score (three regions) | J7 AB10 (not established, 0.277); LGU 11.1 LGU-A1 | Rule 1.1 (d); pool stated |
| A11 | Abstract s12 | AB8: R1 lower error than P0 (−2.64 [−3.48, −1.85]) and than P* (−1.64) with all labels | J7 AB8 and note; LG 7.1a (R1 − P* superior) | Rule 1.1 (a); J7 note requires P* and regional skew |
| A12 | Abstract s13 | The gain came from one region | LG 7.1 L8 (only Russia W significant); J7 AB8 row | J7 note (region count) |
| A13 | Abstract s14 | Absent from a three-region mean with Alaska | LG 7.1a (three-region mean inferior, +0.77); LG 7.2 decision 3 | Qualifier required by decision 3; written without a verdict word (rule 1.1) [DECISION] |
| A14 | Abstract s15 | AB9: gain against P1 below 0.5 cm, −0.40 [−0.76, −0.22] | J7 AB9 (superior, 0.001) | Rule 1.1 (a); effect-size label |
| A15 | Abstract (omitted) | AB2 not in the abstract | J7 AB2 (not established, 1.0) | Omission for word limit; no verdict word used [DECISION] |
| R0.1 | Overview | Run completeness of LG, LGX, LGT and LGU | LG 7.1, 7.2, 7.4; LGU 11.1 | Facts as recorded |
| R0.2 | Overview | Label rows and 1 km locations by region; unit stated for comparisons | WRAPUP 2.1, 2.2, 2.5 (data description, not results) | WRAPUP 2.5 |
| R0.3 | Overview | Re-test of reused regions versus independent-region confirmation; blinding labels | WRAPUP 1.2; LG 7.2, 7.4 | WRAPUP 1.2 |
| R2.1 | Fig 2 | L1: 0 of 4 regions; Russia E larger error at every n; Canada n = 3 not established; L1 supported | LG 7.1 L1 | Rule 1.4 (a); two-weighting significance; verdict name after the result |
| R2.2 | Fig 2 | L2 R2 − R1 at n = 10, 40, 160; AB6 equivalence; 'augmentation is for zero labels only' | LG 7.1 L2; LG 7.1a; J7 AB6 | Rule 1.4 (a); effect-size label; separate values for L2 (1,000) and AB6 (10,000) |
| R2.3 | Fig 2, SI | L18: no superiority with α = 10, 100; larger error at n = 40, 160 | LG 7.2 L18 | LGX registered wording |
| R2.4 | SI | R3 lower error than R1 at n = 40, 160 (partial); n = 10 not established | LG 7.1 L2; LG 7.1a | Rule 1.4 (a); partial label; no comparison with P1* [MISSING] |
| R2.5 | Fig 2, Fig 4b | L4 by n; AB5 before and after correction; region rows; three-region mean with Alaska | LG 7.1 L4; LG 7.1a; J7 AB5 | Rule 1.4 (a); rule 1.1 (b) body wording; effect-size label |
| R2.6 | Fig 2 | P1* = P1 at n = 10; R1 − P1* not established at n = 40, 160; no 'improves on recalibrated physics' at these n | LG 7.1a; LG 7.2 decision 2 | L29 registered rule |
| R2.7 | Fig 2 | L4 not maintained in the extended pool | LG 7.3 L4e | LG 6B.5 conclusion wording |
| R3.1 | Fig 3a | L15 values at n = 0 and 10; 'mixed'; AB3 | LG 7.2 L15; J7 AB3 | LGX 'mixed' wording, described by n |
| R3.2 | Fig 3b | L10 four contrasts; auxiliary n = 3 and all labels; opposite direction at n = 40, 160 in the same paragraph; restriction of the conclusion; AB7 | LG 7.2 L10 and decision 4; J7 AB7 | Decision 4 (same paragraph, verdict unchanged) |
| R3.3 | Fig 3b | L12: RM larger error than R1; equivalence rejected; RMc rows in SI | LG 7.2 L12 | LGX wording; effect-size label |
| R3.4 | Fig 3b | L17: equivalent at n = 10; R1 lower error with all labels | LG 7.2 L17 | LGX registered wording |
| R3.5 | Fig 3c | L8 mean and region rows; Russia W source; Russia E one-weighting note; three-region mean with Alaska; R1 − P*; AB8; AB9; L31 | LG 7.1 L8; LG 7.1a; LG 7.2 L29, L31, decision 3; J7 AB8, AB9 | Rule 1.4 (a); two-weighting significance (WRAPUP 1.2); decision 3 (both qualifiers); no region-general statement (WRAPUP 11.2) |
| R3.6 | Fig 3d | AB4 mean and regions; S-a viewing note; n* of P1–P3; L7 rejected; L13 not supported | J7 AB4; LG 7.1 n* summary and L7; LG 7.2 L13 | J7 note; S-a viewing note (J7 registration label) |
| R4.1 | Fig 4a | n* by region; P0 stronger baseline than most methods; n* only against P0 and P1 | LG 7.1 n* summary; DISPLAY_ITEMS decision of 13:50 | Label net value required [MISSING] |
| R4.2 | Fig 4b | L4 minimum n = 10; n = 80 not run | LG 7.1 | Facts as recorded |
| R4.3 | Fig 4c | L3 supported; nested α reduced n* in 0 of 4; not evidence for α; R1 auxiliary rejected | LG 7.1 L3 and interpretive note | LG 7.1 note |
| R4.4 | Fig 4d | Target-level verdict counts | [MISSING] | WRAPUP 1.3 (counts for sub-regions) |
| R5.1 | SI | Licence-verified version primary; pools PE1, PE2; cross-environment label; check (i) and (c)2 status | LG 7.3 | LG revision 15 (m); WRAPUP 1.2 |
| R5.2 | SI | L1e supported; maintained in the extended pool (six regions) | LG 7.3 L1e | LG 6B.5 conclusion wording |
| R5.3 | SI | L8e values; maintained | LG 7.3 L8e | LG 6B.5 conclusion wording |
| R5.4 | SI | L4e not maintained; PE1 rows; new-region mean; PE2 from Tibet; L4 sentence limited to four regions | LG 7.3 L4e and manuscript sentence | LG 6B.5 ('two verdicts together, no region-general sentence') |
| R5.5 | SI | L38: Russia_C (a) met, (b) not met; Tibet (a) not met, (b) met; replication not confirmed in Russia_C; P4 reference 4/4 and 1/4 | LG 7.3 L38 | WRAPUP 11.2 ('replication was not confirmed') |
| R5.6 | SI | Tibetan contrasts not interpreted as ML gains; match the prior expectation | LG 7.3 interpretation; LG 6B.5; Methods draft | See conflict 5 [DECISION] |
| R5.7 | SI | L39 robust; L40 weakened; L41 and L42 by variant | LG 7.3 | As recorded; cross-environment (auxiliary) for L40 |
| R6.1 | Fig 5a | L43: placement effect not established | J7 L43 | WRAPUP 4 sentence; method-level guidance only |
| R6.2 | Fig 5b | L23 components (a)–(c); proximity sentence for (b) | LG 7.2 L23 | LGX registered wording for (b) |
| R6.3 | Fig 5c | L24: distance dependence not confirmed; recalibration gain independent of distance | LG 7.2 L24 | LGX registered wording |
| R7.1 | Fig 6a | AB1; L30 values and sentence; L11; L34 (a); LGF-F1 and LGF-N1 placeholders; L34 (b) | J7 AB1; LG 7.2 L30, L11; LG 7.4 L34 | Rule 1.4 (a) for the L1 part; registered wording for L30 ('did not surpass') and L34 (b) ('indistinguishable within 0.5 cm') |
| R7.2 | Fig 6a | L29: P* = P0@tddm lower error than P0; other baselines; three-region mean; AB2 | LG 7.2 L29 and decision 1; J7 AB2 | L29 rule; rule 1.1 (d) |
| R7.3 | SI, Fig 6c | LGU-B1 under-coverage; LGU-B2 not determined; LGU-B3, B4 | LGU 11.1 | LGU §10 (no transfer of width, no coverage guarantee) |
| R7.4 | Fig 6c | LGU-A1 values and registered sentence; AB10; LGU-A3 | LGU 11.1; J7 AB10 | LGU-A1 registered sentence |
| R7.5 | Fig 6d | L25 described without the registered sentence | LG 7.2 L25 | [DECISION] conflict 1 |
| R7.6 | SI | LGU-C1 ranges and distances; residual kriging not run; map in SI with constant normalizer | LGU 11.1 | LGU §10 ('no spatial correlation' not written) |
| R8.1 | SI | L26 by learner; FT-Transformer-type three-split verdict; no learner lower error in pooled contrasts; one conditional-flow-matching target row | LG 7.2 L26; LG 7.1 L5; LG 7.1a; LG 7.4 note | WRAPUP 1.5 (per learner, split count stated; 'on par' not used); see conflict 3 |
| R8.2 | SI | L32 difference not established; registration label | LG 7.4 L32 | Registered wording; WRAPUP 1.5 label |
| R8.3 | SI | L33, L35, L36, L37; no P1* or P* contrasts in LGT | LG 7.4 | LG 7.4 note on L29 |
| R8.4 | SI | LGF-F2 to F6, N2 to N4 | Pending (LGF §10 empty) | LGF 2.2, 9.2 |
| R8.5 | SI | L20 ladder; L19 degradation size only; L21 | LG 7.2 L19, L20, L21 and decision 5 | L19 rule (degradation size only) |
| R9.1 | Fig 7b | SC1w: 4 of 10 cells non-inferior; unconfirmed cells named; 5 of 10 with 1.0 cm | J7 SC1w | WRAPUP 3.3 'safety not confirmed' sentence; WRAPUP 11.2 |
| R9.2 | Fig 7b | SC1w-P: P1 increased error in Canada at n = 3 and 10 | J7 SC1w-P | WRAPUP 3.3 rejection sentence; S-a viewing note |
| R9.3 | Fig 7b | SC2w sentence A with the P1* qualification | J7 SC2w; LG 7.1a | WRAPUP 3.4 sentence A; L29 application in J7 |
| R9.4 | Fig 7c | SC3w inoperative (0.91); τ sensitivity descriptive; rule not in guidance | J7 SC3w | WRAPUP 3.5 'inoperative' sentence |
| R9.5 | SI | S-a: point labels give smaller gain (Canada n = 10, −0.86 cm); guidance unit 1 km location mean. S-b: single-year labels keep the gain | J7 S-a, S-b | WRAPUP 2.4 interpretation rules |
| R9.6 | Fig 7d | Bundle table AB1–AB10 | J7; LGU 11.1 (AB10 interval) | WRAPUP 1.1 |
| R10.1 | SI | L28: restriction to source conditions with GPR-derived labels; anchors; κ and λ | LG 7.2 L28 | L28 restriction |
| R10.2 | SI | L27, L16, L9, L14 | LG 7.2 | As recorded |
| R10.3 | SI | L22; L6 not determined; AK1w supported with target-level 2 of 5 | LG 7.2 L22; LG 7.1 L6; J7 AK1w | WRAPUP 5 wording ('simulated new region'; no transfer statement) |
| D2 | Discussion | Deployment table | Rows R2.5–R9.5 above | WRAPUP 3.3–3.5, 4, 11.2; J7 conclusion for frame (2) |

---

## Placeholders

**LGF results (window closes 2026-10-02 05:01)**

- `[k]` in the abstract: number of learners counted for AB1 (see the note under the abstract).
- `[RESULT: LGF-F1, LGF-N1]` (abstract note, D1, D2).
- `[RESULT: LGF-F and LGF-N completeness at the window close …]` (Results overview).
- `[RESULT: LGF-F1, TabICL v2 direct prediction at n = 0 …]`, `[RESULT: LGF-N1, direct prediction at n = 0 of the four source-tuned neural learners …]` (Fig 6a paragraph).
- `[RESULT: LGF-F2 …]`, `[RESULT: LGF-F3 …]`, `[RESULT: LGF-F4 …]`, `[RESULT: LGF-F5 …]`, `[RESULT: LGF-F6 …]`, `[RESULT: LGF-N2 …]`, `[RESULT: LGF-N3 …]`, `[RESULT: LGF-N4 …]` (learner paragraph).
- `[RESULT: J9 auxiliary verdict of L26 for the FT-Transformer-type learner …]` (learner paragraph).
- `[RESULT: LGF limitations after the window]` (D5 item 9).

**Missing from the recorded sections**

1. R3 − P1* at n = 40 and 160.
2. P1* at n = 3, at n = 320 and with all labels (LG 7.1a records P1* only for n = 10, 40, 160).
3. Region rows of R1 − P* with all labels.
4. Names of the two regions of L31.
5. LGX-N2 error floor (Fig 3c band), LGX-N4 region-level inference, WRAPUP 1.6 SD/SE-ratio flags.
6. Δ of P2 and P3 against P0 by n (Fig 3d).
7. Label net value Δ(n) − Δ(0) for the n* = 3 rows (Fig 4a).
8. Target-level four-way verdict counts (Fig 4d, `n1_target_summary`).
9. PE2 mean in relative units and the 'scale dependent' label.
10. Composition of the 'mean over the new regions' in LG 7.3 (licence-verified version).
11. Δ_L43 and its two-stage intervals.
12. Interval and verdict of L23 (a).
13. D0 − P* and R0 − P* at n = 0.
14. Coverage and mean width by normalizer (Fig 6b).
15. n and λ of the L26 rows for the diffusion model, RealMLP and ridge regression.
16. n and pool of the two FT-Transformer-type inferior rows; target and n of the conditional-flow-matching superior row.
17. P1* at n = 3 for the LGT sentence (same gap as item 2); LGT contrasts against P* with all labels.
18. Δ_SC3 and the distribution of stopping stages.
19. Block-weighted intervals and unadjusted p of AB1–AB10.
20. δ_rel margin-dependence labels (after the h39 rerun).
21. L28 rows by variant and contrast.

**Decisions for the authors**

1. AB2 in the abstract (word limit).
2. Wording of the three-region Alaska qualifier in the abstract (A13).
3. L25 sentence versus LGU §10 (conflict 1).
4. Placement sentence of WRAPUP 4 with L43 not established (D2).
5. Link of L29 to the year-mismatch term in Discussion D3.
6. Earlier-protocol values of NOVELTY section 4 moved to the SI.
7. Tibet interpretation (conflict 5).

---

## 기록 절 사이의 충돌과 공백

판정은 바꾸지 않았다. 아래는 초안을 쓰면서 확인한 원천 사이의 어긋남이며, 초안이 택한 처리를 함께 적는다.

1. **L25 문장 대 LGU 10절.** LGX L25 의 등록 문장은 c2 ≥ 4 일 때 '라벨이 구간을 좁힌다' 이고 기록된 판정도 이 문장이다(LG 7.2, c1 = 4, c2 = 4). LGU 10절은 같은 문장을 쓰지 않을 문장으로 둔다(폭 감소는 모형 구조에서 자동으로 생긴다). 초안은 커버리지와 폭 비교만 서술했다. 저자 결정이 필요하다.
2. **AB10 의 단위와 재표집 횟수.** `docs/DISPLAY_ITEMS_2026-09-30.md` 8절은 Fig 7d 의 AB10 을 '단위 %, 재표집 2,000회' 로 적었다. LGU 11.1 과 J7 은 구간 점수 차를 cm(+9.51)로, 재표집은 1,000회로 기록했다(WRAPUP 개정 2 (b)의 정정). 그림 캡션은 cm 와 1,000회로 고쳐야 한다.
3. **학습기 문장.** LG 7.4 는 본문 학습기 문장을 'CatBoost 를 넘는 학습기는 없었다' 로 정했다. LG 7.1a 의 L5 4분 보조 열(대상 행 포함)에는 cfm 우세 1행이 있고, WRAPUP 1.5 는 학습기마다 판정 문장을 쓰고 한 문장으로 묶지 말라고 정했다. 초안은 '주 4지역 층화 평균 대비에서' 로 한정하고, 학습기별 문장과 cfm 1행을 함께 적었다.
4. **P1* 결정의 공백.** L29 는 '라벨 n개 조건은 P1 에 대해 같은 절차를 쓴다' 고 정했으나 LG 7.1a 는 n = 10, 40, 160 의 P1* 만 적었다. n = 3, 320, 전량의 P1* 여부가 없다. 전량에서는 R1 − P1@tddm 이 미결정으로 기록되어 있어, AB9·L4 전량·LGT L33(n = 3)의 P1 대비 문장이 P1* 확인 없이 남는다. 초안은 P1 대비로만 적고 [MISSING] 으로 표시했다.
5. **티베트 해석.** 방법 초안은 'Contrasts against P0 in Tibet are … not interpreted' 로 적었고, LG 7.3 해석 문단은 '원천 계수 물리식의 오차가 커서 D0 와 R1 이 P0 를 크게 넘는다(사전 예상과 같다)' 로 적었다. 초안은 사전 예상과 일치한다는 사실만 쓰고 ML 이득으로 해석하지 않았다.
6. **한 가중만의 '유의'.** LG 7.1 L8 행의 '러시아 E +0.84(ns, 블록 등가중은 유의하게 나쁨)' 은 한 가중만으로 '유의' 를 쓴다. WRAPUP 1.2 와 RESEARCH_FRAME 5.1 (ii)는 두 가중 CI 가 모두 0 을 제외할 때만 '유의' 를 쓰게 한다. 초안은 '블록 등가중 구간만 0 을 제외했고 두 가중 규칙으로는 차이를 확인하지 못했다' 로 적었다.
7. **L4 표현.** LG 7.2 결정 2 는 'R1 은 P1 을 작게 넘는다' 로 적었다. WRAPUP 1.4 (a)는 L1–L8 결과 문장을 네 형식('오차가 작다' 등)으로 제한한다. 판정 내용은 같고 표현만 다르다. 초안은 'lower error' 로 썼다.
8. **tddlin 위약의 이름.** LG 6A.3 의 정의는 (E_n/E0)·(a + b·TDD)(TDD 에 선형)이고, LG 7.2 L15 행은 '√TDD 선형 변환 위약' 으로 적었다. 초안은 'linear-TDD placebo' 로 썼다. 정의 대조가 필요하다.
9. **재표집 횟수에 따른 CI 차이.** 같은 대비의 CI 가 LG 7.1(1,000회)과 J7(10,000회)에서 조금 다르다: L4 n = 10 [−0.52, −0.02] 대 AB5 [−0.53, −0.01], L8 [−3.47, −1.90] 대 AB8 [−3.48, −1.85], L4 전량 [−0.76, −0.23] 대 AB9 [−0.76, −0.22], L2 n = 10 [−0.31, 0.24] 대 AB6 [−0.31, 0.22]. 판정은 같다. FT-T 의 n = 0 L5 행만 판정이 갈린다(7.1 열세, 7.1a 미결정. '재표집 의존' 표지).
10. **AB4 의 지역 수.** J7 AB4 행은 러시아 W 의 판정어를 직접 적지 않았다(러시아 W −11.96 지배, 캐나다 열세, 레나·러시아 E 미결정). J7 표 아래 주의 예시('개선이 유의한 지역 1/4')에 따라 1/4 = 러시아 W 로 읽었다. 칸 단위 표로 확인이 필요하다.
11. **L1e·L8e 결론 문구의 지역 수.** LG 7.3 결론 문구는 '확장 풀(지역 6개)' 이다. 약관 확인분 판의 PE1 은 5지역, PE2 는 6지역이므로 N 은 PE2 기준으로 읽었다.
