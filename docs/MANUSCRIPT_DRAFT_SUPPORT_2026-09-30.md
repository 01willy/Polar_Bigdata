# Manuscript drafting support: Stefan coefficient, SI registers and equivalence margin (2026-09-30)

**Purpose**: Draft text and table frames for tasks M3, M4 and M5 of `docs/EXECUTION_PLAN_REMAINING_2026-09-30.md` (section 3.5; inputs in section 2.4). M3 is a Discussion paragraph set on the Stefan coefficient E and a Methods description of the source coefficient E0. M4 is a set of SI table frames (registered hypotheses, past negative results, selection and withdrawal history). M5 is Methods and SI text on the 0.5 cm equivalence margin and on the pre-result calculations S-a, S-b and SC3w.
**Status**: Drafting aid. This is not a registration document. It does not add, remove or reword any hypothesis, verdict rule, scope or confirmatory set. Text that depends on a registered result carries a `[RESULT: ...]` placeholder. Items that need a check against a source before use carry `[VERIFY: ...]`. Literature citations carry `[CITE: ...]` and were not verified in this session.
**Written**: 2026-09-30, completed at 12:09 KST.
**Viewing state**: While writing this document, no curve, test, minimum-n, target, pool or shard file of LG, LGX, LGT, LGU, LGF or LGD was opened, and nothing under `results/rescale_*` or `data/processed/lgw/sealed/` was opened. The document was built from the plan documents (LG, LGU, LGF, WRAPUP, NEXT, NOVELTY, RESEARCH_FRAME, WRAPUP_REVIEW, AUDIT_2026-09-26, the execution plan), the code (`scripts/3_deep_learning/h40_label_grid.py`, `scripts/2_evaluation/h39_scenarios.py`, `src/polar/h4_common.py`, `src/polar/m1_core.py`, `scripts/1_data_prep/era5land_covariates.py`) and `git log`. Pre-result values quoted in M5 are copied from the text of `docs/EXPERIMENT_PLAN_WRAPUP_2026-09-30.md` (results addendum 1 and its fix 1).
**Computed here**: The number of source cells per region for each main target (Table M3-1), from `data/processed/fidelity_base_v3.csv` through `polar.m1_core.load_base`, using only the finite-label mask, coordinates and region codes (single thread, nice 10, under 1 min). E0 values were not computed and are not quoted.
**Conventions**: Times are KST (UTC+9). Commit times are committer times from `git log`; two amended commits differ from their author times by 5 s (`028900a`) and 12 s (`ff1be02`). "Registered" means the first commit that contains the hypothesis text. T_res is the reference time of the first result retrieval, 2026-09-30 07:20:22 KST (WRAPUP 0.3, LG revision 15 (l)).

---

## M3. Physical interpretation of the Stefan coefficient E

### M3.1 Construction of E0 and E_n in this study (Methods text)

**Stefan scaling.** For each labelled cell i, the Stefan approximation is written as ALT_i ≈ E·s_i, with s_i = √TDD_i. TDD_i is the thawing degree-day sum (°C·d) from the ERA5-Land 2 m air temperature monthly climatology for 2015–2020 at the nearest grid cell. It is the sum over months of the positive monthly mean temperature multiplied by the number of days in the month (`scripts/1_data_prep/era5land_covariates.py`, column `e5_sqrt_tdd`). The coefficient E therefore has units of cm (°C·d)^−0.5.

**Label table.** The label table holds 17,467 cells with a direct ALT label (source class `F4_direct` in `fidelity_base_v3.csv`). Alaska contributes 13,606 of these cells (77.9 %), Lena 3,037, Canada 750, Russia W 31, Russia E 30, Russia C 7, Greenland 3, and three other regions one cell each.

**Source pool.** For a target t, the source pool contains every labelled cell outside t, after removal of every cell within 100 km (great-circle distance) of any target cell (`h40.Data.source_idx`). For a sub-region evaluated in mode x, all cells of its parent macro-region are also removed. In mode i, only the sub-region itself and its 100 km buffer are removed. The source pool depends on the target and the mode only.

**Source coefficient E0.** E0 is the least-squares slope through the origin over the source pool, E0 = Σ s_i·y_i / Σ s_i² (`h40.ls_E`). This estimator equals the s²-weighted mean of the cell ratios y_i/s_i. Every cell has the same weight regardless of its region, so regions with many cells dominate E0. All 17,467 cells have a finite, positive s_i, so no source cell is dropped by the positivity condition. E0 does not depend on the split, the label count n, the draw, the seed or the learner. P0 predicts E0·s for every scored cell. The value of E0 for each unit is stored in the `E0` field of the unit metadata; it is not quoted here.

**Table M3-1. Source pool composition for the targets in mode x** (computed in this document; cell counts only)

| Target (mode) | Target cells | Source cells | Removed by 100 km buffer | Alaskan cells in source | Alaskan share | Other main contributors |
|---|---|---|---|---|---|---|
| Lena (x) | 3,037 | 14,429 | 1 (Russia C) | 13,606 | 94.3 % | Canada 750 |
| Canada (x) | 750 | 16,697 | 20 (Alaska) | 13,586 | 81.4 % | Lena 3,037 |
| Russia W (x) | 31 | 17,436 | 0 | 13,606 | 78.0 % | Lena 3,037, Canada 750 |
| Russia E (x) | 30 | 17,437 | 0 | 13,606 | 78.0 % | Lena 3,037, Canada 750 |
| Russia C (x) | 7 | 16,208 | 1,252 (Lena) | 13,606 | 83.9 % | Lena 1,785, Canada 750 |
| Greenland (x) | 3 | 17,464 | 0 | 13,606 | 77.9 % | Lena 3,037, Canada 750 |
| Alaska (x) | 13,606 | 3,860 | 1 (Canada) | 0 | 0 % | Lena 3,037 (78.7 %), Canada 749 |

For the Alaskan sub-regions in mode i, 60.7–77.3 % of the source cells are Alaskan (WRAPUP 5).

**Recalibrated coefficient E_n (P1).** With n labels drawn from the target half A, P1 predicts E_n·s with E_n = (n·E_ls + κ·E0)/(n + κ) and κ = 10. E_ls = Σ s_j·y_j / Σ s_j² over the n drawn labels. In this weighted mean, E0 carries the weight of 10 labels. At n = 0, E_n = E0 and P1 equals P0. At n = 10, E0 and E_ls have equal weight. As n grows, E_n approaches E_ls, the P2 coefficient. The value κ = 10 was fixed before the LG run (LG §3, commit `40be64c`, 2026-09-29 15:45:46). Sensitivity to κ ∈ {3, 30} is registered in LGX X9 and judged under L28.

**Offset estimate (P3).** P3 estimates log E by a normal-normal update (`h4_common.offset_mle_estimate`). The prior mean is log E0. The prior variance τ² is the between-region variance of the mean of log(y/s) over source macro-regions with at least 3 cells, minus the sampling part. The shrinkage weight is n/(n + σ²/τ²), where σ² is the pooled within-region variance of log(y/s).

**Region-equal-weight alternative (not performed).** A source coefficient that gives each source region equal weight was proposed in the wrap-up review (WRAPUP_REVIEW item 11). It was not registered (execution plan decision 9; LG revision 15 (o); WRAPUP revision 2 (e), commit `ff1be02`, 2026-09-30 11:44:30). A registration would have had to precede the retrieval of LG P0 values. The analysis was not performed. The manuscript reports it as a limitation, not as a result.

Paste-ready Methods sentences:

> The source coefficient E0 was the least-squares slope through the origin of ALT on √TDD over all labelled source cells, E0 = Σ s_i y_i / Σ s_i², where s_i is the square root of the ERA5-Land air thawing index (2015–2020 monthly climatology). The source pool for a target excluded all target cells and all cells within 100 km of them. Cells were weighted equally, and Alaskan cells formed 78.0–94.3 % of the source pool for the four main target regions (Lena, Canada, Russia W and Russia E). With n target labels, the recalibrated coefficient was E_n = (n E_ls + 10 E0)/(n + 10), where E_ls is the same slope estimated from the n labels. We did not evaluate an alternative E0 that weights source regions equally.

### M3.2 Discussion paragraphs (draft)

**D1. What E represents.** The Stefan coefficient E is a lumped parameter. In the classical Stefan solution, the thaw depth is Z = (2 k_t I_s / L)^0.5 [CITE: Stefan solution for active-layer thaw, e.g. Riseborough et al. 2008, Permafrost Periglac. Process.]. Here k_t is the thermal conductivity of the thawed layer (W m^−1 K^−1), I_s is the ground-surface thawing index (K·s), and L = ρ_w L_f θ is the volumetric latent heat of the thawed layer (J m^−3). ρ_w is 1,000 kg m^−3, L_f is 3.34 × 10^5 J kg^−1, and θ is the volumetric water or ice content. When the thawing index is taken from air temperature, I_s = n_t I_a, where n_t is the thaw n-factor [CITE: Klene et al. 2001, Arct. Antarct. Alp. Res.]. In the form ALT = E·√TDD_air, the coefficient becomes E = (2 k_t n_t · 86,400 s d^−1 / (ρ_w L_f θ))^0.5, expressed in cm (°C·d)^−0.5. E thus combines the thermal conductivity of the soil, the latent heat set by moisture and ground-ice content, and the surface energy transfer summarized by n_t.

**D2. Site factors that act through E.** Organic surface layers lower k_t when dry and raise θ when wet. Both effects reduce E. Vegetation, moss and shading lower n_t. Late-lying snow delays the start of ground thaw, so the air thawing index overstates the surface thawing index and the fitted E decreases. Soil texture and bulk density change both k_t and θ. Edaphic factors of this kind were the basis of earlier regional Stefan mapping with land-cover specific coefficients [CITE: Nelson et al. 1997, Arct. Alp. Res.; Shiklomanov & Nelson 2002, Permafrost Periglac. Process.].

**D3. Non-physical terms absorbed by E.** E also absorbs terms that are not soil physics. The air thawing index is computed from monthly means of a 0.1° reanalysis grid for 2015–2020. E therefore absorbs the grid-to-site temperature bias, the elevation mismatch within the grid cell, the difference between monthly and daily degree-day sums, and the difference between the climatology period and the observation years. The label definition enters as well. Probe, ground-penetrating radar (GPR) and temperature-derived labels differ, and a label measured before the end of the thaw season underestimates the seasonal maximum. Finally, E absorbs the error of the square-root law itself, which neglects sensible heat, layered soils and unfrozen water [CITE: Romanovsky & Osterkamp 1997, Permafrost Periglac. Process.].

**D4. Magnitude.** For illustration, k_t = 1.0 W m^−1 K^−1, n_t = 1.0 and θ = 0.40 give E ≈ 3.6 cm (°C·d)^−0.5. The combination k_t = 0.5 W m^−1 K^−1, n_t = 0.8 and θ = 0.60 gives E ≈ 1.9 cm (°C·d)^−0.5. The median label ratios ALT/√TDD in the input table are 1.62 cm (°C·d)^−0.5 in Alaska, 1.28 cm (°C·d)^−0.5 in Lena and 1.37 cm (°C·d)^−0.5 in Canada (LG plan 6B.7, label statistics). These medians lie below the second illustrative value. This is consistent with low thermal conductivity, high moisture content, n_t below 1, or a combination of these. The data used in this study do not separate these factors.

**D5. Why E transfers poorly between regions.** A coefficient estimated in one set of regions carries that set's mixture of soils, ground ice, vegetation, snow regimes and label types. Air temperature explains the thawing index but not these controls. Three observations in the input data show the size of the variation. First, the standard deviation of ALT among labels in the same 1 km cell is 11.3 cm in Alaska, 13.0 cm in Lena and 17.7 cm in Canada (WRAPUP 2.2). Second, across Canadian 0.5° blocks with at least 3 labels, the coefficient of variation of the block-mean ratio ALT/√TDD is 0.22 (unit metadata viewed in WRAPUP revision 1; WRAPUP 0.1). Third, the label ratio in the Tibetan candidate data is 11.2 cm (°C·d)^−0.5 for GPR labels and 12.9 cm (°C·d)^−0.5 for temperature-derived labels, about 7–9 times the Arctic values (LG plan 6B.7). The label type also differs by region. Alaskan labels are mostly ABoVE point observations, many from GPR, with some CALM site means. Canadian labels are ABoVE point observations with some CALM site means. Lena labels are mostly single-visit points. Russia W and E labels are multi-year CALM site means (WRAPUP 2.1–2.2).

**D6. What P0 tests in this design.** E0 is pooled over source cells without regional weights. For each of the four main targets, 78.0–94.3 % of the source cells are Alaskan (Table M3-1). P0 therefore tests how well an Alaska-dominated coefficient transfers. It does not test a generic physical coefficient. The registered covariate-dependent coefficient models test whether covariates recover the between-region variation of E: V1 under LG L7 [RESULT: L7 verdict] and V2 under LGX L13 [RESULT: L13 verdict]. The wrap-up plan records earlier negative results for models that predict E (WRAPUP 11.1, H27 and V1) and adds no new E model.

**D7. Relation of the two baselines.** P0 gives the error of a coefficient transferred from other regions with no target label. P1 gives the error after the mean level of E in the target is estimated from n labels, with E0 as a prior worth 10 labels. P1 changes one multiplicative factor. It corrects the regional level of E but not the variation of E within the target. The contrast P1 − P0 at n = 10 (AB4) measures the value of a few labels for the coefficient level [RESULT: AB4 4-way verdict and Holm-corrected wording]. The contrasts R1 − P1 at n = 10 (AB5) and at full labels (AB9), and the minimum n of LG L4, measure what residual learning adds after this level correction [RESULT: AB5, AB9, L4]. Two pre-result sensitivity analyses concern the label unit and the label year (M5.4, M5.5). A sequential stopping rule based on the jackknife precision of the target coefficient rarely stopped early at the registered threshold (M5.6).

**D8. Limitation.** The construction of E0 is a design choice with two consequences. First, P0 and every method anchored on E0 (R0, D1 pseudo-labels at n = 0) inherit the Alaskan dominance of the source pool. Second, their performance relative to direct ML may differ under another pooling rule. We did not evaluate a region-equal-weight E0, so the direction of such a change is unknown. The dependence on E0 decreases as n grows, because E0 carries a fixed weight of 10 labels in E_n.

**Reference candidates for M3** (not verified in this session; check the original papers before citing): Nelson, F. E. et al. (1997) Arctic and Alpine Research 29, 367–378 (Kuparuk regional Stefan mapping with edaphic coefficients). Shiklomanov, N. I. & Nelson, F. E. (2002) Permafrost and Periglacial Processes 13, 219–230. Klene, A. E., Nelson, F. E., Shiklomanov, N. I. & Hinkel, K. M. (2001) Arctic, Antarctic, and Alpine Research 33, 140–148 (n-factors). Romanovsky, V. E. & Osterkamp, T. E. (1997) Permafrost and Periglacial Processes 8, 1–22. Riseborough, D. et al. (2008) Permafrost and Periglacial Processes 19, 137–156.

---

## M4. SI table frames

### M4.1 Conventions

- **Status vocabulary** (filled after the verdicts, from each plan's own rule): supported, partially supported, rejected, equivalent, undetermined, not decidable, partial (regions k/4 or splits k/K), rule not operating, descriptive. The status column is left as `[RESULT]`. Verdict wording follows each plan (WRAPUP 1.4 (a) applies only to LG L1–L8).
- **Role**: C = confirmatory (LGX L10, L12, L15, L19, L29, L30; LGU-A1, B2; LGF-F1, N1). P = primary LG hypothesis (L1–L8, judged by the h40 aggregation). A = auxiliary. D = descriptive or diagnostic (no verdict). N = direction-neutral report. abstract = contrast of the abstract bundle (WRAPUP 1.1), recomputed by h39 with 10,000 resamples in both weightings; it sets the abstract wording through the Holm rule and does not change the verdict of its basis hypothesis.
- **Blinding label** (as stated in the plans): B = blind. PU = partially unblinded (direction seen in an earlier experiment with a different protocol). RU = replication (unblinded). RE = result existed on disk at registration, not viewed. Sa = S-a viewing label (M5.7). NS = not listed in the plan's blinding table (LG §4 has no such table; LGU §6 lists only unblinded items).
- **Registration**: the first commit that contains the hypothesis text (hash, date and KST time). Later amendments that change operational definitions, not wording, are listed in M4.2.

### M4.2 Registration events

| Plan (section) | Hypotheses | Registered | Later amendments (no change of hypothesis wording) |
|---|---|---|---|
| LG §4 | L1–L8 | `40be64c` 2026-09-29 15:45:46 (initial text with revisions 1–2) | revision 3 `249005c` 16:35:29; revision 4 `19f3f93` 17:04:45 (ZovWo submitted); revision 6 `d8355d2` 18:18:36 (D0 description "no physics anchor") |
| LG 6A (LGX) | L9–L31 | `8513cc2` 2026-09-29 18:17:29 (revision 5) | revisions 7–8 `f29bfc8` 21:28:41; revision 11 (first committed as "revision 9" in `6799b74` 22:43:12, renumbered in `cec5719` 23:29:45); revision 15 (h) `ff1be02` 2026-09-30 11:44:30 |
| LG 6C (LGT) | L32–L37 | `6799b74` 2026-09-29 22:43:12 (revision 9; section written 21:45) | revision 12 `45ef3b6` 2026-09-30 01:39:42 |
| LG 6B (LGD) | L1e, L4e, L8e, L38–L42 | `6799b74` 2026-09-29 22:43:12 (revision 10; section written 22:42) | revision 13 `896a15e` 02:35:11; revision 14 `028900a` 04:19:28; revision 15 (m) `ff1be02` 11:44:30 |
| LGU | A1–A7, B1–B6, C1–C4 | `a112cef` 2026-09-29 23:30:02 (written 22:45) | revisions 1–3 `582be9a` 2026-09-30 03:36:00; revision 4 `ff1be02` 11:44:30 |
| LGF | F1–F6, N1, N2, N2s, N3, N4 | `3a4fe84` 2026-09-30 01:44:27 (written 01:30) | revision 1 `f208516` 02:22:24; revisions 2–4 `43df2f8` 04:02:52; revision 5 `b1a4349` 05:00:40; revision 6 `ff1be02` 11:44:30 |
| WRAPUP | L43, SC1w–SC3w, AK1w, AB1–AB10 | `316714c` 2026-09-30 03:02:42 (initial draft with revision 1) | revision 2 `ff1be02` 11:44:30 |

Timing facts for the register: ZovWo (LG) was submitted at 2026-09-29 17:04:45. LGT started at 2026-09-30 01:40:24 and its first shard appeared at 01:47:33. The first retrieval (T_res) was 2026-09-30 07:20:22. All registration commits above precede T_res. The four amendments in `ff1be02` follow T_res and carry the label described in M4.5 row 21.

### M4.3 Table S-R1. Register of registered hypotheses (frame)

| ID | Plan | Role | Statement (abridged) | Registered | Blinding | Status |
|---|---|---|---|---|---|---|
| L1 | LG §4 | P | Direct ML (D0) does not outperform the source-coefficient Stefan model (P0) at n ≤ 40 (at most 1 of 4 main regions with both CI upper bounds < 0) | `40be64c` 09-29 15:45:46 | PU (M1 n = 0 direct ML; P2/W3) | [RESULT] |
| L2 | LG §4 | P | Augmented combinations R2 or R3 have lower error than R1 at n ∈ {10, 40, 160}; otherwise "augmentation is for n = 0 only" | `40be64c` 09-29 15:45:46 | B | [RESULT] |
| L3 | LG §4 | P | Target weighting α ∈ {10, 100, continued training}, chosen by within-A cross-validation, gives n* ≤ 40 in ≥ 2 of 4 structure targets | `40be64c` 09-29 15:45:46 | NS | [RESULT] |
| L4 | LG §4 | P | Residual ML (R1) outperforms the recalibrated Stefan model (P1); report the minimum n | `40be64c` 09-29 15:45:46 | PU (a2, h25b) | [RESULT] |
| L5 | LG §4 | P | Neural and generative learners are on par with CatBoost in R1 (manuscript uses L26, WRAPUP 1.5) | `40be64c` 09-29 15:45:46 | NS | [RESULT] |
| L6 | LG §4 | P | Combinations that pass in Alaskan sub-regions (mode i) also pass in the main 4 regions at the same n (share ≥ 0.5) | `40be64c` 09-29 15:45:46 | NS | [RESULT] |
| L7 | LG §4 | P | The covariate-dependent coefficient V1 (or V1r) removes the degradation in Canada and CA-3 (n ≥ 40, point estimate) | `40be64c` 09-29 15:45:46 | NS | [RESULT] |
| L8 | LG §4 | P | With all labels, R1 outperforms P0 (main-4 stratified mean, both weightings) | `40be64c` 09-29 15:45:46 | PU (M1 H13) | [RESULT] |
| L9 | LGX X1 | A | F1a (D0 plus E0·s input) and D0 are indistinguishable | `8513cc2` 09-29 18:17:29 | B | [RESULT] |
| L10 | LGX X1 | C | Residual structure has lower error than physics-input structure (four contrasts at n = 0 and n = 10) | `8513cc2` 09-29 18:17:29 | B | [RESULT] |
| L11 | LGX X1 | A | Direct ML with process-model inputs (F1k) does not outperform P0 at n = 0 | `8513cc2` 09-29 18:17:29 | RU (M1 n = 0 direct ML) | [RESULT] |
| L12 | LGX X1 | C | Multiplicative correction RM and additive residual R1 are equivalent (n ∈ {10, all}, λ 0.25 and 1.0) | `8513cc2` 09-29 18:17:29 | B | [RESULT] |
| L13 | LGX X1 | A | V2 (log-ratio model) is better than P1 and not different from R1 at n = 10 | `8513cc2` 09-29 18:17:29 | RU for Canada and CA-3 | [RESULT] |
| L14 | LGX X1 | N | Effect of removing physics-related inputs from D0 (D0t, D0c, D0m) | `8513cc2` 09-29 18:17:29 | B | [RESULT] |
| L15 | LGX X2 | C | Stefan pseudo-labels (D1) have lower error than four placebos at n ∈ {0, 10} | `8513cc2` 09-29 18:17:29 | RU at n = 0 (M1 H3) | [RESULT] |
| L16 | LGX X2 | A | Verdicts are insensitive to the pseudo-label ratio r; D1 approaches the physics model as r grows | `8513cc2` 09-29 18:17:29 | B | [RESULT] |
| L17 | LGX X2 | A | R1 and the label-shuffle control R1s are equivalent (n ∈ {10, all}) | `8513cc2` 09-29 18:17:29 | B | [RESULT] |
| L18 | LGX X2 | A | With target weighting, the augmented combination is not better than R1 | `8513cc2` 09-29 18:17:29 | B | [RESULT] |
| L19 | LGX X3a | C | Direct ML beats the physics model under random splits (V-R) but not under region hold-out (V-G) | `8513cc2` 09-29 18:17:29 | RU (2026-06 split leakage table, M1) | [RESULT] |
| L20 | LGX X3a | A | D0 RMSE increases in the order V-R < V-B < V-C100 < V-C500 | `8513cc2` 09-29 18:17:29 | RU | [RESULT] |
| L21 | LGX X3a | A | Degradation from V-R to V-G is smaller for anchor plus residual than for D0 | `8513cc2` 09-29 18:17:29 | B | [RESULT] |
| L22 | LGX X3b | A | Within a region, R1w outperforms P1w (count of regions out of 3) | `8513cc2` 09-29 18:17:29 | RU for Alaska (M1 X-ak) | [RESULT] |
| L23 | LGX X3c | A | Labels near scored cells reduce error more than the same number of distant labels | `8513cc2` 09-29 18:17:29 | B | [RESULT] |
| L24 | LGX X5 | A | Residual-learning net value concentrates near labels; recalibration gain does not depend on distance | `8513cc2` 09-29 18:17:29 | B | [RESULT] |
| L25 | LGX X5 | N | R1 90 % interval coverage and width versus the R0 n = 0 interval (6 cells) | `8513cc2` 09-29 18:17:29 | B | [RESULT] |
| L26 | LGX X6 | A | Each learner minus CatBoost (R1) lies within the 0.5 cm margin at n ∈ {0, 10, all}, 5 splits | `8513cc2` 09-29 18:17:29 | B | [RESULT] |
| L27 | LGX X6 | A | The sign of R0 − P0 in Canada does not depend on the learner (≥ 6 of 8 learners) | `8513cc2` 09-29 18:17:29 | B | [RESULT] |
| L28 | LGX X9 | A | 4-way verdicts of core contrasts are unchanged under κ, label-definition, anchor, input-set and matched-year variants | `8513cc2` 09-29 18:17:29 | RU for κ, λ and anchors | [RESULT] |
| L29 | LGX N1 | C | No physics or product baseline is better than P0 at n = 0 | `8513cc2` 09-29 18:17:29 | PU for B:ens − P0 (LG revision 15 (h)) | [RESULT] |
| L30 | LGX N3 | C | The n = 0 direct ML result does not depend on learner capacity (catboost, rf, catboost_tuned) | `8513cc2` 09-29 18:17:29 | RU (M1 n = 0 direct ML) | [RESULT] |
| L31 | LGX N4 | A | The full-label R1 − P0 improvement is robust to split composition (50 splits) | `8513cc2` 09-29 18:17:29 | RU for Canada, CA-3, AL-5; Sa | [RESULT] |
| L32 | LGT | A | R1[TabPFN] − R1[CatBoost, same context] lies within the 0.5 cm margin at n ∈ {0, 10, 40, 160, all} | `6799b74` 09-29 22:43:12 | RU at n ∈ {10, 40} (b4); B at n = 0, 160, all | [RESULT] |
| L33 | LGT | A | TabPFN residual (R1) outperforms P1; report the minimum n | `6799b74` 09-29 22:43:12 | RU at n ∈ {3, 10, 40} (b4); B otherwise | [RESULT] |
| L34 | LGT | A | (a) TabPFN direct prediction does not outperform P0 at n = 0; (b) R0[TabPFN] − P0 reported | `6799b74` 09-29 22:43:12 | B; CatBoost rows RU | [RESULT] |
| L35 | LGT | A | With all labels, TabPFN R1 outperforms P0 | `6799b74` 09-29 22:43:12 | B; CatBoost rows RU | [RESULT] |
| L36 | LGT | N | D0[TabPFN] − R1[TabPFN] at five n | `6799b74` 09-29 22:43:12 | B | [RESULT] |
| L37 | LGT | A | Core-contrast verdicts are stable under context variants (d3, d10, rid, tgt) | `6799b74` 09-29 22:43:12 | RU for d3, rid | [RESULT] |
| L1e | LGD | A | L1 recomputed on the extended pools PE1 and PE2 | `6799b74` 09-29 22:43:12 | as L1 for P4; RE under revision 15 (m) | [RESULT] |
| L4e | LGD | A | L4 recomputed on PE1 and PE2 (minimum n with region count k/N) | `6799b74` 09-29 22:43:12 | as L4 for P4; RE under revision 15 (m) | [RESULT] |
| L8e | LGD | A | L8 recomputed on PE1 and PE2 | `6799b74` 09-29 22:43:12 | Tibet unblinded (label statistics); RE under revision 15 (m) | [RESULT] |
| L38 | LGD | A | In each new eligible region, the region-level verdicts follow the hypothesis directions | `6799b74` 09-29 22:43:12 | Tibet unblinded; NAtlantic, Russia C blind | [RESULT] |
| L39 | LGD | A | In Tibet, changing the label definition leaves core verdicts unchanged | `6799b74` 09-29 22:43:12 | unblinded (label statistics) | [RESULT] |
| L40 | LGD | A | Adding new cells to Russia W, Russia E and Canada leaves region-level verdicts unchanged | `6799b74` 09-29 22:43:12 | RU | [RESULT] |
| L41 | LGD | A | New-region verdicts are robust to label-set variants (a)–(c) | `6799b74` 09-29 22:43:12 | as L38 | [RESULT] |
| L42 | LGD | A | New-region verdicts are unchanged when other new regions join the source | `6799b74` 09-29 22:43:12 | as L38 | [RESULT] |
| L43 | WRAPUP 4 | A | Block-dispersed label placement gives lower P1 error than cell-random placement (n ∈ {10, 40}) | `316714c` 09-30 03:02:42 | PU (a2_spread, h29b) | [RESULT] |
| SC1w | WRAPUP 3.3 | A | At T3 and T10, recipe R1 is non-inferior to P0 (0.5 cm) in all 5 independent regions (10 cells) | `316714c` 09-30 03:02:42 | PU; SC1w-P: RE and Sa | [RESULT] |
| SC2w | WRAPUP 3.4 | N | Number of regions (Lena, Canada, Alaska x) where R1 beats P1 at n = 40 | `316714c` 09-30 03:02:42 | PU | [RESULT] |
| SC3w | WRAPUP 3.5 | A | A sequential stopping rule (jackknife SE of log E ≤ τ = 0.05) halves labels versus always 40 with error increase ≤ 0.3 cm and no inferior region | `316714c` 09-30 03:02:42 | PU; computed before results (M5.6) | [RESULT] |
| AK1w | WRAPUP 5 | A | Combinations that pass in Alaskan sub-regions with a within-Alaska source (mode i) also pass with an outside-only source (mode x); r ≥ 0.5 | `316714c` 09-30 03:02:42 | PU (h25b, h25x) | [RESULT] |
| AB1 | WRAPUP 1.1 | abstract | D0 − P0, n = 0, λ 1.0, main 4 (basis L1) | `316714c` 09-30 03:02:42 | PU | [RESULT] |
| AB2 | WRAPUP 1.1 | abstract | B:ens − P0, n = 0 (basis L29) | `316714c` 09-30 03:02:42 | PU | [RESULT] |
| AB3 | WRAPUP 1.1 | abstract | D1 − D1@shuffle, n = 0, λ 1.0 (basis L15) | `316714c` 09-30 03:02:42 | RU | [RESULT] |
| AB4 | WRAPUP 1.1 | abstract | P1 − P0, n = 10 (curve core, L28 core contrast) | `316714c` 09-30 03:02:42 | PU; RE (LGT P0, P1); Sa | [RESULT] |
| AB5 | WRAPUP 1.1 | abstract | R1 − P1, n = 10, λ 0.25 (basis L4) | `316714c` 09-30 03:02:42 | PU | [RESULT] |
| AB6 | WRAPUP 1.1 | abstract | R2 − R1, n = 10, λ 0.25 (basis L2) | `316714c` 09-30 03:02:42 | B | [RESULT] |
| AB7 | WRAPUP 1.1 | abstract | R1 − F1k, n = 10 (basis L10) | `316714c` 09-30 03:02:42 | B | [RESULT] |
| AB8 | WRAPUP 1.1 | abstract | R1 − P0, all labels, λ 0.25 (basis L8) | `316714c` 09-30 03:02:42 | PU | [RESULT] |
| AB9 | WRAPUP 1.1 | abstract | R1 − P1, all labels, λ 0.25 (basis L4) | `316714c` 09-30 03:02:42 | PU | [RESULT] |
| AB10 | WRAPUP 1.1 | abstract | Interval score (α 0.1), rung (iii) − B4, n = 10, Lena, Canada, Alaska x (basis LGU-A1) | `316714c` 09-30 03:02:42 | per LGU §6; RE (LGU revision 4) | [RESULT] |
| LGU-A1 | LGU 3.6 | C | The hierarchical predictive interval (rung iii) has a lower interval score than the constant-width calibrated interval B4 at n ∈ {10, 40} | `a112cef` 09-29 23:30:02 | NS; RE under revision 4 | [RESULT] |
| LGU-A2 | LGU 3.6 | D | Share of infinite intervals at n = 3 is 0 (implementation check) | `a112cef` 09-29 23:30:02 | RU (C2 wconf) | [RESULT] |
| LGU-A3 | LGU 3.6 | N | Median RMSE, rung (iii) − P1, n ∈ {3, 10}, δ 0.5 cm | `a112cef` 09-29 23:30:02 | RU (B2 h32 offsets) | [RESULT] |
| LGU-A4 | LGU 3.6 | gate | CRPS (log scale), rung (v) − (iv), per target, leave-one-region-out | `a112cef` 09-29 23:30:02 | NS | [RESULT] |
| LGU-A5 | LGU 3.6 | D | n = 0 coverage and interval score under hyperprior variants (exploratory) | `a112cef` 09-29 23:30:02 | RU | [RESULT] |
| LGU-A6 | LGU 3.6 | D | Russia W at n ∈ {3, 10}: interval score, coverage, width | `a112cef` 09-29 23:30:02 | NS | [RESULT] |
| LGU-A7 | LGU 3.6 | A | Ladder contrasts (ii)−(i), (iii)−(ii), (iv)−(iii), (vi)−(iii), (v)−(iv) | `a112cef` 09-29 23:30:02 | NS | [RESULT] |
| LGU-B1 | LGU 4.4 | A | Pooled generative quantile intervals (nflow, cfm) cover less than 0.85 in new regions (limit check) | `a112cef` 09-29 23:30:02 | unblinded (M1 H15, H17) | [RESULT] |
| LGU-B2 | LGU 4.4 | C | nflow-normalized conformal intervals have a lower interval score than their σ-shuffled placebo at n = 0 (4 conditions) | `a112cef` 09-29 23:30:02 | NS; RE under revision 4 | [RESULT] |
| LGU-B3 | LGU 4.4 | A | Interval score nflow − cbq (5 % margin) | `a112cef` 09-29 23:30:02 | NS | [RESULT] |
| LGU-B4 | LGU 4.4 | A | Each normalizer − constant, raw and width-matched | `a112cef` 09-29 23:30:02 | NS | [RESULT] |
| LGU-B5 | LGU 4.4 | D | Coverage and width by instrument, season, width quintile and source distance | `a112cef` 09-29 23:30:02 | NS | [RESULT] |
| LGU-B6 | LGU 4.4 | A | nflow − const and cbq − const at n ∈ {10, 40, 160} | `a112cef` 09-29 23:30:02 | NS | [RESULT] |
| LGU-C1 | LGU 5 | D | Variogram of the centred log ratio (no verdict) | `a112cef` 09-29 23:30:02 | RU | [RESULT] |
| LGU-C2 | LGU 5 | D | Within-block correlation (no verdict) | `a112cef` 09-29 23:30:02 | RU | [RESULT] |
| LGU-C3 | LGU 5 | D | Share of scored cells with a label within d km (no verdict) | `a112cef` 09-29 23:30:02 | RU | [RESULT] |
| LGU-C4 | LGU 5 | D | Alignment of SAR products with probe labels (no verdict) | `a112cef` 09-29 23:30:02 | RU | [RESULT] |
| LGF-F1 | LGF 3.6 | C | TabICL direct prediction does not outperform P0 at n = 0 (context cap and full source) | `3a4fe84` 09-30 01:44:27 | B; CatBoost rows RU | [RESULT] |
| LGF-F2 | LGF 3.6 | A | R1[TabICL] − R1[CatBoost, same context] lies within 0.5 cm at five n | `3a4fe84` 09-30 01:44:27 | B | [RESULT] |
| LGF-F3 | LGF 3.6 | N | TabICL versus TabPFN (six contrasts) | `3a4fe84` 09-30 01:44:27 | B; TabPFN column RE at revision 1 | [RESULT] |
| LGF-F4 | LGF 3.6 | A | TabICL residual (R1) outperforms P1; full-label R1 − P0 | `3a4fe84` 09-30 01:44:27 | B; TabPFN rows at n ≤ 40 RU | [RESULT] |
| LGF-F5 | LGF 3.6 | N | n = 0 R0[TabICL] − P0 | `3a4fe84` 09-30 01:44:27 | B | [RESULT] |
| LGF-F6 | LGF 3.6 | N | Effect of the context cap (full source minus 10,000 rows) | `3a4fe84` 09-30 01:44:27 | B | [RESULT] |
| LGF-N1 | LGF 4.5 | C | Direct ML from four tuned discriminative networks does not outperform P0 at n = 0 | `3a4fe84` 09-30 01:44:27 | PU (default-version direction, M1) | [RESULT] |
| LGF-N2 | LGF 4.5 | A | Tuned and default networks differ by less than the 0.5 cm margin (9 contrasts per learner) | `3a4fe84` 09-30 01:44:27 | B | [RESULT] |
| LGF-N2s | LGF 4.5 | A | Core-contrast verdicts are the same for default and tuned networks | `3a4fe84` 09-30 01:44:27 | B | [RESULT] |
| LGF-N3 | LGF 4.5 | A | R1[tuned network] − R1[tuned CatBoost] lies within 0.5 cm at n ∈ {0, 10, all} | `3a4fe84` 09-30 01:44:27 | B; default rows RU | [RESULT] |
| LGF-N4 | LGF 4.5 | D | Selection tables and tuning diagnostics (no hypothesis) | `3a4fe84` 09-30 01:44:27 | NS | [RESULT] |

Row count: LG §4 8, LGX 23, LGT 6, LGD 8, WRAPUP 15 (L43, SC1w–SC3w, AK1w, AB1–AB10), LGU 17, LGF 11; total 88. LGX items without a hypothesis (N2 error floor, X5 variable contributions, X3c S_buf, X3a V-P and V-S) are reporting rules and are not listed.

Planned SI columns after the verdicts: verdict wording from the plan, CI pool size (regions k/4), split completeness (k/K), Holm-adjusted p (auxiliary column where the plan defines one), cross-environment label (WRAPUP 8.6), and the full blinding label.

### M4.4 Table S-R2. Past negative results, reclassified (frame, WRAPUP 1.4 (b))

Rules: rows with both cell-weighted and block-equal CIs are reclassified into the 4-way verdict (δ 0.5 cm) and labelled "reclassified (unblinded)". Rows with one weighting only are "4-way verdict not possible (CI protocol differs)". Rows with row-resampling CIs or ratio formats are "protocol differs, descriptive". Stored predictions with block ids are re-scored with `h4_common.boot_delta_blocks` and labelled "re-scored (unblinded)". Original verdicts are not changed. The input list is the corrected 29-input specification (WRAPUP results addendum 1 fix 1; h39 `RECLASS_SPEC`). Values come from `lgw_rescore.csv` (execution plan C7). FINAL-plan hypotheses are written with the prefix FP- to avoid a clash with LGF-F1–F6.

| Past ID | Source table | CI type | Original resamples | Reclassification | Value | Related registered test | Audit and withdrawal history |
|---|---|---|---|---|---|---|---|
| M1 H-series | `m1/m1_sc_tests.csv` | [VERIFY] | not recorded | [RESULT] | [RESULT] | H13 to L8 and AB8; H3 to L15 and AB3; H15, H17 to LGU-B1 | [VERIFY] |
| FP-F4 | `h4/b1_summary.csv` (d_phys columns), `b1_protocol.csv` | [VERIFY] | 1,000 | [RESULT] | [RESULT] | SC1w, SC3w (b1 "always shrink plus residual") | FINAL verdict rejected: the 3-label two-stage protocol was not better than always shrink plus residual (`f098031` 2026-09-26 21:42:16) |
| FP-F5, FP-F6 | `h4/b2_curve.csv` (d_phys, d_shrink columns), `b2_regional.csv` (cell-weighted only), `b2_f5.csv`, `b2_theory_corr.csv` | mixed | 1,000 | [RESULT] | [RESULT] | AB4, L28 (κ), P3 | FINAL verdicts rejected: adaptive pooling did not beat κ = 10 (F5); theoretical label count uncorrelated (F6) |
| FP-F7 | `h4/b3_summary.csv` (blk_d_phys, blk_d_rand columns), `b3_region.csv`, `b3_f7.csv` | [VERIFY] | 1,000 | [RESULT] | [RESULT] | none (selection rules at n ≥ 80 excluded, LG 6A.10) | FINAL verdict rejected: no gain from D-optimal selection |
| FP-F8 | `h4/b4_tests.csv` | [VERIFY] | not recorded | [RESULT] | [RESULT] | L32, L33 (TabPFN) | FINAL verdict supported; "TabPFN on par" wording to be corrected (J10) |
| FP-F9 | `h4/c1_tests.csv` | [VERIFY] | 1,000 | [RESULT] | [RESULT] | L29, AB2 (physics and product baselines) | FINAL verdict rejected: no gain from multi-layer CCI combination |
| H18–H22 | `h2/h_tests_all.csv` (block-equal CIs present, 1,055 rows); predictions `m1/h1819_shard*_preds.npz`, `h2/h21_preds.npz` (includes H20), `h2/h22_preds.npz` | two weightings | not recorded | [RESULT] | re-scored [RESULT] | none | WRAPUP 1.4 (b) body text corrected by revision 2 (c) |
| H23, H24 | `h2/h23_tests.csv`, `h2/h24_tests.csv` | row resampling | not recorded | protocol differs, descriptive | [RESULT] | L43 (placement) | H24 verdict "A (adopted)" withdrawn (AUDIT action 4, `f098031` 2026-09-26 21:42:16); "H24 k-center adopted" discarded (RESEARCH_FRAME A.4, `6830277` 2026-09-29 13:18:02) |
| H25 | `h3/h25b_curve.csv`, `h3/h25x_curve.csv` (two weightings); `h3/h25_curve.csv` (row CI); break-even tables (descriptive) | mixed | 1,000 (h25b, h25x); not recorded (h25) | [RESULT] | [RESULT] | L4, AK1w | "break-even n set by abs(log E ratio), ρ −0.84" redefined (AUDIT action 1, `f098031`) and discarded (RESEARCH_FRAME A.4, `6830277`) |
| H26 | registered paired test not produced | none | none | source missing | none | none | rejected without the registered paired CI; criterion unattainable at registration (AUDIT workflow:F5, `f098031`) |
| H27 | `h3/h27*_rule.csv` (4 files) | rule tables | not recorded | protocol differs, descriptive | [RESULT] | none | redefined with H25 (AUDIT action 1) |
| H28 | `h3/h28_tests.csv` | [VERIFY] | not recorded | [RESULT] | [RESULT] | L3 (nested α selection) | H28 = rejected (AUDIT; RESEARCH_FRAME A.4) |
| H29 | `h3/h29_summary.csv`, `h3/h29b_summary.csv` | summary | not recorded | protocol differs, descriptive | [RESULT] | L43 (h29b harmful single-block share) | block-CI re-run h29b (`f098031` 2026-09-26 21:42:16) |
| H30 | `h3/h30_deploy_tests.csv` | [VERIFY] | not recorded | [RESULT] | [RESULT] | [VERIFY] | [VERIFY] |

### M4.5 Table S-H1. Selection and withdrawal history

| # | Item | Type | Origin (commit, KST) | Action and reason | Recorded in (commit, KST) | Timing relative to LG-family results |
|---|---|---|---|---|---|---|
| 1 | "Component decomposition: 72–95 % of the few-label gain in level regions is coefficient recalibration" (NOVELTY N3; FINAL plan §10.3) | dropped claim | FINAL plan and session records, 2026-09-26 | Rejected as a contribution claim (2 reviewers reject, 1 conditional). Reasons: the residual CI includes 0 only in Russia W; CA-2 degrades under recalibration (+7.7 cm); the share keeps only targets where recalibration worked; the classification uses scored labels; at α = 1 the residual does not respond to target labels. Remaining form: case values in cm, no percentage | NOVELTY §1, §5, §10 (`d8355d2` 2026-09-29 18:18:36) | before (no LG result) |
| 2 | "The ranking of ML and physics reverses with the validation scheme" (NOVELTY N8) | dropped claim | earlier internal answers | Rejected. The paired Alaskan block contrast failed the two-weighting rule (cell-weighted +1.66 [−1.32, 3.93], block-equal −1.87 [−3.15, −0.72] cm); direct ridge was below Stefan in every scheme; no CI; pattern already in Gautam 2025. Remaining form: methods justification. The registered test is L19, whose rule forbids the word "reversal" unless direct ML wins under random splits | NOVELTY §5 (`d8355d2` 2026-09-29 18:18:36); L19 (`8513cc2` 18:17:29) | before |
| 3 | NOVELTY N5 placement clause (dispersed versus concentrated labels) | dropped clause | NOVELTY draft | Deleted: no block-equal CI on mean rows; sign of the n = 160 value depends on the target set. A placement sentence is allowed only through L43 | NOVELTY §5 (`d8355d2`); L43 (`316714c` 2026-09-30 03:02:42) | before |
| 4 | NOVELTY N2 (two baselines) as an independent contribution | demoted | NOVELTY draft | Moved to Methods; depends on N1 and has precedents | NOVELTY §5 (`d8355d2`) | before |
| 5 | "Break-even n predicted by abs(log E ratio), ρ −0.84"; "H24 k-center adopted"; "Lena and Canada cannot exceed at n ≤ 320"; contest result "CCI combination cancels bias"; control ladder "all rungs significant" | dropped claims | H24, H25, H27, contest 2026-07 | Discarded or withdrawn, no reuse | RESEARCH_FRAME A.4 (`6830277` 2026-09-29 13:18:02); AUDIT (`f098031` 2026-09-26 21:42:16) | before |
| 6 | SC1–SC3 (NEXT P3; RESEARCH_FRAME §5.1 rank 5, §5.2) | withdrawn hypotheses | NEXT `f227fc0` 2026-09-28 17:40:56; RESEARCH_FRAME `6830277` | Withdrawn and replaced by SC1w–SC3w. Reasons: the level/structure classification became a post hoc variable; the 3-split h25b protocol was replaced by the LG grid. Never computed | WRAPUP 3.1 (`316714c` 2026-09-30 03:02:42); NEXT revision line (`ff1be02` 11:44:30) | before (withdrawal); the NEXT record line after T_res |
| 7 | AK1 (NEXT §8.3) | withdrawn hypothesis | NEXT `6830277` 2026-09-29 13:18:02 | Withdrawn and replaced by AK1w. Reason: stage 2 re-scores the same Alaskan sub-regions with another source, so "transfers to new regions" was broader than the design. Never computed | WRAPUP 5 (`316714c`); NEXT revision line (`ff1be02`) | before; NEXT line after T_res |
| 8 | SC1w decision rule "no inferior cell and ≥ 8 decided cells" | replaced rule | WRAPUP initial draft (uncommitted, file hash 5976615c…) | Replaced by cell-wise non-inferiority before commit; the draft rule counted undetermined cells toward support | WRAPUP revision 1 (2) (`316714c`) | before |
| 9 | Three-way classification in WRAPUP 7.2 (a)3 | withdrawn rule | WRAPUP initial draft | Withdrawn before commit; L38 follows the LG 6B.5 wording | WRAPUP revision 1 (3) (`316714c`) | before |
| 10 | Abstract bundle selection: the review example (L1, L2, L4 n = 10, L8, P1 − P0 n = 10, L10, L15, L29, L30) became AB1–AB10 | selection | WRAPUP_REVIEW item 4 (`422c34d` 2026-09-30 01:45:19) | L30 removed (learner robustness is written as a qualifier); AB9 (second baseline) and AB10 (uncertainty) added. Selection made with knowledge of earlier experiments (M1, P2, W3, E1–E3, h25b, a2, b1, b2) | WRAPUP 1.1 (`316714c`) | before |
| 11 | LGX X7: existing product comparison and CALM trend reproduction | excluded analysis | LG 6A.10 (`8513cc2` 2026-09-29 18:17:29) | Excluded; product comparison later registered only as a conditional descriptive SI table (WRAPUP 10, executed only if the map is in the main text) | WRAPUP 10 (`316714c`) | before |
| 12 | LG 6A.10 exclusions of the TabPFN label grid and of independent-region expansion | exclusions replaced | LG 6A.10 (`8513cc2`) | Replaced by LGT (LG revision 9) and LGD (LG revision 10); the Mongolia and Central Asia exclusion is kept | `6799b74` 2026-09-29 22:43:12 | before |
| 13 | WRAPUP 11.1: local re-run of running Rescale jobs; LG refit with site-level draws; time-axis validation, campaign hold-out, CALM trends; site-scale GIPL2 or CryoGrid runs; selection rules at n ≥ 80 and a measurement-priority map; maps for several regions; Ran 2022 ESSD and 15 pending data sets; ERA5-Land 1990–2009 year matching; residual kriging baseline (conditional on LGU-C1); n = 80 grid point (conditional on L4 minimum n = 160); LGF lower-priority tiers; a new ML model of E; Mongolia in the confirmatory pool | excluded analyses | WRAPUP 11.1 (`316714c` 2026-09-30 03:02:42) | Not performed; reasons as in WRAPUP 11.1. The two conditional items (execution plan C12, C13) follow the registration format of LG revision 15 (n) | WRAPUP 11.1 (`316714c`); LG revision 15 (n) (`ff1be02`) | before |
| 14 | Exception to row 13, first item: remaining FT-Transformer shards of the stopped Qjpbeb job | scoped exception | user directive 2026-09-30 08:45 | Run locally as a cross-environment auxiliary only; the primary verdict closes on Rescale shards; auxiliary verdicts are not used in confirmatory sentences or the abstract | WRAPUP revision 2 (d); LG revision 15 (c)(d)(f) (`ff1be02` 11:44:30) | after T_res, bundles not unpacked |
| 15 | Covariate-space uniform selection rule (RESEARCH_FRAME 5.2, FAILURE_ANALYSIS 6), Gautam 2025 set-up reproduction, R0 covariate-support check rule | excluded analyses | RESEARCH_FRAME 5.2 (`6830277`) | Not performed in this paper; LGX X3a covers random versus region hold-out; map panel (e) marks extrapolation | WRAPUP revision 2 (e) (`ff1be02` 11:44:30) | after T_res, not unpacked |
| 16 | NOVELTY 7.4 option 3 (exclusion of ALT values below 20 cm; 138 Alaskan and 89 Lena rows) and option 4 (source-target TDD range overlap table) | excluded analyses | NOVELTY 7.4 (`d8355d2` 2026-09-29 18:18:36) | Not registered by the user; sensitivity analyses unrelated to confirmatory verdicts | WRAPUP revision 2 (e); LG revision 15 (o) (`ff1be02`) | after T_res, not unpacked |
| 17 | Region-equal-weight E0 sensitivity | not registered | WRAPUP_REVIEW item 11 (`422c34d` 2026-09-30 01:45:19) | Not registered and not performed; only the E interpretation paragraph is written (M3) | execution plan decision 9; WRAPUP revision 2 (e); LG revision 15 (o) (`ff1be02`) | after T_res; would have required registration before LG P0 retrieval |
| 18 | LGX-8.6 options: (a) Rescale continuation and (e) Rescale summary job | removed options | WRAPUP 8.6 (`316714c`) | Removed by the user directive of 2026-09-30 08:45 (no further Rescale cost); option (b) local merge into the primary verdict was already withdrawn by LG revision 14 (4) after cross-environment check (i) | LG revision 15 (a) (`ff1be02`); revision 14 (`028900a` 04:19:28) | (a), (e) after T_res; (b) before |
| 19 | AB10 resampling count "2,000" | corrected | WRAPUP 1.1 (`316714c`) | Corrected to 1,000 (LGU revision 2, p resolution 1e-3); the WRAPUP 1.1 body keeps the old text | LGU revision 2 (`582be9a` 03:36:00); WRAPUP revision 2 (b) (`ff1be02`) | LGU change before; WRAPUP record after T_res |
| 20 | LGU candidate sentence "labels correct the interval location but do not reduce its width" | dropped sentence | LGU review document | Not used; width statements belong to LGX L25 | LGU initial (`a112cef` 2026-09-29 23:30:02) | before |
| 21 | Label "revision after retrieval, bundles not unpacked and not viewed" | status label | WRAPUP 0.3 fourth item | Four amendments (LG revision 15, LGF revision 6, LGU revision 4, WRAPUP revision 2) were committed after T_res (07:20:22) in `ff1be02` (11:44:30). WRAPUP 0.3 would label result-dependent rules "registered after viewing results". The amendments change only technical corrections, a map-mask interpretation, the status of auxiliary local runs and the licence-verified data rule; at commit time no bundle was unpacked and no result table was created or opened. They therefore carry the label above instead. The user has not yet confirmed this judgement (execution plan decision 11); if the user decides otherwise, the affected rows carry "registered after viewing results" | LG revision 15 (l); WRAPUP revision 2 (g) (`ff1be02`) | after T_res; LGX bundles unpacked at 11:44:39 |
| 22 | LG revision 15 (b) statement of the remaining nn FT-Transformer shards (16 main plus 35 others; Greenland has no unit) | corrected record | LG revision 15 (`ff1be02` 11:44:30) | After unpacking, the 51 shards were 16 main, AL-1 4, six targets × 5 = 30, and Greenland split 2 = 1; the Greenland sentence was wrong | LG revision 15 post-unpack note (`1a86df0` 2026-09-30 11:46:13) | after unpacking (file names and status fields only) |

---

## M5. Equivalence margin and pre-result calculations

### M5.1 Methods sentences: equivalence margin (paste-ready)

> Equivalence was judged on the difference in RMSE between two methods with a primary margin δ = 0.5 cm and a secondary margin δ = 1.0 cm. A contrast was called equivalent when neither the superior nor the inferior verdict applied and all four limits of the cell-weighted and block-equal confidence intervals lay within [−δ, +δ]. For methods that scale a learned correction by λ, equivalence required equivalence at both the reference λ and λ = 1.0. Both margins were fixed on 2026-09-29 (commit 8513cc2, 18:17:29 KST), before any result of the label-grid runs was retrieved, and were not changed afterwards.

> The margins are conventions and were not derived from measurement error. After fixing them, we checked two reference quantities. In the ABoVE validation data, 92.4 % of 21,957 probe ALT values are whole centimetres, which indicates a reading unit of about 1 cm. Under an earlier protocol with an Alaska-only source, the RMSE of the source-coefficient Stefan model was 21.6–42.5 cm in the four main target regions, so 0.5 cm and 1.0 cm correspond to 1.2–2.3 % and 2.4–4.6 % of that error.

> As a sensitivity rule we also report a relative margin δ_rel = 0.02 × RMSE(P0), fixed on 2026-09-30 (commit 316714c, 03:02:42 KST). Rows whose verdicts differ between δ = 0.5 cm, δ = 1.0 cm and δ_rel are labelled "margin-dependent". The primary verdict is always the δ = 0.5 cm verdict. A superior or inferior contrast whose point estimate is smaller than 0.5 cm in absolute value is labelled "statistically distinguishable, magnitude below 0.5 cm".

Disclosure sentence (recommended, for Methods or SI): 

> The margins were set after earlier experiments in this project had reported effect sizes of the same order (for example, a mean full-label improvement of −0.68 cm in the four main regions under the M1 protocol). They were therefore not chosen without knowledge of typical effect sizes, but they were fixed before any result of the LG, LGX, LGT, LGU, LGD or LGF runs was retrieved.

### M5.2 SI text: evidence for the margin and its level

> Measurement noise does not set the margin. If a noise term with standard deviation σ_e is independent of the predictions, RMSE_obs² = RMSE_sig² + σ_e², and for small differences Δ_obs ≈ (RMSE_sig / RMSE_obs)·Δ_sig. For σ_e = 3–8 cm and RMSE = 15–40 cm, this ratio is about 0.85–1.00. The margin on the observed scale is therefore close to the margin on the signal scale, and measurement noise does not make a 0.5 cm difference undetectable. The within-cell standard deviation of ALT (11–18 cm) describes the uncertainty of one label, not a limit on RMSE differences between methods, and is not used to set δ. We rate the support for the margin as weak (convention).

**Table S-M1. Reference quantities for the equivalence margin** (WRAPUP 9.2; checked 2026-09-30 before any LG-family retrieval)

| Quantity | Value | Source | Level of confirmation |
|---|---|---|---|
| Probe reading unit | 92.4 % of 21,957 ABoVE probe ALT values are whole centimetres | ABoVE V2 raw data (`ABoVE_Soil_ThawDepth_Moisture_Validation_V2.csv`) | computed from data; reading protocol document not found |
| CALM probing protocol | graduated metal rod about 1 cm in diameter and about 1 m long, pushed to refusal; usually two readings averaged per node; reading unit not stated | GTOS 62 T7 (Smith & Brown, Rome 2009) | original document checked; reading-unit sentence not found |
| ABoVE ALT_err | probe: 25th, 50th and 75th percentiles all 3.0 cm over 12,653 rows (appears to be a fixed value); GPR: median 8.3 cm over 192,548 rows (interquartile range 6.3–10.3 cm) | ABoVE V2 raw data | computed from data; definition of ALT_err not found |
| GPR accuracy | "not fully known, appears to be within ±15 % in fine-grained soils" | GTOS 62 T7 | document checked |
| Within-1 km SD of ALT | Alaska 11.3 cm, Lena 13.0 cm, Canada 17.7 cm | input table v3 (WRAPUP 2.2) | computed from data |
| Within-cell point SD and inter-annual SD | within-cell point SD (n_obs ≥ 2), rms: Alaska 6.4 cm (median 2.3 cm), Canada 16.7 cm (median 10.8 cm); CALM inter-annual SD, median: Alaska 5.2, Canada 5.9, Russia W 9.4, Russia E 6.0 cm; standard error of multi-year means, rms 1.6–6.0 cm | `data/processed/m1/l2_noise_floor.csv` (2026-09-21) | earlier internal table |
| Ratio to P0 RMSE | source-coefficient Stefan RMSE under the M1 protocol (Alaska-only source): Lena 21.6, Canada 26.5, Russia W 42.5, Russia E 28.4 cm; δ 0.5 cm is 1.2–2.3 % and δ 1.0 cm is 2.4–4.6 % | same table | earlier internal table; the LG P0 uses a different source, so its RMSE differs |
| LGX-N2 error floor | available after LGX aggregation | `lgx_floor.csv` | not used (after results) |

Not confirmed: the CALM reading unit in the original protocol text, the definition of ABoVE ALT_err, and the LGX-N2 floor.

### M5.3 Pre-result calculations: common facts (Methods or SI)

> Three calculations that do not use model predictions of the label-grid runs were registered together with their interpretation rules on 2026-09-30 (WRAPUP sections 2.4 and 3.5; commit 316714c, 03:02:42 KST). They were first computed at 03:25–03:35 KST. After a code review, the corrected aggregator re-ran them at 04:08 KST, and a full-table comparison showed identical values. At both times no LG or LGX result folder existed (checked at 03:35 and 04:07 KST). The outputs were committed at 04:19:28 KST (commit 028900a), before the first result retrieval at 07:20:22 KST. The calculations used only the input label table, the soil thaw table, the sub-region map, the CALM raw data and the split and draw functions of the label-grid harness (h39 modes sens-unit, sens-year and sc3; one thread; Python 3.9.18, NumPy 1.26.4, pandas 2.1.4, SciPy 1.11.4; h39 first version SHA-256 prefix 325f28c224a90284, corrected version 51a18650052e6b2d).

### M5.4 S-a: label unit (point labels versus 1 km location means)

Methods text:

> S-a compared recalibration with point labels and with 1 km location means in Lena, Canada and Alaska (mode x). We used split seeds 1–200 with the split rules of the label grid (valid splits: Lena 197, Canada 200, Alaska 200), five draws per split and n ∈ {3, 10, 40}. In the row condition, n label rows were drawn from the target half A as in the label grid. In the location-mean condition, the rows of A were first averaged within 1 km locations (mean ALT and mean √TDD), and n locations were drawn. In both conditions, E_ls was the slope through the origin of the drawn labels and E_n = (n E_ls + 10 E0)/(n + 10). Both conditions were scored on the same B-half rows. For each split we averaged Δ(P1 − P0) over draws and computed Δ_unit = Δ_location-mean − Δ_row. In Canada at n = 40, the location-mean condition was skipped in the 100 splits where A had at most 40 locations.

Results text (SI):

> Median Δ_unit [10th, 90th percentile] was +0.31 [−0.16, +1.37] cm (n = 3) and +0.34 [−0.26, +1.99] cm (n = 10) in Lena, −0.47 [−1.76, +0.81] cm and −0.86 [−3.48, +1.05] cm in Canada, and +0.07 [−0.19, +0.41] cm and +0.09 [−0.36, +0.58] cm in Alaska. At n = 40 (descriptive), the medians were +0.56 cm (Lena), −0.96 cm (Canada, 100 splits) and +0.08 cm (Alaska). Following the registered rule, the Canadian median at n = 10 (below −0.5 cm) gives the sentence: "With point labels, the gain from few-label recalibration was smaller than with 1 km location means (Canada, n = 10, −0.86 cm)". The deployment guideline therefore states its label unit as "1 km location mean". Lena and Alaska stayed within ±0.5 cm at n ∈ {3, 10} with the opposite sign, so the sentences are written per region. No median exceeded +0.5 cm at n ∈ {3, 10}. At n ∈ {3, 10}, the 10–90 % intervals of Lena and Canada did not overlap the Russia W reference interval in either condition, so the sentence "part of the between-region difference in recalibration gain is explained by the label unit" is not written. With B also averaged to 1 km locations (auxiliary), the medians at n = 10 were −0.65 cm (Canada), −0.17 cm (Lena) and −0.02 cm (Alaska).

Viewing and interpretation labels for S-a (to state in the SI):

1. The registered plan stated the n range {3, 10} only for the first rule. The same range was applied to the second and third rules after the first calculation (03:25 KST). This range is an interpretation fixed after the values were seen (h39 convention 9). If the second and third rules were extended to n = 40, Canada at n = 40 would give the same sentence, and Lena at n = 40 (+0.56 cm) would add the opposite sentence ("with 1 km location means, the gain was smaller than with point labels").
2. S-a computed the split distribution of the analytic P1 − P0 contrast (and P2 − P0) for Lena, Canada and Alaska (n ∈ {3, 10, 40}) and for Russia W (reference, n ∈ {3, 10}). This is the same quantity as LGX-N4a (splits 1–200, five draws, analytic P0, P1, P2), with different draw seeds only. It has the same form as AB4 and SC1w-P (different draw seeds and split range). The person running the calculation saw the regional summaries and the Russia W 10–90 % interval. See label text in M5.7 item 1.

### M5.5 S-b: single-year labels in Russia W and Russia E

Methods text:

> S-b replaced the multi-year CALM site means of Russia W and Russia E by single-year values. From the CALM raw data (PANGAEA 972777; 3,819 rows), we removed values with inequality signs (24 ">" and 10 "<"), inactive rows (117) and empty values (122), leaving 3,663 rows (the conditions can overlap). Label cells were matched to CALM sites by coordinates within 0.005° in latitude and longitude. Of the 31 Russia W cells, 27 matched one site, 4 matched several sites and none was unmatched. Of the 30 Russia E cells, the counts were 27, 3 and 0. Cells with multiple matches kept their multi-year means. For the 54 single-match cells, the multi-year mean of the valid years reproduced the label value (maximum absolute difference below 1e-13 cm). The median number of valid years was 13 (Russia W) and 20 (Russia E). For split seeds 1–200, five draws and n ∈ {3, 10}, one observation year was drawn for each drawn site, E_ls and E_n were computed from the single-year values, and scoring used the multi-year means of the B half. Δ_year = Δ(P1 − P0)_single-year − Δ(P1 − P0)_multi-year was paired within split and draw.

Results text (SI):

> The share of drawn labels replaced by a single-year value was 0.88 (Russia W) and 0.90 (Russia E). Median Δ_year [10th, 90th percentile] was +0.02 [−0.62, +0.63] cm (n = 3) and +0.05 [−0.48, +0.52] cm (n = 10) in Russia W, and +0.01 [−0.19, +0.21] cm and +0.04 [−0.25, +0.28] cm in Russia E. Both medians stayed within ±0.5 cm at both n, so the registered sentence applies: "The recalibration gain from 3–10 labels was maintained with single-year labels."

Labels for S-b: the plan fixed only the seed for the year choice; the seed for the row draw (`seed_of("lgw-sb-row", target, split, n, draw)`) was set by the implementation (h39 convention 3). No viewing label is recorded for S-b. The split output holds only Δ_year and the replacement share (checked in the h39 code, not in the table). The run metadata records the source E0 of Russia W and Russia E.

### M5.6 SC3w: sequential stopping rule

Methods text:

> SC3w tested a sequential rule that decides from the target labels alone whether to stop at 3, 10 or 40 labels. For each valid split (1–5) and each of 20 repetitions, a random permutation of A fixed nested label sets of 3, 10 and 40. At 3 and 10 labels, the jackknife standard error of log E (leave-one-label-out least-squares slopes before shrinkage) was compared with τ = 0.05. The rule stopped when the standard error was at most τ and otherwise continued, with 40 labels as the cap. The stopped stage gave E_n (κ = 10) and P1. The comparison was P1 from the first 40 labels of the same permutation. Δ_SC3 = RMSE(P1_rule) − RMSE(P1_40) was scored on the B half with 10,000 block bootstrap resamples, and the label ratio was the mean number of labels used divided by 40. Support required (i) label ratio ≤ 0.5, (ii) a three-region mean point estimate ≤ 0.3 cm, (iii) both CI upper limits of the three-region mean < 0.5 cm and (iv) no inferior region. A label ratio ≥ 0.9 was classified as "rule not operating". Values τ ∈ {0.025, 0.10, 0.20} were descriptive sensitivities.

Results text (SI):

> At τ = 0.05 the rule was classified as not operating. The three-region mean label ratio was 0.909 (condition (i) failed; conditions (ii)–(iv) held). The registered sentence is: "At the main τ of 0.05 the stopping rule rarely operated (label ratio 0.91)". By the registered rule, Δ_SC3 is descriptive and the rule is not part of the deployment guideline. The shares of repetitions stopping at 3, 10 and 40 labels were 5, 0 and 95 % in Lena, 7, 0 and 93 % in Canada, and 11, 8 and 81 % in Alaska. The three-region mean Δ_SC3 was −0.048 cm [−0.079, −0.002] (block-equal −0.077 cm [−0.105, −0.049]), no region was inferior, and each region was equivalent. For τ = 0.025 the label ratio was 0.988 (not operating). For τ = 0.10 it was 0.542, and only condition (i) failed. For τ = 0.20 it was 0.185 and all four conditions held (Δ_SC3 −0.677 cm [−0.909, −0.353]); this value is descriptive and does not change the verdict. In descriptive targets, Russia W (cap equal to its A size, 14–16 labels) was inferior at τ = 0.05 (+0.29 cm [+0.16, +0.40]), and among the ten sub-regions in mode x only AL-2 was inferior (+0.04 cm [+0.02, +0.18]).

Viewing labels for SC3w (to state in the SI):

1. The expected-range text in WRAPUP 3.5 ("most repetitions will reach 40 labels") was written after the label statistics of one Canadian split had been viewed (coefficient of variation of the block-mean ratio ALT/√TDD in Canada, 0.22; WRAPUP 0.1). τ was not changed after this estimate.
2. The blinding label is "partially unblinded": the b2 shrinkage curves (κ = 10) and the Canadian shrinkage results in RESEARCH_FRAME A.2 had been seen. The regional non-inferiority condition (iv) was added so that Canada could not mask degradation elsewhere.
3. The full SC3w table with the P0 contrasts and RMSE columns is sealed (`data/processed/lgw/sealed/lgw_sc3_full.csv`) until the LG retrieval. For Russia W and Russia E, the cap−P0 rows equal the LG contrast P1 − P0 at full labels (same formula, splits and scored cells). For the other targets, the cap−P0 rows have the same form as the LG contrast P1 − P0 at n = 40 (different draws). The fix-1 run did not open these rows. The record of the first run does not state whether they were displayed. This remains unresolved and is reported as such (M5.7 item 3).

### M5.7 Viewing labels to attach to later results

1. **S-a label** on the Lena, Canada, Alaska (x) and Russia W cells of AB4, SC1w-P, the LGX-N4a split distribution and region inference, L31, and the WRAPUP 1.6 descriptive column (P1 − P0 at n = 10): "In the S-a calculation (2026-09-30 03:25 KST), the split distribution of the same analytic P1 − P0 quantity, differing only in draw seeds, was viewed." Any other work that opens `lgw_sens_unit.csv` adds the same label.
2. **Result existed at registration (not viewed)** on AB4 and SC1w-P (LGT shards contained P0 and P1 values from 01:47:33), on LGT L32 sentences (WRAPUP 1.5) and on δ_rel rows for LGT (WRAPUP 9.4).
3. **SC3w sealed rows** on LG P1 − P0 at full labels for Russia W and Russia E, and on P1 − P0 at n = 40 for Lena, Canada and Alaska (x): "The same or an equivalent quantity was computed in the SC3w pre-result calculation; whether it was displayed in the first run is not recorded."
4. **S-a interpretation label** on the second and third S-a rules: "n range {3, 10} fixed after the first calculation."
5. **Cross-environment label** on any local `lgx_lg_aux.csv` and the columns derived from it (`lgw_bundle.csv` h42_verdict4, resample_dependence): "cross-environment reproduction not confirmed", unless a Rescale table with the same content passes the 1e-6 cm comparison (WRAPUP results addendum 1 fix 1).

---

## Record: inconsistencies found in identifiers and commit records

1. Commit `316714c` (2026-09-30 03:02:42) names the abstract bundle "B1–B10" in its message. The committed document uses AB1–AB10 (renamed in WRAPUP revision 1 (8) before the commit). The WRAPUP initial-draft entry in section 12 also uses the pre-rename names SC1–SC3 and AK1.
2. Commit `6799b74` (2026-09-29 22:43:12) calls its LGX pre-check record "LG revision 9". In the document this content is revision 11 (renumbered in `cec5719`, 23:29:45); revision 9 is the LGT section and revision 10 the LGD section, both first committed in `6799b74`. LG revision 11 records this.
3. LGU revision 3 is dated "2026-09-30 03:50", but it first appears in commit `582be9a`, committed at 03:36:00.
4. LGF revision 5 is dated "05:05 KST", but it was committed in `b1a4349` at 05:00:40. The LGF window deadline (2026-10-02 05:01) implies a window start of 05:01, consistent with the commit time rather than the label.
5. Two amended commits have different author and committer times: `ff1be02` (11:44:18 and 11:44:30) and `028900a` (04:19:23 and 04:19:28). The execution plan §10 cites 11:44:18; this document uses committer times.
6. LG revision 15 (b) described the remaining nn FT-Transformer shards as 16 main plus 35 others and stated that Greenland had no unit. The post-unpack check (`1a86df0`, 11:46:13) found AL-1 4, six targets × 5 and Greenland split 2 (1 shard). The execution plan 1.1 and 6.1 (b) keep the pre-correction text.
7. WRAPUP revision 2 (g) and LG revision 15 (l) apply the label "revision after retrieval, bundles not unpacked and not viewed" to four amendments. LGU revision 4 and LGF revision 6, in the same commit `ff1be02`, are headed "before viewing results" and do not carry this label.
8. WRAPUP 1.1 still states 2,000 LGU resamples for AB10, and WRAPUP 1.4 (b) still states that H18–H22 lack block-equal CIs and that the H20 predictions were not found. The corrections exist only in WRAPUP revision 2 (b) and (c), because sections 1–11 are frozen. The manuscript must use the corrected values.
9. WRAPUP 3.5 gives the main-4 P0 RMSE under the M1 protocol as "about 14–42 cm", while WRAPUP 9.2 gives 21.6–42.5 cm for the same four regions from the same table.
10. The "72–95 %" decomposition claim, rejected in NOVELTY (`d8355d2`), is still written as confirmed in `docs/EXPERIMENT_PLAN_FINAL_PAPER_2026-09-26.md` (line 272), `SESSION_HANDOFF.md` (line 42), `docs/EXPERIMENT_LOG.md` (line 40) and `docs/RESEARCH_FRAME_2026-09-29.md` (line 298). The correction is scheduled as execution plan J10.
11. NOVELTY 7.4 (committed in `d8355d2` at 18:18:36) states that LG revision 5 was not yet committed. Revision 5 had been committed 67 s earlier in `8513cc2` (18:17:29).
12. Identifier collisions: the FINAL-plan hypotheses F1–F11 (used as F4–F9 in the h39 reclassification inputs) share letters with LGF-F1–F6. WRAPUP prefixes LGX-N, LGF-N and NOVELTY-N but not the FINAL F series. This document uses FP-F.
13. LGU-C1–C4 and LGF-N4 are listed among hypothesis ids (execution plan 1.4, WRAPUP identifier note), and LGU-A2, A5, A6 and B5 sit in the LGU hypothesis tables, but all of them are implementation checks, exploratory, descriptive or diagnostic outputs without a verdict. The SI should not count them as tested hypotheses.
