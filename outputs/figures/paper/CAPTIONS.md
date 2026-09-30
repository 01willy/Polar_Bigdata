# Figure captions

## Fig2_recipe_decomp

Fig. 2 | Recipe and gain decomposition in labelled regions. ΔRMSE is method RMSE minus comparator RMSE on held-out B blocks (cm); negative is better. The comparator is the physics anchor (Stefan, E0) unless the axis names a control. Recipe: Stefan anchor plus CatBoost residual with weight λ. Parentheses give |log(E_own/E0)|; AL, CA and LE denote subregions of Alaska, Canada and Lena. Error bars are unadjusted 95 % bootstrap CIs from 1,000 resamples of scoring blocks within a region, paired by seed, with splits pooled; regions with fewer than 8 blocks have no CI. Means (a, b, d) combine the regional distributions of Lena, Canada, Russia W and Russia E, stratified by region. Significance follows Holm-adjusted p (Supplementary Table 2), not CI overlap with zero. Panel c shows point estimates averaged over 1–3 splits, without CI. a, Stefan pseudo-label augmentation versus five controls (covariates-only, direct CatBoost); grey dots, single regions. b, Recipe per region (M1 H13); upper points λ = 0.25, lower λ = 0.5; Russia C and Greenland have fewer than 8 blocks. c, Gain with all A-block labels split into level (E shrinkage, κ = 10) and structure (residual on the shrunk-E anchor) parts; segment, structure part (solid, gain; dashed, loss); open chevron, its sign where shorter than 0.5 cm. Grey/white marks the pre-classified |log(E_own/E0)| ≥ 0.15 / < 0.15 threshold only; it does not indicate which part is larger. Russia W is on the broken axis. d, Filled, cell-weighted; open, block-equal. Registered nested selection (H28) includes zero (H28 rejected). Pre-specified λ = 0.25 remains significant after Holm adjustment over eight H28 contrasts; λ = 0.5 does not. The post hoc best recipe is chosen on target scores (reference only). Source: data/processed/h4/a1_ladder.csv, m1/m1_sc_tests.csv, h4/a1_decomp.csv, h3/h28_tests.csv. Scoring cells per region: Supplementary Table 1; cell-weighted and block-equal values: Supplementary Table 4.

<!-- words: definition 64, statistics 76, panels 142, data 30, total 312 -->

## Fig6_label0

Fig. 6 | Zero-label regions. ΔRMSE, method RMSE minus physics-anchor RMSE (Stefan model, Alaska E0), cm; negative is better. AB4, unweighted mean over Lena, Canada, Russia W and Russia E. H18, Stefan forcing from MODIS LST or ERA5-Land soil temperature; H19, residual ML on reduced covariate sets (x14, terrain and climate plus new groups; x16, without SoilGrids); H20, E from CCI thaw depth; H21, E borrowed from labelled regions; H22, invariance or domain-adversarial objectives; H23, meta-learning. c, d score 3,760 cells in 93 blocks (4 AB4 regions). All intervals: 95 %, 1,000 bootstrap resamples; a, stratified over regions (H18 to H22); c, d, scoring blocks within regions (at least 8 blocks), averaged across regions by resample index. Holm-adjusted p: Supplementary Table 2. F10 partly supported: cdf coverage 0.86 [0.80, 0.92], width 81 cm; 4 of 6 regions within 0.85–0.95 (Russia W 0.75); pool 0.73. a, Solid and dashed intervals: no target covariates and target covariates only; residual ML uses λ = 0.25; H23 is a point estimate on its own evaluation blocks. b, Filled marker, AB4 mean RMSE; tick, worst AB4 region (Russia W for every rule); dotted lines, physics values; AOA, area of applicability; CCI gate, CCI where within 20 cm of physics. c, Filled and open markers, cell-weighted and block-equal AB4 means; light points, single AB4 regions. F9 not supported: 0 of 4 CCI combinations Holm-significant (uncertainty weight −0.43 [−0.84, 0.04] cm). d, Leave-one-region-out coverage versus mean width; grey band, 0.85–0.95. Filled, two-level hierarchical conformal: cdf, region CDF pooling (pre-specified primary); sub, one-cell-per-region subsampling; blk, 0.5° blocks as groups. Open grey, references without CI: pool, cell pooling; H17 CQR and anchor-residual (anc) intervals, Alaska-calibrated (ak) or density-ratio weighted (iw). Omitted: exact hierarchical (infinite width), weighted conformal (uses target labels). Source data: data/processed/h2/h_tests_all.csv, h2/h23_summary.csv, h2/h_deploy_gating.csv, h3/h30_deploy_worst.csv, h4/c1_tests.csv and h4/c2_coverage.csv.

<!-- words: definition 75, statistics 71, panels 147, data 26, total 319 -->

## Fig5_label_placement

Fig. 5 | Label placement. Block label value is ΔRMSE on B blocks (method minus physics anchor, cm; negative is better) when only one A block is labelled (E shrinkage κ = 10 plus residual ML, λ = 0.25), averaged over splits for each unique block. Rules: k-medoid, weighted k-medoid (cluster-size weights), D-optimal (√TDD leverage), Active PPI; random is the reference. a, b: hatching, ΔRMSE > 0 by sign only, not significance; harmful fraction with Clopper-Pearson 95 % CI. c: V_tot = between-split variance plus block-bootstrap variance of E (100 resamples of pooled A blocks, rule re-applied). F7 not supported: D-optimal cut V_tot by ≥ 30 % in 2 of 8 primary targets (criterion 6). d: target points, block-bootstrap estimates (1,000 resamples, paired); pooled means, stratified block-bootstrap 95 % CIs with the 4 main regions or the 10 subregions as strata. a, Canada: 9 of 33 blocks harmful (27 %, 95 % CI 13–46 %). b, Lena Delta: 10 of 18 blocks harmful (56 %, 95 % CI 31–78 %). Polar stereographic, true scale 70°N; circle area ∝ labelled cells (clipped at 27 and 200); circles displaced to avoid overlap (31 moved, ≤ 2.2 mm), grey lines join 6 to true centroids (dots). Symlog colour scale, linear within ±2 cm. Circles at labelled-cell centroids; 4 in b fall on delta channels (coastline: Natural Earth 1:50 m in a, 1:10 m in b). Black outline: block receiving ≥ 0.5 of 10 k-medoid labels (mean). c, n = 3; blue, 8 targets with block-level bootstrap (primary set); grey, 4 targets with cell-level fallback; AL-1 and AL-6 lack V_tot. Dotted line, pre-specified 0.7 criterion. d, n = 10, 14 targets; smaller markers, subregions; D-optimal minus random, main regions +1.10 [0.29, 2.26] cm. Source: data/processed/h3/h29b_blocks.csv, h29b_summary.csv; data/processed/h4/b3_evar.csv, b3_f7.csv, b3_summary.csv, b3_region.csv. Per-target values in Supplementary Table 3.

<!-- words: definition 58, statistics 80, panels 149, data 25, total 312 -->

## Fig7_workflow

Fig. 7 | Deployment procedure and its confirmatory tests. ΔRMSE, method minus physics-anchor RMSE (E0 from source regions) on held-out B blocks; negative is better. Error type, oracle |log(E_own/E0)|: level ≥ 0.20, structure < 0.15. E0 + res., anchor plus CatBoost residual ML (λ 0.25); Shrink, E shrunk toward E0 (κ = 10); Re-fit, unshrunk E; S3, shrink plus residual ML; Protocol, the dashed rule in a. † Exploratory (post hoc). Under 5 scoring blocks: AL-1, AL-3, AL-4, AL-6, CA-2, CA-3, LE-1, LE-2. b, 14 targets, 3 splits × 20 repeats × 2 seeds, k-medoid labels (nested at n = 10); no CI on means. c, 95 % block-bootstrap CI (1,000 resamples of scoring blocks within split). F4 (pre-registered: fewer worsened targets than S3, mean Δ within 0.5 cm of the oracle branch) not supported: at n = 3, 7 of 14 targets worsen (4 with block CI above 0) versus 6 for S3; mean Δ −0.50 cm, 0.66 cm above the oracle. a, Solid arrows, recommended; dashed, pre-registered branch (b). Tags, expected Δ, evidence; 3–10 tag, range of S3 target means in b (n = 3, 10) per error type, without targets worse by > 1 cm. Bottom leaf: 6 of 8 structure targets lack n* ≤ 40 (A2, E0 fixed); AL-5, Canada beat physics at n = 0. b, τ per target by leave-one-target-out. Filled, mean; tick, worst target; arrowheads, off-scale (37.0, 11.0 cm). Bars, upper n = 3. c, D1 holdout (Mongolia, Central Asia; 46 cells, 21 blocks); Δ as % of physics RMSE (270.6 cm). Target labels, ground-temperature thaw depths (1990–2011); training labels, direct probing (ABoVE, Lena: 99 % of 17,467 cells); region and label definition confounded. Bold row, pre-registered prediction Δ < 0. F11 supported with this caveat: protocol n = 3 (τ 0.10–0.20 identical), −52.8 [−56.9, −49.2] cm. CI hidden by symbol: E0 + res., n = 0, −5.3 [−6.0, −4.6] %. Source: data/processed/h4/b1_protocol.csv, b1_summary.csv, b1_meta.json, c2_coverage.csv, a1_recipe_table.csv, a2_minn.csv, d1_summary.csv; outputs/figures/paper/workflow_tags.json. τ per target: b1_summary.csv tau_loto, rule b1_meta.json. Per-target values: Supplementary Tables 4, 7.

<!-- words: definition 80, statistics 77, panels 150, data 40, total 347 -->

## Fig1_problem

Fig. 1 | Regional Stefan coefficients and transfer error without target information. The physics anchor predicts active-layer thickness (ALT) as E·√TDD, with TDD the ERA5-Land thawing degree-day sum and E fitted on the source region (Alaska; E_AK in b). Table 1's E0 is a different, leave-region-out fit. ΔRMSE = direct-ML RMSE minus physics-anchor RMSE on the same cells (cm); negative favours ML. Region numbers follow Table 1. b, E = exp(mean z), z = log ALT − log √TDD per cell; 95 % CI of the mean from 1,000 resamples of 0.5° blocks for regions with ≥ 8 blocks, boxes for ≥ 10 cells; Russia C (7 cells) and Greenland (3 cells): mean only. Table 1's least-squares |log(E_own/E0)| ranks regions 2–5 as Canada < Lena < Russia E < Russia W. d, CIs resample (split, repeat) rows for regions and strata of blocks for the mean. a, 17,464 label cells in 183 blocks of 0.5° (17,467-cell population minus 3 single-cell regions); circle area proportional to cells per block (bounded at 6 and 200). Region 3 has mainland and Arctic Archipelago clusters. Dark circles: external holdout 8 (46 cells, thaw depth derived from ground temperature). Triangle: 6 of its blocks (13 cells) beyond the frame. Background: ESA CCI Permafrost PFR v4.0, mean of 1997–2021 (ERA5-based to 2002, MODIS LST CryoGrid from 2003), 0.1°; polar stereographic, true scale at 70°N. b, cells (subsampled to ≤ 500 per region; 20 off-axis omitted), interquartile boxes and means, ordered by |log(E/E_AK)|. c, training data and scored cells per condition; A and B are halves of the target 0.5° blocks (three splits); hatching, physics pseudo-labels on A-block covariates. d, CatBoost trained on Alaska, three seeds, all target cells; cell counts in Supplementary Table 1. Source data: data/processed/fidelity_base_v3.csv (polar.m1_core.load_base), data/processed/cci_pfr_mean_1997_2021.nc, data/processed/m1/m1_sc_tests.csv (X-model, noinfo, catboost direct minus Stefan anchor).

<!-- words: definition 67, statistics 78, panels 148, data 30, total 323 -->

## Fig3_label_budget

Fig. 3 | Label-budget staircase and partial pooling of the scaling coefficient E. ΔRMSE, method minus physics-anchor RMSE (E0 from source regions, n = 0) on held-out B blocks, cm; negative is better (0 line, physics anchor). E shrinkage: target least-squares E shrunk toward E0 with κ = 10; augmentation and residual ML use this E as anchor. All A-block labels is a reference, not an upper bound. F5 not supported: no adaptive pooling estimator improves on fixed κ = 10. a–d: means over 3 splits × 20 label draws (E shrinkage) or 3 splits × 5 draws × 2 seeds (augmentation, residual ML); e: 3 splits × 10 draws per target, averaged over targets at n = 3, 5, 10 (14, CA-1 excluded), 20–160 (12) and 320 (9). All intervals are 95 % block-bootstrap CIs (1,000 resamples of scoring blocks within each split, paired across methods and repeats); in e, averaged index-wise across targets. a–d, Main regions, |log(E_own/E0)|; header fill, error type (none, structure < 0.15; grey, level ≥ 0.20; hatched, intermediate). y symmetric-log, linear within ±2 cm. Right whiskers, reference split range; T-ticks, largest tested n; grey shading in axes, n above label pool. Residual ML CIs: filled band, k-medoid; dashed outline, random. In Lena, residual ML improves at n ≥ 20 with random labels (−1.38 to −0.62 cm) but worsens at n = 10–80 with k-medoid (+0.6 to +1.2 cm). e, Target-stratified mean of four E estimators (anchor only, random labels), broken at target-set changes. κ = 10: −0.62 [−0.74, −0.37] cm at n = 3, −1.16 [−1.44, −0.77] cm at n = 10, lowest at every n, but CIs overlap from n = 5 (margin 0.00 cm at n = 160). Offset MLE, +0.33 [−0.10, 1.15]; PPI++, +1.37 [0.86, 2.35] cm at n = 3. Arrows, mixed-effects boosting (+5.3 to +10.4 cm, off scale). Source data: data/processed/h3/h25b_curve.csv (a–d); data/processed/h4/b2_regional.csv (e; vs = phys, region_set = ALL, variant = E, rule = random).

<!-- words: definition 77, statistics 72, panels 148, data 24, total 321 -->

## Fig4_min_labels

Fig. 4 | Minimum target labels for residual learning, E treatment separated from label placement (A2). ΔRMSE, method minus physics-anchor RMSE (E0 from source regions, n = 0) on held-out B blocks (cm); negative is better. n* is the smallest n at which the block-bootstrap 95 % CI upper bound of ΔRMSE is below 0, at least two of three splits improve, and at least 75 % of repeats improve (pre-specified). Residual ML: CatBoost, λ = 0.25. Targets: 14 (CA-1 excluded, |A| = 3). 95 % block bootstrap CIs: 1,000 resamples of scoring blocks within each split, paired; 3 splits × 10 label draws × 2 seeds. CIs shown for targets with at least 8 scoring blocks across splits. †, a split with fewer than 5 scoring blocks (Supplementary Table 3). Reached (7 of 14) versus not: Mann-Whitney U exact p = 0.71; censoring-aware Kendall τ_b(|log(E_own/E0)|, n*) = −0.13 (p = 0.58). F1 supported: 6 of 8 structure targets lack n* ≤ 40. a–d, ΔRMSE against n, all-block placement: a, Russia W (level error); b, Lena; c, Canada; d, AL-2 (structure error). Parentheses: |log(E_own/E0)|, two decimals. Shared symmetric-log y axis, linear within ±2 cm. Filled or open: n* criteria met or not at that n. n = 0: source residual only. Band, bar: E0-fixed CI. Dotted: E_own known. Smaller, x-offset markers: E fitted to labels. Grey: n untested. a, table: ΔRMSE (cm) at n = 3 and 10, where markers hide the E_own line. e, E0-fixed n* against oracle |log(E_own/E0)| (square-root axis); ring, met at n = 0; band, not reached by the largest tested n, n_max (open triangle, n_max < 40). Structure < 0.15 ≤ intermediate (hatched) < 0.20 ≤ level. f, spread minus concentrated at n = 10 and 40: diamonds, target mean with CI (14 and 12 targets); dots, targets. Symmetric-log x axis, linear within ±1 cm. Source: h4/a2_curve.csv, a2_minn.csv, a2_spread.csv; |log(E_own/E0)|: h3/h25b_targets.csv (least-squares E). n* for other E treatments: Supplementary Table 5, Supplementary Fig. 12. n = 40 excludes Russia W and E (n_max 10).

<!-- words: definition 80, statistics 78, panels 149, data 38, total 345 -->

## Table1_region_summary

Table 1 | Per-target summary by label regime. E_own is the least-squares Stefan coefficient (ALT on √TDD, zero intercept) of a target's labelled cells; E0, the same fit on source cells over 100 km from the target. Error type uses |log(E_own/E0)|: level ≥ 0.20, structure < 0.15, intermediate between (post hoc). ΔRMSE: method minus physics anchor (E0), cm; negative better. AL, CA and LE: Alaskan, Canadian and Lena subregions (Supplementary Fig. 3), numbered (No., Fig. 1a) by parent region. Brackets: 95 % CIs from 1,000 bootstrap resamples of scoring blocks within each split, paired across methods, repeats and seeds, split-averaged; the recipe pools splits within region. n* (pre-specified) is the smallest tested n (3 to 320) at which the ΔRMSE CI upper bound is below 0, at least two of three splits improve and at least 75 % of repeats improve; '> n': not reached by the largest tested n (no imputed value). Residual ΔRMSE: CatBoost residual (λ = 0.25) trained on source labels only, anchored on E0. Coverage: two-level hierarchical conformal 90 % interval (CDF pooling, pre-specified primary method), leave-one-region-out, shown for six regions (Russia W, Russia E, Lena, Canada, plus label-free-only Russia C and Greenland); Width, mean interval width. n* (H25): E shrunk toward E0 (κ = 10) plus residual ML, labels placed by k-medoid or at random, 3 splits × 5 draws × 2 seeds. n* (A2): E0 fixed plus residual ML on source and target labels, labels spread at random over all A blocks, 3 splits × 10 draws × 2 seeds. Reached: 6, 5 and 7 of 14 targets (CA-1, 3 labels, excluded); criteria are re-tested at each n and can fail after being met at n = 0. Recipe: pre-specified Stefan anchor plus CatBoost residual (λ = 0.25) with all labels (Holm p = 0.004 for the mean). Source data (data/processed): h3/h25b_targets.csv, h4/a2_curve.csv, h4/c2_coverage.csv, h3/h25b_breakeven.csv, h4/a2_minn.csv, h4/a1_recipe_table.csv, m1/m1_sc_tests.csv; fidelity_base_v3.csv (Russia C, Greenland counts).

<!-- words: definition 80, statistics 74, panels 142, data 34, total 330 -->

## Supplementary_index

Status of Supplementary Figures S1–S12 and Tables ST1–ST10. Numbering is fixed and was not renumbered: S10 (C4, InSAR weak-label multi-fidelity) was not performed because of time constraints and keeps its slot as a limitation note, so S11 and S12 keep their original numbers. S11 has no figure; it is the software and run metadata in Supplementary Tables 10a and 10b. S12 is paired with Supplementary Table 5.

- S1: done; files: FigS1_stefan_scatter, TableS9_covariates
- S2: done; files: FigS2_scoring_protocol
- S3: done; files: FigS3_subregions
- S4: done; files: FigS4_subregion_budget
- S5: done; files: FigS5_obs_design, TableS8_h18_h24_tests
- S6: done; files: FigS6_label_number_aux
- S7: done; files: FigS7_tfm
- S8: done; files: FigS8_cci_bias
- S9: done; files: FigS9_label_year
- S10: not performed; files: none (note in FigS10_c4_not_performed); note: C4 InSAR multi-fidelity: not performed (time constraint); limitation described from the literature (Chang 2024; Wendt 2026)
- S11: done; files: TableS10a_software, TableS10b_runs (no figure; software and run metadata only)
- S12+ST5: done; files: FigS12_a2_matrix, TableS5_a2_nstar
- ST1: done; files: TableS1_cells_blocks_E
- ST2: done; files: TableS2_holm_families
- ST3: done; files: TableS3_ci_row_vs_block
- ST4: done; files: TableS4a_b1_protocol, TableS4b_h28_two_scorings
- ST6: done; files: TableS6a_h27_breakeven, TableS6b_h27_rule_stats, TableS6c_c2_coverage_blockeq
- ST7: done; files: TableS7_d1_external

<!-- words: total 231 -->

## FigS1_stefan_scatter

Stefan relation by region. ALT is the probe-measured active-layer thickness label; √TDD is the square root of the ERA5-Land thawing degree-day sum (2015–2020). The slope E of ALT = E·√TDD is fitted two ways: least squares through the origin in linear space and E = exp(mean z), z = log ALT − log √TDD (Fig. 1b). Numbers give E for the two fits (linear / log). Fits use all label cells of the region; points are a random subsample. a to g, Regions in Table 1 order; h, all regions pooled. Axes are clipped at 250 cm; the count of cells above is shown. The 25 covariates are listed in Supplementary Table 9. Source data: data/processed/fidelity_base_v3.csv via polar.m1_core.load_base (17,467 cells).

<!-- words: definition 56, statistics 22, panels 34, data 14, total 126 -->

## FigS2_scoring_protocol

Scoring protocol. Each target's 0.5° blocks are split into A blocks, the pool from which n labels are drawn, and B blocks, on which every method and the physics anchor are scored. Three random A/B splits are used; sources exclude the target and a 100 km buffer. Block bootstrap: within a split, B blocks are resampled with replacement 1,000 times; the same indices are applied to both methods and to all repeats and seeds; RMSE is √(ΣSSE/Σcells) per draw. Draw b is averaged across splits to give the CI of ΔRMSE. Regional means use a stratified block bootstrap. a, A/B blocks. b, Paired block resampling in one split. c, Combination over splits and over the four main regions (AB4). Cell-weighted scoring pools cells; block-equal scoring weights each block equally (Supplementary Table 4b). Implementation: src/polar/h4_common.py (boot_delta_blocks), src/polar/m1_stats.py (strat).

<!-- words: definition 48, statistics 53, panels 35, data 13, total 149 -->

## FigS3_subregions

Subregion definition. Subregions are k-means clusters of scoring-block centroids within each parent region (Alaska k = 6, Canada k = 3, Lena Delta k = 2; square-root cell weights; labels not used), named north to south. Each subregion is a transfer target whose source excludes cells within 100 km. CA-1 (6 cells, 3 A-block labels) is shown but excluded from analysis, leaving 10 subregions and 14 analysed targets with the 4 main regions. Cell and block counts, E_own, E0 and |log(E_own/E0)| are in Supplementary Table 1. a, Alaska. b, Canada. c, Lena Delta. Circles are scoring blocks with area proportional to cells per block; outlines are the smoothed union of 2.6 mm discs around the block centroids of one subregion; detached parts are drawn dashed and 1 mm tighter, carry the subregion name in plain type and are unrelated to any neighbouring outline they approach. Polar stereographic projection, true scale at 70°N; insets show location. Source data: data/processed/h3/subregions.csv; clustering reproduced from data/processed/fidelity_base_v3.csv.

<!-- words: definition 46, statistics 41, panels 70, data 14, total 171 -->

## FigS4_subregion_budget

Label-budget staircase for subregions. ΔRMSE is method RMSE minus physics-anchor RMSE (E0 fitted on source cells, n = 0) on held-out B blocks, in cm; negative is better. E re-fit shrinks the target E toward E0 (κ = 10). All A-block labels is a reference, not an upper bound. Error type uses the oracle in-region E_own (structure < 0.15, level ≥ 0.2, intermediate between). Curves are means over 3 splits × 20 label draws (E re-fit) or 3 splits × 5 draws × 2 seeds (augmentation, residual ML); duplicate splits are counted once. k, l show 95 % block-bootstrap CIs (1,000 resamples of scoring blocks within each split, indices shared across methods, repeats and seeds; splits averaged); panels a to j show no CI. a to j, Ten subregions ordered by |log(E_own/E0)| (in parentheses); symmetric-log y axis, linear within ±2 cm; grey shading marks n beyond the A-block label pool; filled triangles at the axis edge mark off-scale means (range of values shown). k, l, All 14 analysed targets at n = 3 and n = 10 (k-medoid labels); grey band marks level targets, int. marks intermediate targets; CIs are clipped at the axis limits and off-scale means are drawn as filled triangles. Negative ΔRMSE (down in a to j, left in k and l) is better in every panel; the level band and int. mark label the same rows in k and l. Source data: data/processed/h3/h25b_curve.csv. Supplementary Tables 1 and 3.

<!-- words: definition 61, statistics 58, panels 111, data 12, total 242 -->

## FigS5_obs_design

Observation design for sparse labels (H24). ΔRMSE is method RMSE minus physics-anchor RMSE (n = 0) on held-out B blocks, in cm; negative is better. Rules choose which n A-block cells are labelled; the estimator is fixed within a row. Alaska is a sandbox with a five-region equal-weight E0 anchor. Means over 3 splits × 20 draws (E re-fit) or 3 splits × 10 draws × 3 seeds (residual ML); no CI is drawn. H24 was not in the Holm family and is reported as exploratory: the pre-registered gain over random labels held only at n = 3. Hypothesis tests H18 to H24 are listed in Supplementary Table 8. a to e, E re-fit with κ = 10. f to j, Physics anchor plus residual ML. Columns share the log n axis; y axes are independent. Grey shading marks n beyond the tested range; the grey horizontal line is the physics anchor (Δ = 0). Negative ΔRMSE (down) is better in every panel. Source data: data/processed/h2/h24_curve.csv; tests in h2/h24_tests.csv and h2/h_tests_all.csv.

<!-- words: definition 49, statistics 55, panels 52, data 17, total 173 -->

## FigS6_label_number_aux

Auxiliary label-number analyses. Break-even n (H27, recovery rule) is the smallest n at which residual ML (k-medoid labels, λ = 0.25) recovers at least 50 % of the gap between physics and the all-A-block-label reference with win rate ≥ 0.75; n* (A2) is the pre-specified CI rule of Fig. 4. Not-reached targets are shown as > n_max only. a: 4 of 14 targets reach break-even; reaching is associated with |log(E_own/E0)| (Mann-Whitney U = 40, exact p = 0.002, complete separation; Kendall τ_b = −0.64, p = 0.004, censored targets tied at the maximum rank); under the block-CI rule 6 of 14 reach it (τ_b = −0.39, p = 0.093). |log E| uses the oracle in-region E_own. d: censored-tied Spearman ρ = −0.04 (p = 0.89; 8 of 14 not reached); pre-registered F6 threshold ρ ≥ 0.6. a, Upper axis, reached targets; band, not-reached targets (n_max = 320 except Russia E 10; AL-4, CA-3 160; grey, n_max < 40); open triangles (8 targets) mark targets whose reference was worse than physics at every tested n; triangles closer than 1.5 mm are stacked in rows (x unchanged). b, Fraction of targets without n* by n (E0 fixed, all blocks), by error type (count in parentheses; level ≥ 0.2, structure < 0.15). c, Median over 14 targets of the ΔRMSE change from the default setting (α = 1, λ = 0.25, κ = 10); α from the original H25 run. d, Theory n versus empirical n*; the band above holds targets not reached by n_max (grey, n_max < 40), stacked in rows where they would overlap (x unchanged). Lena and Russia W coincide at n* = 3. Source data: data/processed/h3/h27c_targets.csv, h27c_rule.csv (a); data/processed/h4/a2_minn.csv (b); h3/h25_curve.csv, data/processed/h3/h25b_curve.csv (c); data/processed/h4/b2_theory.csv, data/processed/h3/h25b_breakeven.csv (d). Supplementary Table 6.

<!-- words: definition 56, statistics 75, panels 133, data 39, total 303 -->

## FigS7_tfm

Tabular foundation model as residual learner (B4). TFM replaces CatBoost as the residual model on the physics anchor (λ = 0.25, k-medoid labels). ΔRMSE in cm; negative favours TFM in a and the method in b. Target rows: 95 % block-bootstrap CIs (1,000 resamples, indices shared across methods). Mean row: stratified block bootstrap over the targets tested at that n (13 targets at n = 3 and 10; 11 at n = 40). Holm-adjusted p values in Supplementary Table 2. a, TFM minus CatBoost residual at n = 3, 10 and 40 (line style). b, Both residual learners against the physics anchor at n = 10. Rows ordered by |log(E_own/E0)|; values outside the axis are drawn as filled triangles at the edge. Source data: data/processed/h4/b4_summary.csv, data/processed/h4/b4_tests.csv.

<!-- words: definition 35, statistics 43, panels 43, data 12, total 133 -->

## FigS8_cci_bias

ESA CCI Permafrost ALT as a label-free input (C1). CCI error is CCI ALT minus the observed label; anchor error is prediction minus label for the physics anchor (E0 × √TDD) and for the CCI-combined anchor. Cells of the six label-free targets. b shows medians within CCI-error bins holding ≥ 20 cells. In c the uncertainty-weighted combination has RMSE 22.5 cm with CCI uncertainty only and 22.4 cm when source-region variance is added. a, Observed versus CCI ALT (log axes; up to 400 cells per target shown), 1:1 dotted. b, Propagation of CCI error into the combined anchor; dotted 1:1 is full propagation. c, Sensitivity of the uncertainty-weighted combination to source variance (1,500 cells shown). d, Block-mean CCI error, circle area proportional to cells; polar stereographic, true scale at 70°N, extent fitted to the labelled blocks. Source data: None; coordinates from data/processed/fidelity_base_v3.csv.

<!-- words: definition 41, statistics 30, panels 67, data 10, total 148 -->

## FigS9_label_year

Label-year sensitivity (D3). ERA5-Land covariates are 2015–2020 climatologies while labels span 1993–2023. a, Alaska in-domain: models trained on labels from 2015–2020 (overlap) or before 2015, minus the same model trained on all labels, scored on the same test cells. b, Transfer from Alaska: RMSE difference between label-year subsets of the target cells. 95 % CIs from 400 block-bootstrap resamples (paired on cells in a; unpaired subsets in b). Six-fold block cross-validation, three seeds for CatBoost. a, Rows: training subset and test subset; point estimates outside −3 to 5 cm are drawn as filled triangles at the axis edge with the value printed beside them; CIs are clipped at the axis limits. b, Rows: region and subset contrast (inside, labels within 2015–2020; before, before 2015; main regions, cell-pooled Lena, Canada and Russia); CIs are clipped at the axis limits. Source data: data/processed/m1/label_year_sensitivity.csv (tables delta_trainset and delta_subset).

<!-- words: definition 55, statistics 23, panels 64, data 14, total 156 -->

## FigS10_c4_not_performed

Supplementary Fig. 10 slot (C4, not performed). The planned weak-label multi-fidelity analysis, which would use InSAR seasonal subsidence as low-fidelity ALT labels, was not performed because of time constraints; no figure, table or result is reported. It is treated as a limitation on literature grounds: InSAR-derived ALT proxies depend on ground ice and soil moisture and have been unstable outside Alaska (negative correlation on the Qinghai-Tibet Plateau, Chang 2024; R² 0.03 in Svalbard, Wendt 2026; see also Sadeghi Chorsi 2024), and no validated InSAR-ALT grid exists outside Alaska. The slot is kept so that Supplementary items S11 and S12 retain their original numbers (no renumbering).

<!-- words: total 105 -->

## FigS12_a2_matrix

Confounder-free minimum label number (A2), all targets. Cells show ΔRMSE of residual ML (λ = 0.25, labels spread over all blocks) minus the physics anchor, in cm, for three E treatments. n* is the smallest n at which the block-bootstrap 95 % CI upper bound of ΔRMSE is below 0, at least two of three splits improve, and at least 75 % of repeats improve (pre-specified). Colour scale is shared across panels, centred at 0 (broc). Outlined cells meet all three conditions; n* is the leftmost outlined cell per row. F1 (pre-registered) is supported: with E0 fixed, 6 of 8 structure targets (AL-6, AL-2, AL-4, LE-2, Lena, LE-1) do not reach n* at n ≤ 40; Lena reaches it at n = 320 (ΔRMSE −0.23 cm [−0.53, −0.11]), the others not by n_max (160 or 320). a to c, E0 fixed, E shrunk toward E0 (κ = 10) and log E offset (MLE). Rows ordered by |log(E_own/E0)|. Canada and AL-5 meet the rule at n = 3, the smallest n tested, because the source residual already beats physics at n = 0 (Canada −0.95 cm [−1.18, −0.57], AL-5 −0.53 cm [−0.73, −0.38]); target labels leave the curve flat (Canada −0.72 cm at n = 3, −0.66 cm at n = 320). Error types as in Supplementary Table 6a: 8 structure, 2 intermediate, 4 level (oracle |log(E_own/E0)| < 0.15, 0.15 to 0.20, ≥ 0.20). Source data: data/processed/h4/a2_curve.csv with block CIs from h4/a2_blk.csv. Supplementary Table 5.

<!-- words: definition 65, statistics 69, panels 96, data 17, total 247 -->

## TableS1_cells_blocks_E

Supplementary Table 1. Scoring cells, blocks and Stefan coefficients. Upper rows: regions of Fig. 1b with E = exp(mean z), z = log ALT − log √TDD, and its 95 % block-bootstrap CI (regions with ≥ 8 blocks). Lower rows: the 15 transfer targets, with E_own fitted on all target cells (oracle) and E0 fitted on source cells. A-block cells give the label pool range over three splits; E0 is the split mean. Error type: structure |log(E_own/E0)| < 0.15, level ≥ 0.2, intermediate between; CA-1 is excluded. Same-parent source is the share of source cells from the target's parent region. Cell counts follow polar.m1_core.load_base (population 17,467 cells). Source: outputs/figures/paper/source_data/Fig1_b.csv; data/processed/h3/h25b_targets.csv.

<!-- words: definition 57, statistics 43, panels 10, data 13, total 123 -->

## TableS2_holm_families

Supplementary Table 2. Holm families and adjusted p values for the confirmatory tests referenced in the figures. p is the two-sided paired bootstrap p value; Holm p is adjusted within the listed family. H28 is adjusted over all eight AB4 mean tests (nested selection, exploratory and reference contrasts): the pre-specified λ = 0.25 recipe remains significant (Holm p 0.032) while λ = 0.5 does not (0.112); the registered H28 nested-selection hypothesis is rejected. H23 and H24 were not part of any Holm family. All rows with a Holm p value; region-level rows are in the CSV file together with the family sizes. Source: data/processed/m1/m1_sc_tests.csv, h2/h_tests_all.csv, h3/h28_tests.csv, h3/h30_deploy_tests.csv, h4/c1_tests.csv, h4/b4_tests.csv.

<!-- words: definition 33, statistics 48, panels 19, data 24, total 124 -->

## TableS3_ci_row_vs_block

Supplementary Table 3. Row-resampling versus block-bootstrap 95 % CIs for ΔRMSE vs physics. The block CI resamples scoring blocks within each split (1,000 draws, indices shared across methods, repeats and seeds) and is the CI drawn in figures; the row CI resamples (split, repeat) rows and ignores scoring-block variation. Width ratio is block CI width divided by row CI width per row. Row CI only is the share of rows whose row CI excludes 0 while the block CI does not. Summary by experiment; all paired rows with both intervals are in the CSV file. Source: data/processed/h3/h25b_curve.csv; data/processed/h4/b1_blk.csv, b2_pooling.csv, b3_blk.csv, b4_summary.csv, d1_blk.csv, a2_curve.csv.

<!-- words: definition 50, statistics 32, panels 14, data 21, total 117 -->

## TableS4a_b1_protocol

Supplementary Table 4a. Two-stage protocol (B1): all methods and gating thresholds τ over 14 targets. Δ is RMSE minus physics-anchor RMSE (cm). Worse counts targets with Δ > 0 (point) or with the 95 % block-bootstrap CI above 0; Better counts CIs below 0. Gap to oracle branch is the mean Δ difference to the post hoc best branch. protocol@loto selects τ by leave-one-target-out; its selected τ counts are in the CSV file. Explore rows were added after results were viewed and are outside the pre-registered analysis. Rows by method and n (3 or 10 target labels). Source: data/processed/h4/b1_protocol.csv; block CIs from h4/b1_blk.csv.

<!-- words: definition 58, statistics 29, panels 10, data 12, total 109 -->

## TableS4b_h28_two_scorings

Supplementary Table 4b. Recipe selection (H28, Fig. 2d) under two scorings: cell-weighted RMSE and block-equal RMSE (each scoring block weighted equally). Δ is RMSE minus physics-anchor RMSE (cm). CIs are stratified block-bootstrap intervals over the AB4 regions (1,000 resamples). Rows: registered nested selection, exploratory variants, pre-specified references and the Alaska fold check. Source: data/processed/h3/h28_tests.csv.

<!-- words: definition 28, statistics 12, panels 13, data 6, total 59 -->

## TableS5_a2_nstar

Supplementary Table 5. Minimum label number n* (A2) for residual ML (λ = 0.25) by E treatment and label placement (all blocks, concentrated, spread over five blocks). n* uses the pre-specified three-condition rule of Fig. 4. Not-reached targets are reported as > n_max, the largest n tested; range-limited marks n_max < 40. No values are imputed. Printed columns: labels over all blocks for four E treatments (E_own fixed uses the oracle in-region E), and E0 fixed with concentrated or five-block placement. Rows ordered by |log(E_own/E0)|; other combinations, stages and λ = 0.5 are in the CSV file. Source: data/processed/h4/a2_minn.csv.

<!-- words: definition 35, statistics 20, panels 44, data 6, total 105 -->

## TableS6a_h27_breakeven

Supplementary Table 6a. H27 break-even n for residual ML (λ = 0.25) by label rule and definition. Recovery rule: at least 50 % recovery of the physics to all-A-block-label gap with win rate ≥ 0.75. Block CI rule: block-bootstrap 95 % CI upper bound below 0, split win rate ≥ 2/3 and repeat win rate ≥ 0.75. Not-reached targets are reported as > n_max (the largest n tested), never as imputed values; Russia E (n_max 10) is range-limited. The row-resampling CI rule is in the CSV file only. † marks targets whose all-label reference was worse than physics at every tested n, so recovery is undefined. |log E| (oracle) uses all target cells; 3 labels uses pooled three-label estimates. Rows ordered by oracle |log(E_own/E0)|; error type: structure < 0.15, level ≥ 0.2, intermediate between. Source: data/processed/h3/h27c_targets.csv (derived from the H25 re-run, h3/h25b_curve.csv).

<!-- words: definition 54, statistics 62, panels 16, data 14, total 146 -->

## TableS6b_h27_rule_stats

Supplementary Table 6b. Association between reaching break-even and |log(E_own/E0)| for each label rule, stage and definition (S1, E re-fit; S3, residual ML λ = 0.25). Kendall τ_b treats not-reached targets as tied at the maximum rank; Mann-Whitney U compares |log E| of reached and not-reached targets (exact p). Logistic coefficients are not reported under complete separation. Spearman correlations are in the CSV file and are not used as the primary statistic. Printed rows: recovery and block CI rules; 14 targets with oracle and three-label pooled |log E|, and the 10 subregions refitted with same-parent source cells excluded (excl.). Other variants are in the CSV file. Source: data/processed/h3/h27c_rule.csv, h27x_rule.csv.

<!-- words: definition 27, statistics 47, panels 34, data 8, total 116 -->

## TableS6c_c2_coverage_blockeq

Supplementary Table 6c. Label-free 90 % conformal intervals (C2): leave-one-region-out coverage with cell-weighted and block-equal scoring. pooled, single pooled quantile; hier2_cdf, hierarchical two-level (primary); hier2_blk, block-level calibration. Block-equal coverage weights each scoring block equally. Russia C and Greenland have fewer than 8 blocks. hier2_exact is infinite when the number of calibration regions is below 1/α and is listed only in the CSV file. Rows by target region and method. Source: data/processed/h4/c2_coverage.csv.

<!-- words: definition 29, statistics 38, panels 6, data 6, total 79 -->

## TableS7_d1_external

Supplementary Table 7. External hold-out D1 (Mongolia and Central Asia, 46 labelled cells). Δ is RMSE minus physics-anchor RMSE (cm); bias is mean prediction minus label. refit_allA and shrink_allA use all A-block labels (reference, not an upper bound). 95 % block-bootstrap CIs over 3 splits. All D1 labels are thaw depths derived from ground-temperature profiles (fidelity 3), unlike the probe-based source labels, so the physics bias mixes regional and label-definition differences. offset_only was added after results were viewed (exploratory, outside F11). Rows by method, n target labels and gating threshold τ (protocol only). Source: data/processed/h4/d1_summary.csv; block CIs from h4/d1_blk.csv.

<!-- words: definition 40, statistics 44, panels 12, data 12, total 108 -->

## TableS8_h18_h24_tests

Supplementary Table 8. Label-free and sparse-label hypothesis tests H18 to H24, four-region (AB4) means. Δ is RMSE of the first method minus the comparator, in cm; negative favours the first method. H18 to H22 CIs are stratified block-bootstrap intervals over the AB4 regions with Holm adjustment within each pre-registered family. H23 and H24 means have no CI and are outside the Holm family; H24 is exploratory (the pre-registered gain over random labels held only at n = 3). Printed rows: confirmatory H18 to H22 tests, H23 at λ = 0.25 and H24 k-medoid rows; all rows in the CSV file. Source: data/processed/h2/h_tests_all.csv, h23_tests.csv, h24_tests.csv.

<!-- words: definition 31, statistics 46, panels 21, data 11, total 109 -->

## TableS9_covariates

Supplementary Table 9. The 25 covariates available in every region (input set x25), used by all ML models. Label-derived quantities (observation counts, years, spread) are excluded from inputs. Non-missing share over the 17,467 label cells after load_base. Rows by covariate group. Source: polar.m1_core.INPUT_SETS, data/processed/fidelity_base_v3.csv, fidelity_base_v3_meta.json.

<!-- words: definition 28, statistics 11, panels 4, data 13, total 56 -->

## TableS10a_software

Supplementary Table 10a. Software versions used for the analyses and figures. Versions are read from the analysis environment at figure build time. One row per package. Source: runtime environment of scripts/4_visualization/paper_figs.py.

<!-- words: definition 11, statistics 11, panels 4, data 9, total 35 -->

## TableS10b_runs

Supplementary Table 10b. Experiment runs: seeds, splits, repeats, bootstrap resamples and wall-clock run time. Run times are single-node wall-clock hours on the shared server (CPU, 4 to 6 threads per process). Runs with recorded run time; all metadata files are listed in the CSV file. Source: data/processed/*/*_meta.json.

<!-- words: definition 14, statistics 17, panels 14, data 4, total 49 -->

## v2 (F3 part 1, 2026-09-30): main-text Fig 2–4

Files: `outputs/figures/paper/v2/Fig{2,3,4}_*.{pdf,svg,png}` (PDF with embedded TrueType text, SVG with live text, PNG 600 dpi); plotted values in `source_data/v2/`; QA in `_qa/v2_*`. The 2026-09-26 sections above are kept unchanged. Platform: every LG and LGX value comes from shards computed on the Rescale platform (LG job ZovWo; LGX shards on Rescale). The LGX tables (`lgx_tests`, `lgx_lg_aux`), the fixed-composition pooled curves and the N1 target counts are local re-aggregations of those shards; columns derived from `lgx_lg_aux` carry the label "cross-environment check not performed". Verdict symbols are copied from `lg_tests`, `lgx_lg_aux` and `lgx_tests`; none are recomputed in the plotting code. Markers for the abstract contrast bundle (AB1–AB10, `lgw_bundle`) are not yet drawn because those results have not been recorded.

## v2/Fig2_label_curve

**Error change with the number of target labels, relative to the source-coefficient Stefan model.** ΔRMSE = RMSE(method) − RMSE(P0) on the scored B half of each target; negative is lower error. P0 applies the source least-squares Stefan coefficient E0; P1 re-fits E on the n labels with κ = 10 shrinkage; R0 and R1 add residual CatBoost (λ = 0.25) to P0 and P1; R2 adds physics pseudo-label rows to R1; D0 is direct CatBoost without a physics anchor. P* and P1* are the stronger physics baselines found under L29. a–d, cell-weighted means over five splits with block-bootstrap 95% CI (1,000 resamples; bands for P1 and R1 only). e, f, fixed-composition stratified means (10,000 resamples). Verdict symbols are copied from the LGX tables (10,000 resamples): both cell-weighted and block-equal-weighted CIs, equivalence margin 0.5 cm; † marks |Δ| < 0.5 cm. a–d, Lena, Canada, Russia W and Russia E (source excludes the target; mode x); hatched, n larger than the labelled pool. e, four-region pool P4 (n ≤ 10). f, Lena and Canada pool at every n present in both (n = 1,000 exists for Lena only). Dotted line, P* − P0; open squares, P1* − P0 at n = 40 and 160. In f both are point estimates (means of region values). Table right of f, registered pooled contrasts; its pool row gives the regions behind each column (L+C, Lena and Canada). Symbols above the grey line in a–d, region-level D0 − P0 verdicts for L1 (registered verdict: supported). All values come from LG and LGX runs on the Rescale platform; the LGX tables are a local re-aggregation of the same shards (cross-environment check not performed).

<!-- words: definition 75, statistics 52, panels 110, data 27, total 264 -->

## v2/Fig3_physics_use

**How physics enters the learner: pseudo-label controls, combination structure, all-label effect and coefficient pooling.** ΔRMSE is the RMSE difference in cm (first method minus second; negative is lower error for the first). D1 adds Stefan pseudo-labels (κ = 10 re-fit E times √TDD) to direct CatBoost; placebos replace them. F1k, F1a and F1n give physics outputs to direct ML as inputs; RM is a multiplicative residual; R1s shuffles the target labels. P* is P0 with year-matched TDD and P1* the edaphic Stefan re-fit (L29). Points and bars, cell-weighted stratified means with block-bootstrap 95% CI (10,000 resamples), four-region pool P4 unless stated. Verdict symbols (right of each forest) are copied from the LGX tables and use both cell-weighted and block-equal-weighted CIs with a 0.5 cm margin; † marks |Δ| < 0.5 cm. Panel d has no verdicts (descriptive). a, placebo controls (L15, registered verdict: mixed). b, combination structure: physics as input versus residual (L10, supported), multiplicative versus additive residual (L12, rejected), shuffled target labels (L17). c, all target labels: R1 against P0 (L8), P* and P1 (L4) by region, P4 and the three-region mean with Alaska (x); Alaska is shown as a reference row and is not pooled with P4. Grey ticks, covariate-conditional error floor minus P0 (LGX-N2). Bars, share of 50 splits (49 for Lena) with Δ < 0 (L31, dashed line 0.8). d, coefficient models against P0; the Lena and Canada P* line and P1* points are point estimates; CIs are drawn for P1 only (all CIs in Source Data). LG and LGX runs on the Rescale platform; LGX tables are a local re-aggregation of the same shards (cross-environment check not performed).

<!-- words: definition 69, statistics 52, panels 113, data 22, total 256 -->

## v2/Fig4_min_labels

**Minimum number of target labels and the distribution of improvement and deterioration across targets.** n* is the smallest grid n at which the method has lower error than its baseline under three registered conditions (block-CI upper bound < 0, split win rate ≥ 2/3, repeat win rate ≥ 0.75). It is reported as an interval n_lo < n* ≤ n_hi; an arrow means not reached up to the largest tested n (no imputed value). n* is registered against P0 and P1 only; no n* was computed against P* or P1*. a, c, block-bootstrap CIs with 1,000 resamples. b, fixed-composition stratified means with cell-weighted 95% CI (10,000 resamples). Verdicts use both cell-weighted and block-equal-weighted CIs with a 0.5 cm margin: symbols in b are copied from the LGX tables (10,000 resamples); counts in d apply the same rule to the per-target LG curve CIs (1,000 resamples). † marks |Δ| < 0.5 cm. a, n* by target; sub-regions share cells or sources with their parent region and are not independent. b, R1 − P1 (L4, registered verdict: net value of ML at sparse labels, minimum n = 10 for the four-region mean) and R1 − P1* at n = 40 and 160, where P1* (edaphic Stefan re-fit) outperformed P1 (L29); the pool row gives the regions behind each verdict. c, L3 targets, R0 with α = 1 versus nested α (registered verdict: supported; nested α reduced n* for no target). d, number of targets per verdict at each n; sub-regions are counted, not averaged. LG and LGX runs on the Rescale platform; LGX tables and the pooled curves are local re-aggregations of the same shards (cross-environment check not performed). The n = 80 grid point (C12) was not run.

<!-- words: definition 74, statistics 63, panels 98, data 34, total 269 -->

