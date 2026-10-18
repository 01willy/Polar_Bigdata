#!/usr/bin/env python3
"""si_x_results.py : Supplementary Table S13 parts c to n (results of XB to XL), used by make_si_tables.py.

Values are copied, not recomputed, from (round 3, 5 October 2026):
  * docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md section 8.2 (XB), 8.3 (XH), 8.4 (XI), 8.5 (XJ), 8.6 (XE, xh0 only),
    8.7 (XG), 8.8 (XD-alg) and 8.9 (XD-4); registered interpretation sentences of the same sections, rendered in English;
  * docs/EXPERIMENT_PLAN_FINAL_BATCH_ADDENDUM_XK_XL_2026-10-05.md, results section (XK, XL);
  * paper/claims/*/README.md section 6.1 (evidence index; same values).
XM rows come from docs/EXPERIMENT_PLAN_FINAL_BATCH_ADDENDUM_XM_2026-10-05.md, results section (sealed tables first opened
5 October 2026, 14:20:47 KST; commit e454a71). XC rows come from section 8.10 (sealed tables opened 5 October 2026, 05:20:39 KST); XD-5 from 8.11, the XB CCI v5 sensitivity from 8.12 and XE-e from the end of 8.6; XF, XE-a/XE-b and XC-F3 carry [PENDING] markers.
Minus signs are U+2212. Cell entries are LaTeX-ready (no escaping by the caller).
"""

PEND_XF = "Pending: registered data deadline 11 October 2026 (XF)."   # round 10: plain sentence, no tag
PEND_XE = "Pending: registered data deadline 8 October 2026 (XE stage 2: XE-a, XE-b, XE-a0 and the xt2 table)."


def note(text):
    return "\\par\\noindent{\\footnotesize " + text + "}\\par\\smallskip"


# ---------------------------------------------------------------------------------------------- XB (8.2)
XB_H = ["Hypothesis and contrast", "$n$", "Pool", "$\\Delta$ cell [95\\% CI]", "$\\Delta$ block-equal [95\\% CI]", "Fallback", "Holm $P$", "Verdict"]
XB = [
    ["XB-1 Stack $-$ P1 (transfer)", "10", "four main regions, partial (3/4; no stacking in E Russia, fallback 0.52)", "−0.28 [−1.11, +1.16]", "+1.42 [+0.46, +2.39]", "0.36", "1.00", "undetermined"],
    ["", "40", "Lena Delta, Canada (2/2)", "−1.86 [−2.44, −0.55]", "+0.10 [−0.44, +0.64]", "0.14", "1.00", "undetermined"],
    ["", "160", "Lena Delta, Canada (2/2)", "−2.73 [−3.38, −1.35]", "−0.46 [−1.14, +0.22]", "0.10", "1.00", "undetermined"],
    ["", "all", "four main regions (4/4)", "−1.54 [−2.28, −0.47]", "−0.39 [−1.13, +0.35]", "0.20", "1.00", "undetermined; hypothesis rejected"],
    ["XB-2 StackR(0.25) $-$ R1(0.25)", "10", "partial (3/4)", "−0.33 [−1.10, +0.99]", "+1.16 [+0.33, +2.00]", "0.36", "1.00", "undetermined"],
    ["", "40", "Lena Delta, Canada (2/2)", "−1.65 [−2.15, −0.49]", "−0.01 [−0.47, +0.44]", "0.14", "1.00", "undetermined"],
    ["", "160", "Lena Delta, Canada (2/2)", "−2.27 [−2.82, −1.05]", "−0.35 [−0.94, +0.23]", "0.10", "1.00", "undetermined"],
    ["", "all", "four main regions (4/4)", "−1.38 [−2.00, −0.41]", "−0.43 [−1.05, +0.20]", "0.20", "1.00", "undetermined; hypothesis rejected"],
    ["XB-3 StackR $-$ R1 (CV $\\lambda$), within region", "200", "three targets (3/3)", "−0.57 [−0.71, −0.20]", "+0.42 [+0.24, +0.60]", "0.08", "0.004", "undetermined (the two CIs exclude zero on opposite sides)"],
    ["", "500", "partial (2/3; Canada $|A| < 500$)", "−0.55 [−0.69, −0.35]", "−0.33 [−0.47, −0.19]", "0.05", "0.0012", "lower error"],
    ["", "1,000", "partial (2/3)", "−0.498 [−0.62, −0.34]", "−0.29 [−0.44, −0.15]", "0.01", "0.0012", "lower error (distinguishable, size below 0.5 cm)"],
    ["", "all", "three targets (3/3)", "−0.99 [−1.19, −0.43]", "+0.19 [−0.05, +0.43]", "0.01", "1.00", "undetermined; hypothesis partially supported"],
    ["XB-4 Stack $-$ P1, non-inferiority (margin 0.5 cm), Alaska within region", "200", "Alaska", "−0.47 [−0.61, −0.28]", "−0.73 [−0.92, −0.55]", "0.16", "$<$0.0001", "non-inferior (four-way lower error, size below 0.5 cm)"],
    ["", "500", "Alaska", "−0.63 [−0.77, −0.44]", "−0.86 [−1.06, −0.67]", "0.09", "$<$0.0001", "non-inferior (four-way lower error)"],
    ["", "1,000", "Alaska", "−0.72 [−0.84, −0.51]", "−0.85 [−1.05, −0.65]", "0.03", "$<$0.0001", "non-inferior (four-way lower error)"],
    ["", "all", "Alaska", "−0.75 [−0.89, −0.54]", "−0.90 [−1.12, −0.69]", "0.04", "$<$0.0001", "non-inferior (four-way lower error); hypothesis supported"],
]
XB_NOTE = (
    "Flags: designed after earlier results were viewed; registered before running; partly unblinded as in the registered hypothesis table. "
    "Holm families: XB-1 to XB-3 ($m$ = 12), XB-4 non-inferiority ($m$ = 4); verdicts use the uncorrected CIs. Fallback: share of draws in which stacking was replaced by P1. "
    "Registered sentences (rendered): XB-1, `No difference between stacking and the recalibrated Stefan model was established' (40 and 160 labels: Lena Delta and Canada, two regions); "
    "the statement that the gain from combining physics models and products is region-conditional (WF7, L29) is retained; at ten labels the stacking fallback rate was 0.36, so XB-1 at ten labels "
    "may in effect compare P1 with P1. XB-2, `No difference between stacked residuals and recalibrated-anchor residuals was established' (same qualifiers). "
    "XB-3, `Within label-rich regions, stacked residuals had 0.55 cm (cell-weighted) lower error than recalibrated-anchor residuals with 500 labels (partial, 2 of 3 regions)' and "
    "`... 0.498 cm lower error with 1,000 labels (partial, 2 of 3 regions); the difference is statistically distinguishable but below 0.5 cm in size'; `no difference was established "
    "with 200 and all labels (3 of 3 regions)'. XB-4, `Within Alaska, where the physics model is close to the error floor, the error increase of stacking relative to the recalibrated "
    "Stefan model was within 0.5 cm (non-inferior).' XB-3 was not retained when CCI v5 was added as a candidate (sensitivity edition, part c continued); the registered verdict remains the main-run one. Regional rows: no higher-error row. XB-3 Canada rows had opposite signs in the two weightings ($n$ 200 −0.95 [−1.29, +0.14] / "
    "+1.68 [+1.24, +2.13]; all −2.05 [−2.60, −0.47] / +1.05 [+0.41, +1.69]; undetermined). Weights are reported descriptively only (no causal sentence): mean weight with all labels, "
    "Alaska soil-thaw-index Stefan 0.82, Canada Kudryavtsev 0.57, Lena Delta year-matched Stefan 0.77, Tibetan Plateau affine-calibrated CCI v4 0.92; Spearman $\\rho$ of the CCI weight "
    "with the P1 RMSE of the target 0.31 ($P$ 0.094, 31 stores). Auxiliary XB-5 (not counted): Stack0 $-$ P0 without labels, four main regions +0.51 [+0.09, +0.69] / +0.09 [−0.14, +0.31], "
    "undetermined. Source: data/processed/xbatch/XB\\_multisource\\_stacking/sealed/ (xb\\_tests.csv, xb\\_weights\\_summary.csv, xb\\_fallback.csv); reproduction gate passed (xb\\_gate.csv).")

# ---------------------------------------------------------------------------------------------- XG (8.7)
XG_H = ["Hypothesis and contrast", "$n$", "Pool", "5 km block mask: cell / block-equal", "25 km block mask: cell / block-equal", "Holm $P$ (5 / 25 km)", "Verdict", "Product values imputed (5 / 25 km)"]
XG = [
    ["XG-1w B:wei $-$ P0", "0", "Lena Delta, Canada (2/2)", "+1.44 [−0.02, +4.30] / +4.09 [+2.10, +6.12]", "+0.97 [−0.41, +3.87] / +4.01 [+1.99, +6.12]", "0.156 / 0.383", "undetermined", "47.4\\% / 26.5\\%"],
    ["XG-2w R1(0.25) $-$ P1@wei", "10", "Lena Delta, Canada", "+1.34 [+0.00, +1.83] / −0.49 [−1.36, +0.37]", "+1.25 [−0.07, +1.75] / −0.52 [−1.43, +0.35]", "0.542 / 0.504", "undetermined", "47.4\\% / 26.5\\%"],
    ["XG-3w R1(0.25) $-$ P1@wei", "all", "Lena Delta, Canada", "+0.87 [−0.41, +1.37] / −0.39 [−1.11, +0.33]", "+0.61 [−0.61, +1.19] / −0.43 [−1.17, +0.30]", "0.542 / 0.504", "undetermined", "47.4\\% / 26.5\\%"],
    ["XG-1c B:cci5y $-$ P0", "0", "four main regions (4/4)", "+23.79 [+15.34, +30.16] / +19.17 [+14.83, +23.72]", "same", "0.0006 / 0.0006", "higher error", "below 0.1\\%"],
    ["XG-2c R1(0.25) $-$ P1@cci5y", "10", "four main regions", "−8.37 [−10.97, −7.39] / −12.87 [−14.69, −11.10]", "same", "0.0006 / 0.0006", "lower error (HK region-level CI [−16.02, −0.50])", "below 0.1\\%"],
    ["XG-3c R1(0.25) $-$ P1@cci5y", "all", "four main regions", "−7.51 [−10.31, −6.27] / −11.75 [−13.63, −9.88]", "same", "0.0006 / 0.0006", "lower error (HK CI [−17.05, +2.43])", "below 0.1\\%"],
]
XG_NOTE = (
    "Flags: registration deviation (WRAPUP 10); designed after earlier results were viewed; XG-1w to XG-3w blind; XG-1c partly unblinded (raw CCI v4 minus P0 viewed in LGX L29); "
    "XG-2c and XG-3c partly unblinded (CCI v4 scale recalibration viewed); all confirmatory rows carry a few-blocks flag after masking (fewer than eight scoring blocks per split). "
    "The two masks are co-primary and gave the same branch, so no mask-dependence flag applies. CCI v5 has no training sites, so its values are the same under both masks. "
    "B:product, the product map without labels; P1@product, the product recalibrated with the same target labels. ALT definitions differ: CCI is the annual maximum thaw depth of a "
    "model pixel, Wei v2 is a machine-learning map trained on CALM labels (including CALM sites inside the target regions), and our labels mix GPR and probe measurements. "
    "Source-coefficient RMSE before and after masking (cell-weighted): Wei pool 24.93 cm, 25.28 cm (5 km) and 28.34 cm (25 km); Lena Delta 21.69, 21.74 and 27.86 cm; Canada 28.16 and "
    "28.82 cm (both masks); CCI pool 30.64 cm (unchanged). Regional XG-1c rows (5 km): Lena Delta +1.19 [+0.13, +5.38] / +9.34 [+6.57, +12.22] (depends on split independence), "
    "Canada +82.38 [+49.01, +99.54] / +53.00 [+41.74, +64.55], E Russia +5.84 [+0.08, +18.20] / +10.44 [+4.25, +17.24] (depends on split independence), all higher error; W Russia "
    "+5.78 [−5.86, +17.69] / +3.89 [−6.71, +15.83], undetermined. XG-2c lower error in all four regions; XG-3c undetermined in Canada (−1.47 [−7.55, +0.96] / −10.94 [−13.71, −8.20]) and "
    "lower error elsewhere. Wei rows: XG-3w lower error in the Lena Delta only (5 km −1.08 [−1.76, −0.59] / −1.49 [−2.24, −0.77]); the Lena comparison depends strongly on imputed "
    "product values (excluding blocks with imputed values removes all Lena blocks, so the Wei pool row cannot be decided; CCI v5 verdicts are unchanged). Cell-level 5 km sensitivity "
    "(refitted): XG-1w +2.09 [+0.41, +5.23] / +4.45 [+2.50, +6.46], higher error ($P$ 0.016); XG-2w and XG-3w undetermined; sensitivities do not change the confirmatory verdicts. "
    "Registered sentences (rendered): XG-1c, `Without labels, the existing CCI v5 map had 23.79 cm higher error than the source-coefficient Stefan model (4 of 4 regions, after 5 and 25 km "
    "block masks of training sites); error was higher in the Lena Delta ($\\Delta$ +1.19 cm, CI [0.13, 5.38]), Canada (+82.38 cm, [49.01, 99.54]) and E Russia (+5.84 cm, [0.08, 18.20]).' "
    "XG-2c, `With the same ten target labels, the recalibrated-anchor residual had 8.37 cm lower error than CCI v5 recalibrated with those labels' (the HK region-level CI excludes zero, "
    "so a region-general sentence is allowed). XG-3c, `... with all target labels, 7.51 cm lower error' (the HK CI includes zero; no region-general sentence). XG-1w, `Without labels, no "
    "difference was established between the existing Wei v2 map and the source-coefficient Stefan model (2 of 2 regions, after 5 and 25 km block masks; product values imputed for 47.4\\%). "
    "Wei v2 also used CALM sites inside the target regions for training, whereas the source-coefficient Stefan model used only labels outside the target region and its 100 km buffer.' "
    "XG-2w and XG-3w, `No difference was established between the recalibrated-anchor residual with the same 10 (all) target labels and Wei v2 recalibrated with those labels (product values "
    "imputed for 47.4\\%).' No sentence states that physics-anchored ML is worse than a product that shares training data (WRAPUP 10). "
    "Source: data/processed/xbatch/XG\\_product\\_comparison/sealed/ (xg\\_tests.csv, xg\\_confirm\\_rows.csv, xg\\_tests\\_cell.csv, xg\\_fill\\_sensitivity.csv); gate passed 5 October 2026, 03:42:02.")

# ---------------------------------------------------------------------------------------------- XH (8.3)
XH_H = ["Region", "Item", "W1R random cell", "W1S 0.05° site", "W1B 0.5° block", "W1K kNNDM", "V-G region holdout"]
XH = [
    ["Alaska", "median distance to nearest training cell (km)", "0.00", "0.88", "26.06", "58.73", ""],
    ["", "D0", "11.54 [9.61, 14.99] / 11.94", "13.28 / 13.65", "16.21 / 14.02", "17.30 [14.15, 19.68] / 14.12", "35.56 / 31.91"],
    ["", "P1 (V-G: P0)", "14.24 [12.08, 18.60] / 15.37", "14.30 / 15.38", "14.41 / 15.43", "14.53 [12.43, 18.84] / 15.70", "14.68 / 14.31"],
    ["", "R1 (V-G: R0)", "13.27 / 14.39", "13.82 / 14.88", "13.97 / 14.97", "14.07 / 15.41", "15.00 / 16.76"],
    ["Lena Delta", "median distance (km)", "0.01", "1.02", "9.55", "26.42", ""],
    ["", "D0", "13.94 [10.86, 21.12] / 21.44", "18.28 / 23.25", "19.77 / 24.51", "22.10 [17.26, 37.01] / 26.42", "25.31 / 24.60"],
    ["", "P1 (V-G: P0)", "20.89 [13.42, 37.22] / 25.92", "21.11 / 26.07", "21.18 / 26.23", "21.40 [13.67, 38.01] / 26.37", "21.64 / 24.95"],
    ["", "R1 (V-G: R0)", "18.35 / 24.08", "19.54 / 24.55", "20.17 / 24.94", "21.03 / 25.48", "21.93 / 24.67"],
    ["Canada", "median distance (km)", "0.01", "3.86", "25.33", "236.53", ""],
    ["", "D0", "18.96 [14.01, 24.14] / 20.90", "23.90 / 20.08", "24.48 / 21.46", "33.16 [22.52, 39.09] / 28.48", "25.24 / 20.22"],
    ["", "P1 (V-G: P0)", "26.50 [18.43, 33.13] / 20.99", "27.98 / 21.31", "29.53 / 21.22", "30.78 [20.19, 39.06] / 23.40", "26.48 / 20.96"],
    ["", "R1 (V-G: R0)", "23.59 / 19.80", "26.27 / 19.83", "27.83 / 20.07", "30.94 / 22.57", "25.60 / 20.05"],
]
XHD_H = ["Item", "Learner", "Alaska", "Lena Delta", "Canada", "Three-region stratified mean"]
XHD = [
    ["XH-1 ($M$ = P1)", "CatBoost (main)", "+5.46 [+2.13, +8.53] / +1.85 [+0.75, +2.97]", "+7.66 [+4.20, +15.86] / +4.52 [+0.59, +8.68]", "+9.92 [+3.35, +17.28] / +5.17 [+0.94, +9.26]", "+7.68 [+4.51, +11.25] / +3.85 [+1.84, +5.83]"],
    ["", "CatBoost (second setting)", "+5.27 / +2.70", "+7.61 / +4.76", "+8.77 / +5.09", "+7.22 [+4.84, +10.42] / +4.18 [+2.28, +6.11]"],
    ["", "random forest", "+6.95 / +3.26", "+9.99 / +5.78", "+6.41 / +1.30", "+7.78 [+5.37, +10.26] / +3.44 [+1.97, +5.06]"],
    ["", "CatBoost (main), Canada ±2°", "", "", "+7.79 [+1.82, +14.86] / +5.03 [+1.08, +8.92]", "+6.97 [+3.99, +10.49] / +3.80 [+1.90, +5.76]"],
    ["XH-2 ($M$ = R1, $\\lambda$ 0.25)", "CatBoost (main)", "+4.96 [+1.51, +8.44] / +1.15 [+0.29, +2.03]", "+5.49 [+2.99, +11.04] / +3.58 [+0.62, +6.58]", "+6.85 [+2.42, +11.97] / +4.81 [+1.04, +8.58]", "+5.77 [+3.25, +8.43] / +3.18 [+1.54, +4.79]"],
    ["", "CatBoost (second setting)", "+4.15 / +1.70", "+5.20 / +3.32", "+5.83 / +4.78", "+5.06 [+3.35, +7.32] / +3.27 [+1.74, +4.80]"],
    ["", "random forest", "+5.59 / +2.04", "+7.22 / +4.21", "+3.99 / +0.82", "+5.60 [+3.67, +7.45] / +2.36 [+1.24, +3.55]"],
]
XH_NOTE = (
    "Flags: designed after earlier results were viewed; descriptive (no verdict words, no multiplicity); Alaska rows are an unblinded replication, Lena Delta and Canada rows blind. "
    "Values: RMSE in cm, cell-weighted [95\\% CI] / block-equal. D0 direct CatBoost; P1 Stefan coefficient fitted on the training fold; R1 residual ML ($\\lambda$ 0.25). V-G rows reuse "
    "the region-holdout results of an earlier run (source-coefficient Stefan P0 and residual R0, no target labels); the sealed reuse table was empty, so these values were read from "
    "data/processed/lgx/ladder/lgv\\_metrics.csv without new fits. $\\Delta\\Delta$ = [RMSE$_{K}$(D0) $-$ RMSE$_{R}$(D0)] $-$ [RMSE$_{K}$($M$) $-$ RMSE$_{R}$($M$)], in cm, cell-weighted [95\\% CI] / "
    "block-equal [95\\% CI] from 10,000 block resamples of cell-level squared errors. Canada prediction-domain sensitivity ($\\pm$2°, W1K): D0 29.99 / 26.97, P1 29.74 / 22.03, R1 29.31 / 21.06 cm. "
    "Alaska contest-continuity rows (ridge, W1R to W1K, cell-weighted): direct ridge 12.65 to 14.18 cm, residual ($\\lambda$ 0.75) 12.77 to 13.72 cm. "
    "Registered descriptive sentence (rendered): `From random-cell to kNNDM splits, the RMSE of direct ML increased by 5.76 cm (Alaska), 8.17 cm (Lena Delta) and 14.20 cm (Canada), and that of "
    "the recalibrated Stefan model by 0.29, 0.51 and 4.28 cm (cell-weighted; block-equal, direct ML 2.18, 4.97 and 7.58 cm, recalibrated Stefan 0.32, 0.45 and 2.41 cm; the difference of the "
    "two increases, $\\Delta\\Delta$, was +5.46 [+2.13, +8.53], +7.66 [+4.20, +15.86] and +9.92 [+3.35, +17.28] cm, three-region stratified mean +7.68 [+4.51, +11.25] cm, block-equal three-region "
    "+3.85 [+1.84, +5.83] cm).' With random-cell splits, RMSE was 11.54, 13.94 and 18.96 cm for direct ML and 14.24, 20.89 and 26.50 cm for the recalibrated Stefan model (cell-weighted). "
    "Accompanying sentence: random validation has been argued to estimate map accuracy validly for probability samples (Wadoux et al., 2021); the labels of this study are not a probability sample. "
    "Source: data/processed/xbatch/XH\\_validation\\_ladder/sealed/ (xh\\_ladder.csv, xh\\_dd.csv, xh\\_stage\\_contrasts.csv); gates passed (xh\\_gate.csv).")

# ---------------------------------------------------------------------------------------------- XI (8.4)
XI_H = ["Hypothesis and contrast", "$n$", "warm\\_trim (main): cell / block-equal", "Verdict, main", "warm (basic): cell / block-equal", "Verdict, basic", "Reported verdict (weaker)", "Holm $P$"]
XI = [
    ["XI-a EP(D0 $-$ P1), higher-error criterion", "100", "−3.12 [−3.95, −1.02] / +0.33 [−1.52, +2.18]", "undetermined", "−4.68 [−5.90, −2.50] / −2.29 [−4.55, +0.02]", "undetermined", "undetermined", "1.00"],
    ["", "all", "−6.17 [−7.00, −3.02] / −1.12 [−3.36, +1.22]", "undetermined", "−6.49 [−7.66, −3.77] / −3.07 [−5.59, −0.51]", "lower error (depends on split independence)", "undetermined (both reported); hypothesis rejected", "1.00"],
    ["XI-b EP(R1 $-$ D0), lower-error criterion", "100", "+0.79 [−0.23, +1.41] / −0.86 [−2.27, +0.54]", "undetermined", "+2.37 [+1.20, +3.34] / +1.30 [−0.43, +3.04]", "undetermined", "undetermined", "0.90"],
    ["", "all", "+3.99 [+1.90, +4.61] / +0.49 [−1.47, +2.38]", "undetermined", "+3.58 [+1.95, +4.57] / +2.30 [+0.31, +4.31]", "higher error (depends on split independence)", "undetermined (both reported); hypothesis rejected", "1.00"],
    ["XI-c R1 (CV $\\lambda$) $-$ P1 in W, non-inferiority", "100", "−2.44 [−2.83, −1.78] / −2.35 [−2.92, −1.79]", "lower error, non-inferior", "−2.91 [−3.31, −2.34] / −2.94 [−3.59, −2.23]", "lower error, non-inferior", "lower error", "$<$0.0001"],
    ["", "all", "−2.41 [−3.09, −2.03] / −2.88 [−3.40, −2.36]", "lower error, non-inferior", "−2.88 [−3.47, −2.27] / −3.20 [−3.98, −2.38]", "lower error, non-inferior", "lower error; hypothesis supported", "$<$0.0001"],
]
XI_NOTE = (
    "Flags: designed after earlier results were viewed; Canada (licence-verified expanded edition) rows blind; Alaska rows an unblinded replication; Holm $m$ = 6 (auxiliary column). "
    "This is a spatial proxy with warm blocks of the same period, not a true extrapolation test. EP(A $-$ B) = $\\Delta_W$(A $-$ B) $-$ $\\Delta_I$(A $-$ B), the contrast in the extrapolation "
    "blocks W minus that in the interpolation blocks I (cm). Extrapolation width (median $\\sqrt{\\mathrm{TDD}}$ of W scoring cells minus the maximum of A, split mean): warm\\_trim 0.97 "
    "($^{\\circ}$C~d)$^{1/2}$ (all W cells outside A); basic warm $-$0.002 ($^{\\circ}$C~d)$^{1/2}$ (58.4\\% outside A); cold 1.74; Alaska warm 0.63. RMSE levels (cell-weighted, $n$ 100 and all): "
    "in W, P1 44.38 and 45.10 cm, R1 41.95 and 42.66 cm, D0 41.99 and 40.32 cm; in I, P1 21.93 and 21.98 cm, R1 21.94 and 21.81 cm, D0 23.00 and 23.59 cm. Alaska warm and Canada cold rows "
    "were undetermined for all three contrasts. Registered sentences (rendered): XI-a, `In a spatial proxy using warm blocks of the same period (extrapolation width 0.97 ($^{\\circ}$C~d)$^{1/2}$ in "
    "$\\sqrt{\\mathrm{TDD}}$), no difference was established in the extrapolation loss of direct ML relative to the recalibrated Stefan model (100 labels, all labels)'; XI-b, `In the same "
    "spatial proxy, no difference was established between the extrapolation losses of the recalibrated-anchor residual and of direct ML (100 labels, all labels)'; with all labels the two "
    "editions disagree (basic edition lower error for XI-a and higher error for XI-b), both are reported and the weaker verdict is used; XI-c, `In the extrapolation blocks of the same proxy, "
    "the recalibrated-anchor residual (cross-validated $\\lambda$) had 2.44 cm (100 labels) and 2.41 cm (all labels) lower error than the recalibrated Stefan model (cell-weighted) and met the "
    "non-inferiority criterion (upper bounds of both CIs below 0.5 cm).' Source: data/processed/xbatch/XI\\_climate\\_extrapolation\\_retest/sealed/ (xi\\_hyp.csv, xi\\_tests.csv, xi\\_holm.csv, "
    "xi\\_curve.csv); gate passed (xi\\_gate.csv).")

# ---------------------------------------------------------------------------------------------- XE (8.6, xh0 only)
XE_H = ["Contrast (within-grid RMSE, R1 with CV $\\lambda$)", "$n$", "Pool", "Three-target mean: cell / block-equal", "Four-way verdict", "Target rows"]
XE = [
    ["XE-c R1(xh0) $-$ R1(x25)", "200", "3/3", "−0.09 [−0.10, +0.06] / +0.06 [+0.01, +0.12]", "equivalent", "Alaska, Canada equivalent; Lena Delta equivalent (margin-dependent, depends on split independence, few blocks)"],
    ["", "500", "partial (2/3)", "−0.16 [−0.18, +0.05] / +0.05 [−0.05, +0.15]", "equivalent", "Alaska equivalent; Lena Delta equivalent (same flags)"],
    ["", "1,000", "partial (2/3)", "−0.17 [−0.22, +0.05] / −0.01 [−0.11, +0.11]", "equivalent", "Alaska equivalent; Lena Delta equivalent (same flags)"],
    ["", "all", "3/3", "−0.12 [−0.16, +0.02] / −0.08 [−0.18, +0.02]", "equivalent", "Alaska, Canada equivalent; Lena Delta undetermined (−0.34 [−0.45, +0.05] / −0.22 [−0.50, +0.06])"],
    ["XE-c R1(xh0) $-$ P1", "200", "3/3", "−0.02 [−0.06, +0.21] / +0.23 [+0.15, +0.30]", "equivalent (margin-dependent)", "Alaska equivalent; Lena Delta undetermined; Canada higher error (+0.33 [+0.25, +0.45] / +0.30 [+0.22, +0.39], size below 0.5 cm)"],
    ["", "500", "partial (2/3)", "−0.24 [−0.30, +0.15] / +0.27 [+0.12, +0.41]", "equivalent (margin-dependent; depends on split independence)", "Alaska equivalent; Lena Delta undetermined"],
    ["", "1,000", "partial (2/3)", "−0.19 [−0.27, +0.33] / +0.36 [+0.18, +0.54]", "undetermined (margin-dependent)", "Alaska equivalent; Lena Delta undetermined"],
    ["", "all", "3/3", "−0.03 [−0.08, +0.35] / +0.29 [+0.16, +0.43]", "equivalent (margin-dependent; depends on split independence)", "Alaska equivalent; Lena Delta undetermined; Canada higher error (+0.25 [+0.16, +0.38] / +0.21 [+0.12, +0.31], size below 0.5 cm)"],
    ["XE-a, XE-b, XE-a0", "", "", PEND_XE, "", ""],
]
XE_NOTE = (
    "Flags: designed after earlier results were viewed; xh0 partly unblinded (the ten H0 inputs were inputs of H19, and the within-grid R1 $-$ P1 contrast of WF9-a had been viewed). "
    "All rows are the auxiliary contrast XE-c (four-way verdict, not counted as confirmatory, outside the XE Holm family). xh0 = the 25 covariates plus ten H0 inputs (topographic wetness "
    "index and its standard deviation, flow accumulation, two curvatures, relief, surface-water occurrence and distance, tree cover). Within-grid share explained, 1 $-$ SSE$_w$(M)/SSE$_w$(P1), "
    "all labels (descriptive, no CI): Alaska 0.07\\% (x25) and 0.69\\% (xh0), Lena Delta $-$0.06\\% and 3.88\\%, Canada $-$2.18\\% and $-$2.15\\%. Total RMSE, x25 to xh0: Alaska 14.06 to 14.00 cm, "
    "Lena Delta 20.89 to 20.31 cm, Canada 29.72 to 26.75 cm (between-grid component 23.62 to 19.75 cm), P1 14.53, 20.62 and 29.10 cm; these total and between-grid contrasts are not registered "
    "and not judged. No pre-fixed sentence applies to XE-c; only facts are reported. The reproduction gate was rescored at the registered local-to-cloud tolerance (gate level corrected, not a "
    "design change). Source: data/processed/xbatch/XE\\_hires\\_covariates/sealed/ (xe\\_r1b\\_xh0\\_tests.csv, xe\\_r1b\\_xh0\\_decomp.csv).")

# ---------------------------------------------------------------------------------------------- XD (8.8, 8.9)
XD_H = ["Hypothesis and contrast", "$n$", "Pool or target", "$\\Delta$ cell [95\\% CI]", "$\\Delta$ block-equal [95\\% CI]", "Four-way verdict", "Non-inferiority ($P\\times2$)", "Holm $P$"]
XD = [
    ["XD-1 S8a $-$ S1", "10", "PE1 (5/5)", "−0.33 [−1.20, +0.66]", "−0.16 [−0.91, +0.57]", "undetermined; hypothesis rejected", "", "1.00"],
    ["", "40", "PE1, partial (2/5)", "−0.70 [−1.61, +0.47]", "−0.96 [−1.76, −0.22]", "undetermined", "", "1.00"],
    ["XD-2 S8w $-$ S8a (post hoc design, Lena target)", "10", "Lena Delta (transfer)", "−1.86 [−3.30, +0.10]", "+0.49 [−0.96, +1.97]", "undetermined; hypothesis rejected (higher error in eight Alaska and Canada rows)", "", "1.00"],
    ["", "40", "Lena Delta (transfer)", "−0.18 [−1.32, +0.67]", "−0.51 [−1.49, +0.41]", "undetermined", "", "1.00"],
    ["XD-3 S8a $-$ S2", "10", "PE1 (5/5)", "+0.32 [+0.01, +0.61]", "+0.28 [+0.00, +0.56]", "higher error (depends on split independence; size below 0.5 cm)", "not met (0.20)", "1.00"],
    ["", "40", "PE1, partial (2/5)", "−0.63 [−1.06, −0.21]", "−0.62 [−0.98, −0.27]", "lower error", "met ($<$0.0001)", "$<$0.0001"],
    ["XD-3 S8a $-$ S4", "10", "PE1 (5/5)", "+0.09 [−0.36, +0.43]", "−0.07 [−0.40, +0.27]", "equivalent (depends on split independence)", "met (0.020; before correction only)", "0.27"],
    ["", "40", "PE1, partial (2/5)", "+0.22 [−0.20, +0.62]", "+0.06 [−0.27, +0.39]", "undetermined (margin-dependent; depends on split independence)", "not met (0.16)", "1.00"],
    ["XD-3 S8b $-$ S2", "10", "PE1 (5/5)", "+0.45 [+0.12, +0.79]", "+0.42 [+0.13, +0.72]", "higher error (size below 0.5 cm)", "not met (0.75)", "1.00"],
    ["", "40", "PE1, partial (2/5)", "−0.48 [−0.85, −0.08]", "−0.47 [−0.80, −0.17]", "lower error", "met ($<$0.0001)", "$<$0.0001"],
    ["XD-3 S8b $-$ S4", "10", "PE1 (5/5)", "+0.22 [−0.15, +0.53]", "+0.07 [−0.22, +0.38]", "undetermined (margin-dependent; draw-dependent)", "not met (0.074)", "0.89"],
    ["", "40", "PE1, partial (2/5)", "+0.38 [−0.01, +0.79]", "+0.21 [−0.13, +0.57]", "undetermined (margin-dependent)", "not met (0.61)", "1.00"],
    ["XD-3 S8c $-$ S2", "10", "PE1 (5/5)", "−0.36 [−0.87, +0.01]", "−0.34 [−0.71, +0.03]", "undetermined (margin-dependent; draw-dependent)", "met ($<$0.0001)", "$<$0.0001"],
    ["", "40", "PE1, partial (2/5)", "−1.13 [−1.67, −0.60]", "−0.90 [−1.37, −0.42]", "lower error", "met ($<$0.0001)", "$<$0.0001"],
    ["XD-3 S8c $-$ S4", "10", "PE1 (5/5)", "−0.59 [−1.21, −0.19]", "−0.68 [−1.09, −0.29]", "lower error (depends on split independence)", "met ($<$0.0001)", "$<$0.0001"],
    ["", "40", "PE1, partial (2/5)", "−0.28 [−0.57, −0.00]", "−0.22 [−0.47, +0.04]", "undetermined (margin-dependent; draw-dependent)", "met ($<$0.0001)", "$<$0.0001"],
    ["XD-3 S8p $-$ S2", "10", "PE1 (5/5)", "+0.52 [−0.37, +1.28]", "+0.31 [−0.44, +1.06]", "undetermined", "not met (0.93)", "1.00"],
    ["", "40", "PE1, partial (2/5)", "−0.75 [−1.92, +0.12]", "−0.50 [−1.33, +0.34]", "undetermined", "met (0.019; before correction only)", "0.27"],
    ["XD-3 S8p $-$ S4", "10", "PE1 (5/5)", "+0.29 [−0.49, +0.88]", "−0.04 [−0.63, +0.57]", "undetermined (margin-dependent)", "not met (0.39)", "1.00"],
    ["", "40", "PE1, partial (2/5)", "+0.10 [−0.92, +0.77]", "+0.18 [−0.42, +0.78]", "undetermined (margin-dependent)", "not met (0.31)", "1.00"],
]
XD4_H = ["Test task or family", "$v$ cell-weighted", "$v$ block-equal", "Direction"]
XD4 = [
    ["LE-1", "+0.408", "+0.149", "higher error in both weightings"],
    ["LE-2", "−0.027", "−0.006", "lower error in both weightings"],
    ["CA-2", "−0.008", "+0.005", "weightings disagree"],
    ["CA-3", "−0.036", "−0.004", "lower error in both weightings"],
    ["Tibetan Plateau", "+0.132", "+0.086", "higher error in both weightings"],
    ["Central Russia (not counted; no $n$ 40)", "+0.031", "+0.018", "descriptive"],
    ["Family Lena Delta (LE-1, LE-2)", "+0.190", "+0.071", "higher error in both weightings"],
    ["Family Canada (CA-2, CA-3)", "−0.022", "+0.0002", "not consistent"],
    ["Family Tibetan Plateau", "+0.132", "+0.086", "higher error in both weightings"],
]
XD6_H = ["Target (transfer)", "Blocks", "Effective blocks", "Covariate between-block share", "ln(cell-weighted $E$ / block-mean $E$)", "S8a $-$ S1, $n$ 10", "S8a $-$ S1, $n$ 40"]
XD6 = [
    ["Alaska", "36.2", "3.49", "0.78", "+0.095", "−0.96 / −1.97 (lower error)", "−1.14 / −2.23 (lower error)"],
    ["Canada", "17.8", "3.89", "0.83", "−0.044", "−0.81 / +0.52 (undetermined)", "−2.39 / −1.71 (lower error)"],
    ["Lena Delta", "7.6", "1.72", "0.39", "−0.069", "+1.70 / +1.02 (undetermined)", "+0.98 / −0.22 (undetermined)"],
    ["W Russia", "10.6", "7.73", "0.92", "+0.114", "−0.06 / −0.05 (undetermined)", ""],
    ["E Russia", "10.8", "7.81", "0.90", "−0.301", "+0.56 / −0.72 (undetermined)", ""],
    ["Central Russia", "5.4", "2.87", "0.76", "+0.075", "−3.03 / −1.55 (undetermined)", ""],
    ["Tibetan Plateau", "16.8", "10.25", "0.78", "+0.009", "+2.30 / +2.37 (undetermined)", "+3.01 / +1.15 (undetermined)"],
    ["AL-1", "3.75", "1.26", "0.35", "+0.006", "+0.35 / −1.65 (undetermined)", "−1.11 / −1.82 (lower error)"],
    ["AL-2", "8.6", "1.17", "0.32", "−0.102", "−0.31 / −0.71 (lower error)", "−0.34 / −0.81 (lower error)"],
    ["AL-3", "10.2", "2.89", "0.71", "−0.009", "+1.43 / +1.30 (higher error)", "+0.35 / +0.15 (undetermined)"],
    ["AL-4", "1.33", "1.00", "0.18", "−0.001", "not decidable", "not decidable"],
    ["AL-5", "6.8", "1.95", "0.72", "+0.204", "−2.10 / −2.90 (lower error)", "−2.39 / −4.30 (undetermined)"],
    ["AL-6", "3.25", "1.00", "0.14", "−0.000", "not decidable", "not decidable"],
]
XD_NOTE = (
    "Flags: exploratory (Supplementary Information); designed after earlier results were viewed; XD-1 and XD-3 partly unblinded (S2 and S4 of WF8 viewed); XD-2 partly unblinded and a post hoc "
    "design targeting the Lena Delta loss; XD-4 blind (placements of the test tasks never computed before), descriptive, no verdict words and no $P$ values (three families; minimum family-level "
    "sign-test $P$ 0.125). Contrasts use R1 ($\\lambda$ 0.25); contrasts between different label sets use two-stage CIs, S8w $-$ S8a uses block resampling with the same labels; Holm $m$ = 20 "
    "(auxiliary column). S8a, $\\gamma$ 0 with farthest-first block order and the cell nearest the block centre first; S8b, farthest-first within blocks; S8c, $\\gamma$ 0.5; S8p, $\\gamma$ 1; "
    "S8w, S8a labels with design-weighted coefficients. Algorithm P (appendix XC-0): rule 2 excluded S2, S4, S8a, S8b, S8c and S8p on Alaskan targets, so Algorithm P = S1 (cell-random "
    "sampling); XC-5b, XC-5c and S-XC3 were therefore not tested and remain in the XC Holm family with $P$ = 1. XD-2 higher-error rows: Alaska transfer $n$ 10 +2.02 [+1.36, +2.80] / +1.94 "
    "[+1.02, +2.81], $n$ 40 +1.44 / +3.06; Canada transfer $n$ 10 +4.89 [+1.07, +6.65] / +2.82, $n$ 40 +1.99 / +1.21 (both depend on split independence); Alaska within region $n$ 20 +1.21 / +2.35; "
    "Canada within region $n$ 50 +1.78 / +1.07, $n$ 100 +1.30 / +0.87 (both depend on split independence), $n$ 200 +1.02 / +0.86. Post hoc rows outside the registered hypotheses (not counted): "
    "S8c $-$ S1 lower error in the four-way verdict at $n$ 10 and 40 but undetermined with the two-stage common CI; these rows do not change Algorithm P. XD-4: $v$ = mean over splits, draws, "
    "$n \\in \\{20, 40\\}$ and policy seeds of (RMSE(S9*) $-$ RMSE(Algorithm P)) / RMSE(Algorithm P); task RMSE differences (cm, cell-weighted [two-stage CI] (block-equal)): LE-1 $n$ 20 +7.60 "
    "[+0.61, +9.12] (+2.43), $n$ 40 +8.82 [+1.30, +10.43] (+3.49); LE-2 $n$ 20 −0.62, $n$ 40 −0.29; CA-2 $n$ 20 +0.19 [−2.01, +2.14], $n$ 40 −0.34 [−1.31, +1.23]; CA-3 $n$ 20 −1.11 [−1.89, +0.30], "
    "$n$ 40 −1.18 [−2.06, +0.20]; Tibetan Plateau $n$ 20 +21.72 [+16.47, +26.48] (+13.65), $n$ 40 +7.45 [+4.03, +10.32] (+2.65). Registered sentences (rendered): XD-1, `No difference between "
    "placement combining block allocation and covariate coverage and random placement was established with 10 labels' and `... with 40 labels' (transfer, PE1); XD-2, `Design-weighted "
    "coefficients increased error in Alaska (transfer $n$ 10 and 40, within region $n$ 20) and Canada (transfer $n$ 10 and 40, within region $n$ 50, 100 and 200)' (post hoc design, Lena "
    "target); XD-3, `The error increase of S8 variant V relative to S2 (or S4) was within 0.5 cm' for S8a $-$ S2 ($n$ 40), S8a $-$ S4 ($n$ 10; significant before correction only), S8b $-$ S2 "
    "($n$ 40), S8c $-$ S2 ($n$ 10, 40), S8c $-$ S4 ($n$ 10, 40) and S8p $-$ S2 ($n$ 40; significant before correction only), `... was larger' for S8a $-$ S2 and S8b $-$ S2 at $n$ 10 "
    "(statistically distinguishable but below 0.5 cm in size), and `non-inferiority was not established' for the six other rows; XD-4, `A placement policy learned and frozen in Alaskan "
    "sub-regions had lower error than Algorithm P in 2 of 5 test tasks outside Alaska (0 of 3 families; both weightings, no test)' and `... higher error in 2 of 5 test tasks (2 of 3 families; "
    "both weightings, no test).' Common limitation: the candidate pool consists of already measured cells, so gains are not guaranteed to transfer to field candidates (all land grid cells). "
    "XD-6 is descriptive (trait means over splits; effects as cell-weighted / block-equal cm); correlations of traits with effects were not computed. XD-5: part k (continued). "
    "Source: data/processed/xbatch/XD\\_placement\\_policy/sealed/ (xd\\_alg\\_tests.csv, xd\\_alg\\_contrasts.csv, xd\\_alg\\_holm.csv, xd6\\_traits.csv; xd4\\_summary.json, xd4\\_tasks.csv, "
    "xd4\\_families.csv, xd4\\_ci.csv); gate passed (gate/xd\\_gate.csv).")

# ---------------------------------------------------------------------------------------------- XJ (8.5)
XJ_H = ["Hypothesis", "$w$", "$n$", "$\\Delta$ cell [95\\% CI]", "$\\Delta$ block-equal [95\\% CI]", "Four-way verdict", "Holm $P$ ($m$ 15)"]
XJ = [
    ["XJ-1 R1@$w$ $-$ R1, four main regions, $n$ 0 and 10, $w$ 0.1, 0.3, 1 (six rows)", "", "", "", "", "not tested (licence basis not recorded)", "1.00"],
    ["XJ-2 D1@$w$ $-$ D1, $n$ 0, $w$ 0.1, 0.3, 1 (three rows)", "", "", "", "", "not tested (licence basis not recorded)", "1.00"],
    ["XJ-3 P1\\_aux($w$) $-$ P1, Tibetan Plateau", "0.1", "3", "−20.85 [−21.80, −19.89]", "−21.81 [−23.20, −20.11]", "lower error", "0.0015"],
    ["", "0.1", "10", "−8.14 [−8.88, −7.10]", "−5.94 [−7.67, −4.02]", "lower error", "0.0015"],
    ["", "0.3", "3", "−47.54 [−50.12, −44.50]", "−45.18 [−50.11, −39.44]", "lower error", "0.0015"],
    ["", "0.3", "10", "−19.63 [−21.74, −16.53]", "−12.66 [−17.30, −7.67]", "lower error", "0.0015"],
    ["", "1", "3", "−83.17 [−90.21, −73.41]", "−66.86 [−79.47, −53.35]", "lower error", "0.0015"],
    ["", "1", "10", "−37.50 [−43.30, −28.88]", "−20.57 [−31.04, −9.55]", "lower error", "0.004"],
]
XJ_NOTE = (
    "Flags: designed after earlier results were viewed; XJ-3 partly unblinded (L39). Target-side auxiliary rows: 39 temperature-derived rows (F3, CC BY 4.0), of which rows outside the "
    "B blocks of a split entered the A pool (15, 16, 8, 24 and 28 rows in splits 1 to 5); only direct-label cells are scored. Baseline RMSE (cell-weighted): P0 244.51 cm; P1 201.05 cm "
    "($n$ 3) and 148.14 cm ($n$ 10); P1\\_aux($w$ 1) 117.89 and 110.63 cm. Descriptive $n$ 0 rows: $-$34.85 ($w$ 0.1), $-$75.81 (0.3) and $-$124.60 cm (1). Registered sentences (rendered): "
    "XJ-3, `Temperature-derived auxiliary rows (weight $w$) reduced the transfer error with $k$ labels by $a$ cm. Temperature-derived labels differ in definition from direct measurements (L39), "
    "so this result is not generalized as evidence for label augmentation', with ($w$, $k$, $a$) = (0.1, 3, 20.85), (0.1, 10, 8.14), (0.3, 3, 47.54), (0.3, 10, 19.63), (1, 3, 83.17), (1, 10, 37.50); "
    "XJ-1 and XJ-2, `The source-side design of temperature-derived auxiliary rows was not tested (licence basis not recorded).' All rows concern one target (Tibetan Plateau). "
    "Source: data/processed/xbatch/XJ\\_tempderived\\_aux\\_labels/sealed/ (xj\\_hyp.csv, xj\\_tests.csv, xj\\_holm.csv); gate passed (xj\\_gate.csv).")

# ---------------------------------------------------------------------------------------------- XK (addendum)
XK_H = ["Region", "Grid", "Grid cells", "Blocks", "P1 cell / block", "R1 cell / block", "P1 $-$ R1 cell [95\\% CI] / block-equal [95\\% CI]", "CCI v4 cell / block"]
XK = [
    ["Alaska", "1 km", "13,606", "74", "14.30 / 18.16", "13.80 / 17.31", "0.50 [0.18, 1.10] / 0.86 [0.29, 1.54]", "19.90 / 28.22"],
    ["", "0.05°", "105", "36", "12.94 / 15.29", "11.27 / 13.60", "1.67 [0.60, 2.69] / 1.70 [0.67, 2.68]", "20.96 / 26.78"],
    ["", "0.1°", "83", "35", "12.76 / 15.13", "11.22 / 13.40", "1.54 [0.50, 2.54] / 1.72 [0.68, 2.69]", "21.30 / 26.66"],
    ["", "0.25°", "56", "35", "13.16 / 14.78", "11.55 / 13.05", "1.61 [0.54, 2.72] / 1.72 [0.64, 2.81]", "23.13 / 26.35"],
    ["Lena Delta", "1 km", "2,958", "20", "21.02 / 34.20", "21.01 / 32.70", "0.01 [−1.39, 1.41] / 1.50 [−0.38, 3.20]", "22.45 / 40.23"],
    ["", "0.05°", "56", "13", "24.66 / 26.74", "24.97 / 25.90", "−0.30 [−3.32, 1.32] / 0.84 [−2.19, 3.48]", "28.65 / 31.17"],
    ["", "0.1°", "37", "12", "26.38 / 31.93", "26.75 / 31.20", "−0.38 [−3.77, 1.72] / 0.72 [−2.23, 3.70]", "30.62 / 36.25"],
    ["", "0.25°", "22", "12", "15.85 / 18.64", "15.07 / 17.39", "0.78 [−1.25, 3.04] / 1.25 [−1.60, 4.26]", "20.14 / 23.33"],
    ["Canada", "1 km", "747", "36", "28.84 / 27.00", "26.22 / 26.64", "2.62 [0.00, 4.42] / 0.36 [−3.38, 3.34]", "30.05 / 40.77"],
    ["", "0.05°", "37", "16", "20.29 / 18.75", "19.18 / 16.40", "1.11 [−1.15, 4.49] / 2.34 [0.34, 4.64]", "32.50 / 29.61"],
    ["", "0.1°", "26", "15", "20.16 / 19.27", "18.80 / 16.84", "1.35 [−0.50, 4.22] / 2.43 [0.53, 4.70]", "31.78 / 29.51"],
    ["", "0.25°", "19", "14", "15.85 / 16.27", "14.89 / 13.53", "0.96 [−1.93, 4.45] / 2.74 [−0.04, 4.87]", "33.87 / 28.93"],
]
XKP_H = ["Region (cells)", "Grid", "CCI v4", "CCI v5", "Wei 2026", "Wei 2026 (5 km mask)", "Yi-Kimball"]
XKP = [
    ["Alaska (13,605)", "1 km", "6.10 [2.54, 9.11] / 10.87", "54.22 [12.08, 90.76] / 53.21", "2.23 [0.69, 5.01] / −0.94", "3.02 [0.83, 5.73] / 3.32 (9,131 cells)", "46.09 [10.97, 99.91] / 51.90 (10,605 cells)"],
    ["", "0.1°", "10.08 [6.42, 14.12] / 13.26", "53.20 [34.47, 71.94] / 63.02", "3.88 [0.77, 6.78] / 3.14", "4.03 [0.74, 7.71] / 3.08", "39.10 [26.43, 52.86] / 48.49"],
    ["Lena Delta (1,342)", "1 km", "0.78 [−0.73, 5.49] / 7.69", "−0.31 [−1.87, 1.44] / 1.79", "9.91 [−4.04, 14.05] / −2.08", "10.00 [−4.81, 14.13] / −2.03 (1,330 cells)", ""],
    ["", "0.1°", "3.14 [−0.04, 6.40] / 4.08", "−0.61 [−4.72, 2.64] / 0.09", "−1.14 [−6.72, 8.57] / −1.33", "−1.26 [−7.10, 9.24] / −1.32", ""],
    ["Canada (745)", "1 km", "3.76 [−2.75, 16.12] / 13.75", "54.25 [23.70, 78.39] / 61.55", "2.99 [−3.66, 10.78] / −2.61", "3.32 [−4.23, 11.33] / 7.36 (726 cells)", ""],
    ["", "0.1°", "12.98 [1.58, 23.37] / 12.67", "60.09 [31.09, 87.72] / 67.23", "5.42 [−0.14, 13.07] / 7.27", "5.42 [0.36, 13.70] / 7.27", ""],
]
XK_NOTE = (
    "Flags: added registration of 5 October 2026 (commit d721d24, before computation); designed after earlier results were viewed; descriptive, no verdict words. Out-of-block predictions "
    "from 0.5° block five-fold cross-validation; grid cells with fewer than three label cells dropped; 1,000 block resamples. First table: models set (cells where P1, R1 and CCI v4 are finite), "
    "no mask; RMSE in cm, cell-weighted / block-equal. Second table: RMSE(product) $-$ RMSE(R1) in cm, cell-weighted [95\\% CI] / block-equal, common set (all product values finite; Alaska "
    "Yi-Kimball on its own set), no mask, plus the 5 km mask edition for Wei 2026. Bias at label cells (product $-$ label, 1 km, cell-weighted, common set), reported as fact without a cause: "
    "CCI v5 +35.63 cm (Alaska), +58.60 cm (Canada) and +0.27 cm (Lena Delta); CCI v4 $-$5.09, +10.35 and $-$6.04 cm. Pre-fixed interpretation rules: rule 1 (all RMSE decrease with grid size) "
    "does not hold, so its sentence is not used; rule 2 does not fix the weighting and the two weightings pick different grid sizes in all three regions, so only the fact is reported: "
    "`In Alaska the RMSE difference between P1 and R1 was 0.50 cm at 1 km and 1.54 to 1.67 cm at 0.05°, 0.1° and 0.25° (cell-weighted).' Products are compared only as RMSE differences at the same "
    "grid size, for example `At the 0.1° grid in Alaska, the RMSE of CCI v4, CCI v5, Wei 2026 and Yi-Kimball was larger than that of R1 by 10.08, 53.20, 3.88 and 39.10 cm' and `At the 0.1° grid "
    "in the Lena Delta, the RMSE of CCI v5 and Wei 2026 was smaller than that of R1 by 0.61 and 1.14 cm (CIs include zero)' (cell-weighted); product rankings are not generalized, and Wei 2026 "
    "used CALM sites for training (WRAPUP 10). Supplementary Fig.~S15. Source: data/processed/xbatch/XK\\_support\\_scale/xk\\_support\\_scale\\_v1.csv and xk\\_meta.json.")

# ---------------------------------------------------------------------------------------------- XL (addendum)
XL_H = ["Region", "Map", "Mean", "Quartiles (25--75\\%)", "Mean difference", "Mean absolute difference", "Pearson $r$", "Spearman $\\rho$", "Share of mapped cells with values"]
XL = [
    ["Alaska", "this study (recalibrated anchor plus residual ML)", "56.09", "48.41–63.61", "", "", "", "", ""],
    ["", "recalibrated Stefan", "57.46", "48.92–66.06", "−1.38", "2.57", "0.956", "0.961", "1.000"],
    ["", "this map, 0.1° cell means", "56.09", "48.49–63.53", "0.00", "1.37", "0.983", "0.983", "1.000"],
    ["", "CCI v5 (1997–2023 mean)", "105.46", "61.30–133.89", "−49.37", "50.45", "0.464", "0.642", "0.999"],
    ["", "Wei 2026 (2000–2024 mean)", "70.79", "57.64–71.53", "−14.66", "18.14", "−0.154", "0.090", "0.994"],
    ["", "Aalto 2018 (2000–2014)", "78.82", "60.40–102.61", "−22.63", "37.16", "−0.104", "0.106", "0.991"],
    ["", "Yi-Kimball (2001–2015 mean)", "105.15", "54.33–137.00", "−49.33", "53.37", "0.338", "0.461", "0.862"],
    ["Lena Delta", "this study", "42.18", "40.90–43.19", "", "", "", "", ""],
    ["", "recalibrated Stefan", "41.09", "40.12–42.10", "1.10", "1.61", "0.361", "0.387", "1.000"],
    ["", "this map, 0.1° cell means", "42.18", "41.14–43.16", "0.00", "0.81", "0.783", "0.816", "1.000"],
    ["", "CCI v5", "41.24", "32.96–48.15", "0.94", "8.99", "−0.226", "−0.262", "1.000"],
    ["", "Wei 2026", "58.64", "53.32–63.07", "−16.50", "16.51", "0.045", "0.068", "0.940"],
    ["", "Aalto 2018", "55.16", "48.81–60.46", "−12.98", "13.09", "−0.015", "−0.023", "0.986"],
]
XL_NOTE = (
    "Flags: added registration of 5 October 2026 (commit d721d24, before computation); designed after earlier results were viewed; descriptive, no verdict. Mapped cells (cm; difference = this "
    "map minus the compared map): Alaska 858,517 of 899,633 domain cells (permafrost fraction at least 50\\%), Lena Delta 38,086 of 53,011. The 0.1° cell means explain 97\\% (Alaska, $r$ 0.983) "
    "and 61\\% (Lena Delta, $r$ 0.783) of the variance of the 1 km map; the standard deviation of 1 km values within a 0.1° cell is 1.8 cm in Alaska (total 9.5 cm) and 1.1 cm in the Lena Delta "
    "(total 1.8 cm). Registered map sentence (rendered): `The 1 km map is a display resolution; its information resolution is bounded by the climate (about 10 km) and soil (about 5 km) "
    "inputs.' The maps carry no accuracy claim; accuracy is compared only at label cells (XG and XK). Areas of large difference are named without a cause: Interior Alaska (CCI v5 and "
    "Yi-Kimball, more than 50 cm), about 61–63° N, 141–150° W (Wei 2026), and west of about 148° W at 62–64° N (this map deeper than Aalto 2018); in the Lena Delta, the northern and eastern delta "
    "(this map deeper than CCI v5) and most of the grid (this map shallower than Wei 2026 and Aalto 2018). In Alaska, 73.2\\% of mapped cells lie outside the training range of at least one of "
    "the 25 covariates (mainly terrain roughness, slope and topographic position); the constant-width split-conformal interval is $\\pm$20.15 cm (Alaska) and $\\pm$25.39 cm (Lena Delta), and its "
    "coverage is not guaranteed in extrapolated cells. Supplementary Figs~S10 to S14. Source: data/processed/xbatch/XL\\_map\\_products/xl\\_summary\\_v1.csv and xl\\_meta.json.")


# ---------------------------------------------------------------------------------------------- XB sensitivity (8.12)
XBS_H = ["Hypothesis", "$n$", "Main-run verdict", "CCI v5 edition: cell / block-equal [95\\% CI]", "Verdict, CCI v5 edition", "Holm $P$ (CCI v5 edition)"]
XBS = [
    ["XB-1 Stack $-$ P1", "10", "undetermined", "+0.27 [−0.78, +1.84] / +1.64 [+0.68, +2.67]", "undetermined (fallback 0.37)", "1.00"],
    ["", "40", "undetermined", "−2.17 [−2.79, −0.41] / +0.70 [+0.06, +1.33]", "undetermined (CIs exclude zero on opposite sides)", "0.33"],
    ["", "160", "undetermined", "−2.79 [−3.43, −1.06] / +0.38 [−0.43, +1.19]", "undetermined", "1.00"],
    ["", "all", "undetermined", "−1.57 [−2.24, −0.44] / −0.04 [−0.82, +0.75]", "undetermined", "1.00"],
    ["XB-1 hypothesis", "", "rejected", "", "rejected", ""],
    ["XB-2 StackR(0.25) $-$ R1(0.25)", "10", "undetermined", "+0.16 [−0.84, +1.64] / +1.39 [+0.55, +2.28]", "undetermined", "1.00"],
    ["", "40", "undetermined", "−2.02 [−2.60, −0.36] / +0.65 [+0.07, +1.22]", "undetermined (opposite sides)", "0.33"],
    ["", "160", "undetermined", "−2.35 [−2.92, −0.74] / +0.53 [−0.21, +1.28]", "undetermined", "1.00"],
    ["", "all", "undetermined", "−1.42 [−1.98, −0.38] / −0.04 [−0.71, +0.64]", "undetermined", "1.00"],
    ["XB-2 hypothesis", "", "rejected", "", "rejected", ""],
    ["XB-3 StackR $-$ R1 (CV $\\lambda$), within region", "200", "undetermined", "−1.24 [−1.34, −0.67] / +0.32 [+0.11, +0.54]", "undetermined (opposite sides)", "0.031"],
    ["", "500", "lower error", "−0.50 [−0.62, −0.19] / −0.06 [−0.24, +0.11]", "undetermined (margin-dependent)", "1.00"],
    ["", "1,000", "lower error", "−0.36 [−0.54, −0.03] / +0.08 [−0.14, +0.30]", "undetermined (margin-dependent)", "1.00"],
    ["", "all", "undetermined", "−1.44 [−1.59, −0.74] / +0.17 [−0.10, +0.43]", "undetermined", "1.00"],
    ["XB-3 hypothesis", "", "partially supported", "", "rejected", ""],
    ["XB-4 Stack $-$ P1, non-inferiority, Alaska", "200 to all", "non-inferior at all four $n$", "all −0.70 [−0.82, −0.49] / −0.92 [−1.13, −0.73]; $n$ 200 −0.46 / −0.75", "non-inferior at all four $n$ (four-way lower error)", "$<$0.0001"],
    ["XB-4 hypothesis", "", "supported", "", "supported", ""],
]
XBS_NOTE = (
    "Sensitivity edition registered in plan 2.2 as a second-stage candidate: the year-matched CCI v5 map (affine-calibrated; missing cells take the P1 value) added as an "
    "eighth candidate, not as a replacement ($K$ = 8 in all 219 store-by-$n$ rows). Local run on 5 October 2026, 05:20 to 05:38 KST (211 units, no failure); the keys shared "
    "with the main run agreed within 6.0$\\times$10$^{-14}$ cm; sealed tables first opened 05:43:31 KST. Flags as in the main run; designed after earlier results were "
    "viewed. The registered verdicts remain those of the main run (XB-1 and XB-2 rejected, XB-3 partially supported, XB-4 supported). In this edition XB-1, XB-2 and XB-4 "
    "kept their verdicts, and XB-3 was not retained: the lower error at 500 and 1,000 labels became undetermined (block-equal point estimates −0.06 and +0.08 cm). "
    "Regional rows: XB-1 Lena Delta at $n$ 10 changed from undetermined to higher error (+0.31 / +1.98 cm), XB-1 Canada at $n$ 160 and all labels from undetermined to lower error "
    "(−5.54 / −1.59 and −6.03 / −1.71 cm), XB-3 Alaska and Lena Delta within region at $n$ 1,000 from lower error to undetermined and Alaska with all labels from equivalent to "
    "undetermined; XB-2 and XB-4 regional rows did not change. Weights (descriptive, no causal statement): mean CCI v5 weight with all labels, Canada 0.21, Lena Delta 0.23, "
    "Tibetan Plateau 0.32 and Alaska 0.03 in transfer, Canada 0.46, Lena Delta 0.10 and Alaska 0.10 within regions; Spearman $\\rho$ of the total CCI weight with the P1 RMSE "
    "0.35 ($P$ 0.051, 31 transfer stores) and 0.50 ($P$ 0.67, three within-region targets). Source: data/processed/xbatch/XB\\_multisource\\_stacking/sealed/ "
    "(xb\\_tests\\_cci5y.csv, xb\\_weights\\_summary\\_cci5y.csv, xb\\_cci\\_relation\\_cci5y.csv); plan 8.12.")

# ---------------------------------------------------------------------------------------------- XE-e (8.6, end)
XEE_H = ["Region", "Model", "Non-GPR VWC, registered subset (main)", "Non-GPR VWC, finite-predictor cells", "GPR VWC, registered subset", "GPR VWC, finite cells"]
XEE = [
    ["Alaska", "linear regression", "−3.54\\%", "−0.79\\%", "−0.54\\%", "−5.56\\%"],
    ["", "CatBoost (mean of seeds 0 and 1)", "−4.89\\%", "−2.40\\%", "+0.82\\%", "−4.16\\%"],
    ["Canada", "linear regression", "−4.58\\%", "+2.55\\%", "n.a.", "n.a."],
    ["", "CatBoost", "−4.48\\%", "−6.59\\%", "n.a.", "n.a."],
]
XEE_NOTE = (
    "XE-e, diagnostic (exploratory): share of the within-grid residual of the recalibrated Stefan model P1 explained by on-site volumetric water content (shallow and deep "
    "layers) under five-fold block cross-validation. Flags: designed after earlier results were viewed; blind (the relation of VWC with P1 residuals had not been computed); "
    "the registered rule says XE-e changes no verdict. Site data: ABoVE soil thaw depth and moisture validation, version 2 (15 m radius). Registered subsets: Alaska 2,468 cells "
    "in 31 blocks, Canada 602 cells in 15 blocks; shares computed on cells in grid groups of at least two cells (2,466 and 599); non-GPR VWC finite in 1,329 and 595 cells. "
    "GPR VWC shares error with ALT and is a sensitivity only; Canada has no GPR VWC (n.a.). Reference: a 0.5 cm reduction of within-grid RMSE corresponds to explained shares of "
    "7.9\\% (Alaska) and 5.0\\% (Canada). Description: under block cross-validation on-site VWC did not explain the within-grid residual of P1 (main shares −4.89\\% to −3.54\\%, "
    "sensitivities −6.59\\% to +2.55\\%). Run locally on 5 October 2026 (05:41, 8 s); sealed table first opened 05:42:35 KST. Source: data/processed/xbatch/XE\\_hires\\_covariates/"
    "sealed/xe\\_e\\_xe\\_e.csv and xe\\_e\\_xe\\_e\\_meta.json; plan 8.6.")

# ---------------------------------------------------------------------------------------------- XM (addendum XM, results)
XM_H = ["Hypothesis", "Component", "$n$", "Pool or target", "$\\Delta$ cell [95\\% CI]", "$\\Delta$ block-equal [95\\% CI]", "Four-way verdict"]
XM = [
    ["XM-a (main)", "within-grid", "all", "three-target mean (3/3)", "−0.15 [−0.25, −0.05]", "−0.25 [−0.35, −0.14]", "lower error (size below 0.5 cm); Holm $P$ 0.019"],
    ["", "within-grid", "all", "Alaska / Lena Delta / Canada", "−0.03 / −0.52 / +0.09", "−0.05 / −0.72 / +0.03", "lower error / lower error / equivalent"],
    ["", "within-grid", "1,000", "Alaska and Lena Delta mean (partial, 2/3)", "−0.21 [−0.28, −0.03]", "−0.25 [−0.39, −0.11]", "lower error (size below 0.5 cm); Holm $P$ 0.04"],
    ["", "within-grid", "500", "Alaska and Lena Delta mean (partial, 2/3)", "−0.17 [−0.20, +0.03]", "−0.13 [−0.25, −0.01]", "equivalent; Holm $P$ 0.135"],
    ["XM-b (auxiliary)", "total", "all", "three-target mean (3/3)", "−0.70 [−0.93, −0.44]", "−0.74 [−1.03, −0.44]", "lower error"],
    ["", "total", "all", "Alaska / Lena Delta / Canada", "−0.12 / −1.50 / −0.49", "+0.03 / −2.08 / −0.16", "equivalent / lower error / undetermined"],
    ["XM-c (auxiliary)", "between-grid", "all", "Lena Delta", "−1.80 [−2.35, −1.10]", "−2.11 [−2.96, −1.25]", "lower error"],
]
XM_NOTE = (
    "XM, added registration (docs/EXPERIMENT\\_PLAN\\_FINAL\\_BATCH\\_ADDENDUM\\_XM\\_2026-10-05.md; commit ff7c97f, 5 October 2026, 13:23 KST, before feature extraction). "
    "Flags: designed after earlier results were viewed (XE xh0 and XE-e tables); exploratory (Supplementary Information); partly unblinded (the tree-cover and water-occurrence "
    "inputs of xh0 overlap with the WorldCover tree and water fractions, and the xh0 within-grid contrast had been viewed); outside the XE Holm family. Inputs: xw = the 25 "
    "covariates plus the fractions of 11 ESA WorldCover 2021 v200 classes (10 m) in the 1 km label cell and three Sentinel-2 L2A (20 m) summer indices (July and August 2019 to "
    "2023; cell median NDVI, median NDMI and NDVI standard deviation); the WorldCover-only variant xw\\_lc is auxiliary and not shown. Contrast: R1 with $\\lambda$ chosen by block "
    "cross-validation, xw minus x25, on the same labels, splits and seeds; components as in Methods. Holm family: XM-a at $n$ 500, 1,000 and all ($m$ = 3). Verdict: XM-a partially "
    "supported (lower error at 1,000 and all labels, equivalent at 500). Registered sentence (rendered): `Adding the WorldCover 10 m land-cover fractions and the Sentinel-2 20 m "
    "summer vegetation indices of the 1 km cell reduced the ERA5-Land within-grid error of R1 by 0.15 cm (within-grid share explained 1.2\\%) (statistically distinguishable, size "
    "below 0.5 cm). As an exploratory result it does not change the within-grid statement or the map wording of the main text and is reported as a candidate for a confirmatory "
    "test. XM does not distinguish the causes of the remaining variation.' Facts, without interpretation: the total RMSE gain was concentrated in the Lena Delta (−1.50 cm), mostly "
    "in the between-grid component (−1.80 cm); in Alaska and Canada the total RMSE change was within 0.5 cm or undetermined. Coverage of label rows: WorldCover 100\\% in all three "
    "targets; Sentinel-2 100\\% (Alaska), 99.28\\% (Lena Delta) and 99.87\\% (Canada); no group was dropped by the 90\\% rule, so xw is the main variant. Sealed tables first opened "
    "5 October 2026, 14:20:47 KST, after the reproduction gate of the re-used x25 shards passed (35,442 keys, no failure). Source: data/processed/xbatch/XM\\_landcover\\_vegetation/"
    "sealed/ (xm\\_tests.csv, xm\\_hypotheses.csv, xm\\_holm.csv); addendum XM, results.")

# ---------------------------------------------------------------------------------------------- XD-5 (8.11)
XD5_H = ["Test task", "U(S9*)", "U(Algorithm P = S1)", "U(oracle)", "Median of 200 candidate sets", "Regret, S9*", "Regret, Algorithm P"]
XD5 = [
    ["CA-2", "−0.005", "0", "−0.089", "+0.002", "0.084", "0.089"],
    ["CA-3", "−0.024", "0", "−0.085", "−0.028", "0.061", "0.085"],
    ["LE-1", "+0.275", "0", "−0.069", "+0.001", "0.344", "0.069"],
    ["LE-2", "−0.017", "0", "−0.051", "−0.024", "0.034", "0.051"],
    ["Tibetan Plateau", "+0.106", "0", "−0.090", "−0.009", "0.195", "0.090"],
    ["Central Russia (not counted)", "+0.023", "0", "n.a.", "n.a.", "n.a.", "n.a."],
]
XD5CV_H = ["Task", "Family", "S9-DS, leave family out", "S9-DS, leave task out", "S9-GBM, leave family out", "S9-GBM, leave task out"]
XD5CV = [
    ["LE-1", "Lena Delta", "+0.074", "+0.034", "−0.024", "−0.013"],
    ["LE-2", "Lena Delta", "−0.015", "−0.008", "−0.029", "−0.020"],
    ["CA-2", "Canada", "+0.066", "+0.081", "−0.004", "+0.082"],
    ["CA-3", "Canada", "−0.031", "−0.037", "−0.044", "−0.025"],
    ["Tibetan Plateau", "Tibetan Plateau", "+0.053", "+0.072", "−0.009", "−0.009"],
]
XD5_NOTE = (
    "XD-5, descriptive (Supplementary Information; no verdict words and no $P$ values), built after the XD-4 tables were opened; local run 5 October 2026, 03:30 to 05:37 KST; "
    "sealed tables first opened 05:38:08 KST. U is the relative change of RMSE against the mean of S1 draws, averaged over the two weightings (negative, lower error); regret = "
    "U $-$ U(oracle), with the oracle the best of 200 candidate label sets (mean over splits and $n$). Because Algorithm P = S1, the S9* $-$ S1 description equals XD-4. "
    "S9* had lower regret than Algorithm P in 3 of 5 test tasks (CA-2, CA-3, LE-2) and higher regret in 2 (LE-1, Tibetan Plateau); the LE-1 mean is set by four rows of splits 2 "
    "and 4 (U(S9*) +0.38, +0.73, +0.71, +0.64; the other four rows −0.11 to −0.03). Cross-validation utility (post hoc comparison after S9* = S9-DS had been frozen): leaving a "
    "family out, S9-DS was negative in 2 of 5 tasks (LE-2, CA-3) and S9-GBM in 5 of 5 (−0.044 to −0.004). For the Tibetan Plateau both schemes use the same training set; S9-GBM "
    "gave the same value (−0.009) and S9-DS gave +0.053 and +0.072, because the minibatch-order seed includes the fold name. The S9-DS utility is therefore sensitive to the "
    "minibatch-order seed; this is not a determinism failure (the frozen model gave identical selections on two GPUs, Jaccard 1.000). Alaska-family values (leave the Alaska "
    "family out) are in plan 8.11. Common limitation: the candidate pool consists of already measured cells, so gains are not guaranteed to transfer to field candidates. "
    "Source: data/processed/xbatch/XD\\_placement\\_policy/sealed/ (xd5\\_regret.csv, xd5\\_cv\\_ds.csv, xd5\\_cv\\_gbm.csv, xd5\\_meta.json); plan 8.11.")

# ---------------------------------------------------------------------------------------------- XC (8.10)
XC_H = ["Hypothesis and contrast", "$n$", "Pool", "$\\Delta$ cell [95\\% CI]", "$\\Delta$ block-equal [95\\% CI]", "Four-way verdict", "Holm $P$ (rank, multiplier)", "Branch"]
XC = [
    ["XC-1b A1 $-$ A2 (workflow $-$ random placement with fixed recipe)", "40", "partial (2/5: Lena Delta, Canada)", "−0.86 [−1.03, −0.39]", "−0.36 [−0.72, +0.003]", "undetermined", "0.21 (5, 4)", "undetermined (minimum detectable effect about 0.5 cm; two-stage 0.46 cm)"],
    ["XC-1c A1 $-$ A2", "160", "partial (2/5)", "−0.93 [−1.25, −0.40]", "−0.54 [−0.94, −0.13]", "lower error", "0.040 (4, 5)", "lower error"],
    ["XC-2a A1 $-$ A3 (workflow $-$ recalibrated Stefan), non-inferiority 0.5 cm", "10", "5/5", "+0.08 [−0.07, +0.20]", "−0.005 [−0.15, +0.14]", "equivalent", "$<$0.0001 (1, 8)", "non-inferior; XC-2s not superior"],
    ["XC-2b A1 $-$ A3", "40", "partial (2/5)", "−1.35 [−1.64, −0.80]", "−1.00 [−1.42, −0.59]", "lower error", "$<$0.0001 (2, 7)", "non-inferior and superior (XC-2s adjusted $P$ 0.0008)"],
    ["XC-2c A1 $-$ A3", "160", "partial (2/5)", "−1.51 [−1.99, −0.88]", "−1.29 [−1.78, −0.80]", "lower error", "$<$0.0001 (3, 6)", "non-inferior and superior (XC-2s adjusted $P$ 0.0008)"],
    ["XC-5b A1 $-$ A5 (placement under the selection rule)", "40", "", "", "", "not tested", "1.00 (6, 3)", "not decidable (Algorithm P = S1, rule 5)"],
    ["XC-5c A1 $-$ A5", "160", "", "", "", "not tested", "1.00 (7, 2)", "not decidable (Algorithm P = S1, rule 5)"],
    ["XC-F3 A1 $-$ A2, external family F3", "10", "F3", "", "", "not tested", "1.00 (8, 1)", "not decidable ($A1 = A2$ at $n$ 10); appendix pending: registered data deadline 11 October 2026 (XC-F3, after XF)"],
]
XCR_H = ["Contrast", "$n$", "Region", "Cell-weighted [95\\% CI]", "Block-equal [95\\% CI]", "Verdict"]
XCR = [
    ["A1 $-$ A2", "40", "Lena Delta", "+0.21 [+0.02, +0.50]", "+0.11 [−0.10, +0.32]", "undetermined (margin-dependent)"],
    ["", "40", "Canada", "−1.94 [−2.27, −1.07]", "−0.83 [−1.52, −0.15]", "lower error"],
    ["", "160", "Lena Delta", "+0.54 [−0.05, +0.82]", "−0.10 [−0.49, +0.27]", "undetermined (margin-dependent)"],
    ["", "160", "Canada", "−2.41 [−2.77, −1.31]", "−0.98 [−1.69, −0.28]", "lower error"],
    ["A1 $-$ A3", "10", "Lena Delta", "+0.08 [−0.26, +0.14]", "−0.35 [−0.54, −0.18]", "undetermined"],
    ["", "10", "Canada", "−0.84", "−0.85", "lower error"],
    ["", "10", "W Russia", "−0.28", "−0.27", "undetermined"],
    ["", "10", "E Russia", "+1.34 [+0.95, +1.84]", "+1.30 [+0.82, +1.78]", "higher error"],
    ["", "10", "Central Russia", "+0.10", "+0.14", "equivalent (depends on split independence)"],
    ["", "40", "Lena Delta", "+0.25 [−0.13, +0.41]", "−0.24 [−0.49, −0.01]", "equivalent (depends on split independence)"],
    ["", "40", "Canada", "−2.96", "−1.75", "lower error"],
    ["", "160", "Lena Delta", "+0.53 [−0.35, +0.75]", "−0.61 [−1.15, −0.10]", "undetermined"],
    ["", "160", "Canada", "−3.55", "−1.96", "lower error"],
    ["A1 $-$ A3, selection family (S-XC4 b, S-XC8)", "40", "Alaska (parent excluded)", "+1.51 [+0.73, +1.79]", "+0.48 [+0.28, +0.67]", "higher error"],
    ["", "160", "Alaska (parent excluded)", "+2.34 [+0.73, +2.86]", "+0.68 [+0.36, +1.04]", "higher error"],
]
XC_NOTE = (
    "Flags: designed after earlier results were viewed; re-test in reused regions (new random splits of the same cells, seeds 201 to 210); XC-1, XC-2 and XC-5 partly "
    "unblinded (WF2, WF4, WF8, LGX-N4), XC-F3 blind (registered before running). Holm $m$ = 8; the Holm-adjusted $P$ sets the wording; no row needed the 'significant before "
    "correction only' flag (XC-1c 0.040). Run on the local server, 5 October 2026, 03:16 to 05:19 KST (208 units, no failure); sealed tables first opened 05:20:39 KST; "
    "reproduction gate passed (128 keys, local-to-cloud tolerance). Algorithm P = S1 (appendix XC-0, rule 5): the placement step of the workflow A1 was random nested "
    "labels, so A1 = A2 at $n$ 10 (both R1 with $\\lambda$ 0.25) and A1 = A5 at $n$ 40 and 160 (both the selection rule W); XC-1b and XC-1c therefore equal the auxiliary "
    "contrast S-XC1 at $n$ 40 and 160 (rule W against the fixed recipe on the same labels), and the placement step contributed nothing. XC-5b, XC-5c, XC-F3, S-XC0 and S-XC3 "
    "were not tested and stay in the Holm family with $P$ = 1. Because the two arms of the main contrasts share label sets, the main CI is the same-label block-resampling CI; "
    "common-resampling and two-stage upper bounds gave the same four-way and non-inferiority verdicts. HK region-level intervals include zero for all five tested rows "
    "(XC-2a [−0.92, +1.04] cm), so no region-general sentence is written. Leave one region out (auxiliary): XC-1b and XC-1c are undetermined with the Lena Delta only (+0.21, +0.54) "
    "and lower error with Canada only (−1.94, −2.41); without Canada, XC-2a has higher error (+0.31 [+0.12, +0.44] / +0.21 [+0.04, +0.37], both upper bounds below 0.5 cm); "
    "without E Russia, XC-2a has lower error (−0.24 / −0.33). "
    "S-XC4 (a), targets more than 2 cm worse than P0 (A1 / A2 / A3), independent-region family: $n$ 10 1 (E Russia) / 1 (E Russia) / 0; $n$ 40 0 / 0 / 1 (Canada); "
    "$n$ 160 1 (Alaska) / 1 (Canada) / 1 (Canada); all labels 1 (Alaska) / 2 (Canada, E Russia) / 1 (Canada). S-XC4 (b), regions more than 0.5 cm worse than the recalibrated "
    "Stefan model (cell-weighted point estimate): $n$ 10, 1 of 7 (E Russia +1.34); $n$ 40, 1 of 4 (Alaska, selection family, +1.51); $n$ 160, 2 of 3 (Lena Delta +0.53, Alaska +2.34); "
    "Alaska at $n$ 10 was +0.05 cell-weighted but +1.75 [+1.34, +2.16] block-equal; Tibetan Plateau $n$ 10 −21.00 / −17.54, $n$ 40 −29.68 / −23.46. "
    "Auxiliary contrasts (verdict words, not counted): S-XC1 (rule W $-$ fixed R1, same labels) $n$ 10 −0.87 [−1.38, −0.38] / −0.89 [−1.43, −0.35], lower error (depends on split "
    "independence); S-XC2 (stacking candidates added to W) $n$ 40 −0.12 / +0.30 equivalent, $n$ 160 −0.41 / +0.03 undetermined; S-XC5 regret A1 $-$ A6 $n$ 10 +1.73 / +1.41 "
    "(higher error), $n$ 40 +0.24 / +0.21, $n$ 160 +0.42 / +0.20 (undetermined); S-XC6 (post hoc threshold) median RMSE(P0) $-$ RMSE(A1 at the largest $n$) 0.37 / 0.21 cm for "
    "16 targets with $|b_{10}| \\geq 8$ cm and −1.81 / −0.57 cm for 5 targets below; S-XC8, PE1 all labels A1 $-$ A2 −1.29 / −1.33 and A1 $-$ A3 −1.43 / −1.56 (lower error); "
    "S-XC9, equal and $|b_{10}|$-proportional budgets gave the same allocation; S-XC10 (A1 $-$ P0, margin 0.5 cm) $n$ 10 −1.87 / −1.58 (lower error, non-inferior), $n$ 40 −0.22 / −0.14 "
    "and $n$ 160 −0.06 / −0.20 (undetermined, non-inferior, margin-dependent), E Russia $n$ 10 +2.30 [+1.35, +4.54] / +4.00 (higher error); S-XC11 (descriptive) share of reducible "
    "error removed, Canada 0.20 to 0.25, Lena Delta −0.04 to −0.09, Alaska −0.42 to −1.64; S-XC12 $\\rho$ 0.17 [−0.30, 0.86] / −0.06 [−0.49, 0.65] (21 targets, 7 families). "
    "Within-region auxiliary arm XC-r (descriptive verdicts, outside the Holm family): rule W $-$ R1 (cross-validated $\\lambda$) three targets $n$ 200 −0.35 [−0.49, +0.06] / +0.15 "
    "[−0.14, +0.46] equivalent (depends on split independence, margin-dependent), $n$ 500 +0.03 / +0.14 and $n$ 1,000 +0.36 / +0.05 undetermined (Alaska rows equivalent at all three). "
    "Registered sentences (rendered; all with 'in new random splits of the same regions'): common sentence, `Algorithm P was chosen in the Alaska family, and the cells of the "
    "evaluation regions were used to design the components (placement, rule W); the evaluation splits are new partitions of the same cells', with the fact that the placement "
    "step was random placement; XC-1b, `the workflow that applies the placement algorithm, the ten-label bias diagnostic and the label-count method rule in sequence established "
    "no difference from random placement with the fixed residual recipe at 40 labels (minimum detectable effect about 0.5 cm)', Lena Delta row +0.21 cm [0.02, 0.50], undetermined; "
    "XC-1c, `... had 0.93 cm (cell-weighted) lower error than random placement with the fixed residual recipe at 160 labels (Lena Delta and Canada, two regions)', Lena Delta row "
    "+0.54 cm [−0.05, 0.82], undetermined; XC-2a, `the error increase of the workflow relative to the recalibrated Stefan model was within 0.5 cm (non-inferior, 10 labels)', "
    "`in E Russia the error was larger ($\\Delta$ +1.34 cm, CI [0.95, 1.84])', `of seven independent regions (five PE1 regions, Alaska, Tibetan Plateau) the workflow was more than "
    "0.5 cm worse than the recalibrated Stefan model in one (E Russia, +1.34 cm)' and `non-inferiority refers to the pooled mean and did not hold in E Russia', Lena Delta row +0.08, "
    "undetermined; XC-2b and XC-2c, non-inferior at 40 and 160 labels and `1.35 cm' and `1.51 cm lower error than the recalibrated Stefan model', with one of four evaluated regions "
    "(Alaska, selection family, +1.51 cm) and two of three (Lena Delta +0.53 cm, Alaska +2.34 cm) more than 0.5 cm worse, and `non-inferiority refers to the pooled mean and did not "
    "hold' in those regions; Lena Delta rows +0.25 (equivalent) and +0.53 (undetermined); XC-5b, XC-5c, not tested (Algorithm P = S1); XC-F3, not tested ($A1 = A2$ at $n$ 10); "
    "the registered 'no eligible region' sentence is written after the XF data deadline. Source: data/processed/xbatch/XC\\_workflow\\_end\\_to\\_end/sealed/ (xc\\_hyp.csv, "
    "xc\\_contrasts\\_main.csv, xc\\_holm.csv, xc\\_sentences.json, xc\\_harm\\_a.csv, xc\\_harm\\_b.csv, xc\\_loo.csv, xc\\_contrasts\\_aux.csv, xc\\_xcr\\_contrasts.csv); "
    "docs/EXPERIMENT\\_PLAN\\_FINAL\\_BATCH\\_2026-10-04.md 8.10.")

def build(longtable):
    """return the LaTeX of S13 parts c to n; longtable is make_si_tables.longtable."""
    out = []
    out.append(longtable(XB_H, XB, ws=[0.17, 0.05, 0.17, 0.14, 0.14, 0.07, 0.08, 0.18],
                         title="c. XB label-weighted stacking of physics models and products (transfer: XB-1, XB-2; within region: XB-3, XB-4)"))
    out.append(note(XB_NOTE))
    out.append(longtable(XBS_H, XBS, ws=[0.20, 0.07, 0.14, 0.27, 0.20, 0.12],
                         title="c (continued). XB sensitivity edition with CCI v5 added as an eighth candidate (registered verdicts remain those of the main run)"))
    out.append(note(XBS_NOTE))
    out.append(longtable(XG_H, XG, ws=[0.14, 0.04, 0.11, 0.19, 0.19, 0.09, 0.13, 0.11],
                         title="d. XG existing ALT maps scored on the same blocks (registration deviation)"))
    out.append(note(XG_NOTE))
    out.append(longtable(XH_H, XH, ws=[0.09, 0.17, 0.17, 0.12, 0.12, 0.17, 0.16],
                         title="e. XH validation ladder: RMSE by scoring scheme (cm, cell-weighted [95\\% CI] / block-equal; descriptive)"))
    out.append(longtable(XHD_H, XHD, ws=[0.12, 0.13, 0.19, 0.19, 0.19, 0.18],
                         title="f. XH optimism difference $\\Delta\\Delta$ from random-cell to kNNDM splits (cm; descriptive)"))
    out.append(note(XH_NOTE))
    out.append(longtable(XI_H, XI, ws=[0.15, 0.04, 0.17, 0.10, 0.17, 0.12, 0.15, 0.10],
                         title="g. XI warm-block spatial proxy for climate extrapolation, Canada licence-verified expanded edition (not a true extrapolation test)"))
    out.append(note(XI_NOTE))
    out.append(longtable(XE_H, XE, ws=[0.16, 0.05, 0.08, 0.22, 0.17, 0.32],
                         title="h. XE public inputs at 10 to 500 m within regions, stage 1 (xh0; auxiliary contrast XE-c)"))
    out.append(note(XE_NOTE))
    out.append(longtable(XEE_H, XEE, ws=[0.10, 0.22, 0.19, 0.19, 0.15, 0.15],
                         title="h (continued). XE-e diagnostic: share of the within-grid residual of P1 explained by on-site soil moisture (block cross-validation)"))
    out.append(note(XEE_NOTE))
    out.append(longtable(XD_H, XD, ws=[0.15, 0.04, 0.11, 0.13, 0.13, 0.22, 0.12, 0.08],
                         title="i. XD placement algorithm (XD-1 to XD-3; R1 $\\lambda$ 0.25, cm; exploratory)"))
    out.append(longtable(XD4_H, XD4, ws=[0.40, 0.16, 0.16, 0.28],
                         title="j. XD-4 learned placement policy S9* (S9-DS) against Algorithm P (= S1) in test tasks outside Alaska (descriptive, no test)"))
    out.append(longtable(XD6_H, XD6, ws=[0.13, 0.07, 0.09, 0.12, 0.13, 0.23, 0.23],
                         title="k. XD-6 region traits and placement effect (descriptive)"))
    out.append(note(XD_NOTE))
    out.append(longtable(XD5_H, XD5, ws=[0.22, 0.11, 0.14, 0.11, 0.16, 0.13, 0.13],
                         title="k (continued). XD-5 regret against the best of 200 candidate sets (descriptive)"))
    out.append(longtable(XD5CV_H, XD5CV, ws=[0.16, 0.16, 0.17, 0.17, 0.17, 0.17],
                         title="k (continued). XD-5 cross-validation utility of the learned policies (post hoc)"))
    out.append(note(XD5_NOTE))
    out.append(longtable(XJ_H, XJ, ws=[0.30, 0.05, 0.05, 0.17, 0.17, 0.15, 0.11],
                         title="l. XJ temperature-derived auxiliary labels"))
    out.append(note(XJ_NOTE))
    out.append(longtable(XK_H, XK, ws=[0.09, 0.06, 0.08, 0.06, 0.13, 0.13, 0.30, 0.15],
                         title="m. XK error by grid size within regions (added registration; descriptive)"))
    out.append(longtable(XKP_H, XKP, ws=[0.12, 0.05, 0.16, 0.16, 0.16, 0.18, 0.17],
                         title="m (continued). Products against R1 at the same grid size, RMSE(product) $-$ RMSE(R1)"))
    out.append(note(XK_NOTE))
    out.append(longtable(XL_H, XL, ws=[0.08, 0.24, 0.07, 0.12, 0.09, 0.09, 0.08, 0.09, 0.14],
                         title="n. XL 1 km maps compared with ALT products (added registration; descriptive)"))
    out.append(note(XL_NOTE))
    out.append(longtable(XC_H, XC, ws=[0.20, 0.04, 0.12, 0.13, 0.13, 0.10, 0.10, 0.18],
                         title="o. XC workflow applied in sequence, in new random splits of the same regions (PE1 pool; Algorithm P = S1)"))
    out.append(longtable(XCR_H, XCR, ws=[0.24, 0.05, 0.17, 0.17, 0.17, 0.20],
                         title="o (continued). Regional rows"))
    out.append(note(XC_NOTE))
    out.append(longtable(["Analysis", "Status"], [["XF new independent regions", PEND_XF]], ws=[0.5, 0.5],
                         title="p. XF"))
    out.append(longtable(XM_H, XM, ws=[0.11, 0.09, 0.05, 0.15, 0.185, 0.185, 0.23],
                         title="q. XM open 10 to 20 m land-cover and vegetation inputs of the 1 km label cell, within regions (added registration; exploratory; R1 with CV $\\lambda$, xw $-$ x25, cm)"))
    out.append(note(XM_NOTE))
    return "\n".join(out)
