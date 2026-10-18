#!/usr/bin/env python3
"""si_fig_captions.py : trimmed captions of Supplementary Figs S7-S14 (round 9, 5 October 2026), used by make_si_figs.py.

The figure agent wrote one legend per figure (outputs/figures/paper/v3_restructure/si/FigS<file>_legend.md, 14:48).
The captions below shorten those legends to the SI caption style (bold title sentence, panel letters, shared conventions)
and keep every number and flag of the legend; check() verifies that every numeric token of the legend occurs in the
caption and that the legend file is the one the caption was trimmed from (sha256 prefix).
Display numbers differ from file numbers after S13 (independent-region prediction maps) was dropped: S13 = file FigS14,
S14 = file FigS15. Supplementary Table numbers are the first-citation numbers (the legend of file FigS15 cited the
abstract contrast bundle as Table S3, its former number; here Table S14). Fig. 1e is the manuscript number.
"""
import hashlib
import os
import re

# display number -> (file number, sha256 prefix of the legend the caption was trimmed from, caption)
CAPTIONS = {
    7: (7, "f25070eb1e97c32a",
        "\\textbf{Four scoring schemes in Alaska (M1 six-fold comparison): split maps, distances and RMSE.} "
        "This is the comparison cited in Supplementary Note 2, not the five-stage XH validation ladder of Fig.~1e. "
        "\\textbf{a}--\\textbf{d}, Fold assignment of the 13,606 labelled 1~km cells (six folds, seed 0) for random cells, 0.05° sites, 0.5° blocks and $k$-fold "
        "nearest-neighbour distance matching (kNNDM; 13 $k$-means clusters merged into six folds). "
        "\\textbf{e}--\\textbf{h}, Distances from test cells to the nearest training cell (blue) and from 20,000 sampled permafrost cells of the 0.02° prediction grid "
        "(mean annual air temperature below 0~°C) to the nearest label (grey), share of cells per logarithmic bin (seed 0); medians (mean of three seeds), 0.004, "
        "1.11, 25.8 and 53.5~km test-to-training and 81.6~km map-to-label. "
        "\\textbf{i}, RMSE (mean of three seeds) of direct ML (CatBoost), the Stefan model fitted to the training folds and the Stefan model plus a ridge residual "
        "(weight 0.75), rising from random cells to kNNDM from 11.6 to 17.9~cm, 14.2 to 14.5~cm and 12.77 to 13.70~cm. "
        "Alaska Albers equal-area projection; coastlines, Natural Earth 50~m."),
    8: (8, "6e3c4692337be205",
        "\\textbf{Transfer without target labels: residual models by target and learner detail.} "
        "Error change relative to the source-coefficient Stefan model without target labels. "
        "\\textbf{a}, \\textbf{b}, Source anchor plus residual ML (CatBoost) with residual weight 0.25 (low-weight residual, \\textbf{a}) and 1.0 (\\textbf{b}) at each "
        "target's labelled-cell centre (17 targets; source excludes the parent region; post hoc re-analysis of the label-grid curves); below −40~cm, end colour "
        "(Tibetan Plateau in \\textbf{b}, −47.34~cm); Alaska targets enlarged, Lena Delta targets offset with leader lines. "
        "\\textbf{c}, Direct ML (ten learners, TabICL v2 with two context settings, CatBoost with physics inputs) and low-weight residual (three learners), "
        "four-region stratified mean and region rows, one axis per column; the main CatBoost uses the abstract-bundle contrast for the four regions and the "
        "label-grid curve for Alaska. "
        "Intervals, 95\\% block-bootstrap confidence intervals (CIs) weighted by cell (thick) and by 0.5° block (thin); band, ±0.5~cm. "
        "Polar stereographic projections (central meridian 127°~E; Alaska enlargement, 157°~W), scale bars true at 70°~N; Tibetan Plateau inset, Lambert azimuthal "
        "equal-area. Permafrost zones, ESA CCI Permafrost fraction v4.0 (1997--2021 mean); coastlines, Natural Earth 50~m."),
    9: (9, "7f82ef55ed47ff02",
        "\\textbf{Coefficient recalibration: coefficient error and ten-label gain by target, soil-property Stefan curves and regional contrasts.} "
        "\\textbf{a}, Source-coefficient error $\\ln(E_{\\mathrm{own}}/E_{\\mathrm{source}})$, $E_{\\mathrm{own}}$ fitted by least squares to all labels of the target "
        "(label-grid target table; positive, target coefficient larger; Tibetan Plateau not in the table, grey). "
        "\\textbf{b}, Error change of the Stefan model recalibrated with ten target labels relative to the source coefficient (post hoc re-analysis of the label-grid "
        "curves); below −15~cm, end colour (Tibetan Plateau, −96.37~cm). "
        "\\textbf{c}, Recalibrated and soil-property recalibrated Stefan models minus the source-coefficient Stefan model by number of labels, each region a new "
        "target (cell-weighted 95\\% intervals). "
        "\\textbf{d}, Four ten-label contrasts by region and four-region stratified mean (abstract contrast bundle): recalibrated minus source Stefan "
        "(−2.45~cm; 95\\% CI −3.04 to −1.88), anchor plus residual ML minus recalibrated Stefan (−0.18; −0.53 to −0.01), pseudo-label augmentation of the anchor "
        "plus residual (−0.13; −0.31 to 0.22) and residual minus physics-input structure (−3.74; −5.23 to −2.75); 95\\% block-bootstrap CIs weighted by cell "
        "(thick) and block (thin); band, ±0.5~cm. Map conventions as in Supplementary Fig.~S8."),
    10: (10, "06156553113c0959",
         "\\textbf{Method selection: choices of the cross-validation rule by target and source cross-validation tuning of neural networks.} "
         "\\textbf{a}, Method chosen most often by five-fold block cross-validation within ten target labels (25 selections per target and scoring mode, "
         "5 splits × 5 draws; scoring modes pooled for sub-regions); the rule was designed after the label-grid results had been viewed. "
         "\\textbf{b}, RMSE of the rule minus the fixed residual recipe (anchor plus residual ML, residual weight 0.25), ten labels, point estimates; beyond ±3~cm, "
         "end colours (Tibetan Plateau, −29.53~cm). "
         "\\textbf{c}, Neural networks configured by leave-one-source-region-out cross-validation minus their default configuration, four-region stratified mean, "
         "for residual ML (weights 0.25 and 1.0) and direct ML, without labels and with ten and all labels; one axis per column. "
         "\\textbf{d}, Change in source cross-validation error against change in target error without labels (32 points: four networks, two model kinds, four "
         "regions; Spearman $\\rho = -0.32$, descriptive); open circles, the 12 points where tuning kept the default. "
         "Intervals, 95\\% block-bootstrap CIs weighted by cell (thick) and block (thin); band, ±0.5~cm. Map conventions as in Supplementary Fig.~S8."),
    11: (11, "301ac0a74df39616",
         "\\textbf{Gains within label-rich regions: error change by scoring block, high-resolution inputs, climate extrapolation and stacking.} "
         "All analyses were designed after earlier results had been viewed. "
         "\\textbf{a}, \\textbf{b}, Residual ML (cross-validated residual weight, mean of two seeds) minus the Stefan model recalibrated on the same labels, by 0.5° "
         "scoring block, all labels within regions (half of the blocks scored per split; Lena Delta 20 blocks over 24 splits, Canada 36 over 25; split means +0.47 "
         "and +0.45~cm); in \\textbf{b} blocks are enlarged squares at their centres, and one Canadian block (+23.77~cm) takes the end colour. "
         "\\textbf{c}, Residual ML with ten added within-grid inputs minus residual ML with the baseline inputs and minus recalibrated Stefan, three-region stratified "
         "mean (500 and 1000 labels: Alaska and Lena Delta only; auxiliary contrast, partly unblinded). "
         "\\textbf{d}, Spatial proxy for climate extrapolation in Canada (warm blocks scored; not a true extrapolation test): penalty of direct ML relative to "
         "recalibrated Stefan, of residual ML relative to direct ML, and residual ML minus recalibrated Stefan in warm blocks; filled, main edition (extrapolation "
         "width 0.97 in $\\sqrt{\\mathrm{TDD}}$ units); open, basic edition (width −0.002). "
         "\\textbf{e}, Label-weighted stacking of physics models and products minus recalibrated Stefan, and stacked residual minus residual, transfer pool and within "
         "regions (partly unblinded). "
         "Intervals, 95\\% block-bootstrap CIs weighted by cell (thick) and block (thin); band, ±0.5~cm. Polar stereographic projections; coastlines, Natural Earth 50~m."),
    12: (12, "f3647bb4ba25d9ab",
         "\\textbf{Placement of new labels: strategy contrasts by target, label proximity and learned placement policies.} "
         "\\textbf{a}, \\textbf{b}, Error change of residual ML (anchor plus residual, weight 0.25) with ten transfer labels placed by block stratification "
         "(\\textbf{a}) or covariate maximin-distance spreading (\\textbf{b}) instead of random cells (13 targets with a placement test; grey, not tested); above "
         "4~cm, end colour (Tibetan Plateau in \\textbf{b}, +12.71~cm). "
         "\\textbf{c}, Proximity contrasts, four-region stratified mean (symbols, regions): direct ML on a random minus a block split; error relative to recalibrated "
         "Stefan with labels near the scored cells minus labels in held-out blocks, for direct and residual ML; residual ML with ten labels inside minus outside the "
         "scored blocks. "
         "\\textbf{d}, Nearest minus farthest third of scored cells, for residual ML minus recalibrated Stefan and recalibrated minus source Stefan (cell-weighted "
         "95\\% intervals). "
         "\\textbf{e}, Learned placement policy and three variants minus block stratification and minus covariate spreading (five-region mean, 10 labels; Lena Delta "
         "and Canada, 40 labels), learned policy minus random cells, and design-weighted coefficient minus learned policy in the Lena Delta (two-stage 95\\% "
         "intervals; XD, exploratory). "
         "Panels \\textbf{a}, \\textbf{b} and \\textbf{e} were designed after earlier results had been viewed. Intervals in \\textbf{c} and \\textbf{e}, 95\\% CIs "
         "weighted by cell (thick) and block (thin); band, ±0.5~cm. Map conventions as in Supplementary Fig.~S8."),
    13: (14, "f9c65deeb1341671",
         "\\textbf{Prediction intervals for the Lena Delta without target labels.} "
         "\\textbf{a}, ALT predicted by the source-coefficient Stefan model (source regions exclude the Lena Delta and a 100~km buffer; ERA5-Land 2015--2020 "
         "climatology). "
         "\\textbf{b}, Width of the 90\\% prediction interval of \\textbf{a}: hierarchical conformal quantile of leave-one-region-out log errors ($q$ = 0.71; calibration "
         "regions Alaska, Canada, E Russia and W Russia), so the width is proportional to the prediction; with four calibration regions the interval carries no "
         "finite-sample guarantee. "
         "\\textbf{c}, Number of covariates outside the 0.5--99.5 percentile range of the source rows, missing values included; all mapped cells have three or more "
         "(most often mean annual air temperature, freezing degree-days and coldest-month temperature); the count is not an error-risk indicator. "
         "38,086 of 53,011 grid cells shown; 87 of 38,173 land cells (0.2\\%) lack inputs. 1~km display grid; north polar stereographic projection centred at "
         "126.7°~E; coastlines, Natural Earth 10~m."),
    14: (15, "3e335494a671a363",
         "\\textbf{Deployment scenarios: scenario matrix, sequential stopping rule and abstract contrast bundle.} "
         "\\textbf{a}, Four-way verdict by stage (labels in the new region) and contrast for five regions and the four-region stratified mean (95\\% block-bootstrap "
         "intervals weighted by cell and by block; equivalence margin ±0.5~cm); frames, non-inferior (margin 0.5~cm); hatching, more labels than the region provides; "
         "dagger, Lena Delta and Canada pool; source and recalibrated, source-coefficient and recalibrated Stefan models. "
         "\\textbf{b}, Sequential stopping rule for recalibration (3, 10, then 40 labels; stop when the jackknife standard error of the log coefficient is at most "
         "$\\tau$): RMSE change relative to always using 40 labels against the mean share of labels used, three-region mean (Lena Delta, Canada, Alaska; cell-weighted "
         "95\\% intervals) and regional values; with $\\tau$ = 0.05 the rule used 0.91 of the labels and changed RMSE by −0.05~cm (95\\% CI −0.08 to −0.002). "
         "\\textbf{c}, Abstract contrast bundle in the order of Supplementary Table S14: four-region stratified mean (95\\% block-bootstrap intervals by cell, thick, "
         "and block, thin), regional values (symbols) and wording after Holm correction over ten contrasts; ensemble, mean of scale-calibrated Stefan, Kudryavtsev and "
         "ESA CCI anchors; last row, interval scores in the Lena Delta, Canada and Alaska (+9.51~cm; 95\\% CI −0.75 to 22.11). "
         "Horizontal axis linear within ±2~cm, logarithmic beyond."),
}

NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def visible(md_text):
    t = re.sub(r"<!--.*?-->", " ", md_text, flags=re.S)
    return " ".join(ln for ln in t.split("\n") if ln.strip() and not ln.lstrip().startswith("#"))


def numbers(text):
    return [m.group(0).rstrip(",") for m in NUM.finditer(text)]


def check(si_dir):
    """return a list of problems: changed legend files, numbers of the legend missing from the caption."""
    problems = []
    for disp, (fnum, sha, cap) in CAPTIONS.items():
        md = os.path.join(si_dir, f"FigS{fnum}_legend.md")
        if not os.path.exists(md):
            problems.append(f"S{disp}: legend file FigS{fnum}_legend.md missing")
            continue
        raw = open(md, encoding="utf-8").read()
        if hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16] != sha:
            problems.append(f"S{disp}: FigS{fnum}_legend.md changed since trimming (re-sync the caption)")
        capnum = set(numbers(cap.replace("--", " ")))
        for x in numbers(visible(raw).replace("–", " ")):
            if x not in capnum:
                problems.append(f"S{disp}: number {x} of the legend not in the caption")
    return problems


def words(cap):
    t = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cap)
    t = re.sub(r"\$[^$]*\$", "X", t).replace("~", " ")
    return len(t.split())
