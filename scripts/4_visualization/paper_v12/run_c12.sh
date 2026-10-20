#!/usr/bin/env bash
# Regenerate the v11 figures of role C: Fig. 1, 5, 6, 7 and Supplementary Figs S4-S9 (v11 numbering). CPU only, one map script at a time.
# Copied from scripts/4_visualization/paper_v10/run_fig5_6_si.sh and edited for v11.
# Outputs: outputs/figures/paper/v12/Fig{1,5,6,7}_*.{pdf,svg,png} + source_data/Fig{1,5,6,7}_*.json,
#          outputs/figures/paper/v12_si/FigS{4..9}_*.{pdf,png} + SI_figures.json (keys S4-S9; other keys are written by the other v11 SI scripts)
set -euo pipefail
cd /home/willy010313/Polar_Bigdata
export POLAR_PALETTE=paper12 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
D=scripts/4_visualization/paper_v12
for s in fig1_data fig5_products fig6_product_maps fig7_maps si_s04_external si_s05_training_sites si_s06_product_anchor si_s07_map_detail \
         si_s08_correction_maps si_s09_site_series; do
  echo "== $s"; nice -n 10 python3 "$D/$s.py" 2>&1 | grep -E "QA|outside|overlaps|text_ink|small|color|font|Error|Traceback" || true
done
