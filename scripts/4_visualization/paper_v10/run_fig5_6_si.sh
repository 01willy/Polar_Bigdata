#!/usr/bin/env bash
# Regenerate Fig. 5, Fig. 6 and Supplementary Figs S1-S11 (v10 manuscript). CPU only, one map script at a time.
# Outputs: outputs/figures/paper/v10/Fig{5,6}_*.{pdf,svg,png} + source_data/Fig{5,6}_*.json, outputs/figures/paper/v10_si/FigS*.{pdf,png} + SI_figures.json
set -euo pipefail
cd /home/willy010313/Polar_Bigdata
export POLAR_PALETTE=paper CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
D=scripts/4_visualization/paper_v10
for s in fig5_products fig6_maps si_s01_stefan si_s02_structures si_s03_shap si_s04_product_anchor si_s05_external si_s06_s07_product_maps \
         si_s08_site_series si_s09_map_detail si_s10_training_sites si_s11_label_ladder; do
  echo "== $s"; nice -n 10 python3 "$D/$s.py" 2>&1 | grep -E "QA|outside|overlaps|text_ink|small|color|Error|Traceback" || true
done
