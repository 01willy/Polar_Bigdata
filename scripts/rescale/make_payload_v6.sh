#!/bin/bash
# v6 Rescale 묶음: 코드(testbed·하네스·집계·시계열·공통 셸) + v5 입력 표(제품 긴 표·격자 제외). 산출 work/v6_payload.tar.gz
set -euo pipefail; cd "$(dirname "$0")/../.."
mkdir -p work
FILES="src/polar/__init__.py src/polar/testbed_v5.py src/polar/physics.py src/polar/fidelity.py src/polar/tab_models.py
scripts/3_deep_learning/v5_compare.py scripts/2_evaluation/v5_aggregate.py scripts/2_evaluation/v5_site_series.py
scripts/rescale/lg_common.sh scripts/rescale/run_v6.sh
data/processed/v5/labels_cell_year_v5.csv data/processed/v5/static_v5.csv data/processed/v5/forcing_year.csv data/processed/v5/forcing_clim.csv
data/processed/v5/modis_cell_year_v5.csv data/processed/v5/hires_cell_v5.csv design/style_tokens_v4.json"
for f in $FILES; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
python3 -m py_compile src/polar/testbed_v5.py scripts/3_deep_learning/v5_compare.py scripts/2_evaluation/v5_aggregate.py scripts/2_evaluation/v5_site_series.py
bash -n scripts/rescale/run_v6.sh
tar czf work/v6_payload.tar.gz $FILES
{ echo "created $(date -u +%FT%TZ)"; echo "git_commit $(git rev-parse --short HEAD)"; for f in $FILES; do echo "$(sha256sum "$f" | cut -c1-16) $f"; done; } > work/v6_payload_info.txt
ls -la work/v6_payload.tar.gz | awk '{print $5" bytes"}'
