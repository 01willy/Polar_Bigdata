#!/bin/bash
# v9 Rescale 묶음: 코드(testbed·추가 실험·집계·공통 셸) + v5 입력 표 + v9 외부 지역 표. 산출 work/v9_payload.tar.gz
set -euo pipefail; cd "$(dirname "$0")/../.."
mkdir -p work
FILES="src/polar/__init__.py src/polar/testbed_v5.py src/polar/physics.py src/polar/fidelity.py src/polar/tab_models.py
scripts/rescale/lg_common.sh scripts/rescale/run_v9.sh scripts/3_deep_learning/v9_extra.py scripts/2_evaluation/v9_extra_agg.py
data/processed/v5/labels_cell_year_v5.csv data/processed/v5/static_v5.csv data/processed/v5/forcing_year.csv data/processed/v5/forcing_clim.csv
data/processed/v5/modis_cell_year_v5.csv data/processed/v5/hires_cell_v5.csv data/processed/v5/products_cell_year_v5.csv
data/processed/v9_ext/labels_cell_year_v5.csv data/processed/v9_ext/static_v5.csv data/processed/v9_ext/forcing_year.csv data/processed/v9_ext/forcing_clim.csv
data/processed/v9_ext/products_cell_year_v5.csv design/style_tokens_v4.json"
for f in $FILES; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
python3 -m py_compile src/polar/testbed_v5.py scripts/3_deep_learning/v9_extra.py scripts/2_evaluation/v9_extra_agg.py
bash -n scripts/rescale/run_v9.sh
tar czf work/v9_payload.tar.gz $FILES
{ echo "created $(date -u +%FT%TZ)"; echo "git_commit $(git rev-parse --short HEAD)"; for f in $FILES; do echo "$(sha256sum "$f" | cut -c1-16) $f"; done; } > work/v9_payload_info.txt
ls -la work/v9_payload.tar.gz | awk '{print $5" bytes"}'
