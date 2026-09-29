#!/bin/bash
# LG(H40) Rescale 업로드 묶음 작성. 코드·시험·입력 자료를 저장소 경로 구조 그대로 work/lg_payload.tar.gz 로 묶는다.
# 학습을 하지 않는다(tar 와 해시 계산만, 수 초). 실행: bash scripts/rescale/make_payload.sh
#
# 묶음 내용
#   src/polar/                                   패키지(캐시 제외)
#   scripts/3_deep_learning/h40_label_grid.py    실험 본체
#   scripts/rescale/{run_lg.sh, run_lg_smoke.sh, lg_common.sh, lg_timing_table.py}
#   tests/                                       단위 시험(Rescale 스모크 작업에서 test_h40_smoke.py 를 실행한다)
#   data/processed/{fidelity_base_v3.csv, e5_soil_tdd_v3.csv}   입력 자료
#   data/processed/lg_subregion_map_v1.csv       하위 지역 대응표(없으면 h40 이 k-means 로 계산한다)
#   lg_payload_info.txt                          작성 시각·git 커밋·파일 해시
# 기존 결과(data/processed/lg/)는 넣지 않는다. 크기 상한 20 MB 를 넘으면 실패한다.
set -euo pipefail
cd "$(dirname "$0")/../.."

OUT=work/lg_payload.tar.gz
INFO=work/lg_payload_info.txt
MAX_BYTES=$(( 20 * 1024 * 1024 ))
FILES=(
  src/polar
  scripts/3_deep_learning/h40_label_grid.py
  scripts/rescale/run_lg.sh
  scripts/rescale/run_lg_smoke.sh
  scripts/rescale/lg_common.sh
  scripts/rescale/lg_timing_table.py
  tests
  data/processed/fidelity_base_v3.csv
  data/processed/e5_soil_tdd_v3.csv
  data/processed/lg_subregion_map_v1.csv
)
for f in "${FILES[@]}"; do
  [ -e "$f" ] || { echo "[payload] 없음: $f" >&2; exit 1; }
done
for f in scripts/rescale/run_lg.sh scripts/rescale/run_lg_smoke.sh scripts/rescale/lg_common.sh; do
  bash -n "$f" || { echo "[payload] 문법 오류: $f" >&2; exit 1; }
done

mkdir -p work
{
  echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"
  echo "git_uncommitted_paths=$(git status --porcelain -- src/polar scripts/3_deep_learning/h40_label_grid.py scripts/rescale tests 2>/dev/null | wc -l)"
  for f in scripts/3_deep_learning/h40_label_grid.py src/polar/m1_core.py src/polar/tab_models.py src/polar/h4_common.py src/polar/m1_stats.py \
           tests/test_h40_smoke.py data/processed/fidelity_base_v3.csv data/processed/e5_soil_tdd_v3.csv data/processed/lg_subregion_map_v1.csv; do
    echo "sha256 $(sha256sum "$f" | cut -c1-16) $f"
  done
} > "$INFO"

tar czf "$OUT.part" --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' "${FILES[@]}" -C work lg_payload_info.txt
gzip -t "$OUT.part"
SIZE=$(stat -c %s "$OUT.part")
if [ "$SIZE" -gt "$MAX_BYTES" ]; then
  rm -f "$OUT.part"
  echo "[payload] 실패: 묶음 크기 $(( SIZE / 1024 / 1024 )) MB 가 상한 20 MB 를 넘는다" >&2
  exit 1
fi
mv -f "$OUT.part" "$OUT"

LIST=$(tar tzf "$OUT")                                             # 목록을 한 번만 읽는다(pipefail 에서 grep -q 의 조기 종료 회피)
echo "[payload] $OUT · $(awk -v s="$SIZE" 'BEGIN{printf "%.2f MB", s / 1024 / 1024}') (상한 20 MB) · 파일 $(grep -vc '/$' <<< "$LIST")개"
tar tzvf "$OUT" | awk '{print $3, $6}' | sort -k1,1nr | awk 'NR <= 8 {printf "[payload]   %10d  %s\n", $1, $2}'
for need in scripts/rescale/run_lg.sh scripts/rescale/run_lg_smoke.sh scripts/rescale/lg_common.sh scripts/rescale/lg_timing_table.py \
            scripts/3_deep_learning/h40_label_grid.py src/polar/m1_core.py tests/test_h40_smoke.py \
            data/processed/fidelity_base_v3.csv data/processed/e5_soil_tdd_v3.csv data/processed/lg_subregion_map_v1.csv lg_payload_info.txt; do
  grep -qx "$need" <<< "$LIST" || { echo "[payload] 실패: 묶음에 $need 가 없다" >&2; exit 1; }
done
if grep -q -E '__pycache__|\.pyc$|data/processed/lg/' <<< "$LIST"; then
  echo "[payload] 실패: 묶음에 캐시 또는 기존 결과가 들어 있다" >&2
  exit 1
fi
sed 's/^/[payload] /' "$INFO"
