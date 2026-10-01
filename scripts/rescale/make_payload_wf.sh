#!/bin/bash
# WF(H54) Rescale 업로드 묶음 작성. 코드, 시험, 입력 자료, LGD 약관 확인분 새 지역 실행 표를 저장소 경로 구조 그대로
# work/wf_payload.tar.gz 로 묶는다. 학습을 하지 않는다(tar 와 해시 계산, 수 초). 실행: bash scripts/rescale/make_payload_wf.sh
#
# 묶음 내용
#   src/polar/                                       패키지(캐시 제외)
#   scripts/3_deep_learning/{h40_label_grid.py, h42_label_grid_ext.py, h54_workflow.py}   h40·h42 는 동결(해시 대조), h54 는 WF 하네스
#   scripts/rescale/{lg_common.sh, run_wf.sh}
#   tests/test_h54_workflow.py                       단위 시험(사전 점검 작업에서 실행한다)
#   data/processed/{fidelity_base_v3.csv, e5_soil_tdd_v3.csv, lg_subregion_map_v1.csv, e5_soil_tdd_v4.csv}
#   data/processed/lgd/run_tables/{Tibet, NAtlantic_lic, Russia_C}/{fidelity_base_v3.csv, e5_soil_tdd_v3.csv}
#       LGD 약관 확인분 새 지역 실행 표(약관 미확인 행 0 을 이 스크립트가 다시 확인한다). e5_soil_tdd_v3.csv 는 v4 토양 표로의 상대 기호 연결이다
#   wf_payload_info.txt                              작성 시각, git 커밋, 파일 해시
# 기존 결과(data/processed/wf/)는 넣지 않는다. 크기 상한 40 MB 를 넘으면 실패한다.
set -euo pipefail
cd "$(dirname "$0")/../.."

OUT=work/wf_payload.tar.gz
INFO=work/wf_payload_info.txt
MAX_BYTES=$(( 40 * 1024 * 1024 ))
H40=scripts/3_deep_learning/h40_label_grid.py
H42=scripts/3_deep_learning/h42_label_grid_ext.py
H54=scripts/3_deep_learning/h54_workflow.py
H40_SHA16=7098f59dabe2b73f                                          # 본 실행(ZovWo)의 h40 해시(sha256 앞 16자). h40 은 동결이다
LGD_SPECS=(Tibet NAtlantic_lic Russia_C)
FILES=(
  src/polar
  "$H40" "$H42" "$H54"
  scripts/rescale/lg_common.sh
  scripts/rescale/run_wf.sh
  tests/test_h54_workflow.py
  data/processed/fidelity_base_v3.csv
  data/processed/e5_soil_tdd_v3.csv
  data/processed/e5_soil_tdd_v4.csv
  data/processed/lg_subregion_map_v1.csv
)
for s in "${LGD_SPECS[@]}"; do
  FILES+=("data/processed/lgd/run_tables/$s/fidelity_base_v3.csv" "data/processed/lgd/run_tables/$s/e5_soil_tdd_v3.csv")
done
for f in "${FILES[@]}"; do
  [ -e "$f" ] || { echo "[payload] 없음: $f" >&2; exit 1; }
done
for f in scripts/rescale/run_wf.sh scripts/rescale/lg_common.sh; do
  bash -n "$f" || { echo "[payload] 문법 오류: $f" >&2; exit 1; }
done
python3 -m py_compile "$H54" || { echo "[payload] 문법 오류: $H54" >&2; exit 1; }

sha16() { sha256sum "$1" | cut -c1-16; }
[ "$(sha16 "$H40")" = "$H40_SHA16" ] || { echo "[payload] 실패: $H40 의 해시 $(sha16 "$H40") 가 본 실행의 값 $H40_SHA16 과 다르다(h40 은 동결)" >&2; exit 1; }
for s in "${LGD_SPECS[@]}"; do                                      # 토양 표는 v4 로의 상대 기호 연결이어야 한다(묶음 안에서 풀린다)
  l="data/processed/lgd/run_tables/$s/e5_soil_tdd_v3.csv"
  if [ -L "$l" ]; then
    [ "$(readlink "$l")" = "../../../e5_soil_tdd_v4.csv" ] || { echo "[payload] 실패: $l 의 연결 대상이 예상과 다르다: $(readlink "$l")" >&2; exit 1; }
  fi
done
# 약관 확인: 세 실행 표의 loc_id 가운데 fidelity_base_v4_labels.csv 의 lic_unverified = 1 인 셀이 없어야 한다
python3 - "${LGD_SPECS[@]}" <<'PY'
import sys
import pandas as pd
lab = pd.read_csv("data/processed/fidelity_base_v4_labels.csv", usecols=["loc_id", "lic_unverified"], low_memory=False)
bad = set(lab.loc[lab.lic_unverified == 1, "loc_id"])
for s in sys.argv[1:]:
    t = pd.read_csv(f"data/processed/lgd/run_tables/{s}/fidelity_base_v3.csv", usecols=["loc_id"], low_memory=False)
    n = int(t.loc_id.isin(bad).sum())
    if n:
        sys.exit(f"[payload] 실패: {s} 실행 표에 약관 미확인 셀이 {n}개 있다")
    print(f"[payload] 약관 확인: {s} 실행 표 {len(t):,}행, 약관 미확인 셀 0")
PY

mkdir -p work
{
  echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"
  echo "git_uncommitted_paths=$(git status --porcelain -- src/polar scripts/3_deep_learning/h54_workflow.py scripts/rescale tests/test_h54_workflow.py 2>/dev/null | wc -l)"
  echo "h40_sha16_main_run=$H40_SHA16"
  for f in "$H40" "$H42" "$H54" src/polar/m1_core.py src/polar/h4_common.py src/polar/m1_stats.py src/polar/fidelity.py src/polar/physics.py \
           tests/test_h54_workflow.py scripts/rescale/run_wf.sh scripts/rescale/lg_common.sh \
           data/processed/fidelity_base_v3.csv data/processed/e5_soil_tdd_v3.csv data/processed/e5_soil_tdd_v4.csv data/processed/lg_subregion_map_v1.csv; do
    echo "sha256 $(sha16 "$f") $f"
  done
  for s in "${LGD_SPECS[@]}"; do echo "sha256 $(sha16 "data/processed/lgd/run_tables/$s/fidelity_base_v3.csv") data/processed/lgd/run_tables/$s/fidelity_base_v3.csv"; done
} > "$INFO"

tar czf "$OUT.part" --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' "${FILES[@]}" -C work wf_payload_info.txt
gzip -t "$OUT.part"
SIZE=$(stat -c %s "$OUT.part")
if [ "$SIZE" -gt "$MAX_BYTES" ]; then
  rm -f "$OUT.part"
  echo "[payload] 실패: 묶음 크기 $(( SIZE / 1024 / 1024 )) MB 가 상한 40 MB 를 넘는다" >&2
  exit 1
fi
mv -f "$OUT.part" "$OUT"

LIST=$(tar tzf "$OUT")                                             # 목록을 한 번만 읽는다(pipefail 에서 grep -q 의 조기 종료 회피)
echo "[payload] $OUT · $(awk -v s="$SIZE" 'BEGIN{printf "%.2f MB", s / 1024 / 1024}') (상한 40 MB) · 파일 $(grep -vc '/$' <<< "$LIST")개"
tar tzvf "$OUT" | awk '{print $3, $6}' | sort -k1,1nr | awk 'NR <= 8 {printf "[payload]   %10d  %s\n", $1, $2}'
for need in "$H40" "$H42" "$H54" scripts/rescale/run_wf.sh scripts/rescale/lg_common.sh src/polar/m1_core.py tests/test_h54_workflow.py \
            data/processed/fidelity_base_v3.csv data/processed/e5_soil_tdd_v3.csv data/processed/e5_soil_tdd_v4.csv \
            data/processed/lg_subregion_map_v1.csv wf_payload_info.txt; do
  grep -qx "$need" <<< "$LIST" || { echo "[payload] 실패: 묶음에 $need 가 없다" >&2; exit 1; }
done
if grep -q -E '__pycache__|\.pyc$|data/processed/wf/' <<< "$LIST"; then
  echo "[payload] 실패: 묶음에 캐시 또는 기존 결과가 들어 있다" >&2
  exit 1
fi
sed 's/^/[payload] /' "$INFO"
