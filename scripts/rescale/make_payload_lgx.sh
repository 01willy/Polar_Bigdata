#!/bin/bash
# LGX(H41·H42) Rescale 업로드 묶음 작성. 코드, 시험, 입력 자료, 고정 입력 표, 재현 점검 기준 조각을 저장소 경로 구조 그대로
# work/lgx_payload.tar.gz 로 묶는다. 학습을 하지 않는다(tar 와 해시 계산, 수 초).
#
# 실행
#   bash scripts/rescale/make_payload_lgx.sh                 고정 입력 표 3종이 있어야 한다(없으면 만드는 명령을 출력하고 1 로 끝난다)
#   bash scripts/rescale/make_payload_lgx.sh --make-tables   없는 표를 먼저 만든 뒤 묶는다(로컬 선행 작업. 학습 없음, 스레드 1, 한 번에 하나, 각 60 s 제한)
#
# 묶음 내용
#   src/polar/                                         패키지(캐시 제외). 기존 모듈은 본 실행(ZovWo)과 같은 판이어야 한다(해시 대조)
#   scripts/3_deep_learning/h40_label_grid.py          본 실행 하네스(동결. h42, h41 이 import 한다)
#   scripts/3_deep_learning/h42_label_grid_ext.py      확장 하네스
#   scripts/2_evaluation/h41_validation_ladder.py      검증 사다리
#   scripts/rescale/{lg_common.sh, lgx_common.sh, run_lgx.sh, run_lgx_smoke.sh, lgx_timing_table.py, lgx_gate_check.py}
#   tests/                                             단위 시험(사전 점검 작업에서 test_h42_ext.py, test_h41_ladder.py 를 실행한다)
#   data/processed/{fidelity_base_v3.csv, e5_soil_tdd_v3.csv, lg_subregion_map_v1.csv}     입력 자료(본 실행과 같은 파일, 해시 대조)
#   data/processed/{lgx_label_flags_v1.csv, lgx_tdd_matched_v1.csv, lgx_cluster_map_v1.csv} 고정 입력 표(계획서 §6A.8)
#   data/processed/lg/shards/lg_{precheck,smoke}__cpu__*   LG 사전 점검(qOkSo)의 cpu 조각. 재현 점검의 기준으로만 쓴다(읽기 전용)
#   data/processed/lg/shards/{lg,lgrm}_precheck__gpu__*_unit.json   LG 사전 점검의 gpu 조각 표지. 분할 4·5 축의 시간 환산에만 쓴다
#   lgx_payload_info.txt                               작성 시각, git 커밋, 파일 해시
# 본 실행 결과(data/processed/lg 의 tag lg 조각)와 확장 결과(data/processed/lgx)는 넣지 않는다. 크기 상한 20 MB 를 넘으면 실패한다.
set -euo pipefail
cd "$(dirname "$0")/../.."

MAKE_TABLES=0
for arg in "$@"; do
  case "$arg" in
    --make-tables) MAKE_TABLES=1 ;;
    *) echo "[payload] 알 수 없는 인자: $arg" >&2; exit 1 ;;
  esac
done

OUT=work/lgx_payload.tar.gz
INFO=work/lgx_payload_info.txt
MAX_BYTES=$(( 20 * 1024 * 1024 ))
H40=scripts/3_deep_learning/h40_label_grid.py
H42=scripts/3_deep_learning/h42_label_grid_ext.py
H41=scripts/2_evaluation/h41_validation_ladder.py
H40_SHA16=7098f59dabe2b73f                                          # 본 실행(ZovWo)의 h40 해시(sha256 앞 16자)
LG_INFO=work/lg_payload_info.txt                                   # 본 실행 묶음의 해시 기록
REF_ROOT=results/rescale_lg_smoke                                  # LG 사전 점검(qOkSo) 결과
REF_DIR=data/processed/lg/shards
TABLES=(data/processed/lgx_label_flags_v1.csv data/processed/lgx_tdd_matched_v1.csv data/processed/lgx_cluster_map_v1.csv)
SCRIPTS=(scripts/rescale/lg_common.sh scripts/rescale/lgx_common.sh scripts/rescale/run_lgx.sh scripts/rescale/run_lgx_smoke.sh)
FILES=(
  src/polar
  "$H40" "$H42" "$H41"
  "${SCRIPTS[@]}"
  scripts/rescale/lgx_timing_table.py
  scripts/rescale/lgx_gate_check.py
  tests
  data/processed/fidelity_base_v3.csv
  data/processed/e5_soil_tdd_v3.csv
  data/processed/lg_subregion_map_v1.csv
  "${TABLES[@]}"
)

# ---------------------------------------------------------------- 고정 입력 표
# 표를 만드는 명령(로컬 선행 작업). 학습을 하지 않는다. 공유 서버이므로 스레드 1, 낮은 우선순위, 한 번에 하나, 60 s 제한으로 돌린다.
table_cmd() {
  case "$1" in
    data/processed/lgx_label_flags_v1.csv) echo "python3 $H42 --write-label-flags --threads 1" ;;
    data/processed/lgx_tdd_matched_v1.csv) echo "python3 $H42 --write-tdd-matched --threads 1" ;;
    data/processed/lgx_cluster_map_v1.csv) echo "python3 $H41 --write-cluster-map --threads 1" ;;
  esac
}
missing=()
for t in "${TABLES[@]}"; do [ -s "$t" ] || missing+=("$t"); done
if [ ${#missing[@]} -gt 0 ] && [ "$MAKE_TABLES" -eq 1 ]; then
  for t in "${missing[@]}"; do
    cmd=$(table_cmd "$t")
    echo "[payload] 표 작성: $cmd"
    t0=$(date +%s)
    # shellcheck disable=SC2086
    env -u LG_RESCALE OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
      nice -n 19 timeout 60 $cmd || { echo "[payload] 실패: 표를 만들지 못했다($t, 종료 코드 $?)" >&2; exit 1; }
    echo "[payload] 표 작성 끝: $t · $(( $(date +%s) - t0 )) s"
  done
  missing=()
  for t in "${TABLES[@]}"; do [ -s "$t" ] || missing+=("$t"); done
fi
if [ ${#missing[@]} -gt 0 ]; then
  echo "[payload] 실패: 고정 입력 표가 없다. 표가 없으면 h42 의 x9·x9f 축과 h41 의 V-C·V-G 단이 실행 전에 중단한다" >&2
  for t in "${missing[@]}"; do echo "[payload]   없음: $t  →  $(table_cmd "$t")" >&2; done
  echo "[payload]   한 번에 만들려면: bash scripts/rescale/make_payload_lgx.sh --make-tables" >&2
  exit 1
fi

# ---------------------------------------------------------------- 파일 확인
for f in "${FILES[@]}"; do
  [ -e "$f" ] || { echo "[payload] 없음: $f" >&2; exit 1; }
done
for f in "${SCRIPTS[@]}"; do
  bash -n "$f" || { echo "[payload] 문법 오류: $f" >&2; exit 1; }
done

# 동결 확인. h40 과 src/polar 의 기존 모듈, 입력 자료가 본 실행 묶음과 같은 판이어야 재현 점검과 X6 의 분할 결합이 성립한다.
sha16() { sha256sum "$1" | cut -c1-16; }
[ "$(sha16 "$H40")" = "$H40_SHA16" ] || { echo "[payload] 실패: $H40 의 해시 $(sha16 "$H40") 가 본 실행의 값 $H40_SHA16 과 다르다" >&2; exit 1; }
if [ -f "$LG_INFO" ]; then
  while read -r _ want path; do
    case "$path" in tests/*) continue ;; esac
    [ -f "$path" ] || { echo "[payload] 실패: 본 실행 묶음에 있던 $path 가 없다" >&2; exit 1; }
    [ "$(sha16 "$path")" = "$want" ] || { echo "[payload] 실패: $path 의 해시가 본 실행 묶음의 값($want)과 다르다" >&2; exit 1; }
  done < <(grep '^sha256 ' "$LG_INFO")
  echo "[payload] 동결 확인: h40, src/polar 의 기존 모듈 4개, 입력 자료 3개의 해시가 본 실행 묶음($LG_INFO)과 같다"
else
  echo "[payload] 경고: $LG_INFO 가 없어 src/polar 와 입력 자료의 해시를 본 실행 묶음과 대조하지 못했다(h40 해시만 확인했다)" >&2
fi

# 재현 점검 기준 조각과 시간 환산용 표지(LG 사전 점검 qOkSo 의 산출). 없으면 경고만 하고 뺀다.
REFS=()
if [ -d "$REF_ROOT/$REF_DIR" ]; then
  while IFS= read -r f; do REFS+=("$REF_DIR/$(basename "$f")"); done < <(
    find "$REF_ROOT/$REF_DIR" -maxdepth 1 -type f \( -name 'lg_precheck__cpu__*' -o -name 'lg_smoke__cpu__*' \
         -o -name 'lg_precheck__gpu__*_unit.json' -o -name 'lgrm_precheck__gpu__*_unit.json' \) | LC_ALL=C sort)
fi
if [ ${#REFS[@]} -eq 0 ]; then
  echo "[payload] 경고: $REF_ROOT/$REF_DIR 에 LG 사전 점검 조각이 없다. 사전 점검 작업의 재현 점검은 '기준 조각 없음'이 된다" >&2
fi

mkdir -p work
{
  echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"
  echo "git_uncommitted_paths=$(git status --porcelain -- src/polar scripts/3_deep_learning scripts/2_evaluation/h41_validation_ladder.py scripts/rescale tests 2>/dev/null | wc -l)"
  echo "h40_sha16_main_run=$H40_SHA16"
  echo "reference_shards=${#REFS[@]} (from $REF_ROOT)"
  for f in "$H40" "$H42" "$H41" src/polar/m1_core.py src/polar/tab_models.py src/polar/h4_common.py src/polar/m1_stats.py src/polar/physics.py \
           src/polar/fidelity.py src/polar/m1_ext.py tests/test_h42_ext.py tests/test_h41_ladder.py "${SCRIPTS[@]}" \
           scripts/rescale/lgx_timing_table.py scripts/rescale/lgx_gate_check.py \
           data/processed/fidelity_base_v3.csv data/processed/e5_soil_tdd_v3.csv data/processed/lg_subregion_map_v1.csv "${TABLES[@]}"; do
    echo "sha256 $(sha16 "$f") $f"
  done
} > "$INFO"

TAR_ARGS=(--exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' "${FILES[@]}")
[ ${#REFS[@]} -gt 0 ] && TAR_ARGS+=(-C "$REF_ROOT" "${REFS[@]}" -C "$PWD")
TAR_ARGS+=(-C work lgx_payload_info.txt)
tar czf "$OUT.part" "${TAR_ARGS[@]}"
gzip -t "$OUT.part"
SIZE=$(stat -c %s "$OUT.part")
if [ "$SIZE" -gt "$MAX_BYTES" ]; then
  rm -f "$OUT.part"
  echo "[payload] 실패: 묶음 크기 $(( SIZE / 1024 / 1024 )) MB 가 상한 20 MB 를 넘는다" >&2
  exit 1
fi
mv -f "$OUT.part" "$OUT"

LIST=$(tar tzf "$OUT")                                             # 목록을 한 번만 읽는다(pipefail 에서 grep -q 의 조기 종료 회피)
echo "[payload] $OUT · $(awk -v s="$SIZE" 'BEGIN{printf "%.2f MB", s / 1024 / 1024}') (상한 20 MB) · 파일 $(grep -vc '/$' <<< "$LIST")개 · 기준 조각 파일 ${#REFS[@]}개"
tar tzvf "$OUT" | awk '{print $3, $6}' | sort -k1,1nr | awk 'NR <= 8 {printf "[payload]   %10d  %s\n", $1, $2}'
for need in "$H40" "$H42" "$H41" "${SCRIPTS[@]}" scripts/rescale/lgx_timing_table.py scripts/rescale/lgx_gate_check.py \
            src/polar/m1_core.py src/polar/physics.py src/polar/fidelity.py src/polar/m1_ext.py tests/test_h42_ext.py tests/test_h41_ladder.py \
            data/processed/fidelity_base_v3.csv data/processed/e5_soil_tdd_v3.csv data/processed/lg_subregion_map_v1.csv "${TABLES[@]}" \
            lgx_payload_info.txt; do
  grep -qx "$need" <<< "$LIST" || { echo "[payload] 실패: 묶음에 $need 가 없다" >&2; exit 1; }
done
if grep -q -E '__pycache__|\.pyc$|data/processed/lgx/|data/processed/lg/shards/lg__' <<< "$LIST"; then
  echo "[payload] 실패: 묶음에 캐시, 확장 결과 또는 본 실행 조각이 들어 있다" >&2
  exit 1
fi
sed 's/^/[payload] /' "$INFO"
