#!/bin/bash
# xbatch(XA–XJ) Rescale 업로드 묶음 작성. 계획 docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md 4절 묶음 목록.
# 목록은 xbatch_core.PAYLOAD_CODE_GLOBS·PAYLOAD_INPUTS·PAYLOAD_OPTIONAL(있을 때만, PAYLOAD_REQUIRES 동반 확인)을 그대로 쓴다.
# 학습을 하지 않는다(목록 확인, 해시, tar). 실행: bash scripts/rescale/make_payload_xbatch.sh
# 산출: work/xbatch_payload.tar.gz, work/xbatch_payload_info.txt(작성 시각, git 커밋, 미커밋 경로 수, 파일 sha256 앞 16자)
set -euo pipefail
cd "$(dirname "$0")/../.."

OUT=work/xbatch_payload.tar.gz
INFO=work/xbatch_payload_info.txt
LIST=work/xbatch_payload_files.txt
MAX_BYTES=$(( 200 * 1024 * 1024 ))
mkdir -p work

CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python3 - "$LIST" <<'PY'
import glob, os, sys
sys.path.insert(0, "scripts/3_deep_learning")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import xbatch_core as XB
files = []
for g in XB.PAYLOAD_CODE_GLOBS:
    hit = sorted(p for p in glob.glob(g) if "__pycache__" not in p)
    if not hit:
        if g in ("scripts/3_deep_learning/xbatch_core.py", "scripts/3_deep_learning/x_*.py", "src/polar/*.py"):
            sys.exit(f"[payload] 실패: 코드 패턴에 맞는 파일이 없다: {g}")
        print(f"[payload] 패턴에 맞는 파일 없음(건너뜀): {g}")
    files += hit
for p in XB.PAYLOAD_INPUTS:
    if not os.path.exists(p):
        sys.exit(f"[payload] 실패: 필수 입력이 없다: {p}")
    files.append(p)
for p in XB.PAYLOAD_OPTIONAL:
    if os.path.exists(p):
        for q in XB.PAYLOAD_REQUIRES.get(p, ()):
            if not os.path.exists(q):
                sys.exit(f"[payload] 실패: {p} 의 동반 파일이 없다: {q}")
        files.append(p)
        print(f"[payload] 선택 입력 포함: {p}")
    else:
        print(f"[payload] 선택 입력 없음(건너뜀): {p}")
files += ["scripts/rescale/lg_common.sh", "scripts/rescale/run_xbatch.sh"]
seen, out = set(), []
for p in files:
    if p not in seen:
        seen.add(p); out.append(p)
open(sys.argv[1], "w").write("\n".join(out) + "\n")
print(f"[payload] 파일 {len(out)}개")
PY

bash -n scripts/rescale/run_xbatch.sh
for f in scripts/3_deep_learning/xbatch_core.py scripts/3_deep_learning/x_*.py scripts/2_evaluation/xa_*.py; do
  python3 -m py_compile "$f"
done
# 동결 모듈 해시(계획 머리말): h40, h42, h54, h41
sha16() { sha256sum "$1" | cut -c1-16; }
[ "$(sha16 scripts/3_deep_learning/h40_label_grid.py)" = 7098f59dabe2b73f ] || { echo "[payload] 실패: h40 해시 불일치" >&2; exit 1; }
[ "$(sha16 scripts/3_deep_learning/h42_label_grid_ext.py)" = 22e215e435e157be ] || { echo "[payload] 실패: h42 해시 불일치" >&2; exit 1; }
[ "$(sha16 scripts/3_deep_learning/h54_workflow.py)" = cf9a1f6d0929c892 ] || { echo "[payload] 실패: h54 해시 불일치" >&2; exit 1; }
[ "$(sha16 scripts/2_evaluation/h41_validation_ladder.py)" = 492373e4d37b5ea4 ] || { echo "[payload] 실패: h41 해시 불일치" >&2; exit 1; }

# 약관 확인: LGD 실행 표 네 개에 lic_unverified = 1 셀이 없어야 한다(계획 4절)
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python3 - <<'PY'
import pandas as pd
lab = pd.read_csv("data/processed/fidelity_base_v4_labels.csv", usecols=["loc_id", "lic_unverified"], low_memory=False)
bad = set(lab.loc[lab.lic_unverified == 1, "loc_id"])
for s in ("Tibet", "NAtlantic_lic", "Russia_C", "Canada_expanded_lic"):
    t = pd.read_csv(f"data/processed/lgd/run_tables/{s}/fidelity_base_v3.csv", usecols=["loc_id"], low_memory=False)
    n = int(t.loc_id.isin(bad).sum())
    if n:
        raise SystemExit(f"[payload] 실패: {s} 실행 표에 약관 미확인 셀이 {n}개 있다")
    print(f"[payload] 약관 확인: {s} {len(t):,}행, 약관 미확인 셀 0")
PY

{
  echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit=$(git rev-parse --short HEAD)"
  echo "git_uncommitted_paths=$(git status --porcelain -- $(cat "$LIST" | grep -v '^data/') 2>/dev/null | wc -l)"
  while read -r f; do
    [ -f "$f" ] && echo "sha256 $(sha16 "$f") $f"
  done < "$LIST"
} > "$INFO"

# LGD 실행 표의 토양 표는 v4 로의 상대 기호 연결이다(tar 가 연결을 보존하고 대상 v4 표도 목록에 있다)
tar czf "$OUT.part" --exclude='__pycache__' --exclude='*.pyc' -T "$LIST" -C work xbatch_payload_info.txt
gzip -t "$OUT.part"
SIZE=$(stat -c %s "$OUT.part")
if [ "$SIZE" -gt "$MAX_BYTES" ]; then rm -f "$OUT.part"; echo "[payload] 실패: 크기 $(( SIZE / 1048576 )) MB > 200 MB" >&2; exit 1; fi
mv -f "$OUT.part" "$OUT"
echo "[payload] $OUT · $(( SIZE / 1048576 )) MB · 파일 $(wc -l < "$LIST")개 · sha256 $(sha256sum "$OUT" | cut -c1-16)"
grep -E "^(created|git_)" "$INFO" | sed 's/^/[payload] /'
