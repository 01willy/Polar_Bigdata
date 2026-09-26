"""논문 그림 진입점(Sci Rep 규격, 스펙 figures/PAPER_FIGURE_REDESIGN_2026-09-26.md §6).

그림별 모듈 scripts/4_visualization/paper/<key>.py 의 build(draft: bool) 를 레지스트리 순서대로 호출한다.
모듈이 없으면 경고 후 건너뛰고, 모듈 안 MissingData(자료 없음)는 경고로 처리한다. 그 밖의 예외는 추적을 출력하고 종료 코드 1.
스타일은 src/polar/paperstyle.py, 로더·캡션·QA 는 paper/_common.py 한 곳에서 정의한다.

실행
  python3 scripts/4_visualization/paper_figs.py                  # 전부
  python3 scripts/4_visualization/paper_figs.py --only fig3,fig4 # 일부
  python3 scripts/4_visualization/paper_figs.py --draft          # h4 본 실행 파일이 없으면 스모크 자료로 *_draft 저장
  python3 scripts/4_visualization/paper_figs.py --qa             # _qa/<name>_100pct.png 확인 시트 + _qa/summary.json
산출: outputs/figures/paper/Fig{N}_*.pdf/.png(600 dpi), CAPTIONS.md, source_data/, _qa/
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
import warnings
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "paper"))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import matplotlib                                   # noqa: E402
matplotlib.use("Agg")

REG = {                                             # 키 → 모듈 이름(scripts/4_visualization/paper/<모듈>.py)
    "fig1": "fig1", "fig2": "fig2", "fig3": "fig3", "fig4": "fig4", "fig5": "fig5", "fig6": "fig6", "fig7": "fig7",
    "table1": "table1", "supp": "supp",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="", help="쉼표 구분 키(fig1…fig7, table1, supp)")
    ap.add_argument("--draft", action="store_true", help="h4 본 실행 파일이 없으면 스모크 자료 허용(_draft 저장)")
    ap.add_argument("--qa", action="store_true", help="100 %% 확인 시트와 _qa/summary.json 작성")
    a = ap.parse_args(argv)
    import _common as C                               # noqa: E402
    from polar import paperstyle as ps                # noqa: E402
    ps.use_paper()
    C.QA_OPTS["sheets"] = bool(a.qa)
    keys = [k.strip() for k in a.only.split(",") if k.strip()] or list(REG)
    unknown = [k for k in keys if k not in REG]
    if unknown:
        ap.error(f"알 수 없는 키: {unknown} (가능: {list(REG)})")
    summary, rc = {}, 0
    for k in keys:
        mod_path = HERE / "paper" / f"{REG[k]}.py"
        if not mod_path.exists():
            warnings.warn(f"[paper] {k}: 모듈 {mod_path.relative_to(HERE.parents[1])} 없음, 건너뜀")
            summary[k] = "missing module"
            continue
        try:
            mod = importlib.import_module(REG[k])
            if not hasattr(mod, "build"):
                raise AttributeError(f"{REG[k]}.build(draft) 없음")
            res = mod.build(draft=a.draft)
            res = res if isinstance(res, list) else [res] if res else []
            summary[k] = [dict(name=r.get("name"), ok=r.get("ok"), fails=r.get("fails")) for r in res if isinstance(r, dict)]
            if any(isinstance(r, dict) and not r.get("ok", True) for r in res):
                rc = max(rc, 2)
        except C.MissingData as e:
            warnings.warn(f"[paper] {k}: 자료 없음({e}), 건너뜀. 스모크 자료로 초안을 보려면 --draft")
            summary[k] = f"missing data: {e}"
        except Exception:                             # noqa: BLE001
            traceback.print_exc()
            summary[k] = "error"; rc = 1
    if a.qa:
        (C.QA_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    for k, v in summary.items():
        print(f"  {k:7s} {v}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
