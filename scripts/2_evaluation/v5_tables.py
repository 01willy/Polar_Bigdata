"""v5 표: Table 2(방법 × 검증 단 RMSE, 지역별, 연도 정합 강제력)와 보조 표(강제력 효과, 제품 비교)를 CSV·Markdown·LaTeX 로 쓴다.
실행: python3 scripts/2_evaluation/v5_tables.py --run data/processed/v5/runs/core_cb --out outputs/figures/paper/v5
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
METH = ["P0", "P1", "P1a", "Ku", "D0", "F1", "D1", "R1", "S"]
SCHEMES = ["random", "block", "year", "block_year", "region"]
SCH_EN = {"random": "Random", "block": "Block", "year": "Year", "block_year": "Block × year", "region": "Region"}


def fmt(v, nd=1):
    return "" if (v is None or (isinstance(v, float) and not np.isfinite(v))) else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--out", default="outputs/figures/paper/v5")
    ap.add_argument("--lam", type=float, default=0.5)
    a = ap.parse_args(); run = ROOT / a.run; out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    sm = pd.read_csv(run / "summary_metrics.csv")
    sm = sm[(sm.featset == "F8TS") & ((sm.method != "R1") | (sm.lam == a.lam))]
    dl = pd.read_csv(run / "deltas_vs_ref.csv")
    dl = dl[(dl.featset == "F8TS") & (dl.ref == "P1") & ((dl.method != "R1") | (dl.lam == a.lam))]
    rows = []
    for forcing in ("year", "clim"):
        for r in ("Alaska", "Lena", "Canada"):
            for m in METH:
                rec = dict(forcing=forcing, region=r, method=m)
                for s in SCHEMES:
                    q = sm[(sm.forcing == forcing) & (sm.region == r) & (sm.scheme == s) & (sm.method == m)]
                    rec[f"rmse_{s}"] = float(q.rmse.iloc[0]) if len(q) else np.nan
                    d = dl[(dl.forcing == forcing) & (dl.region == r) & (dl.scheme == s) & (dl.method == m)]
                    rec[f"{s}_dP1"] = f"{d.delta.iloc[0]:+.1f} [{d.lo.iloc[0]:+.1f}, {d.hi.iloc[0]:+.1f}]" if len(d) else ("ref" if m == "P1" else "")
                q = sm[(sm.forcing == forcing) & (sm.region == r) & (sm.scheme == "year") & (sm.method == m)]
                rec["r_anom_year"] = float(q.r_anom.iloc[0]) if len(q) else np.nan
                rows.append(rec)
    t = pd.DataFrame(rows); t.to_csv(out / "Table2_methods_v5.csv", index=False)
    # Markdown(연도 정합 판)
    md = ["# Table 2. RMSE (cm) by method and validation design, year-matched forcing, F8TS, CatBoost (R1 λ %g). ΔRMSE vs P1 [95 %% block-bootstrap CI]" % a.lam, ""]
    for r in ("Alaska", "Lena", "Canada"):
        md += [f"## {r}", "", "| Method | " + " | ".join(f"{SCH_EN[s]} RMSE | Δ vs P1" for s in SCHEMES) + " | r_anom (year) |", "|---|" + "---|" * (2 * len(SCHEMES) + 1)]
        for m in METH:
            q = t[(t.forcing == "year") & (t.region == r) & (t.method == m)].iloc[0]
            md.append(f"| {m} | " + " | ".join(f"{fmt(q[f'rmse_{s}'])} | {q[f'{s}_dP1']}" for s in SCHEMES) + f" | {fmt(q['r_anom_year'], 2)} |")
        md.append("")
    (out / "Table2_methods_v5.md").write_text("\n".join(md))
    # LaTeX(요약: 블록·연도·지역 홀드아웃만)
    tex = ["\\begin{tabular}{llrrrrrr}", "\\toprule", "Region & Method & Random & Block & Year & Block$\\times$year & Region & $r_{anom}$ \\\\", "\\midrule"]
    for r in ("Alaska", "Lena", "Canada"):
        for i, m in enumerate(METH):
            q = t[(t.forcing == "year") & (t.region == r) & (t.method == m)].iloc[0]
            tex.append(f"{r if i == 0 else ''} & {m} & " + " & ".join(fmt(q[f'rmse_{s}']) for s in SCHEMES) + f" & {fmt(q['r_anom_year'], 2)} \\\\")
        tex.append("\\midrule")
    tex[-1] = "\\bottomrule"; tex.append("\\end{tabular}")
    (out / "Table2_methods_v5.tex").write_text("\n".join(tex))
    print((out / "Table2_methods_v5.md").read_text()[:3000])


if __name__ == "__main__":
    main()
