# -*- coding: utf-8 -*-
"""xg_alaska_label_cell_metrics.py: 알래스카 라벨 셀에서 우리 블록 교차검증 예측(OOF)과 기존 ALT 제품의 r·R²·RMSE 요약(사후 서술).

보고 덱의 '기존 제품 대비 정확도' 쪽(method_figs_v4.PRODUCT_RESULTS)과 알래스카 실측 대비 예측 쪽의 값이 저장된 요약 표 없이
하드코딩되어 있어(2026-10-08 정리 감사), 같은 값을 셀 단위 파일에서 다시 계산해 요약 표로 남긴다. 새 적합은 하지 않는다.

입력(둘 다 git 밖, 셀 단위 실측값을 담아 배포하지 않는다):
  data/processed/xbatch/XK_support_scale/xk_oof_cells_v1.csv   y(실측), p1(재보정 Stefan OOF), r1(잔차 결합 OOF)
  data/processed/xbatch/XG_product_comparison/xg_product_values_v1.csv   제품 값(value_cm = 연도 정합, value_static_cm = 기간 평균)
출력(집계값만):
  data/processed/xbatch/XG_product_comparison/alaska_label_cell_metrics_v1.csv 와 _meta.json
실행: OMP_NUM_THREADS=1 nice -n 10 python3 scripts/2_evaluation/xg_alaska_label_cell_metrics.py
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OOF = ROOT / "data/processed/xbatch/XK_support_scale/xk_oof_cells_v1.csv"
PROD = ROOT / "data/processed/xbatch/XG_product_comparison/xg_product_values_v1.csv"
OUT = ROOT / "data/processed/xbatch/XG_product_comparison/alaska_label_cell_metrics_v1.csv"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    r = float(np.corrcoef(y, p)[0, 1])
    return dict(n_cells=int(len(y)), r=r, r2_ols=r * r, r2_1mse=float(1 - np.mean(e ** 2) / np.var(y)),
                rmse_cm=float(np.sqrt(np.mean(e ** 2))), bias_cm=float(np.mean(e)), mean_pred_cm=float(np.mean(p)),
                mean_obs_cm=float(np.mean(y)), sd_obs_cm=float(np.std(y)))


def main():
    oof = pd.read_csv(OOF)
    oof = oof[oof["region"] == "Alaska"].copy()
    prod = pd.read_csv(PROD)
    rows = []
    for col, name in (("p1", "Recalibrated Stefan (block-CV OOF)"), ("r1", "Anchor + residual ML (block-CV OOF)")):
        rows.append(dict(subset="all_label_cells", series=col, definition=name, **metrics(oof["y"], oof[col])))
    for key in sorted(prod["product"].unique()):
        sub = prod[(prod["product"] == key) & (prod["valid"] == 1)]
        for vcol, vdef in (("value_cm", "year-matched"), ("value_static_cm", "period mean")):
            m = oof.merge(sub[["loc_id", vcol]], on="loc_id", how="inner").dropna(subset=[vcol])
            if len(m) < 2:
                continue
            rows.append(dict(subset="all_label_cells", series=key, definition=vdef, **metrics(m["y"], m[vcol])))
            if key == "yk":   # Yi–Kimball 이 있는 셀에서 우리 OOF 도 같이 낸다
                rows.append(dict(subset="yk_cells_" + vdef.replace(" ", "_"), series="r1", definition="Anchor + residual ML (block-CV OOF)",
                                 **metrics(m["y"], m["r1"])))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, float_format="%.6g")
    meta = dict(created=datetime.now().astimezone().isoformat(timespec="seconds"), script=str(Path(__file__).relative_to(ROOT)),
                inputs={str(p.relative_to(ROOT)): sha256(p) for p in (OOF, PROD)},
                note="사후 서술(등록 판정 아님). 셀 가중, 알래스카 라벨 셀. r2_ols = r², r2_1mse = 1 − MSE/Var(y). "
                     "덱 PRODUCT_RESULTS 는 연도 정합(value_cm) 정의와 같다. SI 의 XK 표는 기간 평균(value_static_cm) 정의다.",
                n_alaska_cells=int(len(oof)))
    OUT.with_name(OUT.stem + "_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(df[["subset", "series", "definition", "n_cells", "r", "r2_ols", "rmse_cm", "bias_cm", "sd_obs_cm"]].to_string(index=False))


if __name__ == "__main__":
    main()
