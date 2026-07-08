"""표준 평가지표 — 모든 실험이 동일한 metric을 출력하도록 강제하는 단일 소스.

원칙(2026-07-06 방향 확정, docs/PLAN_FORWARD.md):
- RMSE 단독 보고 금지. 반드시 R²·target_SD·skill-over-mean 병기.
- skill_over_mean = 1 - RMSE/std(y): 평균예측기 대비 실제 설명력.
  범위를 좁히면(큐레이션) RMSE는 작아져도 skill은 오히려 낮아질 수 있음 → 아티팩트 탐지용.
- 헤드라인은 공간블록/LORO/kNNDM CV. 무작위 CV는 참고용.

사용:
    from polar.eval_metrics import all_metrics, grouped_metrics, STD_COLS
    m = all_metrics(y, yhat, lower=lo, upper=hi)   # dict (rmse_cm, mae_cm, bias_cm, r2, target_sd_cm, skill_over_mean, coverage_90, width_90, n)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

STD_COLS = [
    "n", "rmse_cm", "mae_cm", "bias_cm", "r2", "target_sd_cm", "skill_over_mean",
    "coverage_90", "width_90",
]


def _clean(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    return y[m], yhat[m]


def rmse(y, yhat):
    y, yhat = _clean(y, yhat)
    if y.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y, yhat):
    y, yhat = _clean(y, yhat)
    return float(np.mean(np.abs(y - yhat))) if y.size else np.nan


def bias(y, yhat):
    """예측 - 관측의 평균(양수=과대예측)."""
    y, yhat = _clean(y, yhat)
    return float(np.mean(yhat - y)) if y.size else np.nan


def r2(y, yhat):
    """1 - SSE/SST. 음수면 평균예측기보다 나쁨."""
    y, yhat = _clean(y, yhat)
    if y.size == 0:
        return np.nan
    sse = np.sum((y - yhat) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - sse / sst) if sst > 0 else np.nan


def target_sd(y):
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    return float(np.std(y)) if y.size else np.nan


def skill_over_mean(y, yhat):
    """1 - RMSE/std(y). 평균예측기 대비 RMSE 개선율. R²와 monotone(=1-(1-R²)^0.5는 아님; 정의상 1-RMSE/SD)."""
    y, yhat = _clean(y, yhat)
    if y.size == 0:
        return np.nan
    sd = np.std(y)
    return float(1.0 - rmse(y, yhat) / sd) if sd > 0 else np.nan


def coverage(y, lower, upper):
    """관측이 [lower, upper] 안에 든 비율(0~1). nominal과 비교(예 0.90)."""
    y = np.asarray(y, float); lower = np.asarray(lower, float); upper = np.asarray(upper, float)
    m = np.isfinite(y) & np.isfinite(lower) & np.isfinite(upper)
    if m.sum() == 0:
        return np.nan
    return float(np.mean((y[m] >= lower[m]) & (y[m] <= upper[m])))


def interval_width(lower, upper):
    lower = np.asarray(lower, float); upper = np.asarray(upper, float)
    m = np.isfinite(lower) & np.isfinite(upper)
    return float(np.mean(upper[m] - lower[m])) if m.sum() else np.nan


def all_metrics(y, yhat, lower=None, upper=None, nominal=0.90):
    """표준 지표 dict. 구간 주면 coverage/width도 채움(컬럼명은 nominal 90 고정 관례)."""
    y, yhat = _clean(y, yhat)
    out = {
        "n": int(y.size),
        "rmse_cm": round(rmse(y, yhat), 3),
        "mae_cm": round(mae(y, yhat), 3),
        "bias_cm": round(bias(y, yhat), 3),
        "r2": round(r2(y, yhat), 4),
        "target_sd_cm": round(target_sd(y), 3),
        "skill_over_mean": round(skill_over_mean(y, yhat), 4),
        "coverage_90": np.nan,
        "width_90": np.nan,
    }
    if lower is not None and upper is not None:
        out["coverage_90"] = round(coverage(y, lower, upper), 4)
        out["width_90"] = round(interval_width(lower, upper), 3)
    return out


def grouped_metrics(df, group_cols, y_col="alt_cm", pred_col="pred",
                    lower_col=None, upper_col=None, nominal=0.90):
    """df를 group_cols로 나눠 표준 지표 계산 → DataFrame(그룹키 + STD_COLS)."""
    rows = []
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        lo = g[lower_col] if lower_col else None
        hi = g[upper_col] if upper_col else None
        m = all_metrics(g[y_col].values, g[pred_col].values,
                        lower=(lo.values if lo is not None else None),
                        upper=(hi.values if hi is not None else None),
                        nominal=nominal)
        row = dict(zip(group_cols if isinstance(group_cols, (list, tuple)) else [group_cols], keys))
        row.update(m)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # self-test: 완벽예측·평균예측·잡음
    rng = np.random.default_rng(0)
    y = rng.normal(50, 15, 5000)
    print("perfect :", all_metrics(y, y))
    print("mean    :", all_metrics(y, np.full_like(y, y.mean())))  # skill≈0, r2≈0
    print("noisy   :", all_metrics(y, y + rng.normal(0, 10, y.size),
                                   lower=y - 20, upper=y + 20))
