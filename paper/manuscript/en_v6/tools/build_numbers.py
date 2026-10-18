#!/usr/bin/env python3
"""build_numbers.py : regional summaries for the [MISSING] items of QA rule M-04 (MANUSCRIPT_SPEC 12), read-only.

Definitions (round 6, 2026-10-05; recorded in COMPLIANCE.md 12.6):
  * regional median  = median of the cell-weighted regional deltas of the contrast (one value per region of the pool);
  * mean without W Russia = equal-weight mean of the remaining regional deltas, the same stratification as the pooled
    mean (the pooled MEAN row equals the simple mean of the regional rows; checked below);
  * regions with lower / higher error = regional four-way verdicts of the registered rule (both 95% block-bootstrap CIs,
    cell-weighted and block-equal, below / above zero). The stored verdict is used when present; otherwise it is derived
    from the stored CI columns with the same rule.
Sources (opened tables, no refit):
  * R3: paper/claims/C1_label0_safety/tables/lgw_bundle.csv (copy of data/processed/lgw/lgw_bundle.csv), contrasts AB4
    (P1 - P0, 10 labels) and AB7 (R1 - F1k, 10 labels), four main regions;
  * R7: data/processed/lgd/lgd_tests_lic.csv (licence-verified edition, main verdict), pool PE1 (four main regions plus
    Central Russia), L8e item R1-P0 with all labels and L4e item R1-P1 with 10 labels.
Usage: python3 tools/build_numbers.py   (run from paper/manuscript/en); writes build/numbers_m04.json
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
OUT = os.path.join(HERE, 'build', 'numbers_m04.json')
VERD = {'우세': 'lower', '열세': 'higher', '동등': 'equivalent', '미결정': 'undetermined'}


def verdict(row):
    v = row.get('verdict4')
    if isinstance(v, str) and v in VERD:
        return VERD[v], 'stored'
    cols = ['ci_lo', 'ci_hi', 'ci_lo_beq', 'ci_hi_beq']
    if all(c in row and pd.notna(row[c]) for c in cols):
        if row['ci_hi'] < 0 and row['ci_hi_beq'] < 0:
            return 'lower', 'derived'
        if row['ci_lo'] > 0 and row['ci_lo_beq'] > 0:
            return 'higher', 'derived'
        if max(abs(row[c]) for c in cols) < 0.5:
            return 'equivalent', 'derived'
        return 'undetermined', 'derived'
    return 'n.a.', 'none'


def summarize(reg, mean_row, label):
    reg = reg.copy()
    reg['region'] = reg['target'].str.replace('|x', '', regex=False)
    vals = reg.set_index('region')['delta']
    res = {
        'contrast': label,
        'regions': {k: round(float(v), 4) for k, v in vals.items()},
        'pooled_mean_row': round(float(mean_row['delta']), 4),
        'pooled_mean_check': round(float(vals.mean()), 4),
        'median': round(float(np.median(vals.values)), 4),
        'mean_without_W_Russia': round(float(vals.drop('Russia_W').mean()), 4),
    }
    vv = [verdict(r) for _, r in reg.iterrows()]
    res['verdicts'] = {reg.iloc[i]['region']: vv[i][0] + ('' if vv[i][1] == 'stored' else ' (derived)') for i in range(len(reg))}
    res['n_lower'] = sum(v[0] == 'lower' for v in vv)
    res['n_higher'] = sum(v[0] == 'higher' for v in vv)
    res['n_regions'] = len(vv)
    assert abs(res['pooled_mean_row'] - res['pooled_mean_check']) < 1e-3, (label, res['pooled_mean_row'], res['pooled_mean_check'])
    return res


out = {}
b = pd.read_csv(os.path.join(ROOT, 'paper/claims/C1_label0_safety/tables/lgw_bundle.csv'))
for ab, label in [('AB4', 'R3 recalibration P1 - P0, 10 labels, four main regions'),
                  ('AB7', 'R3 residual structure R1 - F1k (physics-input ML), 10 labels, four main regions')]:
    q = b[b.ab == ab]
    out[ab] = summarize(q[q.scope == 'region'], q[q.scope == 'MEAN'].iloc[0], label)

d = pd.read_csv(os.path.join(ROOT, 'data/processed/lgd/lgd_tests_lic.csv'))
for key, tid, item, n, label in [('PE1_R1-P0_all', 'L8e', 'R1-P0', -1.0, 'R7 five-region pool, residual ML - source-coefficient Stefan, all labels'),
                                 ('PE1_R1-P1_10', 'L4e', 'R1-P1', 10.0, 'R7 five-region pool, residual ML - recalibrated Stefan, 10 labels')]:
    q = d[(d.test_id == tid) & (d.pool == 'PE1') & (d.item == item) & (d.n == n)]
    out[key] = summarize(q[q.scope == 'region'], q[q.scope == 'MEAN'].iloc[0], label)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for k, v in out.items():
    print(k, '| median', v['median'], '| without W Russia', v['mean_without_W_Russia'], '| lower', v['n_lower'], 'higher', v['n_higher'],
          'of', v['n_regions'], '|', v['verdicts'])
