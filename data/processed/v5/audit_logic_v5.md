# v5 논리 감사 (자동)

| 항목 | 결과 | 세부 |
|---|---|---|
| label-year equals forcing-year (TDD identical after join) | PASS | n=19172 |
| forcing varies across years within cells | PASS | frac cells with sd>0 = 1.000, median sd 114 °C·d |
| climatological forcing constant within cell | PASS |  |
| lag feature equals previous-year TDD | PASS |  |
| monthly cumulative TDD monotone and tdd_cum_12 == e5_tdd | PASS |  |
| observation months within Aug–Oct (CALM = Sep) | PASS | {8: 16214, 9: 2001, 10: 957} |
| product value is the label-year value | PASS |  |
| block folds disjoint in blocks | PASS | folds 5 |
| year hold-out folds contain one year each | PASS | folds 15 |
| prediction rows cover all cell-years once per seed | PASS |  |
| cell-year counts match registry | PASS | {'Alaska': 15540, 'Canada': 1196, 'Lena': 2436} |
| MODIS yearly values only inside 2015–2020, else mean-filled flag | PASS |  |
| winter_partial flag only for 2010 | PASS |  |