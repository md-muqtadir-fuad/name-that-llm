### 4.x score set `tfidf` (clustering thr 0.85), balanced accuracy, mean over seeds


**grouping = true**  (rows: decoder; cells: delta vs base, +-sd over seeds)

| decoder | n2-3 |
|---|---|
| base bacc | 0.3718 |
| share of rows in groups>=2 | 1.00 |
| center0.5+distinct_t1 | +0.0080 +-0.013 |
| center0.5+hungarian | +0.0089 +-0.012 |
| center0.5+sinkhorn0.5 | +0.0130 +-0.015 |
| center0.5+sinkhorn0.5+distinct_t1 | +0.0138 +-0.015 |
| center1+hungarian | -0.0537 +-0.031 |
| center1+sinkhorn0.5 | -0.0456 +-0.033 |
| center_b0.25 | +0.0102 +-0.011 |
| center_b0.5 | +0.0065 +-0.013 |
| center_b0.75 | -0.0099 +-0.018 |
| center_b1.0 | -0.0554 +-0.031 |
| count_balance | -0.0106 +-0.013 |
| distinct_marg_t0.5 | +0.0107 +-0.004 |
| distinct_marg_t1 | +0.0122 +-0.004 |
| hungarian | +0.0084 +-0.006 |
| repulse_a0.2 | +0.0103 +-0.011 |
| repulse_a0.35 | +0.0094 +-0.014 |
| repulse_a0.5 | +0.0006 +-0.012 |
| repulse_a0.7 | -0.0217 +-0.023 |
| sinkhorn0.5+distinct_t1 | +0.0153 +-0.008 |
| sinkhorn0.5+hungarian | +0.0095 +-0.005 |
| sinkhorn_a0.5 | +0.0017 +-0.003 |
| sinkhorn_a1 | -0.0009 +-0.009 |
| twin_add1+distinct_t1 | +0.0228 +-0.018 |
| twin_add1+hungarian | +0.0217 +-0.017 |
| twin_add1+sinkhorn0.5 | +0.0240 +-0.021 |
| twin_add1+sinkhorn0.5+distinct | +0.0245 +-0.022 |
| twin_add_w0.5 | +0.0213 +-0.017 |
| twin_add_w1.0 | +0.0232 +-0.016 |
| twin_add_w2.0 | +0.0141 +-0.023 |
| twin_replace | -0.0347 +-0.023 |

**grouping = rec**  (rows: decoder; cells: delta vs base, +-sd over seeds)

| decoder | n2-3 |
|---|---|
| base bacc | 0.3718 |
| share of rows in groups>=2 | 0.84 |
| center0.5+distinct_t1 | +0.0113 +-0.018 |
| center0.5+hungarian | +0.0100 +-0.014 |
| center0.5+sinkhorn0.5 | +0.0165 +-0.015 |
| center0.5+sinkhorn0.5+distinct_t1 | +0.0159 +-0.014 |
| center1+hungarian | -0.0438 +-0.022 |
| center1+sinkhorn0.5 | -0.0377 +-0.024 |
| center_b0.25 | +0.0087 +-0.007 |
| center_b0.5 | +0.0104 +-0.015 |
| center_b0.75 | -0.0032 +-0.016 |
| center_b1.0 | -0.0446 +-0.022 |
| count_balance | -0.0106 +-0.013 |
| distinct_marg_t0.5 | +0.0073 +-0.006 |
| distinct_marg_t1 | +0.0104 +-0.006 |
| hungarian | +0.0057 +-0.006 |
| repulse_a0.2 | +0.0099 +-0.009 |
| repulse_a0.35 | +0.0108 +-0.016 |
| repulse_a0.5 | +0.0057 +-0.015 |
| repulse_a0.7 | -0.0132 +-0.018 |
| sinkhorn0.5+distinct_t1 | +0.0118 +-0.009 |
| sinkhorn0.5+hungarian | +0.0067 +-0.006 |
| sinkhorn_a0.5 | +0.0017 +-0.003 |
| sinkhorn_a1 | -0.0009 +-0.009 |
| twin_add1+distinct_t1 | +0.0247 +-0.018 |
| twin_add1+hungarian | +0.0221 +-0.015 |
| twin_add1+sinkhorn0.5 | +0.0245 +-0.021 |
| twin_add1+sinkhorn0.5+distinct | +0.0245 +-0.023 |
| twin_add_w0.5 | +0.0216 +-0.013 |
| twin_add_w1.0 | +0.0249 +-0.016 |
| twin_add_w2.0 | +0.0162 +-0.019 |
| twin_replace | -0.0271 +-0.021 |

**grouping = rec_dis**  (rows: decoder; cells: delta vs base, +-sd over seeds)

| decoder | n2-3 |
|---|---|
| base bacc | 0.3718 |
| share of rows in groups>=2 | 0.85 |
| center0.5+distinct_t1 | +0.0120 +-0.018 |
| center0.5+hungarian | +0.0113 +-0.016 |
| center0.5+sinkhorn0.5 | +0.0063 +-0.009 |
| center0.5+sinkhorn0.5+distinct_t1 | +0.0054 +-0.010 |
| center1+hungarian | -0.0453 +-0.025 |
| center1+sinkhorn0.5 | -0.0393 +-0.025 |
| center_b0.25 | +0.0088 +-0.008 |
| center_b0.5 | +0.0108 +-0.017 |
| center_b0.75 | -0.0042 +-0.018 |
| center_b1.0 | -0.0465 +-0.024 |
| count_balance | -0.0596 +-0.009 |
| distinct_marg_t0.5 | +0.0075 +-0.006 |
| distinct_marg_t1 | +0.0104 +-0.006 |
| hungarian | +0.0061 +-0.006 |
| repulse_a0.2 | +0.0102 +-0.009 |
| repulse_a0.35 | +0.0125 +-0.018 |
| repulse_a0.5 | +0.0063 +-0.017 |
| repulse_a0.7 | -0.0140 +-0.020 |
| sinkhorn0.5+distinct_t1 | -0.0117 +-0.011 |
| sinkhorn0.5+hungarian | -0.0145 +-0.010 |
| sinkhorn_a0.5 | -0.0190 +-0.011 |
| sinkhorn_a1 | -0.0543 +-0.012 |
| twin_add1+distinct_t1 | +0.0257 +-0.018 |
| twin_add1+hungarian | +0.0236 +-0.016 |
| twin_add1+sinkhorn0.5 | +0.0082 +-0.016 |
| twin_add1+sinkhorn0.5+distinct | +0.0066 +-0.015 |
| twin_add_w0.5 | +0.0215 +-0.015 |
| twin_add_w1.0 | +0.0256 +-0.016 |
| twin_add_w2.0 | +0.0162 +-0.021 |
| twin_replace | -0.0280 +-0.023 |

**grouping = rec_k18**  (rows: decoder; cells: delta vs base, +-sd over seeds)

| decoder | n2-3 |
|---|---|
| base bacc | 0.3718 |
| share of rows in groups>=2 | 1.00 |
| center0.5+distinct_t1 | -0.0119 +-0.018 |
| center0.5+hungarian | -0.0263 +-0.017 |
| center0.5+sinkhorn0.5 | +0.0057 +-0.012 |
| center0.5+sinkhorn0.5+distinct_t1 | -0.0033 +-0.017 |
| center1+hungarian | -0.0380 +-0.018 |
| center1+sinkhorn0.5 | -0.0165 +-0.018 |
| center_b0.25 | -0.0015 +-0.010 |
| center_b0.5 | -0.0045 +-0.013 |
| center_b0.75 | -0.0102 +-0.018 |
| center_b1.0 | -0.0200 +-0.023 |
| count_balance | -0.0106 +-0.013 |
| distinct_marg_t0.5 | -0.0137 +-0.008 |
| distinct_marg_t1 | +0.0008 +-0.011 |
| hungarian | -0.0239 +-0.017 |
| repulse_a0.2 | +0.0000 +-0.007 |
| repulse_a0.35 | +0.0006 +-0.014 |
| repulse_a0.5 | -0.0050 +-0.012 |
| repulse_a0.7 | -0.0113 +-0.018 |
| sinkhorn0.5+distinct_t1 | +0.0055 +-0.011 |
| sinkhorn0.5+hungarian | -0.0244 +-0.019 |
| sinkhorn_a0.5 | +0.0017 +-0.003 |
| sinkhorn_a1 | -0.0009 +-0.009 |
| twin_add1+distinct_t1 | -0.0159 +-0.012 |
| twin_add1+hungarian | -0.0311 +-0.012 |
| twin_add1+sinkhorn0.5 | -0.0008 +-0.011 |
| twin_add1+sinkhorn0.5+distinct | -0.0181 +-0.014 |
| twin_add_w0.5 | +0.0091 +-0.009 |
| twin_add_w1.0 | +0.0026 +-0.013 |
| twin_add_w2.0 | -0.0072 +-0.016 |
| twin_replace | -0.0326 +-0.018 |