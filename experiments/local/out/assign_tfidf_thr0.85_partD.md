### 4.x score set `tfidf` (clustering thr 0.85), balanced accuracy, mean over seeds


**grouping = true**  (rows: decoder; cells: delta vs base, +-sd over seeds)

| decoder | n1-2 | dup_pairs | dup_n2-4 |
|---|---|---|---|
| base bacc | 0.3742 | 0.3773 | 0.3695 |
| share of rows in groups>=2 | 0.68 | 1.00 | 1.00 |
| center0.5+distinct_t1 | +0.0129 +-0.018 | +0.0211 +-0.007 | +0.0025 +-0.019 |
| center0.5+hungarian | +0.0147 +-0.016 | -0.0178 +-0.013 | -0.0026 +-0.020 |
| center0.5+sinkhorn0.5 | +0.0139 +-0.014 | +0.0160 +-0.007 | +0.0023 +-0.011 |
| center0.5+sinkhorn0.5+distinct_t1 | +0.0149 +-0.016 | +0.0229 +-0.007 | +0.0041 +-0.016 |
| center1+hungarian | -0.0633 +-0.025 | -0.0215 +-0.013 | -0.0646 +-0.029 |
| center1+sinkhorn0.5 | -0.0582 +-0.025 | +0.0125 +-0.006 | -0.0566 +-0.030 |
| center_b0.25 | +0.0078 +-0.012 | +0.0075 +-0.003 | +0.0066 +-0.013 |
| center_b0.5 | +0.0139 +-0.017 | +0.0069 +-0.007 | -0.0027 +-0.016 |
| center_b0.75 | -0.0090 +-0.021 | +0.0018 +-0.007 | -0.0235 +-0.027 |
| center_b1.0 | -0.0633 +-0.025 | -0.0068 +-0.006 | -0.0651 +-0.030 |
| count_balance | -0.0096 +-0.023 | +0.0011 +-0.006 | -0.0166 +-0.020 |
| distinct_marg_t0.5 | +0.0057 +-0.006 | +0.0141 +-0.008 | +0.0025 +-0.007 |
| distinct_marg_t1 | +0.0064 +-0.005 | +0.0227 +-0.006 | +0.0019 +-0.004 |
| hungarian | +0.0055 +-0.005 | -0.0126 +-0.016 | +0.0040 +-0.011 |
| repulse_a0.2 | +0.0059 +-0.012 | +0.0061 +-0.005 | +0.0040 +-0.012 |
| repulse_a0.35 | +0.0123 +-0.017 | +0.0078 +-0.008 | -0.0009 +-0.012 |
| repulse_a0.5 | +0.0016 +-0.019 | +0.0067 +-0.007 | -0.0060 +-0.017 |
| repulse_a0.7 | -0.0211 +-0.023 | +0.0022 +-0.010 | -0.0320 +-0.027 |
| sinkhorn0.5+distinct_t1 | +0.0085 +-0.011 | +0.0262 +-0.006 | +0.0037 +-0.010 |
| sinkhorn0.5+hungarian | +0.0094 +-0.010 | -0.0121 +-0.015 | +0.0016 +-0.011 |
| sinkhorn_a0.5 | +0.0041 +-0.012 | +0.0038 +-0.004 | +0.0005 +-0.009 |
| sinkhorn_a1 | +0.0034 +-0.023 | +0.0054 +-0.005 | -0.0008 +-0.012 |
| twin_add1+distinct_t1 | +0.0108 +-0.017 | -0.0005 +-0.008 | -0.0121 +-0.020 |
| twin_add1+hungarian | +0.0094 +-0.016 | -0.0142 +-0.011 | -0.0126 +-0.020 |
| twin_add1+sinkhorn0.5 | +0.0156 +-0.020 | +0.0141 +-0.005 | -0.0082 +-0.015 |
| twin_add1+sinkhorn0.5+distinct | +0.0156 +-0.021 | +0.0030 +-0.005 | -0.0118 +-0.018 |
| twin_add_w0.5 | +0.0123 +-0.013 | +0.0157 +-0.007 | +0.0065 +-0.011 |
| twin_add_w1.0 | +0.0091 +-0.017 | +0.0136 +-0.006 | -0.0094 +-0.016 |
| twin_add_w2.0 | -0.0007 +-0.025 | +0.0072 +-0.008 | -0.0174 +-0.021 |
| twin_replace | -0.0490 +-0.019 | -0.0049 +-0.010 | -0.0574 +-0.034 |

**grouping = rec**  (rows: decoder; cells: delta vs base, +-sd over seeds)

| decoder | n1-2 |
|---|---|
| base bacc | 0.3742 |
| share of rows in groups>=2 | 0.55 |
| center0.5+distinct_t1 | +0.0078 +-0.014 |
| center0.5+hungarian | +0.0096 +-0.012 |
| center0.5+sinkhorn0.5 | +0.0114 +-0.014 |
| center0.5+sinkhorn0.5+distinct_t1 | +0.0093 +-0.015 |
| center1+hungarian | -0.0554 +-0.014 |
| center1+sinkhorn0.5 | -0.0508 +-0.023 |
| center_b0.25 | +0.0047 +-0.010 |
| center_b0.5 | +0.0091 +-0.013 |
| center_b0.75 | -0.0101 +-0.015 |
| center_b1.0 | -0.0554 +-0.014 |
| count_balance | -0.0096 +-0.023 |
| distinct_marg_t0.5 | +0.0038 +-0.006 |
| distinct_marg_t1 | +0.0045 +-0.005 |
| hungarian | +0.0039 +-0.005 |
| repulse_a0.2 | +0.0035 +-0.009 |
| repulse_a0.35 | +0.0075 +-0.013 |
| repulse_a0.5 | -0.0010 +-0.014 |
| repulse_a0.7 | -0.0204 +-0.015 |
| sinkhorn0.5+distinct_t1 | +0.0087 +-0.011 |
| sinkhorn0.5+hungarian | +0.0087 +-0.010 |
| sinkhorn_a0.5 | +0.0041 +-0.012 |
| sinkhorn_a1 | +0.0034 +-0.023 |
| twin_add1+distinct_t1 | +0.0069 +-0.013 |
| twin_add1+hungarian | +0.0060 +-0.013 |
| twin_add1+sinkhorn0.5 | +0.0118 +-0.016 |
| twin_add1+sinkhorn0.5+distinct | +0.0115 +-0.017 |
| twin_add_w0.5 | +0.0068 +-0.005 |
| twin_add_w1.0 | +0.0060 +-0.013 |
| twin_add_w2.0 | -0.0040 +-0.019 |
| twin_replace | -0.0432 +-0.016 |

**grouping = rec_dis**  (rows: decoder; cells: delta vs base, +-sd over seeds)

| decoder | n1-2 |
|---|---|
| base bacc | 0.3742 |
| share of rows in groups>=2 | 0.57 |
| center0.5+distinct_t1 | +0.0067 +-0.014 |
| center0.5+hungarian | +0.0092 +-0.013 |
| center0.5+sinkhorn0.5 | -0.0098 +-0.022 |
| center0.5+sinkhorn0.5+distinct_t1 | -0.0079 +-0.023 |
| center1+hungarian | -0.0593 +-0.014 |
| center1+sinkhorn0.5 | -0.0536 +-0.020 |
| center_b0.25 | +0.0044 +-0.009 |
| center_b0.5 | +0.0088 +-0.013 |
| center_b0.75 | -0.0140 +-0.017 |
| center_b1.0 | -0.0590 +-0.014 |
| count_balance | -0.0577 +-0.013 |
| distinct_marg_t0.5 | +0.0033 +-0.007 |
| distinct_marg_t1 | +0.0040 +-0.006 |
| hungarian | +0.0034 +-0.005 |
| repulse_a0.2 | +0.0024 +-0.009 |
| repulse_a0.35 | +0.0070 +-0.013 |
| repulse_a0.5 | -0.0036 +-0.014 |
| repulse_a0.7 | -0.0242 +-0.016 |
| sinkhorn0.5+distinct_t1 | -0.0151 +-0.019 |
| sinkhorn0.5+hungarian | -0.0176 +-0.016 |
| sinkhorn_a0.5 | -0.0205 +-0.018 |
| sinkhorn_a1 | -0.0552 +-0.012 |
| twin_add1+distinct_t1 | +0.0032 +-0.016 |
| twin_add1+hungarian | +0.0030 +-0.016 |
| twin_add1+sinkhorn0.5 | -0.0149 +-0.013 |
| twin_add1+sinkhorn0.5+distinct | -0.0154 +-0.013 |
| twin_add_w0.5 | +0.0060 +-0.005 |
| twin_add_w1.0 | +0.0030 +-0.015 |
| twin_add_w2.0 | -0.0084 +-0.025 |
| twin_replace | -0.0486 +-0.014 |

**grouping = rec_k18**  (rows: decoder; cells: delta vs base, +-sd over seeds)

| decoder | n1-2 |
|---|---|
| base bacc | 0.3742 |
| share of rows in groups>=2 | 1.00 |
| center0.5+distinct_t1 | -0.0161 +-0.015 |
| center0.5+hungarian | -0.0376 +-0.027 |
| center0.5+sinkhorn0.5 | +0.0032 +-0.009 |
| center0.5+sinkhorn0.5+distinct_t1 | -0.0092 +-0.018 |
| center1+hungarian | -0.0479 +-0.029 |
| center1+sinkhorn0.5 | -0.0194 +-0.013 |
| center_b0.25 | -0.0005 +-0.009 |
| center_b0.5 | -0.0134 +-0.011 |
| center_b0.75 | -0.0156 +-0.008 |
| center_b1.0 | -0.0289 +-0.017 |
| count_balance | -0.0096 +-0.023 |
| distinct_marg_t0.5 | -0.0130 +-0.014 |
| distinct_marg_t1 | -0.0058 +-0.015 |
| hungarian | -0.0337 +-0.020 |
| repulse_a0.2 | +0.0036 +-0.009 |
| repulse_a0.35 | -0.0044 +-0.011 |
| repulse_a0.5 | -0.0153 +-0.012 |
| repulse_a0.7 | -0.0178 +-0.011 |
| sinkhorn0.5+distinct_t1 | -0.0031 +-0.019 |
| sinkhorn0.5+hungarian | -0.0318 +-0.022 |
| sinkhorn_a0.5 | +0.0041 +-0.012 |
| sinkhorn_a1 | +0.0034 +-0.023 |
| twin_add1+distinct_t1 | -0.0267 +-0.018 |
| twin_add1+hungarian | -0.0418 +-0.029 |
| twin_add1+sinkhorn0.5 | -0.0109 +-0.008 |
| twin_add1+sinkhorn0.5+distinct | -0.0271 +-0.020 |
| twin_add_w0.5 | -0.0084 +-0.013 |
| twin_add_w1.0 | -0.0149 +-0.010 |
| twin_add_w2.0 | -0.0223 +-0.015 |
| twin_replace | -0.0437 +-0.016 |