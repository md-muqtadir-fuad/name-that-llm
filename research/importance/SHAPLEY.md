# E049 Source Shapley (strong OOF blend, bacc 0.5396, mean of 3 scenarios)

| source | Shapley bacc | leave-one-out | folds with Shapley > 0 |
|---|---|---|---|
| svm | +0.0555 | +0.0053 | 5/5 |
| style | +0.0328 | +0.0046 | 5/5 |
| deberta | +0.0656 | +0.0105 | 5/5 |
| roberta | +0.0435 | +0.0006 | 5/5 |
| qwen8 | +0.0923 | +0.0063 | 5/5 |
| qwen14 | +0.0916 | +0.0038 | 5/5 |
| q35 | +0.1028 | +0.0121 | 5/5 |

## Shapley per family (fam = right family; within = right model inside the true family)

| source | fam:claude | fam:gemini | fam:glm | fam:gpt | fam:kimi | fam:qwen | within:claude | within:gemini | within:glm | within:gpt |
|---|---|---|---|---|---|---|---|---|---|---|
| svm | +0.091 | +0.118 | +0.029 | +0.089 | +0.006 | +0.082 | +0.048 | +0.050 | +0.005 | +0.034 |
| style | +0.073 | +0.098 | +0.032 | +0.072 | -0.005 | +0.051 | +0.028 | +0.024 | +0.011 | +0.021 |
| deberta | +0.089 | +0.121 | +0.058 | +0.109 | +0.060 | +0.065 | +0.060 | +0.051 | +0.003 | +0.044 |
| roberta | +0.066 | +0.106 | +0.028 | +0.088 | +0.033 | +0.012 | +0.045 | +0.034 | +0.006 | +0.037 |
| qwen8 | +0.100 | +0.117 | +0.064 | +0.111 | +0.076 | +0.089 | +0.093 | +0.065 | +0.018 | +0.087 |
| qwen14 | +0.113 | +0.126 | +0.080 | +0.129 | +0.094 | +0.074 | +0.084 | +0.057 | +0.012 | +0.074 |
| q35 | +0.104 | +0.125 | +0.049 | +0.113 | +0.102 | +0.095 | +0.117 | +0.057 | +0.020 | +0.111 |

## Top source per label (Shapley on recall)

- claude-fable-5: q35 +0.051, qwen8 +0.036, qwen14 +0.036
- claude-opus-4-6: q35 +0.140, qwen14 +0.113, svm +0.110
- claude-opus-4-8: q35 +0.124, qwen8 +0.106, qwen14 +0.099
- claude-opus-5: q35 +0.157, qwen8 +0.156, qwen14 +0.139
- claude-sonnet-5: q35 +0.128, qwen8 +0.101, qwen14 +0.094
- gemini-3.1-pro-preview: q35 +0.130, qwen14 +0.122, qwen8 +0.122
- gemini-3.6-flash: svm +0.090, qwen8 +0.083, q35 +0.081
- gemini-3.7-flash: deberta +0.073, qwen14 +0.071, qwen8 +0.068
- gemini-3.8-flash: q35 +0.110, qwen8 +0.084, qwen14 +0.066
- glm-5.3: qwen8 +0.072, q35 +0.057, qwen14 +0.054
- glm-5.3-flash: qwen14 +0.075, deberta +0.043, qwen8 +0.032
- gpt-5.5: qwen8 +0.090, qwen14 +0.083, q35 +0.080
- gpt-5.6-luna: q35 +0.146, qwen14 +0.110, qwen8 +0.097
- gpt-5.6-sol: q35 +0.097, qwen8 +0.090, qwen14 +0.055
- gpt-5.6-terra: qwen14 +0.113, q35 +0.096, qwen8 +0.076
- gpt-6-astra: q35 +0.173, qwen8 +0.148, qwen14 +0.145
- kimi-k3: q35 +0.118, qwen14 +0.110, qwen8 +0.092
- qwen3.8-max: q35 +0.111, qwen8 +0.105, svm +0.098
