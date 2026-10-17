## 1. Stylometric EDA

### 1.1 Global discriminativeness (Kruskal-Wallis H across 18 models; eta^2 from one-way ANOVA)

| feature | H | eta^2 |
|---|---|---|
| uses_em_dash | 1244 | 0.461 |
| em_dash_pw | 1205 | 0.366 |
| straight_apos_pw | 858 | 0.230 |
| uses_straight_apos | 857 | 0.318 |
| uses_curly_apos | 646 | 0.239 |
| curly_apos_pw | 645 | 0.171 |
| non_ascii_pc | 574 | 0.146 |
| colon_pw | 460 | 0.148 |
| mean_word_len | 445 | 0.172 |
| comma_pw | 360 | 0.134 |
| semicolon_pw | 316 | 0.109 |
| rule_of_three | 309 | 0.128 |
| func_word_pc | 243 | 0.094 |
| words_per_sent | 228 | 0.031 |
| contraction_pw | 202 | 0.067 |
| curly_dq_pw | 185 | 0.056 |
| sent_len_std | 133 | 0.064 |
| straight_dq_pw | 126 | 0.033 |
| upper_pc | 123 | 0.026 |
| type_token | 109 | 0.042 |
| n_sents | 83 | 0.036 |
| not_x_but_y | 76 | 0.027 |
| digit_pc | 65 | 0.016 |
| markdown_italic | 46 | 0.017 |
| you_pw | 45 | 0.013 |
| paren_pw | 35 | 0.012 |
| we_pw | 32 | 0.011 |
| ends_period | 26 | 0.010 |
| first_the | 25 | 0.009 |
| n_chars | 25 | 0.011 |

### 1.2 Per-model means (per-word rates x100 for *_pw)

| model | n_words | em_dash_pw | en_dash_pw | semicolon_pw | colon_pw | curly_apos_pw | straight_apos_pw | curly_dq_pw | straight_dq_pw | comma_pw | words_per_sent | mean_word_len | type_token | paren_pw | question_pw | exclam_pw | you_pw | not_x_but_y | markdown_any | has_newline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-fable-5 | 113.95 | 1.37 | 0.00 | 0.35 | 0.68 | 0.00 | 1.02 | 0.00 | 0.15 | 9.20 | 24.44 | 5.33 | 0.81 | 0.04 | 0.04 | 0.00 | 0.60 | 0.05 | 0.00 | 0.00 |
| claude-opus-4-6 | 115.27 | 1.14 | 0.00 | 0.12 | 0.37 | 0.00 | 0.99 | 0.00 | 0.17 | 7.80 | 27.30 | 5.64 | 0.82 | 0.00 | 0.03 | 0.00 | 0.71 | 0.08 | 0.00 | 0.00 |
| claude-opus-4-8 | 124.53 | 0.86 | 0.00 | 0.19 | 0.50 | 0.00 | 0.91 | 0.00 | 0.13 | 9.54 | 24.09 | 5.44 | 0.80 | 0.01 | 0.04 | 0.00 | 0.93 | 0.06 | 0.00 | 0.00 |
| claude-opus-5 | 123.85 | 1.23 | 0.01 | 0.59 | 0.80 | 0.00 | 1.01 | 0.00 | 0.20 | 8.41 | 23.96 | 5.06 | 0.78 | 0.01 | 0.05 | 0.02 | 0.72 | 0.07 | 0.00 | 0.00 |
| claude-sonnet-5 | 121.27 | 1.23 | 0.00 | 0.23 | 0.59 | 0.00 | 1.50 | 0.00 | 0.24 | 9.69 | 26.51 | 5.65 | 0.82 | 0.00 | 0.01 | 0.01 | 0.53 | 0.07 | 0.00 | 0.00 |
| gemini-3.1-pro-preview | 119.31 | 0.00 | 0.00 | 0.16 | 0.16 | 0.05 | 0.13 | 0.00 | 0.03 | 7.57 | 20.25 | 5.50 | 0.80 | 0.00 | 0.05 | 0.01 | 0.60 | 0.04 | 0.00 | 0.00 |
| gemini-3.6-flash | 111.11 | 0.00 | 0.00 | 0.20 | 0.17 | 0.21 | 0.26 | 0.00 | 0.06 | 8.37 | 24.65 | 5.72 | 0.83 | 0.01 | 0.04 | 0.00 | 0.56 | 0.04 | 0.00 | 0.00 |
| gemini-3.7-flash | 117.11 | 0.01 | 0.00 | 0.20 | 0.17 | 0.36 | 0.18 | 0.00 | 0.00 | 8.27 | 23.87 | 5.94 | 0.84 | 0.01 | 0.03 | 0.00 | 0.53 | 0.05 | 0.00 | 0.00 |
| gemini-3.8-flash | 100.68 | 0.00 | 0.00 | 0.28 | 0.18 | 0.32 | 0.15 | 0.05 | 0.01 | 8.53 | 22.73 | 5.90 | 0.86 | 0.00 | 0.03 | 0.00 | 0.59 | 0.04 | 0.00 | 0.00 |
| glm-5.3 | 111.71 | 1.52 | 0.00 | 0.37 | 0.75 | 0.00 | 1.07 | 0.00 | 0.23 | 9.06 | 23.63 | 5.32 | 0.81 | 0.01 | 0.04 | 0.01 | 0.70 | 0.06 | 0.00 | 0.00 |
| glm-5.3-flash | 120.48 | 1.38 | 0.01 | 0.53 | 0.74 | 0.00 | 1.11 | 0.00 | 0.21 | 8.77 | 23.50 | 5.36 | 0.81 | 0.00 | 0.03 | 0.00 | 0.70 | 0.06 | 0.00 | 0.00 |
| gpt-5.5 | 110.47 | 0.18 | 0.00 | 0.76 | 0.82 | 0.69 | 0.15 | 0.27 | 0.05 | 11.06 | 26.33 | 5.19 | 0.83 | 0.00 | 0.02 | 0.01 | 0.33 | 0.11 | 0.00 | 0.00 |
| gpt-5.6-luna | 115.82 | 0.49 | 0.02 | 0.45 | 0.88 | 0.72 | 0.02 | 0.23 | 0.00 | 10.32 | 23.79 | 5.51 | 0.82 | 0.14 | 0.05 | 0.00 | 0.31 | 0.08 | 0.00 | 0.00 |
| gpt-5.6-sol | 119.21 | 0.34 | 0.01 | 0.64 | 0.79 | 0.82 | 0.00 | 0.36 | 0.00 | 10.17 | 24.97 | 5.43 | 0.83 | 0.01 | 0.03 | 0.01 | 0.24 | 0.08 | 0.00 | 0.00 |
| gpt-5.6-terra | 113.65 | 0.37 | 0.02 | 0.52 | 0.90 | 0.71 | 0.05 | 0.16 | 0.00 | 10.63 | 26.36 | 5.41 | 0.83 | 0.06 | 0.01 | 0.00 | 0.18 | 0.09 | 0.00 | 0.00 |
| gpt-6-astra | 117.76 | 0.14 | 0.00 | 0.59 | 0.76 | 0.68 | 0.14 | 0.15 | 0.00 | 7.90 | 21.13 | 5.26 | 0.82 | 0.01 | 0.06 | 0.00 | 0.69 | 0.08 | 0.00 | 0.00 |
| kimi-k3 | 108.15 | 0.89 | 0.03 | 0.82 | 0.99 | 0.46 | 0.43 | 0.21 | 0.02 | 10.03 | 25.91 | 5.29 | 0.84 | 0.00 | 0.02 | 0.00 | 0.47 | 0.09 | 0.00 | 0.00 |
| qwen3.8-max | 108.40 | 0.05 | 0.00 | 0.38 | 0.49 | 0.52 | 0.17 | 0.02 | 0.00 | 11.78 | 22.23 | 5.46 | 0.84 | 0.00 | 0.00 | 0.00 | 0.48 | 0.10 | 0.00 | 0.00 |

### 1.3 Share of passages containing a mark at least once (%)

| model | em dash — | en dash – | semicolon | colon | curly apostrophe ’ | straight apostrophe ' | curly quotes “ | straight quotes " | parenthesis | question ? | exclamation ! | spaced hyphen ' - ' | newline | ** | ... |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-fable-5 | 83 | 0 | 34 | 59 | 0 | 60 | 0 | 9 | 3 | 5 | 0 | 0 | 0 | 0 | 0 |
| claude-opus-4-6 | 75 | 0 | 15 | 42 | 0 | 57 | 0 | 7 | 0 | 3 | 1 | 0 | 0 | 0 | 0 |
| claude-opus-4-8 | 63 | 0 | 24 | 49 | 0 | 60 | 0 | 7 | 1 | 5 | 0 | 0 | 0 | 0 | 0 |
| claude-opus-5 | 81 | 1 | 54 | 69 | 0 | 63 | 0 | 9 | 2 | 3 | 1 | 0 | 0 | 0 | 1 |
| claude-sonnet-5 | 73 | 0 | 23 | 51 | 0 | 73 | 0 | 10 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| gemini-3.1-pro-preview | 0 | 0 | 19 | 17 | 6 | 11 | 0 | 3 | 0 | 5 | 1 | 0 | 0 | 0 | 0 |
| gemini-3.6-flash | 0 | 0 | 23 | 20 | 19 | 21 | 0 | 3 | 1 | 4 | 0 | 0 | 0 | 0 | 0 |
| gemini-3.7-flash | 1 | 0 | 23 | 23 | 31 | 14 | 0 | 0 | 1 | 3 | 0 | 0 | 0 | 0 | 0 |
| gemini-3.8-flash | 0 | 0 | 28 | 18 | 25 | 11 | 2 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| glm-5.3 | 82 | 0 | 34 | 62 | 0 | 58 | 0 | 9 | 1 | 4 | 1 | 0 | 0 | 0 | 0 |
| glm-5.3-flash | 79 | 1 | 48 | 65 | 0 | 67 | 0 | 12 | 0 | 4 | 0 | 0 | 0 | 0 | 0 |
| gpt-5.5 | 13 | 0 | 61 | 65 | 42 | 11 | 9 | 1 | 0 | 2 | 1 | 0 | 0 | 0 | 0 |
| gpt-5.6-luna | 37 | 1 | 43 | 71 | 51 | 1 | 11 | 0 | 5 | 4 | 0 | 0 | 0 | 0 | 0 |
| gpt-5.6-sol | 27 | 1 | 62 | 64 | 53 | 0 | 15 | 0 | 1 | 3 | 1 | 0 | 0 | 0 | 0 |
| gpt-5.6-terra | 23 | 1 | 52 | 73 | 47 | 5 | 8 | 0 | 2 | 1 | 0 | 0 | 0 | 0 | 0 |
| gpt-6-astra | 9 | 0 | 53 | 65 | 44 | 9 | 7 | 0 | 1 | 5 | 0 | 0 | 0 | 0 | 0 |
| kimi-k3 | 52 | 1 | 61 | 77 | 32 | 23 | 9 | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 0 |
| qwen3.8-max | 3 | 0 | 39 | 47 | 29 | 12 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |

### 1.4 Same, per lab (%)

| lab | em dash — | en dash – | semicolon | colon | curly apostrophe ’ | straight apostrophe ' | curly quotes “ | straight quotes " | parenthesis | question ? | exclamation ! | spaced hyphen ' - ' | newline | ** | ... |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| anthropic | 75 | 0 | 30 | 54 | 0 | 63 | 0 | 9 | 1 | 3 | 0 | 0 | 0 | 0 | 0 |
| google | 0 | 0 | 23 | 19 | 20 | 14 | 0 | 2 | 0 | 4 | 0 | 0 | 0 | 0 | 0 |
| moonshot | 52 | 1 | 61 | 77 | 32 | 23 | 9 | 1 | 1 | 2 | 0 | 0 | 0 | 0 | 0 |
| openai | 22 | 1 | 54 | 68 | 47 | 5 | 10 | 0 | 2 | 3 | 0 | 0 | 0 | 0 | 0 |
| qwen | 3 | 0 | 39 | 47 | 29 | 12 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| zhipu | 81 | 0 | 41 | 64 | 0 | 63 | 0 | 11 | 1 | 4 | 1 | 0 | 0 | 0 | 0 |

### 1.5 Most distinctive stylometric features per model (standardized mean diff vs other 17 models)

| model | top + features (d) | top - features (d) |
|---|---|---|
| claude-fable-5 | uses_em_dash +0.96, em_dash_pw +0.86, uses_straight_apos +0.66, straight_apos_pw +0.54, non_ascii_pc +0.34 | uses_curly_apos -0.55, curly_apos_pw -0.44, mean_word_len -0.26, not_x_but_y -0.18 |
| claude-opus-4-6 | uses_em_dash +0.79, uses_straight_apos +0.60, em_dash_pw +0.60, straight_apos_pw +0.50, markdown_italic +0.40 | uses_curly_apos -0.55, comma_pw -0.49, semicolon_pw -0.49, curly_apos_pw -0.44 |
| claude-opus-4-8 | uses_straight_apos +0.67, uses_em_dash +0.51, straight_apos_pw +0.41, markdown_italic +0.30, em_dash_pw +0.28 | uses_curly_apos -0.55, curly_apos_pw -0.44, semicolon_pw -0.37, type_token -0.25 |
| claude-opus-5 | uses_em_dash +0.92, sent_len_std +0.78, uses_straight_apos +0.74, em_dash_pw +0.70, func_word_pc +0.66 | mean_word_len -0.79, uses_curly_apos -0.55, type_token -0.51, curly_apos_pw -0.44 |
| claude-sonnet-5 | straight_apos_pw +1.04, uses_straight_apos +0.96, uses_em_dash +0.75, em_dash_pw +0.70, contraction_pw +0.65 | uses_curly_apos -0.55, curly_apos_pw -0.44, func_word_pc -0.30, semicolon_pw -0.30 |
| gemini-3.1-pro-preview | n_sents +0.45, we_pw +0.29, upper_pc +0.24, first_the +0.17, first_imagine +0.16 | uses_em_dash -0.85, non_ascii_pc -0.79, em_dash_pw -0.72, colon_pw -0.65 |
| gemini-3.6-flash | mean_word_len +0.49, ends_question +0.13, first_when +0.10, type_token +0.08, question_pw +0.05 | uses_em_dash -0.85, em_dash_pw -0.72, non_ascii_pc -0.68, colon_pw -0.65 |
| gemini-3.7-flash | mean_word_len +0.93, uses_curly_apos +0.25, type_token +0.24, markdown_italic +0.11, curly_apos_pw +0.07 | uses_em_dash -0.83, em_dash_pw -0.71, func_word_pc -0.64, colon_pw -0.64 |
| gemini-3.8-flash | mean_word_len +0.84, type_token +0.48, uses_curly_apos +0.10, first_a +0.09, first_when +0.03 | uses_em_dash -0.85, em_dash_pw -0.72, func_word_pc -0.63, colon_pw -0.62 |
| glm-5.3 | em_dash_pw +1.04, uses_em_dash +0.93, uses_straight_apos +0.62, straight_apos_pw +0.59, non_ascii_pc +0.47 | uses_curly_apos -0.55, curly_apos_pw -0.44, mean_word_len -0.28, curly_dq_pw -0.18 |
| glm-5.3-flash | em_dash_pw +0.88, uses_em_dash +0.88, uses_straight_apos +0.83, straight_apos_pw +0.63, non_ascii_pc +0.33 | uses_curly_apos -0.55, curly_apos_pw -0.44, rule_of_three -0.22, mean_word_len -0.21 |
| gpt-5.5 | rule_of_three +0.68, comma_pw +0.59, semicolon_pw +0.58, uses_curly_apos +0.55, curly_apos_pw +0.54 | uses_em_dash -0.56, mean_word_len -0.54, em_dash_pw -0.51, uses_straight_apos -0.45 |
| gpt-5.6-luna | uses_curly_apos +0.79, curly_apos_pw +0.59, paren_pw +0.42, colon_pw +0.42, rule_of_three +0.39 | uses_straight_apos -0.69, straight_apos_pw -0.53, sent_len_std -0.18, straight_dq_pw -0.18 |
| gpt-5.6-sol | uses_curly_apos +0.82, curly_apos_pw +0.73, curly_dq_pw +0.61, non_ascii_pc +0.41, semicolon_pw +0.40 | uses_straight_apos -0.71, straight_apos_pw -0.55, em_dash_pw -0.32, uses_em_dash -0.25 |
| gpt-5.6-terra | rule_of_three +0.69, uses_curly_apos +0.67, curly_apos_pw +0.58, colon_pw +0.45, comma_pw +0.45 | uses_straight_apos -0.59, straight_apos_pw -0.50, uses_em_dash -0.35, em_dash_pw -0.29 |
| gpt-6-astra | uses_curly_apos +0.60, curly_apos_pw +0.53, func_word_pc +0.35, n_sents +0.34, semicolon_pw +0.30 | uses_em_dash -0.66, em_dash_pw -0.56, uses_straight_apos -0.51, comma_pw -0.46 |
| kimi-k3 | semicolon_pw +0.69, colon_pw +0.59, non_ascii_pc +0.50, em_dash_pw +0.31, uses_curly_apos +0.29 | mean_word_len -0.34, n_sents -0.18, uses_straight_apos -0.17, ends_period -0.14 |
| qwen3.8-max | comma_pw +0.83, rule_of_three +0.62, curly_apos_pw +0.30, not_x_but_y +0.24, first_in +0.23 | uses_em_dash -0.77, em_dash_pw -0.66, uses_straight_apos -0.43, straight_apos_pw -0.36 |

### 1.6 Per lab

| lab | top + features (d) | top - features (d) |
|---|---|---|
| anthropic | uses_em_dash +1.03, uses_straight_apos +0.95, em_dash_pw +0.82, straight_apos_pw +0.79, sent_len_std +0.37 | uses_curly_apos -0.71, curly_apos_pw -0.57, semicolon_pw -0.25, type_token -0.24 |
| google | mean_word_len +0.71, type_token +0.18, ends_question +0.10, we_pw +0.08, first_when +0.07 | uses_em_dash -1.02, em_dash_pw -0.87, non_ascii_pc -0.79, colon_pw -0.78 |
| moonshot | semicolon_pw +0.69, colon_pw +0.59, non_ascii_pc +0.50, em_dash_pw +0.31, uses_curly_apos +0.29 | mean_word_len -0.34, n_sents -0.18, uses_straight_apos -0.17, ends_period -0.14 |
| openai | uses_curly_apos +0.89, curly_apos_pw +0.78, rule_of_three +0.48, colon_pw +0.46, curly_dq_pw +0.44 | uses_straight_apos -0.77, straight_apos_pw -0.62, uses_em_dash -0.49, em_dash_pw -0.48 |
| qwen | comma_pw +0.83, rule_of_three +0.62, curly_apos_pw +0.30, not_x_but_y +0.24, first_in +0.23 | uses_em_dash -0.77, em_dash_pw -0.66, uses_straight_apos -0.43, straight_apos_pw -0.36 |
| zhipu | em_dash_pw +1.02, uses_em_dash +0.96, uses_straight_apos +0.77, straight_apos_pw +0.65, non_ascii_pc +0.42 | uses_curly_apos -0.58, curly_apos_pw -0.47, mean_word_len -0.26, rule_of_three -0.20 |

### 1.7 Length: does it carry label signal?

Word count by model: eta^2 = 0.010; per-model median range 84-118 words.
Mean within-prompt percentile rank of length (0.5 = no effect): gemini-3.8-flash 0.45, kimi-k3 0.49, qwen3.8-max 0.50, gpt-5.5 0.51, glm-5.3 0.52, gemini-3.6-flash 0.52, claude-opus-4-6 0.53, claude-fable-5 0.53, gpt-5.6-luna 0.53, gpt-5.6-terra 0.53, claude-sonnet-5 0.53, gpt-5.6-sol 0.54, gemini-3.7-flash 0.54, gpt-6-astra 0.54, gemini-3.1-pro-preview 0.55, claude-opus-5 0.56, glm-5.3-flash 0.56, claude-opus-4-8 0.57

Per-prompt min/max words: median min 44, median max 231.

### 1.8 Opening words

Top first word per model (share %):

| model | most common first words |
|---|---|
| claude-fable-5 | the 12, in 9, when 5, imagine 5, at 4 |
| claude-opus-4-6 | the 14, in 8, imagine 6, when 5, consider 3 |
| claude-opus-4-8 | the 11, in 9, imagine 7, when 5, at 3 |
| claude-opus-5 | the 11, in 5, a 5, imagine 3, two 3 |
| claude-sonnet-5 | the 15, in 11, beneath 6, imagine 5, a 4 |
| gemini-3.1-pro-preview | the 15, imagine 7, in 7, when 5, to 4 |
| gemini-3.6-flash | when 7, in 6, beneath 6, the 5, deep 5 |
| gemini-3.7-flash | beneath 7, the 7, deep 6, when 6, in 5 |
| gemini-3.8-flash | beneath 6, the 6, across 5, when 5, in 5 |
| glm-5.3 | the 7, imagine 7, when 5, in 5, beneath 4 |
| glm-5.3-flash | the 13, in 8, deep 6, imagine 4, beneath 4 |
| gpt-5.5 | at 8, the 7, in 5, when 5, on 4 |
| gpt-5.6-luna | the 11, in 8, at 7, imagine 5, when 4 |
| gpt-5.6-sol | the 9, in 6, at 6, when 5, a 4 |
| gpt-5.6-terra | the 8, at 7, when 5, on 5, in 5 |
| gpt-6-astra | the 9, in 5, at 5, imagine 5, across 4 |
| kimi-k3 | the 9, in 8, a 6, at 6, when 4 |
| qwen3.8-max | in 13, the 11, when 5, across 3, deep 3 |

Within a prompt, the most common 2-word opening is shared by on average 42% of the 18 answers (median 39%). => openings are prompt-driven, a clustering cue, weak label cue.

### 1.9 Most model-specific tokens (log-odds ratio with informative Dirichlet prior, z-score)

| model | most over-used tokens (z) | most under-used (z) |
|---|---|---|
| claude-fable-5 | `—` 12.0, `n` 6.4, `she` 6.2, `c` 5.0, `pa` 4.8, `her` 4.1, `nothing` 4.1, `complete` 4.0, `"` 3.9, `shorter` 3.7 | `s` -4.7, `’` -4.3, `an` -2.9, `i` -2.7, `while` -2.3 |
| claude-opus-4-6 | `—` 9.4, `that` 5.1, `roughly` 5.0, `comfortable` 4.6, `most` 4.4, `doesn't` 4.0, `deeply` 4.0, `percent` 3.9, `underlying` 3.8, `emerges` 3.8 | `;` -5.7, `s` -4.7, `’` -4.3, `:` -3.2, `dark` -2.9 |
| claude-opus-4-8 | `you` 6.9, `quietly` 5.3, `we` 5.1, `glimpse` 4.4, `that` 4.4, `isn't` 3.9, `immensity` 3.9, `here` 3.8, `illuminate` 3.7, `retreat` 3.6 | `s` -5.4, `;` -4.6, `’` -4.4, `an` -2.7, `:` -2.5 |
| claude-opus-5 | `—` 11.1, `the` 7.9, `a` 7.8, `is` 7.7, `he` 7.2, `hundred` 6.2, `never` 5.7, `so` 5.4, `it` 5.4, `thing` 5.3 | `s` -5.1, `’` -4.4, `into` -3.9, `while` -3.8, `these` -3.6 |
| claude-sonnet-5 | `—` 8.1, `it's` 5.6, `itself` 5.4, `"` 5.1, `isn't` 4.6, `this` 4.2, `requires` 4.0, `entirely` 3.9, `since` 3.8, `earth's` 3.8 | `s` -4.8, `’` -4.3, `;` -4.2, `the` -3.7, `is` -3.6 |
| gemini-3.1-pro-preview | `of` 14.9, `profound` 9.3, `our` 8.2, `the` 8.0, `absolute` 7.5, `this` 7.1, `biological` 6.7, `microscopic` 6.5, `relentless` 6.2, `ultimate` 6.2 | `and` -7.7, `:` -7.1, `—` -6.4, `;` -5.0, `s` -5.0 |
| gemini-3.6-flash | `of` 9.0, `expanse` 6.8, `dynamic` 6.6, `void` 6.4, `breathtaking` 6.1, `mathematical` 6.0, `into` 6.0, `an` 5.9, `microscopic` 5.8, `physical` 5.7 | `:` -6.3, `—` -6.2, `is` -6.0, `and` -5.6, `each` -4.4 |
| gemini-3.7-flash | `an` 14.2, `of` 10.6, `profound` 7.8, `fundamentally` 6.5, `expanse` 5.8, `biological` 5.7, `these` 5.7, `intricate` 5.5, `this` 5.5, `across` 5.4 | `a` -7.0, `—` -6.9, `it` -6.8, `:` -6.3, `is` -4.9 |
| gemini-3.8-flash | `an` 13.7, `transforming` 6.2, `into` 6.1, `of` 6.0, `enduring` 6.0, `abyss` 5.8, `across` 5.8, `permanently` 5.6, `architecture` 5.5, `absolute` 5.4 | `a` -7.2, `—` -6.0, `:` -5.9, `is` -5.8, `it` -5.7 |
| glm-5.3 | `—` 14.2, `"` 6.4, `itself` 5.0, `every` 4.9, `balls` 4.7, `now` 4.6, `wire` 4.6, `billion` 4.3, `believe` 4.1, `nothing` 3.9 | `s` -4.9, `’` -4.3, `while` -2.8, `or` -2.4, `an` -2.4 |
| glm-5.3-flash | `—` 12.8, `"` 5.9, `overnight` 4.9, `p` 4.4, `scientists` 4.0, `lived` 3.9, `never` 3.9, `organ` 3.8, `crucially` 3.7, `ago` 3.7 | `s` -4.8, `’` -4.4, `an` -3.2, `into` -3.0, `system` -2.4 |
| gpt-5.5 | `’` 7.6, `s` 7.2, `;` 6.2, `speech` 4.9, `like` 4.7, `“` 4.5, `”` 4.5, `habits` 4.4, `less` 4.4, `and` 4.2 | `—` -6.6, `of` -6.0, `this` -5.0, `an` -4.0, `the` -3.5 |
| gpt-5.6-luna | `’` 10.0, `s` 9.4, `)` 8.1, `(` 8.1, `may` 6.1, `while` 5.6, `”` 5.6, `“` 5.6, `can` 5.2, `therefore` 5.1 | `of` -6.1, `the` -5.9, `our` -4.0, `within` -3.5, `you` -3.5 |
| gpt-5.6-sol | `’` 10.5, `s` 9.4, `”` 8.5, `“` 8.5, `;` 5.4, `and` 5.2, `another` 4.5, `travel` 4.5, `continually` 4.4, `or` 4.4 | `of` -7.5, `—` -4.9, `you` -3.8, `the` -3.7, `your` -3.6 |
| gpt-5.6-terra | `s` 9.4, `’` 9.2, `t` 6.3, `(` 6.0, `)` 6.0, `may` 5.7, `crowded` 5.4, `can` 4.4, `entered` 4.3, `institutions` 4.3 | `of` -6.2, `—` -4.6, `the` -4.6, `you` -4.4, `we` -4.2 |
| gpt-6-astra | `’` 7.3, `s` 6.7, `familiar` 6.2, `can` 6.0, `helps` 5.3, `helped` 5.2, `could` 5.2, `still` 5.2, `its` 5.1, `shelter` 5.0 | `—` -7.9, `of` -5.1, `in` -4.8, `so` -3.6, `the` -3.6 |
| kimi-k3 | `;` 8.7, `:` 5.5, `keeps` 5.5, `b` 5.4, `patience` 5.2, `bright` 5.1, `s` 5.0, `debt` 4.5, `before` 4.3, `after` 4.3 | `of` -7.3, `this` -5.1, `an` -3.3, `silent` -3.0, `will` -2.7 |
| qwen3.8-max | `turning` 5.9, `luminous` 5.3, `becomes` 5.2, `and` 4.6, `shape` 4.5, `lift` 4.0, `compassion` 3.9, `becoming` 3.9, `delayed` 3.8, `d` 3.8 | `—` -7.5, `of` -5.1, `the` -5.1, `an` -4.7, `its` -2.5 |

| lab | most over-used tokens (z) | most under-used (z) |
|---|---|---|
| anthropic | `—` 22.3, `"` 9.2, `that` 9.2, `so` 7.8, `never` 7.3, `you` 6.9, `years` 6.3, `in` 5.9, `itself` 5.8, `thousand` 5.7 | `s` -11.5, `an` -7.4, `;` -6.9, `’` -6.6, `can` -5.0 |
| google | `of` 22.5, `an` 19.8, `this` 12.1, `profound` 11.8, `our` 11.5, `absolute` 10.8, `these` 10.5, `void` 9.8, `biological` 9.7, `microscopic` 9.6 | `:` -14.0, `and` -12.0, `—` -10.3, `each` -9.6, `it` -9.1 |
| moonshot | `;` 8.7, `:` 5.5, `keeps` 5.5, `b` 5.4, `patience` 5.2, `bright` 5.1, `s` 5.0, `debt` 4.5, `before` 4.3, `after` 4.3 | `of` -7.3, `this` -5.1, `an` -3.3, `silent` -3.0, `will` -2.7 |
| openai | `’` 21.7, `s` 20.6, `can` 11.8, `may` 10.6, `“` 10.5, `”` 10.5, `;` 9.4, `:` 9.3, `and` 8.5, `therefore` 8.2 | `of` -16.1, `—` -14.4, `the` -11.1, `this` -8.5, `you` -7.4 |
| qwen | `turning` 5.9, `luminous` 5.3, `becomes` 5.2, `and` 4.6, `shape` 4.5, `lift` 4.0, `compassion` 3.9, `becoming` 3.9, `delayed` 3.8, `d` 3.8 | `—` -7.5, `of` -5.1, `the` -5.1, `an` -4.7, `its` -2.5 |
| zhipu | `—` 19.6, `"` 9.0, `itself` 5.7, `every` 5.3, `nothing` 5.2, `never` 5.0, `now` 4.9, `:` 4.2, `billion` 4.1, `way` 3.9 | `s` -6.7, `’` -5.2, `an` -4.1, `while` -3.0, `physical` -2.9 |
