# Lexical and stylometric properties per model variant

Train set only (2,699 passages, 150 prompts). Each prompt is answered by every model, so differences are not topic effects.
Rates: /100w = per 100 words; 0/1 features are the share of passages containing the item.

## 1. Full profile (per-model means)

| feature | gpt-5.5 | gpt-5.6-luna | gpt-5.6-sol | gpt-5.6-terra | gpt-6-astra | gem-3.1-pro | gem-3.6-flash | gem-3.7-flash | gem-3.8-flash | cl-fable-5 | cl-opus-4-6 | cl-opus-4-8 | cl-opus-5 | cl-sonnet-5 | glm-5.3 | glm-5.3-flash | qwen3.8-max | kimi-k3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| words | 110.47 | 115.82 | 119.21 | 113.65 | 117.76 | 119.31 | 111.11 | 117.11 | _100.68_ | 113.95 | 115.27 | **124.53** | 123.85 | 121.27 | 111.71 | 120.48 | 108.40 | 108.15 |
| sentences | 4.33 | 4.89 | 4.94 | 4.45 | 5.55 | **5.77** | 4.59 | 4.81 | 4.42 | 4.71 | _4.27_ | 5.07 | 5.16 | 4.55 | 4.85 | 5.37 | 4.97 | 4.49 |
| words/sentence | 26.33 | 23.79 | 24.97 | 26.36 | 21.13 | _20.25_ | 24.65 | 23.87 | 22.73 | 24.44 | **27.30** | 24.44 | 23.96 | 26.51 | 23.63 | 23.50 | 22.23 | 25.91 |
| sent_len_sd | 5.69 | 5.37 | 6.15 | 6.14 | 5.23 | 5.21 | _5.00_ | 5.01 | 5.13 | 6.08 | 7.15 | 6.59 | **8.49** | 5.97 | 6.62 | 6.02 | 5.32 | 6.10 |
| sent_len_cv | 0.23 | 0.23 | 0.26 | 0.25 | 0.25 | 0.26 | 0.22 | _0.21_ | 0.23 | 0.26 | 0.27 | 0.28 | **0.37** | 0.24 | 0.30 | 0.28 | 0.25 | 0.26 |
| max_sent_len | 34.56 | 31.41 | 33.97 | 35.17 | 28.83 | _28.20_ | 31.86 | 31.32 | 29.91 | 32.85 | **37.19** | 33.84 | 35.83 | 35.11 | 32.98 | 32.01 | 29.76 | 34.49 |
| min_sent_len | 19.02 | 16.63 | 16.98 | 18.60 | 13.94 | 13.24 | 18.45 | 17.48 | 16.29 | 16.16 | 18.25 | 15.44 | _12.15_ | **19.12** | 14.94 | 15.15 | 15.27 | 17.89 |
| word_len | 5.19 | 5.51 | 5.43 | 5.41 | 5.26 | 5.50 | 5.72 | **5.94** | 5.90 | 5.33 | 5.64 | 5.44 | _5.06_ | 5.65 | 5.32 | 5.36 | 5.46 | 5.29 |
| long_words(>=10) | 0.06 | 0.09 | 0.07 | 0.08 | 0.06 | 0.09 | 0.10 | **0.12** | 0.12 | 0.08 | 0.10 | 0.08 | _0.06_ | 0.10 | 0.07 | 0.07 | 0.08 | 0.07 |
| short_words(<=3) | 0.30 | 0.28 | 0.27 | 0.28 | 0.29 | 0.29 | 0.26 | 0.25 | _0.24_ | 0.29 | 0.27 | 0.29 | **0.33** | 0.26 | 0.29 | 0.29 | 0.27 | 0.29 |
| type_token | 0.83 | 0.82 | 0.83 | 0.83 | 0.82 | 0.80 | 0.83 | 0.84 | **0.86** | 0.81 | 0.82 | 0.80 | _0.78_ | 0.82 | 0.81 | 0.81 | 0.84 | 0.84 |
| hapax_ratio | 0.74 | 0.73 | 0.74 | 0.74 | 0.73 | 0.72 | 0.74 | 0.76 | **0.79** | 0.72 | 0.72 | 0.71 | _0.68_ | 0.73 | 0.72 | 0.71 | 0.75 | 0.75 |
| commas/sentence | **2.93** | 2.46 | 2.57 | 2.86 | 1.67 | _1.53_ | 2.13 | 1.92 | 1.99 | 2.29 | 2.15 | 2.39 | 2.10 | 2.61 | 2.19 | 2.13 | 2.62 | 2.52 |
| commas/100w | 11.06 | 10.32 | 10.17 | 10.63 | 7.90 | _7.57_ | 8.37 | 8.27 | 8.53 | 9.20 | 7.80 | 9.54 | 8.41 | 9.69 | 9.06 | 8.77 | **11.78** | 10.03 |
| semicolons/100w | 0.76 | 0.45 | 0.64 | 0.52 | 0.59 | 0.16 | 0.20 | 0.20 | 0.28 | 0.35 | _0.12_ | 0.19 | 0.59 | 0.23 | 0.37 | 0.53 | 0.38 | **0.82** |
| colons/100w | 0.82 | 0.88 | 0.79 | 0.90 | 0.76 | _0.16_ | 0.17 | 0.17 | 0.18 | 0.68 | 0.37 | 0.50 | 0.80 | 0.59 | 0.75 | 0.74 | 0.49 | **0.99** |
| em_dash/100w | 0.18 | 0.49 | 0.34 | 0.37 | 0.14 | _0.00_ | _0.00_ | 0.01 | _0.00_ | 1.37 | 1.14 | 0.86 | 1.23 | 1.23 | **1.52** | 1.38 | 0.05 | 0.89 |
| en_dash/100w | _0.00_ | 0.02 | 0.01 | 0.02 | _0.00_ | _0.00_ | _0.00_ | _0.00_ | _0.00_ | _0.00_ | _0.00_ | _0.00_ | 0.01 | _0.00_ | _0.00_ | 0.01 | _0.00_ | **0.03** |
| spaced_hyphen/100w | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** |
| hyphenated_words/100w | 0.66 | 0.49 | 0.51 | 0.60 | 0.32 | _0.11_ | 0.13 | 0.23 | 0.17 | 0.54 | 0.73 | 0.55 | 0.57 | **0.81** | 0.60 | 0.61 | 0.53 | 0.59 |
| parens/100w | _0.00_ | **0.14** | 0.01 | 0.06 | 0.01 | _0.00_ | 0.01 | 0.01 | _0.00_ | 0.04 | _0.00_ | 0.01 | 0.01 | 0.00 | 0.01 | _0.00_ | _0.00_ | 0.00 |
| curly_apos | 0.42 | 0.51 | **0.53** | 0.47 | 0.44 | 0.06 | 0.19 | 0.31 | 0.25 | _0.00_ | _0.00_ | _0.00_ | _0.00_ | _0.00_ | _0.00_ | _0.00_ | 0.29 | 0.32 |
| straight_apos | 0.11 | 0.01 | _0.00_ | 0.05 | 0.09 | 0.11 | 0.21 | 0.14 | 0.11 | 0.60 | 0.57 | 0.60 | 0.63 | **0.73** | 0.58 | 0.67 | 0.12 | 0.23 |
| curly_dquote | 0.09 | 0.11 | **0.15** | 0.08 | 0.07 | _0.00_ | _0.00_ | _0.00_ | 0.02 | _0.00_ | _0.00_ | _0.00_ | _0.00_ | _0.00_ | _0.00_ | _0.00_ | 0.01 | 0.09 |
| straight_dquote | 0.01 | _0.00_ | _0.00_ | _0.00_ | _0.00_ | 0.03 | 0.03 | _0.00_ | 0.01 | 0.09 | 0.07 | 0.07 | 0.09 | 0.10 | 0.09 | **0.12** | _0.00_ | 0.01 |
| ellipsis | _0.00_ | _0.00_ | _0.00_ | _0.00_ | _0.00_ | _0.00_ | _0.00_ | _0.00_ | _0.00_ | _0.00_ | _0.00_ | _0.00_ | **0.01** | _0.00_ | _0.00_ | _0.00_ | _0.00_ | _0.00_ |
| asterisk | _0.00_ | 0.01 | _0.00_ | _0.00_ | _0.00_ | _0.00_ | _0.00_ | 0.01 | _0.00_ | _0.00_ | **0.03** | 0.03 | 0.01 | 0.01 | _0.00_ | _0.00_ | _0.00_ | _0.00_ |
| exclaim | 0.01 | _0.00_ | 0.01 | _0.00_ | _0.00_ | 0.01 | _0.00_ | _0.00_ | _0.00_ | _0.00_ | 0.01 | _0.00_ | 0.01 | 0.01 | **0.01** | _0.00_ | _0.00_ | _0.00_ |
| question | 0.02 | 0.04 | 0.03 | 0.01 | 0.05 | **0.05** | 0.04 | 0.03 | 0.02 | 0.05 | 0.03 | 0.05 | 0.03 | 0.01 | 0.04 | 0.04 | _0.01_ | 0.02 |
| ends_with_period | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | _0.99_ |
| digits/100c | 0.06 | 0.07 | 0.07 | 0.06 | 0.07 | 0.03 | 0.02 | 0.03 | _0.02_ | 0.10 | 0.10 | 0.06 | 0.10 | 0.07 | **0.11** | 0.10 | 0.03 | 0.09 |
| non_ascii/100c | 0.19 | 0.23 | 0.24 | 0.20 | 0.16 | _0.02_ | 0.04 | 0.06 | 0.06 | 0.23 | 0.18 | 0.14 | 0.21 | 0.19 | 0.25 | 0.23 | 0.10 | **0.26** |
| capitalized_mid/100w | 4.33 | 4.74 | 4.71 | _4.22_ | 4.75 | **5.43** | 4.55 | 4.39 | 4.57 | 4.96 | 4.52 | 4.77 | 4.98 | 4.46 | 5.20 | 5.24 | 4.59 | 4.64 |
| contractions/100w | 0.84 | 0.73 | 0.81 | 0.76 | 0.82 | _0.18_ | 0.46 | 0.53 | 0.47 | 1.00 | 0.99 | 0.91 | 1.01 | **1.49** | 1.06 | 1.08 | 0.69 | 0.89 |
| hedges/100w | 0.25 | 0.33 | 0.27 | 0.31 | 0.29 | 0.09 | _0.05_ | 0.07 | 0.07 | **0.34** | 0.20 | 0.24 | 0.22 | 0.21 | 0.26 | 0.31 | 0.20 | 0.13 |
| intensifiers/100w | 0.06 | 0.09 | 0.06 | 0.09 | 0.05 | **0.66** | 0.49 | 0.53 | 0.38 | 0.13 | 0.27 | 0.23 | _0.03_ | 0.11 | 0.09 | 0.11 | 0.11 | 0.10 |
| first_person/100w | 0.92 | 0.91 | 0.91 | _0.68_ | 1.00 | **1.53** | 0.97 | 0.86 | 0.88 | 0.95 | 0.89 | 1.25 | 0.70 | 0.71 | 0.93 | 1.11 | 1.02 | 0.96 |
| second_person/100w | 0.33 | 0.31 | 0.24 | _0.18_ | 0.69 | 0.60 | 0.56 | 0.54 | 0.60 | 0.60 | 0.71 | **0.94** | 0.71 | 0.53 | 0.70 | 0.71 | 0.47 | 0.47 |
| the/100w | 4.52 | 4.29 | 4.54 | 4.47 | 4.64 | 6.03 | 5.20 | 5.29 | 5.16 | 5.14 | 5.13 | 5.09 | **6.10** | 4.12 | 5.04 | 5.19 | _4.11_ | 4.75 |
| and/100w | 3.11 | 3.15 | **3.23** | 2.99 | 2.55 | _1.54_ | 1.80 | 1.89 | 1.99 | 2.64 | 2.33 | 2.67 | 2.71 | 2.39 | 2.44 | 2.34 | 3.17 | 2.86 |
| of/100w | 1.65 | 1.67 | 1.50 | 1.64 | 1.75 | **3.73** | 3.22 | 3.17 | 2.80 | 2.31 | 2.58 | 2.35 | 2.33 | 2.11 | 2.18 | 2.15 | 1.60 | _1.40_ |
| this/100w | 0.22 | 0.28 | 0.32 | 0.26 | 0.40 | **0.86** | 0.67 | 0.65 | 0.62 | 0.44 | 0.40 | 0.65 | 0.27 | 0.66 | 0.53 | 0.46 | 0.37 | _0.19_ |
| not/100w | 1.13 | 1.00 | 1.03 | 0.93 | **1.19** | 0.44 | _0.43_ | 0.51 | 0.48 | 0.67 | 0.81 | 0.76 | 1.06 | 0.87 | 0.97 | 0.96 | 1.07 | 1.05 |
| ly_adverbs/100w | 1.22 | 1.70 | 1.87 | 1.89 | 1.42 | **2.94** | 2.72 | 2.91 | 2.80 | 1.99 | 2.05 | 2.40 | 1.72 | 2.12 | 1.69 | 1.97 | 1.54 | _1.18_ |
| ing/100w | 4.22 | 4.80 | 4.38 | _4.17_ | 4.51 | 5.84 | 5.93 | **5.95** | 5.95 | 4.73 | 5.59 | 5.69 | 4.46 | 5.75 | 4.71 | 4.85 | 5.44 | 4.55 |
| tion/100w | 1.36 | **1.91** | 1.49 | 1.53 | 1.22 | 1.44 | 1.29 | 1.56 | 1.47 | 1.44 | 1.71 | 1.58 | _1.22_ | 1.84 | 1.30 | 1.35 | 1.50 | 1.36 |
| sent_starts_conj | 0.02 | 0.04 | 0.02 | 0.04 | 0.05 | _0.01_ | 0.02 | 0.02 | 0.03 | 0.03 | 0.02 | 0.04 | **0.05** | 0.03 | 0.04 | 0.03 | 0.02 | 0.03 |
| sent_starts_this_it | 0.04 | 0.06 | 0.05 | 0.05 | 0.04 | **0.12** | 0.07 | 0.08 | 0.05 | 0.06 | 0.10 | 0.11 | 0.05 | 0.10 | _0.03_ | 0.07 | 0.06 | 0.04 |
| sent_starts_the | 0.09 | 0.10 | 0.08 | 0.10 | 0.10 | 0.09 | _0.04_ | 0.04 | 0.05 | 0.10 | **0.13** | 0.11 | 0.11 | 0.09 | 0.11 | 0.09 | 0.10 | 0.10 |
| not_x_but_y | 0.23 | 0.18 | 0.19 | 0.21 | **0.30** | 0.19 | 0.17 | 0.16 | 0.15 | _0.05_ | 0.07 | 0.07 | 0.07 | 0.10 | 0.07 | 0.09 | 0.23 | 0.16 |
| its_not_x_its_y | 0.01 | 0.02 | 0.02 | **0.05** | 0.01 | 0.01 | _0.00_ | _0.00_ | 0.01 | _0.00_ | 0.01 | 0.01 | 0.02 | 0.01 | 0.03 | 0.01 | 0.03 | 0.01 |
| imagine/picture | 0.06 | 0.11 | 0.10 | 0.09 | 0.11 | 0.11 | 0.05 | 0.05 | 0.05 | 0.09 | 0.09 | **0.12** | 0.09 | 0.08 | 0.11 | 0.09 | 0.07 | _0.04_ |

Bold = highest model, italic = lowest.

## 2. What separates sibling variants

eta² = share of within-prompt variance explained by which model wrote the passage (prompt mean removed). Classifier = logistic regression on these ~55 hand features only, 5-fold GroupKFold by prompt, balanced accuracy (chance = 1/k). "prompt-centred" = features minus their prompt mean, the oracle upper bound if test prompts were known.

### OpenAI (5 variants, chance 0.20)

Hand-feature classifier: **0.392** balanced accuracy (prompt-centred: 0.508).

| feature | eta² | gpt-5.5 | gpt-5.6-luna | gpt-5.6-sol | gpt-5.6-terra | gpt-6-astra |
|---|---|---|---|---|---|---|
| commas/100w | 0.233 | 11.06 | 10.32 | 10.17 | 10.63 | 7.90 |
| word_len | 0.206 | 5.19 | 5.51 | 5.43 | 5.41 | 5.26 |
| commas/sentence | 0.167 | 2.93 | 2.46 | 2.57 | 2.86 | 1.67 |
| long_words(>=10) | 0.132 | 0.06 | 0.09 | 0.07 | 0.08 | 0.06 |
| words/sentence | 0.109 | 26.33 | 23.79 | 24.97 | 26.36 | 21.13 |
| max_sent_len | 0.084 | 34.56 | 31.41 | 33.97 | 35.17 | 28.83 |
| min_sent_len | 0.072 | 19.02 | 16.63 | 16.98 | 18.60 | 13.94 |
| ly_adverbs/100w | 0.066 | 1.22 | 1.70 | 1.87 | 1.89 | 1.42 |
| short_words(<=3) | 0.061 | 0.30 | 0.28 | 0.27 | 0.28 | 0.29 |
| second_person/100w | 0.056 | 0.33 | 0.31 | 0.24 | 0.18 | 0.69 |
| tion/100w | 0.054 | 1.36 | 1.91 | 1.49 | 1.53 | 1.22 |
| and/100w | 0.052 | 3.11 | 3.15 | 3.23 | 2.99 | 2.55 |

### Google (4 variants, chance 0.25)

Hand-feature classifier: **0.468** balanced accuracy (prompt-centred: 0.547).

| feature | eta² | gemini-3.1-pro-preview | gemini-3.6-flash | gemini-3.7-flash | gemini-3.8-flash |
|---|---|---|---|---|---|
| word_len | 0.359 | 5.50 | 5.72 | 5.94 | 5.90 |
| short_words(<=3) | 0.259 | 0.29 | 0.26 | 0.25 | 0.24 |
| long_words(>=10) | 0.177 | 0.09 | 0.10 | 0.12 | 0.12 |
| capitalized_mid/100w | 0.141 | 5.43 | 4.55 | 4.39 | 4.57 |
| type_token | 0.106 | 0.80 | 0.83 | 0.84 | 0.86 |
| hapax_ratio | 0.099 | 0.72 | 0.74 | 0.76 | 0.79 |
| sent_starts_this_it | 0.096 | 0.12 | 0.07 | 0.08 | 0.05 |
| sentences | 0.090 | 5.77 | 4.59 | 4.81 | 4.42 |
| curly_apos | 0.082 | 0.06 | 0.19 | 0.31 | 0.25 |
| first_person/100w | 0.077 | 1.53 | 0.97 | 0.86 | 0.88 |
| min_sent_len | 0.077 | 13.24 | 18.45 | 17.48 | 16.29 |
| sent_starts_the | 0.070 | 0.09 | 0.04 | 0.04 | 0.05 |

### Anthropic (5 variants, chance 0.20)

Hand-feature classifier: **0.443** balanced accuracy (prompt-centred: 0.579).

| feature | eta² | claude-fable-5 | claude-opus-4-6 | claude-opus-4-8 | claude-opus-5 | claude-sonnet-5 |
|---|---|---|---|---|---|---|
| word_len | 0.406 | 5.33 | 5.64 | 5.44 | 5.06 | 5.65 |
| long_words(>=10) | 0.293 | 0.08 | 0.10 | 0.08 | 0.06 | 0.10 |
| short_words(<=3) | 0.227 | 0.29 | 0.27 | 0.29 | 0.33 | 0.26 |
| sent_len_cv | 0.128 | 0.26 | 0.27 | 0.28 | 0.37 | 0.24 |
| semicolons/100w | 0.121 | 0.35 | 0.12 | 0.19 | 0.59 | 0.23 |
| the/100w | 0.120 | 5.14 | 5.13 | 5.09 | 6.10 | 4.12 |
| commas/100w | 0.114 | 9.20 | 7.80 | 9.54 | 8.41 | 9.69 |
| min_sent_len | 0.114 | 16.16 | 18.25 | 15.44 | 12.15 | 19.12 |
| colons/100w | 0.083 | 0.68 | 0.37 | 0.50 | 0.80 | 0.59 |
| this/100w | 0.081 | 0.44 | 0.40 | 0.65 | 0.27 | 0.66 |
| ing/100w | 0.075 | 4.73 | 5.59 | 5.69 | 4.46 | 5.75 |
| sent_len_sd | 0.063 | 6.08 | 7.15 | 6.59 | 8.49 | 5.97 |

### ZAI (2 variants, chance 0.50)

Hand-feature classifier: **0.573** balanced accuracy (prompt-centred: 0.580).

| feature | eta² | glm-5.3 | glm-5.3-flash |
|---|---|---|---|
| sent_starts_this_it | 0.088 | 0.03 | 0.07 |
| semicolons/100w | 0.036 | 0.37 | 0.53 |
| ly_adverbs/100w | 0.031 | 1.69 | 1.97 |
| sentences | 0.028 | 4.85 | 5.37 |
| sent_len_sd | 0.021 | 6.62 | 6.02 |
| straight_apos | 0.020 | 0.58 | 0.67 |
| non_ascii/100c | 0.018 | 0.25 | 0.23 |
| sent_starts_the | 0.015 | 0.11 | 0.09 |
| em_dash/100w | 0.013 | 1.52 | 1.38 |
| exclaim | 0.013 | 0.01 | 0.00 |
| parens/100w | 0.013 | 0.01 | 0.00 |
| sent_len_cv | 0.012 | 0.30 | 0.28 |

### Confusable: GLM vs claude-opus-5 (3 variants, chance 0.33)

Hand-feature classifier: **0.500** balanced accuracy (prompt-centred: 0.564).

| feature | eta² | glm-5.3 | glm-5.3-flash | claude-opus-5 |
|---|---|---|---|---|
| word_len | 0.216 | 5.32 | 5.36 | 5.06 |
| short_words(<=3) | 0.166 | 0.29 | 0.29 | 0.33 |
| sent_len_cv | 0.119 | 0.30 | 0.28 | 0.37 |
| sent_len_sd | 0.119 | 6.62 | 6.02 | 8.49 |
| long_words(>=10) | 0.077 | 0.07 | 0.07 | 0.06 |
| the/100w | 0.073 | 5.04 | 5.19 | 6.10 |
| min_sent_len | 0.058 | 14.94 | 15.15 | 12.15 |
| this/100w | 0.056 | 0.53 | 0.46 | 0.27 |
| sent_starts_this_it | 0.047 | 0.03 | 0.07 | 0.05 |
| type_token | 0.040 | 0.81 | 0.81 | 0.78 |
| max_sent_len | 0.038 | 32.98 | 32.01 | 35.83 |
| first_person/100w | 0.031 | 0.93 | 1.11 | 0.70 |

### Confusable: kimi vs gpt-5.5 (2 variants, chance 0.50)

Hand-feature classifier: **0.650** balanced accuracy (prompt-centred: 0.713).

| feature | eta² | kimi-k3 | gpt-5.5 |
|---|---|---|---|
| em_dash/100w | 0.253 | 0.89 | 0.18 |
| commas/100w | 0.073 | 10.03 | 11.06 |
| word_len | 0.061 | 5.29 | 5.19 |
| straight_apos | 0.060 | 0.23 | 0.11 |
| commas/sentence | 0.053 | 2.52 | 2.93 |
| non_ascii/100c | 0.050 | 0.26 | 0.19 |
| hedges/100w | 0.050 | 0.13 | 0.25 |
| short_words(<=3) | 0.038 | 0.29 | 0.30 |
| colons/100w | 0.033 | 0.99 | 0.82 |
| sent_len_cv | 0.025 | 0.26 | 0.23 |
| curly_apos | 0.024 | 0.32 | 0.42 |
| of/100w | 0.022 | 1.40 | 1.65 |

## 3. Signature words and bigrams

Log-odds ratio with informative Dirichlet prior (z-score), model vs its siblings (vs all other models for single-model labs). Only items in at least 5 of the model's 150 passages. Format: item (docs with it in this model / mean docs in siblings).

- **gpt-5.5** vs siblings: "of" (113/114), "in" (109/94), "the" (139/144), "you" (14/13), "as if" (14/3), "your" (15/11), "so" (28/18), "never" (11/11), "within" (9/11), "begin to" (5/0), "whispering" (5/1), "with a" (17/6), "to the" (15/12), "of the" (17/22), "upon" (5/3), "bloom" (8/2), "in this" (7/3), "speech" (7/1)
- **gpt-5.6-luna** vs siblings: "of" (117/113), "this" (46/46), "records" (9/1), "architecture" (10/4), "never" (11/11), "an" (70/54), "entire" (7/4), "these" (23/16), "of the" (19/22), "you" (11/14), "within" (7/12), "the" (144/143), "transformed" (6/1), "structure" (8/2), "shell" (5/1), "down" (5/3), "was" (15/12), "transforming" (5/2)
- **gpt-5.6-sol** vs siblings: "of" (105/116), "the" (148/142), "within" (14/10), "ever" (12/4), "this" (51/45), "entire" (8/3), "finally" (6/1), "never" (12/11), "that the" (14/6), "itself" (30/20), "thousand" (7/3), "again" (15/5), "transforming" (6/2), "limited" (5/0), "we are" (5/2), "starlight" (6/1), "cosmic" (13/8), "flame" (6/1)
- **gpt-5.6-terra** vs siblings: "of" (118/112), "within" (12/10), "the" (143/143), "this" (43/47), "of the" (21/21), "early" (7/1), "instant" (7/2), "was" (12/13), "human" (21/13), "never" (7/12), "in" (93/98), "in a" (27/19), "silent" (13/10), "to the" (15/12), "tastes" (6/1), "into the" (18/13), "where the" (7/3), "to speak" (5/1)
- **gpt-6-astra** vs siblings: "of" (115/113), "our" (36/20), "your" (24/8), "you" (20/11), "this" (61/42), "we" (33/24), "the" (142/144), "of a" (24/13), "these" (29/15), "never" (14/10), "earth's" (5/0), "a familiar" (6/0), "here" (16/6), "of our" (5/1), "of the" (28/19), "nothing" (19/8), "drawing" (7/0), "something" (16/6)
- **gemini-3.1-pro-preview** vs siblings: "a" (142/125), "is" (83/52), "it" (74/45), "it is" (33/5), "we" (43/22), "was a" (10/1), "they" (38/19), "would" (19/10), "not" (51/42), "are" (51/25), "can" (24/15), "navigate the" (5/0), "strips away" (6/0), "but" (46/39), "each" (15/15), "exactly" (7/0), "is the" (23/7), "one" (25/20)
- **gemini-3.6-flash** vs siblings: "and" (122/124), "a" (134/127), "each" (16/15), "it" (53/52), "so" (11/7), "like" (28/25), "instant the" (5/0), "can" (14/18), "not" (42/45), "what" (11/10), "of physical" (5/0), "one" (21/21), "on" (35/26), "could" (8/6), "no" (25/26), "whose" (7/3), "expands" (5/0), "beyond" (14/7)
- **gemini-3.7-flash** vs siblings: "and" (127/122), "than" (23/13), "an intricate" (10/1), "each" (15/15), "no" (31/24), "what" (14/9), "like" (29/25), "can" (17/17), "stillness of" (5/0), "ground" (6/0), "and the" (18/14), "one" (21/21), "not" (43/45), "only" (24/16), "or" (38/28), "then" (5/5), "itself" (19/14), "is" (55/61)
- **gemini-3.8-flash** vs siblings: "and" (127/122), "each" (15/15), "still" (8/3), "so" (10/7), "like" (26/26), "than" (14/16), "not" (42/45), "one" (17/22), "can" (13/18), "and the" (13/15), "who" (5/4), "no" (23/27), "even" (15/8), "more" (13/11), "becomes" (11/10), "cannot" (7/6), "those" (6/2), "that" (78/88)
- **claude-fable-5** vs siblings: "an" (52/52), "when" (40/28), "can" (29/28), "rings" (6/1), "conversation" (6/1), "computer" (5/1), "may" (13/11), "difference" (6/1), "also" (8/4), "instantly" (5/1), "archive" (5/1), "time" (26/18), "merely" (11/12), "thus" (11/5), "end" (6/2), "cross" (5/2), "bright" (6/2), "works" (5/1)
- **claude-opus-4-6** vs siblings: "deeply" (7/0), "an" (50/53), "human" (21/11), "the architecture" (9/1), "comfortable" (5/0), "mathematical" (9/4), "merely" (15/11), "void" (11/5), "produce" (6/1), "carrying" (10/4), "profound" (5/2), "molecular" (12/3), "more than" (8/3), "even" (20/12), "these" (29/20), "unprecedented" (5/1), "complex" (9/3), "ancient" (20/14)
- **claude-opus-4-8** vs siblings: "an" (61/50), "my" (5/2), "when we" (5/0), "while" (33/24), "our" (33/23), "remember" (8/2), "dark" (20/15), "glimpse" (7/1), "immensity" (6/1), "hidden" (14/7), "beneath" (34/20), "delicate" (5/1), "person" (10/3), "quietly" (25/9), "within the" (5/2), "appear" (6/1), "distant" (16/8), "how" (20/12)
- **claude-opus-5** vs siblings: "a" (146/134), "is" (96/74), "he" (13/4), "rumor" (7/0), "and it" (10/2), "part" (7/1), "meant" (5/0), "learned to" (9/1), "the small" (6/0), "behind" (22/8), "which" (31/14), "room" (12/6), "the" (148/144), "learned" (10/2), "has learned" (5/0), "can" (32/27), "that is" (9/1), "turns" (10/2)
- **claude-sonnet-5** vs siblings: "into" (94/74), "while" (38/23), "rich" (7/0), "an" (52/52), "through" (62/46), "life itself" (5/0), "sophisticated" (5/0), "simple" (10/4), "near" (12/3), "this" (80/61), "even" (19/12), "requires" (9/2), "relentless" (6/2), "hush" (7/2), "minds" (10/3), "imagined" (5/1), "violently" (5/1), "digital" (7/2)
- **glm-5.3** vs siblings: "wire" (5/0), "by its" (6/0), "ball" (5/1), "of their" (5/0), "exchange" (5/0), "stand" (9/2), "note" (6/3), "so" (39/24), "per" (6/1), "nowhere" (5/0), "system" (13/5), "fragments" (9/3), "and then" (5/1), "someone" (9/3), "as a" (20/12), "your own" (5/0), "words" (8/3), "believe" (7/3)
- **glm-5.3-flash** vs siblings: "overnight" (6/0), "crucially" (5/0), "producing" (5/0), "humans" (6/1), "minds" (12/3), "reasoning" (5/0), "real" (8/1), "door" (7/2), "a time" (5/1), "waits" (6/0), "scientists" (11/4), "we are" (12/3), "threading" (5/0), "at its" (5/1), "becomes the" (5/0), "we" (41/32), "the result" (5/0), "adrift" (5/0)
- **qwen3.8-max** vs all others: "turning" (23/7), "becomes a" (22/7), "luminous" (19/6), "becomes" (38/21), "shape" (14/4), "lift" (6/1), "and" (136/133), "a luminous" (6/1), "pressure and" (5/1), "as if" (14/4), "becoming" (10/3), "making" (14/5), "a whisper" (8/2), "pressure" (15/6), "loss" (8/2), "not only" (7/2), "maps" (7/2), "forests" (7/2)
- **kimi-k3** vs all others: "the patience" (5/0), "patience of" (5/0), "debt" (5/0), "keeps" (16/4), "patience" (11/2), "a held" (5/0), "bright" (16/4), "held breath" (5/0), "steering" (5/0), "shining" (5/1), "patiently" (6/1), "lightning" (7/2), "after" (19/8), "not the" (11/3), "before" (37/23), "lets" (7/2), "becomes" (35/21), "go" (7/1)

## 4. Most common first two words per model

- **gpt-5.5**: "across the" (4), "at the" (3), "in a" (3), "when a" (3), "on a" (3)
- **gpt-5.6-luna**: "at the" (5), "in the" (5), "imagine a" (4), "in a" (3), "across the" (3)
- **gpt-5.6-sol**: "in the" (5), "at the" (4), "when a" (3), "deep within" (2), "imagine a" (2)
- **gpt-5.6-terra**: "at the" (6), "across the" (4), "when the" (4), "within the" (3), "beyond the" (3)
- **gpt-6-astra**: "across the" (4), "in the" (4), "imagine a" (4), "at the" (3), "on a" (2)
- **gemini-3.1-pro-preview**: "in the" (7), "imagine a" (5), "deep within" (4), "beneath the" (3), "when a" (3)
- **gemini-3.6-flash**: "in the" (8), "beneath the" (8), "deep within" (6), "at the" (4), "long before" (4)
- **gemini-3.7-flash**: "beneath the" (9), "deep within" (7), "at the" (6), "when the" (5), "across the" (4)
- **gemini-3.8-flash**: "beneath the" (7), "across the" (6), "in the" (6), "at the" (5), "deep within" (4)
- **claude-fable-5**: "in the" (8), "at the" (5), "beneath the" (4), "consider the" (3), "when the" (2)
- **claude-opus-4-6**: "in the" (8), "at the" (4), "imagine a" (2), "if every" (2), "deep beneath" (2)
- **claude-opus-4-8**: "in the" (10), "imagine a" (5), "at the" (4), "when a" (3), "beneath the" (3)
- **claude-opus-5**: "in the" (4), "there is" (4), "beneath the" (3), "imagine a" (3), "on a" (3)
- **claude-sonnet-5**: "in the" (12), "beneath the" (5), "at the" (3), "in a" (2), "there is" (2)
- **glm-5.3**: "beneath the" (4), "deep within" (4), "at the" (3), "in the" (3), "when a" (3)
- **glm-5.3-flash**: "beneath the" (5), "in the" (4), "deep in" (4), "in a" (3), "long before" (3)
- **qwen3.8-max**: "in the" (7), "in a" (5), "deep in" (3), "across the" (3), "at the" (3)
- **kimi-k3**: "in the" (7), "at the" (4), "within the" (3), "in a" (3), "beneath the" (3)
