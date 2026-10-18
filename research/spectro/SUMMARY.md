# Text spectrograms: CV and most distinctive cells per model

Figures: [char](char.png) / [char fingerprint](char_fingerprint.png), [word](word.png) / [word fingerprint](word_fingerprint.png), [sentence](sentence.png) / [sentence fingerprint](sentence_fingerprint.png), [token](token.png) / [token fingerprint](token_fingerprint.png). Colour = log2(model / all-model mean). Numbers: experiments/spectro/out/*_cells.csv, *_fingerprint_*.csv.

| level | standalone | delta on strong | member w | folds+ |
|---|---|---|---|---|
| char | 0.1919 | -0.0063 | 0.1 | 0/5 |
| char_collapsed | 0.2611 | -0.0026 | 0.1 | 1/5 |
| word | 0.0974 | -0.0019 | 0.1 | 1/5 |
| word_collapsed | 0.1382 | -0.0000 | 0.25 | 2/5 |
| sentence | 0.0989 | -0.0004 | 0.25 | 2/5 |
| sentence_collapsed | 0.0907 | -0.0000 | 0.25 | 3/5 |
| token | 0.0989 | -0.0015 | 0.25 | 2/5 |
| token_collapsed | 0.1193 | -0.0022 | 0.5 | 2/5 |
| all | 0.1786 | -0.0007 | 0.1 | 1/5 |

| model | level | top cells (band @ position: ratio vs all-model mean) |
|---|---|---|
| claude-fable-5 | char | ’ @ 0%: x0.06; — @ 20%: x2.86; ' @ 0%: x2.62 |
| claude-opus-4-6 | char | ’ @ 0%: x0.06; : @ 70%: x0.16; : @ 10%: x0.36 |
| claude-opus-4-8 | char | ’ @ 0%: x0.06; : @ 15%: x0.12; . @ 35%: x0.51 |
| claude-opus-5 | char | ’ @ 0%: x0.06; — @ 85%: x3.37; — @ 15%: x2.74 |
| claude-sonnet-5 | char | ’ @ 0%: x0.06; ' @ 0%: x2.99; . @ 5%: x0.42 |
| gemini-3.1-pro-preview | char | — @ 15%: x0.07; — @ 85%: x0.07; — @ 80%: x0.07 |
| gemini-3.6-flash | char | : @ 15%: x0.06; — @ 15%: x0.07; — @ 85%: x0.07 |
| gemini-3.7-flash | char | ' @ 0%: x0.05; — @ 15%: x0.07; — @ 85%: x0.07 |
| gemini-3.8-flash | char | : @ 10%: x0.04; : @ 15%: x0.06; — @ 15%: x0.07 |
| glm-5.3 | char | ’ @ 0%: x0.06; — @ 75%: x2.92; — @ 80%: x2.80 |
| glm-5.3-flash | char | ’ @ 0%: x0.06; — @ 10%: x2.93; — @ 30%: x2.92 |
| gpt-5.5 | char | — @ 15%: x0.07; — @ 10%: x0.08; 0-9 @ 5%: x0.08 |
| gpt-5.6-luna | char | ' @ 0%: x0.14; — @ 75%: x0.18; — @ 20%: x0.39 |
| gpt-5.6-sol | char | ' @ 0%: x0.05; — @ 15%: x0.07; — @ 20%: x0.30 |
| gpt-5.6-terra | char | ' @ 0%: x0.05; — @ 20%: x0.07; — @ 85%: x0.15 |
| gpt-6-astra | char | — @ 20%: x0.07; — @ 75%: x0.09; — @ 30%: x0.09 |
| kimi-k3 | char | — @ 20%: x0.33; : @ 15%: x2.97; : @ 5%: x2.07 |
| qwen3.8-max | char | — @ 85%: x0.07; — @ 70%: x0.08; — @ 90%: x0.09 |
| claude-fable-5 | word | 1 @ 95%: x0.58; 8 @ 95%: x0.58; 11-13 @ 45%: x0.63 |
| claude-opus-4-6 | word | 1 @ 90%: x0.53; 1 @ 95%: x0.56; 11-13 @ 55%: x1.50 |
| claude-opus-4-8 | word | 1 @ 95%: x0.51; 1 @ 50%: x0.55; 1 @ 55%: x0.59 |
| claude-opus-5 | word | 11-13 @ 60%: x0.53; 1 @ 60%: x1.84; 1 @ 20%: x1.71 |
| claude-sonnet-5 | word | 1 @ 45%: x0.36; 11-13 @ 50%: x1.60; 11-13 @ 85%: x1.57 |
| gemini-3.1-pro-preview | word | 5 @ 10%: x0.68; 4 @ 25%: x0.68; 11-13 @ 45%: x0.69 |
| gemini-3.6-flash | word | 11-13 @ 25%: x1.63; 5 @ 90%: x0.66; 11-13 @ 55%: x1.48 |
| gemini-3.7-flash | word | 1 @ 95%: x0.44; 1 @ 55%: x0.47; 1 @ 70%: x0.47 |
| gemini-3.8-flash | word | 1 @ 75%: x0.29; 1 @ 45%: x0.42; 1 @ 10%: x0.42 |
| glm-5.3 | word | 1 @ 15%: x0.47; 1 @ 85%: x0.48; 11-13 @ 50%: x0.55 |
| glm-5.3-flash | word | 1 @ 65%: x0.43; 1 @ 95%: x1.90; 11-13 @ 5%: x0.60 |
| gpt-5.5 | word | 11-13 @ 20%: x0.46; 11-13 @ 45%: x0.54; 11-13 @ 60%: x0.59 |
| gpt-5.6-luna | word | 11-13 @ 85%: x1.58; 1 @ 20%: x1.45; 1 @ 65%: x1.45 |
| gpt-5.6-sol | word | 11-13 @ 25%: x0.63; 1 @ 15%: x0.65; 2 @ 65%: x0.66 |
| gpt-5.6-terra | word | 11-13 @ 25%: x0.51; 1 @ 35%: x0.62; 1 @ 85%: x1.47 |
| gpt-6-astra | word | 11-13 @ 25%: x0.38; 11-13 @ 70%: x0.55; 11-13 @ 60%: x0.59 |
| kimi-k3 | word | 11-13 @ 95%: x0.45; 11-13 @ 60%: x0.58; 11-13 @ 70%: x0.61 |
| qwen3.8-max | word | 11-13 @ 85%: x0.55; 11-13 @ 35%: x0.56; 11-13 @ 75%: x0.62 |
| claude-fable-5 | sentence | 41+ @ sent 6: x0.00; 6-10 @ sent 8: x0.00; 6-10 @ sent 7: x0.00 |
| claude-opus-4-6 | sentence | 41+ @ sent 8: x0.00; 6-10 @ sent 6: x0.00; 6-10 @ sent 7: x0.00 |
| claude-opus-4-8 | sentence | 41+ @ sent 8: x0.00; 41+ @ sent 7: x0.00; 1-5 @ sent 7: x0.01 |
| claude-opus-5 | sentence | 1-5 @ sent 2: x6.15; 1-5 @ sent 7: x5.78; 1-5 @ sent 3: x5.69 |
| claude-sonnet-5 | sentence | 11-15 @ sent 8: x0.00; 6-10 @ sent 8: x0.00; 41+ @ sent 4: x0.00 |
| gemini-3.1-pro-preview | sentence | 41+ @ sent 7: x0.00; 41+ @ sent 1: x0.00; 41+ @ sent 2: x0.00 |
| gemini-3.6-flash | sentence | 11-15 @ sent 8: x0.00; 6-10 @ sent 4: x0.00; 6-10 @ sent 6: x0.00 |
| gemini-3.7-flash | sentence | 21-25 @ sent 7: x0.00; 11-15 @ sent 8: x0.00; 6-10 @ sent 4: x0.00 |
| gemini-3.8-flash | sentence | 11-15 @ sent 7: x0.00; 6-10 @ sent 5: x0.00; 6-10 @ sent 6: x0.00 |
| glm-5.3 | sentence | 41+ @ sent 5: x0.00; 1-5 @ sent 7: x0.01; 1-5 @ sent 8: x0.01 |
| glm-5.3-flash | sentence | 41+ @ sent 3: x0.00; 1-5 @ sent 2: x0.01; 6-10 @ sent 1: x2.67 |
| gpt-5.5 | sentence | 11-15 @ sent 6: x0.00; 41+ @ sent 7: x0.00; 6-10 @ sent 8: x0.00 |
| gpt-5.6-luna | sentence | 41+ @ sent 8: x0.00; 41+ @ sent 7: x0.00; 1-5 @ sent 4: x0.01 |
| gpt-5.6-sol | sentence | 6-10 @ sent 8: x0.00; 6-10 @ sent 7: x0.00; 1-5 @ sent 7: x0.01 |
| gpt-5.6-terra | sentence | 11-15 @ sent 8: x0.00; 6-10 @ sent 6: x0.00; 6-10 @ sent 8: x0.00 |
| gpt-6-astra | sentence | 41+ @ sent 7: x0.00; 41+ @ sent 2: x0.00; 41+ @ sent 5: x0.00 |
| kimi-k3 | sentence | 41+ @ sent 6: x0.00; 6-10 @ sent 3: x0.00; 1-5 @ sent 7: x0.01 |
| qwen3.8-max | sentence | 41+ @ sent 6: x0.00; 41+ @ sent 5: x0.00; 1-5 @ sent 7: x0.01 |
| claude-fable-5 | token | 10k+ @ 0%: x0.68; 10k+ @ 80%: x0.73; 10k+ @ 90%: x0.75 |
| claude-opus-4-6 | token | 10k+ @ 5%: x1.37; 10k+ @ 55%: x1.32; 11-30 @ 20%: x0.77 |
| claude-opus-4-8 | token | 10k+ @ 0%: x0.48; 10k+ @ 35%: x0.60; 10k+ @ 70%: x0.64 |
| claude-opus-5 | token | 10k+ @ 5%: x1.53; 10k+ @ 10%: x1.46; 10k+ @ 45%: x1.39 |
| claude-sonnet-5 | token | 10k+ @ 0%: x0.64; 10k+ @ 60%: x0.68; 10k+ @ 20%: x1.37 |
| gemini-3.1-pro-preview | token | 11-30 @ 45%: x0.58; 10k+ @ 55%: x0.61; 10k+ @ 45%: x0.63 |
| gemini-3.6-flash | token | 11-30 @ 15%: x0.51; 11-30 @ 55%: x0.62; 11-30 @ 70%: x0.65 |
| gemini-3.7-flash | token | 11-30 @ 65%: x0.61; 11-30 @ 40%: x0.65; 10k+ @ 0%: x1.53 |
| gemini-3.8-flash | token | 11-30 @ 95%: x0.52; 101-300 @ 10%: x0.64; 10k+ @ 0%: x1.55 |
| glm-5.3 | token | 10k+ @ 0%: x1.77; 11-30 @ 75%: x1.77; 10k+ @ 70%: x1.56 |
| glm-5.3-flash | token | 11-30 @ 35%: x1.38; 31-100 @ 90%: x1.33; 10k+ @ 5%: x0.76 |
| gpt-5.5 | token | 10k+ @ 50%: x0.67; 10k+ @ 0%: x0.67; 10k+ @ 60%: x0.69 |
| gpt-5.6-luna | token | 10k+ @ 20%: x0.61; 10k+ @ 45%: x0.61; 10k+ @ 10%: x0.62 |
| gpt-5.6-sol | token | 10k+ @ 10%: x0.35; 10k+ @ 5%: x0.47; 10k+ @ 40%: x0.70 |
| gpt-5.6-terra | token | 10k+ @ 20%: x0.64; 10k+ @ 95%: x1.35; 11-30 @ 35%: x1.34 |
| gpt-6-astra | token | 10k+ @ 5%: x0.28; 10k+ @ 0%: x0.37; 10k+ @ 35%: x0.38 |
| kimi-k3 | token | 10k+ @ 60%: x1.64; 10k+ @ 90%: x1.59; 10k+ @ 40%: x1.51 |
| qwen3.8-max | token | 10k+ @ 5%: x0.56; 10k+ @ 60%: x0.65; 10k+ @ 75%: x0.67 |
