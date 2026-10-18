# Sibling signatures (prompt-paired, all train)

rate = mean relative frequency in the model's passages; vs sib = mean over its siblings; t = paired t over prompts;
cons = share of prompts where the difference has the majority sign. Full tables: experiments/spectro/out/signatures_<family>.csv


## claude (5 models, 149 paired prompts)

| model | top signatures (feature: ratio vs siblings, t, consistency) |
|---|---|
| claude-fable-5 | `char:— ` 0.00x t=-8.3 c=1.00; `char: —` 0.00x t=-8.3 c=1.00; `char: — ` 0.00x t=-8.3 c=1.00; `char:s —` 0.01x t=-5.3 c=1.00; `char:— t` 0.00x t=-5.0 c=1.00; `char:evi` 0.35x t=-4.8 c=0.96; `char:— a` 0.01x t=-4.7 c=1.00; `char:en` 0.88x t=-4.3 c=0.68 |
| claude-opus-4-6 | `spec_char:a-z` 1.01x t=+10.7 c=0.87; `char:, ` 0.81x t=-8.9 c=0.78; `spec_char:,` 0.82x t=-8.8 c=0.75; `char:,` 0.82x t=-8.8 c=0.77; `char:. ` 0.82x t=-7.7 c=0.76; `func:,` 0.86x t=-7.3 c=0.72; `char:al ` 1.59x t=+7.1 c=0.72; `spec_word:11-13` 1.33x t=+7.1 c=0.72 |
| claude-opus-4-8 | `char:— ` 0.00x t=-8.3 c=1.00; `char: — ` 0.00x t=-8.3 c=1.00; `char: —` 0.00x t=-8.3 c=1.00; `char:s —` 0.01x t=-5.3 c=1.00; `word:—` 0.68x t=-5.2 c=0.69; `char:— t` 0.00x t=-5.0 c=1.00; `char:—` 0.69x t=-5.0 c=0.68; `spec_char:—` 0.69x t=-4.9 c=0.68 |
| claude-opus-5 | `spec_char:a-z` 0.98x t=-16.7 c=0.92; `char: ` 1.08x t=+15.7 c=0.87; `spec_char:space` 1.07x t=+15.3 c=0.86; `func:CONTENT` 0.95x t=-10.8 c=0.79; `spec_word:9-10` 0.74x t=-10.6 c=0.82; `func:CONTENT CONTENT` 0.85x t=-9.9 c=0.80; `spec_token:1-10` 1.11x t=+9.7 c=0.78; `spec_word:11-13` 0.66x t=-9.5 c=0.80 |
| claude-sonnet-5 | `char: ` 0.95x t=-9.4 c=0.78; `spec_char:space` 0.95x t=-9.2 c=0.79; `spec_char:a-z` 1.01x t=+9.1 c=0.80; `char: —` 0.00x t=-8.3 c=1.00; `char: — ` 0.00x t=-8.3 c=1.00; `char:— ` 0.00x t=-8.3 c=1.00; `func:CONTENT` 1.04x t=+7.9 c=0.70; `char:he ` 0.74x t=-7.5 c=0.77 |

## gemini (4 models, 150 paired prompts)

| model | top signatures (feature: ratio vs siblings, t, consistency) |
|---|---|
| gemini-3.1-pro-preview | `spec_char:a-z` 0.99x t=-15.0 c=0.90; `char: ` 1.05x t=+13.1 c=0.86; `spec_char:space` 1.05x t=+12.9 c=0.86; `char:. ` 1.28x t=+10.4 c=0.85; `spec_token:1-10` 1.11x t=+9.5 c=0.79; `func:CONTENT` 0.95x t=-9.1 c=0.79; `char:.` 1.20x t=+8.1 c=0.77; `spec_char:.` 1.20x t=+8.1 c=0.77 |
| gemini-3.6-flash | `word:.` 0.90x t=-6.1 c=0.71; `char:. ` 0.91x t=-5.2 c=0.63; `func:.` 0.92x t=-4.8 c=0.63; `func:CONTENT .` 0.92x t=-4.6 c=0.65; `char: a ` 1.26x t=+4.4 c=0.63; `func:a` 1.24x t=+4.3 c=0.62; `func:a CONTENT` 1.24x t=+4.3 c=0.62; `spec_char:.` 0.93x t=-4.3 c=0.63 |
| gemini-3.7-flash | `spec_char:a-z` 1.01x t=+9.8 c=0.78; `char: ` 0.97x t=-8.6 c=0.77; `spec_char:space` 0.97x t=-8.5 c=0.78; `char:. ` 0.89x t=-6.8 c=0.72; `func:a CONTENT CONTENT` 0.67x t=-6.8 c=0.76; `func:a` 0.74x t=-6.3 c=0.74; `spec_char:A-Z` 0.88x t=-6.3 c=0.73; `func:a CONTENT` 0.74x t=-6.2 c=0.74 |
| gemini-3.8-flash | `char: a ` 0.64x t=-8.0 c=0.81; `char:a ` 0.67x t=-7.9 c=0.78; `char: ` 0.97x t=-7.9 c=0.71; `spec_char:space` 0.97x t=-7.8 c=0.71; `func:a` 0.68x t=-7.4 c=0.76; `func:a CONTENT` 0.68x t=-7.4 c=0.76; `spec_word:1` 0.69x t=-6.9 c=0.75; `spec_token:1-10` 0.93x t=-6.8 c=0.73 |

## glm (2 models, 150 paired prompts)

| model | top signatures (feature: ratio vs siblings, t, consistency) |
|---|---|
| glm-5.3 | `char:nds` 0.52x t=-3.5 c=0.61; `char:die` 3.06x t=+3.4 c=0.79; `char:l c` 0.46x t=-3.2 c=0.72; `char: pu` 2.01x t=+3.2 c=0.72; `char:f.` 3.10x t=+3.1 c=0.75; `char:m ` 1.31x t=+3.1 c=0.63; `char:uci` 0.06x t=-3.0 c=0.92; `char:het` 0.39x t=-3.0 c=0.78 |
| glm-5.3-flash | `char:nds` 1.93x t=+3.5 c=0.61; `char:die` 0.33x t=-3.4 c=0.79; `char:l c` 2.16x t=+3.2 c=0.72; `char: pu` 0.50x t=-3.2 c=0.72; `char:f.` 0.32x t=-3.1 c=0.75; `char:m ` 0.76x t=-3.1 c=0.63; `char:uci` 17.42x t=+3.0 c=0.92; `char:het` 2.54x t=+3.0 c=0.78 |

## gpt (5 models, 150 paired prompts)

| model | top signatures (feature: ratio vs siblings, t, consistency) |
|---|---|
| gpt-5.5 | `spec_char:a-z` 0.99x t=-8.8 c=0.79; `char: ` 1.03x t=+7.7 c=0.75; `char:, ` 1.18x t=+7.7 c=0.69; `char:,` 1.17x t=+7.5 c=0.69; `spec_char:,` 1.17x t=+7.5 c=0.69; `spec_char:space` 1.03x t=+7.4 c=0.76; `func:,` 1.12x t=+6.3 c=0.67; `func:CONTENT ,` 1.13x t=+6.1 c=0.67 |
| gpt-5.6-luna | `spec_char:space` 0.97x t=-7.7 c=0.71; `char: ` 0.97x t=-7.4 c=0.71; `spec_word:11-13` 1.37x t=+6.4 c=0.67; `char:tio` 1.38x t=+6.2 c=0.66; `char:io` 1.31x t=+5.9 c=0.66; `char:c` 1.10x t=+5.9 c=0.67; `char:ion` 1.31x t=+5.4 c=0.66; `spec_word:9-10` 1.17x t=+5.1 c=0.68 |
| gpt-5.6-sol | `char:a ` 0.79x t=-5.6 c=0.71; `char: a ` 0.80x t=-5.0 c=0.67; `func:'` 0.00x t=-4.9 c=1.00; `func:' CONTENT` 0.00x t=-4.9 c=1.00; `word:'` 0.00x t=-4.9 c=1.00; `func:CONTENT ' CONTENT` 0.00x t=-4.9 c=1.00; `func:CONTENT '` 0.00x t=-4.9 c=1.00; `char:'` 0.00x t=-4.8 c=1.00 |
| gpt-5.6-terra | `func:.` 0.91x t=-6.7 c=0.69; `spec_char:.` 0.91x t=-6.7 c=0.72; `char:.` 0.91x t=-6.6 c=0.71; `word:.` 0.92x t=-6.2 c=0.70; `char:. ` 0.89x t=-5.3 c=0.69; `char:. A` 0.54x t=-4.4 c=0.78; `func:CONTENT , CONTENT` 1.15x t=+4.4 c=0.63; `char:f n` 0.01x t=-4.4 c=1.00 |
| gpt-6-astra | `word:,` 0.76x t=-13.5 c=0.87; `func:.` 1.20x t=+12.8 c=0.87; `char:,` 0.77x t=-12.5 c=0.87; `char:, ` 0.77x t=-12.5 c=0.87; `spec_char:,` 0.77x t=-12.4 c=0.87; `func:,` 0.78x t=-12.3 c=0.84; `func:CONTENT ,` 0.77x t=-12.2 c=0.85; `spec_char:.` 1.18x t=+11.4 c=0.85 |
