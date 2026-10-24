# Own prompt set: sources and provenance

Collected by the user (manual browsing + a table-scraper browser extension), cleaned by `tools/prompts/clean_prompts.py`.
Raw files: `research/ext_data/own_prompts/raw/` (local only, git-ignored, never deleted). Output: `research/ext_data/own_prompts/prompts.csv`;
every raw item with its keep/drop reason: `clean_report.csv`. Only titles are used (post bodies, usernames and comments are ignored).

| Raw file (renamed copy) | Collected | Source | Raw titles | Kept |
|---|---|---|---|---|
| reddit_search_physics_2026-10-24.txt | 2026-10-24 | Reddit search "physics" (pasted page) | 126 | 46 |
| reddit_askphysics_scrape_2026-10-24.csv | 2026-10-24 | r/AskPhysics listing | 139 | 112 |
| reddit_philosophy_p1_2026-10-24.csv | 2026-10-24 | philosophy listing, part 1 | 281 | 220 |
| reddit_philosophy_p2_2026-10-24.csv | 2026-10-24 | philosophy listing, part 2 | 178 | 70 |
| quora_mathquestions_2026-10-24.csv | 2026-10-24 | Quora space "mathquestions" | 227 | 76 |
| reddit_history_p1/p2/p3_2026-10-25.csv | 2026-10-25 | history listings (r/AskHistorians style) | 1,269 | 1,101 |
| reddit_psychology_p1_2026-10-25.csv | 2026-10-25 | psychology listing | 389 | 365 |
| reddit_society_business_2026-10-25.csv | 2026-10-25 | AskReddit-style listing ("business.csv") | 920 | 904 |
| reddit_society_p2/p3_2026-10-25.csv | 2026-10-25 | society listings (p3: Bangladesh community) | 860 | 347 |
| reddit_everyday_p1_2026-10-25.csv | 2026-10-25 | everyday/society listing | 971 | 855 |
| reddit_tech_datasciencecareers_2026-10-25.csv | 2026-10-25 | r/datasciencecareers ("old (1).csv") | 391 | 153 |
| reddit_tech_askprogramming_2026-10-25.csv | 2026-10-25 | r/AskProgramming | 883 | 571 |
| reddit_economics_eli5_listing_2026-10-25.csv | 2026-10-25 | r/explainlikeimfive economics listing | 901 | 854 |
| stackexchange_economics_2026-10-25.csv | 2026-10-25 | Economics Stack Exchange: **pages 1-51, 50 questions per page** (user note); titles in the `s-link` column | 5,093 | 2,516 |

| reddit_cooking_askculinary_2026-10-25.csv | 2026-10-25 | r/AskCulinary ("old.csv") | 969 (both cooking files) | 471 (both) |
| reddit_cooking_askfoodcooking_2026-10-25.csv | 2026-10-25 | r/AskFoodCooking | | |

Total after cleaning (2026-10-25): **8,664 prompts** from 19 files. Prompt IDs are permanent (`own_` + md5 of the normalised
text); Benchmarks batches o001+ take the next unused IDs in the interleaved order (`tools/kbench/daily.py --own`).

Licensing for publication: Stack Exchange content is CC BY-SA 4.0 (publishable with attribution). Reddit and Quora terms limit
redistribution of their text: publish post URLs/IDs or paraphrases, not the scraped titles.
