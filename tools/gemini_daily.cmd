@echo off
rem Daily Gemini AI Studio free-tier collection (gemini-3.6-flash, gemini-3.8-flash; 20 requests/day each).
rem Quota resets at midnight Pacific (08:00 UK summer time). Output: research\ext_data\gemini_api\responses.csv
cd /d I:\venv_python\that_llm
echo ==== %date% %time% >> research\ext_data\gemini_api\daily.log
.venv\Scripts\python.exe -u tools\gemini_collect.py --models gemini-3.6-flash gemini-3.8-flash >> research\ext_data\gemini_api\daily.log 2>&1
