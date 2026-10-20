@echo off
setlocal
cd /d "%~dp0"
echo [1/3] Crawl-only labeled evaluation (50 documents)...
python evaluate_crawled_corpus.py evaluate --tokenizer underthesea
if errorlevel 1 goto :failed
echo [2/3] Known-positive diagnostic (887 documents)...
python evaluate_887_diagnostic.py --tokenizer underthesea
if errorlevel 1 goto :failed
echo [3/3] Refresh SEG301 report...
python complete_lab.py --out data/preprocessing --tokenizer underthesea
if errorlevel 1 goto :failed
echo.
echo SUCCESS - see data\preprocessing\crawl_eval\metric_summary.csv
pause
exit /b 0
:failed
echo.
echo FAILED. Check installed packages, CSVs, and underthesea. Existing data is not deleted.
pause
exit /b 1
