$ErrorActionPreference = 'Stop'
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m trading.cli generate-demo
& .\.venv\Scripts\python.exe -m trading.cli backtest
& .\.venv\Scripts\python.exe -m pytest -q
Write-Host 'Demo READY: run .\.venv\Scripts\python.exe -m trading.cli train-xgb'
