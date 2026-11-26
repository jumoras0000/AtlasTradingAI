#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-rl.txt
python -m trading.cli generate-demo
python -m trading.cli train-ppo --steps 15000
python -m pytest -q
