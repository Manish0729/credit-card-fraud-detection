#!/usr/bin/env bash
set -euo pipefail

THR="${1:-0.77}"

clear
python run_real_fraud_detection.py --threshold "$THR"
python threshold_tuning.py
mkdir -p visualizations/final
mv -f visualizations/report_*.png visualizations/final/ || true
open visualizations/final/*.png