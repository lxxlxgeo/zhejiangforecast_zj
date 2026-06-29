#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8000}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /home/lxce/miniconda3/etc/profile.d/conda.sh
conda activate ml_sc

cd "$PROJECT_DIR"
mkdir -p runtime_api
export PYTHONPATH="$PROJECT_DIR/src"
export ZJ_FORECAST_HOME="$PROJECT_DIR/runtime_api"

exec uvicorn zhejiangforecast_zj.api.main:app --host 0.0.0.0 --port "$PORT"

