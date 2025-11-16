#!/usr/bin/env bash
set -euo pipefail

# Directory of this script (the "code" folder)
ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

# Python executable (can override with: PYTHON=/opt/anaconda3/envs/carbon_ok/bin/python bash run_all.sh)
PY="${PYTHON:-python}"

# Config files directory
CFG_DIR="$ROOT_DIR/configs"

# Regions to run
regions=(ercot nyiso pjm)

# Run from the code root so that "src" is on the Python path
cd "$ROOT_DIR"

for r in "${regions[@]}"; do
  cfg="$CFG_DIR/$r.json"
  echo "=============================="
  echo "🚀 Running TFT for region: $r"
  echo "📄 Config: $cfg"
  echo "=============================="
  "$PY" -m src.train_tft --config "$cfg"
  echo "✅ Done: $r"
  echo
done

echo "🎉 All regions finished."