#!/bin/bash
# Download all required data sources for the SF Intersection Safety Analysis
# Data Sources:
#   1. SF Open Data: Traffic Crashes Resulting in Injury (ubvf-ztfx)
#   2. SF Open Data: Traffic Signals (ybh5-27n2)
#   3. SF Open Data: Stop Signs (4542-gpa3)
#   4. SFMTA: Intersection Traffic Counts 2014-2022

set -e

DATA_DIR="${1:-data}"
mkdir -p "$DATA_DIR"

echo "Downloading SF intersection safety data into $DATA_DIR/ ..."
echo ""

echo "[1/4] Downloading crash data (SF Open Data: ubvf-ztfx) ..."
curl -L -o "$DATA_DIR/crashes.csv" \
  "https://data.sfgov.org/api/views/ubvf-ztfx/rows.csv?accessType=DOWNLOAD"
echo "  -> $(wc -l < "$DATA_DIR/crashes.csv") lines"

echo "[2/4] Downloading traffic signals (SF Open Data: ybh5-27n2) ..."
curl -L -o "$DATA_DIR/traffic_signals.csv" \
  "https://data.sfgov.org/api/views/ybh5-27n2/rows.csv?accessType=DOWNLOAD"
echo "  -> $(wc -l < "$DATA_DIR/traffic_signals.csv") lines"

echo "[3/4] Downloading stop signs (SF Open Data: 4542-gpa3) ..."
curl -L -o "$DATA_DIR/stop_signs.csv" \
  "https://data.sfgov.org/api/views/4542-gpa3/rows.csv?accessType=DOWNLOAD"
echo "  -> $(wc -l < "$DATA_DIR/stop_signs.csv") lines"

echo "[4/4] Downloading SFMTA intersection traffic counts ..."
curl -L -o "$DATA_DIR/intersection_counts.csv" \
  "https://www.sfmta.com/media/40634/download?inline"
echo "  -> $(wc -l < "$DATA_DIR/intersection_counts.csv") lines"

echo ""
echo "All data downloaded to $DATA_DIR/"
echo "Run: python3 analysis.py --data-dir $DATA_DIR"
