#!bin/bash

set -e

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

cd $SCRIPT_DIR/preprocess

echo "***** Extract Concept Net *****"
python3.8 extract_cpnet.py

echo "***** Graph Construction *****"
python3.8 graph_construction.py
