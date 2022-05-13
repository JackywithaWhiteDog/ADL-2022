#!bin/bash

set -e

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

cd ./data/in_domain

python3.8 $SCRIPT_DIR/write_csv.py

cd ../out_of_domain

python3.8 $SCRIPT_DIR/write_csv.py
