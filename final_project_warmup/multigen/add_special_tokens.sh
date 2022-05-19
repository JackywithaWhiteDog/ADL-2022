#!bin/bash

set -e

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

cd $SCRIPT_DIR/scripts

echo "***** Add Special Tokens *****"
python3.8 add_special_tokens.py
