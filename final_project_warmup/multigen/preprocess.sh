#!bin/bash

set -e

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
DATA=out_of_domain

cd $SCRIPT_DIR/preprocess

echo "***** Ground Concepts Simple *****"
python3.8 ground_concepts_simple.py $DATA

echo "***** Find Neightbours *****"
python3.8 find_neighbours.py $DATA

echo "***** Filter Triple *****"
python3.8 filter_triple.py $DATA
