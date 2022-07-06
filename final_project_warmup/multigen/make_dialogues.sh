#!bin/bash

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

DATA_TYPE=out_of_domain
SPLIT=test
PRE_TRAINED=anlg

python3.8 ${SCRIPT_DIR}/make_dialogues.py \
    ./data/${DATA_TYPE}/test/source.csv \
    ./models/${DATA_TYPE}/grf-${DATA_TYPE}_${PRE_TRAINED}_eval/result_ep:test.txt \
    ./models/${DATA_TYPE}/grf-${DATA_TYPE}_${PRE_TRAINED}_eval/dialogues.txt
