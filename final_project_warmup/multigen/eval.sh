#!bin/bash

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

DATA=anlg

python ${SCRIPT_DIR}/evaluation/eval.py --dataset ${DATA} --output_dir ./models/${DATA}/grf-${DATA}_eval/result_ep:test.txt
