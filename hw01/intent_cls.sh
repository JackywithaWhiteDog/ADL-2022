#!bin/bash

python intent_test.py \
    --test_file "${1}" \
    --pred_file "${2}" \
    --ckpt_path ckpt/intent/intent-best.ckpt \
    "${@:3}"
