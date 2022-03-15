#!bin/bash

python slot_test.py \
    --test_file "${1}" \
    --pred_file "${2}" \
    --ckpt_path ckpt/slot/slot-best.ckpt \
    --device cuda
