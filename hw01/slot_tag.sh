#!bin/bash

python slot_test.py \
    --test_file "${1}" \
    --pred_file "${2}" \
    --ckpt_path ckpt/slot/20220303_1645/slot-epoch=12-val_acc=0.97-val_loss=0.12.ckpt \
    --device cuda:7
