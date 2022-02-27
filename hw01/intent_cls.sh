#!bin/bash

python intent_test.py \
    --test_file "${1}" \
    --pred_file "${2}" \
    --ckpt_path ckpt/intent/20220226_2338/intent-epoch=12-val_acc=0.92-val_loss=0.91.ckpt \
    --device cuda:7
