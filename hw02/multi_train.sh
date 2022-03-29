#!bin/bash

export CUDA_VISIBLE_DEVICES="5"

python ./multi_train.py \
    --pretrained_model hfl/chinese-roberta-wwm-ext \
    --max_seq_length 512
