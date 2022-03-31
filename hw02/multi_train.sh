#!bin/bash

export CUDA_VISIBLE_DEVICES="1"

python ./multi_train.py \
    --model_name_or_path ckiplab/albert-tiny-chinese \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --output_dir ./ckpt/multiple_choice/sample \
    --per_device_eval_batch_size 16 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --overwrite_output \
    --save_steps 1000 \
    --cache_dir ./cache \
    --train_file ./data/train.json \
    --validation_file ./data/valid.json \
    --context_file ./data/context.json
