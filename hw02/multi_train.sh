#!bin/bash

# export CUDA_VISIBLE_DEVICES="4"

python3.8 ./multi_train.py \
    --model_name_or_path hfl/chinese-roberta-wwm-ext \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --per_device_eval_batch_size 16 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --output_dir ./ckpt/multiple_choice/roberta \
    --overwrite_output \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_strategy epoch \
    --save_total_limit 1 \
    --cache_dir ./cache \
    --train_file ./data/train.json \
    --validation_file ./data/valid.json \
    --context_file ./data/context.json

# python3.8 ./multi_train.py \
#     --model_name_or_path hfl/chinese-roberta-wwm-ext \
#     --no_pretrained \
#     --do_train \
#     --do_eval \
#     --max_seq_length 512 \
#     --learning_rate 5e-5 \
#     --num_train_epochs 3 \
#     --per_device_eval_batch_size 16 \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 8 \
#     --output_dir ./ckpt/multiple_choice/roberta_no_pretrained \
#     --overwrite_output \
#     --evaluation_strategy steps \
#     --eval_steps 500 \
#     --save_strategy epoch \
#     --save_total_limit 1 \
#     --cache_dir ./cache \
#     --train_file ./data/train.json \
#     --validation_file ./data/valid.json \
#     --context_file ./data/context.json

# python3.8 ./multi_train.py \
#     --model_name_or_path ckiplab/albert-tiny-chinese \
#     --do_train \
#     --do_eval \
#     --max_seq_length 512 \
#     --learning_rate 5e-5 \
#     --num_train_epochs 3 \
#     --output_dir ./ckpt/multiple_choice/albert-tiny \
#     --per_device_eval_batch_size 16 \
#     --per_device_train_batch_size 8 \
#     --gradient_accumulation_steps 2 \
#     --overwrite_output \
#     --evaluation_strategy steps \
#     --eval_steps 500 \
#     --save_strategy epoch \
#     --save_total_limit 1 \
#     --cache_dir ./cache \
#     --train_file ./data/train.json \
#     --validation_file ./data/valid.json \
#     --context_file ./data/context.json
