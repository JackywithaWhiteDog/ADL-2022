#!bin/bash

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

DEVICE=6
ROOT_PATH=.
DATA_TYPE=out_of_domain
PRE_TRAINED=anlg
PRE_TRAINED_DIR=anlg/grf-anlg_gpt2-small

CUDA_VISIBLE_DEVICES=${DEVICE} \
python3 ${SCRIPT_DIR}/scripts/main.py \
--train_data_file ${ROOT_PATH}/data/${DATA_TYPE}/train \
--dev_data_file ${ROOT_PATH}/data/${DATA_TYPE}/dev \
--test_data_file ${ROOT_PATH}/data/${DATA_TYPE}/test \
--graph_path 2hops_100_directed_triple_filter.json \
--output_dir ${ROOT_PATH}/models/${DATA_TYPE}/grf-${DATA_TYPE}_${PRE_TRAINED} \
--source_length 50 \
--target_length 75 \
--model_type gpt2 \
--model_name_or_path ${ROOT_PATH}/models/${PRE_TRAINED_DIR} \
--do_train \
--per_gpu_train_batch_size 8 \
--gradient_accumulation_steps 2 \
--per_gpu_eval_batch_size 16 \
--workers 7 \
--seed 1123 \
--evaluate_metrics bleu \
--overwrite_output_dir \
--num_train_epochs 20 \
--learning_rate 5e-5 \
--aggregate_method max \
--alpha 3 \
--beta 5 \
--gamma 0.5 \
--weight_decay 0.0 \
--warmup_ratio 0.0 \
--logging_steps 20 \
