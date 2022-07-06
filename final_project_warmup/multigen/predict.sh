#!bin/bash

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

DEVICE=5
ROOT_PATH=.
DATA_TYPE=in_domain
PRE_TRAINED=anlg

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
--model_name_or_path ${ROOT_PATH}/models/${DATA_TYPE}/grf-${DATA_TYPE}_${PRE_TRAINED} \
--do_eval \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--workers 7 \
--seed 1123 \
--evaluate_metrics bleu \
--overwrite_output_dir \
--aggregate_method max \
--gamma 0.5 \
--sampling \
--sampling_topk 2
