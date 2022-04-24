# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# export CUDA_VISIBLE_DEVICES="4"

python3.8 slot.py \
  --model_name_or_path ./ckpt/slot/checkpoint-2718 \
  --do_train \
  --do_eval \
  --do_predict \
  --overwrite_output_dir \
  --logging_steps 500 \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --save_strategy epoch \
  --save_total_limit 1 \
  --cache_dir ./cache \
  --train_file ./data/slot/train.json \
  --validation_file ./data/slot/eval.json \
  --test_file ./data/slot/test.json \
  --label_column_name tags \
  --output_dir ./ckpt/slot
