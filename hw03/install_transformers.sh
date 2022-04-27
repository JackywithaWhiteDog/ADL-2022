#!bin/bash

git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout t5-fp16-no-nans
pip install -e .
