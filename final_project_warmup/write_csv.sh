#!bin/bash

set -e

cd ./data/in_domain

python3.8 ../../write_csv.py

cd ../out_of_domain

python3.8 ../../write_csv.py
