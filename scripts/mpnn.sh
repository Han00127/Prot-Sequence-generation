#!/bin/bash
source /opt/conda/bin/activate base
out_folder='./results/mpnn_reproduce'
cathDataPath='./data/chain_set.jsonl'
splits='./data/chain_set_splits.json'
shortPath='./data/test_split_L100.json'
scPath='./data/test_split_sc.json'

python mpnn_train.py --model MPNN --epoch 100 --out_folder $out_folder --jsonl_path $cathDataPath --file_splits $splits --test_short_path $shortPath --test_single_path $scPath
