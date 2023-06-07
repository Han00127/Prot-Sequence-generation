#!/bin/bash
source /opt/conda/bin/activate base
save_dir='/data/private/LMDESIGN/test/mpnn_results'
save_dir2='/data/private/LMDESIGN/test/mpnn_cmlm_results'
saved_weight='/data/project/rw/mpnn_results/reproduce2/model_weights/epoch100.pt'
saved_weight2='/data/project/rw/lmdesign_results/mpnn_cmlm/model_weights/epoch100.pt'

cath_file='/data/project/rw/cath4.2/chain_set.jsonl'
cath_splits='/data/project/rw/cath4.2/chain_set_splits.json'
short_splits='/data/project/rw/cath4.2/test_split_L100.json'
single_splits='/data/project/rw/cath4.2/test_split_sc.json'
ts_dir='/data/project/rw/ts/'

# Test trained model on CATH4.2 and TS50 & 500, P1 == ProteinMPNN P2 == ProteinMPNN CMLM 
python mpnn_test.py --use_pretrained_weights $saved_weight --out_folder $save_dir --jsonl_path $cath_file --file_splits $cath_splits --test_short_path $short_splits --test_single_path $single_splits --test_ts_directory $ts_dir & 
python mpnn_test.py --use_pretrained_weights $saved_weight2 --num_decoder_layers 4 --num_neighbors 64 --out_folder $save_dir2 --jsonl_path $cath_file --file_splits $cath_splits --test_short_path $short_splits --test_single_path $single_splits --test_ts_directory $ts_dir

wait