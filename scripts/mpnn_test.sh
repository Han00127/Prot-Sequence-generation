#!/bin/bash
source /opt/conda/bin/activate base

## set data path if run download_preprocessed_datasets.sh in data directory, don't need to modify here. 
cath_file='./data/chain_set.jsonl'
cath_splits='./data/chain_set_splits.json'
short_splits='./data/test_split_L100.json'
single_splits='./data/test_split_sc.json'
ts_dir='./data/ts/'

# #################################################
# ## set models - ProtMPNN
# #################################################
save_dir='./result/mpnn_reproduce/test'
saved_weight='/data/project/rw/LMDesign_weights/mpnn_trained_weight.pt'

# Test trained model on CATH4.2 and TS50 & 500
python mpnn_test.py --use_pretrained_weights $saved_weight --out_folder $save_dir --jsonl_path $cath_file --file_splits $cath_splits --test_short_path $short_splits --test_single_path $single_splits --test_ts_directory $ts_dir 


#################################################
## set models - ProtMPNN+CMLM
#################################################
save_dir='./result/mpnn_cmlm_reproduce/test'
saved_weight='/data/project/rw/LMDesign_weights/mpnn_cmlm_trained_weight.pt' 

## This is model trained mpnn+cmlm with 4 decoder layer
## If you trained set num_encoder_layers and num_decoder_layers different, use them here. 
python mpnn_test.py --use_pretrained_weights $saved_weight --out_folder $save_dir --jsonl_path $cath_file --file_splits $cath_splits --test_short_path $short_splits --test_single_path $single_splits --test_ts_directory $ts_dir --num_decoder_layers 4

