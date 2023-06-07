#!/bin/bash
source /opt/conda/bin/activate base
save_dir='/data/private/LMDESIGN/test/lmdesign_exp1_results'
save_dir2='/data/private/LMDESIGN/test/lmdesign_exp2_results'
save_dir3='/data/private/LMDESIGN/test/lmdesign_exp3_results'

saved_weight='/data/project/rw/lmdesign_results/lmdesign2_full_encode/model_weights/epoch100.pt' # put path for saved weights
saved_weight2='/data/project/rw/lmdesign_results/lmdesign_exp3/model_weights/epoch100.pt' # put path for saved weights
saved_weight3='/data/project/rw/lmdesign_results/lmdesign_exp4/model_weights/epoch100.pt' # put path for saved weights

cath_file='/data/project/rw/cath4.2/chain_set.jsonl'
cath_splits='/data/project/rw/cath4.2/chain_set_splits.json'
short_splits='/data/project/rw/cath4.2/test_split_L100.json'
single_splits='/data/project/rw/cath4.2/test_split_sc.json'
ts_dir='/data/project/rw/ts/'

# Test LMDESIGN exp1 model on CATH4.2  and TS50 & TS500  
python lmdesign_test.py --use_pretrained_weights $saved_weight --out_folder $save_dir --embed_dim 1280 --num_heads 10 --structure_model MPNN  --structure_weight "" \
    --jsonl_path $cath_file --file_splits $cath_splits --test_short_path $short_splits --test_single_path $single_splits --test_ts_directory $ts_dir

# Training LMDESIGN on CATH4.2 exp2
python lmdesign_test.py --use_pretrained_weights $saved_weight2 --out_folder $save_dir2 --embed_dim 1280 --num_heads 10 --structure_model MPNN --structure_weight "" --num_decoder_layers 4 \
    --jsonl_path $cath_file --file_splits $cath_splits --test_short_path $short_splits --test_single_path $single_splits --test_ts_directory $ts_dir

# Training LMDESIGN on CATH4.2 exp3
python lmdesign_test.py --use_pretrained_weights $saved_weight3 --out_folder $save_dir3 --embed_dim 1280 --num_heads 10 --structure_model MPNN --structure_weight "" --num_decoder_layers 4 \
    --jsonl_path $cath_file --file_splits $cath_splits --test_short_path $short_splits --test_single_path $single_splits --test_ts_directory $ts_dir