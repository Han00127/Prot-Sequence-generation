#!/bin/bash

## set data path if run download_preprocessed_datasets.sh in data directory, don't need to modify here. 
cath_file='./data/chain_set.jsonl'
cath_splits='./data/chain_set_splits.json'
short_splits='./data/test_split_L100.json'
single_splits='./data/test_split_sc.json'
ts_dir='./data/ts/'

# #################################################
# ## set models - LM-Design1 test on CATH 4.2 and TS
# #################################################
save_dir='/data/private/LMDESIGN/test/lmdesign_exp1_results'
saved_weight='/data/project/rw/LMDesign_weights/lmdesign1_trained_weight.pt' # put path for saved weights
python lmdesign_test.py --use_pretrained_weights $saved_weight --out_folder $save_dir --embed_dim 1280 --num_heads 10 --structure_model MPNN  --structure_weight "" \
    --jsonl_path $cath_file --file_splits $cath_splits --test_short_path $short_splits --test_single_path $single_splits --test_ts_directory $ts_dir --num_refinement 6


# #################################################
# ## set models - LM-Design2 test on CATH 4.2 and TS
# #################################################
save_dir2='/data/private/LMDESIGN/test/lmdesign_exp2_results'
saved_weight2='/data/project/rw/LMDesign_weights/lmdesign2_trained_weight.pt' # put path for saved weights
python lmdesign_test.py --use_pretrained_weights $saved_weight2 --out_folder $save_dir2 --embed_dim 1280 --num_heads 10 --structure_model MPNN --structure_weight "" --num_decoder_layers 4 \
    --jsonl_path $cath_file --file_splits $cath_splits --test_short_path $short_splits --test_single_path $single_splits --test_ts_directory $ts_dir --num_refinement 6

# #################################################
# ## set models - LM-Design3 test on CATH 4.2 and TS
# #################################################
save_dir3='/data/private/LMDESIGN/test/lmdesign_exp3_results'
saved_weight3='/data/project/rw/LMDesign_weights/lmdesign3_trained_weight.pt' # put path for saved weights
python lmdesign_test.py --use_pretrained_weights $saved_weight3 --out_folder $save_dir3 --embed_dim 1280 --num_heads 10 --structure_model MPNN --structure_weight "" --num_decoder_layers 4 \
    --jsonl_path $cath_file --file_splits $cath_splits --test_short_path $short_splits --test_single_path $single_splits --test_ts_directory $ts_dir --num_refinement 6
