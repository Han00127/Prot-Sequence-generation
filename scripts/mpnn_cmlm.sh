#!/bin/bash
source /opt/conda/bin/activate base
out_folder='./result/mpnn_cmlm_reproduce'
cathDataPath='./data/chain_set.jsonl'
splits='./data/chain_set_splits.json'
shortPath='./data/test_split_L100.json'
scPath='./data/test_split_sc.json'

# ## Training from pretrained model vanilla ProtMPNN model - v_48_020.pt)) encoder decoder 3 / 3 as same as official model
# python mpnn_train.py --model MPNN_CMLM --epoch 100 --out_folder $out_folder --jsonl_path $cathDataPath --file_splits $splits --test_short_path $shortPath --test_single_path $scPath

## Prev model trained with 4 decoder layers training from scratch
python mpnn_train.py --model MPNN_CMLM --epoch 100 --out_folder $out_folder --jsonl_path $cathDataPath --file_splits $splits --test_short_path $shortPath --test_single_path $scPath --use_pretrained_weights "" --num_decoder_layers 4 
