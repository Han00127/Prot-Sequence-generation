#!/bin/bash
out_folder='./results/lmdesign1_reproduce'
cathDataPath='./data/chain_set.jsonl'
splits='./data/chain_set_splits.json'
shortPath='./data/test_split_L100.json'
scPath='./data/test_split_sc.json'
pretrained_weights='/data/project/rw/LMDesign_weights/mpnn_cmlm_trained_weight.pt' # if you can access to braincloud project dir, use this for pretrained MPNN+CMLM

####################################
## Single implmenation 
#####################################
## Single implementation if make separated shell script for each experiment like lmdesign1.sh lmdesign2.sh lmdesign3.sh with given scripts
# Training LMDESIGN on CATH4.2 exp1 
python lmdesign_train.py --epoch 1 --out_folder $out_folder --jsonl_path $cathDataPath --file_splits $splits --test_short_path $shortPath --test_single_path $scPath \
                         --embed_dim 1280 --num_heads 10 --structure_model MPNN --structure_trainable True --num_decoder_layers 4  --structure_weight ""

# Training LMDESIGN on CATH4.2 exp2
python lmdesign_train.py --epoch 1 --out_folder $out_folder --jsonl_path $cathDataPath --file_splits $splits --test_short_path $shortPath --test_single_path $scPath \
                        --embed_dim 1280 --num_heads 10 --structure_model MPNN --structure_trainable True --structure_weight $pretrained_weights --num_decoder_layers 4

# Training LMDESIGN on CATH4.2 exp3
python lmdesign_train.py --epoch 10 --out_folder$out_folder --jsonl_path $cathDataPath --file_splits $splits --test_short_path $shortPath --test_single_path $scPath \
                         --embed_dim 1280 --num_heads 10 --structure_model MPNN --structure_trainable False --structure_weight $pretrained_weights --num_decoder_layers 4

#######################################
## V100 32G -> exp1 & exp2 are possibly parallely executed -> This makes a bit faster results.
########################################
cathDataPath='./data/chain_set.jsonl'
splits='./data/chain_set_splits.json'
shortPath='./data/test_split_L100.json'
scPath='./data/test_split_sc.json'
out_folder='./results/lmdesign1_reproduce'
out_folder2='./results/lmdesign2_reproduce'
pretrained_weights='/data/project/rw/LMDesign_weights/mpnn_cmlm_trained_weight.pt' # if you can access to braincloud project dir, use this for pretrained MPNN+CMLM
######################
## ESM model weights
## ESM model weight saved somewhere, please give the path for esm weights as argument e.g., --plm_weight /data/project/rw/ESM_weight/esm1b_t33_650M_UR50S.pt 
######################


# Training LMDESIGN on CATH4.2 exp1 
python lmdesign_train.py --epoch 1 --out_folder $out_folder --jsonl_path $cathDataPath --file_splits $splits --test_short_path $shortPath --test_single_path $scPath \
                         --embed_dim 1280 --num_heads 10 --structure_model MPNN --structure_trainable True --num_decoder_layers 4  --structure_weight "" &

# Training LMDESIGN on CATH4.2 exp2
python lmdesign_train.py --epoch 1 --out_folder $out_folder2 --jsonl_path $cathDataPath --file_splits $splits --test_short_path $shortPath --test_single_path $scPath \
                        --embed_dim 1280 --num_heads 10 --structure_model MPNN --structure_trainable True --structure_weight $pretrained_weights --num_decoder_layers 4

wait
