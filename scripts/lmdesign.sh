#!/bin/bash
# source /opt/conda/bin/activate base
# P1="python lmdesign_train.py --epoch 20 --out_folder /data/project/rw/codetest_results/lmdesign/lmdesign_exp1_reproduce/ --embed_dim 1280 --num_heads 10 --structure_model MPNN --structure_trainable True"
# P2="python lmdesign_train.py --epoch 20 --out_folder /data/project/rw/codetest_results/lmdesign/lmdesign_exp2_reproduce/ --embed_dim 1280 --num_heads 10 --structure_model MPNN --structure_trainable True --structure_weight /data/project/rw/lmdesign_results/mpnn_cmlm/model_weights/epoch100.pt --num_decoder_layers 4"
# # Training MPNN on CATH4.2
# $P1 & $P2
# wait 


# Training LMDESIGN on CATH4.2 exp1 /data/project/rw/lmdesign_results/mpnn_reproduce
# python lmdesign_train.py --epoch 20 --out_folder /data/project/rw/codetest_resultss/lmdesign/lmdesign_exp1_reproduce/ --embed_dim 1280 --num_heads 10 --structure_model MPNN --structure_trainable True --num_decoder_layers 4 & python lmdesign_train.py --epoch 20 --out_folder /data/project/rw/codetest_results/lmdesign/lmdesign_exp2_reproduce/ --embed_dim 1280 --num_heads 10 --structure_model MPNN --structure_trainable True --structure_weight /data/project/rw/lmdesign_results/mpnn_cmlm/model_weights/epoch100.pt --num_decoder_layers 4

# # Training LMDESIGN on CATH4.2 exp2
# python lmdesign_train.py --epoch 10 --out_folder /data/project/rw/lmdesign_results/lmdesign_reproduce/lmdesign_repro_exp2/ --embed_dim 1280 --num_heads 10 --structure_model MPNN --structure_trainable True --structure_weight /data/project/rw/mpnn_cmlm/model_weights/epoch100.pt &

# # Training LMDESIGN on CATH4.2 exp3
# python lmdesign_train.py --epoch 10 --out_folder /data/project/rw/lmdesign_results/lmdesign_reproduce/lmdesign_repro_exp3/ --embed_dim 1280 --num_heads 10 --structure_model MPNN --structure_trainable False --structure_weight /data/project/rw/mpnn_cmlm/model_weights/epoch100.pt


source /opt/conda/bin/activate base
P1="python lmdesign_train.py --epoch 1 --out_folder /data/private/LMDESIGN/test/lmdesign_exp1_reproduce/ --embed_dim 1280 --num_heads 10 --structure_model MPNN --structure_trainable True"
P2="python lmdesign_train.py --epoch 1 --out_folder /data/private/LMDESIGN/test/lmdesign_exp2_reproduce/ --embed_dim 1280 --num_heads 10 --structure_model MPNN --structure_trainable True --structure_weight /data/project/rw/lmdesign_results/mpnn_cmlm/model_weights/epoch100.pt --num_decoder_layers 4"
# Training MPNN on CATH4.2
$P1 & $P2
wait 