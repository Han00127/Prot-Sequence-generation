#!/bin/bash
source /opt/conda/bin/activate base
# Training LMDESIGN on CATH4.2 exp3    
python lmdesign_train.py --epoch 1 --out_folder /data/private/LMDESIGN/test/lmdesign_exp3_reproduce/ --embed_dim 1280 --num_heads 10 --structure_model MPNN --structure_weight /data/project/rw/lmdesign_results/mpnn_cmlm/model_weights/epoch100.pt --num_decoder_layers 4
# python lmdesign_train.py --epoch 100 --out_folder /data/project/rw/codetest_results/lmdesign/lmdesign_exp3_reproduce/ --embed_dim 1280 --num_heads 10 --structure_model MPNN --structure_weight /data/project/rw/lmdesign_results/mpnn_cmlm/model_weights/epoch100.pt --num_decoder_layers 4