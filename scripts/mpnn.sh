# #!/bin/bash
# source /opt/conda/bin/activate base
# P1="python ../mpnn_train.py --model MPNN --epoch 100 --out_folder /data/project/rw/codetest_results/mpnn/mpnn_repro --jsonl_path /data/project/rw/cath4.2/chain_set.jsonl --file_splits /data/project/rw/cath4.2/chain_set_splits.json"
# P2="python ../mpnn_train.py --model MPNN_CMLM --epoch 100 --out_folder /data/project/rw/codetest_results/mpnn/mpnn_cmlm_repro --jsonl_path /data/project/rw/cath4.2/chain_set.jsonl --file_splits /data/project/rw/cath4.2/chain_set_splits.json"
# # Training MPNN on CATH4.2
# $P1 & $P2
# wait 



#!/bin/bash
source /opt/conda/bin/activate base
P1="python mpnn_train.py --model MPNN --epoch 2 --out_folder /data/private/LMDESIGN/test/mpnn --jsonl_path /data/project/rw/cath4.2/chain_set.jsonl --file_splits /data/project/rw/cath4.2/chain_set_splits.json"
P2="python mpnn_train.py --model MPNN_CMLM --epoch 2 --out_folder /data/private/LMDESIGN/test/mpnn_cmlm --jsonl_path /data/project/rw/cath4.2/chain_set.jsonl --file_splits /data/project/rw/cath4.2/chain_set_splits.json"
# Training MPNN on CATH4.2
$P1 & $P2
wait 