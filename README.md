# Structure-informed Langugage Models Are Protein Designers

This is the unofficial code of the arxiv paper *Structure-informed Language Models Are Protein Designers* published by *Zhang Zaixiang et al* 

![LMDesign](https://github.com/Han00127/Structure-informed-Language-Models-Are-Protein-Designers/assets/93216105/6bcdcb65-8ce7-4736-ae29-d0302a535c1f)

## Summary of implmentation: 

We achieve similar performance in base model implementation like ProtMPNN, ProtMPNN-CMLM models. However, our implementation on LM-Design achieves surprising results on CATH4.2 and TS test datset as shown in below Table. Please refer to "results/" together to verify the experimental results. 

![image](https://github.com/Han00127/Structure-informed-Language-Models-Are-Protein-Designers/assets/93216105/bc5c04a8-a81d-4f46-9eb1-3f75bfe18e5c)

Code implmentation
-----------------------------------------------------------------------------------------------------
To run LM-Design, clone my github repo and install corresponding **lmdesign.yaml** environemnt on your conda or if you are using Braincloud, use **lmdesing_final** shared image.

For yaml installation, please check dependency in yaml files and install with the following command: 
conda env create --file lmdesign.yaml 

All methods are light weight models so that 1 GPU (V100) is enough. But it still has depency on CPU computing on featurizations on both structure and PLM models. 

Code organization:
* `data/` - directory contains dataset download code and even native pdb files.
* `helper_scripts/` - helper functions for future works like multi chains parsing and residue fix, adding AA bias, tying residue etc.
* `models/` - directory of models for LM-Design, Structure adapter, ProteinMPNN, Pifold, ESM models and utilities.   
* `results/` - contains conducted experimental results like ProtMPNN, ProtMPNN-CMLM, LM-Design1, LM-Design2, and LM-Design2.
* `scripts/` - contains the shell script to reproduce our models 
* `weights/` - directory of base inverse fold model weight e.g, ProteinMPNN
* `mpnn_train.py` - train and validate protMPNN and protMPNN-CMLM models 
* `mpnn_test.py` - test protMPNN and protMPNN-CMLM models on CATH4.2 and extra test dataset TS50 and TS500
* `lmdesign_train.py` - train and validate the variate LMDesign models on CATH4.2
* `lmdesign_test.py` - test LM-Design models on CATH4.2 and extra test dataset TS50 and TS500
-----------------------------------------------------------------------------------------------------

## Data preparation 
Main dataset used in this models are CATH4.2 and TS dataset. To install the dataset, go to data and follow command:

    ./download_preprocessed_dataset.sh 
    
This will automatically download dataset on data directory. This should be done before any training or test.

-----------------------------------------------------------------------------------------------------

## Trained model weight preparation 
Trained ProtMPNN, ProtMPNN-CMLM, LM-Design1 (ProtMPNN-CMLM), LM-Design2 (Pretrained ProtMPNN-CMLM:finetune), LM-Design3 (Pretrained ProtMPNN-CMLM:freeze), ESM weights can be found https://shorturl.at/bopqS. To train the model, **ESM weight** should be ready.  

-----------------------------------------------------------------------------------------------------
## Reproduce ProtMPNN and ProtMPNN-CMLM models 
To test pretrained ProtMPNN and ProtMPNN-CMLM model, please refer to "scripts/mpnn_test.sh". 
To do so, set mendatory data path in shell scripts :
```
## set data path if run download_preprocessed_datasets.sh in data directory, don't need to modify here. ##
cath_file='./data/chain_set.jsonl'
cath_splits='./data/chain_set_splits.json'
short_splits='./data/test_split_L100.json'
single_splits='./data/test_split_sc.json'
ts_dir='./data/ts/'

save_dir='./result/mpnn_reproduce/' ## results directory 
## If you are able to access project folder, you can use this code. 
saved_weight='/data/project/rw/LMDesign_weights/mpnn_trained_weight.pt'
## Otherwise, should be defined by the path where you install the weight
saved_weight='' 

# Test trained model on CATH4.2 and TS50 & 500
# #################################################
# ## set models - ProtMPNN
# #################################################
python mpnn_test.py --use_pretrained_weights $saved_weight --out_folder $save_dir --jsonl_path $cath_file --file_splits $cath_splits --test_short_path $short_splits --test_single_path $single_splits --test_ts_directory $ts_dir 

#################################################
## set models - ProtMPNN+CMLM
#################################################
save_dir='./result/mpnn_cmlm_reproduce/'
saved_weight='/data/project/rw/LMDesign_weights/mpnn_cmlm_trained_weight.pt' 

## This is model trained mpnn+cmlm with 4 decoder layer
## If you trained set num_encoder_layers and num_decoder_layers different, use them here. 
python mpnn_test.py --use_pretrained_weights $saved_weight --out_folder $save_dir --jsonl_path $cath_file --file_splits $cath_splits --test_short_path $short_splits --test_single_path $single_splits --test_ts_directory $ts_dir --num_decoder_layers 4

```
after setting up above, run **mpnn_test.sh** file. Note that I set parallel execution on protMPNN and protMPNN-CMLM. Both setting should be configured. If not, please modify the code for single run.

To train ProtMPNN model, please refer to "scripts/mpnn.sh" and "mpnn_train.py"
```
## Define the datapath as mentioned above
out_folder='./results/reproduce_mpnn_test'
cathDataPath='./data/chain_set.jsonl'
splits='./data/chain_set_splits.json'
shortPath='./data/test_split_L100.json'
scPath='./data/test_split_sc.json'

python mpnn_train.py --model MPNN --epoch 100 --out_folder $out_folder --jsonl_path $cathDataPath --file_splits $splits --test_short_path $shortPath --test_single_path $scPath
```
after setting up above, run mpnn_train.sh file.

To train ProtMPNN-CMLM model, please refer to "scripts/mpnn_cmlm.sh" and "mpnn_train.py"
```
# Training from pretrained model vanilla ProtMPNN model - v_48_020.pt)) encoder decoder 3 / 3 as same as official model
# python mpnn_train.py --model MPNN_CMLM --epoch 1 --out_folder $out_folder --jsonl_path $cathDataPath --file_splits $splits --test_short_path $shortPath --test_single_path $scPath

## My trained models has 4 decoder layers training from scratch
python mpnn_train.py --model MPNN_CMLM --epoch 1 --out_folder $out_folder --jsonl_path $cathDataPath --file_splits $splits --test_short_path $shortPath --test_single_path $scPath --use_pretrained_weights "" --num_decoder_layers 4 
```
run **mpnn_cmlm.sh** in script.

-----------------------------------------------------------------------------------------------------

## Reproduce LM-Design models 
LM-Design has three main modules Structure encoder (e.g., protMPNN), Protein Language Model (e.g., ESM1b) and strcuture adapter.
### LMDesign1 (ProtMPNN-CMLM)
To test pretrained LM-Design1 model, please refer to "scripts/lmdesign_test.sh" and "lmdesign_test.py"
To do so, set mendatory data path in shell scripts :
```
## set data path as usual as above 
cath_file='./data/chain_set.jsonl'
cath_splits='./data/chain_set_splits.json'
short_splits='./data/test_split_L100.json'
single_splits='./data/test_split_sc.json'
ts_dir='./data/ts/'

save_dir='./results/lmdesign_exp1_reproduce' # save result path 
## If you are able to access project folder, you can use this code. 
saved_weight='/data/project/rw/LMDesign_weights/mpnn_trained_weight.pt'
## Otherwise, should be defined by the path where you install the weight
saved_weight='' 

# #################################################
# ## set models - LM-Design1 test on CATH 4.2 and TS
# #################################################
python lmdesign_test.py --use_pretrained_weights $saved_weight --out_folder $save_dir --embed_dim 1280 --num_heads 10 --structure_model MPNN  --structure_weight "" --jsonl_path $cath_file --file_splits $cath_splits --test_short_path $short_splits --test_single_path $single_splits --test_ts_directory $ts_dir --num_refinement 6
```
To train the LMDesign1 model, please refer to "script/lmdesign.sh" and "lmdesign_train.py".
```
## Define dataset 
# out_folder='./results/lmdesign1_reproduce'
# cathDataPath='./data/chain_set.jsonl'
# splits='./data/chain_set_splits.json'
# shortPath='./data/test_split_L100.json'
# scPath='./data/test_split_sc.json'

# ####################################
# ## Single implmenation 
# #####################################
# # Training LMDESIGN on CATH4.2 exp1 
# python lmdesign_train.py --epoch 100 --out_folder $out_folder --jsonl_path $cathDataPath --file_splits $splits --test_short_path $shortPath --test_single_path $scPath --embed_dim 1280 --num_heads 10 --structure_model MPNN --structure_trainable True --num_decoder_layers 4  --structure_weight ""
```
run lmdesign.sh to train LM-Design1. 

## LMDesign2 (pretrained ProtMPNN-CMLM: fine-tune)
To test pretrained LMDesign2 model, please refer to "scripts/lmdesign_test.sh" and "lmdesign_test.py"
To do so, set mendatory data path in shell scripts :
```
## set data path as usual as above 
cath_file='./data/chain_set.jsonl'
cath_splits='./data/chain_set_splits.json'
short_splits='./data/test_split_L100.json'
single_splits='./data/test_split_sc.json'
ts_dir='./data/ts/'

save_dir2='./results/lmdesign_exp2_reproduce'
## If you are able to access project folder, you can use this code. 
saved_weight2='/data/project/rw/LMDesign_weights/lmdesign2_trained_weight.pt'
## Otherwise, should be defined by the path where you install the weight
saved_weight2='' 

# #################################################
# ## set models - LM-Design2 test on CATH 4.2 and TS
# #################################################
save_dir2='./results/lmdesign_exp2_reproduce'
saved_weight2='/data/project/rw/LMDesign_weights/lmdesign2_trained_weight.pt' # put path for saved weights
python lmdesign_test.py --use_pretrained_weights $saved_weight2 --out_folder $save_dir2 --embed_dim 1280 --num_heads 10 --structure_model MPNN --structure_weight "" --num_decoder_layers 4 --jsonl_path $cath_file --file_splits $cath_splits --test_short_path $short_splits --test_single_path $single_splits --test_ts_directory $ts_dir --num_refinement 6
```
Our pretrained protMPNN-CMLM model utlizes --num_decoder_layers 4. This would be depending on how to train the protMPNN-CMLM model. Run the code above.

To train the LMDesign2 model, please refer to "script/lmdesign.sh" and "lmdesign_train.py".
```
## Define dataset 
# out_folder='./results/lmdesign1_reproduce'
# cathDataPath='./data/chain_set.jsonl'
# splits='./data/chain_set_splits.json'
# shortPath='./data/test_split_L100.json'
# scPath='./data/test_split_sc.json'

# ####################################
# ## Single implmenation 
# #####################################
# # Training LMDESIGN on CATH4.2 exp2
# python lmdesign_train.py --epoch 100 --out_folder $out_folder --jsonl_path $cathDataPath --file_splits $splits --test_short_path $shortPath --test_single_path $scPath --embed_dim 1280 --num_heads 10 --structure_model MPNN --structure_trainable True --structure_weight $pretrained_weights --num_decoder_layers 4

```
Run the code above. In this experiment, structure encoder and structure adapters are trainable and PLM are frozen.

## LMDesign3 (pretrained ProtMPNN-CMLM: freeze)
To test pretrained LMDesign3 model, please refer to "scripts/lmdesign_test.sh" and "lmdesign_test.py" 
To do so, set mendatory data path in shell scripts :
```
## set data path as usual as above 
cath_file='./data/chain_set.jsonl'
cath_splits='./data/chain_set_splits.json'
short_splits='./data/test_split_L100.json'
single_splits='./data/test_split_sc.json'
ts_dir='./data/ts/'

save_dir3='/data/private/LMDESIGN/test/lmdesign_exp3_results'
## If you are able to access project folder, you can use this code. 
saved_weight3='/data/private/LMDESIGN/test/lmdesign_exp3_results'
## Otherwise, should be defined by the path where you install the weight
saved_weight3='' 

# #################################################
# ## set models - LM-Design3 test on CATH 4.2 and TS
# #################################################

python lmdesign_test.py --use_pretrained_weights $saved_weight3 --out_folder $save_dir3 --embed_dim 1280 --num_heads 10 --structure_model MPNN --structure_weight "" --num_decoder_layers 4 --jsonl_path $cath_file --file_splits $cath_splits --test_short_path $short_splits --test_single_path $single_splits --test_ts_directory $ts_dir --num_refinement 6

```
Our pretrained protMPNN-CMLM model utlizes --num_decoder_layers 4. This would be depending on how pretrain the protMPNN-CMLMmodel. Run the code above.

To train the LMDesign3 model, please refer to "script/lmdesign.sh" and "lmdesign_train.py".
```
## Define dataset 
# out_folder='./results/lmdesign1_reproduce'
# cathDataPath='./data/chain_set.jsonl'
# splits='./data/chain_set_splits.json'
# shortPath='./data/test_split_L100.json'
# scPath='./data/test_split_sc.json'

# ####################################
# ## Single implmenation 
# #####################################
# # Training LMDESIGN on CATH4.2 exp3
# python lmdesign_train.py --epoch 10 --out_folder$out_folder --jsonl_path $cathDataPath --file_splits $splits --test_short_path $shortPath --test_single_path $scPath --embed_dim 1280 --num_heads 10 --structure_model MPNN --structure_trainable False --structure_weight $pretrained_weights --num_decoder_layers 4
```
Run the code above. In this experiment, LMDesign3 trains only structure adapter. 

I strongly recommend to use parallel execution just adds & in the scripts. This is because the trainable weights are relatively small. To maximize utilization of GPU, please process_train1 & process_train2. For the details of it, please refer to "./scripts/lmdesign.sh 

If you struggle with reproducing on my code,  please contact me via email (구일kthan 엣 gmail.com). Or any verification or feedback on my code will be welcome :)



