### Structure-informed Langugage Models Are Protein Designers

This is the unofficial code of the arxiv paper *Structure-informed Language Models Are Protein Designers* published by *Zhang Zaixiang et al* 

![LMDesign](https://github.com/Han00127/Structure-informed-Language-Models-Are-Protein-Designers/assets/93216105/6bcdcb65-8ce7-4736-ae29-d0302a535c1f)

One line summary of implmentation: 
We achieve relativly similar performance for base model like ProtMPNN, ProtMPNN-CMLM models. However, our implementation achieve surprising results on CATH4.2 and TS test datset such that we reach state-of-the-art performance on both dataset.
our result / paper results. Please refer to "results/" together to verify the experimental results. 
![image](https://github.com/Han00127/Structure-informed-Language-Models-Are-Protein-Designers/assets/93216105/bc5c04a8-a81d-4f46-9eb1-3f75bfe18e5c)

Code implmentation
-----------------------------------------------------------------------------------------------------
To run LMDesign clone this github repo and install corresponding **lmdesign.yaml** environemnt on your conda or if you are using Braincloud, use **lmdesing_final** shared image.

If use yaml installation, please check dependency in yaml files and install with the following command: 
conda env create --file lmdesign.yaml 

All methods are light weight models so that 1 GPU (V100) is enough. But it still has depency on CPU computing on featurizations on both structure and PLM models. 

Code organization:
* `data/` - directory contains dataset download code and even native pdb files.
* `helper_scripts/` - helper functions for future works like multi chains parsing and residue fix, adding AA bias, tying residue etc.
* `models/` - directory of models for LMDesign, Structure adapter, ProteinMPNN, Pifold, ESM models and utilities.   
* `results/` - contains conducted experimental results like ProtMPNN, ProtMPNN-CMLM, LMDesign1, LMDesign2, and LMDesign2.
* `scripts/` - contains the shell script to reproduce our models 
* `weights/` - directory of base inverse fold model weight e.g, ProteinMPNN, PiFold
* `mpnn_train.py` - train and validate protMPNN and protMPNN-CMLM models 
* `mpnn_test.py` - test protMPNN and protMPNN-CMLM models on CATH4.2 and extra test dataset TS50 and TS500
* `lmdesign_train.py` - train and validate the variate LMDesign models on CATH4.2
* `lmdesign_test.py` - test LMDesign models on CATH4.2 and extra test dataset TS50 and TS500
-----------------------------------------------------------------------------------------------------

## Data preparation 
Main dataset used in this models are CATH4.2. To install the dataset, go to data and follow command:
    ./download_preprocessed_dataset.sh 
This will automatically download dataset on current folder. For TS dataset, it should be unzip after installation.

## Trained model weight preparation 
Trained ProtMPNN, ProtMPNN-CMLM, LMDesign1 (ProtMPNN-CMLM), LMDesign2 (Pretrained ProtMPNN-CMLM:finetune), LMDesign3 (Pretrained ProtMPNN-CMLM:freeze), ESM weights can be found https://shorturl.at/bopqS
For reproducing the reported experimental results, install ProtMPNN, ProtMPNN-CMLM, LMDesign* weights from above. 
For initial training model, ESM weights are mendatory.

## Reproduce ProtMPNN and ProtMPNN-CMLM models 
To test pretrained ProtMPNN and ProtMPNN-CMLM model, please refer to "scripts/mpnn_test.sh". 
To do so, set mendatory data path in shell scripts :
```
save_dir='where you want to save the result of experiment'
saved_weight='Trained weight path and id e.g., /data/project/rw/mpnn_results/reproduce2/model_weights/epoch100.pt'
cath_file='path for CATH jsonl file e.g., /data/project/rw/cath4.2/chain_set.jsonl'
cath_splits='path for split train/valid/test json file e.g., /data/project/rw/cath4.2/chain_set_splits.json'
short_splits='path for short test json file e.g., /data/project/rw/cath4.2/test_split_L100.json'
single_splits='path for single chain test json file e.g., /data/project/rw/cath4.2/test_split_sc.json'
ts_dir='directory contains ts50, ts500 json file e.g., /data/project/rw/ts/'
```
after setting up above, run mpnn_test.sh file. Note that I set parallel execution on protMPNN and protMPNN-CMLM. Both setting should be configured. If not, please modify the code for single run.

To train ProtMPNN model, please refer to "scripts/mpnn.sh" and "mpnn_train.py"
```
    - Default setting of ProtMPNN model parameter 
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_layers", type=int, default=3, help="number of encoder layers") 
    argparser.add_argument("--num_neighbors", type=int, default=48, help="number of neighbors for the sparse graph")   
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.2, help="amount of noise added to backbone during training") 
    - Data path like CATH data path
    argparser.add_argument("--out_folder", type=str, default='/data/project/rw/mpnn_results/MPNN/', help="Path to a folder to output sequences, e.g. /home/out/")
    argparser.add_argument("--jsonl_path", type=str,default='/data/project/rw/cath4.2/chain_set.jsonl',help="Path to parsed pdb into jsonl")
    argparser.add_argument("--file_splits", type=str, default='/data/project/rw/cath4.2/chain_set_splits.json', help='Path to train/valid/test split info')
    argparser.add_argument("--test_short_path", type=str, default="/data/project/rw/cath4.2/test_split_L100.json", help="Path to Short test split")
    argparser.add_argument("--test_single_path", type=str, default="/data/project/rw/cath4.2/test_split_sc.json", help="Path to Single test split")
    argparser.add_argument("--chain_id_jsonl",type=str, default='', help="Path to a dictionary specifying which chains need to be designed and which ones are fixed, if not specied all chains will be designed.")
```
after setting up above, run mpnn_train.sh file.

To train ProtMPNN-CMLM model, everything same as above except :
```
    argparser.add_argument("--model", type=str, default="MPNN_CMLM", help="Select base models")
```
run mpnn.sh in script.

## Reproduce LMDesign models 
