# !/bin/bash
# CATH 4.2
wget http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set.jsonl
wget http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set_splits.json
wget http://people.csail.mit.edu/ingraham/graph-protein-design/data/SPIN2/test_split_L100.json
wget http://people.csail.mit.edu/ingraham/graph-protein-design/data/SPIN2/test_split_sc.json

# TS 50&500 
wget https://github.com/A4Bio/PiFold/releases/download/Training%26Data/ts.zip
unzip ts.zip
