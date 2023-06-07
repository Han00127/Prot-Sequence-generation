#!/bin/bash
#SBATCH --mem=32g
#SBATCH -c 2
#SBATCH --output=parse_multiple_chains.out

source activate base
python parse_multiple_chains.py --input_path='../inputs/PDB_complexes/pdbs/' --output_path='/data/private/ProteinMPNN-main/inputs/PDB_complexes/parsed_pdbs.jsonl'
