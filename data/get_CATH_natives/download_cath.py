import json
import os
import sys

from tqdm import tqdm
from Bio.PDB import PDBList
IN = sys.argv[1] # train

def get_pdb(p_id, c_id):
    # # Create a PDBList object
    pdbl = PDBList()

    # Enter the PDB code and the chain you want to download
    pdb_code = p_id  # Example: Change this to your desired PDB code
    chain_id = c_id  # Example: Change this to your desired chain

    # Download the PDB file
    url = f"https://files.rcsb.org/download/{pdb_code.upper()}.pdb -O /data/project/rw/cath4.2/natives/{IN}/{pdb_code}.{chain_id}.pdb"
    os.system(f"wget {url}")

if __name__ == "__main__":
    # pdb_id = "1A20"
    # chain_id = "A"
    path = '/data/project/rw/cath4.2/chain_set_splits.json' ## changed here where it contains splits informations
    count = 0
    with open(path, 'r') as file:
        data = json.load(file)
        for id in tqdm(data[f'{IN}']):
            pdb_id, chain_id = id.split('.')

            get_pdb(pdb_id, chain_id)
            count +=1

    print("Done all downloaded files ", count )