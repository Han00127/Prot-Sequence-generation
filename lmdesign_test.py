import argparse

def main(args):

    import json, time, os
    import numpy as np
    import torch
    from torch.utils.data.dataset import Subset
    import copy
    import torch.nn.functional as F
    import random
    import os.path

    from models.mpnn_utils import loss_nll, calculate_recovery 
    from models.mpnn_utils import StructureDataset, TS50, TS500
    from models.lmdesign import LMDesign
    import csv

    if args.seed:
        seed=args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)   

    from tqdm import tqdm
    
    skips = [0,1,2,3,29,30,31,32]
    base_folder = time.strftime(args.out_folder, time.localtime())

    if base_folder[-1] != '/':
        base_folder += '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolders = ['results']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    dataset = StructureDataset(args.jsonl_path, truncate=None, max_length=args.max_length)
    chain_id_dict = None
    
    ################################
    ## Load Test datasets 
    ################################
    
    ## TS datasets
    ts50_set = TS50(path=args.test_ts_directory) 
    ts500_set = TS500(path=args.test_ts_directory)
    print(f"TS50 {len(ts50_set)} TS500 {(len(ts500_set))}")

    # CATH dataset 
    dataset_indices = {d['name']:i for i,d in enumerate(dataset)}
    with open(args.file_splits) as f:
        dataset_splits = json.load(f)
    _, valid_set, all_test_set = [
        Subset(dataset, [
            dataset_indices[chain_name] for chain_name in dataset_splits[key] 
            if chain_name in dataset_indices
        ])
        for key in ['train', 'validation', 'test']
    ]

    with open(args.test_short_path) as f:
        dataset_splits = json.load(f)
    short_test_set = Subset(dataset, [dataset_indices[chain_name] for chain_name in dataset_splits['test'] if chain_name in dataset_indices])

    with open(args.test_single_path) as f:
        dataset_splits = json.load(f)
    sc_test_set = Subset(dataset, [dataset_indices[chain_name] for chain_name in dataset_splits['test'] if chain_name in dataset_indices])
    
    
    print(f"Test all : {len(all_test_set)} Test short: {len(short_test_set)} Test sc : {len(sc_test_set)}")

    ## Load saved models
    model = LMDesign(args, device)
    model.to(device)
    print(sum(p.numel() for p in model.parameters()))
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    if args.use_pretrained_weights:
        checkpoint = torch.load(args.use_pretrained_weights, map_location=device) 
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # #############
    # ## TS TEST 
    # #############
    f = open(base_folder + 'results/ts50.csv','w', newline='')
    wr = csv.writer(f)
    header = ['Index', 'GT', 'PD', 'Perplexity', 'Acc']
    wr.writerow(header)
    with torch.no_grad():
        ts50_test_sum, ts50_test_weights = 0., 0.
        ts50_recovery = []
        test_ts50_pbar = tqdm(ts50_set)
        for ix, protein in enumerate(test_ts50_pbar):
            batch_clones = [copy.deepcopy(protein) for i in range(1)] ## GT model.alphabet.encode(batch_clones[0]['seq'])
            GT = torch.tensor([model.alphabet.encode(protein['seq'])]).to(device)
            results = model.iterative_refine_inference(batch_clones, iterations=3,temp=1.0) ## base = 2
            loss, loss_av = loss_nll(GT, results['log_probs'], results['mask_for_loss'])
            
            sub_sum = torch.sum(loss * results['mask_for_loss']).cpu().data.numpy()
            sub_weights = torch.sum(results['mask_for_loss']).cpu().data.numpy()

            ts50_test_sum += sub_sum
            ts50_test_weights += sub_weights
            sub_accs = calculate_recovery(results['log_probs'],GT)
            ts50_recovery.append(sub_accs)
            
            pd = torch.argmax(F.softmax(results['log_probs'], dim=-1), dim=-1)
            pd = ''.join([model.alphabet.all_toks[s] if not s in skips else 'X' for s in pd[0]])
            wr.writerow([ix,protein['seq'], pd, np.exp(sub_sum / sub_weights), sub_accs])
    f.close()
    
    f = open(base_folder + 'results/ts500.csv','w', newline='')
    wr = csv.writer(f)
    header = ['Index', 'GT', 'PD', 'Perplexity', 'Acc']
    wr.writerow(header)
    with torch.no_grad():
        ts500_test_sum, ts500_test_weights = 0., 0.
        ts500_recovery = []
        test_ts500_pbar = tqdm(ts500_set)
        for ix, protein in enumerate(test_ts500_pbar):
            batch_clones = [copy.deepcopy(protein) for i in range(1)] ## GT model.alphabet.encode(batch_clones[0]['seq'])
            GT = torch.tensor([model.alphabet.encode(protein['seq'])]).to(device)
            try:
                results = model.iterative_refine_inference(batch_clones, iterations=3,temp=1.0) ## base = 2
                loss, loss_av = loss_nll(GT, results['log_probs'], results['mask_for_loss'])
                
                sub_sum = torch.sum(loss * results['mask_for_loss']).cpu().data.numpy()
                sub_weights = torch.sum(results['mask_for_loss']).cpu().data.numpy()

                ts500_test_sum += sub_sum
                ts500_test_weights += sub_weights
                sub_accs = calculate_recovery(results['log_probs'],GT)
                ts500_recovery.append(sub_accs)
                
                pd = torch.argmax(F.softmax(results['log_probs'], dim=-1), dim=-1)
                pd = ''.join([model.alphabet.all_toks[s] if not s in skips else 'X' for s in pd[0]])
                wr.writerow([ix,protein['seq'], pd, np.exp(sub_sum / sub_weights), sub_accs])
            except:
                pass
    f.close()
    ts50_test_loss = ts50_test_sum / ts50_test_weights
    ts50_test_perplexity = np.exp(ts50_test_loss)
    ts50_test_recovery = np.median(ts50_recovery)
    
    ts500_test_loss = ts500_test_sum / ts500_test_weights
    ts500_test_perplexity = np.exp(ts500_test_loss)
    ts500_test_recovery = np.median(ts500_recovery)
    TSoutput = "TS50 perplexity : {:.4f} TS50 median recovery {:.4f} TS500 perplexity : {:.4f} TS500 median recovery {:.4f}".format(ts50_test_perplexity.item(), ts50_test_recovery.item(),ts500_test_perplexity.item(), ts500_test_recovery.item())
    # #############
    # ## CATH TEST 
    # #############
    f = open(base_folder + 'results/all_results.csv','w', newline='')
    wr = csv.writer(f)
    header = ['Index', 'GT', 'PD', 'Perplexity', 'Acc']
    wr.writerow(header)
    with torch.no_grad():
        all_test_sum, all_test_weights = 0., 0.
        all_recovery = []
        test_all_pbar = tqdm(all_test_set)
        total_count = 0
        for ix, protein in enumerate(test_all_pbar):
            batch_clones = [copy.deepcopy(protein) for i in range(1)] ## GT model.alphabet.encode(batch_clones[0]['seq'])
            GT = torch.tensor([model.alphabet.encode(protein['seq'])]).to(device)
            results = model.iterative_refine_inference(batch_clones, iterations=args.num_refinement,temp=args.temperature) ## base = 2
            loss, loss_av = loss_nll(GT, results['log_probs'], results['mask_for_loss'])
            
            sub_sum = torch.sum(loss * results['mask_for_loss']).cpu().data.numpy()
            sub_weights = torch.sum(results['mask_for_loss']).cpu().data.numpy()

            all_test_sum += sub_sum
            all_test_weights += sub_weights
            sub_accs = calculate_recovery(results['log_probs'],GT)
            all_recovery.append(sub_accs)
            
            pd = torch.argmax(F.softmax(results['log_probs'], dim=-1), dim=-1)
            pd = ''.join([model.alphabet.all_toks[s] if not s in skips else 'X' for s in pd[0]])
            wr.writerow([ix,protein['seq'], pd, np.exp(sub_sum / sub_weights), sub_accs])
    f.close()
    
    f = open(base_folder + 'results/short_results.csv','w', newline='')
    wr = csv.writer(f)
    header = ['Index', 'GT', 'PD', 'Perplexity', 'Acc']
    wr.writerow(header)
    with torch.no_grad():
        short_test_sum, short_test_weights = 0., 0.
        short_recovery = []
        test_pbar = tqdm(short_test_set)
        total_count = 0
        for ix, protein in enumerate(test_pbar):
            batch_clones = [copy.deepcopy(protein) for i in range(1)] ## GT model.alphabet.encode(batch_clones[0]['seq'])
            results = model.iterative_refine_inference(batch_clones, iterations=args.num_refinement,temp=args.temperature)
            GT = torch.tensor([model.alphabet.encode(protein['seq'])]).to(device)
            loss, loss_av = loss_nll(GT, results['log_probs'], results['mask_for_loss'])

            sub_sum = torch.sum(loss * results['mask_for_loss']).cpu().data.numpy()
            sub_weights = torch.sum(results['mask_for_loss']).cpu().data.numpy()

            short_test_sum += sub_sum
            short_test_weights += sub_weights
            sub_accs = calculate_recovery(results['log_probs'],GT)
            short_recovery.append(sub_accs)
            
            pd = torch.argmax(F.softmax(results['log_probs'], dim=-1), dim=-1)
            pd = ''.join([model.alphabet.all_toks[s] for s in pd[0]])

            wr.writerow([ix,protein['seq'], pd, np.exp(sub_sum / sub_weights), sub_accs])
    f.close()
    
    f = open(base_folder + 'results/sc_results.csv','w', newline='')
    wr = csv.writer(f)
    header = ['Index', 'GT', 'PD', 'Perplexity', 'Acc']
    wr.writerow(header)
    with torch.no_grad():
        sc_test_sum, sc_test_weights = 0., 0.
        sc_recovery = []
        test_pbar = tqdm(sc_test_set)
        for ix, protein in enumerate(test_pbar):
            batch_clones = [copy.deepcopy(protein) for i in range(1)]
            results = model.iterative_refine_inference(batch_clones, iterations=args.num_refinement,temp=args.temperature)
            GT = torch.tensor([model.alphabet.encode(protein['seq'])]).to(device)
            loss, loss_av = loss_nll(GT, results['log_probs'], results['mask_for_loss'])

            sub_sum = torch.sum(loss * results['mask_for_loss']).cpu().data.numpy()
            sub_weights = torch.sum(results['mask_for_loss']).cpu().data.numpy()

            sc_test_sum += sub_sum
            sc_test_weights += sub_weights
            sub_accs = calculate_recovery(results['log_probs'],GT)
            sc_recovery.append(sub_accs)
            
            pd = torch.argmax(F.softmax(results['log_probs'], dim=-1), dim=-1)
            pd = ''.join([model.alphabet.all_toks[s] for s in pd[0]])

            wr.writerow([ix,protein['seq'], pd, np.exp(sub_sum / sub_weights), sub_accs])
    f.close()

    all_test_loss = all_test_sum / all_test_weights
    all_test_perplexity = np.exp(all_test_loss)
    all_test_recovery = np.median(all_recovery)


    short_test_loss = short_test_sum / short_test_weights
    short_test_perplexity = np.exp(short_test_loss)
    short_recovery = np.median(short_recovery)

    sc_test_loss = sc_test_sum / sc_test_weights
    sc_test_perplexity = np.exp(sc_test_loss)
    sc_recovery = np.median(sc_recovery)

    CATHoutput = "All perplexity : {:.4f} All median recovery {:.4f} Shot perplexity : {:.4f} Short median recovery : {:.4f} Single chain perplexity : {:.4f} Single  median recovery: {:.4f} ".format(all_test_perplexity.item(), all_test_recovery.item(),short_test_perplexity.item(), short_recovery.item(), sc_test_perplexity.item(), sc_recovery.item())
    print(TSoutput)
    print(CATHoutput)
        
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ## Setup parameters 
    argparser.add_argument("--seed", type=int, default=2020, help="If set to 0 then a random seed will be picked;")
    argparser.add_argument("--out_folder", type=str, default='/data/project/rw/lmdesign_results/debug', help="Path to a folder to save information, e.g. /home/out/")
    argparser.add_argument("--use_pretrained_weights", type=str, default="/data/private/ProteinMPNN-main/vanilla_model_weights/v_48_020.pt", help="pretraIned MPNN weights")
    
    ## Test datasets 
    argparser.add_argument("--jsonl_path", type=str,default='/data/project/rw/cath4.2/chain_set.jsonl',help="Path to a folder with parsed pdb into jsonl")
    argparser.add_argument("--file_splits", type=str, default='/data/project/rw/cath4.2/chain_set_splits.json', help='set train/valid/test info')
    argparser.add_argument("--test_single_path", type=str, default="/data/project/rw/cath4.2/test_split_sc.json", help="Path to Single test split")
    argparser.add_argument("--test_short_path", type=str, default='/data/project/rw/cath4.2/test_split_L100.json', help='Path to a Single test split')
    argparser.add_argument("--test_ts_directory", type=str, default='/data/project/rw/ts/',help="Directory for ts datasets")  
    argparser.add_argument("--max_length", type=int, default=200000, help="Max sequence length")

    ## Structure model 
    argparser.add_argument("--structure_model", type=str, default='MPNN', help='Select base structure models')
    argparser.add_argument("--structure_trainable", type=bool, default=False, help='Structure models are frozen')
    argparser.add_argument('--structure_weight', type=str, default='/data/private/ProteinMPNN-main/vanilla_model_weights/v_48_020.pt')
    argparser.add_argument("--hidden_dim", type=int, default=128, help='hidden model dimension')
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers") 
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=48, help="number of neighbors for the sparse graph")   
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.0, help="amount of noise added to backbone during training")   
    argparser.add_argument("--ca_only", action="store_true", default=False, help="Parse CA-only structures and use CA-only models (default: false)")   
    
    ## PLM model 
    argparser.add_argument("--plm_model", type=str, default='ESM1b', help='Select base PLM models')
    argparser.add_argument("--plm_weight", type=str, default="/data/project/rw/ESM_weight/esm1b_t33_650M_UR50S.pt", help="Path to model weights folder;") 
    
    ## Structure Adapter 
    argparser.add_argument("--embed_dim", type=int, default=768, help='Embed dimension of both structure and plm representations')
    argparser.add_argument("--num_heads", type=int, default=12, help='Number of heads used in attention')
    
    ## Iterative refinement configuration 
    argparser.add_argument("--num_refinement", type=int, default=5, help='Smallest step set to 2, T=1')
    argparser.add_argument("--temperature", type=float, default=1.0, help='Configuration of temperature 0.1, 0.5 1.0 1.2 1.5 and more for diversity 1.0 default')
    
    args = argparser.parse_args()    
    main(args)   
