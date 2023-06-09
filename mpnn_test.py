import argparse
import os.path

def main(args):

    import json, time, os, sys, glob
    import numpy as np
    import torch
    from torch.utils.data.dataset import random_split, Subset
    import copy
    import torch.nn.functional as F
    import random
    import os.path

    from models.mpnn_utils import loss_nll, CATH_tied_featurize,calculate_recovery 
    from models.mpnn_utils import StructureDataset, ProteinMPNN, TS50, TS500
    import csv

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)   

    from tqdm import tqdm
    base_folder = time.strftime(args.out_folder, time.localtime())

    if base_folder[-1] != '/':
        base_folder += '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolders = ['results']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)
    
    BATCH_COPIES = 1
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
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
    _, _, all_test_set = [
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
    print(f"Test all: {len(all_test_set)} Test short: {len(short_test_set)} Test sc : {len(sc_test_set)}")


    ##############################
    ## load models 
    #############################
    model = ProteinMPNN(ca_only=False, num_letters=21, node_features=args.hidden_dim, edge_features=args.hidden_dim, hidden_dim=args.hidden_dim,
                         num_encoder_layers=args.num_encoder_layers, num_decoder_layers=args.num_decoder_layers, augment_eps=args.backbone_noise,
                           k_neighbors=args.num_neighbors)
    model.to(device)
    if args.use_pretrained_weights:
        checkpoint = torch.load(args.use_pretrained_weights, map_location=device) 
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    f = open(base_folder + 'results/ts50.csv','w', newline='')
    wr = csv.writer(f)
    header = ['Index', 'GT', 'PD', 'Perplexity', 'Acc']
    wr.writerow(header)
    with torch.no_grad():
        ts50_test_sum, ts50_test_weights = 0., 0.
        ts50_recovery = []
        test_ts50_pbar = tqdm(ts50_set)
        for ix, protein in enumerate(test_ts50_pbar):
            batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
            X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = CATH_tied_featurize(batch_clones, device, chain_id_dict)
            randn_1 = torch.randn(chain_M.shape, device=X.device)
            log_probs,_ = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
            mask_for_loss = mask*chain_M*chain_M_pos
            loss, loss_av = loss_nll(S,log_probs, mask_for_loss)
            
            sub_sum = torch.sum(loss * mask_for_loss).cpu().data.numpy()
            sub_weights = torch.sum(mask_for_loss).cpu().data.numpy()

            ts50_test_sum += sub_sum
            ts50_test_weights += sub_weights
            sub_accs = calculate_recovery(log_probs,S)
            ts50_recovery.append(sub_accs)
            
            pd = torch.argmax(F.softmax(log_probs, dim=-1), dim=-1)
            pd = ''.join([alphabet[s] for s in pd[0]])
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
            try:
                batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
                X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = CATH_tied_featurize(batch_clones, device, chain_id_dict)
                randn_1 = torch.randn(chain_M.shape, device=X.device)
                log_probs,_ = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
                mask_for_loss = mask*chain_M*chain_M_pos
                loss, loss_av = loss_nll(S,log_probs, mask_for_loss)
                
                sub_sum = torch.sum(loss * mask_for_loss).cpu().data.numpy()
                sub_weights = torch.sum(mask_for_loss).cpu().data.numpy()

                ts500_test_sum += sub_sum
                ts500_test_weights += sub_weights
                sub_accs = calculate_recovery(log_probs,S)
                ts500_recovery.append(sub_accs)
                
                pd = torch.argmax(F.softmax(log_probs, dim=-1), dim=-1)
                pd = ''.join([alphabet[s] for s in pd[0]])
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



    # # #############
    # # ## CATH TEST 
    # # #############
    f = open(base_folder + 'results/all_results.csv','w', newline='')
    wr = csv.writer(f)
    header = ['Index', 'GT', 'PD', 'Perplexity', 'Acc']
    wr.writerow(header)
    with torch.no_grad():
        all_test_sum, all_test_weights = 0., 0.
        all_test_recovery = []
        all_test_pbar = tqdm(all_test_set)
        for ix, protein in enumerate(all_test_pbar):
            batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
            X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = CATH_tied_featurize(batch_clones, device, chain_id_dict)
            randn_1 = torch.randn(chain_M.shape, device=X.device)
            log_probs,_ = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
            mask_for_loss = mask*chain_M*chain_M_pos
            loss, loss_av = loss_nll(S,log_probs, mask_for_loss)
            
            sub_sum = torch.sum(loss * mask_for_loss).cpu().data.numpy()
            sub_weights = torch.sum(mask_for_loss).cpu().data.numpy()

            all_test_sum += sub_sum
            all_test_weights += sub_weights
            sub_accs = calculate_recovery(log_probs,S)
            all_test_recovery.append(sub_accs)
            
            pd = torch.argmax(F.softmax(log_probs, dim=-1), dim=-1)
            pd = ''.join([alphabet[s] for s in pd[0]])
            wr.writerow([ix,protein['seq'], pd, np.exp(sub_sum / sub_weights), sub_accs])
    f.close()


    f = open(base_folder + 'results/short_results.csv','w', newline='')
    wr = csv.writer(f)
    header = ['Index', 'GT', 'PD', 'Perplexity', 'Acc']
    wr.writerow(header)
    with torch.no_grad():
        short_test_sum, short_test_weights = 0., 0.
        short_test_recovery = []
        test_pbar = tqdm(short_test_set)
        for ix, protein in enumerate(test_pbar):
            batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
            X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = CATH_tied_featurize(batch_clones, device, chain_id_dict)
            randn_1 = torch.randn(chain_M.shape, device=X.device)
            log_probs,_ = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
            mask_for_loss = mask*chain_M*chain_M_pos
            loss, loss_av = loss_nll(S,log_probs, mask_for_loss)
            
            sub_sum = torch.sum(loss * mask_for_loss).cpu().data.numpy()
            sub_weights = torch.sum(mask_for_loss).cpu().data.numpy()

            short_test_sum += sub_sum
            short_test_weights += sub_weights
            sub_accs = calculate_recovery(log_probs,S)
            short_test_recovery.append(sub_accs)
            
            pd = torch.argmax(F.softmax(log_probs, dim=-1), dim=-1)
            pd = ''.join([alphabet[s] for s in pd[0]])
            wr.writerow([ix,protein['seq'], pd, np.exp(sub_sum / sub_weights), sub_accs])
    f.close()

    f = open(base_folder + 'results/sc_results.csv','w', newline='')
    wr = csv.writer(f)
    header = ['Index', 'GT', 'PD', 'Perplexity', 'Acc']
    wr.writerow(header)
    with torch.no_grad():
        sc_test_sum, sc_test_weights = 0., 0.
        sc_test_recovery = []
        test_pbar = tqdm(sc_test_set)
        for ix, protein in enumerate(test_pbar):
            batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
            X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = CATH_tied_featurize(batch_clones, device, chain_id_dict)
            randn_1 = torch.randn(chain_M.shape, device=X.device)
            log_probs,_ = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
            mask_for_loss = mask*chain_M*chain_M_pos
            loss, loss_av = loss_nll(S,log_probs, mask_for_loss)
            
            sub_sum = torch.sum(loss * mask_for_loss).cpu().data.numpy()
            sub_weights = torch.sum(mask_for_loss).cpu().data.numpy()

            sc_test_sum += sub_sum
            sc_test_weights += sub_weights
            sub_accs = calculate_recovery(log_probs,S)
            sc_test_recovery.append(sub_accs)
            
            pd = torch.argmax(F.softmax(log_probs, dim=-1), dim=-1)
            pd = ''.join([alphabet[s] for s in pd[0]])
            wr.writerow([ix,protein['seq'], pd, np.exp(sub_sum / sub_weights), sub_accs])
    f.close()
    


    all_test_loss = all_test_sum / all_test_weights
    all_test_perplexity = np.exp(all_test_loss)
    all_recovery = np.median(all_test_recovery)
    short_test_loss = short_test_sum / short_test_weights
    short_test_perplexity = np.exp(short_test_loss)
    short_recovery = np.median(short_test_recovery)
    sc_test_loss = sc_test_sum / sc_test_weights
    sc_test_perplexity = np.exp(sc_test_loss)
    sc_recovery = np.median(sc_test_recovery)

    output = "Test all perlexity : {:.4f} Test all medain recovery : {:.4f} \n \
            Shot perplexity : {:.4f} Short median recovery : {:.4f} \n \
            Single chain perplexity : {:.4f} Single chain median recovery : {:.4f}".format(all_test_perplexity.item(), all_recovery.item(),
                                                                                            short_test_perplexity.item(), short_recovery.item(),
                                                                                            sc_test_perplexity.item(), sc_recovery.item())
    print(TSoutput)
    print(output)
        
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--ca_only", action="store_true", default=False, help="Parse CA-only structures and use CA-only models (default: false)") 
    argparser.add_argument("--use_pretrained_weights", type=str, default="./weights/vanilla_model_weights/v_48_020.pt", help="pretraIned MPNN weights")
    argparser.add_argument("--max_length", type=int, default=200000, help="Max sequence length")
    argparser.add_argument("--seed", type=int, default=2020, help="If set to 0 then a random seed will be picked;")

    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers") 
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of encoder layers") 
    argparser.add_argument("--num_neighbors", type=int, default=48, help="number of neighbors for the sparse graph")   
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.0, help="amount of noise added to backbone during training") 
    
    argparser.add_argument("--out_folder", type=str, default='/data/project/rw/mpnn_results/MPNN/', help="Path to a folder to output sequences, e.g. /home/out/")
    argparser.add_argument("--jsonl_path", type=str,default='/data/project/rw/cath4.2/chain_set.jsonl',help="Path to parsed pdb into jsonl")
    argparser.add_argument("--file_splits", type=str, default='/data/project/rw/cath4.2/chain_set_splits.json', help='Path to train/valid/test split info')
    argparser.add_argument("--test_short_path", type=str, default="/data/project/rw/cath4.2/test_split_L100.json", help="Path to Short test split")
    argparser.add_argument("--test_single_path", type=str, default="/data/project/rw/cath4.2/test_split_sc.json", help="Path to Single test split")
    argparser.add_argument("--test_ts_directory", type=str, default='/data/project/rw/ts/',help="Directory for ts datasets")
    argparser.add_argument("--chain_id_jsonl",type=str, default='', help="Path to a dictionary specifying which chains need to be designed and which ones are fixed, if not specied all chains will be designed.")
    
    args = argparser.parse_args()    
    main(args)   
