import argparse
import os.path

def main(args):
    import json, time, os, sys, glob
    import shutil
    import warnings
    import numpy as np
    import torch
    from torch import optim
    from torch.utils.data import DataLoader

    import copy
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    import os.path  

    ## KTHAN modified
    from torch.utils.data.dataset import Subset
    from models.mpnn_utils import loss_nll, CATH_tied_featurize,get_std_opt, s_encoder_cmlm_mask,graph_loss_smoothed,calculate_recovery 
    from models.mpnn_utils import StructureDataset, ProteinMPNN, StructureLoader
    from tqdm import tqdm
    from torch.utils.tensorboard import SummaryWriter

    #######################
    ## Set general settings
    #######################
    if args.seed:
        seed=args.seed
    else:
        seed=int(np.random.randint(0, high=999, size=1, dtype=int)[0])
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed) 

    scaler = torch.cuda.amp.GradScaler()
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    base_folder = time.strftime(args.out_folder, time.localtime())

    if base_folder[-1] != '/':
        base_folder += '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolders = ['model_weights', 'train_logs']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    writer = SummaryWriter(os.path.join(args.out_folder, "train_logs"))

    #######################
    ## Processing Datasets 
    #######################
    dataset = StructureDataset(args.jsonl_path, truncate=None, max_length=args.max_length)
    chain_id_dict = None

    ## Loads CATH 4.2 Train / Valid / Test 
    dataset_indices = {d['name']:i for i,d in enumerate(dataset)}
    with open(args.file_splits) as f:
        dataset_splits = json.load(f)
    train_set, validation_set, test_set = [
        Subset(dataset, [
            dataset_indices[chain_name] for chain_name in dataset_splits[key] 
            if chain_name in dataset_indices
        ])
        for key in ['train', 'validation', 'test']
    ]
    loader_train = StructureLoader(train_set, batch_size=args.batch_size)
    loader_valid = StructureLoader(validation_set, batch_size=args.batch_size)
    loader_test = StructureLoader(test_set, batch_size=args.batch_size)
    print(f"Train : {len(train_set)}, Valida : {len(validation_set)} Test : {len(test_set)}")

    with open(args.test_short_path) as f:
        dataset_splits = json.load(f)
    short_test_set = Subset(dataset, [dataset_indices[chain_name] for chain_name in dataset_splits['test'] if chain_name in dataset_indices])

    with open(args.test_single_path) as f:
        dataset_splits = json.load(f)
    sc_test_set = Subset(dataset, [dataset_indices[chain_name] for chain_name in dataset_splits['test'] if chain_name in dataset_indices])
    
    print(f" Test short: {len(short_test_set)} Test sc : {len(sc_test_set)}")

    ## Set training models 
    model = ProteinMPNN(ca_only=args.ca_only, num_letters=21, node_features=args.hidden_dim, edge_features=args.hidden_dim, hidden_dim=args.hidden_dim, num_encoder_layers=args.num_layers, num_decoder_layers=args.num_layers, augment_eps=args.backbone_noise, k_neighbors=args.num_neighbors)
    model.to(device)
    print("Trainable parameters :", sum(p.numel() for p in model.parameters()))
    
    if args.use_pretrained_weights:
        model.load_state_dict(torch.load(args.use_pretrained_weights ,map_location=device)['model_state_dict'])

    if args.resume:
        checkpoint = torch.load(args.previous_checkpoint)
        total_step = checkpoint['step'] #write total_step from the checkpoint
        epoch = checkpoint['epoch'] #write epoch from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        total_step = 0
        epoch = 0

    optimizer = get_std_opt(model.parameters(), args.hidden_dim, total_step)
    if args.resume:
        optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        

    ## Start training
    for e in range(args.epoch):
        t0 = time.time()
        e = epoch + e
        model.train()
        train_sum, train_weights = 0., 0.
        train_pbar = tqdm(loader_train)
        for _, batch in enumerate(train_pbar):
            start_batch = time.time()
            X, S, mask, _, chain_M, chain_encoding_all, _, _, _, _, chain_M_pos, _, residue_idx, _, _, _, _, _, _, _ = CATH_tied_featurize(batch, device, chain_id_dict)
            elapsed_featurize = time.time() - start_batch
            optimizer.zero_grad()
            mask_for_loss = mask*chain_M
            randn_1 = torch.randn(chain_M.shape, device=X.device)
            if args.model == 'MPNN':
                with torch.cuda.amp.autocast():
                    log_probs,_ = model(X, S, mask, chain_M, residue_idx, chain_encoding_all,randn_1)
                    _, loss_av_smoothed = graph_loss_smoothed(S, log_probs, mask_for_loss)
            elif args.model == 'MPNN_CMLM':
                _, mask_ = s_encoder_cmlm_mask(S,mask) # tokens 그대로, mask_ S_masked 
                with torch.cuda.amp.autocast():
                    log_probs,_ = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
                    newS = torch.masked_select(S, mask_.bool())
                    newP = log_probs[mask_.bool(),:]
                    loss, loss_av_smoothed = graph_loss_smoothed(newS, newP, mask_)

            scaler.scale(loss_av_smoothed).backward()  
            if args.gradient_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

            scaler.step(optimizer)
            scaler.update()
        
            loss, _ = loss_nll(S, log_probs, mask_for_loss)
            train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
            train_weights += torch.sum(mask_for_loss).cpu().data.numpy()
            writer.add_scalar('Train loss', loss_av_smoothed.item(), total_step)
            total_step += 1

        t1 = time.time()

        model.eval()
        with torch.no_grad():
            validation_sum, validation_weights = 0., 0.
            valid_pbar = tqdm(loader_valid)
            for _, batch in enumerate(valid_pbar):
                X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = CATH_tied_featurize(batch, device, chain_id_dict)
                randn_1 = torch.randn(chain_M.shape, device=X.device)
                log_probs,_ = model(X, S, mask, chain_M, residue_idx, chain_encoding_all,randn_1)
                mask_for_loss = mask*chain_M
                loss, loss_av = loss_nll(S, log_probs, mask_for_loss)
                
                validation_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                validation_weights += torch.sum(mask_for_loss).cpu().data.numpy()
        
        with torch.no_grad():
            test_sum, test_weights = 0., 0.
            test_pbar = tqdm(loader_test)
            for _, batch in enumerate(test_pbar): 
                X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = CATH_tied_featurize(batch, device, chain_id_dict)
                randn_1 = torch.randn(chain_M.shape, device=X.device)
                log_probs,_ = model(X, S, mask, chain_M, residue_idx, chain_encoding_all,randn_1)
                mask_for_loss = mask*chain_M
                loss, loss_av = loss_nll(S, log_probs, mask_for_loss)
                
                test_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                test_weights += torch.sum(mask_for_loss).cpu().data.numpy()

        with torch.no_grad():
            short_test_sum, short_test_weights = 0., 0.
            short_test_recovery = []
            test_pbar = tqdm(short_test_set)
            for ix, protein in enumerate(test_pbar):
                batch_clones = [copy.deepcopy(protein) for i in range(1)]
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

        with torch.no_grad():
            sc_test_sum, sc_test_weights = 0., 0.
            sc_test_recovery = []
            test_pbar = tqdm(sc_test_set)
            for ix, protein in enumerate(test_pbar):
                batch_clones = [copy.deepcopy(protein) for i in range(1)]
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

        train_loss = train_sum / train_weights
        train_perplexity = np.exp(train_loss)
        validation_loss = validation_sum / validation_weights
        validation_perplexity = np.exp(validation_loss)
        test_loss = test_sum / test_weights
        test_perplexity = np.exp(test_loss)

        short_test_loss = short_test_sum / short_test_weights
        short_test_perplexity = np.exp(short_test_loss)
        short_recovery = np.median(short_test_recovery)
        sc_test_loss = sc_test_sum / sc_test_weights
        sc_test_perplexity = np.exp(sc_test_loss)
        sc_recovery = np.median(sc_test_recovery)

        writer.add_scalar('Train perplexity', train_perplexity, e)
        writer.add_scalar('Validation perplexity', validation_perplexity, e)
        writer.add_scalar('Test perplexity', test_perplexity, e)
        writer.add_scalar('Test Short perplexity', short_test_perplexity, e)
        writer.add_scalar('Test Short median recovery', short_recovery, e)
        writer.add_scalar('Test Single Chain perplexity', sc_test_perplexity, e)
        writer.add_scalar('Single median recovery', sc_recovery, e)
        output = "Epoch : {} Train perplexity : {:.3f} Valid perplexity : {:.3f} Test perplexity : {:.3f} \
                  Short perplexity {:.3f} Short median recovery {:.3f} Single perplexity {:.3f} \
                  Single median reocvery {:.3f}".format(e, train_perplexity, validation_perplexity,test_perplexity,short_test_perplexity,short_recovery, sc_test_perplexity,sc_recovery)

        print(output)
        with open(os.path.join(base_folder,'log.txt'), 'a') as f:
            f.write(output+'\n')

        checkpoint_filename = base_folder+'model_weights/epoch{}.pt'.format(e+1)
        torch.save({
                'epoch': e+1,
                'step': total_step,
                'num_edges' : args.num_neighbors,
                'noise_level': args.backbone_noise, 
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.optimizer.state_dict(),
                }, checkpoint_filename)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--model", type=str, default="MPNN", help="Select base models")
    argparser.add_argument("--ca_only", action="store_true", default=False, help="Parse CA-only structures and use CA-only models (default: false)") 
    argparser.add_argument("--resume", type=bool, default=False, help="resume training")
    argparser.add_argument("--previous_checkpoint", type=str, default="", help="path for previous model weights, e.g. file.pt")
    argparser.add_argument("--use_pretrained_weights", type=str, default="./weights/vanilla_model_weights/v_48_020.pt", help="pretraIned MPNN weights")
    argparser.add_argument("--epoch", type=int, default=100, help="training epochs")
    argparser.add_argument("--batch_size", type=int, default=6000, help="Batch size; can set higher for titan, quadro GPUs, reduce this if running out of GPU memory")
    argparser.add_argument("--max_length", type=int, default=200000, help="Max sequence length")
    argparser.add_argument("--gradient_norm", type=float, default=0.0, help="gradient norm")
    argparser.add_argument("--mixed_precision", type=bool, default=True, help="train with mixed precision")
    argparser.add_argument("--seed", type=int, default=0, help="If set to 0 then a random seed will be picked;")

    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_layers", type=int, default=3, help="number of encoder layers") 
    argparser.add_argument("--num_neighbors", type=int, default=48, help="number of neighbors for the sparse graph")   
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.0, help="amount of noise added to backbone during training") 
    
    argparser.add_argument("--out_folder", type=str, default='/data/project/rw/mpnn_results/MPNN/', help="Path to a folder to output sequences, e.g. /home/out/")
    argparser.add_argument("--jsonl_path", type=str,default='/data/project/rw/cath4.2/chain_set.jsonl',help="Path to parsed pdb into jsonl")
    argparser.add_argument("--file_splits", type=str, default='/data/project/rw/cath4.2/chain_set_splits.json', help='Path to train/valid/test split info')
    argparser.add_argument("--test_short_path", type=str, default="/data/project/rw/cath4.2/test_split_L100.json", help="Path to Short test split")
    argparser.add_argument("--test_single_path", type=str, default="/data/project/rw/cath4.2/test_split_sc.json", help="Path to Single test split")
    argparser.add_argument("--chain_id_jsonl",type=str, default='', help="Path to a dictionary specifying which chains need to be designed and which ones are fixed, if not specied all chains will be designed.")
 
    args = argparser.parse_args()    
    main(args)
