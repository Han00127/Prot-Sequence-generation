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
    from torch.utils.data.dataset import random_split, Subset
    import copy
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    import os.path

    from models.mpnn_utils import loss_nll, loss_smoothed, get_std_opt,calculate_recovery 
    from models.mpnn_utils import StructureDataset, StructureLoader
    from models.lmdesign import LMDesign
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
    base_folder = time.strftime(args.out_folder, time.localtime())

    if base_folder[-1] != '/':
        base_folder += '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolders = ['model_weights', 'train_logs']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)
    
    writer = SummaryWriter(os.path.join(args.out_folder, "train_logs"))
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    
    #######################
    ## Processing Datasets  
    #######################
    dataset = StructureDataset(args.jsonl_path, truncate=None, max_length=args.max_length)
    chain_id_dict = None
    
    ## make CATH set
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

    with open(args.test_short_path) as f:
        dataset_splits = json.load(f)
    short_test_set = Subset(dataset, [dataset_indices[chain_name] for chain_name in dataset_splits['test'] if chain_name in dataset_indices])

    with open(args.test_single_path) as f:
        dataset_splits = json.load(f)
    sc_test_set = Subset(dataset, [dataset_indices[chain_name] for chain_name in dataset_splits['test'] if chain_name in dataset_indices])
    
    
    print(f"Test all : {len(train_set)}, Valida : {len(validation_set)} Test short: {len(short_test_set)} Test sc : {len(sc_test_set)}")

    loader_train = StructureLoader(train_set, batch_size=args.batch_size)
    loader_valid = StructureLoader(validation_set, batch_size=args.batch_size)
    loader_test = StructureLoader(test_set, batch_size=args.batch_size)
    ##############################
    ## load models     checkpoint_path="/data/private/ProteinMPNN-main/vanilla_model_weights/v_48_020.pt"
    ## reproduce 
    #############################
    # checkpoint_path = "/data/project/rw/mpnn_results/reproduce2/model_weights/epoch97.pt"
    # checkpoint = torch.load(checkpoint_path, map_location=device) 
    # noise_level_print = checkpoint['noise_level']
    # num_edges = checkpoint['num_edges']
    model = LMDesign(args, device)
    model.to(device)
    print("Total parameters ", sum(p.numel() for p in model.parameters()))
    print("Total trainable parameters ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    if args.resume:
        checkpoint = torch.load(args.previous_checkpoint,map_location=device)
        total_step = checkpoint['step'] #write total_step from the checkpoint
        epoch = checkpoint['epoch'] #write epoch from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        total_step = 0
        epoch = 0
    optimizer = get_std_opt(model.parameters(), 128, total_step)
    if args.resume:
        optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    import time 
    for e in range(args.epoch):
        e += epoch
        model.train()
        train_sum, train_weights = 0., 0. 
        train_pbar = tqdm(loader_train)
        for _, batch in enumerate(train_pbar):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                results = model(batch)
                loss, loss_av_smoothed = loss_smoothed(results['batch_tokens'][:,1:-1],results['log_probs'],results['c_mask'][:,1:-1])
            scaler.scale(loss_av_smoothed).backward()
            if args.gradient_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)
            scaler.step(optimizer)
            scaler.update()
            
            # masked perplexity 
            loss, loss_av = loss_nll(results['batch_tokens'][:,1:-1], results['log_probs'], results['c_mask'][:,1:-1])
            masked_sum = torch.sum(loss * results['c_mask'][:,1:-1]).cpu().data.numpy()
            masked_weight = torch.sum(results['c_mask'][:,1:-1]).cpu().data.numpy()
            masked_perplexity = np.exp(masked_sum / masked_weight)
            train_sum += masked_sum
            train_weights += masked_weight
            writer.add_scalar('Train loss', loss_av_smoothed.item(), total_step)
            writer.add_scalar('Train perplexity', masked_perplexity.item(), total_step)
            total_step += 1
        
        model.eval()
        with torch.no_grad():
            valid_sum, valid_weights = 0., 0. 
            valid_pbar = tqdm(loader_valid)
            for _, batch in enumerate(valid_pbar):
                results = model(batch)             
                loss, loss_av = loss_nll(results['batch_tokens'][:,1:-1], results['log_probs'], results['mask_for_loss'])
                valid_sum += torch.sum(loss * results['mask_for_loss']).cpu().data.numpy()
                valid_weights += torch.sum(results['mask_for_loss']).cpu().data.numpy()


        with torch.no_grad():
            test_sum, test_weights = 0., 0.
            test_recovery = []
            test_pbar = tqdm(loader_test)
            for _, batch in enumerate(test_pbar): 
                results = model(batch)
                loss, loss_av = loss_nll(results['batch_tokens'][:,1:-1], results['log_probs'], results['mask_for_loss'])
                test_sum += torch.sum(loss * results['mask_for_loss']).cpu().data.numpy()
                test_weights += torch.sum(results['mask_for_loss']).cpu().data.numpy()
                test_recovery.append(calculate_recovery(results['log_probs'],results['batch_tokens'][:,1:-1]))

        with torch.no_grad():
            short_test_sum, short_test_weights = 0., 0.
            short_recovery = []
            test_pbar = tqdm(short_test_set)
            for ix, protein in enumerate(test_pbar):
                batch_clones = [copy.deepcopy(protein) for i in range(1)]
                results = model.iterative_refine_inference(batch_clones, iterations=5)
                loss, loss_av = loss_nll(results['batch_tokens'][:,1:-1], results['log_probs'], results['mask_for_loss'])

                short_test_sum += torch.sum(loss * results['mask_for_loss']).cpu().data.numpy()
                short_test_weights += torch.sum(results['mask_for_loss']).cpu().data.numpy()
                short_recovery.append(calculate_recovery(results['log_probs'],results['batch_tokens'][:,1:-1]))
        
        with torch.no_grad():
            sc_test_sum, sc_test_weights = 0., 0.
            sc_recovery = []
            test_pbar = tqdm(sc_test_set)
            for ix, protein in enumerate(test_pbar):
                batch_clones = [copy.deepcopy(protein) for i in range(1)]
                results = model.iterative_refine_inference(batch_clones, iterations=5)
                loss, loss_av = loss_nll(results['batch_tokens'][:,1:-1], results['log_probs'], results['mask_for_loss'])

                sc_test_sum += torch.sum(loss * results['mask_for_loss']).cpu().data.numpy()
                sc_test_weights += torch.sum(results['mask_for_loss']).cpu().data.numpy()
                sc_recovery.append(calculate_recovery(results['log_probs'],results['batch_tokens'][:,1:-1]))
    
        train_loss = train_sum / train_weights
        train_perplexity = np.exp(train_loss)

        valid_loss = valid_sum / valid_weights
        valid_perplexity = np.exp(valid_loss)

        test_loss = test_sum / test_weights
        test_perplexity = np.exp(test_loss)
        test_recovery = np.median(test_recovery)

        short_test_loss = short_test_sum / short_test_weights
        short_test_perplexity = np.exp(short_test_loss)
        short_recovery = np.median(short_recovery)

        sc_test_loss = sc_test_sum / sc_test_weights
        sc_test_perplexity = np.exp(sc_test_loss)
        sc_recovery = np.median(sc_recovery)

        output = "Epoch {} Train perplexity : {:.4f} Valid perplexity {:.4f} Test perplexity {:.4f} Test median recovery {:.4f} \
                 Shot perplexity : {:.4f} Short median recovery : {:.4f} Single chain perplexity : {:.4f} Single  median recovery: {:.4f} ".format(e,train_perplexity.item(),valid_perplexity.item(), test_perplexity.item(), test_recovery, short_test_perplexity.item(), short_recovery.item(), sc_test_perplexity.item(), sc_recovery.item())
        print(output)
        with open(os.path.join(base_folder,'log.txt'), 'a') as f:
            f.write(output+'\n')

        writer.add_scalar('Train perplexity', train_perplexity.item(), e)
        writer.add_scalar('Valid perplexity', valid_perplexity.item(), e)
        writer.add_scalar('Test perplexity', test_perplexity.item(), e)
        writer.add_scalar('Test median recovery', test_recovery.item(), e)
        writer.add_scalar('Test Short perplexity', short_test_perplexity.item(), e)
        writer.add_scalar('Test Short median recovery', short_recovery.item(), e)
        writer.add_scalar('Test Single Chain perplexity', sc_test_perplexity.item(), e)
        writer.add_scalar('Test Single Chain median recovery', sc_recovery.item(), e)
        checkpoint_filename_last = base_folder+'model_weights/epoch{}.pt'.format(e+1)
        torch.save({
                    'epoch': e+1,
                    'step': total_step,
                    'num_edges' : args.num_neighbors,
                    'plm': args.plm_model,
                    'noise_level': args.backbone_noise,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.optimizer.state_dict(),
                    }, checkpoint_filename_last)
        
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ## Setup parameters 
    argparser.add_argument("--seed", type=int, default=0, help="If set to 0 then a random seed will be picked;")
    argparser.add_argument("--epoch", type=int, default=1, help="training epochs")
    argparser.add_argument("--lr", type=int, default=0, help="training lr")
    argparser.add_argument("--out_folder", type=str, default='/data/project/rw/lmdesign_results/debug', help="Path to a folder to save information, e.g. /home/out/")
    argparser.add_argument("--resume", type=bool, default=False, help="resume training")
    argparser.add_argument("--previous_checkpoint", type=str, default="", help="path for previous model weights, e.g. file.pt")
    
    argparser.add_argument("--jsonl_path", type=str,default='/data/project/rw/cath4.2/chain_set.jsonl',help="Path to a folder with parsed pdb into jsonl")
    argparser.add_argument("--file_splits", type=str, default='/data/project/rw/cath4.2/chain_set_splits.json', help='set train/valid/test info')
    argparser.add_argument("--test_single_path", type=str, default="/data/project/rw/cath4.2/test_split_sc.json", help="Path to Single test split")
    argparser.add_argument("--test_short_path", type=str, default='/data/project/rw/cath4.2/test_split_L100.json', help='Path to a Single test split')
    argparser.add_argument("--batch_size", type=int, default=6000, help="Batch size; can set higher for titan, quadro GPUs, reduce this if running out of GPU memory")
    argparser.add_argument("--max_length", type=int, default=200000, help="Max sequence length")
    argparser.add_argument("--gradient_norm", type=float, default=1.0, help="gradient norm")   

    ## Structure model 
    argparser.add_argument("--structure_model", type=str, default='MPNN', help='Select base structure models')
    argparser.add_argument("--structure_trainable", type=bool, default=False, help='Structure models are frozen')
    argparser.add_argument('--structure_weight', type=str, default='/data/private/ProteinMPNN-main/vanilla_model_weights/v_48_020.pt')
    argparser.add_argument("--hidden_dim", type=int, default=128, help='hidden model dimension')
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers") 
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=48, help="number of neighbors for the sparse graph")   
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.2, help="amount of noise added to backbone during training")   
    argparser.add_argument("--ca_only", action="store_true", default=False, help="Parse CA-only structures and use CA-only models (default: false)")   
    
    ## PLM model 
    argparser.add_argument("--plm_model", type=str, default='ESM1b', help='Select base PLM models')
    argparser.add_argument("--plm_weight", type=str, default="/data/project/rw/ESM_weight/esm1b_t33_650M_UR50S.pt", help="Path to model weights folder;") 
    
    ## Structure Adapter 
    argparser.add_argument("--embed_dim", type=int, default=768, help='Embed dimension of both structure and plm representations')
    argparser.add_argument("--num_heads", type=int, default=12, help='Number of heads used in attention')
    
    args = argparser.parse_args()    
    main(args)   
