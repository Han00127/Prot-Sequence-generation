import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.customize_esm.pretrained import load_model_and_alphabet
from models.structure_adapter import StructureAdapter
from models.mpnn_utils import s_seq2plm_seq,CATH_tied_featurize,ProteinMPNN,cmlm_mask
import copy 


class LMDesign(nn.Module): ## REAL LMDesign models
    def __init__(self, args, device):
        super(LMDesign, self).__init__()
        self.args = args
        self.device = device
        
        ########################
        ## Load structure models
        ## This can be further devolped to accomodate more structure models
        ######################## 
        if args.structure_model == 'MPNN':
            if args.structure_weight: ## Pretrained mpnn+cmlm 
                weights = torch.load(args.structure_weight)
                self.s_model = ProteinMPNN(ca_only=args.ca_only, num_letters=21, node_features=args.hidden_dim, edge_features=args.hidden_dim, 
                                        hidden_dim=args.hidden_dim, num_encoder_layers=args.num_encoder_layers, num_decoder_layers=args.num_decoder_layers,
                                        augment_eps=args.backbone_noise, k_neighbors=args.num_neighbors)

                self.s_model.load_state_dict(weights['model_state_dict'])
            else:
                self.s_model = ProteinMPNN(ca_only=args.ca_only, num_letters=21, node_features=args.hidden_dim, edge_features=args.hidden_dim, 
                                        hidden_dim=args.hidden_dim, num_encoder_layers=args.num_encoder_layers, num_decoder_layers=args.num_decoder_layers,
                                        augment_eps=args.backbone_noise, k_neighbors=args.num_neighbors)
        self.s_model.to(self.device)

        ########################
        ## Load structure models
        ## This can be further devolped to accomodate more PLM  
        ######################## 
        
        self.plm_model, self.alphabet = load_model_and_alphabet(args.plm_weight)
        # self.plm_model, self.alphabet = load_model_and_alphabet('/data/project/rw/ESM_weight/esm1b_t33_650M_UR50S.pt')
        self.tokenizer = self.alphabet.get_batch_converter()
        self.plm_model.to(self.device)

        ########################
        ## Load structure adapter and LM Head from ESM!!
        ## This can be further devolped to accomodate more PLM  
        ######################## 
        self.structure_adapter = StructureAdapter(args)
        self.structure_adapter.to(self.device)
        
        self.head = copy.deepcopy(self.plm_model.lm_head)

        # ## FREEEZE MODELS 
        self.freeze()

        ## Print parameters in the models
        pifold_params = sum(p.numel() for p in self.s_model.parameters())
        pifold_trainable_params = sum(p.numel() for p in self.s_model.parameters() if p.requires_grad)
        print("##########################################################")
        print(f"Entire parameters of structure model {pifold_params}")
        print(f"trainable parameters in structure {pifold_trainable_params}")
        print("##########################################################")
        plm_params = sum(p.numel() for p in self.plm_model.parameters())
        plm_trainable_params = sum(p.numel() for p in self.plm_model.parameters() if p.requires_grad)
        print(f"Entire parameters of pLM {plm_params}")
        print(f"trainable parameters in pLM {plm_trainable_params}")
        print("##########################################################")
        sa_params = sum(p.numel() for p in self.structure_adapter.parameters())
        sa_trainable_params = sum(p.numel() for p in self.structure_adapter.parameters() if p.requires_grad)
        print(f"Entire parameters of sa {sa_params}")
        print(f"trainable parameters in sa {sa_trainable_params}")
        print("##########################################################")
        lm_head_params = sum(p.numel() for p in self.head.parameters())
        lm_head_trainable_params = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        print(f"Entire parameters of lm_head {lm_head_params}")
        print(f"trainable parameters in lm_head {lm_head_trainable_params}")
        print("##########################################################")
    
    def forward(self, inputs):
        '''
            inputs == batch 
            inputs = [name, seq, coords,...]
        '''
        X, S, mask, _, chain_M, chain_encoding_all, _, _, _, _, chain_M_pos, _, residue_idx, _, _, _, _, _, _, _ = CATH_tied_featurize(inputs, self.device, None)
        randn_1 = torch.randn(chain_M.shape, device=X.device)
        s_log_probs, s_repre = self.s_model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)    
        mask_for_loss = mask*chain_M*chain_M_pos

        ## seq 변환 for PLM
        seqs = s_seq2plm_seq(S, mask)
        
        ## Reprocess data for PLM ## ESM LIKE TOKENS 
        data = []
        for i, seq in enumerate(seqs):
            name = "batch_" + str(i)
            data.append((name, seq))
        batch_tokens = self.tokenizer(data)[-1]
        batch_tokens = batch_tokens.to(self.device)

        cmlm_tokens, c_mask = cmlm_mask(batch_tokens,mask) # bert style mask input data & corresponding mask
        cmlm_tokens, c_mask = cmlm_tokens.clone().to(self.device), c_mask.clone().to(self.device)

        plm_repre = self.plm_model(cmlm_tokens)
        out,attention = self.structure_adapter(s_repre, plm_repre[:,1:-1,:])
        out = out + plm_repre[:,1:-1,:]
        x = self.head(out)
        result = {'log_probs': x , ## Prediction
                  'batch_tokens': batch_tokens, ## GT
                  'c_mask': c_mask, 
                  'mask_for_loss' : mask_for_loss, ## entire GT mask
                  'attentions': attention
                  }
        return result
    
    def freeze(self):
        if not self.args.structure_trainable: ## False -> True
            for k,v in self.s_model.named_parameters():
                v.requires_grad=False
        else:
            for k,v in self.s_model.named_parameters():
                v.requires_grad=True
        for k,v in self.plm_model.named_parameters():
            v.requires_grad=False
        for k,v in self.head.named_parameters():
            v.requires_grad=False

    def iterative_refine_inference(self, inputs, iterations, temp=1.0):
        
        X, S, mask, _, chain_M, chain_encoding_all, _, _, _, _, chain_M_pos, _, residue_idx, _, _, _, _, _, _, _ = CATH_tied_featurize(inputs, self.device, None)
        GT_seqs = s_seq2plm_seq(S, mask)
        GT_data = []
        for i, seq in enumerate(GT_seqs):
            name = "batch_" + str(i)
            GT_data.append((name, seq))
        GT_tokens = self.tokenizer(GT_data)[-1]
        GT_tokens = GT_tokens.to(self.device) 

        randn_1 = torch.randn(chain_M.shape, device=X.device)
        log_probs, s_repre = self.s_model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)    
        mask_for_loss = mask*chain_M*chain_M_pos

        ############################################
        ## iteration 만큼해서 refinement temperature 가능 
        ############################################
        for i in range(iterations-1): 
            s_seqs = torch.argmax(log_probs, dim=-1)
            
            if i == 0: # from Structure encoder 
                s_seqs = s_seq2plm_seq(s_seqs, mask)
                data = []
                for i, seq in enumerate(s_seqs):
                    name = "batch_" + str(i)
                    data.append((name, seq))
                batch_tokens = self.tokenizer(data)[-1]
                batch_tokens = batch_tokens.to(self.device)
            else:
                aa = F.softmax(log_probs, dim=-1) # make probabilities of each class  
                predicted_probs, tokens = torch.topk(aa, 1) # gives top 1 probabilities 
                s_seqs[:,torch.where(torch.lt(predicted_probs, 0.5))[1]] = 32 # if it is not cofident, mask
                batch_tokens[:,1:-1] = s_seqs

            plm_repre = self.plm_model(batch_tokens)
            out, attention = self.structure_adapter(s_repre, plm_repre[:,1:-1,:])
            out = out + plm_repre[:,1:-1,:]
            log_probs = self.head(out,temp)
        result = {'log_probs': log_probs , ## Prediction
                  'batch_tokens': GT_tokens, ## GT
                  'c_mask': None, 
                  'mask_for_loss' : mask_for_loss, ## entire GT mask 
                  'attentions' : attention  
                  }
        return result
