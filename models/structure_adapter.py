## THis contains Structure Adapter
import torch 
import torch.nn as nn
import math
import torch.nn.functional as F
from models.customize_esm.rotary_embedding import RotaryEmbedding

class StructureAdapter(nn.Module):

    def __init__(self, args):
        self.args = args
        super(StructureAdapter, self).__init__()

        # bottle neck layers. This is mimic "Parameter-Efficient Transfer Learning for NLP" bottleneck
        self.ffn_down_project = nn.Linear(1280, 640)
        self.ffn_activation = nn.GELU()
        self.ffn_up_project = nn.Linear(640,1280)

        # Multi Head Cross Attention modules - takes structure representation as key and value and plm representation as query
        self.attn = CrossMultiheadAttention(input_dim=args.hidden_dim, embed_dim=args.embed_dim, num_heads=args.num_heads)
        self.LN = torch.nn.LayerNorm(1280)

        self._reset_ffn_parameters()

    def _reset_ffn_parameters(self):
        nn.init.xavier_uniform_(self.ffn_down_project.weight)
        nn.init.xavier_uniform_(self.ffn_up_project.weight)
        self.ffn_down_project.bias.data.fill_(0)
        self.ffn_up_project.bias.data.fill_(0)

    def forward(self,s_repre, plm_repre):
        out,attention = self.attn(s_repre,plm_repre)
        out = out + plm_repre
        out = self.LN(out) 
        # BottleNeck FFN 
        residual = out
        out = self.ffn_down_project(out)
        out = self.ffn_activation(out)
        out = self.ffn_up_project(out)
        out = out + residual
        return out, attention

class CrossMultiheadAttention(nn.Module):

    def __init__(self, input_dim=128, embed_dim=768, num_heads=12):
        super(CrossMultiheadAttention, self).__init__()
        '''
            논문에서 다르게, 실제 github issue 질문에
            저자는 query - pLM          
                 key, value - structure 로 사용했다고 밝힘
        '''
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = int(self.embed_dim / self.num_heads)
        self.input_dim = input_dim

        self.w_q = nn.Linear(1280,self.embed_dim) # 1280 -> 768 from 
        self.w_k = nn.Linear(128, self.embed_dim) # 128 -> 768
        self.w_v = nn.Linear(128, self.embed_dim) # 128 -> 768

        self.o_proj = nn.Linear(self.embed_dim, 1280)

        ## Adding ROPE 
        self.rot_emb = RotaryEmbedding(dim=self.head_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight) 
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        
        self.w_q.bias.data.fill_(0)
        self.w_k.bias.data.fill_(0)
        self.w_v.bias.data.fill_(0)
        self.o_proj.bias.data.fill_(0)
        
    
    def _scaled_dot_product(self,q,k,v, mask=None):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention

    def forward(self, s_repre, plm_repre, mask=None, return_attention=False): 
        '''
            Structure representation used for K and V
            PLM representation used for Q 
        '''
        B, seqs = s_repre.size()[:2]
        q = self.w_q(plm_repre).reshape(B,seqs,self.num_heads,self.head_dim).permute(0,2,1,3)
        k = self.w_k(s_repre).reshape(B,seqs,self.num_heads,self.head_dim).permute(0,2,1,3)
        v = self.w_v(s_repre).reshape(B,seqs,self.num_heads,self.head_dim).permute(0,2,1,3)

        q, k = self.rot_emb(q, k)
        values, attention = self._scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0,2,1,3) # [SeqLen, Head, Dims]
        values = values.reshape(B,seqs, self.embed_dim)
        o = self.o_proj(values)

        return o, attention
