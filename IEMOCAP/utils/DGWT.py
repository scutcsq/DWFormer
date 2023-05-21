import torch
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable
from einops import rearrange
import time
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import random
from fairseq.modules.multihead_attention import MultiheadAttention
from .modules import FeedForwardNetwork

class GlobalMHA(nn.Module):
    def __init__(self, dim, head):
        super(GlobalMHA, self).__init__()
        self.softmax2 = nn.Softmax(dim=-1)
        self.mha = MultiheadAttention(embed_dim=dim, num_heads=head)
        
    def forward(self, x, mask):
        '''
        x:data shape,(t, batch, c)

        mask:attention mask,mask window tokens padding, (batch, t)
        '''
        attn_output, attn_output_weights = self.mha(query=x, key=x, value=x,key_padding_mask = mask.bool())
        haltingscore = torch.sum(attn_output_weights, dim=1)
        #calculate important scores
        mask2 = mask * (-1e10)
        haltingscore = self.softmax2(haltingscore + mask2)
        return attn_output, haltingscore
    
class DynamicGlobalWindowTransformer(nn.Module):
    def __init__(self, dim, head, FFNdim) -> None:
        super(DynamicGlobalWindowTransformer, self).__init__()
        self.MHSA = GlobalMHA(dim, head)
        self.FFN = FeedForwardNetwork(dim, FFNdim)
        self.ln1 = nn.LayerNorm(dim, eps=1e-5)
        self.ln2 = nn.LayerNorm(dim, eps=1e-5)

    def forward(self, x, mask):
        x = x.permute([1,0,2])
        residual = x
        x1, attn = self.MHSA(x, mask)
        x = residual + x1
        x = self.ln1(x)
        residual = x
        x2 = self.FFN(x)
        x = self.ln2(residual + x2)
        x = x.permute([1,0,2])
        return x, attn