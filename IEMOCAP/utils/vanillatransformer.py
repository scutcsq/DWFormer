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
# import modules.FeedForwardNetwork as FeedForwardNetwork


class vanilla_transformer_block(nn.Module):
    def __init__(self, dim, head, FFNdim) -> None:
        super(vanilla_transformer_block, self).__init__()
        self.mha = MultiheadAttention(embed_dim=dim, num_heads=head)
        self.FFN = FeedForwardNetwork(dim, FFNdim)
        self.ln1 = nn.LayerNorm(dim, eps=1e-5)
        self.ln2 = nn.LayerNorm(dim, eps=1e-5)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, x, haltingscore):
        '''
        x: (batch, t, c)
        haltingscore:(batch, t)
        '''
        x = x.permute([1,0,2])
        residual = x
        x1, attn = self.mha(key = x, value = x, query = x)
        x = residual + x1
        x = self.ln1(x)
        residual = x
        x2 = self.FFN(x)
        x = self.ln2(residual + x2)
        x = x.permute([1,0,2])
        attn = torch.sum(attn,dim = 1)
        attn = self.softmax(attn)
        haltingscore += attn
        return x, haltingscore, attn