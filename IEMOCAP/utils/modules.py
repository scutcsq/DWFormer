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

def arbitrary_mask_v2(start_index: torch.Tensor, end_index: torch.Tensor, len_out: int, reverse: bool = False, return_bool: bool = True):
    ''''
    generate attention mask matrix
    Args:
        start_index: (b t), the first true value
        end_index: (b t), the last true value
        len_out: length of mask
        reverse: reverse the output mask (Default: False)
        return_bool: if True, return torch.BoolTensor, otherwise torch.FloatTensor
    Returns:
        mask: (b t len_out), the padded values are marked as True if reverse is False
    '''
    b, t = start_index.shape
    start_index = start_index.unsqueeze(dim=-1)# b t 1
    end_index = end_index.unsqueeze(dim=-1)

    mask = torch.arange(0, len_out, device=start_index.device).unsqueeze(dim=0).unsqueeze(dim=1).expand(b, t, -1)
    mask = ((mask - start_index) < 0) | ((mask - end_index) > 0)
    if reverse:
        mask = ~mask
    if not return_bool:
        mask = mask.float()
    return mask

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return x
    

class FeedForwardNetwork(nn.Module):
    "Implement the FFN function"
    def __init__(self, dim, FFNdim,dropout = 0.3) -> None:
        super(FeedForwardNetwork, self).__init__()
        self.FFN1 = nn.Linear(dim, FFNdim)
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.FFN2 = nn.Linear(FFNdim, dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.FFN1(x)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        x1 = self.FFN2(x1)
        x1 = self.dropout2(x1)
        return x1