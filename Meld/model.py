import torch
import torch.nn as nn
import random
from utils.vanillatransformer import vanilla_transformer_block

from utils.modules import PositionalEncoding
from utils.DWFormerBlock import DWFormerBlock


class classifier(nn.Module):
    def __init__(self, feadim, classnum):
        super(classifier, self).__init__()
        self.fc1 = nn.Linear(feadim, feadim // 2)
        self.fc2 = nn.Linear(feadim // 2, feadim // 4)
        self.fc3 = nn.Linear(feadim // 4, classnum)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        x = x.permute([0,2,1])
        x = self.avgpool(x).squeeze(-1)
        x = self.fc1(x)
        x = self.dropout(self.relu(x))
        x = self.fc2(x)
        x = self.dropout(self.relu(x))
        out = self.fc3(x)
        return out


class DWFormer(nn.Module):
    def __init__(self, feadim, n_head, FFNdim, classnum):
        super(DWFormer, self).__init__()
        '''
        feadim:input dimension of the feature
        n_head:numbers of the attention head
        FFNdim:dimension of FeedForward Network
        classnum: numbers of emotion
        '''
        self.or1 = vanilla_transformer_block(feadim, n_head, FFNdim)
        self.dt1 = DWFormerBlock(feadim, n_head, FFNdim)
        self.dt2 = DWFormerBlock(feadim, n_head, FFNdim)
        self.dt3 = DWFormerBlock(feadim, n_head, FFNdim)
        self.classifier = classifier(feadim, classnum)
        self.PE = PositionalEncoding(feadim)
        self.ln1 = nn.LayerNorm(feadim,eps = 1e-5)
        self.ln2 = nn.LayerNorm(feadim,eps = 1e-5)
        self.ln3 = nn.LayerNorm(feadim,eps = 1e-5)
        self.ln4 = nn.LayerNorm(feadim,eps = 1e-5)
        self._reset_parameters()
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def forward(self, x, x_mask):
        '''
        x:input data, (b, t, c)
        x_mask:
        '''
        batch, times, _ = x.shape
        haltingscore = torch.zeros((batch, times), device=x.device)

        # propose random attention scores instead of vanilla transformer
        # randomdata = self.getNumList(0,323,120)
        # haltingscore[:,randomdata] += 1e-10
        # Vanilla Transformer Block

        x = self.ln1(x)
        x1,_,attn = self.or1(x, haltingscore)

        # DWFormer Block

        x2,thresholds1,attn11 = self.dt1(x1, x_mask, attn)
        x3 = self.ln2(x2)
        x4,thresholds2,attn12 = self.dt2(x3, x_mask, attn11)
        # x5 = self.ln3(x4)
        # x6,thresholds3,attn13 = self.dt3(x5, x_mask, attn12)

        # Classifier

        out = self.classifier(x4)

        return out
    
        # attn4 = torch.cat([attn.unsqueeze(0).data,attn11.unsqueeze(0).data,attn12.unsqueeze(0).data,attn13.unsqueeze(0)],dim = 0)#ori结果
        # thresholds = torch.cat([thresholds1.unsqueeze(0).data,thresholds2.unsqueeze(0).data,thresholds3.unsqueeze(0)],dim = 0)#分窗
        # return out,attn4,thresholds

    def getNuthresholdsist(self,start, end, n):
        '''
        generate random init
        '''
        numsArray = set()
        while len(numsArray) < n:
            numsArray.add(random.randint(start, end))

        return list(numsArray)