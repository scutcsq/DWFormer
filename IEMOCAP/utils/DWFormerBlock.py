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
from .DLWT import DynamicLocalWindowTransformer
from .DGWT import DynamicGlobalWindowTransformer
from .modules import arbitrary_mask_v2

class DWFormerBlock(nn.Module):
    def __init__(self, feadim, n_head, ffndim):
        super(DWFormerBlock, self).__init__()
        self.DLWT = DynamicLocalWindowTransformer(feadim, n_head, ffndim)
        self.DGWT = DynamicGlobalWindowTransformer(feadim, n_head, ffndim)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.ln1 = nn.LayerNorm(feadim,eps = 1e-5)
        self.ln2 = nn.LayerNorm(feadim,eps = 1e-5)
    def forward(self, x, mask, haltingscore):
        #---Dynamic Local Window Splitting Module---

        attention_mask,mappingmask,attention_mask2,thresholds,lengths,wise = self.mask_generation_function(haltingscore, x, mask,threshold = 0.5, lambdas= 0.85)
        
        #---Dynamic Local Window Transformer Module---

        x = self.ln1(x)
        local_x, local_score = self.DLWT(x, attention_mask,mappingmask)

        #---Dynamic Global Window Transformer Module
        local_x = local_x * wise.unsqueeze(-1)
        pre_global_x = self.beforeDGWT(local_x,mappingmask,local_score,lengths)
        pre_global_x = self.ln2(pre_global_x)
        global_x, hh2 = self.DGWT(pre_global_x, attention_mask2)

        #Summation
        
        data, attn = self.afterDGWT(local_x,global_x,local_score,hh2,mappingmask)
        attn = self.softmax(attn)
        
        return data, thresholds, attn
    def mask_generation_function(self,haltingscore, x, mask,threshold=0.5, lambdas=0.85):
        '''
        input:
            haltingscore:(batch,token_length)
            x:(batch,token_length,feadim)
            mask:(batch,token_length)
            threshold
            lambdas
        output:
            attention_mask:(batch )
            outdata(windownum,max_token_length,feadim)
            outmask(windownum,max_token_length)
            calwindow(batch)
            calwindowlength(windownum)
            attention_mask,mappingmask,attention_mask2,thresholds,totallength,wise
        '''
        batch, token_length, fea = x.shape
        zero_mat = torch.zeros_like(haltingscore)
        one_mat = torch.ones_like(haltingscore)

        # calculate data length
        
        mask11 = one_mat - mask
        mask1 = torch.sum(mask11, dim=-1)  # real data length
        token_length1 = torch.ones((batch), device=x.device) * token_length

        # threshold method

        med = mask1 * threshold + token_length1 - mask1 
        thresholds, _ = torch.sort(haltingscore, dim=-1)
        med = list(map(int, med.cpu().data.numpy()))
        thresholds1 = thresholds[:, med]
        thresholds = torch.diag(thresholds1)

        # divide window,and find the begin and the end of the windows by first order difference
        # we use convolution operation to prevent the individual token from forming independent window 

        x1 = torch.where(haltingscore >= thresholds.unsqueeze(1), one_mat, zero_mat)  # (batch,token_length)#bigger than the threshold is set as 1, while set as 0.
        wise = torch.where(haltingscore>= thresholds.unsqueeze(1), one_mat, one_mat * lambdas) # those token_lengths in weak emotional information places are multiplied by lambda = 0.85
        x2_1 = x1[:, 1:]  # (batch,token_length)
        x2_2 = 1 - x1[:, -1]
        x2 = torch.cat([x2_1, x2_2.unsqueeze(-1)], axis=1)
        x3 = x2 - x1  # (batch,token_length)#一阶差分得到
        x3 = torch.where(x3 == -1, one_mat, x3)
        x4 = x3.view(1, 1, batch, token_length)
        b = Variable(torch.ones((1, 1, 1, 2), device=x.device))
        x4 = F.conv1d(x4, b, padding=(0, 1)).view(batch, -1)
        x4 = x4[:, :-1]
        x3 = torch.where(x4 == 2, zero_mat, x3)
        x3[:,-1] = 1

        # utilize the begin and the end of the windows, split the window 
        
        zerodim = torch.zeros((1), device=x.device)
        onedim = torch.ones((1), device=x.device) * (-1)
        result = torch.where(x3 != 0)  # 返回切割位置
        onedim2 = torch.ones((1), device=x.device) * len(result[0])
        result2 = torch.cat((onedim, result[1][:-1]))  # 返回起点位置,但对batch切换数据后得改变起点重置为0
        result3 = torch.cat((result[0][1:], zerodim))
        result4 = result3 - result[0]
        result4 = torch.cat((zerodim, result4[:-1]))
        zero = torch.ones_like(result4) * (-1)
        result2 = torch.where(result4 == 1, zero, result2)
        length = result[1] - result2  # calculate each length of the windows.
        result6 = torch.where(result4 == 1)
        result7 = torch.cat([result6[0], onedim2])
        result8 = torch.cat([zerodim, result6[0]])
        calwindow = result7 - result8
        result2 += 1

        # calculate the begin and the end of the windows

        maxwindow = int(max(calwindow)) #maximum length of the window
        mappingmask = torch.zeros((batch,int(maxwindow),token_length),device = x.device)#batch,maxwindownum,token_length
        attention_mask = torch.ones((batch,token_length,token_length),device = x.device)
        calwindow1 = list(map(int, calwindow.cpu().data.numpy()))
        beginplace = torch.split(result2,calwindow1,dim = 0)
        endplace = torch.split(result[1],calwindow1,dim = 0)
        beginplace = rnn.pad_sequence(beginplace,batch_first = True)#calculate the begin places of the window
        endplace = rnn.pad_sequence(endplace,batch_first = True)#calculate the end places of the window

        # generate the attention mask for Local Window Tranformer Block

        a1 = torch.arange(1,maxwindow+1).unsqueeze(0).unsqueeze(-1).to(x.device)
        a1 = a1.expand(batch,-1,token_length)
        a2 = calwindow.unsqueeze(-1).unsqueeze(-1)
        a2 = a2.expand(-1,maxwindow,token_length)
        zeromat = torch.zeros_like(a2)
        mappingmask = arbitrary_mask_v2(beginplace,endplace,token_length,reverse = True,return_bool= False)
        mappingmask = torch.where(a1<=a2,mappingmask,zeromat)
        mappingmask_t = mappingmask.transpose(1,2)
        attention_mask = torch.matmul(mappingmask_t,mappingmask)
        attention_mask = 1- attention_mask

        # generate the attention mask for Global Window Transformer Block

        highattn1 = torch.arange(1,maxwindow+1).unsqueeze(0).to(x.device)
        highattn1 = highattn1.expand(batch,-1)
        calwindow2= calwindow.unsqueeze(-1)
        calwindow2 = calwindow2.expand(-1,maxwindow)
        attention_mask2 = torch.zeros((batch,maxwindow),device = x.device)
        one_mat = torch.ones((batch,maxwindow),device = x.device)
        attention_mask2 = torch.where(calwindow2>=highattn1,attention_mask2,one_mat)
        totallength = endplace - beginplace
        onemat = torch.ones_like(totallength)
        totallength = torch.where(totallength ==0,onemat,totallength)

        # return results

        return attention_mask,mappingmask,attention_mask2,thresholds,totallength,wise
    def beforeDGWT(self,inputdata, mappingmask,attn1,lengths):
        '''
        Transform feature into window sequence.
        input:
        inputdata:(b,t,fea)
        mappingmask(b,maxwindownum,t)
        attn1(batch,t)
        output:
        outdata:(batch,maxwindownum,fea)
        '''

        data = inputdata * attn1.unsqueeze(-1)
        outdata = torch.matmul(mappingmask,data)
        return outdata

    def afterDGWT(self,inputdata1,inputdata2,attn1,attn2,mappingmask):
        '''
        Sum the DLWT features and DGWT features by upsampling.
        input:inputdata1:(b,t,fea)
            inputdata2:(b,maxwindownum,fea)
            attn1(b,t)
            attn2(b,maxwindownum)
            mappingmask(b,maxwindownum,t)
        output:
            outdata(b,t,fea)
            outattn(b,t)
        '''
        mappingmask = mappingmask.transpose(-1,-2)
        inputdata2 = torch.matmul(mappingmask,inputdata2)
        attn2 = torch.matmul(mappingmask,attn2.unsqueeze(-1)).squeeze(-1)
        outattn = attn1 * attn2
        outdata = inputdata1  + inputdata2
        return outdata,outattn