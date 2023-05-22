import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import fairseq
import lmdb
import shutil
import math
from WavLM import WavLM,WavLMConfig
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False
'''
目标:将一整段很长的语音进行切割，只保留
'''

checkpoint = torch.load('/148Dataset/PretrainedModel/model.audio/english/WavLM-Large.pt')#Pretrained model checkpoint path.
cfg = WavLMConfig(checkpoint['cfg'])
model = WavLM(cfg)
model.load_state_dict(checkpoint['model'])
model = model.to(device)
model.eval()

#IEMOCAP:
for i in range(1,6):
    csv = r'iemocap_data/data'# csv path. 
    inputdir = r'/148Dataset/data-chen.shuaiqi/IEMOCAP/IEMOCAP_full_release/Session'# data path.
    outputdir = r'/148Dataset/data-chen.shuaiqi/IEMOCAP/IEMOCAP_full_release/Feature/WavLM/Session' #feature path.
    dir = inputdir + str(i) + r'/sentences/wav/'
    csvs = csv + str(i) + '.csv'
    csv = pd.read_csv(csvs)
    name1 = csv.dataname.values
    name2 = csv.newdataname.values
    for j in range(len(name2)):
        data, _ = librosa.load(dir+name1[j]+'/'+name2[j]+'.wav',sr = 16000)
        data = data[np.newaxis,:]
        data = torch.Tensor(data).to(device)
        with torch.no_grad():
            feature = model.extract_features(source = data, output_layer = 12)[0]
        feature = feature.squeeze(0)
        feature = feature.cpu().data.numpy()        
        np.save(outputdir + str(i) + '/'+name2[j] + '.npy',feature)


#-------------------------------------------------------------------------------------
#MELD:

def MELD_extractor(inputdir,outputdir):
    name = os.listdir(inputdir)
    for i in range(len(name)):
        data1 = inputdir + name[i]
        names = name[i]
        data,_ = librosa.load(data1,sr = 16000)
        data = data[np.newaxis,:]
        data = torch.Tensor(data).to(device)
        with torch.no_grad():
            feature = model.extract_features(source = data, output_layer = 12)[0]
        feature = feature.squeeze(0)
        feature = feature.cpu().data.numpy()
        print(feature.shape)
        np.save(outputdir + names[:-4] + '.npy', feature)    
#train:
inputdir = r'/148Dataset/data-chen.shuaiqi/MELD/MELD.Raw/train_splits/'
outputdir = r'/148Dataset/data-chen.shuaiqi/MELD/MELD.Raw/feature/WavLM12/train/'
MELD_extractor(inputdir,outputdir)
#dev:
inputdir = r'/148Dataset/data-chen.shuaiqi/MELD/MELD.Raw/dev_splits_complete/'
outputdir = r'/148Dataset/data-chen.shuaiqi/MELD/MELD.Raw/feature/WavLM12/dev/'
MELD_extractor(inputdir,outputdir)
#test:
inputdir = r'/148Dataset/data-chen.shuaiqi/MELD/MELD.Raw/output_repeated_splits_test/'
outputdir = r'/148Dataset/data-chen.shuaiqi/MELD/MELD.Raw/feature/WavLM12/test/'
MELD_extractor(inputdir,outputdir)

