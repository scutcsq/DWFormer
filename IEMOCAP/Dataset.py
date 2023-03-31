import os
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import fairseq
import lmdb
import shutil
def preprocess_lmdb(out_path,data_path,mode,csvname,reset):
    if os.path.exists(out_path + mode):
        if reset == False:
            return None
        else:
            shutil.rmtree(out_path + mode)
    print('start preprocessing .')
    data = pd.read_csv(csvname)
    names = data.name.values
    labels = data.label.values
    env = lmdb.open(out_path + mode,map_size = 409951162700)
    count = 0
    ones = np.ones((324),dtype = np.float32)
    with env.begin(write = True) as txn:
        for i in range(len(labels)):
            name1 = names[i]
            data1 = np.load(data_path + 'Session' + name1[4]+'/'+name1+'.npy')
            newdata1 = np.zeros((324,1024),dtype = np.float32)
            mask = np.zeros((324),dtype = np.float32)
            lens = data1.shape[0]
            if lens > 324:
                newlens = 324
            else:
                newlens = lens
                mask[newlens:] = ones[newlens:]
            newdata1[:newlens, :] = data1[:newlens, :]
            key_data = 'data-%05d'%count
            key_label = 'label-%05d'%count
            key_mask = 'mask-%05d'%count
            txn.put(key_data.encode(),newdata1)
            txn.put(key_label.encode(),labels[i])
            txn.put(key_mask.encode(),mask)
            count += 1
    env.close()
    print(' preprocess is finished !')

out_path = r'./new_database_wavlm_mask_324/' #save the Dataset
os.mkdir(out_path)
for i in range(1,6):
    csvname1 = r'train'+ str(i) + '.csv'
    data_path = r'./Feature/WavLM/'#place where saves the WavLM features
    mode = r'train' + str(i)
    os.mkdir(out_path+mode)
    reset = True
    preprocess_lmdb(out_path,data_path,mode,csvname1,reset)
    mode = r'valid' + str(i)
    os.mkdir(out_path+mode)
    csvname2 = r'valid' + str(i) + '.csv'
    preprocess_lmdb(out_path,data_path,mode,csvname2,reset)