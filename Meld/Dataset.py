import os
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import fairseq
import lmdb
import shutil
from scipy import io

def label_MELD_change(label):
    if label == 'neutral':
        datalabel = 0
    elif label == 'joy':
        datalabel = 1
    elif label == 'sadness':
        datalabel = 2
    elif label == 'anger':
        datalabel = 3
    elif label == 'surprise':
        datalabel = 4
    elif label == 'fear':
        datalabel = 5
    elif label == 'disgust':
        datalabel = 6
    return datalabel

def gender_MELD_change(label):
    if label == 'Chandler':
        datalabel = 0
    elif label == 'Joey':
        datalabel = 0
    elif label == 'Rachel':
        datalabel = 1
    elif label == 'Monica':
        datalabel = 1
    elif label == 'Phoebe':
        datalabel = 1
    elif label == 'Ross':
        datalabel = 0
    else:
        datalabel = 2
    return datalabel

def preprocess_lmdb_MELD(out_path,mode,csvname,reset,length=225):
    if os.path.exists(out_path + mode):
        if reset == False:
            return None
        else:
            shutil.rmtree(out_path + mode)
    data = pd.read_csv(csvname)
    Dialogue_ID = data.Dialogue_ID.values
    Utterance_ID = data.Utterance_ID.values
    emotion = data.Emotion.values
    speaker = data.Speaker.values
    env = lmdb.open(out_path + mode,map_size = 409951162700)
    count = 0
    with env.begin(write = True) as txn:    
        for i in range(len(emotion)):
            name1 = 'dia' + str(Dialogue_ID[i]) + '_utt' + str(Utterance_ID[i]) + '.npy'
            # namepath = '/148Dataset/data-chen.weidong/meld/feature/wavlm_large_L12_mat/'+ mode + '/' + name1
            namepath = r'./feature/MFCC/'+mode+'/'+name1
            if os.path.exists(namepath):
                # data1 = io.loadmat(namepath)['wavlm']
                data1 = np.load(namepath)
                newdata1 = np.zeros((length,1024),dtype = np.float32)
                maskdata = np.zeros((length), dtype = np.float32)
                ones = np.ones((length), dtype = np.float32)
                lens = data1.shape[0]
                if lens >= length:
                    lens = length
                newlabel = label_MELD_change(emotion[i])
                sexlabel = gender_MELD_change(speaker[i])
                # print(speaker[i])
                # print(sexlabel)
                if sexlabel!= 2:
                    newdata1[:lens, :] = data1[:lens, :]
                    maskdata[lens:] = ones[lens:] 
                    key_data = 'data-%05d'%count
                    key_label = 'label-%05d'%count
                    key_mask = 'mask-%05d'%count
                    txn.put(key_data.encode(),newdata1)
                    txn.put(key_label.encode(),np.array([newlabel]))
                    txn.put(key_mask.encode(),maskdata)
                    # txn.put(key_mask.encode(),mask)
                    # print(newlabel)
                    count += 1  
    env.close()  
    print(count)        
out_path = r'./WavLM12/'
mode = 'train'
csvname = r'train_sent_emo.csv'
reset = True
preprocess_lmdb_MELD(out_path,mode,csvname,reset)
mode = 'dev'
csvname = r'dev_sent_emo.csv'
preprocess_lmdb_MELD(out_path,mode,csvname,reset)
mode = 'test'
csvname = r'test_sent_emo.csv'
preprocess_lmdb_MELD(out_path,mode,csvname,reset)
