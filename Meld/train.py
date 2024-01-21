import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import torch
import torch.nn as nn
import numpy as np
import time
import random
from sklearn import metrics
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from statistics import mode
from model import DWFormer
import lmdb
torch.set_num_threads(1)
gen_data = False
voting = True

#-----------------------------------------------限制随机------------------------------------------

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark= False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False

#-------------------------------------------------生成DataLoader-----------------------------------

class lmdb_dataset(Dataset):
    def __init__(self,out_path,mode):
        self.env = lmdb.open(out_path + mode)
        self.txn = self.env.begin(write = False)
        self.len = self.txn.stat()['entries']
    def __getitem__(self,index):
        key_data = 'data-%05d' %index
        key_label = 'label-%05d' %index
        key_mask = 'mask-%05d' %index

        data = np.frombuffer(self.txn.get(key_data.encode()),dtype = np.float32)
        data = torch.FloatTensor(data.reshape(-1,1024).copy())
        label = np.frombuffer(self.txn.get(key_label.encode()),dtype = np.int64)
        label = torch.LongTensor(label.copy()).squeeze()
        mask = np.frombuffer(self.txn.get(key_mask.encode()),dtype = np.float32)
        mask = torch.FloatTensor(mask.copy())

        return data, label, mask
    def __len__(self):
        return int(self.len / 3)
    
final_wa = []
final_ua = []

seed = 2
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


out_path = r'./WavLM/'
trainplace = r'train'
validplace = r'dev'
testplace = r'test'
train_dataset = lmdb_dataset(out_path,trainplace)
develop_dataset = lmdb_dataset(out_path,validplace)
test_dataset = lmdb_dataset(out_path,testplace)
trainDataset = DataLoader(dataset=train_dataset,batch_size=32,shuffle=True,drop_last = True)
developDataset = DataLoader(dataset=develop_dataset,batch_size=32,shuffle= False)
testDataset = DataLoader(dataset = test_dataset,batch_size = 32, shuffle = False)

model = DWFormer(feadim = 1024, n_head = 8, FFNdim = 512, classnum = 7).to(device)

modelname = 'dwformer'
WD = 1e-3
LR_DECAY = 0.5
EPOCH = 100
STEP_SIZE = 5
lr = 5e-4

optimizer = torch.optim.SGD(model.parameters(),lr = lr, momentum = 0.9)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0 = 3,T_mult = 2, eta_min = 1e-4 * 0.1)
loss = nn.CrossEntropyLoss().to(device)

#--------------------------------------------------train--------------------------------------------
best_wa = 0
best_ua = 0
best_test_wa = 0
best_test_ua = 0
act_best_test_wa = 0
act_best_test_ua = 0
num = 0
for epoch in range(EPOCH):
    model.train()
    loss_tr = 0.0
    start_time = time.time()
    pred_all,actu_all = [],[]
    for step, (datas, labels, mask) in enumerate(trainDataset, 0):
        mask = mask.to(device)
        datas = datas.to(device)
        labels = labels.view(len(labels))
        labels = labels.to(device)
        optimizer.zero_grad()

        out = model(datas,mask)
        

        err1 = loss(out,labels.long())
        err1.backward()
        optimizer.step()
        pred = torch.max(out.cpu().data, 1)[1].numpy()
        actu = labels.cpu().data.numpy()
        pred_all += list(pred)
        actu_all += list(actu)
        loss_tr += err1.cpu().item()
    loss_tr = loss_tr / len(trainDataset.dataset)
    pred_all, actu_all = np.array(pred_all), np.array(actu_all)
    wa_tr = metrics.accuracy_score(actu_all, pred_all)
    # ua_tr = metrics.recall_score(actu_all, pred_all,average='macro')
    ua_tr = metrics.f1_score(actu_all, pred_all, average = 'weighted')
    end_time = time.time()
    print('当前学习率',str(optimizer.param_groups[0]['lr']))
    print('TRAIN:: Epoch: ', epoch, '| Loss: %.3f' % loss_tr, '| wa: %.3f' % wa_tr, '| ua: %.3f' % ua_tr)
    print('所耗时长:',str(end_time-start_time),'s')
    scheduler.step()

# #---------------------------------------------------develop-----------------------------------------
    model.eval()
    loss_de = 0.0
    start_time = time.time()
    pred_all,actu_all = [],[]
    for step, (datas, labels, mask) in enumerate(developDataset, 0):
        mask = mask.to(device)
        datas = datas.to(device)
        labels = labels.view(len(labels))
        labels = labels.to(device)
        #原有
        with torch.no_grad():
            out = model(datas, mask)
        err1 = loss(out,labels.long())
        pred = torch.max(out.cpu().data, 1)[1].numpy()
        actu = labels.cpu().data.numpy()
        pred_all += list(pred)
        actu_all += list(actu)
        loss_de += err1.cpu().item()
    loss_de = loss_de / len(developDataset.dataset)
    pred_all, actu_all = np.array(pred_all,dtype=int), np.array(actu_all,dtype=int)

    wa_de = metrics.accuracy_score(actu_all, pred_all)
    # ua_de = metrics.recall_score(actu_all, pred_all,average='macro')
    ua_de = metrics.f1_score(actu_all, pred_all, average = 'weighted')
    if ua_de > best_ua:
        torch.save(model.state_dict(), modelname + '/model-best_seed'+str(seed)+'.txt')
        best_ua = ua_de
        best_wa = wa_de
        num = epoch
    elif (ua_de == best_ua) and (wa_de > best_wa):
        torch.save(model.state_dict(), modelname + '/model-best_seed'+str(seed)+'.txt')
        best_ua = ua_de
        best_wa = wa_de
        num = epoch        
    end_time = time.time()
    print('VALID:: Epoch: ', epoch, '| Loss: %.3f' % loss_de, '| wa: %.3f' % wa_de, '| ua: %.3f' % ua_de)
    print('所耗时长:  ',str(end_time-start_time),'s')
# # #------------------------------------test------------------------------------------------------------
print('验证集最好结果: | wa: %.3f' %best_wa, '|ua: %.3f' %best_ua)
torch.cuda.empty_cache()
model = DWFormer(feadim = 1024, n_head = 8, FFNdim = 512, classnum = 7)
model.load_state_dict(torch.load(modelname + '/model-best_seed'+str(seed)+'.txt'))
model = model.to(device)
model.eval()
loss_te = 0.0
# start_time = time.time()
pred_all,actu_all = [],[]
for step, (datas,labels,mask) in enumerate(testDataset, 0):
    mask = mask.to(device)
    datas = datas.to(device)
    labels = labels.view(len(labels))
    labels = labels.to(device)
    #原有
    with torch.no_grad():
        out = model(datas,mask)
        
    err1 = loss(out,labels.long())
    pred = torch.max(out.cpu().data, 1)[1].numpy()
    actu = labels.cpu().data.numpy()
    pred_all += list(pred)
    actu_all += list(actu)
    loss_te += err1.cpu().item()
loss_te = loss_te / len(testDataset.dataset)
pred_all, actu_all = np.array(pred_all,dtype=int), np.array(actu_all,dtype=int)
epoch = 1
wa_te = metrics.accuracy_score(actu_all, pred_all)
# ua_te = metrics.recall_score(actu_all, pred_all,average='macro')
ua_te = metrics.f1_score(actu_all, pred_all, average = 'weighted')
print('TEST:: Epoch: ', epoch, '| Loss: %.3f' % loss_te, '| wa: %.3f' % wa_te, '| ua: %.3f' % ua_te)

