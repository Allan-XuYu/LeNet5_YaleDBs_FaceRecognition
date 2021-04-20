'''
Author: Allanxu
Date: 2021-04-16 12:54:14
LastEditors: Allanxu
LastEditTime: 2021-04-19 14:45:12
Description: ---
'''
from scipy.io import loadmat
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import time
import os
from collections import defaultdict

import datatools as dt

class CNN_32(nn.Module):
    def __init__(self,cat_labels,dropout_rate):
        super(CNN_32,self).__init__()
        
        self.conv_layer1 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=5, stride=1,padding = 1,padding_mode ='zeros',groups= 1)
        self.max_layer1 = nn.MaxPool2d(3, stride=2)
        self.conv_layer2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=5, stride=1,padding = 1,padding_mode ='zeros',groups= 8)
        self.max_layer2 = nn.MaxPool2d(2, stride=2)

        self.fc1 = nn.Linear(16*6*6, 120)  # 32*6*6 is from maxpooling layer 2
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, cat_labels)
        
        self.dropout = nn.Dropout(dropout_rate)  
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        
    def forward(self,data):
        C1 = self.relu(self.conv_layer1(data))
        S2 = self.max_layer1(C1)
        C3 = self.relu(self.conv_layer2(S2))
        S4 = self.max_layer2(C3)
        F_in   = S4.view(data.size(0),-1) # flatten
        F5 = self.relu(self.fc1(F_in))
        F6 = self.relu(self.fc2(F5))
        F7 = self.fc3(self.dropout(F6))
        result = self.softmax(F7)
        return result

class CNN_64(nn.Module):
    def __init__(self,cat_labels,dropout_rate):
        super(CNN_64,self).__init__()
        
        self.conv_layer1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5, stride=2,padding = 1,padding_mode ='zeros',groups= 1)
        self.max_layer1 = nn.MaxPool2d(3, stride=2)
        self.conv_layer2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5, stride=1,padding = 1,padding_mode ='zeros',groups= 16)
        self.max_layer2 = nn.MaxPool2d(2, stride=2)

        self.fc1 = nn.Linear(32*6*6, 120) 
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, cat_labels)
        
        self.dropout = nn.Dropout(dropout_rate)  
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        
    def forward(self,data):
        C1 = self.relu(self.conv_layer1(data))
        S2 = self.max_layer1(C1)
        C3 = self.relu(self.conv_layer2(S2))
        S4 = self.max_layer2(C3)
        F_in   = S4.view(data.size(0),-1) # flatten
        F5 = self.relu(self.fc1(F_in))
        F6 = self.relu(self.fc2(F5))
        F7 = self.fc3(self.dropout(F6))
        result = self.softmax(F7)
        return result

def model_train(data,label,c_label,batch_size,outfile_dir,setName,n_epoch=3000,pixel_struc='64'):
    # make sure output dir
    if not os.path.exists(outfile_dir):
        os.makedirs(outfile_dir)
    # training data load
    if pixel_struc=='64':
        train_data=Variable(torch.FloatTensor(data)).reshape(batch_size,64,64).unsqueeze(1)
        train_label=Variable(torch.LongTensor(label-1).squeeze(1))
        cnn_model = CNN_64(c_label, 0.1 )
    elif pixel_struc=='32':
        train_data=Variable(torch.FloatTensor(data)).reshape(batch_size,32,32).unsqueeze(1)
        train_label=Variable(torch.LongTensor(label-1).squeeze(1))
        cnn_model = CNN_32(c_label, 0.1 )
    # model instance / optimizer intialization
    optimizer = torch.optim.Adadelta(cnn_model.parameters(), rho = 0.95, weight_decay = 0.001)
    #optimizer = torch.optim.SGD(cnn_model.parameters(), lr=0.01, momentum=0.9)

    # some parameters
    best_epoch = 0
    best_loss = 1111111
    loss_epoch = []

    # train model
    for epoch in range(n_epoch):
        optimizer.zero_grad()
        model_out = cnn_model(train_data)
        loss = F.cross_entropy(model_out, train_label)
        loss.backward()
        optimizer.step()
        loss_epoch.append(loss.data)
        
        if loss.data < best_loss:
            best_loss = loss.data
            best_epoch = epoch
            torch.save(cnn_model.state_dict(), outfile_dir + '\\'+ setName +'_cnn.pth' )
    # save epoch loss        
    np.save(outfile_dir+'\\'+setName+'_loss',loss_epoch)


if __name__ == "__main__":  

    main_dir = r"C:\Users\Allan\Desktop\AIS\Bigdata\COMP7930_Final_Project"
    data_dir = r"\Data"

    # various datasets
    DBs=defaultdict(list)
    DBs['Ya64'] = ['2Train','3Train','4Train','5Train','6Train','7Train','8Train']
    DBs['Ya32'] = ['5Train','10Train','20Train','30Train','40Train','50Train']
    DBs['ORL64'] = ['2Train','3Train','4Train','5Train','6Train','7Train','8Train']
    DBs['ORL32'] = ['2Train','3Train','4Train','5Train','6Train','7Train','8Train']

    # raw data loading
    X_Ya64,Y_Ya64 = dt.Load_RawData(main_dir+data_dir+r'\Yale_64x64.mat')
    X_Ya32,Y_Ya32 = dt.Load_RawData(main_dir+data_dir+r'\YaleB_32x32.mat')
    X_ORL64,Y_ORL64 = dt.Load_RawData(main_dir+data_dir+r'\ORL_64x64.mat')
    X_ORL32,Y_ORL32 = dt.Load_RawData(main_dir+data_dir+r'\ORL_32x32.mat')


    outpath = r"C:\Users\Allan\Desktop\AIS\Bigdata\COMP7930_Final_Project\outfiles\cnn"
    train_mode = '64'
    n_ind = 0

    for i in DBs.keys(): # different Data
        
        for j in DBs[i]:  #   different splited proportion
            print("DB:%s, Set:%s" % (i,j))
            #for k in range(50):  # randomly splits
                #print(main_dir+data_dir+ "\\"+ i+"\\"+ j +"\\" + str(k+1) + '.mat')
            splited_ind = loadmat(main_dir+data_dir+ "\\"+ i+"\\"+ j +"\\" + str(1) + '.mat')
            ind_train = splited_ind['trainIdx'].squeeze()
            ind_test = splited_ind['testIdx'].squeeze()
            if i=='Ya64':
                train_mode = '64' 
                n_ind = 15
                train,train_y,test,test_y = dt.SetsSplit(X_Ya64,Y_Ya64,ind_train,ind_test)
            elif i=='Ya32':
                n_ind = 38
                train_mode = '32' 
                train,train_y,test,test_y = dt.SetsSplit(X_Ya32,Y_Ya32,ind_train,ind_test)
            elif i=='ORL64':
                n_ind = 40
                train_mode = '64' 
                train,train_y,test,test_y = dt.SetsSplit(X_ORL64,Y_ORL64,ind_train,ind_test)
            elif i=='ORL32':
                n_ind = 40
                train_mode = '32' 
                train,train_y,test,test_y = dt.SetsSplit(X_ORL32,Y_ORL32,ind_train,ind_test)
            else:
                print('data error')

             
            model_train(train, train_y,c_label=n_ind,batch_size= train.shape[0], outfile_dir=outpath,
            setName=i+'_'+j ,n_epoch=3000,pixel_struc=train_mode)