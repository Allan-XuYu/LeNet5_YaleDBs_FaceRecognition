#!/usr/bin/env

'''
Author: Allanxu
Date: 2021-04-07 14:14:24
LastEditors: Allanxu
LastEditTime: 2021-04-07 15:40:37
Description: ---
'''
from scipy.io import loadmat
import numpy as np
import time
from collections import defaultdict
import datatools as dt

class xu_PCA:
    
    def __init__(self):
        #Pixel point generally is used a variable range 0 to 255 to represent, so normalize data by divide 255 (range 0 to 1)
        self.std_num = 255.0
    
    def PCA(self, raw_data, PCs):
        data = raw_data/self.std_num
        S = np.cov(data.T)
        eigvalues,eigvectors = np.linalg.eig(S)
        indexByValues = np.argsort(eigvalues)[::-1] # sorted by eigenvalues biger to smaller
    
        eigvalues,eigvectors = eigvalues[indexByValues],eigvectors[:,indexByValues]
        eigvalues=np.real(eigvalues)
        eigvectors=np.real(eigvectors)
        subspace = data @ eigvectors[:,:PCs]
        return eigvalues,eigvectors,subspace

    #values,vectors,subs=PCA(X_std,30)
    #print(subs)

    def Plot_img(self, data):  # for this project assignment(Image recognition) only
        try:
            plt.imshow(data)
        except TypeError:
            print("check data structure, ** x **")

    #Plot_img(X_hat[0,:])

    def Reconstruct(self, data, eig_vectors,PCs):
        eig_vectors = eig_vectors[:,:PCs]
        cover_data = data @ eig_vectors.T
        cover_data = self.std_num * cover_data

        return cover_data

    # temp = Reconstruct(Z,eig_vectors[:,:30])
    # temp = temp[0,:].reshape((28,28))
    # Plot_img(temp)
    
    def MSE(self, data, raw_data):
        return np.mean(pow((raw_data-data),2))
    
    def PCA_noStd(self, data, PCs): 
        S = np.cov(data.T)
        eigvalues,eigvectors = np.linalg.eig(S)
        indexByValues = np.argsort(eigvalues)[::-1]

        eigvalues,eigvectors = eigvalues[indexByValues],eigvectors[:,indexByValues]
        eigvalues=np.real(eigvalues)
        eigvectors=np.real(eigvectors)
        subspace = data @ eigvectors[:,:PCs]
        return eigvalues,eigvectors,subspace
    
if __name__ == "__main__":

    main_dir = "/home/xuyu/comp7930"
    data_dir = "/Data"
    out_dir ="/outfiles/PCA"

    DBs=defaultdict(list)
    DBs['Ya64'] = ['2Train','3Train','4Train','5Train','6Train','7Train','8Train']
    DBs['Ya32'] = ['5Train','10Train','20Train','30Train','40Train','50Train']
    DBs['ORL64'] = ['2Train','3Train','4Train','5Train','6Train','7Train','8Train']
    DBs['ORL32'] = ['2Train','3Train','4Train','5Train','6Train','7Train','8Train']

    # raw data loading
    X_Ya64,Y_Ya64 = dt.Load_RawData(main_dir+data_dir+'/Yale_64x64.mat')
    X_Ya32,Y_Ya32 = dt.Load_RawData(main_dir+data_dir+'/YaleB_32x32.mat')
    X_ORL64,Y_ORL64 = dt.Load_RawData(main_dir+data_dir+'/ORL_64x64.mat')
    X_ORL32,Y_ORL32 = dt.Load_RawData(main_dir+data_dir+'/ORL_32x32.mat')
  
    debug = False
    start_time = time.time()
    for i in DBs.keys(): # different Data
        for j in DBs[i]:  #   different splited proportion
            print('DB:%s,Splited:%s' %(i,j))
            #print('time consume:',time.time() - start_time)
            #for k in range(50):  # randomly splits
                #print(main_dir+data_dir+ "\\"+ i+"\\"+ j +"\\" + str(k+1) + '.mat')
            splited_ind = loadmat(main_dir+data_dir+ "/"+ i+"/"+ j +"/" + str(1) + '.mat')
            ind_train = splited_ind['trainIdx'].squeeze()
            ind_test = splited_ind['testIdx'].squeeze()
            if i=='Ya64':
                train,train_y,test,test_y = dt.SetsSplit(X_Ya64,Y_Ya64,ind_train,ind_test)
            elif i=='Ya32':
                train,train_y,test,test_y = dt.SetsSplit(X_Ya32,Y_Ya32,ind_train,ind_test)
            elif i=='ORL64':
                train,train_y,test,test_y = dt.SetsSplit(X_ORL64,Y_ORL64,ind_train,ind_test)
            elif i=='ORL32':
                train,train_y,test,test_y = dt.SetsSplit(X_ORL32,Y_ORL32,ind_train,ind_test)
            else:
                print('data error')
            
            # Loss with different PCs in one set. 
            loss_tr = []
            loss_te = []
            # 50 different top-k PCs
            step = int(train.shape[1]/50)
            PCs = list(range(1,step*50,step))
            for p in PCs:# 1-4095 or 1-1023  
                
                print(p)
                DimRedu = xu_PCA() 
                eigenvalues,eigenvectors,sub = DimRedu.PCA(train,p)
                re_data=DimRedu.Reconstruct(sub,eigenvectors,p)
                train_loss = DimRedu.MSE(re_data,train)
                loss_tr.append(train_loss)

                sub_test = (test/255.0)@ eigenvectors[:,:p]
                re_test=DimRedu.Reconstruct(sub_test,eigenvectors,p)
                test_loss = DimRedu.MSE(re_test,test)
                loss_te.append(test_loss)
                
            outfile1 = open(main_dir+out_dir+ "/"+ i + '_' + j + '_' + str(1)+ '_' + "trainloss","w")
            outfile2 = open(main_dir+out_dir+ "/"+ i + '_' + j + '_' + str(1)+ '_' + "testloss","w")
            for l in range(len(loss_tr)):
                outfile1.write(str(PCs[l]) + '\t'+ str(loss_tr[l]) + '\n')
            outfile1.close()
            for l in range(len(loss_te)):
                outfile2.write(str(PCs[l]) + '\t'+ str(loss_te[l]) + '\n')
            outfile2.close()

            #if debug==True:
            #    break
            print('time consume:',time.time() - start_time)
            if debug==True:
                    break
        if debug==True:
            print(train)
            break
