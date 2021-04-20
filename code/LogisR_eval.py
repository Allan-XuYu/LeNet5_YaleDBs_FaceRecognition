'''
Author: Allanxu
Date: 2021-04-12 14:33:13
LastEditors: Allanxu
LastEditTime: 2021-04-16 14:44:18
Description: ---
'''
from scipy.io import loadmat
import numpy as np
from sklearn.linear_model import LogisticRegression
import time
from collections import defaultdict
import datatools as dt
import PCA_eval as pca 



def accuracy(label,pred_result):
    pred_result = np.argmax(pred_result,axis=1)
    if len(label) != len(pred_result):
        return False
    count = 0
    for i in range(len(label)):
        if label[i] == pred_result[i]:
            count+=1 
    return count/len(label)



if __name__ == "__main__":    

    
    main_dir = r"C:\Users\Allan\Desktop\AIS\Bigdata\COMP7930_Final_Project"
    data_dir = r"\Data"
    out_dir =r"\outfiles\LogisR"
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


    PCs = {'Ya64':105,'Ya32':714,'ORL64':220,'ORL32':203}
    #count =0 # for debug
    for i in DBs.keys(): # different Data
        loss_arr = []
        for j in DBs[i]:  #   different splited proportion
            print("DB:%s, Set:%s" % (i,j))
            #for k in range(50):  # randomly splits
                #print(main_dir+data_dir+ "\\"+ i+"\\"+ j +"\\" + str(k+1) + '.mat')
            splited_ind = loadmat(main_dir+data_dir+ "\\"+ i+"\\"+ j +"\\" + str(1) + '.mat')
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

            # PCA
            DimRedu = pca.xu_PCA()
            eigenvalues,eigenvectors,sub = DimRedu.PCA(train,PCs[i])

            # Logistic regression model
            lr_clf = LogisticRegression(random_state=None, solver='lbfgs',multi_class='auto', verbose = 1,max_iter=10000)
            lr_clf.fit(sub, train_y.reshape(1,-1)[0])

            # train prediction loss
            train_pred = lr_clf.predict_proba(sub)
            label_train = train_y.squeeze() - 1
            print('Classfication accuracy with PCA:',accuracy(label_train,train_pred))

            # test pca
            sub_test = (test/255.0)@ eigenvectors[:,:PCs[i]]
            test_pred = lr_clf.predict_proba(sub_test)
            # tese prediction loss
            label_test = test_y.squeeze() - 1
            test_acc = accuracy(label_test,test_pred)
            print('Classfication accuracy with PCA:',test_acc)
            loss_arr.append(test_acc)
            
            #for debug
            # count+=1
            # if count>1:
            #     break

        outfile = open(main_dir+out_dir+'\\'+ i +'_acc','w')
        for k in range(len(loss_arr)):
            outfile.write(str(loss_arr[k])+'\n')
        outfile.close() 
        #break # for debug
            