# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 19:19:53 2017

@author: Faaiz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

wq = pd.read_csv("winequality-red.csv",sep=";")
wqdata = pd.DataFrame(wq)

# The normalization formula is used to normalize the data.
wqdata_norm = (wqdata-wqdata.min())/(wqdata.max()-wqdata.min())
print(wqdata_norm)

#Then we are supposed to split the data set into test and train data.
#X has all the columns except the last one. Y has the last column, ‘quality’.
splitrandom=np.random.rand(len(wqdata_norm))<=0.8
wqdata_train = wqdata_norm[splitrandom]
wqdata_test = wqdata_norm[~splitrandom]

X_wqdata_train = wqdata_train.iloc[:,0:11]
bias=np.ones((len(X_wqdata_train),1))
X_wqdata_train.insert(0,'bias',bias)

Y_wqdata_train = wqdata_train.iloc[:,11]

X_wqdata_test = wqdata_test.iloc[:,0:11]
bias=np.ones((len(X_wqdata_test),1))
X_wqdata_test.insert(0,'bias',bias)

Y_wqdata_test = wqdata_test.iloc[:,11]

# The ridge regression algorithm is then implemented.
# It takes 9 arguments. X and Y are the data’s trained values. A and B store the RMSE of the trained and test data sets respectively.
def learn_ridge_reg(alpha,L,Y,X,tmax,batch_size,Y_wqdata_test,X_wqdata_test):
    
    A = [] #stores the RMSE of trained data set
    B = [] #stores the RMSE of test data set
    
    X = X.iloc[:batch_size,:]
    Y = Y.iloc[:batch_size]
    XT = X.T
    X_transpose = XT.iloc[:,:batch_size]
    
    # The below 2 lines of code randomly shuffle the X and Y data.
    X = np.random.permutation(X)
    Y = np.random.permutation(Y)

    beta = np.zeros(X.shape[1]) 

    for i in range(0,tmax):
       XB = np.dot(X,beta)
       Y_XB = np.subtract(Y,XB)

       mul = np.dot(X_transpose,Y_XB)
       mul2 =-2*mul
       
       L2 = 2*L
       L2_beta = np.dot(L2,beta)
       add = np.add(mul2,L2_beta)
       alpha_mul = alpha*add
       beta = beta-alpha_mul
      
       Y_hat = np.dot(X, beta)
       residuals = Y-Y_hat
       RMSE_train = np.sqrt(np.mean((residuals)**2))

       Y_hat = np.dot(X_wqdata_test, beta)
       residuals = Y_wqdata_test-Y_hat
       RMSE_test = np.sqrt(np.mean((residuals)**2))
      
    # The first part of the code calculates the RMSE of the trained data set, whereas the second part calculates for the test data set. 
       A = np.append(A,RMSE_train)
       B = np.append(B,RMSE_test)
    # The RMSE values are then appended to the empty arrays assigned earlier.
    
    plt.plot(range(i+1),A,c="r")  
    plt.title("RMSE vs iterations")
    plt.plot(range(i+1),-B)  
    plt.legend()
    plt.figure()
    
# The values of Lambda and alpha are increased thrice.        
alpha = 0.0001
L = 0.2
tmax = 500
batch_size=50

train_1 = learn_ridge_reg(alpha,L,Y_wqdata_train,X_wqdata_train,tmax,batch_size,Y_wqdata_test,X_wqdata_test)

alpha = 0.001
L = 0.4
train_2 = learn_ridge_reg(alpha,L,Y_wqdata_train,X_wqdata_train,tmax,batch_size,Y_wqdata_test,X_wqdata_test)

alpha = 0.01
L = 0.8
train_3 = learn_ridge_reg(alpha,L,Y_wqdata_train,X_wqdata_train,tmax,batch_size,Y_wqdata_test,X_wqdata_test)




    
    



