'''
Model expects following input formats
X.shape = (n,m) where n- number of features and m - number of examples
Y.shape = (c,m) where Y is set of one-hot vectors, c = number of classes, m = number of examples
'''

#necessary libraries
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
import random as rd
import numpy as np
import matplotlib.pyplot as plt

#loading Iris dataset
dataset = datasets.load_iris()
X = dataset.data                #X.shape = (150,4) - needs to be transposed m=150, n=4
Y = dataset.target              #Y.shape = (150,) - rank 1 array needs to be reshaped and one hot encoded


#reshaping and oneHot Encoding
X = X.T 
Y = Y.reshape(Y.shape[0],1)
onehot_encoder = OneHotEncoder(sparse=False)
Y_coded = onehot_encoder.fit_transform(Y)
Y = Y_coded.T 


#dividing data to test and train
def test_train_split(X,Y,percent_train=0.70):
    num_train_examples = int(X.shape[1] * percent_train)
    train_samples = set(rd.sample(range(0,X.shape[1]),num_train_examples))
    total = set(range(0,X.shape[1]))
    test_samples = total - train_samples
    
    test_samples = list(test_samples)
    train_samples = list(train_samples)
    rd.shuffle(train_samples)
    rd.shuffle(test_samples)
    
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    
    for i in train_samples:
        X_train.append(X[:,i])
        Y_train.append(Y[:,i])
    
    for i in test_samples:
        X_test.append(X[:,i])
        Y_test.append(Y[:,i])
    
    X_train = np.array(X_train).T
    Y_train = np.array(Y_train).T
    X_test = np.array(X_test).T
    Y_test = np.array(Y_test).T
    return (X_train, Y_train, X_test, Y_test)    
    
X_train, Y_train, X_test, Y_test = test_train_split(X, Y, percent_train=0.80)


#Using model to run multiclass logistic regression
from Logistic_regression_multiclass import *

'''
X       - data
Y       - target
epochs  - maximum iterations over the dataset
alpha   - learning rate
lambd   - regularization parameter
displayRate - Number of iterations after which cost is printed
'''
W,b,costs,iters = train(X_train, Y_train, epochs = 10000,alpha = 0.09, lambd = 0.1,displayRate=500)


# To check if cost decreases at each iteration
def plot_cost(costs,iters):
    plt.title('cost v/s iterations')
    plt.xlabel('iterations')
    plt.ylabel('cost')
    plt.plot(iters,costs)
    plt.show()

plot_cost(costs,iters)


# measuring test and train accuracy. Note the format model expects. 
A_train = predict(X_train, W, b)
train_acc = evaluate_accuracy(A_train,Y_train)
print('Train_accuracy: ',train_acc)

A_test = predict(X_test,W,b)
test_acc = evaluate_accuracy(A_test,Y_test)
print('Test_accuracy: ',test_acc)