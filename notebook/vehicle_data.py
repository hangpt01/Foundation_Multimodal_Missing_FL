import scipy.io
import os
import numpy as np
import random
import json
from numpy import *
import torch

# all clients -> 1 file



NUM_USER = 23
def preprocess(x):
    means = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    
    x = (x - means) * 1.0 / std
    where_are_NaNs = isnan(x)
    x[where_are_NaNs] = 0
    return x

def generate_data():
    
    mat = scipy.io.loadmat('benchmark/RAW_DATA/VEHICLE/vehicle.mat')
    raw_x, raw_y = mat['X'], mat['Y']
    raw_x.shape, raw_y.shape, raw_x[0][0].shape, raw_y[0][0].shape

    train_samples_each_client = np.zeros(NUM_USER,dtype=int)
    test_samples = 0
    preprocess_X = preprocess(raw_x[0][0])
    train_samples = int(preprocess_X.shape[0]*0.75)
    train_samples_each_client[0] = train_samples
    test_samples += preprocess_X.shape[0] - train_samples
    
    X_train = preprocess_X[:train_samples,:]
    X_test = preprocess_X[train_samples:, :]
    Y_train = raw_y[0][0][:train_samples, :]
    Y_test = raw_y[0][0][train_samples:, :]
    
    for i in range(1,NUM_USER):
        x_processed = preprocess(raw_x[i][0])
        
        train_samples = int(x_processed.shape[0]*0.75)
        train_samples_each_client[i] = train_samples
        test_samples += x_processed.shape[0] - train_samples
        
        x_train = x_processed[:train_samples,:]
        x_test = x_processed[train_samples:,:]
        X_train = np.concatenate((X_train, x_train), axis=0)
        # print(X_test.shape, x_test.shape)
        X_test = np.concatenate((X_test, x_test), axis=0)
        
        y_temp = raw_y[i][0]
        y_train = y_temp[:train_samples,:]
        y_test = y_temp[train_samples:,:]
        Y_train = np.concatenate((Y_train, y_train), axis=0)
        Y_test = np.concatenate((Y_test, y_test), axis=0)
        
        num = 0
        for j in range(len(raw_y[i][0])):
            if raw_y[i][0][j] == 1:
                num += 1
        print("ratio, ", num * 1.0 / len(raw_y[i][0]))
    # return np.array(X), np.array(y)
    Y_train[Y_train == -1] = 0
    Y_test[Y_test == -1] = 0
    print(train_samples_each_client, test_samples)
    print(Y_train[:20], Y_test[:20])
    np.save('benchmark/RAW_DATA/VEHICLE/x_train.npy', np.array(np.transpose(X_train.reshape(-1,2,50),(0,2,1)), dtype=object), allow_pickle=True)
    np.save('benchmark/RAW_DATA/VEHICLE/x_test.npy', np.array(np.transpose(X_test.reshape(-1,2,50),(0,2,1)), dtype=object), allow_pickle=True)
    np.save('benchmark/RAW_DATA/VEHICLE/y_train.npy', np.array(Y_train, dtype=object), allow_pickle=True)
    np.save('benchmark/RAW_DATA/VEHICLE/y_test.npy', np.array(Y_test, dtype=object), allow_pickle=True)
    
    return X_train, X_test, Y_train, Y_test, train_samples_each_client.tolist(), test_samples

def change_labels():
    Y_train = np.load("benchmark/RAW_DATA/VEHICLE/y_train.npy", allow_pickle=True)
    Y_test = np.load("benchmark/RAW_DATA/VEHICLE/y_test.npy", allow_pickle=True)
    Y_train = np.array(Y_train.reshape(-1), dtype='int64')
    Y_test = np.array(Y_test.reshape(-1), dtype='int64')
    Y_train[Y_train == -1] = 0
    Y_test[Y_test == -1] = 0
    print(Y_train[:20], Y_test[:20])
    # np.save('benchmark/RAW_DATA/VEHICLE/y_train.npy', np.array(Y_train, dtype=object), allow_pickle=True)
    # np.save('benchmark/RAW_DATA/VEHICLE/y_test.npy', np.array(Y_test, dtype=object), allow_pickle=True)

if __name__ == '__main__':
    X1, X2, Y1, Y2, ls_train_samples, test_samples = generate_data()
    print(X1.shape, X2.shape, Y1.shape, Y2.shape, ls_train_samples, test_samples)
    # change_labels()