from __future__ import division

import sys
import csv
from os import listdir
from os.path import isfile
from chainer.datasets import tuple_dataset

import numpy as np


def loadSingleFile(fileName):

    with open(fileName, "r") as f:
        return np.array(list(csv.reader(f, delimiter='\t')), dtype=np.float32)

def genDLdata(dataPath, split=0.8, loadFromCache=False):

    if loadFromCache == True:
        X_train = np.load("../data/Cache/trainX.npy")
        Y_train = np.load("../data/Cache/trainY.npy")

        X_test = np.load("../data/Cache/testX.npy")
        Y_test = np.load("../data/Cache/testY.npy")

        train = tuple_dataset.TupleDataset(X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2]), Y_train)
        test = tuple_dataset.TupleDataset(X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]), Y_test)

        return train, test
        
    darwinPath=dataPath+"/darwin"
    neutralPath=dataPath+"/neutral"

    da_data = [loadSingleFile(darwinPath+"/"+f) for f in listdir(darwinPath)]
    ne_data = [loadSingleFile(neutralPath+"/"+f) for f in listdir(neutralPath)]

    da_label = np.ones(len(da_data), dtype=np.int32)
    ne_label = np.zeros(len(ne_data), dtype=np.int32)

    tidx = int(split*len(da_data))
    print("** Total darwin sample [{}]".format(len(da_data)))
    da_data_part1, da_data_part2 = np.split(da_data,[tidx])
    da_label_part1, da_label_part2 = np.split(da_label,[tidx])

    tidx = int(split*len(ne_data))
    print("** Total neutral sample [{}]".format(len(ne_data)))
    ne_data_part1, ne_data_part2 = np.split(ne_data,[tidx])
    ne_label_part1, ne_label_part2 = np.split(ne_label, [tidx])

    X_train = np.concatenate((da_data_part1, ne_data_part1),0)
    X_test = np.concatenate((da_data_part2, ne_data_part2), 0)

    Y_train = np.concatenate((da_label_part1, ne_label_part1),0)
    Y_test  = np.concatenate((da_label_part2, ne_label_part2),0)

    ## reshape, note in the binary classification case Y should be reshaped in 1 dimension.
    train = tuple_dataset.TupleDataset(X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2]), Y_train)
    test = tuple_dataset.TupleDataset(X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]), Y_test)

    # caching the data
    if loadFromCache == False:
        np.save("../data/Cache/trainX", X_train)
        np.save("../data/Cache/trainY", Y_train)
        np.save("../data/Cache/testX", X_test)
        np.save("../data/Cache/testY", Y_test)

    return train, test



    
if __name__ == "__main__":
    
    dataPath="/data/Bioinfo/niida_sim_dl/data/simlationData"
    train, test = genDLdata(dataPath, loadFromCache=False)
    print "@ Loading from cache okay!"

