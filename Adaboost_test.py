import os
import numpy as np 
import pandas as pd 
from utils import my_train_test_split, encoder, file_prep
from machine_learning.adaboost import adaboost

PATH = os.getcwd()
DATAPATH = os.path.join(PATH, 'Dataset')
DATAFILES = os.listdir(DATAPATH)

FILESPATH = os.path.join(DATAPATH, DATAFILES[0])
X, y, features = file_prep(FILESPATH)
X_train, X_test, y_train, y_test = my_train_test_split(
    X, y, test_size=0.20, random_state=42)

def dict_printer(params):
    for key, value in params.items():
        print(key,value)

def main():
    my_adaboost = adaboost()
    params = my_adaboost.get_params()
    dict_printer(params)
    my_adaboost.fit(X_train, y_train)
    for i, clf in enumerate(my_adaboost.weak_clf):
        # dict_printer(clf.get_params())
        print(features[clf.feature_index])
        # print(i, my_adaboost.scores[i])

if __name__ == "__main__":
    main()