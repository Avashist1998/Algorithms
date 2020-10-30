import os
import numpy as np 
import pandas as pd 
from utils import my_train_test_split, encoder
from Machine_Learning.Decision_Stramp import Decision_Stamp

def File_prep(path):
    FILEDF = pd.read_csv(path)
    cols = FILEDF.columns
    y = FILEDF[cols[-1]]
    X = FILEDF.drop(columns = cols[-1])
    X = X.to_numpy()
    y = y.to_numpy()
    y = encoder(y)
    return X, y, cols

def main():
    PATH = os.getcwd()
    DATAPATH = os.path.join(PATH, 'Dataset')
    DATAFILES = os.listdir(DATAPATH)
    my_Stamp = Decision_Stamp()
    for i, FILES in enumerate(DATAFILES):
        FILESPATH = os.path.join(DATAPATH, FILES)
        X, y, features = File_prep(FILESPATH)
        X_train, X_test, y_train, y_test = my_train_test_split(
            X, y, test_size=0.20, random_state=42)
        my_Stamp.fit(X_train, y_train)
        prediction = my_Stamp.predict(X_test)
        feature = features[my_Stamp.feature_index]
        score = my_Stamp.accuracy(prediction,y_test)
        print("The best feature is {} with the accuracy of {}.".format(feature, score))
        
    
if __name__ == "__main__":
    main()