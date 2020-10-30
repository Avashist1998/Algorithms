# Read Chapter 10.1.1 from the textbook and code a 1 level decision tree algorithm

# The code is different from the book because it use multiporcessing to perfrom the decision stamp search
# An ideal use case for this file when working with large number of features. 

# take in input of s = (x1,y1) ... (xm,ym)
# goal is the find j* and theta* for equation 10.1 Ref paper
#initalize: F* = infinty
# for j = 1 ... d
# sort S using the j'th coordinate and denote
# x1,j <= x2,j <= ... <= xm,j <= xm+1,j = xm,j + 1
# F = sumi:yi=1 Di
# if F < F*
# F* = F, theata* = X1,j-1, j* = j
# for i =1 ... m
# F = F - yiDi
# if F< F* and xij not equal xi+1,j
# F* = F, theata* = 0.5 (xi,j + xi+1,j), j* = j
# return j*, theata*
# input of the system


import numpy as np

class Decision_Stamp():
    def __init__(self):
        # The index of the feature used to make classification
        self.feature_index = None
        # The threshold value that the feature should be measured against
        self.threshold = None
        # Pairity of the threshold
        self.pairty = 0


    def fit(self, X=None, y=None, w=None):
        '''
        X, y, W should all be numpy array
        X shape = [N,M] 
        Y shape = [1,N]
        W shape = [1,N]
        '''
        if (X is None and y is None):
            print("Improper input in the function")
            self.feature_index = 0
            self.threshold = 0
        else:
            F_star = float('inf')
            [row, col] = X.shape
            if (w == None):
                w = np.array([1]*row)
            for j in range(col):
                index = X[:,j].argsort()
                Xj = X[:,j][index]
                Yj = y[index]
                Dj = w[index]
                F = sum(Dj[Yj == 1])
                if F<F_star:
                    F_star = F
                    theta_star = Xj[0] - 1
                    j_star = j  
                for i in range(0,row-1):
                    F = F - Yj[i]*Dj[i]
                    if ((F<F_star) &  (Xj[i] != Xj[i+1])):
                        F_star = F
                        theta_star= 0.5*((Xj[i] + Xj[i+1]))
                        j_star=j
            self.feature_index = j_star
            self.threshold = theta_star
            prediction1 = 2*np.array(X[:,j_star] <= theta_star).astype(int)-1
            prediction2 = 2*np.array(X[:,j_star] > theta_star).astype(int)-1
            score1 = self.accuracy(prediction1, y)
            score2 = self.accuracy(prediction2, y)
            if (score2 > score1): self.pairty = 1

    def predict(self, X):
        if self.pairty:
            prediction = X[:,self.feature_index] > self.threshold
            prediction = np.array(prediction).astype(int)
        else:
            prediction = X[:,self.feature_index] <= self.threshold
            prediction = np.array(prediction).astype(int)

        prediction = 2*prediction - 1
        return prediction
    
    def accuracy(self, y, yHat):
        same_check = lambda y1,y2: y1 == y2
        total = sum(map(same_check, y, yHat))
        score = total/len(list(y))
        return score