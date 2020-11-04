from __future__ import absolute_import
from .decision_stamp import decision_stamp
import numpy as np 

class adaboost():
    def __init__(self, n_estimators=10, clf=decision_stamp):
        self.clf = clf
        self.n_estimators = n_estimators
        self.alpha = list()
        self.scores = list()
        self.weak_clf = list()
        self.weight_error = list()
        
    def get_params(self, deep=True):
        return{'n_estimators':self.n_estimators}

    def set_parmas(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def beta_cal (self,epsolon):
        beta = 1/((1-epsolon)/epsolon)
        return beta
    
    def weight_cal(self, weight, label, prediction):
        e = (prediction == label).astype(int)
        epsolan = np.sum(weight*(1-e))
        beta = self.beta_cal(epsolan)
        if epsolan == 0:
            weight_new = weight
            beta = 0.000001
        else:
            weight_new = weight*beta**e
            weight_new = weight_new/sum(weight_new)
        alpha = np.log(1/beta)
        self.alpha.append(alpha)
        return weight_new

    def predict(self,data):
        row, _ = data.shape
        sum_y_pred = np.zeros((row))
        for i, clf in enumerate(self.weak_clf):
            sum_y_pred += self.alpha[i]*clf.predict(data)
        threshold = 0.5*sum(self.alpha)
        y_pred = np.array(list(sum_y_pred) >= threshold).astype(int)
        return y_pred

    def fit(self, data,label,weight= None):
        if weight==None:
            labels, counts = np.unique(label, return_counts=True)
            counts = 0.5*(1/np.array(counts))
            new_init_weight = [counts[np.where(labels == l)[0][0]] for l in list(label)]
            weight = np.array(new_init_weight)
        
        for _ in range(self.n_estimators):
            curr_clf = self.clf()
            curr_clf.fit(data,label,weight)
            curr_pred = curr_clf.predict(data)
            weight = self.weight_cal(weight,label,curr_pred)
            self.weak_clf.append(curr_clf)
            self.scores.append(self.score(data,label))  

    def score(self,data,label):
        pred = self.predict(data)
        equality_check = lambda y1,y2: y1 == y2
        total = sum(map(equality_check, label, pred))
        score = total/len(list(label))
        return score
