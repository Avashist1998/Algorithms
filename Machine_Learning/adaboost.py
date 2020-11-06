from __future__ import absolute_import
import numpy as np 
from .decision_stamp import decision_stamp

class ada_boost():
    def __init__(self, n_estimators=10, clf=decision_stamp):
        self.clf = clf
        self.n_estimators = n_estimators
        self.alpha = list()
        self.scores = list()
        self.weak_clf = list()
        
    def get_params(self):
        return{'n_estimators':self.n_estimators}

    def set_parmas(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def beta_cal(self,epsolon):
        beta = epsolon/(1-epsolon)
        return beta
    
    def weight_cal(self, epsolan, weight, label, prediction):
        miss = (prediction != label).astype(int)
        beta = self.beta_cal(epsolan)
        weight_new = weight*[beta**(1-m) for m in miss]
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
        if weight is None:
            labels, counts = np.unique(label, return_counts=True)
            counts = 0.5*(1/np.array(counts))
            new_init_weight = [counts[np.where(labels == l)[0][0]] for l in list(label)]
            weight = np.array(new_init_weight)
        
        for _ in range(self.n_estimators):
            curr_clf = self.clf()
            curr_clf.fit(data,label,weight)
            curr_pred = curr_clf.predict(data)
            weight = self.weight_cal(curr_clf.weight_error,weight,label,curr_pred)
            self.weak_clf.append(curr_clf)
            self.scores.append(self.score(data,label))  

    def score(self,data,label):
        pred = self.predict(data)
        equality_check = lambda y1,y2: y1 == y2
        total = sum(map(equality_check, label, pred))
        score = total/len(list(label))
        return score
