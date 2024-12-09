# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 16:31:09 2024

@author: Sourav
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def agg_fn(arr):
    return np.max(arr)

class SISA:
    def __init__(self,models,agg_fn):
        self.models = models
        self.fn = agg_fn
        self.n_mod = len(self.models)
    def fit(self,X,y):
        self.X,self.y = X,y
        self.router = {}
        np.random.seed(42)
        mod_select = np.random.choice(self.n_mod,len(X))
        for ind,ch in enumerate(mod_select):
            if(self.router.get(ch) == None):
                self.router[ch] = []
            self.router[ch].append(ind)
        
        ###Training each model
        for mod in range(self.n_mod):
            X_new,y_new = X[self.router[mod]],y[self.router[mod]]
            self.models[mod].fit(X_new,y_new)
        print("Training complete")
    def predict(self,X):
        est = np.zeros((self.n_mod,len(X)))
        ret_val = np.zeros(len(X))
        for mod in range(self.n_mod):
            est[mod] = self.models[mod].predict(X)
        for i in range(len(X)):
            unique,count = np.unique(est[:,i],return_counts=True)
            ret_val[i] = np.argmax(count)
        return ret_val
    
    def unlearn(self,indices):
        self.retrain = {}
        print("Initial members of each router set")
        for mod in range(self.n_mod):
            print('mod:',len(self.router[mod]))
        for i in range(len(indices)):
            for mod in range(self.n_mod):
                set_ = self.router[mod]
                if indices[i] in set_:
                    self.router[mod].remove(indices[i]);
                    self.retrain[mod] = True
        print("Data points removed")
        print("Final members of each router set")
        for mod in self.retrain:
            X_new,y_new = self.X[self.router[mod]],self.y[self.router[mod]]
            #print("X_new:",X_new)
            #print("y_new:",y_new)
            print('mod:',len(X_new))
            self.models[mod].fit(X_new,y_new)

'''X = np.array([[2,3],
     [1,1],
     [1,2],
     [6,7],
     [2,4]])

y = np.array([[1],[0],[0],[1],[0]])

models = [DecisionTreeClassifier(),DecisionTreeClassifier()]

unlearn = SISA(models, agg_fn)
unlearn.fit(X,y)

test_X = np.array([[1,2]])
test_y = np.array([0])

print(unlearn.predict(test_X))
print(test_y)

print("Router values:",unlearn.router)

unlearnt_index = len(X)-1
unlearn.unlearn([unlearnt_index])

print(unlearn.predict(test_X))
print(test_y)'''
                    
                
            
        