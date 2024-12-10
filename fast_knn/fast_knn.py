# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 22:34:01 2024

@author: Sourav
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
class fast_knn:
    def __init__(self,X,y,low,high):
        self.X = X
        self.y = y
        self.low = low
        self.high = high
        self.acc = {}
    def find_best_k(self,X_test,y_test):
        mid = None
        while(self.low<=self.high):
            mid = (self.low+self.high)//2
            knn1 = KNeighborsClassifier(n_neighbors=mid)
            knn2 = KNeighborsClassifier(n_neighbors=mid+1)
            knn1.fit(self.X,self.y)
            knn2.fit(self.X,self.y)
            y_pred = knn1.predict(X_test)
            y_pred2 = knn2.predict(X_test)
            acc1 = accuracy_score(y_test,y_pred)
            acc2 = accuracy_score(y_test,y_pred2)
            self.acc[mid] = acc1
            self.acc[mid+1] = acc2
            if(acc1>acc2):
                self.low = mid+1;
            else:
                self.high = mid-1;
        print("Best k is:",mid)
        return self.acc;

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

fknn = fast_knn(X_train, y_train, 1, len(y_train))
acc_dict = fknn.find_best_k(X_test,y_test)

                
                