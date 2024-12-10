# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 22:34:01 2024

@author: Sourav
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt

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
            y_pred_tr = knn1.predict(self.X)
            y_pred2 = knn2.predict(X_test)
            y_pred2_tr = knn2.predict(self.X)
            acc1 = accuracy_score(y_test,y_pred)
            acc1_tr = accuracy_score(self.y,y_pred_tr)
            acc2 = accuracy_score(y_test,y_pred2)
            acc2_tr = accuracy_score(self.y,y_pred2_tr)
            print("Get the accuracies:",acc1,"==>",acc2,"for k:",mid)
            self.acc[mid] = acc1
            self.acc[mid+1] = acc2
            if(acc1+acc1_tr>acc2+acc2_tr):
                self.low = mid+1;
            else:
                self.high = mid-1;
        print("Best k is:",mid)
        return self.acc;

path = "MBA.csv"
data = pd.read_csv(path)
#print(data.head())

#print(data.columns)

data = data.drop(['application_id','work_industry','race'],axis=1)
#data = data.drop(['race'],axis=1)
data['gender'] = np.where(data['gender']=="Male",1,0)
data['international'] = np.where(data['international']=="True",1,0)
unique_majors = np.unique(data['major'])
major_code = dict(zip(unique_majors,np.arange(1,len(unique_majors)+1)))

data['major'] = data['major'].map(major_code)
data['admission'] = np.where(data['admission']=="Admit",1,np.where(data['admission']=="Waitlist",2,0))

y = data['admission'].values
X = data.drop(['admission'],axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data prepared")
# Training through Elbo method
train_errors = []
test_errors = []

low = 1
high = 16

k_values = range(1, high+1)  # Test k from 1 to 20

start = time.time()
for k in k_values:
    # Create kNN model
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Train on the training set
    knn.fit(X_train, y_train)
    
    # Evaluate on the training and test sets
    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)
    
    # Compute error
    train_errors.append(1 - accuracy_score(y_train, y_train_pred))
    test_errors.append(1 - accuracy_score(y_test, y_test_pred))

# Find the best k based on minimum test error
best_k = k_values[np.argmin(test_errors)]

# 3. Output the results
print(f"Best k: {best_k-1}")
print(f"Training Error for Best k: {train_errors[best_k-1]}")
print(f"Test Error for Best k: {test_errors[best_k-1]}")
print(f"Accuracy score:{1-test_errors[best_k-1]}")
print(f"Number of updates is: {len(X_train)}")
print(f"Time taken by elbo method: {time.time()-start}")

#4. Training using fast_KNN
start = time.time()
fknn = fast_knn(X_train, y_train, 1, high)
acc_dict = fknn.find_best_k(X_test,y_test)
print(f"Time taken by fast-knn method: {time.time()-start}")

plt.plot(np.exp(test_errors)/np.arange(1,len(test_errors)+1))
plt.xlabel('k value in knn')
plt.ylabel('Test errors')
plt.title('Best k-value within a given range')
plt.savefig('knn best k finding')
plt.show()
'''iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

fknn = fast_knn(X_train, y_train, 1, len(y_train))
acc_dict = fknn.find_best_k(X_test,y_test)'''

                
                