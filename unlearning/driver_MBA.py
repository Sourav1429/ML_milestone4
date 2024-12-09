# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 10:23:06 2024

@author: Sourav
"""

import pandas as pd
import numpy as np
from SISA import SISA
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def agg_fn(arr):
    return np.max(arr)

def get_compete_model_pred(model,X_test):
    return model.predict(X_test)

data = pd.read_csv("MBA.csv")
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


corr = data.corr()

# Create a heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.savefig('The dependency of admission on various columns')
plt.show()

models = [DecisionTreeClassifier(),DecisionTreeClassifier(),DecisionTreeClassifier(),DecisionTreeClassifier()]
unlearn = SISA(models, agg_fn)

y = data['admission'].values
X = data.drop(['admission'],axis=1).values

print(X.shape)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state = 20)
unlearn.fit(X_train, y_train)

y_pred = get_compete_model_pred(unlearn, X_test)
print(y_pred)
#print("Length of y_pred",len(y_pred))
print(accuracy_score(y_pred,y_test))

#Random forgetting
samples_ = 0.3
total_len = len(X_train)
indices_to_unlearn = np.random.choice(total_len,size = int(samples_*total_len))
unlearn.unlearn(indices_to_unlearn)

y_pred = get_compete_model_pred(unlearn, X_test)
print(y_pred)
#print("Length of y_pred",len(y_pred))
print(accuracy_score(y_pred,y_test))

### Retraining from scratch
X_new_train,y_new_train = [],[]
i=0
for x,y in zip(X_train,y_train):
    if i not in indices_to_unlearn :
        X_new_train.append(x)
        y_new_train.append(y)
    i+=1

X_new_train = np.array(X_new_train)
y_new_train = np.array(y_new_train)
unlearn.fit(X_new_train, y_new_train)
y_pred = get_compete_model_pred(unlearn, X_test)
print(y_pred)
#print("Length of y_pred",len(y_pred))
print(accuracy_score(y_pred,y_test))


