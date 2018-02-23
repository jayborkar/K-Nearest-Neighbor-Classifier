#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 18:36:59 2018

@author: jayborkar
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

iris = datasets.load_iris()
data2 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])


X_train1,X_test1,y_train1, y_test1 = train_test_split(data2[['sepal length (cm)', 'sepal width (cm)']], data2['target'], test_size=0.3)

results = []
for k in range(1, 51, 2):
    classifier= KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train1,y_train1)


    prediction = classifier.predict(X_test1)

    correct=np.where(prediction==y_test1, 1,0).sum()
    print("Correct prediction using ",k,"=", correct)

    accuracy=correct/len(y_test1)
    print("Accuracy=",accuracy)
    results.append([k, accuracy])
    
    
results = pd.DataFrame(results, columns=["k", "accuracy"])
plt.plot(results.k, results.accuracy)
plt.title("Value of k and corresponding classification accuracy") 
plt.show()