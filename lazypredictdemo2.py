# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 03:26:38 2023

@author: cstan
"""

import pandas as pd
import numpy as np
df = pd.read_csv('https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv')
print(df.head(10))

from sklearn.model_selection import train_test_split

X = df.loc[:,df.columns!='Spending Score (1-100)']
y = df['Spending Score (1-100)']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)



#import lazypredict
from lazypredict.Supervised import LazyRegressor

from lazypredict.Supervised import LazyClassifier

multiple_ML_model = LazyRegressor(verbose=0,
                                  ignore_warnings=True,
                                  predictions=True)

models,predictions = multiple_ML_model.fit(X_train,
                                           X_test,
                                           y_train,
                                           y_test)
import matplotlib.pyplot as plt
#plt.bar(range(0,len(models)), predictions, align='center')
modelnames = models.axes[0].T.T.T.T.array
'''
plt.xticks(range(len(models)), models, size='small')
xaxisnums = np.arange(0,42)
#y = x.copy()
#x_ticks_labels = ['jan','feb','mar','apr']
x_ticks_labels = models.axes[0].T.T.T.T.array
fig, ax = plt.subplots(1,1)
ax.bar(bins=xaxisnums[-1], predictions)

# Set number of ticks for x-axis
ax.set_xticks(xaxisnums)
# Set ticks labels for x-axis
ax.set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=18)
plt.show()
#
'''
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
x_ticks_labels = np.array(modelnames)

ax.bar(x_ticks_labels,predictions.loc[1,:])
ax.set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=12)

plt.show()

multiple_ML_model=LazyClassifier()
from sklearn import datasets
# load data
data = datasets.load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
# fit all models
clf = LazyClassifier(predictions=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
modelnames = models.axes[0].T.T.T.T.array
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
x_ticks_labels = np.array(modelnames)

ax.bar(x_ticks_labels,models['ROC AUC'])
ax.set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=12)

print(models)