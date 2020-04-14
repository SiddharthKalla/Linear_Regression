# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 18:29:09 2020

@author: leno
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1/3,random_state = 0)

"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

plt.scatter(X_train,y_train,color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('salary vs experience(training set)')
plt.xlabel('experience')
plt.ylabel('salary')
plt.show()

plt.scatter(X_test,y_test,color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('salary vs experience(training set)')
plt.xlabel('experience')
plt.ylabel('salary')
plt.show()