# -*- coding: utf-8 -*-
"""
Created on Fri May 29 23:16:27 2020

@author: dhruv
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Train.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 5:6].values

datasettest = pd.read_csv('Test.csv')


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

y_pred = regressor.predict(datasettest)

X = np.append(arr = np.ones((1600, 1)).astype(int), values = X, axis = 1)
X_opt = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype=float)
import statsmodels.api as sm
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#All P values are 0