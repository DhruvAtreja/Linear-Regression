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

