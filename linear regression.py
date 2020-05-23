# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

X_Test = pd.read_csv('Linear_X_Test.csv')
X_Train = pd.read_csv('Linear_X_Train.csv')
Y_Train = pd.read_csv('Linear_Y_Train.csv')

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_Train, Y_Train)

Y_Test = regressor.predict(X_Test)
