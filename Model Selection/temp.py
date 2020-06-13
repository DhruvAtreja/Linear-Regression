# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('datasets.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1:].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#MULTIPLE LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_predMultiple = regressor.predict(X_test)
# Evaluating the Model Performance
from sklearn.metrics import r2_score
print("Multiple linear regression r^2: ")
print(r2_score(y_test, y_predMultiple))
print("\n")


#np.set_printoptions(precision=2)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

X = np.append(arr = np.ones((414, 1)).astype(int), values = X, axis = 1)
X_opt = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype=float)
import statsmodels.api as sm
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()

#---------------------------------------------------------------------------

#POLYNOMIAL REGRESSION
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_poly, y_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)

y_predPoly = lin_reg_2.predict( poly_reg.fit_transform(X_test))



# Evaluating the Model Performance
print("Polynomial regression r^2: ")
print(r2_score(y_test, y_predPoly))
print("\n")

#----------------------------------------------------------------------------

#DECISION TREE REGRESSION

from sklearn.tree import DecisionTreeRegressor
regressorT = DecisionTreeRegressor(random_state = 0)
regressorT.fit(X_train, y_train)

# Predicting the Test set results
y_predDecisionTree = regressorT.predict(X_test)

# Evaluating the Model Performance
print("Decision Tree regression r^2: ")
print(r2_score(y_test, y_predDecisionTree))
print("\n")


#-----------------------------------------------------------------------------

#Random Forest Regression

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressorF = RandomForestRegressor(n_estimators = 200, random_state = 0)
regressorF.fit(X_train, y_train.ravel())

# Predicting the Test set results
y_predForest = regressorF.predict(X_test)

# Evaluating the Model Performance
print("Random Forest regression r^2: ")
print(r2_score(y_test, y_predForest))
print("\n")

#-----------------------------------------------------------------------------

# SVR REGRESSION
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

# Training the SVR model on the Training set
from sklearn.svm import SVR
regressorSVR = SVR(kernel = 'rbf')
regressorSVR.fit(X_train, y_train)

# Predicting the Test set results
y_predSVR = sc_y.inverse_transform(regressorSVR.predict(sc_X.transform(X_test)))

print("SVR regression r^2: ")
print(r2_score(y_test, y_predSVR))
print("\n")












