
# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Car_Purchasing_Data.csv', encoding='latin-1')
X = dataset.iloc[:, 4:-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train.reshape(-1,1))
X_test = sc_X.fit_transform(X_test)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)


y_p = regressor.predict(X_test)
y_p = sc_y.inverse_transform(y_p)

from sklearn.metrics import r2_score
r2_score(y_test, y_p)
# by checking r2 scores, we eliminate useless variables like gender
