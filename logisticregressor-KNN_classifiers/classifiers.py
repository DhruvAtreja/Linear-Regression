#Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#importing the datasets
path = '/Users/apple/Desktop/machine learning template/CLASSIFIER/'
X_train = pd.read_csv(path + 'Diabetes_XTrain.csv')
y_train = pd.read_csv(path + 'Diabetes_YTrain.csv')
X_test = pd.read_csv(path + 'Diabetes_Xtest.csv')

X_train = X_train.iloc[:,:].values
y_train = y_train.iloc[:,:].values
X_test = X_test.iloc[:,:].values

#Visualising the dataset
list = np.arange(len(X_train))
plt.scatter(list, y_train, color = 'red')
plt.title('DATASET')
plt.show()

#Featuring Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Training the dataset KNN
from sklearn.neighbors import KNeighborsClassifier
classifier1 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier1.fit(X_train, y_train)

#Training the dataset KNN
from sklearn.linear_model import LogisticRegression
classifier2 = LogisticRegression(random_state = 0)
classifier2.fit(X_train, y_train)

#Predicting the test set
y_pred1 = classifier1.predict(X_test)
y_pred2 = classifier2.predict(X_test)

#Visualizing the predicted values
list = np.arange(len(X_test))
plt.bar(list, y_pred1, color = 'red', label = 'KNN')
plt.bar(list, y_pred2, color = 'blue', label = 'LOGISTICS REGRESSION')
plt.title('DIABETIC OR NOT')
plt.legend()
plt.show()