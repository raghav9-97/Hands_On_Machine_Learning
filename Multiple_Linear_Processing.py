import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('50_Startups.csv')
X = data.iloc[:, :-1].values
Y = data.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
encoder = LabelEncoder()
X[:, 3] = encoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]

from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int),values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(Y,X_opt)
result = regressor_OLS.fit()
result.summary()

X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(Y,X_opt)
result = regressor_OLS.fit()
result.summary()


X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(Y,X_opt)
result = regressor_OLS.fit()
result.summary()


X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(Y,X_opt)
result = regressor_OLS.fit()
result.summary()
