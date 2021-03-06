import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Position_Salaries.csv')

X = data.iloc[:, 1:2].values
Y = data.iloc[:, 2].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,Y)

regressor.predict(6.5)

X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid),color = 'blue')
plt.title('Decision Tree Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
