import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Position_Salaries.csv')

X = data.iloc[:, 1:2].values
Y = data.iloc[:, 2].values

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_poly,Y)

plt.scatter(X, Y, color='red')
plt.plot(X, lin_reg.predict(poly_reg.fit_transform(X)),color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Position Level in Firm')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, Y, color='red')
plt.plot(X_grid, lin_reg.predict(poly_reg.fit_transform(X_grid)),color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Position Level in Firm')
plt.ylabel('Salary')
plt.show()

lin_reg.predict(poly_reg.fit_transform(6.5))