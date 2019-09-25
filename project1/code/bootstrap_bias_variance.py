import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

import projectfunctions as pf

np.random.seed(2019)

n = 20
noise = 0.1
reg = pf.ridge_regression
maxdegree = 15
hyperparam = 0
n_boostraps = 100

x_val = np.linspace(0,1,n)
y_val = np.linspace(0,1,n)

#making an x and y grid
x_grid, y_grid = np.meshgrid(x_val, y_val)

#flatten x and y
x = x_grid.ravel()
y = y_grid.ravel()

#compute z and flatten it
z_grid = pf.frankefunction(x, y) + np.random.normal(0, noise, x.shape)
z = z_grid.ravel()

mse_test = np.zeros(maxdegree+1)
mse_train = np.zeros(maxdegree+1)
bias = np.zeros(maxdegree+1)
variance = np.zeros(maxdegree+1)
polydegree = np.zeros(maxdegree+1)
x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2)

for degree in range(maxdegree+1):
    #calculating train mse
    X_train = pf.generate_design_2Dpolynomial(x_train, y_train, degree=degree)
    beta_train = reg(X_train, z_train, hyperparam=hyperparam)
    z_train_pred = X_train @ beta_train
    mse_train[degree] = np.mean((z_train - z_train_pred)**2)

    #preforming bootstrap
    X_test = pf.generate_design_2Dpolynomial(x_test, y_test, degree=degree)
    z_pred = np.empty((z_test.shape[0], n_boostraps))
    for i in range(n_boostraps):
        x_, y_, z_ = resample(x_train, y_train, z_train)
        X = pf.generate_design_2Dpolynomial(x_, y_, degree=degree)
        beta = reg(X, z_, hyperparam=hyperparam)
        z_pred_temp = X_test @ beta
        z_pred[:, i] = z_pred_temp.ravel()

    z_test = np.reshape(z_test,(len(z_test),1))
    polydegree[degree] = degree
    mse_test[degree] = np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
    bias[degree] = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
    variance[degree] = np.mean( np.var(z_pred, axis=1, keepdims=True) )
    #print('Polynomial degree:', degree)
    #print('Error:', error[degree])
    #print('Bias^2:', bias[degree])
    #print('Var:', variance[degree])
    #print('{} >= {} + {} = {}'.format(error[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))

plt.plot(polydegree, mse_test
        ,label='mse test'
        )
plt.plot(polydegree, mse_train
        ,label='mse train'
        )
plt.plot(polydegree, bias
        ,label='bias'
        ,ls='--'
        )
plt.plot(polydegree, variance
        ,label='Variance'
        ,ls='--'
        )
#plt.plot(polydegree, bias+variance
#        ,label='bias+variance'
#        )
plt.xlabel('degree')
plt.legend()
plt.show()
