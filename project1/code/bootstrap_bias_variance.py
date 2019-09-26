import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

import projectfunctions as pf

np.random.seed(2019)

n = 20
noise = 0.1
reg = pf.ridge_regression
maxdegree = 15
hyperparam = 0
n_bootstraps = 200

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

    scores = pf.bootstrap(x_train, x_test, y_train, y_test, z_train, z_test, reg, degree=degree, hyperparam=hyperparam)

    polydegree[degree] = degree
    mse_test[degree] = scores[0]
    bias[degree] = scores[2]
    variance[degree] = scores[3]
    
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
