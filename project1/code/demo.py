from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from random import random, seed

import projectfunctions as pf

n = 100
noise = 1
k = 5
reg = pf.ridge_regression
hyperparam = 0   #lambda
max_degree = 15

degrees = np.arange(1,max_degree+1)

x_val = np.linspace(0,1,n)
y_val = np.linspace(0,1,n)

#making an x and y grid
x_grid, y_grid = np.meshgrid(x_val, y_val)

#flatten x and y
x = x_grid.flatten()
y = y_grid.flatten()

#compute z and flatten it
z_grid = pf.frankefunction(x_grid, y_grid) + np.random.normal(0,np.sqrt(noise),x_grid.shape)
z = z_grid.flatten()


#calculationg bias, variance, mse for each fold
'''
Her beregner vi bias og varians inne i hver fold
'''
k_fold_mse = np.zeros(len(degrees))
k_fold_bias = np.zeros(len(degrees))
k_fold_r2 = np.zeros(len(degrees))
k_fold_var = np.zeros(len(degrees))

for degree in degrees:
    [mse, r2, bias, var], betas = pf.k_fold_cross_validation(x, y, z, reg, degree=degree, hyperparam=hyperparam, k=k)

    k_fold_mse[degree-1]=mse
    k_fold_r2[degree-1]=r2
    k_fold_bias[degree-1]=bias
    k_fold_var[degree-1]=var

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(degrees, k_fold_mse
        ,label='test mse'
        )
ax.plot(degrees, k_fold_var
        ,label='variance'
        )
ax.plot(degrees, k_fold_bias
        ,label='bias^2'
        )
ax.plot(degrees, k_fold_var+k_fold_bias
        ,label='bias^2+variance'
        )
ax.legend()
ax.set_xlabel('Degree')
ax.set_title('Bias and variance calculated in each fold')

#calculationg bias, variance, mse globaly
'''
Her beregner vi bias og varians globalt, ved å sammenligne med et test-sett
som  holdes utenfor k-fold cross-validation.
'''
k_fold_mse = np.zeros(len(degrees))
k_fold_bias = np.zeros(len(degrees))
k_fold_r2 = np.zeros(len(degrees))
k_fold_var = np.zeros(len(degrees))

for degree in degrees:
    [mse, r2, bias, var] = pf.bias_variance(x, y, z, reg, degree=degree, hyperparam=hyperparam, k=k)

    k_fold_mse[degree-1]=mse
    k_fold_r2[degree-1]=r2
    k_fold_bias[degree-1]=bias
    k_fold_var[degree-1]=var

fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
ax2.plot(degrees, k_fold_mse
        ,label='test mse'
        )
ax2.plot(degrees, k_fold_var
        ,label='variance'
        )
ax2.plot(degrees, k_fold_bias
        ,label='bias^2'
        )
ax2.plot(degrees, k_fold_var+k_fold_bias
        ,label='bias^2+variance'
        )
ax2.legend()
ax2.set_xlabel('Degree')
ax2.set_title('Bias and variance calculated globaly')


plt.show()
