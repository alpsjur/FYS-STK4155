from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from random import random, seed

import projectfunctions as pf

def plot_bias_variance(ax, x, y, z, max_degree, reg, hyperparam, k=5, noise=1):
    degrees = np.arange(1,max_degree+1)

    k_fold_mse = []
    k_fold_r2 = []
    bias = []
    variance = []

    for degree in degrees:
        """Performing a k-fold cross-validation on training data"""
        evaluation_scores = k_fold_cross_validation_bias_variance(x,y,z, reg, degree=degree,hyperparam=hyperparam,k=k)

        """Calculate bias, variance r2 and mse"""
        k_fold_mse.append(evaluation_scores[0])
        k_fold_r2.append(evaluation_scores[1])
        bias.append(evaluation_scores[2])
        variance.append(evaluation_scores[3])

    #Plot mse
    ax.plot(degrees, k_fold_mse,label='k-fold mse')
    #ax.plot(degrees, error, label='error')
    ax.plot(degrees, bias, label='bias')
    ax.plot(degrees, variance, label='variance')


n = 50
noise = 1
k = 5
reg = pf.ridge_regression
max_degree = 15
hyperparam = 0

x_val = np.linspace(0,1,n)
y_val = np.linspace(0,1,n)

#making an x and y grid
x_grid, y_grid = np.meshgrid(x_val, y_val)

#flatten x and y
x = x_grid.ravel()
y = y_grid.ravel()

#compute z and flatten it
z_grid = pf.frankefunction(x, y)+ np.random.normal(0,np.sqrt(noise),x.shape)
z = z_grid.ravel()

pf.bias_variance(x, y, z, reg, degree=5 , hyperparam=hyperparam, k=k)

'''
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

plot_bias_variance(ax, x, y, z, max_degree, reg, hyperparam=hyperparam)
ax.legend()

plt.show()
'''
