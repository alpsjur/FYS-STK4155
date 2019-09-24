from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from random import random, seed

import projectfunctions as pf

def plot_training(ax, x, y, z, max_degree, reg, hyperparam, noise=1):
    """
    Function for plotting mse evaluated only at the training data where
    ax = matplotlib.axis object
    x, y = coordinates
    z = data
    reg = regression function reg(X, data, hyperparam)
    max_degree = maximum degree of polynomial
    hyperparam = hyperparameter for calibrating model
    noise = standard deviation of the normal distributed noise
    """
    degrees = np.arange(1,max_degree+1)

    mse = []

    for degree in degrees:
        """Simple training with no folds for comparison"""
        X = pf.generate_design_2Dpolynomial(x, y, degree)
        beta = reg(X, z, hyperparam=hyperparam)
        z_model = X @ beta

        #computing the MSE when no train test split is used
        mse.append(pf.mse(z, z_model))

    ax.plot(degrees, mse
            ,label='training mse'
            )

def plot_test(ax, x, y, z, max_degree, reg, hyperparam, k=5, noise=1, show_bias_var=False):
    """
    Function for plotting the mse (and bias, variance)
    calculated using k-fold cross-validation, where
        ax = matplotlib.axis object
        reg = regression function reg(X, data, hyperparam)
        max_degree = maximum degree of polynomial
        hyperparam = hyperparameter for calibrating model
        k = number of folds for cross validation
        show_bias_var = if True the bias and variance will also be plotted
        noise = standard deviation of the normal distributed noise
    """
    degrees = np.arange(1,max_degree+1)

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


    #Plot mse
    ax.plot(degrees, k_fold_mse,label='test mse')

    #Plots bias and variance if show_bias_var is True
    if show_bias_var:
        ax.plot(degrees, k_fold_var
            ,label='variance'
            )
        ax.plot(degrees, k_fold_bias
            ,label='bias^2'
            )
        ax.plot(degrees, k_fold_var+k_fold_bias
            ,label='bias^2+variance'
            )


n = 50
noise = 1
k = 10
reg = pf.ridge_regression
max_degree = 15
#hyperparams = np.linspace(0,4,5)

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


'''
#ploting
fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)

for hyperparam in hyperparams:
    plot_test(ax1, x,y,z, max_degree, reg, hyperparam,noise=noise)

ax1.legend(['$\lambda={}$'.format(i) for i in hyperparams])
'''

fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)

plot_test(ax2, x,y,z, max_degree, reg, hyperparam=0
            ,show_bias_var=True
            )
plot_training(ax2, x,y,z, max_degree, reg, hyperparam=0)
ax2.legend()

plt.show()
