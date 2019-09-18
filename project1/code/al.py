from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from random import random, seed

import projectfunctions as pf

def k_fold_cross_validation_bias_variance(x, y, z, reg, degree=5, hyperparam=0, k=5):
    """
    k-fold CV calculating evaluation scores: MSE, R2, variance, and bias for
    data trained on k folds with an aditional validation set.
    where
        x, y = coordinates
        z = data/model
        reg = regression function reg(X, data, hyperparam)
        degree = degree of polynomial
        hyperparam = hyperparameter for calibrating model
        k = number of folds for cross validation
    """
    x_, x_val, y_, y_val, z_, z_val = train_test_split(x,y,z,test_size=0.2,random_state=42)

    print(z_val)

    X_val = pf.generate_design_2Dpolynomial(x_val, y_val, degree=degree)
    z_val_fit = np.zeros((len(x_val),k))

    #shuffle the data
    x_shuffle, y_shuffle, z_shuffle = shuffle(x_, y_, z_)

    #split the data into k folds
    x_split = np.array_split(x_shuffle, k)
    y_split = np.array_split(y_shuffle, k)
    z_split = np.array_split(z_shuffle, k)

    MSE = []
    R2 = []

    #loop through the folds
    for i in range(k):
        #pick out the test fold from data
        x_test = x_split[i]
        y_test = y_split[i]
        z_test = z_split[i]

        # pick out the remaining data as training data
        # concatenate joins a sequence of arrays into a array
        # ravel flattens the resulting array
        x_train = np.concatenate(x_split[0:i] + x_split[i+1:]).ravel()
        y_train = np.concatenate(y_split[0:i] + y_split[i+1:]).ravel()
        z_train = np.concatenate(z_split[0:i] + z_split[i+1:]).ravel()

        #fit a model to the training set
        X_train = pf.generate_design_2Dpolynomial(x_train, y_train, degree=degree)
        beta = reg(X_train, z_train, hyperparam=hyperparam)

        #evaluate the model on the test set
        X_test = pf.generate_design_2Dpolynomial(x_test, y_test, degree=degree)
        z_fit = X_test @ beta

        #evaluate the model on the global test set
        z_val_fit[:,i] = X_val @ beta

        MSE.append(pf.mse(z_test, z_fit)) #mse
        R2.append(pf.r2(z_test, z_fit)) #r2

    #fra morten sin kode, error fungerer IKKE
    print(z_val.ravel().shape)
    print(z_val_fit.shape)
    error = np.mean( np.mean((z_val - z_val_fit)**2, axis=1, keepdims=True) )
    bias = np.mean( (z_val - np.mean(z_val_fit, axis=1, keepdims=True))**2 )
    variance = np.mean( np.var(z_val_fit, axis=1, keepdims=True) )

    return [np.mean(MSE), np.mean(R2), error, bias, variance]

def plot_bias_variance(ax, x, y, z, max_degree, reg, hyperparam, k=5, noise=1):
    degrees = np.arange(1,max_degree+1)

    k_fold_mse = []
    k_fold_r2 = []
    error = []
    bias = []
    variance = []

    for degree in degrees:
        """Performing a k-fold cross-validation on training data"""
        evaluation_scores = k_fold_cross_validation_bias_variance(x,y,z, reg, degree=degree,hyperparam=hyperparam,k=k)

        """Calculate bias, variance r2 and mse"""
        k_fold_mse.append(evaluation_scores[0])
        k_fold_r2.append(evaluation_scores[1])
        error.append(evaluation_scores[2])
        bias.append(evaluation_scores[3])
        variance.append(evaluation_scores[4])

    #Plot mse
    ax.plot(degrees, k_fold_mse,label='k-fold mse')
    #ax.plot(degrees, error, label='error')
    #ax.plot(degrees, bias, label='bias')
    #ax.plot(degrees, variance, label='variance')


n = 50
noise = 1
k = 10
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
z_grid = pf.frankefunction(x, y, noise=noise)
z = z_grid.ravel()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

plot_bias_variance(ax, x, y, z, max_degree, reg, hyperparam=hyperparam)
ax.legend()

plt.show()
