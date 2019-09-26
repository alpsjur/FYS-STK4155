from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
import seaborn as sns
import projectfunctions as pf

def plot_train_vs_degree(ax, x, y, z, reg, max_degree, hyperparam, **kwargs):
    '''
    Function for plotting mse when the model is evaluated on the training set
        ax = matplotlib.axis object
        reg = regression function reg(X, data, hyperparam)
        max_degree = maximum degree of polynomial
        hyperparam = hyperparameter for calibrating model
    '''
    degrees = np.arange(0,max_degree+1)
    mse = []

    for degree in degrees:
        """Simple training with no cross validation"""
        X = pf.generate_design_2Dpolynomial(x, y, degree)
        beta = reg(X, z, hyperparam=hyperparam)
        z_model = X @ beta

        #computing the MSE when no train test split is used
        mse.append(pf.mse(z, z_model))

    ax.plot(degrees, mse, **kwargs
            ,label='mse training'
            )

def plot_train_vs_lambda(ax, x, y, z, reg, degree, hyperparams, **kwargs):
    '''
    Function for plotting mse ws hyperparam when the model is evaluated on
    the training set
        ax = matplotlib.axis object
        reg = regression function reg(X, data, hyperparam)
        degree = degree of polynomial
        hyperparams = hyperparameter for calibrating model
    '''
    mse = []

    for hyperparam in hyperparams:
        """Simple training with no cross validation"""
        X = pf.generate_design_2Dpolynomial(x, y, degree)
        beta = reg(X, z, hyperparam=hyperparam)
        z_model = X @ beta

        #computing the MSE when no train test split is used
        mse.append(pf.mse(z, z_model))

    ax.plot(hyperparams, mse, **kwargs
            ,label='mse training'
            )


def plot_test_vs_degree(ax, x, y, z,  reg, max_degree, hyperparam ,show_bias_var=False, **kwargs):
    """
    Function for plotting the mse (and bias, variance) vs complexity
    calculated using bootstrap, where
        ax = matplotlib.axis object
        reg = regression function reg(X, data, hyperparam)
        max_degree = maximum degree of polynomial
        hyperparam = hyperparameter for model
        show_bias_var = if True the bias and variance will also be plotted
    """
    degrees = np.arange(0,max_degree+1)

    k_fold_mse = np.zeros(len(degrees))
    k_fold_bias = np.zeros(len(degrees))
    k_fold_r2 = np.zeros(len(degrees))
    k_fold_var = np.zeros(len(degrees))

    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2)

    for degree in degrees:
        [mse, r2, bias, var] = pf.bootstrap(x_train, x_test, y_train, y_test, z_train, z_test, reg, degree=degree, hyperparam=hyperparam)

        k_fold_mse[degree-1]=mse
        k_fold_r2[degree-1]=r2
        k_fold_bias[degree-1]=bias
        k_fold_var[degree-1]=var


    #Plot mse
    ax.plot(degrees, k_fold_mse
            ,label='test mse'
            , **kwargs
            )

    #Plots bias and variance if show_bias_var is True
    if show_bias_var:
        ax.plot(degrees, k_fold_var
            ,label='variance'
            ,ls='--'
            , **kwargs
            )
        ax.plot(degrees, k_fold_bias
            ,label='bias^2'
            ,ls='--'
            , **kwargs
            )

def plot_test_vs_lambda(ax, x, y, z, reg, degree, hyperparams ,show_bias_var=False, **kwargs):
    """
    Function for plotting the mse (and bias, variance) vs hyperparam
    calculated using bootstrap, where
        ax = matplotlib.axis object
        reg = regression function reg(X, data, hyperparam)
        degree = degree of polynomial
        hyperparams = hyperparameters to plot against
        show_bias_var = if True the bias and variance will also be plotted
    """

    boot_mse = np.zeros(len(hyperparams))
    boot_bias = np.zeros(len(hyperparams))
    boot_r2 = np.zeros(len(hyperparams))
    boot_var = np.zeros(len(hyperparams))

    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2)

    counter = 0
    for hyperparam in hyperparams:
        [mse, r2, bias, var] = pf.bootstrap(x_train, x_test, y_train, y_test, z_train, z_test, reg, degree=degree, hyperparam=hyperparam)

        boot_mse[counter]=mse
        boot_r2[counter]=r2
        boot_bias[counter]=bias
        boot_var[counter]=var

        counter += 1

    #Plot mse
    ax.plot(hyperparams, boot_mse
            ,label='test mse'
            , **kwargs
            )

    #Plots bias and variance if show_bias_var is True
    if show_bias_var:
        ax.plot(hyperparams, boot_var
            ,label='variance'
            ,ls='--'
            , **kwargs
            )
        ax.plot(hyperparams, boot_bias
            ,label='bias^2'
            ,ls='--'
            , **kwargs
            )

def plot_test_vs_degree_multiple_lambda(ax, x, y, z,  reg, max_degree, hyperparams , **kwargs):
    """
    Function for plotting the mse vs complexity for multiple lambda
    calculated using bootstrap, where
        ax = matplotlib.axis object
        reg = regression function reg(X, data, hyperparam)
        max_degree = maximum degree of polynomial
        hyperparaml = list of hyperparameters for model

    """
    degrees = np.arange(0,max_degree+1)

    k_fold_mse = np.zeros(len(degrees))
    k_fold_bias = np.zeros(len(degrees))
    k_fold_r2 = np.zeros(len(degrees))
    k_fold_var = np.zeros(len(degrees))

    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2)

    for hyperparam in hyperparams:
        for degree in degrees:
            [mse, r2, bias, var] = pf.bootstrap(x_train, x_test, y_train, y_test, z_train, z_test, reg, degree=degree, hyperparam=hyperparam)

            k_fold_mse[degree-1]=mse
            k_fold_r2[degree-1]=r2
            k_fold_bias[degree-1]=bias
            k_fold_var[degree-1]=var


            #Plot mse
        ax.plot(degrees, k_fold_mse
            ,label='$\lambda$={}'.format(hyperparam)
            , **kwargs
            )

if __name__ == '__main__':
    sns.set()

    n = 20
    noise = 0.1
    reg = pf.ridge_regression
    max_degree = 15
    degree = 5
    hyperparam = 0
    hyperparams = np.logspace(-8,0,9)

    x_val = np.linspace(0,1,n)
    y_val = np.linspace(0,1,n)

    #making an x and y grid
    x_grid, y_grid = np.meshgrid(x_val, y_val)

    #flatten x and y
    x = x_grid.flatten()
    y = y_grid.flatten()

    #compute z and flatten it
    z_grid = pf.frankefunction(x_grid, y_grid) + np.random.normal(0,noise,x_grid.shape)
    z = z_grid.flatten()


    #fig1 = plt.figure()
    #ax1 = fig1.add_subplot(1,1,1)

    #plot_test_vs_degree(ax1, x, y, z, reg, max_degree, hyperparam, show_bias_var=True)
    #plot_train_vs_degree(ax1, x, y, z, reg, max_degree, hyperparam)
    #ax1.legend()

    #fig2 = plt.figure()
    #ax2 = fig2.add_subplot(1,1,1)

    #plot_test_vs_lambda(ax2, x, y, z, reg, degree, hyperparams)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(1,1,1)

    plot_test_vs_degree_multiple_lambda(ax3, x, y, z, reg, max_degree, hyperparams)
    ax3.legend()

    plt.show()
