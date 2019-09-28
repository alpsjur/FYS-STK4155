from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
import seaborn as sns
import projectfunctions as pf

def plot_train_vs_degree(ax, x, y, z, reg, max_degree, hyperparam, plot_r2=False, **kwargs):
    '''
    Function for plotting mse when the model is evaluated on the training set
        ax = matplotlib.axis object
        reg = regression function reg(X, data, hyperparam)
        max_degree = maximum degree of polynomial
        hyperparam = hyperparameter for calibrating model
    '''
    degrees = np.arange(0,max_degree+1)
    error = []

    for degree in degrees:
        """Simple training with no cross validation"""
        X = pf.generate_design_2Dpolynomial(x, y, degree)
        beta = reg(X, z, hyperparam=hyperparam)
        z_model = X @ beta

        if plot_r2:
            #computing the MSE when no train test split is used
            error.append(pf.r2(z, z_model))
            label = 'R2 train'

        else:
            #computing the r2 score when no train test split is used
            error.append(pf.mse(z, z_model))
            label = 'mse train'

    ax.plot(degrees, error, **kwargs
            ,label=label
            )

def plot_train_vs_lambda(ax, x, y, z, reg, degree, hyperparams, r2=False, **kwargs):
    '''
    Function for plotting mse ws hyperparam when the model is evaluated on
    the training set
        ax = matplotlib.axis object
        reg = regression function reg(X, data, hyperparam)
        degree = degree of polynomial
        hyperparams = hyperparameter for calibrating model
    '''
    error = []

    for hyperparam in hyperparams:
        """Simple training with no cross validation"""
        X = pf.generate_design_2Dpolynomial(x, y, degree)
        beta = reg(X, z, hyperparam=hyperparam)
        z_model = X @ beta

        if not r2:
            #computing the MSE when no train test split is used
            error.append(pf.mse(z, z_model))
            label = 'mse train'

        if r2:
            #computing the r2 score when no train test split is used
            error.append(pf.r2(z, z_model))
            label = 'R2 train'

    ax.plot(hyperparams, error, **kwargs
            ,label=label
            )

def plot_test_vs_degree_kfold(ax, x, y, z,  reg, max_degree, hyperparam, plot_r2=False, **kwargs):
    degrees = np.arange(0,max_degree+1)

    kfold_error = []

    for degree in degrees:
        [mse, r2, bias, var] = pf.k_fold_cross_validation(x, y, z, reg, degree=degree, hyperparam=hyperparam)

        if plot_r2:
            kfold_error.append(r2)
            label = 'R2 test'

        else:
            kfold_error.append(mse)
            label = 'mse test'

    ax.plot(degrees, kfold_error, **kwargs
            ,label=label
            )

def plot_test_vs_degree_boot(ax, x, y, z,  reg, max_degree, hyperparam ,show_bias_var=False, plot_r2=False, **kwargs):
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

    boot_error = np.zeros(len(degrees))
    boot_bias = np.zeros(len(degrees))
    boot_var = np.zeros(len(degrees))

    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2)

    for degree in degrees:
        [mse, r2, bias, var] = pf.bootstrap(x_train, x_test, y_train, y_test, z_train, z_test, reg, degree=degree, hyperparam=hyperparam)

        boot_bias[degree]=bias
        boot_var[degree]=var

        if plot_r2:
            boot_error[degree]=r2
            label = 'r2 test'

        else:
            boot_error[degree]=mse
            label = 'mse test'

    if show_bias_var:
        label = 'mse'

    #Plot mse
    ax.plot(degrees, boot_error
            ,label=label
            , **kwargs
            )


    #Plots bias and variance if show_bias_var is True
    if show_bias_var:
        ax.plot(degrees, boot_var
            ,label='variance'
            ,ls='--'
            , **kwargs
            )
        ax.plot(degrees, boot_bias
            ,label='bias$^2$'
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
            ,label='mse test'
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

            k_fold_mse[degree]=mse
            k_fold_r2[degree]=r2
            k_fold_bias[degree]=bias
            k_fold_var[degree]=var


            #Plot mse
        ax.plot(degrees, k_fold_mse
            ,label=f"$\lambda$={hyperparam:.2g}"
            , **kwargs
            )

def plot_bias_confidence(ax, x, y, z, reg, degree, hyperparam, confidence=1.96 ,**kwargs):
    """
    Function plotting betas and their confidence intervalls
    """
    X = pf.generate_design_2Dpolynomial(x, y, degree)
    beta = reg(X, z, hyperparam=hyperparam)

    #weight = np.sqrt( np.diag( np.linalg.inv(X.T.dot(X))))*confidence
    weight = np.sqrt( np.diag( np.linalg.inv( X.T @ X ) ) )*confidence
    ax.errorbar(np.arange(1,len(beta)+1), beta
                ,yerr=weight
                ,fmt='o'
                ,**kwargs
                )

if __name__ == '__main__':
    sns.set()
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    n = 20
    noise = 0.1
    reg = pf.lasso_regression
    #reg = pf.lasso_regression
    max_degree = 12
    degree = 5
    hyperparam = 0
    #hyperparams = [0,1e-7, 1e-6, 1e-5, 1e-4, 1e-3] #np.logspace(-7,-2,6)
    hyperparams = list(np.logspace(-6, -1, 6))

    x_val = np.linspace(0,1,n)
    y_val = np.linspace(0,1,n)

    #making an x and y grid
    x_grid, y_grid = np.meshgrid(x_val, y_val)

    #flatten x and y
    x = x_grid.flatten()
    y = y_grid.flatten()

    #compute z and flatten it
    z_grid = pf.frankefunction(x_grid, y_grid) + np.random.normal(0,noise, x_grid.shape)
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
    ax3.legend(frameon=False, fontsize=14)
    ax3.set_xlabel("Degrees", fontsize=14)
    ax3.set_ylabel("MSE", fontsize=14)
    plt.savefig("../figures/lambdavsdegrees.pdf")

    plt.show()
