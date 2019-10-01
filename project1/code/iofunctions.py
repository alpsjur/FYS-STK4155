from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
import seaborn as sns
import projectfunctions as pf


def generate_train_vs_degree(x, y, z, reg, max_degree, hyperparam, filename):
    '''
    Function for plotting mse when the model is evaluated on the training set
        ax = matplotlib.axis object
        reg = regression function reg(X, data, hyperparam)
        max_degree = maximum degree of polynomial
        hyperparam = hyperparameter for calibrating model
    '''
    degrees = np.arange(0, max_degree+1)
    outfile = open(filename, "a")
    outfile.write("degree mse r2\n")
    for degree in degrees:
        """Simple training with no cross validation"""
        X = pf.generate_design_2Dpolynomial(x, y, degree)
        beta = reg(X, z, hyperparam=hyperparam)
        z_model = X @ beta
        mse = pf.mse(z, z_model)
        r2 = pf.mse(z, z_model)
        outstring = f"{degree} {mse} {r2}\n"
        outfile.write(outstring)
    outfile.close()


def generate_train_vs_lambda(x, y, z, reg, degree, hyperparams, filename):
    '''
    Function for plotting mse ws hyperparam when the model is evaluated on
    the training set
        ax = matplotlib.axis object
        reg = regression function reg(X, data, hyperparam)
        degree = degree of polynomial
        hyperparams = hyperparameter for calibrating model
    '''
    outfile = open(filename, "a")
    outfile.write("lambda mse r2\n")
    for hyperparam in hyperparams:
        """Simple training with no cross validation"""
        X = pf.generate_design_2Dpolynomial(x, y, degree)
        beta = reg(X, z, hyperparam=hyperparam)
        z_model = X @ beta
        mse = pf.mse(z, z_model)
        r2 = pf.mse(z, z_model)
        outstring = f"{hyperparam} {mse} {r2}\n"
        outfile.write(outstring)
    outfile.close()

def generate_test_vs_degree_kfold(x, y, z,  reg, max_degree, hyperparam, filename):
    degrees = np.arange(0, max_degree+1)

    outfile = open(filename, "a")
    outfile.write("degree mse r2 bias var\n")
    for degree in degrees:
        [mse, r2, bias, var] = pf.k_fold_cross_validation(x, y, z, reg, degree=degree, hyperparam=hyperparam)
        outstring = f"{degree} {mse} {r2} {bias} {var}\n"
        outfile.write(outstring)
    outfile.close()

def generate_test_vs_degree_boot(x, y, z,  reg, degrees, hyperparam, filename, return_minimum=True):
    """
    Function for plotting the mse (and bias, variance) vs complexity
    calculated using bootstrap, where
        ax = matplotlib.axis object
        reg = regression function reg(X, data, hyperparam)
        max_degree = maximum degree of polynomial
        hyperparam = hyperparameter for model
        show_bias_var = if True the bias and variance will also be plotted
    """

    boot_error = np.zeros(len(degrees))

    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2)

    outfile = open(filename, "a")
    outfile.write("degree mse r2 bias var\n")
    for degree in degrees:
        [mse, r2, bias, var] = pf.bootstrap(x_train, x_test, y_train, y_test, z_train, z_test, reg, degree=degree, hyperparam=hyperparam)
        outstring = f"{degree} {mse} {r2} {bias} {var}\n"
        outfile.write(outstring)
    outfile.close()

    if return_minimum:
        return  [min(boot_error),np.argmin(boot_error)]


def generate_test_vs_lambda(x, y, z, reg, degree, hyperparams, filename):
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

    outfile = open(filename, "a")
    outfile.write("lambda mse r2 bias var\n")
    for hyperparam in hyperparams:
        [mse, r2, bias, var] = pf.bootstrap(x_train, x_test, y_train, y_test, z_train, z_test, reg, degree=degree, hyperparam=hyperparam)
        outstring = f"{hyperparam} {mse} {r2} {bias} {var}\n"
        outfile.write(outstring)
    outfile.close()

def generate_test_vs_degree_multiple_lambda(x, y, z,  reg, degrees, hyperparams, filename, return_minimum=True):
    """
    Function for plotting the mse vs complexity for multiple lambda
    calculated using bootstrap, where
        ax = matplotlib.axis object
        reg = regression function reg(X, data, hyperparam)
        max_degree = maximum degree of polynomial
        hyperparaml = list of hyperparameters for model

    """

    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2)

    if return_minimum:
        error = np.zeros((len(degrees),len(hyperparams)))



    hyper_index = 0
    for hyperparam in hyperparams:
        outfile = open(filename[:-4] + f"_lambda{hyperparam:.0e}.txt", "a")
        outfile.write("degree mse r2 bias var\n")
        for degree in degrees:
            [mse, r2, bias, var] = pf.bootstrap(x_train, x_test, y_train, y_test, z_train, z_test, reg, degree=degree, hyperparam=hyperparam)
            outstring = f"{degree} {mse} {r2} {bias} {var}\n"
            outfile.write(outstring)

            if return_minimum:
                error[degree,hyper_index] = mse
        hyper_index += 1
        outfile.close()

    if return_minimum:
        index = (np.array(np.where(error == error.min())).flatten())
        return [error.min(), degrees[index[0]], hyperparams[index[1]]]
