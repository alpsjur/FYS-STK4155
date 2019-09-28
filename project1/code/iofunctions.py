from sklearn.utils import shuffle
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
import projectfunctions as pf


def write_test_vs_degree_boot(filename, x, y, z,  reg, degrees, hyperparam ,show_bias_var=False, plot_r2=False):
    """
    Function for writing the mse vs complexity to file
    calculated using bootstrap, where
        filename = name of the output file
        reg = regression function reg(X, data, hyperparam)
        degreess = list or array of degrees
        hyperparam = hyperparameter for model
    """
    file = open(filename, 'a')
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2)

    for degree in degrees:
        [mse, r2, bias, var] = pf.bootstrap(x_train, x_test, y_train, y_test, z_train, z_test, reg, degree=degree, hyperparam=hyperparam)
        file.write(f'{degree} {mse}\n')
    file.close()

def read_file(filename):
    degree, mse = np.loadtxt(filename)
    return degree, mse 
