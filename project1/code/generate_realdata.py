import numpy as np
from imageio import imread
import pandas as pd
import seaborn as sns

import projectfunctions as pf
import iofunctions as io



big_oslo_data = imread('../data/test_data_oslo.tif')

#pick out area around Oslo
oslo_data = big_oslo_data[0:1001,500:1501]

#get the number of points
n_y, n_x = np.shape(oslo_data)

#making an x and y grid (may want to define x and y differently)
x_grid, y_grid = np.meshgrid(np.linspace(0,1,n_x),np.linspace(0,1,n_y))

#downsizing
reduction = 10
oslo_data = oslo_data[::reduction,::reduction]
x_grid = x_grid[::reduction,::reduction]
y_grid = y_grid[::reduction,::reduction]


#flatten the data
x = x_grid.ravel()
y = y_grid.ravel()
z = oslo_data.ravel()


reg = pf.ridge_regression
hyperparams = list(np.logspace(-9, -2, 8))

min_degree = 0
max_degree = 20
degrees = np.linspace(min_degree, max_degree, (max_degree - min_degree) + 1, dtype=int)

return_minimum = False

datadir = "../data/ridge/"
filename = datadir + "realData_boot.txt"

#io.generate_train_vs_degree(x, y, z, reg, max_degree, hyperparam, filename)
#io.generate_train_vs_lambda(x, y, z, reg, degree, hyperparams, filename)
#io.generate_test_vs_degree_kfold(x, y, z,  reg, max_degree, hyperparam, filename)
#io.generate_test_vs_degree_boot(x, y, z,  reg, degrees, hyperparam, filename, return_minimum=False)
#io.generate_test_vs_lambda(x, y, z, reg, degree, hyperparams, filename)
io.generate_test_vs_degree_multiple_lambda(x, y, z,  reg, degrees, hyperparams, filename, return_minimum=False)
