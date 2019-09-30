from sklearn.utils import shuffle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from imageio import imread
import pandas as pd

import projectfunctions as pf
import solvers as sol


import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns

reg = pf.ridge_regression
n = 20
noise = 0.1
max_degree = 12
hyperparam = 0
filename = "../data/test.txt"

#set up intervalls for x and y
x = np.linspace(0,1,n)
y = np.linspace(0,1,n)

#making an x and y grid
x_grid, y_grid = np.meshgrid(x, y)

#flatten x and y
x = x_grid.flatten()
y = y_grid.flatten()

#compute z and flatten it
z_grid = pf.frankefunction(x_grid, y_grid) + np.random.normal(0,noise,x_grid.shape)
z = z_grid.flatten()

sol.generate_train_vs_degree(x, y, z, reg, max_degree, hyperparam, filename)

"""
figdir = "../figures/"
sns.set()
sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

datadir = "../data/"
columns = ["degree", "MSE"]
df = pd.read_csv(datadir + "realData_OLS.txt",
                    names=columns,
                    delim_whitespace=True
                    )
df.set_index("degree", inplace=True)
print(df.head())
fig, ax = plt.subplots(1, 1)
ax.plot(df)
ax.set_xlabel("degree", fontsize=18)
ax.set_ylabel("MSE", fontsize=18)


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


reg = pf.lasso_regression
hyperparams = list(np.logspace(-6, -1, 6))
max_degree = 12
return_minimum = False

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

plf.plot_test_vs_degree_multiple_lambda(ax, x, y, z, reg, max_degree, hyperparams, return_minimum)
ax.legend(frameon=False, fontsize=18)
ax.set_xlabel("Degrees", fontsize=18)
ax.set_ylabel("MSE", fontsize=18)
plt.savefig(figdir+"lambdavsdegreesLASSO_REAL.pdf")

plt.show()
"""
