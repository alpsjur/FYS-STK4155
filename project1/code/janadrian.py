from sklearn.utils import shuffle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from imageio import imread

import projectfunctions as pf
import plottingfunctions as plf


import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns
import projectfunctions as pf

figdir = "../figures/"
sns.set()
sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


# Load the terrain
big_oslo_data = imread('../data/test_data_oslo.tif')

#pick out area around Oslo
oslo_data = big_oslo_data[0:1001,500:1501]

#get the number of points
n_y, n_x = np.shape(oslo_data)

#making an x and y grid (may want to define x and y differently)
x_grid, y_grid = np.meshgrid(np.linspace(0,1,n_x),np.linspace(0,1,n_y))

x = x_grid.ravel()
y = y_grid.ravel()
z = oslo_data.ravel()


max_degree = 12
reg = pf.lasso_regression
hyperparams = list(np.logspace(-6, -1, 6))

fig6 = plt.figure()
ax6 = fig6.add_subplot(1,1,1)

plf.plot_test_vs_degree_multiple_lambda(ax6, x, y, z, reg, max_degree, hyperparams)
ax6.legend(frameon=False, fontsize=14)
ax6.set_xlabel("Degrees", fontsize=14)
ax6.set_ylabel("MSE", fontsize=14)
#plt.savefig(figdir+"lambdavsdegreesLASSO_REAL.pdf")

plt.show()
