import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns
import projectfunctions as pf
import plottingfunctions as plf
import iofunctions as io

sns.set()
sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

figdir = "../figures/"
datadir = "../data/"

# Load the terrain
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

#plotting

"plotting MSE vs degree for terrain data using OLS"
reg = pf.ridge_regression
hyperparam = 0
degrees = np.linspace(0, 15, 16, dtype=int)
print(degrees)

filename = datadir + "realData.txt"
io.write_test_vs_degree_boot(filename, x, y, z,  reg, degrees, hyperparam ,show_bias_var=False, plot_r2=False)

"""
fig = plt.figure()
ax = fig.add_subplot(111)

plf.plot_test_vs_degree_boot(ax, x, y, z,  reg, max_degree, hyperparam, linewidth=2)
ax.tick_params(axis='both', labelsize=14)
ax.set_xlabel('degree', fontsize=18)
ax.set_ylabel('MSE', fontsize=18)

plt.savefig(figdir+'mseVSdegreeOLS_terrain.pdf')

"plotting MSE vs degree for terrain data using Ridge for multiple lambda"
reg = pf.ridge_regression
hyperparams = list(np.logspace(-5, -1, 5))
hyperparams.insert(0, 0)
max_degree = 20
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

plf.plot_test_vs_degree_multiple_lambda(ax2, x, y, z, reg, max_degree, hyperparams)
ax2.legend(frameon=False, fontsize=14)
ax2.set_xlabel("Degrees", fontsize=18)
ax2.set_ylabel("MSE", fontsize=18)

plt.savefig(figdir+"lambdavsdegreesRIDGE_terrain.pdf")


plt.show()
"""
'''
X = pf.generate_design_2Dpolynomial(x, y, degree=degree)
beta = reg(X, z, hyperparam=hyperparam)
z_pred = X @ beta

z_pred_grid = z_pred.reshape(x_grid.shape)

plt.figure()
plt.title('Terrain over Oslo fjord')
plt.imshow(oslo_data, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')

fig = plt.figure()
ax = fig.gca(projection="3d")

# Plot the surface.
surf = ax.plot_surface(x_grid, y_grid, oslo_data,
                        cmap=cm.coolwarm,
                        linewidth=0,
                        antialiased=False,
                        alpha = 0.5,
                        )

# Add a color bar which maps values to colors.
fig.colorbar(surf,
            shrink=0.5,
            aspect=5
            )

fig2 = plt.figure()
ax2 = fig2.gca(projection="3d")

# Plot the surface.
surf = ax2.plot_surface(x_grid, y_grid, z_pred_grid,
                        cmap=cm.coolwarm,
                        linewidth=0,
                        antialiased=False,
                        alpha = 0.5,
                        )

# Add a color bar which maps values to colors.
fig2.colorbar(surf,
            shrink=0.5,
            aspect=5
            )

plt.show()
'''
