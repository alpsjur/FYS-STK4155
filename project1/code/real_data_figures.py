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

reg = pf.ridge_regression
hyperparam = 0
degree = 15

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

X = pf.generate_design_2Dpolynomial(x, y, degree=degree)
beta = reg(X, z, hyperparam=hyperparam)
z_pred = X @ beta

z_pred_grid = z_pred.reshape(x_grid.shape)

'''
plt.figure()
plt.title('Terrain over Oslo fjord')
plt.imshow(oslo_data, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
'''

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
