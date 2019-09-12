from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn import linear_model

import projectfunctions as pf

import seaborn as sns


def linear_regression(X, data, hyperparam=0):
    p = len(X[0, :])
    beta = np.linalg.inv(X.T.dot(X) + hyperparam*np.identity(p)).dot(X.T).dot(data)
    return beta

def lasso_regression(X, data, hyperparam=1):
    reg = linear_model.Lasso(alpha=hyperparam)
    reg.fit(X, data)
    beta = reg.coef_
    return beta

def produce_table(data, header):
    """
    Spagetthi code producing a vertical laTEX table.
    data has shape
        [[x0, x1, x2],
         [y0, y1, y2],
         ...          ]
    where
    header = list/array
    orientation = vertical or horizontal where vertical has a horizontal header
    and horizontal has a vertical header.
    """
    tableString = ""
    n = len(data[:, 0])
    tableString += "\\begin{table}[htbp]\n"
    tableString += "\\begin{{tabular}}{{{0:s}}}\n".format("l"*n)
    # creating header
    for element in header:
        tableString += f"\\textbf{{{element}}} & "
    tableString = tableString[:-2] + "\\\\\n"
    # creating table elements
    for j in range(len(data[0, :])):
        for i in range(len(data[:, 0])):
            tableString += f"{data[i, j]:.2f} & "
        tableString = tableString[:-2] + "\\\\\n"
    tableString = tableString[:-4] + "\n"
    tableString += "\\end{tabular}\n"
    tableString += "\\end{table}\n"
    return tableString





sns.set()
sns.set_style("whitegrid")
sns.set_palette("husl")

header = ["x-direction", "y-direction"]
m = 3
n = 4
data = np.zeros((m, n))
data[0, :] = [1, 2, 3, 5]
data[1, :] = [10, 35, 64, 74]
#data = np.array([np.array(i) for i in data])

tableString = produce_table(data, header)
print(tableString)


fig = plt.figure()
ax = fig.gca(projection="3d")

# Generating grid in x,y in [0, 1] with n=100 points
n = 200

# generating design matrices for both coordinates
x_random = np.random.uniform(0, 1, n) #+ 0.1*np.random.randn(n, 1)
x_sorted = np.sort(x_random, axis=0)

y_random = np.random.uniform(0, 1, n) #+ 0.1*np.random.randn(n, 1)
y_sorted = np.sort(y_random,axis=0)

x_grid, y_grid = np.meshgrid(x_sorted,y_sorted)

z_true = pf.frankefunction(x_grid, y_grid)

X = pf.generate_design_2Dpolynomial(x_sorted, y_sorted, degree=5)
beta = lasso_regression(X, z_true, hyperparam=0.5)
z_model = X @ beta


mse_value = pf.mse(z_true, z_model)
r2_value = pf.r2(z_true, z_model)

print(f"MSE = {mse_value:.3f}")
print(f"R2 = {r2_value:.3f}")

# Plot the surface.
surf = ax.plot_surface(x_grid, y_grid, z_model,
                        cmap=cm.Blues,
                        linewidth=0,
                        antialiased=False,
                        alpha = 0.5,
                        )

ax.scatter(x_grid, y_grid, z_true,
                        #cmap=cm.coolwarm,
                        linewidth=0,
                        antialiased=False,
                        marker = '.',
                        s = 0.1,
                        label="data",
                        c='k'
                        )

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
# Add a color bar which maps values to colors.
fig.colorbar(surf,
            shrink=0.5,
            aspect=5,
            label="model"
            )

ax.legend()
plt.show()
