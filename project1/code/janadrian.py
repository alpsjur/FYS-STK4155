'''
Brukt Anna Linas k-fold cross-validation kode og implementert bias variance
'''
from sklearn.utils import shuffle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed

import projectfunctions as pf


'''
plotter feil mot kompleksitet
'''

n = 103
error = 0.2
degrees = np.arange(1, 25)

x_random = np.random.uniform(0, 1, n)
x_sorted = np.sort(x_random,
                    axis=0
                    )

y_random = np.random.uniform(0, 1, n)
y_sorted = np.sort(y_random,
                    axis=0
                    )

#making an x and y grid
x_grid, y_grid = np.meshgrid(x_sorted, y_sorted)

#flatten x and y
x = x_grid.flatten()
y = y_grid.flatten()

#compute z and flatten it
z_grid = pf.frankefunction(x_grid, y_grid,
                            noise=error
                            )
z = z_grid.flatten()

reg = pf.ridge_regression

k_fold_mse = []
k_fold_bias = []
k_fold_r2 = []
k_fold_var = []
mse = []

print("mse   | bias  | var")
for degree in degrees:
    """Performing a k-fold cross-validation on training data"""
    evaluation_scores = pf.k_fold_cross_validation(x, y, z, reg,
                                                    degree=degree,
                                                    hyperparam=0
                                                    )

    """Calculate bias, variance r2 and mse"""

    k_fold_mse.append(evaluation_scores[0])
    k_fold_r2.append(evaluation_scores[1])
    k_fold_bias.append(evaluation_scores[2])
    k_fold_var.append(evaluation_scores[3])


    """Simple training with no folds for comparison"""
    X = pf.generate_design_2Dpolynomial(x, y, degree)
    beta = reg(X, z)
    z_model = X @ beta

    #computing the MSE when no train test split is used
    mse.append(pf.mse(z, z_model))
    print(f"{k_fold_mse[-1]:5.3f} | {k_fold_bias[-1]:5.3f} | {k_fold_var[-1]:5.3f}")

plt.plot(degrees, k_fold_var,'--',
        label="variance"
        )
plt.plot(degrees, k_fold_bias,'--',
        label="bias"
        )
#plt.plot(degrees, k_fold_mse, ,label="k-fold mse")
plt.plot(degrees, mse,
        label="regular mse training"
        )
plt.plot(degrees, np.array(k_fold_var) + np.array(k_fold_bias),
        label="total error w/testing"
        )
plt.xlabel("degrees")
plt.legend()
plt.show()

###3D plot ###

'''
# Plot the surfacese
fig = plt.figure()
ax = fig.gca(projection="3d")

#reshape z_model to matrices so it can be plottet as a surface
z_model_grid = np.reshape(z_model,(n,n))

surf = ax.plot_surface(x_grid, y_grid, z_model_grid,
                        cmap=cm.Blues,
                        linewidth=0,
                        antialiased=False,
                        alpha = 0.5,
                        )

ax.scatter(x_grid, y_grid, z_grid,
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
'''
