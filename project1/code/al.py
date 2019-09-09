'''
Mitt forsøk på k-fold cross-validation
'''
from sklearn.utils import shuffle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed

import projectfunctions as pf

def k_fold_cross_validation(x, y, z, degree, k=10):
    evaluation_scores=np.zeros(k)

    #shuffle the data
    x_shuffle, y_shuffle, z_shuffle = shuffle(x, y, z, random_state=0)

    #split the data into k folds
    x_split = np.array(np.array_split(x_shuffle, k))
    y_split = np.array(np.array_split(y_shuffle, k))
    z_split = np.array(np.array_split(z_shuffle, k))

    #loop through the folds
    for i in range(k):
        #pick out the test fold from data
        x_test = x_split[i]
        y_test = y_split[i]
        z_test = z_split[i]

        #pick out the remaining data as training data
        mask = np.ones(x_split.shape, dtype=bool)
        mask[i] = False
        x_train = x_split[mask]
        y_train = y_split[mask]
        z_train = z_split[mask]

        #fit a model to the training set
        '''
        Her må vi bruke enten OLS, Ridges eller Lassos.
        Foreløpig OLS 5-grad, vil gjøre dette til en variabel
        '''
        X_train = pf.generate_design_2Dpolynomial(x_train, y_train, degree)
        beta = pf.least_squares(X_train, z_train)

        #evaluate the model on the test set
        '''
        Her må vi bruke modellen til å beregne z_tilde for (x_test, y_test)
        og sammenligne med z_test ved MSE eller R_2_score
        Foreløpig MSE
        '''
        X_test = pf.generate_design_2Dpolynomial(x_test, y_test, degree)
        z_fit = X_test @ beta

        evaluation_scores[i] = pf.mse(z_test, z_fit)

    return np.mean(evaluation_scores)

'''
plotter feil mot kompleksitet
'''

n = 100
degrees = np.arange(1,11)

x_random = np.random.uniform(0, 1, n)
x_sorted = np.sort(x_random, axis=0)

y_random = np.random.uniform(0, 1, n)
y_sorted = np.sort(y_random,axis=0)

#making an x and y grid
x_grid, y_grid = np.meshgrid(x_sorted,y_sorted)

#flatten x and y
x = x_grid.flatten()
y = y_grid.flatten()

#compute z and flatten it
z_grid = pf.frankefunction(x_grid, y_grid)
z = z_grid.flatten()

k_fold_mse = []
mse = []

for degree in degrees:
    #performing a k-fold cross-validation
    k_fold_mse.append(k_fold_cross_validation(x,y,z, degree))

    #fitting a model to all the data
    X = pf.generate_design_2Dpolynomial(x, y, degree)
    beta = pf.least_squares(X, z)
    z_model = X @ beta

    #computing the MSE when no train test split is used
    mse.append(pf.mse(z, z_model))

plt.plot(degrees, k_fold_mse,label="test")
plt.plot(degrees, mse,label="training")
plt.legend()
plt.show()


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
