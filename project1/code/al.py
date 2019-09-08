'''
Mitt forsøk på k-fold cross-validation
Denne fungerer foreløpig ikke som ønsket :'( '
'''
import numpy as np
from sklearn.utils import shuffle

def k_fold_cross_validation(x, y, z, k=10):
    evaluation_scores=np.zeros(k)

    #flatten the data set and shuffle it
    x_shuffle, y_shuffle, z_shuffle = shuffle(x, y, z, random_state=0)

    #split the data into k folds, not necessarily equal length
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
        Her må vi bruke enten OLS, Ridges eller Lassoself.
        Foreløpig OLS 5-grad vil gjøre dette til en variabel
        '''
        X_train = pf.generate_design_2Dpolynomial(x_train, y_train, degree=5)
        beta = pf.least_squares(X_train, z_train)


        #evaluate the model on the test set
        '''
        Her må vi bruke modellen til å beregne z_tilde for (x_test, y_test)
        og sammenligne med z_test ved MSE eller R_2_score
        Foreløpig MSE
        '''
        X_test = pf.generate_design_2Dpolynomial(x_test, y_test, degree=5)
        z_fit = X_test @ beta

        evaluation_scores[i] = pf.mse(z_test, z_fit)

    return np.mean(evaluation_scores)

"""
Tester kode med å flate ut z
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed

import projectfunctions as pf


n = 20

x_random = np.random.uniform(0, 1, n)
x_sorted = np.sort(x_random, axis=0)

y_random = np.random.uniform(0, 1, n)
y_sorted = np.sort(y_random,axis=0)

#making an x and y grid that maches the size of z
x_grid, y_grid = np.meshgrid(x_sorted,y_sorted)

x = x_grid.flatten()
y = y_grid.flatten()

z = pf.frankefunction(x_grid, y_grid).flatten()

print(k_fold_cross_validation(x,y,z))

X = pf.generate_design_2Dpolynomial(x, y, degree=7)
beta = pf.least_squares(X, z)
z_model = X @ beta

mse_value = pf.mse(z, z_model)

print(mse_value)

#print(beta)
