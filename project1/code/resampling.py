# -*- coding: utf-8 -*-
"""
Dette er bare en test av k-fold cross-validation resampling.
Flytter over til projectfunctions når det fungerer

NB: Virker ikke ennå!

[ part b) ]
"""
import numpy as np
from random import random, seed

import projectfunctions as pf

import seaborn as sns

#sklearn imports
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures

# Generating grid in x,y in [0, 1] with n=100 points
n = 100

degree = 5

x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)

# Initialize a KFold instance
k = 10
kfold = KFold(n_splits = k)

p = int(0.5*(degree+1)*(degree+2))

# Perform the cross-validation to estimate MSE
scores_KFold = np.zeros((p, k))

i = 0
for _ in range(p):
    j = 0
    for train_inds, test_inds in kfold.split(x):
        xtrain = x[train_inds]
        ytrain = y[train_inds]

        Xtrain = pf.generate_design_2Dpolynomial(xtrain, ytrain, degree=5)

        xtest = x[test_inds]
        ytest = y[test_inds]
        Xtest= pf.generate_design_2Dpolynomial(xtest, ytest, degree=5)

        xtrain, ytrain = np.meshgrid(xtrain, ytrain)
        ztrain = pf.frankefunction(xtrain, ytrain)

        xtest, ytest = np.meshgrid(xtest,ytest)
        ztest = pf.frankefunction(xtest, ytest)

        z_model= pf.least_squares(Xtrain, ztrain)
        z_pred =  pf.least_squares(Xtest, ztest)

        scores_KFold[i,j] = np.sum((z_model - z_pred)**2)/np.size(z_pred)

        j += 1
    i += 1


estimated_mse_KFold = np.mean(scores_KFold, axis = 1)
print(estimated_mse_KFold)
