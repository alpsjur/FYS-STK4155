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

def bias(funk,model):
    n = len(funk)
    #tror egentlig dette er feil bias
    error = np.sum((funk - np.mean(model))**2)/n
    return error

def variance(model):
    n = len(model)
    error = np.sum((model - np.mean(model))**2)/n
    return error


def frankefunction_noice(x,y,sigma):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    f = term1 + term2 + term3 + term4
    noice = np.random.normal(0,sigma**2,len(f))
    return f + noice

def k_fold_cross_validation(x, y, z, degree, k=10):
    """returns evaluation scores for k degrees including
    [mse[0:k],r2[0:k],bias[0:k],variance[0,k]]
    """
    evaluation_scores=np.zeros((4,k))


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

        evaluation_scores[0,i] = pf.mse(z_test, z_fit) #mse
        evaluation_scores[1,i] = pf.r2(z_test, z_fit) #r2
        evaluation_scores[2,i] = bias(pf.frankefunction(x_test,y_test), z_fit) #bias
        evaluation_scores[3,i] = variance(z_fit) #variance

    return evaluation_scores

'''
plotter feil mot kompleksitet
'''

n = 100
error = 0.2
degrees = np.arange(0,9)

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
z_grid = frankefunction_noice(x_grid, y_grid,error)
z = z_grid.flatten()

k_fold_mse = []
k_fold_bias = []
k_fold_r2 = []
k_fold_var = []
mse = []

print("mse   | bias  | var")
for degree in degrees:
    #performing a k-fold cross-validation
    k_fold_mse.append(np.mean(k_fold_cross_validation(x,y,z, degree)[0]))
    k_fold_r2.append(np.mean(k_fold_cross_validation(x,y,z, degree)[1]))
    k_fold_bias.append(np.mean(k_fold_cross_validation(x,y,z, degree)[2]))
    k_fold_var.append(np.mean(k_fold_cross_validation(x,y,z, degree)[3]))

    #fitting a model to all the data
    X = pf.generate_design_2Dpolynomial(x, y, degree)
    beta = pf.least_squares(X, z)
    z_model = X @ beta

    #computing the MSE when no train test split is used
    mse.append(pf.mse(z, z_model))
    print(f"{k_fold_mse[-1]:5.3f} | {k_fold_bias[-1]:5.3f} | {k_fold_var[-1]:5.3f}")

plt.plot(degrees, k_fold_var,label="variance")
plt.plot(degrees, k_fold_bias, label="bias")
plt.plot(degrees, k_fold_mse, label="mse")
plt.plot(degrees, np.array(k_fold_var)+np.array(k_fold_bias)+error**2 ,label="total error")
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
