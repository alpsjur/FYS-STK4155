import pandas as pd
import numpy as np
from numpy import vectorize
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn import linear_model
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

import sys
sys.path.append("class/")
from NeuralNetwork import NeuralNetwork
import projectfunctions as pf

sns.set()
sns.set_style("whitegrid")
sns.set_palette("husl")

class ReLU:
    def __init__(self):
        return

    @staticmethod
    @vectorize
    def __call__(z):
        a = 0.01
        if z <= 0:
            return a*z
        else:
            return z

    @staticmethod
    @vectorize
    def derivative(z):
        a = 0.01
        if z <= 0:
            return a
        else:
            return 1

class Sigmoid:
    def __init__(self):
        return

    @staticmethod
    @vectorize
    def __call__(z):
        return np.exp(z)/(1+np.exp(z))

    @staticmethod
    @vectorize
    def derivative(z):
        return np.exp(z)/(1 + np.exp(z))**2

#testing NN on Franke's Function
def generate_data(n, noise):
    x = np.linspace(0,1,n)
    y = np.linspace(0,1,n)

    #making an x and y grid
    x_grid, y_grid = np.meshgrid(x, y)

    #flatten x and y
    x = x_grid.flatten()
    y = y_grid.flatten()

    #compute z and flatten it
    z_grid = pf.frankefunction(x_grid, y_grid)
    z = z_grid.flatten() + np.random.normal(0,noise,len(x))

    X = np.array([x,y]).transpose()
    return X, z, x_grid, y_grid, z_grid

n = 20
noise = 0#.1


X, z, x_grid, y_grid, z_grid = generate_data(n, noise)

trainingShare = 0.8
#seed  = 42
X_train, X_test, z_train, z_test = train_test_split(
                                                    X,
                                                    z,
                                                    train_size=trainingShare,
                                                    test_size = 1-trainingShare,
                                                    #random_state=seed
                                                    )

input_neurons = X.shape[1]
layers = [input_neurons, 100, 20, 1]
n_epochs = 500
batch_size = 20
learning_rate = 0.1
regularisation = 0.0001

network = NeuralNetwork(layers, ReLU)

network.train(X_train, z_train, learning_rate, n_epochs, batch_size, \
              X_test, z_test, test='mse', regularisation=regularisation)

z_pred = network.predict_probabilities(X)
z_pred = z_pred.reshape(n,n)


fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot_surface(x_grid,y_grid,z_grid,cmap=cm.coolwarm)
ax.plot_wireframe(x_grid,y_grid,z_pred)

fig=plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot_surface(x_grid,y_grid,np.abs(z_grid-z_pred))
plt.show()


plt.show()
