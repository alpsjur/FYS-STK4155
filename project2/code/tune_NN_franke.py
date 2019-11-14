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
noise = 0.1


seed=42
X, z, x_grid, y_grid, z_grid = generate_data(n, noise)

input_neurons = X.shape[1]
layers = [input_neurons, 100, 20, 1]

network = NeuralNetwork(layers, ReLU())

rate_range = np.logspace(-4, -1, 25, dtype=float)
batch_range = np.logspace(0, 3, 10, dtype=int)


# run tune hyperparameter funcition
df_tuned = pf.tune_hyperparameter_franke(X, z, network, seed,
                                  rate_range,
                                  batch_range,
                                  n_epochs=20,
                                  test=None
                                  )



# store results
datadir = "../data/output/NeuralNetwork/"
pf.create_directories(datadir)
filename = "neural_franke_mse_epochs20.csv"
df_tuned.to_csv(datadir + filename)
print(df_tuned.head())
