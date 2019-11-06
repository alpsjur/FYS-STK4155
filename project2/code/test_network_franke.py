import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

#Setting up data

#testing NN on Franke's Function
n = 20
noise = 0.1

#set up intervalls for x and y used in training
x = np.random.uniform(0,1,n)
y = np.random.uniform(0,1,n)

#making an x and y grid
x_grid, y_grid = np.meshgrid(x, y)

#flatten x and y
x = x_grid.flatten()
y = y_grid.flatten()

#compute z and flatten it
z_grid = pf.frankefunction(x_grid, y_grid) + np.random.normal(0,noise,x_grid.shape)
z = z_grid.flatten()

X = np.array([x,y]).transpose()

trainingShare = 0.8
seed  = 42
X_train, X_test, z_train, z_test = train_test_split(
                                                    X,
                                                    z,
                                                    train_size=trainingShare,
                                                    test_size = 1-trainingShare,
                                                    random_state=seed
                                                    )

#data plotting

layers = [2,30,30,1]
n_epochs = 5
batch_size = 100
learning_rate = 0.1

network = NeuralNetwork(layers, regression=True)

network.train(X_train, z_train,n_epochs, batch_size, \
                learning_rate, X_test, z_test, test=True)
