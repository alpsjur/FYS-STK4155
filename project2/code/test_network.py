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

sns.set()
sns.set_style("whitegrid")
sns.set_palette("husl")

filepath = "../data/input/"
filename = "default_of_credit_card_clients"

df = pd.read_pickle(filepath + filename + "_clean.pkl")
#print(df.head())

data = df.to_numpy()
training_labels = data[:, -1]

training_input = data[:, :-1]

sc = StandardScaler()
training_input = sc.fit_transform(training_input)

layers = [23, 30, 30, 1]
n_epochs = 10
batch_size = 100
learning_rate = 0.1


network = NeuralNetwork(layers)
network.train(training_input, training_labels ,n_epochs, batch_size, learning_rate)
