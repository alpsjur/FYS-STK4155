import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append("class/")
from Regression import Logistic
import projectfunctions as pf


sns.set()
sns.set_style("whitegrid")
sns.set_palette("husl")

filepath = "../data/input/"
filename = "default_of_credit_card_clients"

df = pd.read_pickle(filepath + filename + "_clean.pkl")
print(df.head())

data = df.to_numpy()
designMatrix = data[:, :-1]
labels = data[:, -1]

learning_rate = 0.1

logreg = Logistic(designMatrix, labels, learning_rate)
