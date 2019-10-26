import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append("class/")
from Regression import Logistic


sns.set()
sns.set_style("whitegrid")
sns.set_palette("husl")

filepath = "../data/input/"
filename = "default_of_credit_card_clients"

df = pd.read_pickle(filepath + filename + "_clean.pkl")
#print(df.head())

data = df.to_numpy()
labels = data[:, -1]

designMatrix_nointercept = data[:, :-1]
n, m = np.shape(designMatrix_nointercept)
intercept = np.ones((n, 1))
designMatrix = np.hstack((intercept, designMatrix_nointercept))
print(designMatrix)

learning_rate = 1e-4

logreg = Logistic(designMatrix, labels)
logreg.construct_model(10, 5, learning_rate)
model = logreg.fit()
print(model)
