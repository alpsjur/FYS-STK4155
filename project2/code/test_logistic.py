import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import sys
sys.path.append("class/")
from Regression import Logistic

#np.random.seed(42)


sns.set()
sns.set_style("whitegrid")
sns.set_palette("husl")

filepath = "../data/input/"
filename = "default_of_credit_card_clients"

df = pd.read_pickle(filepath + filename + "_clean.pkl")
#print(df.head())

data = df.to_numpy()
labels = data[:, -1]

designMatrix = data[:, :-1]

# Train-test split
trainingShare = 0.5
seed  = 42
designMatrix_train, designMatrix_test, labels_train, labels_test = train_test_split(
                                                                designMatrix,
                                                                labels,
                                                                train_size=trainingShare,
                                                                test_size = 1-trainingShare,
                                                                random_state=seed
                                                                )
# Input Scaling
sc = StandardScaler()
designMatrix_train = sc.fit_transform(designMatrix_train)

learning_rate = 1e-4

logreg = Logistic(designMatrix_train, labels_train)
logreg.construct_model(100, 1000, learning_rate)
model = logreg.fit()
accuracy = logreg.accuracy()
print(accuracy)
