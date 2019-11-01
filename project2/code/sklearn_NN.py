import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn import linear_model
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
import sklearn.neural_network
import sklearn.model_selection
import sklearn.metrics

import sys

sns.set()
sns.set_style("whitegrid")
sns.set_palette("husl")

filepath = "../data/input/"
filename = "default_of_credit_card_clients"

df = pd.read_pickle(filepath + filename + "_clean.pkl")

# preparing designmatrix by scaling and using one hot encoding for cat data
input = df.loc[:, df.columns != 'default payment next month']
num_attributes = list(input.drop(["SEX", "EDUCATION", "MARRIAGE"], axis=1))
cat_attributes = list(input.iloc[:, 1:4])

"""HVIS JEG HAR MED ONEHOT VIL MATRISEN FAA 6 EKSTRA KOLONNER, SOM KODEN IKKE HAANDTERER"""
input_pipeline = ColumnTransformer([
                                    ("scaler", StandardScaler(), num_attributes),
                                    #("onehot", OneHotEncoder(categories="auto"), cat_attributes)
                                    ],
                                    remainder="passthrough"
                                    )
input_prepared = input_pipeline.fit_transform(input)

# exporting labels to a numpy array
labels = df.loc[:, df.columns == 'default payment next month'].to_numpy().ravel()

layers = [23, 30, 30, 1]
n_epochs = 10
batch_size = 100
learning_rate = 0.1

trainingShare = 0.8
seed  = 42
training_input, test_input, training_labels, test_labels = train_test_split(
                                                                input_prepared,
                                                                labels,
                                                                train_size=trainingShare,
                                                                test_size = 1-trainingShare,
                                                                random_state=seed
)

reg = sklearn.neural_network.MLPRegressor(
    hidden_layer_sizes=(30,30),
    activation='logistic',
    batch_size=100,
    learning_rate="adaptive",
    learning_rate_init=0.1,
    max_iter=10,
    tol=1e-4,
    verbose=True,
)

reg = reg.fit(training_input, training_labels)

# See some statistics
pred = reg.predict(test_input)
for i in range(len(pred)):
    if pred[i] <= 0.5:
        print(f"person {i:4.0f}  : non-risk  (0)")
    else:
        print(f"person {i:4.0f}  : risk      (1)")

print(f"MSe = {sklearn.metrics.mean_squared_error(test_labels,pred)}")
print(f"R2 = {reg.score(test_input,test_labels)}")
