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
num_attributes = list(input.drop(["SEX", "EDUCATION", "MARRIAGE",'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'], axis=1))
cat_attributes = list(input.iloc[:, 1:4]) + list(input.iloc[:,5:11])
print(num_attributes)
print(cat_attributes)

"""HVIS JEG HAR MED ONEHOT VIL MATRISEN FAA 6 EKSTRA KOLONNER, SOM KODEN IKKE HAANDTERER"""
input_pipeline = ColumnTransformer([
                                    ("scaler", StandardScaler(), num_attributes),
                                    ("onehot", OneHotEncoder(categories="auto"), cat_attributes)
                                    ],
                                    remainder="passthrough"
                                    )
input_prepared = input_pipeline.fit_transform(input)
print(input_prepared)

# exporting labels to a numpy array
labels = df.loc[:, df.columns == 'default payment next month'].to_numpy().ravel()

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
    batch_size=1000,
    learning_rate="adaptive",
    learning_rate_init=0.01,
    max_iter=1000,
    tol=1e-7,
    verbose=True,
)

reg = reg.fit(training_input, training_labels)
right_count = 0
# See some statistics
pred = reg.predict(test_input)
for i in range(len(pred)):
    if pred[i] <= 0.5:
        pred[i] = 0
    else:
        pred[i] = 1

    if pred[i] == test_labels[i]:
        right_count += 1
        #print('\033[92m' + f"person {i:4.0f} | guess:  {pred[i]:.0f}     true : {test_labels[i]}" + '\033[0m')
    else:
        pass
        #print('\033[91m' + f"person {i:4.0f} | guess:  {pred[i]:.0f}     true : {test_labels[i]}" + '\033[0m')
print(f"MSe = {sklearn.metrics.mean_squared_error(test_labels,pred)}")
print(f"R2 = {reg.score(test_input,test_labels)}")
print(f"Success rate: {right_count/len(pred)*100} %")
