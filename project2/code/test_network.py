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

filepath = "../data/input/"
filename = "default_of_credit_card_clients"

df = pd.read_pickle(filepath + filename + "_partial_clean.pkl")
#print(df.head())

"""
data = df.to_numpy()
labels = data[:, -1]

input = data[:, :-1]

sc = StandardScaler()
input = sc.fit_transform(input)
"""

# preparing designmatrix by scaling and using one hot encoding for cat data
input = df.loc[:, df.columns != 'default payment next month']
column_indices = pf.pca(input)
print(column_indices)
input = input.iloc[:, column_indices]
num_attributes = list(input.drop(["SEX", "EDUCATION", "MARRIAGE"], axis=1))
cat_attributes = list(input.iloc[:, 1:4])
#num_attributes = list(input.drop(["SEX", "EDUCATION", "MARRIAGE",'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'], axis=1))
#cat_attributes = list(input.iloc[:, 1:4]) + list(input.iloc[:,5:11])

input_pipeline = ColumnTransformer([
                                    ("scaler", StandardScaler(), num_attributes),
                                    ("onehot", OneHotEncoder(categories="auto"), cat_attributes)
                                    ],
                                    remainder="passthrough"
                                    )
input_prepared = input_pipeline.fit_transform(input)

# exporting labels to a numpy array
labels = df.loc[:, df.columns == 'default payment next month'].to_numpy().ravel()

first_layer = input_prepared.shape[1]
layers = [first_layer, 20, 20, 1]
n_epochs = 5
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

network = NeuralNetwork(layers)
network.train(training_input, training_labels ,n_epochs, batch_size, \
                learning_rate, test_input, test_labels, test=True)

output = network.predict_probabilities(test_input)

plt.hist(output)
plt.xlim([0,1])
plt.title('Histogram of output from NN')
plt.xlabel('output')
plt.ylabel('count')
plt.show()
