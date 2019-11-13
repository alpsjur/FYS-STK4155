import pandas as pd
import numpy as np
from numpy import vectorize
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn import linear_model
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

import sys
sys.path.append("class/")
import projectfunctions as pf
from NeuralNetwork import NeuralNetwork


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

def learning_schedule(decreaser, learning_rate_init=0.1):
    return learning_rate_init/(decreaser + 1)

np.random.seed(42)


sns.set()
sns.set_style("whitegrid")
sns.set_palette("husl")


filepath = "../data/input/"
filename = "default_of_credit_card_clients_altered_clean.pkl"

df = pd.read_pickle(filepath + filename)

# preparing designmatrix by scaling and using one hot encoding for cat data
designMatrix = df.loc[:, df.columns != 'default payment next month']
#column_indices = pf.pca(designMatrix, 1e-1)
#print(designMatrix.columns[column_indices])
#designMatrix = designMatrix.iloc[:, column_indices]

designMatrix_num = designMatrix.drop(["SEX", "EDUCATION", "MARRIAGE"], axis=1)
designMatrix_cat = designMatrix.iloc[:, 1:4]

num_attributes = list(designMatrix)
cat_attributes = list(designMatrix_cat)
design_pipeline = ColumnTransformer([
                                    ("scaler", StandardScaler(), num_attributes),
                                    ("onehot", OneHotEncoder(categories="auto"), cat_attributes)
                                    ],
                                    remainder="passthrough"
                                    )
designMatrix_prepared = design_pipeline.fit_transform(designMatrix)

# exporting labels to a numpy array
labels = df.loc[:, df.columns == 'default payment next month'].to_numpy().ravel()
seed = 42

input_neurons = designMatrix_prepared.shape[1]
layers = [input_neurons, 20, 20, 1]

network = NeuralNetwork(layers, ReLU())

rate_range = np.logspace(-4, -0.5, 25, dtype=float)
batch_range = np.logspace(0, 3 ,10, dtype=int)


# run tune hyperparameter funcition
df_tuned = pf.tune_hyperparameter(designMatrix_prepared, labels, network, seed,
                                  rate_range,
                                  batch_range,
                                  n_epochs=10,
                                  test=None
                                  )



# store results
datadir = "../data/output/NeuralNetwork/"
pf.create_directories(datadir)
filename = "neural_acc_auc_run.csv"
df_tuned.to_csv(datadir + filename)
print(df_tuned.head())
