import pandas as pd
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn import linear_model
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, r2_score


import sys
sys.path.append("class/")
from Regression import LogisticRegression
import projectfunctions as pf


def learning_schedule(decreaser, learning_rate_init=0.1):
    return learning_rate_init/(decreaser + 1)

np.random.seed(42)


sns.set()
sns.set_style("whitegrid")
sns.set_palette("husl")


filepath = "../data/input/"
filename = "default_of_credit_card_clients_partial_clean.pkl"

df = pd.read_pickle(filepath + filename)

# preparing designmatrix by scaling and using one hot encoding for cat data
designMatrix = df.loc[:, df.columns != 'default payment next month']
column_indices = pf.pca(designMatrix, 1e-1)
print(designMatrix.columns[column_indices])
designMatrix = designMatrix.iloc[:, column_indices]

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
logreg = LogisticRegression()

start = 0.01
stop = 5
hyperparam_name = "learning_rate_init"  # which parameter to tune
""" Important to define type, i.e. dtype=int"""
hyperparam_range = np.linspace(start, stop, 100, dtype=float)

df_tuned = pf.tune_hyperparameter(designMatrix_prepared, labels, logreg, seed,
                                [hyperparam_name, hyperparam_range],
                                learning_schedule,
                                minibatch_size=34,
                                n_epochs=18,
                                )
datadir = "../data/output/LogisticRegression/"
pf.create_directories(datadir)
filename = hyperparam_name + f"_{start}_{stop}.csv"
df_tuned.to_csv(datadir + filename)
print(df_tuned.head())
