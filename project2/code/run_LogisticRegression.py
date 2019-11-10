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
    return learning_rate_init/(decreaser + 10)

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

# Train-test split
trainingShare = 0.8
seed  = 42
designMatrix_train, designMatrix_test, labels_train, labels_test = train_test_split(
                                                                designMatrix_prepared,
                                                                labels,
                                                                train_size=trainingShare,
                                                                test_size = 1-trainingShare,
                                                                random_state=seed
                                                                )

# %% Our code
logreg = LogisticRegression()

logreg.train(designMatrix_train, labels_train,
        learning_schedule=learning_schedule,
        n_epochs=20,
        minibatch_size=34,
        learning_rate_init=1
        )

model = logreg.fit(designMatrix_test)

# %% Scikit learn
reg = linear_model.LogisticRegression(solver="lbfgs", max_iter=4e2)
reg.fit(designMatrix_train, labels_train)

accuracy = logreg.accuracy(designMatrix_test, labels_test)
accuracy_sklearn = reg.score(designMatrix_test, labels_test)
mse = logreg.mse(model, labels_test)
r2 = logreg.r2(model, labels_test)
r2_sklearn = r2_score(labels_test, model)
bias = logreg.bias(model, labels_test)
variance = logreg.variance(model)
predictions = logreg.predict(designMatrix_test)
guess_rate = np.mean(predictions)

print("TEST")
print(f"ACCURACY           {accuracy}")
print(f"ACCURACY (SKLEARN) {accuracy_sklearn}")
print(f"MSE                {mse}")
print(f"R2                 {r2}")
print(f"R2 (SKLEARN)       {r2_sklearn}")
print(f"BIAS               {bias}")
print(f"VAR                {variance}")
print(f"GUESS RATE         {guess_rate}")

"""
# %% our code bootstrap
mse, r2, bias, variance = logreg.train_bootstrap(designMatrix_train, designMatrix_test,
                                            labels_train, labels_test,
                                            learning_schedule=learning_schedule,
                                            n_epochs=100,
                                            minibatch_size=32
                                            )
print("\nBOOTSTRAP")
print(f"MSE                {mse}")
print(f"R2                 {r2}")
print(f"BIAS               {bias}")
print(f"VAR                {variance}")
"""
