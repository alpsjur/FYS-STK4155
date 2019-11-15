import pandas as pd
import numpy as np
from numpy import vectorize
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn import linear_model
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

import sys
sys.path.append("class/")
from Regression import LogisticRegression
from NeuralNetwork import NeuralNetwork
import projectfunctions as pf

np.random.seed(42)

##Load data##
filepath = "../data/input/"
filename = "default_of_credit_card_clients_altered_clean.pkl"

df = pd.read_pickle(filepath + filename)

# preparing designmatrix by scaling and using one hot encoding for cat data
designMatrix = df.loc[:, df.columns != 'default payment next month']
designMatrix.replace(to_replace=-1, value=0, inplace=True)
designMatrix.replace(to_replace=-2, value=0, inplace=True)

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

###Logistic Regression##

def learning_schedule(decreaser, learning_rate_init=0.1):
    return learning_rate_init/(decreaser + 10)

logreg = LogisticRegression()

logreg.train(designMatrix_train, labels_train,
        learning_schedule=learning_schedule,
        n_epochs=20,
        minibatch_size=870,
        learning_rate_init=0.15
        )

model = logreg.fit(designMatrix_test)

accuracy = logreg.accuracy(designMatrix_test, labels_test)
mse = logreg.mse(model, labels_test)
r2 = logreg.r2(model, labels_test)
bias = logreg.bias(model, labels_test)
variance = logreg.variance(model)
probabilities = logreg.fit(designMatrix_test)
predictions = logreg.predict(designMatrix_test)
guess_rate = np.mean(predictions)

#compute confusion matrix
true_negative, false_positive, false_negative, true_positive = confusion_matrix(labels_test, predictions).ravel()

print('Logistic Regression:')
print(f"ACCURACY           {accuracy}")
print(f"AUC                {roc_auc_score(labels_test, probabilities)}")
print('true positive: ',true_positive)
print('false positive: ',false_positive)
print('true negative: ', true_negative)
print('false negative: ', false_negative)


##Neural Network##
class Sigmoid:
    def __init__(self):
        return

    @staticmethod
    @vectorize
    def __call__(z):
        return np.exp(z)/(1+np.exp(z))

    @staticmethod
    @vectorize
    def derivative(z):
        return np.exp(z)/(1 + np.exp(z))**2

first_layer = designMatrix_prepared.shape[1]
layers = [first_layer, 20, 20, 1]
n_epochs = 8
batch_size = 1
learning_rate = 0.035


network = NeuralNetwork(layers, Sigmoid())
network.train(designMatrix_train, labels_train, learning_rate, n_epochs, batch_size, \
            designMatrix_test, labels_test, test='accuracy')

pred_prob = network.predict_probabilities(designMatrix_test)
pred = network.predict(designMatrix_test)

figurepath = '../figures/'

#compute confusion matrix
true_negative, false_positive, false_negative, true_positive = confusion_matrix(labels_test, pred).ravel()
#normalize

print('Neural Network:')
print(f"AUC                {roc_auc_score(labels_test, pred_prob)}")
print('true positive: ',true_positive)
print('false positive: ',false_positive)
print('true negative: ', true_negative)
print('false negative: ', false_negative)
