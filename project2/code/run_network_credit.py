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
from NeuralNetwork import NeuralNetwork
import projectfunctions as pf

sns.set()
sns.set_style("whitegrid")
sns.set_palette("husl")

filepath = "../data/input/"
filename = "default_of_credit_card_clients"

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

df = pd.read_pickle(filepath + filename + "_altered_clean.pkl")
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
#column_indices = pf.pca(input, thresholdscaler=1e-1)
#input = input.iloc[:, column_indices]
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
layers = [first_layer, 20, 20, 1]           # 20, 20, 1
n_epochs = 5                               # 20
batch_size = 1                            # 100
learning_rate = 0.035                         # 1
#regularisation = 0.6

trainingShare = 0.8
seed  = 42
training_input, test_input, training_labels, test_labels = train_test_split(
                                                                input_prepared,
                                                                labels,
                                                                train_size=trainingShare,
                                                                test_size = 1-trainingShare,
                                                                random_state=seed
                                                                )

network = NeuralNetwork(layers, Sigmoid())
network.train(training_input, training_labels, learning_rate, n_epochs, batch_size, \
            test_input, test_labels, test='accuracy')

pred_prob = network.predict_probabilities(test_input)
pred = network.predict(test_input)

figurepath = '../figures/'

#compute confusion matrix
true_negative, false_positive, false_negative, true_positive = confusion_matrix(test_labels, pred).ravel()
#normalize
'''
n_negative = true_negative+false_negative
true_negative /= n_negative
false_negative /= n_negative

n_positive = true_positive+false_positive
true_positive /= n_positive
false_positive /= n_positive
'''
print('true positive: ',true_positive)
print('false positive: ',false_positive)
print('true_negative: ', true_negative)
print('false_negative: ', false_negative)

plt.figure()
plt.hist(pred_prob)
plt.xlim([0,1])
plt.title('Histogram of output from NN')
plt.xlabel('output')
plt.ylabel('count')
plt.show()
