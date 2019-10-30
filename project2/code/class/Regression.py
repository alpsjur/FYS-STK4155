import numpy as np
from sklearn import linear_model
from sklearn.utils import resample


class Regression:
    def __init__(self):
        self.beta = None
        self.betas = []
        return

    def fit(self, designMatrix):
        model = designMatrix @ self.beta
        return model

    def train(self, designMatrix, labels, *args, **kwargs):
        #
        return self.beta

    def clear_betas(self):
        self.betas = []
        return

    def mse(self, model, labels, **kwargs):
        """
        Calculates the mean square error between data and model.
        """
        error = np.mean( np.mean((labels - model)**2, **kwargs) )
        return error

    def bias(self, model, labels, **kwargs):
        """caluclate bias from k expectation values and data of length n"""
        error = self.mse(labels, np.mean(model), **kwargs)
        return error

    def variance(self, model, **kwargs):
        """
        Calculating the variance of the model: Var[model]
        """
        error = self.mse(model, np.mean(model), **kwargs)
        return error

    def r2(self, model, labels, **kwargs):
        """
        Calculates the R2-value of the model.
        """
        error = np.mean(1 - np.sum((labels - model)**2, **kwargs)\
                     /np.sum((labels - np.mean(labels))**2, **kwargs) )
        return error

    def bootstrap(self, designMatrix_train, designMatrix_test, labels_train, labels_test, *args, \
                 n_bootstraps=200, **kwargs):
        '''
        bootstrap resampling method calculating mse, r2, bias and variance
        arguments
            x_train, y_train = coordinates for training model
            x_test, y_test = coordinates for testing model
            z_train = data to fit model on
            z_test = data to test model on
            reg = regression function reg(X, data, hyperparam)
            degree = degree of polynomial
            hyperparam = hyperparameter for calibrating model
            n_bootstraps = number of bootstrap sycles
        Returns [mse, r2, bias, variance]
        '''

        #initialize matrix for storing the predictions
        labels_pred = np.empty((labels_test.shape[0], n_bootstraps))

        #preforming n_bootstraps sycles
        for i in range(n_bootstraps):
            designMatrix_, labels_ = resample(designMatrix_train, labels_train)
            self.train(designMatrix_, labels_, *args, **kwargs)
            labels_pred_temp = self.fit(designMatrix_test)
            #storing the prediction for evaluation
            labels_pred[:, i] = labels_pred_temp.ravel()
        labels_test = np.reshape(labels_test,(len(labels_test),1))
        #evaluate predictions
        mse = self.mse(labels_pred, labels_test, axis=1, keepdims=True)
        r2 = self.r2(labels_pred, labels_test, axis=1, keepdims=True)
        bias = self.bias(labels_pred, labels_test, axis=1, keepdims=True)
        variance = self.variance(labels_pred, axis=1, keepdims=True)
        return [mse, r2, bias, variance]

    """ UNTESTED! DO NOT USE! """
    def k_fold_cross_validation(designMatrix, labels, *args, k=5, **kwargs):
        """
        k-fold CV calculating evaluation scores: MSE, R2, Bias, variance for
        data trained on k folds. Returns MSE, R2 Bias, variance, and a matrix of beta
        values for all the folds.
        arguments:
            x, y = coordinates (will generalise for arbitrary number of parameters)
            z = data
            reg = regression function reg(X, data, hyperparam)
            degree = degree of polynomial
            hyperparam = hyperparameter for calibrating model
            k = number of folds for cross validation
        """
        #p = int(0.5*(degree + 2)*(degree + 1))
        MSE = np.zeros(k)
        R2 = np.zeros(k)
        BIAS = np.zeros(k)
        VAR = np.zeros(k)
        #betas = np.zeros((p,k))

        #shuffle the data
        designMatrix_shuffle = shuffle(designMatrix)
        labels_shuffle = shuffle(labels)

        #split the data into k folds
        designMatrix_split = np.array_split(designMatrix, k)
        labels_split = np.array_split(labels, k)

        #loop through the folds
        for i in range(k):
            #pick out the test fold from data
            designMatrix_test = designMatrix_split[i]
            labels_test = labels_split[i]

            # pick out the remaining data as training data
            # concatenate joins a sequence of arrays into a array
            # ravel flattens the resulting array

            designMatrix_train = np.delete(designMatrix_split, i, axis=0)
            labels_train = np.delete(labels_split, i, axis=0).ravel()

            #fit a model to the training set
            self.train(designMatrix_train, labels_train, *args, *kwargs)

            #evaluate the model on the test set
            model = self.fit(designMatrix_test)

            #betas[:,i] = beta
            MSE[i] = self.mse(model, labels_test, axis=1, keepdims=True) #mse
            R2[i] = self.r2(model, labels_test, axis=1, keepdims=True) #r2
            BIAS[i] = self.bias(labels_test, model, axis=1, keepdims=True)
            VAR[i]= self.variance(model, axis=1, keepdims=True)
        return [np.mean(MSE), np.mean(R2), np.mean(BIAS), np.mean(VAR)]

class LinearRegression(Regression):
    def train(self, designMatrix, labels):
        self.beta = np.linalg.pinv(designMatrix.T.dot(designMatrix)).dot(designMatrix.T).dot(labels)
        self.betas.append(self.beta)
        return self.beta

class RidgeRegression(Regression):
    def train(self, designMatrix, labels, hyperparameter):
        p = len(designMatrix[0, :])
        self.beta = np.linalg.pinv(designMatrix.T.dot(designMatrix)
                + hyperparameter*np.identity(p)).dot(designMatrix.T).dot(labels)
        self.betas.append(self.beta)
        return self.beta

class LassoRegression(Regression):
    def train(self, designMatrix, labels, hyperparameter, **kwargs):
        reg = linear_model.Lasso(alpha=hyperparameter, **kwargs)
        reg.fit(designMatrix, labels)
        self.beta = reg.coef_
        self.betas.append(self.beta)
        return self.beta

class LogisticRegression(Regression):
    def fit(self, designMatrix):
        model = self.sigmoid(super().fit(designMatrix))
        return model

    def sigmoid(self, x):
        f = np.exp(x)/(np.exp(x) + 1)
        return f

    def calculate_cost_gradient(self, labels, designMatrix):
        m = designMatrix.shape[0]
        probabilities = self.sigmoid(designMatrix.dot(self.beta))
        cost_gradient = designMatrix.T.dot((labels - probabilities))/m
        return cost_gradient

    def train(self, designMatrix, labels, learning_rate=1e-4, n_epochs=50, mini_batch_size=100):
        """
        Stochastic gradient descent for computing the parameters that minimize the cost function.
            n_epochs = number of epochs
            mini_batch_size = number of data points in each mini batch
            learning_rate = the learning rate, often denoted eta
        """
        n = labels.shape[0]
        self.beta = np.random.randn(len(designMatrix[0, :]))
        #self.beta = [np.random.randn(1) for i in range(len(designMatrix[0, :]))]
        for epoch in range(n_epochs):
            idx = np.arange(n)
            np.random.shuffle(idx)
            labels = labels[idx]
            designMatrix = designMatrix[idx]
            labels_mini_batches = [labels[i:i+mini_batch_size] for i in range(0, n, mini_batch_size)]
            designMatrix_mini_batches = [designMatrix[i:i+mini_batch_size] for i in range(0, n, mini_batch_size)]
            for labels_mini_batch, designMatrix_mini_batch in zip(labels_mini_batches, designMatrix_mini_batches):
                cost_gradient = self.calculate_cost_gradient(labels_mini_batch, designMatrix_mini_batch)
                self.beta = self.beta - learning_rate*cost_gradient
            self.betas.append(self.beta)
        return self.beta

    def predict(self, designMatrix):
        n = len(designMatrix[:, 0])
        targets = np.zeros(n)
        model = self.fit(designMatrix)
        for i in range(n):
            if model[i] > 0.5:
                targets[i] = 1
            else:
                targets[i] = 0
        return targets

    def indicator(self, target, label):
        if target == label:
            return 1
        else:
            return 0

    def accuracy(self, designMatrix, labels):
        n = len(labels)
        counter = 0
        targets = self.predict(designMatrix)
        for i in range(n):
            counter += self.indicator(targets[i], labels[i])
        return counter/n


if __name__ == "__main__":
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
    from Regression import Logistic

    np.random.seed(42)


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

    # %% Our code
    logreg = LogisticRegression()
    logreg.train(designMatrix_train, labels_train,
            learning_rate=1e-4,
            n_epochs=500,
            mini_batch_size=1000
            )
    #model = logreg.fit(designMatrix_test)
    print(designMatrix_test)
    print(logreg.accuracy(designMatrix_test, labels_test))

    # %% Scikit learn
    reg = linear_model.LogisticRegression(solver="lbfgs")
    reg.fit(designMatrix_train, labels_train)
    print(reg.score(designMatrix_test, labels_test))
