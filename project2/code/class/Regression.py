import numpy as np
from sklearn import linear_model
from sklearn.utils import resample
from sklearn import metrics


class Regression:
    def __init__(self):
        self.beta = None
        self.betas = []
        return

    def fit(self, designMatrix):
        model = designMatrix @ self.beta
        return model

    def _fit_cv(self, designMatrix, beta):
        model = designMatrix @ beta
        return model

    def train(self, designMatrix, labels, *args, **kwargs):
        # dummy function
        return self.beta

    def clear_betas(self):
        self.betas = []
        return

    def mse(self, model, labels, **kwargs):
        """
        Calculates the mean square error between data and model.
        """
        error = np.mean( np.mean((labels - model)**2, **kwargs), **kwargs)
        return error

    def bias(self, model, labels, **kwargs):
        """caluclate bias from k expectation values and data of length n"""
        error = self.mse(labels, np.mean(model, **kwargs), **kwargs)
        return error

    def variance(self, model, **kwargs):
        """
        Calculating the variance of the model: Var[model]
        """
        error = self.mse(model, np.mean(model, **kwargs), **kwargs)
        return error

    def r2(self, model, labels, **kwargs):
        """
        Calculates the R2-value of the model.
        """
        error = 1 - np.mean(np.sum((labels - model)**2, **kwargs)\
                     /np.sum((labels - np.mean(labels))**2, **kwargs) )
        error = 1 - np.sum((labels - model)**2, **kwargs)\
                    /np.sum((labels - np.mean(labels, **kwargs))**2, **kwargs)
        return error

    def train_bootstrap(self, designMatrix_train, designMatrix_test, labels_train, labels_test, *args, \
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
        mse = np.mean(self.mse(labels_pred, labels_test, axis=1, keepdims=True))
        r2 = np.mean(self.r2(labels_pred, labels_test, axis=0, keepdims=True))
        bias = np.mean(self.bias(labels_pred, labels_test, axis=1, keepdims=True))
        variance = np.mean(self.variance(labels_pred, axis=1, keepdims=True))
        return [mse, r2, bias, variance]

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

    def calculate_cost_gradient(self, designMatrix, labels):
        m = designMatrix.shape[0]
        probabilities = self.fit(designMatrix)
        cost_gradient = -designMatrix.T.dot((labels - probabilities))
        return cost_gradient

    def make_minibatches(self, designMatrix, labels, minibatch_size):
        n = labels.shape[0]
        idx = np.arange(n)
        np.random.shuffle(idx)
        labels_shuffled = labels[idx]
        designMatrix_shuffled = designMatrix[idx]
        minibatches = [(designMatrix_shuffled[i:i+minibatch_size,:],\
                        labels_shuffled[i:i+minibatch_size]) for i in range(0, n, minibatch_size)]
        return minibatches

    def train(self, designMatrix, labels, learning_schedule, learning_rate_init=0.1, n_epochs=50, minibatch_size=100, update_beta=False):
        """
        Stochastic gradient descent for computing the parameters that minimize the cost function.
            n_epochs = number of epochs
            mini_batch_size = number of data points in each mini batch
            learning_rate = the learning rate, often denoted eta
        """
        n = labels.shape[0]
        if not update_beta:
            self.beta = np.random.randn(len(designMatrix[0, :]))
        for epoch in range(n_epochs):
            minibatches = self.make_minibatches(designMatrix, labels, minibatch_size)
            n_minibatches = len(minibatches)
            for i in range(n_minibatches):
                designMatrix_mini, labels_mini = minibatches[i]
                cost_gradient = self.calculate_cost_gradient(designMatrix_mini, labels_mini)
                decreaser = epoch*n_minibatches+i
                learning_rate = learning_schedule(decreaser, learning_rate_init=learning_rate_init)
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

    def auc(self,designMatrix,labels):
        targets = self.predict(designMatrix)
        score = metrics.roc_auc_score(labels,targets)
        return score
