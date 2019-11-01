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
        # dummy function
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
        r2 = self.r2(labels_pred, labels_test, axis=0, keepdims=True)
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

    def train(self, designMatrix, labels, learning_schedule, t0=1, t1=10, n_epochs=50, minibatch_size=100):
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
            minibatches = self.make_minibatches(designMatrix, labels, minibatch_size)
            n_minibatches = len(minibatches)
            for i in range(n_minibatches):
                designMatrix_mini, labels_mini = minibatches[i]
                cost_gradient = self.calculate_cost_gradient(designMatrix_mini, labels_mini)
                t = epoch*n_minibatches+i
                learning_rate = learning_schedule(t, t0, t1)
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
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn import linear_model
    from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

    """
    import sys
    sys.path.append("class/")
    from Regression import Logistic
    """

    def learning_schedule(t, t0, t1):
        return t0/(t+t1)

    def fit_intercept(designMatrix):
        n, m = designMatrix.shape
        intercept = np.ones((n, 1))
        designMatrix = np.hstack((intercept, designMatrix))
        return designMatrix

    np.random.seed(42)


    sns.set()
    sns.set_style("whitegrid")
    sns.set_palette("husl")


    filepath = "../../data/input/"
    filename = "default_of_credit_card_clients"

    """
    nanDict = {}
    df = pd.read_excel(filepath + filename, header=1, skiprows=0, index_col=0, na_values=nanDict)

    df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)

    # Features and targets
    designMatrix = df.loc[:, df.columns != 'defaultPaymentNextMonth'].to_numpy()
    labels = df.loc[:, df.columns == 'defaultPaymentNextMonth'].to_numpy().ravel()
    """

    df = pd.read_pickle(filepath + filename + "_clean.pkl")
    print(df.head())

    # preparing designmatrix by scaling and using one hot encoding for cat data
    designMatrix = df.loc[:, df.columns != 'default payment next month']
    designMatrix_num = designMatrix.drop(["SEX", "EDUCATION", "MARRIAGE"], axis=1)
    designMatrix_cat = designMatrix.iloc[:, 1:4]

    num_attributes = list(designMatrix_num)
    cat_attributes = list(designMatrix_cat)
    design_pipeline = ColumnTransformer([
                                        ("scaler", StandardScaler(), num_attributes),
                                        ("onehot", OneHotEncoder(categories="auto"), cat_attributes)
                                        ])
    designMatrix_prepared = design_pipeline.fit_transform(designMatrix)

    # readying labels by one hot encoding
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
            n_epochs=100,
            minibatch_size=32,
            t0=1,
            t1=10
            )

    model = logreg.fit(designMatrix_test)

    # %% Scikit learn
    reg = linear_model.LogisticRegression(solver="lbfgs", max_iter=2e2)
    reg.fit(designMatrix_train, labels_train)

    accuracy = logreg.accuracy(designMatrix_test, labels_test)
    accuracy_sklearn = reg.score(designMatrix_test, labels_test)
    mse = logreg.mse(model, labels_test)
    r2 = logreg.r2(model, labels_test)
    bias = logreg.bias(model, labels_test)
    variance = logreg.variance(model)
    predictions = logreg.predict(designMatrix_test)
    guess_rate = np.mean(predictions)

    print("TEST")
    print(f"ACCURACY           {accuracy}")
    print(f"ACCURACY (SKLEARN) {accuracy_sklearn}")
    print(f"MSE                {mse}")
    print(f"R2                 {r2}")
    print(f"BIAS               {bias}")
    print(f"VAR                {variance}")
    print(f"GUESS RATE         {guess_rate}")

    # %% our code bootstrap
    mse, r2, bias, variance = logreg.bootstrap(designMatrix_train, designMatrix_test,
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
