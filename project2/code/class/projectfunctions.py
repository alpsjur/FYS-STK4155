import os
import numpy as np
from sklearn.utils import shuffle
from sklearn import linear_model
from sklearn.utils import resample

def generate_design_polynomial(x, degree=1):
    """
    Creates a design matrix for a 1d polynomial of degree p
        1 + x + x**2 + ...
    """
    X = np.zeros((len(x), degree+1))
    for degree in range(0, degree+1):
        X[:, degree] = (x.T)**degree
    return X

def generate_design_2Dpolynomial(x, y, degree=5):
    """
    Creates a design matrix for a 2d polynomial with cross-elements
        1 + x + y + x**2 + xy + y**2 + ...
    """
    X = np.zeros(( len(x), int(0.5*(degree + 2)*(degree + 1)) ))
    p = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            X[:, p] = x**i*y**j
            p += 1
    return X

def fit_intercept(designMatrix):
    n, m = designMatrix.shape
    intercept = np.ones((n, 1))
    designMatrix = np.hstack((intercept, designMatrix))
    return designMatrix

def pca(designMatrix, thresholdscaler):
    designMatrix_centered = designMatrix - designMatrix.mean()
    correlation_matrix = designMatrix_centered.corr().to_numpy()
    eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)
    threshold = np.max(eigenvalues)*thresholdscaler

    column_indices = []
    for index in range(len(eigenvalues)):
        if eigenvalues[index] >= threshold:
            column_indices.append(index)
    return column_indices

def train_mean(designMatrix_train, designMatrix_test, labels_train, labels_test,
                method, n_runs, seed, *args, **kwargs):
    parameters = np.zeros(5)
    for i in range(n_runs):
        method.train(designMatrix, labels, *args, **kwargs)
        model = method.fit(designMatrix_test)
        parameters[0] += method.accuracy(designMatrix_test, labels_test)
        parameters[1] += method.mse(model, labels_test)
        parameters[2] += method.r2(model, labels_test)
        parameters[3] += method.bias(model, labels_test)
        parameters[4] += method.variance(model)
    parameters /= n_runs
    return parameters


def tune_hyperparameter(designMatrix, labels, method, seed, hyperparameters, *args, **kwargs):
    """
    Function training a model for different values of a certain hyperparameter.
    Syntax mimicking ColumnTransformer for hyperparameter input.
    """
    from sklearn.model_selection import train_test_split
    import pandas as pd

    trainingShare = 0.8
    designMatrix_train, designMatrix_test, labels_train, labels_test = train_test_split(
                                                                    designMatrix,
                                                                    labels,
                                                                    train_size=trainingShare,
                                                                    test_size = 1-trainingShare,
                                                                    random_state=seed
                                                                    )

    # %% Our code
    parameters = np.zeros((len(hyperparameters[1]), 6))
    parameters[:, 0] = hyperparameters[1]
    header = [hyperparameters[0]] + ["accuracy", "mse", "r2", "bias", "variance"]
    for i in range(len(hyperparameters[1])):
        exec(f"""method.train(designMatrix_train, labels_train,
                *args,
                {hyperparameters[0]}={hyperparameters[1][i]},
                **kwargs
                )"""
                )
        model = method.fit(designMatrix_test)
        parameters[i, 1] = method.accuracy(designMatrix_test, labels_test)
        parameters[i, 2] = method.mse(model, labels_test)
        parameters[i, 3] = method.r2(model, labels_test)
        parameters[i, 4] = method.bias(model, labels_test)
        parameters[i, 5] = method.variance(model)
    df = pd.DataFrame(parameters, columns=header)
    df.set_index(hyperparameters[0], inplace=True)
    return df



def least_squares(X, data, hyperparam=0):
    """
    Least squares solved using matrix inversion
    """
    return ridge_regression(X,data,hyperparam=0)


def ridge_regression(X, data, hyperparam=0):
    """
    Ridge regression solved using matrix inversion
    """
    p = len(X[0, :])
    beta = np.linalg.pinv(X.T.dot(X) + hyperparam*np.identity(p)).dot(X.T).dot(data)
    return beta

def lasso_regression(X, data, hyperparam=1):
    """
    Lasso regression solved using scikit learn's in-built method Lasso
    """
    reg = linear_model.Lasso(alpha=hyperparam,
                            max_iter=1e6,
                            tol = 3e-2,
                            fit_intercept=False # we already have a column of 1s
                            )
    reg.fit(X, data)
    beta = reg.coef_
    return beta

def mse(data, model):
    """
    Calculates the mean square error between data and model.
    """
    n = len(data)
    error = np.sum((data - model)**2)/n
    return error

def r2(data, model):
    """
    Calculates the R2-value of the model.
    """
    n = len(data)
    error = 1 - np.sum((data - model)**2)/np.sum((data - np.mean(data))**2)
    return error

def bias(data, model):
    """caluclate bias from k expectation values and data of length n"""
    n = len(data)
    error = mse(data, np.mean(model))
    return error

def variance(model):
    """
    Calculating the variance of the model: Var[model]
    """
    n = len(model)
    error = mse(model, np.mean(model))
    return error

def SGD(training_data, cost_gradient, parameters, n_epochs, mini_batch_size, learning_rate):
    """
    Stochastic gradient descent for computing the parameters that minimize the cost function.
        training_data = array containing the data point on the form [[[x11,x12,...],y1],[[x21,x22,...],y2],...]
        cost_gradient = function for computing the gradient of the cost function
        parameters = array containing the parameters to be updated. Ex [beta1, beta2, ...] for linear regression
        n_epochs = number of epochs
        mini_batch_size = number of data points in each mini batch
        learning_rate = the learning rate, often denoted eta
    """
    n = len(training_data)
    for epoch in range(n_epochs):
        np.random.shuffle(training_data)
        mini_batches = [training_data[i:i+mini_batch_size] for i in range(0, n, mini_batch_size)]
        for mini_batch in mini_batches:
            gradients = cost_gradient(mini_batch, parameters)
            parameters = parameters - learning_rate*gradients

def bootstrap(x_train, x_test, y_train, y_test, z_train, z_test, \
              reg, degree=5, hyperparam=0, n_bootstraps=200):
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
    #make a design matrix for the test data
    X_test = generate_design_2Dpolynomial(x_test, y_test, degree=degree)

    #initialize matrix for storing the predictions
    z_pred = np.empty((z_test.shape[0], n_bootstraps))

    #preforming n_bootstraps sycles
    for i in range(n_bootstraps):
        x_, y_, z_ = resample(x_train, y_train, z_train)
        X = generate_design_2Dpolynomial(x_, y_, degree=degree)
        beta = reg(X, z_, hyperparam=hyperparam)
        z_pred_temp = X_test @ beta
        #storing the prediction for evaluation
        z_pred[:, i] = z_pred_temp.ravel()

    z_test = np.reshape(z_test,(len(z_test),1))

    #evaluate predictions
    mse = np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
    r2 = np.mean(1 - np.sum((z_test - z_pred)**2, axis=1, keepdims=True)\
                 /np.sum((z_test - np.mean(z_test))**2, axis=1, keepdims=True) )
    bias = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
    variance = np.mean( np.var(z_pred, axis=1, keepdims=True) )

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
    from sklearn.utils import shuffle
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


def k_fold_cross_validation(x, y, z, reg, degree=5, hyperparam=0, k=5):
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
    x_shuffle, y_shuffle, z_shuffle = shuffle(x, y, z)

    #split the data into k folds
    x_split = np.array_split(x_shuffle, k)
    y_split = np.array_split(y_shuffle, k)
    z_split = np.array_split(z_shuffle, k)

    #loop through the folds
    for i in range(k):
        #pick out the test fold from data
        x_test = x_split[i]
        y_test = y_split[i]
        z_test = z_split[i]

        # pick out the remaining data as training data
        # concatenate joins a sequence of arrays into a array
        # ravel flattens the resulting array

        x_train = np.delete(x_split, i, axis=0).ravel()
        y_train = np.delete(y_split, i, axis=0).ravel()
        z_train = np.delete(z_split, i, axis=0).ravel()

        #fit a model to the training set
        X_train = generate_design_2Dpolynomial(x_train, y_train, degree=degree)
        beta = reg(X_train, z_train, hyperparam=hyperparam)

        #evaluate the model on the test set
        X_test = generate_design_2Dpolynomial(x_test, y_test, degree=degree)
        z_fit = X_test @ beta

        #betas[:,i] = beta
        MSE[i] = mse(z_test, z_fit) #mse
        R2[i] = r2(z_test, z_fit) #r2
        BIAS[i] = bias(z_test, z_fit)
        VAR[i]= variance(z_fit)

    return [np.mean(MSE), np.mean(R2), np.mean(BIAS), np.mean(VAR)] #betas

def frankefunction(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x - 2)**2) - 0.25*((9*y - 2)**2))
    term2 = 0.75*np.exp(-((9*x + 1)**2)/49.0 - 0.1*(9*y + 1))
    term3 = 0.5*np.exp(-(9*x - 7)**2/4.0 - 0.25*((9*y - 3)**2))
    term4 = -0.2*np.exp(-(9*x - 4)**2 - (9*y - 7)**2)
    return term1 + term2 + term3 + term4

def create_directories(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return

def produce_table(data, hheader=None, vheader=None):
    """
    Spagetthi code producing a vertical laTEX table.
    data has shape of an array
        [[x0, x1, x2, ..., xN],
         [y0, y1, y2, ..., yN],
         ...          ]
    where
    hheader = list/array of horizontal header
    vheader = list/array of vertical header
    If greek letters are used, then they must be enclosed by $$,
    i. e. $\lambda$
    """
    tableString = ""
    n = len(data[:][0])
    tableString += "\\begin{table}[htbp]\n"
    tableString += "\\centering\n"
    tableString += "\\begin{{tabular}}[width=0.5\\textwidth]{{l{0:s}}}\n".format("c"*(len(hheader)-1))
    tableString += "\\hline\n"
    # creating header
    if hheader is not None:
        for element in hheader:
            tableString += f"\\textbf{{{element}}} & "
    tableString = tableString[:-2] + "\\\\\n"
    tableString += "\\hline\n"
    # creating table elements
    for j in range(len(data[0, :])):
        if vheader is not None:
            tableString += f"\\textbf{{{vheader[j]}}} & "
        for i in range(len(data[:, 0])):
            tableString += f"{data[i, j]:.2f} & "
        tableString = tableString[:-2] + "\\\\\n"
    tableString = tableString[:-4] + "\n"
    tableString += "\\end{tabular}\n"
    tableString += "\\caption{{}}\n"
    tableString += "\\label{{table:}}\n"
    tableString += "\\end{table}\n"
    return tableString
