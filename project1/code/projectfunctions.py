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
            #print( x**i*y**j)
            X[:, p] = x**i*y**j
            p += 1
    return X

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

def lasso_regression(X, data, hyperparam=1, **kwargs):
    """
    Lasso regression solved using scikit learn's in-built method Lasso
    """
    reg = linear_model.Lasso(alpha=hyperparam,
                            max_iter=1e3,
                            tol = 1e-1,
                            fit_intercept=False
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

def produce_table(data, hheader, vheader=None):
    """
    Spagetthi code producing a vertical laTEX table.
    data has shape
        [[x0, x1, x2, ..., xN],
         [y0, y1, y2, ..., yN],
         ...          ]
    where
    header = list/array
    """
    tableString = ""
    n = len(data[:][0])
    tableString += "\\begin{table}[htbp]\n"
    tableString += "\\begin{{tabular}}{{{0:s}}}\n".format("l"*n)
    # creating header
    for element in hheader:
        tableString += f"\\textbf{{{element}}} & "
    tableString = tableString[:-2] + "\\\\\n"
    # creating table elements
    for j in range(len(data[0][:])):
        if vheader is not None:
            tableString += f"\\textbf{{{vheader[j]}}} & "
        for i in range(len(data[:][0])):
            tableString += f"{data[i][j]:.2g} & "
        tableString = tableString[:-2] + "\\\\\n"
    tableString = tableString[:-4] + "\n"
    tableString += "\\end{tabular}\n"
    tableString += "\\end{table}\n"
    return tableString
