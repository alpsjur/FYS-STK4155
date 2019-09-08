import numpy as np


def generate_design_polynomial(x, p=1):
    X = np.zeros((len(x), p+1))
    for degree in range(0, p+1):
        X[:, degree] = (x.T)**degree
    return X

def generate_design_2Dpolynomial(x, y, degree=5):
    X = np.zeros(( len(x), int(0.5*(degree + 2)*(degree + 1)) ))

    p = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            X[:, p] = (x.T)**i*(y.T)**j
            p += 1
    return X

def least_squares(X, data):
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(data)
    return beta

def mse(data, model):
    n = len(data)
    error = np.sum((data - model)**2)/n
    return error

def r2(data, model):
    n = len(data)
    error = 1 - np.sum((data - model)**2)/np.sum((data - np.mean(data))**2)
    return error

def frankefunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def vector_multiply(x,y):
    """Handles multiplication of vectors of unequal length"""
    if len(x) > len (y):
        z = x
        for i in range(len(y)):
            z[i] *= y[i]
        return z
    elif len(y) > len (x):
        z = y
        for i in range(len(x)):
            z[i] *= x[i]
        return z
    else:
        z = x*y
        return z
