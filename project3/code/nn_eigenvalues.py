import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(4155)


#code for generating random, symmetric nxn matrix
n = 6
Q = np.random.rand(n,n)
A = (Q.T+Q)/2

#compute eigenvalues with numpy.linalg
w_np = np.linalg.eig(A)[0]
w_min_np = np.min(w_np)
w_max_np = np.max(w_np)


def f(x):
    """
    function denoted as f in the paper by YI et al.
    x is array of size (n,1)
    returns array of size (n,1)
    """
    I = np.identity(len(x))
    term1 = np.matmul(x.T,x)*A
    term2 = (1 - np.matmul(np.matmul(x.T,A),x))*I
    return np.matmul((term1 + term2),x)

def compute_eigval(v):
    """
    function for computing eigenvalue, given eigenvector v
    v is array of size (n,1)
    returns a float
    """
    num = np.matmul(np.matmul(v.T,A),v)
    den = np.matmul(v.T,v)
    return float(num/den)

x = np.random.rand(n,1)
