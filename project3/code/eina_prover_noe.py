import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(4155)

n = 6
x = np.linspace(1,n,n)
Nt = 10
t = np.linspace(0,1,Nt)

#construct A
Q = np.random.rand(n,n)
A = (Q.T+Q)/2
A_tf = tf.convert_to_tensor(A,dtype=tf.float64)

#compute eigenvalues with numpy.linalg
eig_val, eig_vec = np.linalg.eig(A)
eig_val_min = np.min(eig_val)
eig_val_max = np.max(eig_val)
print(eig_val_min, eig_val_max)

eig_vec_nn = np.zeros((n,Nt))

# Create mesh and convert to tensors
X, T = np.meshgrid(x, t)

x_ = (X.ravel()).reshape(-1, 1)
t_ = (T.ravel()).reshape(-1, 1)

x_tf = tf.convert_to_tensor(x_)
t_tf = tf.convert_to_tensor(t_)

# fix intial values
v0 = np.random.rand(n)
T_, V = np.meshgrid(t, v0)
v0_ = (V.ravel()).reshape(-1, 1)
v0_tf = tf.convert_to_tensor(v0_, dtype=tf.float64)

points = tf.concat([x_tf, t_tf], 1)

# SET UP NEURAL NETWORK
num_iter = 100

num_hidden_neurons = [30,30]
num_hidden_layers = np.size(num_hidden_neurons)

with tf.variable_scope('nn'):
    # input layer
    previous_layer = points

    # hidden layers
    for l in range(num_hidden_layers):
        current_layer = tf.layers.dense(previous_layer,
                                        num_hidden_neurons[l],
                                        activation=tf.nn.sigmoid)
        previous_layer = current_layer

    # output layer
    nn_output = tf.layers.dense(previous_layer, 1)

def initial(x):
    return v0_tf

trial = initial(x_tf) + x_tf*(1-x_tf)*t_tf*nn_output
print(trial)
# calculate the gradients
trial_dt = tf.gradients(trial, t_tf)

# calculate cost function
#err = tf.square(trial_dt[0] + trial - f(x_tf))
#cost = tf.reduce_sum(err, name='cost')


# with tf.name_scope('cost'):
#     # define trial funcition
#     trial = initial(x_tf) + x_tf*(1-x_tf)*t_tf*nn_output
#
#     # calculate the gradients
#     trial_dt = tf.gradients(trial, t_tf)
#
#     # calculate cost function
#     err = tf.square(trial_dt[0] + trial - f(x_tf))
#     cost = tf.reduce_sum(err, name='cost')


def compute_eigval(v):
    """
    function for computing eigenvalue, given eigenvector v
    v is tensor of size (n,1)
    returns a float
    """
    vT = tf.transpose(v)
    num = tf.matmul(tf.matmul(vT,A_tf),v)
    den = tf.matmul(vT,v)
    return num/den
