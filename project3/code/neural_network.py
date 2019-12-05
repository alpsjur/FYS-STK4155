"""
Using Neural Network to solve equation
u_xx = u_t
for a given inital condition u(x,0) = I(x) and
boundries u(0,t) = u(L,t) = 0

using trial funcition:

g_trial(x,t) = (1-t)I(x) + x(1-x)t*N(x,t,P)

N is output from neural network for input x and weights P
"""
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(4155)

# Decide grid size for space and time
L = 1
dx = 0.1
Nx = int(L/dx) + 1

final_t = 0.02
dt = 0.001
Nt = int(final_t/dt) + 1

x = np.linspace(0, L, Nx)
t = np.linspace(0, final_t, Nt)

# Create mesh and convert to tensors
X, T = np.meshgrid(x, t)

x_ = (X.ravel()).reshape(-1, 1)
t_ = (T.ravel()).reshape(-1, 1)

x_tf = tf.convert_to_tensor(x_)
t_tf = tf.convert_to_tensor(t_)

points = tf.concat([x_tf, t_tf], 1)

# SET UP NEURAL NETWORK
num_iter = 100000

num_hidden_neurons = [20,20]
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

# set up cost function (error^2)
# define initial condition
def initial(x):
    return np.sin(np.pi*x)

with tf.name_scope('cost'):
    # define trial funcition
    trial = (1-t_tf)*initial(x_tf) + x_tf*(1-x_tf)*t_tf*nn_output

    # calculate the gradients
    trial_dt = tf.gradients(trial, t_tf)
    trial_d2x = tf.gradients(tf.gradients(trial, x_tf), x_tf)

    # calculate cost function
    err = tf.square(trial_dt[0] - trial_d2x[0])
    cost = tf.reduce_sum(err, name='cost')

# define learning rate and minimization of cost function
# Define how the neural network should be trained
learning_rate = 0.001
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    traning_op = optimizer.minimize(cost)

# definie itialization of all nodes
init = tf.global_variables_initializer()

# CALCULATE AND SOLVE THE PDE
