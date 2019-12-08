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


def f(x, A):
    """
    function denoted as f in the paper by YI et al.
    x is tensor of size (n,1)
    returns tensor of size (n,1)
    """
    I = tf.eye(n,dtype=tf.float64)
    xT = tf.transpose(x)
    term1 = tf.matmul(xT,x)*A
    term2 = (1 - tf.matmul(tf.matmul(xT,A),x))*I
    return tf.matmul((term1 + term2),x)

def compute_eigval(v, A):
    """
    function for computing eigenvalue, given eigenvector v
    v is tensor of size (n,1)
    returns a float
    """
    vT = tf.transpose(v)
    num = tf.matmul(tf.matmul(vT,A),v)
    den = tf.matmul(vT,v)
    return num/den



#setting up the NN
Nt = 10
Nx = n
t = np.linspace(0,Nt-1, Nt)
x = np.linspace(1, Nx-1, Nx)
#x = np.random.rand(n)

# Create mesh and convert to tensors
X, T = np.meshgrid(x, t)

x_ = (X.ravel()).reshape(-1, 1)
t_ = (T.ravel()).reshape(-1, 1)

x_tf = tf.convert_to_tensor(x_)
t_tf = tf.convert_to_tensor(t_)

points = tf.concat([x_tf, t_tf], 1)

A_tf = tf.convert_to_tensor(A)

num_iter = 10000
num_hidden_neurons = [30,30]
num_hidden_layers = np.size(num_hidden_neurons)

with tf.name_scope('dnn'):

    # Input layer
    previous_layer = points

    # Hidden layers
    for l in range(num_hidden_layers):
        current_layer = tf.layers.dense(previous_layer, \
                                        num_hidden_neurons[l], \
                                        name='hidden%d'%(l+1), \
                                        activation=tf.nn.sigmoid)
        previous_layer = current_layer

    # Output layer
    dnn_output = tf.layers.dense(previous_layer, 1, name='output')

#define loss function
#DETTE MAA ORDNES
#dnn_output maa reshapes slik at funskjonen f kan brukes, tidssteg for tidssteg?
#maa beregne f(x) for hvert tidssteg
with tf.name_scope('cost'):
    trial = t_tf*(1-t_tf)*t_tf*dnn_output

    # calculate the gradients
    trial_dt = tf.gradients(trial, t_tf)

    dnn_output_rs = tf.reshape(dnn_output,(Nx, Nt))
    trial_dt_rs = tf.reshape(trial_dt,(Nx, Nt))

    # calculate cost function
    rhs = f(dnn_output_rs, A_tf) - dnn_output_rs
    err = tf.square(-trial_dt_rs-rhs)
    cost = tf.reduce_sum(err, name='cost')

learning_rate = 0.001
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    traning_op = optimizer.minimize(cost)

v_dnn_tf = None

init = tf.global_variables_initializer()


with tf.Session() as sess:
    # Initialize the whole graph
    init.run()

    # Evaluate the initial cost:
    print('Initial cost: %g'%cost.eval())

    # The training of the network:
    for i in range(num_iter):
        sess.run(traning_op)

        # If one desires to see how the cost function behaves for each iteration:
        #if i % 1000 == 0:
        #    print(cost.eval())

    # Training is done, and we have an approximate solution to the ODE
    print('Final cost: %g'%cost.eval())

    # Store the result
    v_dnn_tf = trial.eval()
