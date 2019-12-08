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
A_tf = tf.convert_to_tensor(A,dtype=tf.float64)

#compute eigenvalues with numpy.linalg
w_np, v_np = np.linalg.eig(A)
w_min_np = np.min(w_np)
w_max_np = np.max(w_np)

def f(x):
    """
    function denoted as f in the paper by YI et al.
    x is tensor of size (n,1)
    returns tensor of size (n,1)
    """
    I = tf.eye(n,dtype=tf.float64)
    xT = tf.transpose(x)
    term1 = tf.matmul(xT,x)*A_tf
    term2 = (1 - tf.matmul(tf.matmul(xT,A_tf),x))*I
    return tf.matmul((term1 + term2),x)

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



#setting up the NN
Nt = 100
Nx = n
t = np.linspace(0,Nt-1, Nt)
x = np.linspace(1, Nx-1, Nx)
v0 = np.random.rand(n)

# Create mesh and convert to tensors
X, T = np.meshgrid(x, t)
V, T = np.meshgrid(v0, t)

x_ = (X.ravel()).reshape(-1, 1)
t_ = (T.ravel()).reshape(-1, 1)
v0_ = (V.ravel()).reshape(-1, 1)

x_tf = tf.convert_to_tensor(x_,dtype=tf.float64)
t_tf = tf.convert_to_tensor(t_,dtype=tf.float64)
v0_tf = tf.convert_to_tensor(v0_,dtype=tf.float64)

points = tf.concat([x_tf, t_tf], 1)

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
#trial solution maa defineres annerledes tror AL
with tf.name_scope('cost'):
    trial = t_tf*(1-t_tf)*dnn_output#(1-t_tf)*v0_tf + x_tf*(1-x_tf)*t_tf*dnn_output

    # calculate the gradients
    trial_dt = tf.gradients(trial, t_tf)

    dnn_output_rs = tf.reshape(dnn_output,(Nt, Nx))
    trial_dt_rs = tf.reshape(trial_dt,(Nt, Nx))

    # calculate cost function
    cost = 0
    for j in range(n):
        dnn_output_temp = tf.reshape(dnn_output_rs[j],(n,1))
        trial_dt_temp = tf.reshape(trial_dt_rs[j],(n,1))
        rhs = f(dnn_output_temp) - dnn_output_temp
        err = tf.square(-trial_dt_temp-rhs)
        cost += tf.reduce_sum(err, name='cost')

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
        if i % 1000 == 0:
            print(cost.eval())

    # Training is done, and we have an approximate solution to the ODE
    print('Final cost: %g'%cost.eval())

    # Store the result
    #v_dnn_tf = trial.eval()
    v_dnn_tf = tf.reshape(trial,(Nt,Nx))
    v_dnn_tf = v_dnn_tf.eval()

print(v_dnn_tf[-1])
print(v_np)
