import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns

sns.set()
sns.set_style("white")
sns.set_palette("Set2")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(4155)
np.random.seed(4155)


#code for generating random, symmetric nxn matrix
n = 6
Q = np.random.rand(n,n)
A = (Q.T+Q)/2
#change sign of A to mind other eigenvalue
A_tf = tf.convert_to_tensor(A,dtype=tf.float64)

#compute eigenvalues with numpy.linalg
w_np, v_np = np.linalg.eig(A)
idx = np.argsort(w_np)
v_np = v_np[:,idx]
w_min_np = np.min(w_np)
w_max_np = np.max(w_np)
v_min_np = v_np[:,0]
v_max_np = v_np[:,-1]

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
    v is vector of size n
    returns a float
    """
    v = v.reshape(n,1)
    vT = v.transpose()
    num = np.matmul(np.matmul(vT,A),v)[0,0]
    den = np.matmul(vT,v)[0,0]
    return num/den



#setting up the NN
Nt = 20
Nx = n
t = np.linspace(0, 1, Nt) #maa gaa fra 0 til 1 for aa faa konvergens
x = np.linspace(1, Nx, Nx)
v0 = np.random.rand(n)

# Create mesh and convert to tensors
X, T = np.meshgrid(x, t)
V, T_ = np.meshgrid(v0, t)

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
    trial = dnn_output*t_tf + v0_tf
    #v0_tf*dnn_output**(-t_tf)#(1-t_tf)*v0_tf + t_tf*dnn_output

    # calculate the gradients
    trial_dt = tf.gradients(trial, t_tf)

    #reshape to allow itterating over time stemps
    trial_rs = tf.reshape(trial,(Nt, Nx))
    trial_dt_rs = tf.reshape(trial_dt,(Nt, Nx))

    # calculate cost function, mse
    cost_temp = 0
    for j in range(Nt):
        trial_temp = tf.reshape(trial_rs[j],(n,1))
        trial_dt_temp = tf.reshape(trial_dt_rs[j],(n,1))
        rhs = f(trial_temp) - trial_temp
        err = tf.square(-trial_dt_temp+rhs)
        cost_temp += tf.reduce_sum(err)
    cost = tf.reduce_sum(cost_temp/(Nx*Nt), name='cost')
learning_rate = 0.001
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    traning_op = optimizer.minimize(cost)

v_dnn = None

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
            print(i,' iterations:', cost.eval())

    # Training is done, and we have an approximate solution to the ODE
    print('Final cost: %g'%cost.eval())

    # Store the result
    #v_dnn_tf = trial.eval()
    v_dnn = tf.reshape(trial,(Nt,Nx))
    v_dnn = v_dnn.eval()

fig, ax = plt.subplots()
ax.plot(v_dnn, color='black')
ax.set_xlabel('Number of timesteps')
ax.set_ylabel('Value of the elements of the estimated eigenvector')

v_max_dnn = v_dnn[-1]
w_max_dnn = compute_eigval(v_max_dnn)
print('v0: \n', v0)
print('v nn: \n',v_max_dnn)
print('unit v nn: \n', v_max_dnn/np.linalg.norm(v_max_dnn))
print('unit v max np: \n',v_max_np)
print('unit v min np: \n',v_min_np)
print('w nn: \n',w_max_dnn)
print('w max numpy: \n',w_max_np)
print('w min numpy: \n',w_min_np)

plt.show()
