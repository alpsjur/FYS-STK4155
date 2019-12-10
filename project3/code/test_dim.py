import numpy as np
import tensorflow.compat.v1 as tf

t = np.array([1,2,3])
x = np.array([4,5])

Nt = len(t)
Nx = len(x)

X, T = np.meshgrid(x,t)

x_ = (X.ravel()).reshape(-1, 1)
t_ = (T.ravel()).reshape(-1, 1)

x_ *= t_

x_tf = tf.convert_to_tensor(x_,dtype=tf.float64)
t_tf = tf.convert_to_tensor(t_,dtype=tf.float64)

x_rs = np.reshape(x_tf,(Nt, Nx))

x_temp = np.reshape(x_rs[0],(Nx,1))

print(x_temp)

#with tf.Session() as sess:
#    print(x_temp.eval())
