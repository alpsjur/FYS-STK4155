"""
Read output from tuning scripts and plot heatmaps of the regularization
parameters and their accuracy
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set()
sns.set_style("white")
sns.set_palette("Set2")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


df_LogReg = pd.read_csv("../data/output/LogisticRegression/logistic_acc_auc_3.csv")
df_LogReg.rename(columns = {'learning_rate_init':'Initial learning rate', 'minibatch_size':'Mini batch size'}, inplace = True)
print("max accuracy LR:")
print(df_LogReg.loc[df_LogReg['accuracy'].idxmax()])
print()
print("max AUC LR:")
print(df_LogReg.loc[df_LogReg['AUC'].idxmax()])
print()

x_dim = 30
y_dim = 30

data_LogReg = df_LogReg.values
accuracy = data_LogReg[:,0].reshape(x_dim,y_dim)
learning_rate_init = data_LogReg[:,1].reshape(x_dim,y_dim)
mini_batch_size = data_LogReg[:,2].reshape(x_dim,y_dim)
auc = data_LogReg[:,3].reshape(x_dim,y_dim)
#auc[auc == 0] = np.nan


fig, ax = plt.subplots()


c = ax.pcolormesh(mini_batch_size, learning_rate_init,accuracy
                ,cmap = 'gnuplot'#'viridis'#'plasma'
                #,vmin = 0.2
                #,vmax = 0.9
                )
#ax.set_title('Accuracy of LogisticRegression')
ax.set_xlabel("minibatch size",fontsize=20)
ax.set_ylabel("initial learning rate",fontsize=20)
ax.tick_params(axis='both', labelsize=14)
ax.set_yscale("log")
ax.set_xscale("log")
fig.colorbar(c, ax=ax)
#plt.savefig("../figures/LogRegTune_accuracy.pdf")


fig, ax = plt.subplots()

c = ax.pcolormesh(mini_batch_size, learning_rate_init,auc
                  ,cmap = 'gnuplot'#'viridis'#'plasma'
                  #,vmin = 0.3
                  #,vmax = 0.8
                  )
#ax.set_title('AUC of LogisticRegression')
ax.set_xlabel("minibatch size",fontsize=20)
ax.set_ylabel("initial learning rate",fontsize=20)
ax.tick_params(axis='both', labelsize=14)
ax.set_yscale("log")
ax.set_xscale("log")
fig.colorbar(c, ax=ax)
#plt.savefig("../figures/LogRegTune_auc.pdf")



df_NN = pd.read_csv("../data/output/NeuralNetwork/neural_acc_auc_3.csv")
df_NN.rename(columns = {'learning_rate_init':'Initial learning rate', 'minibatch_size':'Mini batch size'}, inplace = True)
print("max accuracy NN:")
print(df_NN.loc[df_NN['accuracy'].idxmax()])
print()
print("max AUC NN:")
print(df_NN.loc[df_NN['AUC'].idxmax()])
print()

x_dim = 25
y_dim = 10

data_NN = df_NN.values
accuracy = data_NN[:,0].reshape(x_dim,y_dim)
learning_rate = data_NN[:,1].reshape(x_dim,y_dim)
mini_batch_size = data_NN[:,2].reshape(x_dim,y_dim)
auc = data_NN[:,3].reshape(x_dim,y_dim)


fig, ax = plt.subplots()


cut_off = 1
c = ax.pcolormesh(mini_batch_size[0:-cut_off-1,:], learning_rate[0:-cut_off-1,:],accuracy[0:-cut_off-1,:]
                ,cmap = 'plasma'#'viridis'#'plasma'
                #,vmin = 0.2
                #,vmax = 0.9
                )
#ax.set_title('Accuracy of LogisticRegression')
ax.set_xlabel("minibatch size",fontsize=20)
ax.set_ylabel("learning rate",fontsize=20)
ax.tick_params(axis='both', labelsize=14)
ax.set_yscale("log")
ax.set_xscale("log")
fig.colorbar(c, ax=ax)
#plt.savefig("../figures/NNTune_accuracy.pdf")


fig, ax = plt.subplots()

c = ax.pcolormesh(mini_batch_size[0:-cut_off-1,:], learning_rate[0:-cut_off-1,:],auc[0:-cut_off-1,:]
                  ,cmap = 'plasma'#'viridis'#'plasma'
                  #,vmin = 0.3
                  #,vmax = 0.8
                  )
#ax.set_title('AUC of LogisticRegression')
ax.set_xlabel("minibatch size",fontsize=20)
ax.set_ylabel("learning rate",fontsize=20)
ax.tick_params(axis='both', labelsize=14)
ax.set_yscale("log")
ax.set_xscale("log")
fig.colorbar(c, ax=ax)
#plt.savefig("../figures/NNTune_auc.pdf")


df_NN_franke = pd.read_csv("../data/output/NeuralNetwork/neural_franke_mse_epochs20.csv")
df_NN_franke.rename(columns = {'learning_rate_init':'Initial learning rate', 'minibatch_size':'Mini batch size'}, inplace = True)
print("minimum mse:")
print(df_NN_franke.loc[df_NN_franke['mse'].idxmin()])
print()
print("maximum r2:")
print(df_NN_franke.loc[df_NN_franke["r2"].idxmax()])

x_dim = 25
y_dim = 10

learning_rate = df_NN_franke["Initial learning rate"].to_numpy().reshape(x_dim, y_dim)
mini_batch_size = df_NN_franke["Mini batch size"].to_numpy().reshape(x_dim, y_dim)
mse = df_NN_franke["mse"].to_numpy().reshape(x_dim, y_dim)
r2 = df_NN_franke["r2"].to_numpy().reshape(x_dim, y_dim)

fig, ax = plt.subplots()

c = ax.pcolormesh(mini_batch_size, learning_rate, mse
                  ,cmap = 'plasma_r'#'viridis'#'plasma'
                  #,vmin = 0.3
                  #,vmax = 0.8
                  )
#ax.set_title('AUC of LogisticRegression')
ax.set_xlabel("minibatch size",fontsize=20)
ax.set_ylabel("learning rate",fontsize=20)
ax.tick_params(axis='both', labelsize=14)
ax.set_yscale("log")
ax.set_xscale("log")
fig.colorbar(c, ax=ax)
plt.savefig("../figures/NNTune_franke_mse.pdf")

plt.show()
