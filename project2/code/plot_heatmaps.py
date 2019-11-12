"""
Read output from tuning scripts and plot heatmaps of the regularization
parameters and their accuracy
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
#sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


data_LogReg = pd.read_csv("../data/output/NeuralNetwork/neural_acc_auc.csv")
print("max accuracy:")
print(data_LogReg.loc[data_LogReg['accuracy'].idxmax()])
print()
print("max AUC:")
print(data_LogReg.loc[data_LogReg['AUC'].idxmax()])

data_LogReg = data_LogReg.values

x_dim = 15
y_dim = 15

accuracy = data_LogReg[:,0].reshape(x_dim,y_dim)
learning_rate_init = data_LogReg[:,1].reshape(x_dim,y_dim)
mini_batch_size = data_LogReg[:,2].reshape(x_dim,y_dim)
auc = data_LogReg[:,3].reshape(x_dim,y_dim)


fig, ax = plt.subplots()

c = ax.pcolormesh(mini_batch_size, learning_rate_init,accuracy, cmap="gnuplot")
#ax.set_title('Accuracy of LogisticRegression')
ax.set_xlabel("minibatch size",fontsize=20)
ax.set_ylabel("initial learning rate",fontsize=20)
ax.tick_params(axis='both', labelsize=14)
ax.set_yscale("log")
ax.set_xscale("log")
fig.colorbar(c, ax=ax)
plt.savefig("../figures/NNtuner_accuracy.pdf")
plt.show()


fig, ax = plt.subplots()

c = ax.pcolormesh(mini_batch_size, learning_rate_init,auc, cmap="gnuplot")
#ax.set_title('AUC of LogisticRegression')
ax.set_xlabel("minibatch size",fontsize=20)
ax.set_ylabel("initial learning rate",fontsize=20)
ax.tick_params(axis='both', labelsize=14)
ax.set_yscale("log")
ax.set_xscale("log")
fig.colorbar(c, ax=ax)
plt.savefig("../figures/NNtuner_auc.pdf")
plt.show()
