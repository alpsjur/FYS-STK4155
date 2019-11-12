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


df_LogReg = pd.read_csv("../data/output/LogisticRegression/logistic_acc_auc.csv")
df_LogReg.rename(columns = {'learning_rate_init':'Initial learning rate', 'minibatch_size':'Mini batch size'}, inplace = True)
print("max accuracy:")
print(df_LogReg.loc[df_LogReg['accuracy'].idxmax()])
print()
print("max AUC:")
print(df_LogReg.loc[df_LogReg['AUC'].idxmax()])

heatmap_LogReg_AUC = pd.pivot_table(df_LogReg, values='AUC',
                     index=['Initial learning rate'],
                     columns='Mini batch size')

#fig, ax = plt.subplots()
#sns.heatmap(heatmap_LogReg_AUC, ax=ax)
#plt.show()

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
plt.savefig("../figures/NNtuner_accuracy.pdf")
plt.show()


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
plt.savefig("../figures/NNtuner_auc.pdf")
plt.show()
