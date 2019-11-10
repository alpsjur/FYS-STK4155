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


data_LogReg = pd.read_csv("../data/output/LogisticRegression/logistic_logspace.csv")

print(data_LogReg.loc[data_LogReg['accuracy'].idxmax()])

data_LogReg = data_LogReg.values

accuracy = data_LogReg[:,0].reshape(50,50)
learning_rate_init = data_LogReg[:,1].reshape(50,50)
mini_batch_size = data_LogReg[:,2].reshape(50,50)

fig, ax = plt.subplots()

c = ax.pcolormesh(mini_batch_size, learning_rate_init,accuracy, cmap="coolwarm")
#ax.set_title('Accuracy of LogisticRegression')
ax.set_xlabel("minibatch size",fontsize=20)
ax.set_ylabel("initial learning rate",fontsize=20)
ax.tick_params(axis='both', labelsize=14)
ax.set_yscale("log")
fig.colorbar(c, ax=ax)
plt.savefig("../figures/LogRegTune.pdf")
plt.show()
