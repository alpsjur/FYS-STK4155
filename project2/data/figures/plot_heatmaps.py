import pandas as pd
import matplotlib.pyplot as plt


data_LogReg = pd.read_csv("../output/LogisticRegression/logistic.csv")


data_LogReg = data_LogReg.values

accuracy = data_LogReg[:,0].reshape(100,50)
learning_rate_init = data_LogReg[:,1].reshape(100,50)
mini_batch_size = data_LogReg[:,2].reshape(100,50)

fig, ax = plt.subplots()

c = ax.pcolormesh(mini_batch_size, learning_rate_init,accuracy, cmap="YlGnBu")
ax.set_title('Accuracy of LogisticRegression')
# set the limits of the plot to the limits of the data
ax.set_xlabel("minibatch size")
ax.set_ylabel("initial learning rate")
fig.colorbar(c, ax=ax)

plt.show()
