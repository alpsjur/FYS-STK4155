import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import numpy as np
import seaborn as sns
import pandas as pd


sns.set()
sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.rc("text", usetex=True)
plt.rc("font", family="serif")


datadir = "../data/pde/"


filename = datadir + "neural_network_benchmark_T2E-02_gamma5E-4.dat"
nn1 = pd.read_csv(filename,
                delim_whitespace=True,
                names=["Iterations", "CPU time", "MSE"]
                )
nn1.set_index("Iterations", inplace=True)

filename = datadir + "neural_network_benchmark_T2E-02_gamma2E-03.dat"
nn2 = pd.read_csv(filename,
                delim_whitespace=True,
                names=["Iterations", "CPU time", "MSE"]
                )
nn2.set_index("Iterations", inplace=True)

filename = datadir + "neural_network_benchmark_T2E-02_gamma4E-03.dat"
nn3 = pd.read_csv(filename,
                delim_whitespace=True,
                names=["Iterations", "CPU time", "MSE"]
                )
nn3.set_index("Iterations", inplace=True)

filename = datadir + "neural_network_benchmark_T2E-02_gamma6E-03.dat"
nn4 = pd.read_csv(filename,
                delim_whitespace=True,
                names=["Iterations", "CPU time", "MSE"]
                )
nn4.set_index("Iterations", inplace=True)

filename = datadir + "neural_network_benchmark_T2E-02_gamma8E-03.dat"
nn5 = pd.read_csv(filename,
                delim_whitespace=True,
                names=["Iterations", "CPU time", "MSE"]
                )
nn5.set_index("Iterations", inplace=True)

filename = datadir + "neural_network_benchmark_T2E-02_gamma1E-02.dat"
nn6 = pd.read_csv(filename,
                delim_whitespace=True,
                names=["Iterations", "CPU time", "MSE"]
                )
nn6.set_index("Iterations", inplace=True)


figdir = "../figures/"

fig, ax = plt.subplots(1, 1)

learning_rates = [1e-4, 2e-3, 4e-3, 6e-3, 8e-3, 1e-2]

MSEs = [nn1.MSE, nn2.MSE, nn3.MSE, nn4.MSE, nn5.MSE, nn6.MSE]
v = np.logspace(-10, -1, 10)

c = ax.contourf(nn1.index, learning_rates, MSEs, v,
                locator=ticker.LogLocator(),
                cmap=cm.PuBu_r
                )
ax.set_xlabel("Iterations", fontsize=20)
ax.set_ylabel("Learning rate", fontsize=20)
cbar = fig.colorbar(c)

cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel("MSE", fontsize=20, rotation=270)

"""
ax.semilogy(nn1.MSE, color="k")
ax.semilogy(nn2.MSE, color="k")
ax.semilogy(nn3.MSE, color="k")
ax.semilogy(nn4.MSE, color="k")
ax.semilogy(nn5.MSE, color="k")
ax.semilogy(nn6.MSE, color="k")
"""

plt.show()
