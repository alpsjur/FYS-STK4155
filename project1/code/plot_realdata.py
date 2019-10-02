from sklearn.utils import shuffle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from imageio import imread
import pandas as pd
import seaborn as sns



sns.set()
sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def read_pandas(datadir, lambdamin, lambdamax):
    genericFilename = "realData_boot_lambda1e-0"
    dfs = []
    keys = []
    for i in range(lambdamin, lambdamax+1):
        df_temp = pd.read_csv(datadir + genericFilename + f"{i}.txt",
                            delim_whitespace=True,
                            header=0
                            )
        df_temp.set_index("degree", inplace=True)
        dfs.append(df_temp)
        keys.append(f"1e-{i}")
    df = pd.concat(dfs,
                    axis=0, keys=keys,
                    names = ["lambda"]
                    )
    return df

def plot_all_lambdas(ax, df, lambdamin, lambdamax, parameter):
    for i in range(lambdamin, lambdamax+1):
        ax.plot(df.loc[f"1e-{i}", parameter], label=fr"$\lambda$ = 1e-{i}")
    ax.set_xlabel("degree", fontsize=18)
    ax.set_ylabel("MSE", fontsize=18)
    ax.legend(frameon=False, fontsize=18, ncol=2)
    plt.tight_layout()
    return

datadir_lasso = "../data/lasso/"
datadir_ridge = "../data/ridge/"

df_lasso = read_pandas(datadir_lasso, 1, 6)
df_ridge = read_pandas(datadir_ridge, 2, 9)


figdir = "../figures/"
fig1, ax1 = plt.subplots(1, 1)

plot_all_lambdas(ax1, df_lasso, 1, 6, "mse")
ax1.set_ylim(0, 16000)
plt.savefig(figdir + "lasso_multilambda_mse.pdf")

fig2, ax2 = plt.subplots(1, 1)
plot_all_lambdas(ax2, df_ridge, 2, 9, "mse")
ax2.set_ylim(0, 16000)
plt.savefig(figdir + "ridge_multilambda_mse.pdf")

plt.show()
