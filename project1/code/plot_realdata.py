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



figdir = "../figures/"
sns.set()
sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

datadir = "../data/lasso/"
genericFilename = "realData_boot_lambda1e-0"

df = pd.read_csv(datadir + genericFilename + "1.txt",
                delim_whitespace=True,
                header=0
                )
df.set_index("degree",
            inplace=True
            )
dfs = []
for i in range(2, 7):
    df_temp = pd.read_csv(datadir + genericFilename + f"{i}.txt",
                        delim_whitespace=True,
                        header=0
                        )
    df_temp.set_index("degree",
                inplace=True
                )
    dfs.append(df_temp)
df = pd.concat(dfs, keys=["1e-1", "1e-2", "1e-3", "1e-4", "1e-5", "1e-6"])
print(df)
df["mse"].plot()
plt.show()
