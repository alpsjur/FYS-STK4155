import numpy as np
from imageio import imread
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import projectfunctions as pf
import iofunctions as io

figdir = "../figures/"
sns.set()
sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

datadir = "../data/"

columns = ["degree", "MSE"]
df = pd.read_csv(datadir + "realData_OLS.txt",
                    names=columns,
                    delim_whitespace=True
                    )
df.set_index("degree", inplace=True)
fig, ax = plt.subplots(1, 1)
ax.plot(df)
ax.set_xlabel("degree", fontsize=18)
ax.set_ylabel("MSE", fontsize=18)
ax.set_ylim(0,15000)

plt.savefig(figdir+'mseVSdegreeOSL_terrain.pdf')
