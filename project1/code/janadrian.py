from sklearn.utils import shuffle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from imageio import imread
import pandas as pd

import projectfunctions as pf
import plottingfunctions as plf


import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns
import projectfunctions as pf

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
print(df.head())

df.plot()
plt.show()
