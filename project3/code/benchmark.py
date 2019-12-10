import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
import sys

import finite_difference as fd

sys.path.append("class/")
import projectfunctions as pf


datadir = "../data/"


#first command line argument sets the number of runs (n) for each value of N
n = int(sys.argv[1])
#second command line argument sets the maximum power of n
T = float(sys.argv[2])
N_max = int(sys.argv[3])
#making a list of n power values
N = np.linspace(1, N_max, N_max, dtype=int)

#removing old data-files
#pf.remove_file("../data/finite_difference_timelog.dat")


filename = f"finite_difference_benchmark_T{T:.0E}.dat"
outfile = open(datadir + filename, "w")
counter = 1
for i in N:
    timearray = np.zeros(n)
    msearray = np.zeros(n)
    for j in range(n):
        tic = time.process_time()
        u, x = fd.solve(fd.initial, T, Nx=i)
        toc = time.process_time()
        timearray[j] = toc - tic
        msearray[j] = pf.mse(u[-1, :], fd.exact(x, T))
    outfile.write("{} {:.16f} {:.16f}\n".format(i, np.mean(timearray), np.mean(msearray)))
    print("Iteration {}/{} complete".format(counter, len(N)))
    counter += 1
outfile.close()






#plotting maximum error plot
sns.set()
sns.set_style("whitegrid")
sns.set_palette("husl")
