import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
import sys

import neural_network as nn

sys.path.append("class/")
import projectfunctions as pf


datadir = "../data/pde/"
pf.create_directories(datadir)


#first command line argument sets the number of runs (n) for each value of N
n = int(sys.argv[1])
#second command line argument sets the maximum power of n
T = float(sys.argv[2])
N_max = int(sys.argv[3])
#making a list of n power values
N = np.logspace(1, N_max, N_max*4, dtype=int)

#removing old data-files
#pf.remove_file("../data/finite_difference_timelog.dat")

learning_rate = 4E-3

filename = f"neural_network_benchmark_T{T:.0E}_gamma{learning_rate:.0E}.dat"
outfile = open(datadir + filename, "w")
counter = 1
for i in N:
    timearray = np.zeros(n)
    msearray = np.zeros(n)
    for j in range(n):
        tic = time.process_time()
        u, x, t = nn.solve(nn.initial, T, learning_rate=learning_rate, num_iter=i)
        toc = time.process_time()
        timearray[j] = toc - tic
        msearray[j] = np.mean((u[-1, :] - nn.exact(x, T))**2)
    outfile.write("{} {:.16f} {:.16f}\n".format(i, np.mean(timearray), np.mean(msearray)))
    print("Iteration {}/{} complete".format(counter, len(N)))
    counter += 1
outfile.close()






#plotting maximum error plot
sns.set()
sns.set_style("whitegrid")
sns.set_palette("husl")
