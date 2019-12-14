import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


sns.set()
sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.rc("text", usetex=True)
plt.rc("font", family="serif")


datadir = "../data/pde/"


filename = datadir + "finite_difference_benchmark_T2E-02.dat"
fd1 = pd.read_csv(filename,
                delim_whitespace=True,
                names=["Time points", "CPU time", "MSE"]
                )
#fd_benchmark.drop(0, inplace=True)
fd1.set_index("Time points", inplace=True)
fd1["MSEmean"] = fd1.MSE.rolling(window=50,
                                center=True
                                ).mean()

filename = datadir + "finite_difference_benchmark_T3E-01.dat"
fd2 = pd.read_csv(filename,
                delim_whitespace=True,
                names=["Time points", "CPU time", "MSE"]
                )
#fd_benchmark.drop(0, inplace=True)
fd2.set_index("Time points", inplace=True)
fd2["MSEmean"] = fd2.MSE.rolling(window=50,
                                center=True
                                ).mean()

filename = datadir + "neural_network_benchmark_T2E-02_gamma4E-03.dat"
nn1 = pd.read_csv(filename,
                delim_whitespace=True,
                names=["Iterations", "CPU time", "MSE"]
                )
nn1.set_index("Iterations", inplace=True)

filename = datadir + "neural_network_benchmark_T3E-01_gamma4E-03.dat"
nn2 = pd.read_csv(filename,
                delim_whitespace=True,
                names=["Iterations", "CPU time", "MSE"]
                )
nn2.set_index("Iterations", inplace=True)


figdir = "../figures/"

figname = figdir + "MSEbench.pdf"
fig, ax = plt.subplots(2, 2, sharey=True)
#fig.tight_layout()

bigax = fig.add_subplot(1, 1, 1, frameon=False)
bigax.tick_params(labelcolor="none",
                top=False,
                bottom=False,
                right=False,
                left=False,
                grid_color="none"
                )
bigax.set_ylabel("MSE", fontsize=20)

ax[0, 0].loglog(fd1.MSEmean, color="k")
ax[0, 0].text(0.8, 0.97, "a.",
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax[0, 0].transAxes,
                fontsize=20
                )


ax[0, 1].loglog(nn1.MSE, color="k", marker="o", markersize=2, linestyle="none")
ax[0, 1].text(0.8, 0.97, "b.",
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax[0, 1].transAxes,
                fontsize=20
                )

ax[1, 0].loglog(fd2.MSEmean, color="k")
ax[1, 0].set_xlabel("Time points", fontsize=20)
ax[1, 0].text(0.8, 0.97, "c.",
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax[1, 0].transAxes,
                fontsize=20
                )

ax[1, 1].loglog(nn2.MSE, color="k", marker="o", markersize=2, linestyle="none")
ax[1, 1].set_xlabel("Iterations", fontsize=20)
ax[1, 1].text(0.8, 0.97, "d.",
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax[1, 1].transAxes,
                fontsize=20
                )

plt.savefig(figname)


figname = figdir + "CPUbench.pdf"
fig, ax = plt.subplots(2, 2, sharey=True)

bigax = fig.add_subplot(1, 1, 1, frameon=False)
bigax.tick_params(labelcolor="none",
                top=False,
                bottom=False,
                right=False,
                left=False,
                grid_color="none"
                )
bigax.set_ylabel("CPU time", fontsize=20)

ax[0, 0].loglog(fd1["CPU time"], color="k")
ax[0, 0].text(0.8, 0.04, "a.",
                horizontalalignment="left",
                verticalalignment="bottom",
                transform=ax[0, 0].transAxes,
                fontsize=20
                )


ax[0, 1].loglog(nn1["CPU time"], color="k", marker="o", markersize=2, linestyle="none")
ax[0, 1].text(0.8, 0.04, "b.",
                horizontalalignment="left",
                verticalalignment="bottom",
                transform=ax[0, 1].transAxes,
                fontsize=20
                )

ax[1, 0].loglog(fd2["CPU time"], color="k")
ax[1, 0].set_xlabel("Time points", fontsize=20)
ax[1, 0].text(0.8, 0.04, "c.",
                horizontalalignment="left",
                verticalalignment="bottom",
                transform=ax[1, 0].transAxes,
                fontsize=20
                )

ax[1, 1].loglog(nn2["CPU time"], color="k", marker="o", markersize=2, linestyle="none")
ax[1, 1].set_xlabel("Iterations", fontsize=20)
ax[1, 1].text(0.8, 0.04, "d.",
                horizontalalignment="left",
                verticalalignment="bottom",
                transform=ax[1, 1].transAxes,
                fontsize=20
                )

plt.savefig(figname)


filename = datadir + "neural_network_benchmark_T2E-02_gamma4E-03.dat"
nn_Nt10 = pd.read_csv(filename,
                delim_whitespace=True,
                names=["Iterations", "CPU time", "MSE"]
                )
nn_Nt10.set_index("Iterations", inplace=True)

filename = datadir + "neural_network_benchmark_T2E-02_gamma4E-03_Nt50.dat"
nn_Nt50 = pd.read_csv(filename,
                delim_whitespace=True,
                names=["Iterations", "CPU time", "MSE"]
                )
nn_Nt50.set_index("Iterations", inplace=True)

filename = datadir + "neural_network_benchmark_T2E-02_gamma4E-03_Nt100.dat"
nn_Nt100 = pd.read_csv(filename,
                delim_whitespace=True,
                names=["Iterations", "CPU time", "MSE"]
                )
nn_Nt100.set_index("Iterations", inplace=True)

fig, ax = plt.subplots(1, 1)

ax.loglog(nn_Nt10.MSE, color="k", ls="dashed", label=r"$N_t = 10$")
ax.loglog(nn_Nt50.MSE, color="k", ls="dashdot", label=r"$N_t = 50$")
ax.loglog(nn_Nt100.MSE, color="k", ls="dotted", label=r"$N_t = 100$")
ax.set_xlabel("Iterations", fontsize=20)
ax.set_ylabel("MSE", fontsize=20)
ax.legend(frameon=False, loc="lower left", fontsize=20)

plt.savefig(figdir + "MSE_varying_dt.pdf")
plt.show()
