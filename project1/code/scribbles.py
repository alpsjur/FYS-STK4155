from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed

import seaborn as sns


sns.set()
sns.set_style("whitegrid")
sns.set_palette("husl")


def generate_design_polynomial(x, p=1):
    X = np.zeros((len(x), p+1))
    for degree in range(0, p+1):
        X[:, degree] = (x.T)**degree
    return X

def generate_design_2Dpolynomial(x, y, degree=5):
    X = np.zeros(( len(x), int(0.5*(degree + 2)*(degree + 1)) ))
    #X = np.zeros(( len(x), (degree + 1)**2))

    p = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            X[:, p] = (x.T)**i*(y.T)**j
            p += 1
    return X

def least_squares(X, data):
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(data)
    model = X @ beta
    return model

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


fig = plt.figure()
ax = fig.gca(projection="3d")

# Generating grid in x,y in [0, 1] with n=100 points
n = 1000

x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
x, y = np.meshgrid(x,y)

# generating design matrices for both coordinates
x_random = np.random.uniform(0, 1, n)
x_sorted = np.sort(x_random, axis=0)

y_random = np.random.uniform(0, 1, n)
y_sorted = y_random[np.argsort(x_random, axis=0)].flatten()

z = FrankeFunction(x, y)

X = generate_design_2Dpolynomial(x_sorted, y_sorted, degree=5)
z_model = least_squares(X, z)

print(z)
print(z_model)
# Plot the surface.
surf = ax.plot_surface(x, y, z,
                        cmap=cm.coolwarm,
                        linewidth=0,
                        antialiased=False
                        )
surf = ax.plot_surface(x, y, z_model,
                        cmap=cm.Blues,
                        linewidth=0,
                        antialiased=False
                        )
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
# Add a color bar which maps values to colors.
fig.colorbar(surf,
            shrink=0.5,
            aspect=5
            )
plt.show()
