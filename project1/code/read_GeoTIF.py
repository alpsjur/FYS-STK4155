import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns

# Load the terrain
terrain1 = imread('../data/test_data_oslo.tif')

# Show the terrain
sns.set()
sns.set_style("whitegrid")
sns.set_palette("husl")

fig = plt.figure()
ax = fig.gca(projection="3d")
ax.set_aspect('equal')

#get the number of points
n_y, n_x = np.shape(terrain1)

#making an x and y grid (may want to define x and y differently)
x_grid, y_grid = np.meshgrid(np.arange(n_x),np.arange(n_y))

print(np.shape(terrain1))
print(np.shape(x_grid))

# Plot the surface.
surf = ax.plot_surface(x_grid, y_grid, terrain1,
                        cmap=cm.coolwarm,
                        linewidth=0,
                        antialiased=False,
                        alpha = 0.5,
                        )

# Add a color bar which maps values to colors.
fig.colorbar(surf,
            shrink=0.5,
            aspect=5
            )
plt.show()

'''
plt.figure()
plt.title('Terrain over Oslo fjord')
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
'''
