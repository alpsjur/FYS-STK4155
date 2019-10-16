import numpy as np
from sklearn import linear_model


class Regression:
    def __init__(self, designMatrix, data):
        self.designMatrix = designMatrix
        self.data = data
        self.beta = None
        self.betas = []
        self.model = None
        return

    def OLS(self):
        designMatrix = self.designMatrix
        self.beta = np.linalg.pinv(designMatrix.T.dot(designMatrix)).dot(designMatrix.T).dot(self.data)
        self.betas.append(self.beta)
        return

    def ridge(self, hyperparameter):
        designMatrix = self.designMatrix
        p = len(designMatrix[0, :])
        self.beta = np.linalg.pinv(designMatrix.T.dot(designMatrix)
                + hyperparameter*np.identity(p)).dot(designMatrix.T).dot(self.data)
        self.betas.append(self.beta)
        return

    def lasso(self, hyperparameter, **kwargs):
        reg = linear_model.Lasso(alpha=hyperparameter, **kwargs)
        reg.fit(self.designMatrix, self.data)
        self.beta = reg.coef_
        self.betas.append(self.beta)
        return

    def logistic(self):
        #self.beta =
        return

    def construct_model(self):
        self.model = self.designMatrix @ self.beta
        return

    def clear_betas(self):
        self.betas = []
        return

    def mse(self):
        """
        Calculates the mean square error between data and model.
        """
        n = len(self.data)
        error = np.sum((self.data - self.model)**2)/n
        return error

    def r2(self):
        """
        Calculates the R2-value of the model.
        """
        n = len(self.data)
        error = 1 - np.sum((self.data - self.model)**2)/np.sum((self.data - np.mean(self.data))**2)
        return error

    def bias(self):
        """caluclate bias from k expectation values and data of length n"""
        n = len(self.data)
        error = mse(self.data, np.mean(self.model))
        return error

    def variance(self):
        """
        Calculating the variance of the model: Var[model]
        """
        n = len(self.model)
        error = mse(self.model, np.mean(self.model))
        return error


if __name__ == "__main__":
    import projectfunctions as pf
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm


    n = 20
    noise = 0.1

    x_grid, y_grid = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))

    #flatten x and y
    x = x_grid.flatten()
    y = y_grid.flatten()

    #compute z and flatten it
    z_grid = pf.frankefunction(x_grid, y_grid) + np.random.normal(0, noise, x_grid.shape)
    z = z_grid.flatten()

    X = pf.generate_design_2Dpolynomial(x, y)

    linreg = LinearRegression(X, z)
    linreg.OLS()
    linreg.construct_model()

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    # Plot the surface.
    surf = ax.plot_surface(x_grid, y_grid, linreg.model.reshape(x_grid.shape),
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
