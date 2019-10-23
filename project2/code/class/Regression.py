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

    def fit(self):
        self.model = self.designMatrix @ self.beta
        return self.model

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

class OLS(Regression):
    def construct_model(self):
        designMatrix = self.designMatrix
        self.beta = np.linalg.pinv(designMatrix.T.dot(designMatrix)).dot(designMatrix.T).dot(self.data)
        self.betas.append(self.beta)
        return

class Ridge(Regression):
    def __init__(self, designMatrix, data, hyperparameter):
        self.hyperparameter = hyperparameter
        super().__init__(designMatrix, data)

    def construct_model(self):
        designMatrix = self.designMatrix
        p = len(designMatrix[0, :])
        self.beta = np.linalg.pinv(designMatrix.T.dot(designMatrix)
                + self.hyperparameter*np.identity(p)).dot(designMatrix.T).dot(self.data)
        self.betas.append(self.beta)
        return

class Lasso(Ridge):
    def __init__(self, designMatrix, data, hyperparameter, **kwargs):
        self.kwargs = kwargs
        super().__init__(designMatrix, data, hyperparameter)

    def constuct_model(self):
        reg = linear_model.Lasso(alpha=self.hyperparameter, **self.kwargs)
        reg.fit(self.designMatrix, self.data)
        self.beta = reg.coef_
        self.betas.append(self.beta)
        return

class Logistic(Ridge):
    def __init__(self, designMatrix, data, hyperparameter, learning_rates):
        self.learning_rates = learning_rates
        super().__init__(designMatrix, data, hyperparameter)

    def sigmoid(self, x):
        f = np.exp(x)/(np.exp(x) + 1)
        return f

    def calculate_gradient(self):
        gradient = 2*self.designMatrix.T.dot(self.designMatrix.dot(self.beta) - sigmoid(self.data))
        return gradient

    def gradient_descent(self, tol=1e-8, max_iter=1e4):
        k = 0
        while gradC > tol and k <= max_iter:
            pass


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

    linreg = Lasso(X, z, 1e-5, tol=1e3, max_iter=1e5)
    linreg.construct_model()
    model = linreg.fit()
    print(linreg.r2())

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    # Plot the surface.
    surf = ax.plot_surface(x_grid, y_grid, model.reshape(x_grid.shape),
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
