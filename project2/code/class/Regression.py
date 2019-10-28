import numpy as np
from sklearn import linear_model


class Regression:
    def __init__(self, designMatrix, labels):
        self.designMatrix = designMatrix
        self.labels = labels
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
        n = len(self.labels)
        error = np.sum((self.labels - self.model)**2)/n
        return error

    def r2(self):
        """
        Calculates the R2-value of the model.
        """
        n = len(self.labels)
        error = 1 - np.sum((self.labels - self.model)**2)/np.sum((self.labels - np.mean(self.labels))**2)
        return error

    def bias(self):
        """caluclate bias from k expectation values and data of length n"""
        n = len(self.labels)
        error = mse(self.labels, np.mean(self.model))
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
        self.beta = np.linalg.pinv(designMatrix.T.dot(designMatrix)).dot(designMatrix.T).dot(self.labels)
        self.betas.append(self.beta)
        return

class Ridge(Regression):
    def __init__(self, designMatrix, labels, hyperparameter):
        self.hyperparameter = hyperparameter
        super().__init__(designMatrix, labels)

    def construct_model(self):
        designMatrix = self.designMatrix
        p = len(designMatrix[0, :])
        self.beta = np.linalg.pinv(designMatrix.T.dot(designMatrix)
                + self.hyperparameter*np.identity(p)).dot(designMatrix.T).dot(self.labels)
        self.betas.append(self.beta)
        return

class Lasso(Ridge):
    def __init__(self, designMatrix, labels, hyperparameter, **kwargs):
        self.kwargs = kwargs
        super().__init__(designMatrix, labels, hyperparameter)

    def constuct_model(self):
        reg = linear_model.Lasso(alpha=self.hyperparameter, **self.kwargs)
        reg.fit(self.designMatrix, self.labels)
        self.beta = reg.coef_
        self.betas.append(self.beta)
        return

class Logistic(Regression):
    def __init__(self, designMatrix, labels):
        self.cost_gradient = None
        super().__init__(designMatrix, labels)
        self.beta = np.random.randn(len(self.designMatrix[0, :]))
        self.targets = np.zeros(len(self.labels))

    def sigmoid(self, x):
        f = np.exp(x)/(np.exp(x) + 1)
        return f

    def fit(self):
        self.model = self.sigmoid(self.designMatrix @ self.beta)
        return self.model

    def calculate_cost_gradient(self, labels, designMatrix):
        m = designMatrix.shape[0]
        probabilities = self.sigmoid(designMatrix.dot(self.beta))
        self.cost_gradient = designMatrix.T.dot((labels - probabilities))/m
        return

    def construct_model(self, n_epochs, mini_batch_size, learning_rate):
        """
        Stochastic gradient descent for computing the parameters that minimize the cost function.
            n_epochs = number of epochs
            mini_batch_size = number of data points in each mini batch
            learning_rate = the learning rate, often denoted eta
        """
        n = self.labels.shape[0]
        for epoch in range(n_epochs):
            idx = np.arange(n)
            np.random.shuffle(idx)
            self.labels = self.labels[idx]
            self.designMatrix = self.designMatrix[idx]
            labels_mini_batches = [self.labels[i:i+mini_batch_size] for i in range(0, n, mini_batch_size)]
            designMatrix_mini_batches = [self.designMatrix[i:i+mini_batch_size] for i in range(0, n, mini_batch_size)]
            for labels_mini_batch, designMatrix_mini_batch in zip(labels_mini_batches, designMatrix_mini_batches):
                self.calculate_cost_gradient(labels_mini_batch, designMatrix_mini_batch)
                self.beta = self.beta - learning_rate*self.cost_gradient
            self.betas.append(self.beta)
        return

    def calculate_targets(self):
        for i in range(len(self.labels)):
            if self.model[i] > 0.5:
                self.targets[i] = 1
            else:
                self.targets[i] = 0
        return

    def indicator(self, target, label):
        if target == label:
            return 1
        else:
            return 0

    def accuracy(self):
        self.calculate_targets()
        print(self.targets)
        n = len(self.labels)
        counter = 0
        for i in range(n):
            if self.indicator(self.targets[i], self.labels[i]):
                counter += 1
        return counter/n


if __name__ == "__main__":
    import projectfunctions as pf
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm


    n = 20
    noise = 0.1

    x_grid, y_grid = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))

    #flatten x and y
    x = x_grid.flatten()
    y = y_grid.flatten()

    # compute z and flatten it
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
