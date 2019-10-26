import numpy as np

class NeuralNetwork:
    def __init__(self,
                layer_sizes,
                #epochs=10,
                #batch_size=100,
                #eta=0.1,
                #lmbd=0.0
                ):

        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)

        #initialize weights and biases with random numbers
        self.biases = [np.random.randn(size) for size in layer_sizes[1:]]
        self.weights = [np.random.randn(size, size_prew) for size, size_prew \
                        in zip(layer_sizes[1:], layer_sizes[:-1])]


    def feedforward(self, input):
        for layer in range(self.n_layers-1):
            z = np.dot(self.weights[layer], input) + self.biases[layer]
            input = self.sigmoid(z)
        return input

    def backpropagation(self, input, labels):
        self.biases_gradient = [np.zeros(bias.shape) for bias in self.biases]
        self.weights_gradient = [np.zeros(weight.shape) for weight in self.weights]

        activation = input
        activations = [activation]
        zs = []
        for layer in range(self.n_layers-1):
            z = np.dot(self.weights[layer], activation) + self.biases[layer]
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        delta = (activations[-1]-labels)*sigmoid_derivative(zs[-1])
        self.biases_gradient[-1] = delta
        self.weights_gradient[-1] = np.dot(delta, activations[-2].transpose())

        self.biases[-1] -= self.learning_rate*self.biases_gradient[-1]
        self.weights[-1] -= self.learning_rate*self.weights_gradient[-1]

        for layer in range(2, self.n_layers):
            z = zs[-layer]
            delta = np.dot(self.weights[-layer+1].transpose(), delta)*self.sigmoid_derivative(z)
            self.biases_gradient[-layer] = delta
            self.weights_gradient[-layer] = np.dot(delta, activations[-layer-1].transpose())

            self.biases[-layer] -= self.learning_rate*self.biases_gradient[-layer]
            self.weights[-layer] -= self.learning_rate*self.weights_gradient[-layer]

    def train(self):
        #kode for stochastic gradient decent 

    def predict(self, input):
        probabilities = self.feedforward(input)
        return np.argmax(probabilities)

    def predict_probabilities(self, input):
        probabilities = self.feed_forward(input)
        return probabilities

    def sigmoid(z):
        return np.exp(z)/(1-np.exp(z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
