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
        """
        Function for feeding the input through the Network
            input = array with inputs to the first layer of the network
            returns array with resulting output from the last layer
        """
        for layer in range(self.n_layers-1):
            z = np.dot(self.weights[layer], input) + self.biases[layer]
            input = self.sigmoid(z)
        return input

    def backpropagation(self, input, labels):
        """
        Function for calculationg the backwards propagating correction of the
        weights and biases, given a learning rate, using gradient descent
            input = array with the inputs to the first layer of the network
            labels = array with the output matching the correct labeling of
            the input.
            In the case of binary output:
            If correct label is 1, labels = [0,1]
            If correct label is 0, labels = [1,0]
        """
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
        """
        Function for applying the network on (new) input.
            input = array of inputs to the first layer
        Returns the index of the  output neuron with highest value
        """
        probabilities = self.feedforward(input)
        return np.argmax(probabilities)

    def predict_probabilities(self, input):
        probabilities = self.feed_forward(input)
        return probabilities
        """
        Function for applying the network on (new) input.
            input = array of inputs to the first layer
        Returns the full output of the last layer as an array, i.e. the
        porbability for each class   
        """

    def sigmoid(z):
        return np.exp(z)/(1-np.exp(z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
