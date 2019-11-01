import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):

        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)

        #initialize weights and biases with random numbers
        self.biases = [np.random.randn(size) for size in layer_sizes[1:]]
        self.weights = [np.random.randn(size, size_prew) for size_prew, size \
                        in zip(layer_sizes[:-1], layer_sizes[1:])]

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
        """
        biases_gradient = [np.zeros(bias.shape) for bias in self.biases]
        weights_gradient = [np.zeros(weight.shape) for weight in self.weights]
        activation = input
        #activation = activation[:,np.newaxis]
        activations = [activation]
        zs = []

        for layer in range(self.n_layers-1):
            z = np.dot(self.weights[layer], activation) + self.biases[layer]
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1],labels)*self.sigmoid_derivative(zs[-1])
        biases_gradient[-1] = delta
        weights_gradient[-1] = np.matmul(np.vstack(activations[-2]),delta)

        for layer in range(2, self.n_layers):
            z = zs[-layer]
            delta = np.dot(self.weights[-layer+1].transpose(), delta)*self.sigmoid_derivative(z)
            biases_gradient[-layer] = delta
            weights_gradient[-layer] = np.matmul(np.vstack(activations[-layer-1]),delta)
        return biases_gradient, weights_gradient


    def train(self, training_input, training_labels ,n_epochs, batch_size, learning_rate):
        #kode for stochastic gradient decent
        n = len(training_labels)
        for epoch in range(n_epochs):
            idx = np.arange(n)
            np.random.shuffle(idx)
            training_input = training_input[idx]
            training_labels = training_labels[idx]
            labels_mini_batches = [training_labels[i:i+batch_size] for i in range(0, n, batch_size)]
            input_mini_batches = [training_input[i:i+batch_size] for i in range(0, n, batch_size)]
            for labels_mini_batch, input_mini_batch in zip(labels_mini_batches, input_mini_batches):
                biases_gradient = [np.zeros(bias.shape) for bias in self.biases]
                weights_gradient = [np.zeros(weight.shape) for weight in self.weights]
                for label, input in zip(labels_mini_batch, input_mini_batch):
                    delta_bias_gradient, delta_weight_gradient= self.backpropagation(input, label)
                    biases_gradient = [bg + dbg for  bg, dbg in zip(biases_gradient, delta_bias_gradient)]
                    weights_gradient = [wg + dwg for  wg, dwg in zip(weights_gradient, delta_weight_gradient)]
                self.biases = [b - learning_rate*bg for b, bg in zip(self.biases, biases_gradient)]
                self.weights = [w - learning_rate*wg for w, wg in zip(self.weights, weights_gradient)]

        #if test_data:
        #    print('Epoch {}: {}/{}'.format(j, self.evaluate(test_data), n_test))
        #else:
        #    print('Epoch {} complete'.format(j))

    def predict(self, input):
        """
        Function for applying the network on (new) input.
            input = array of inputs to the first layer
        Returns the index of the  output neuron with highest value
        """
        probabilities = self.feedforward(input)
        for i in range(len(probabilities)):
            if probabilities[i] > 0.5:
                probabilities[i] = 1
            else:
                probabilities[i] = 0
        return probabilities

    def predict_probabilities(self, input):
        """
        Function for applying the network on (new) input.
            input = array of inputs to the first layer
        Returns the probability output
        """
        probabilities = self.feed_forward(input)
        return probabilities

    def cost_derivative(self, output_activations, labels):
        return output_activations-labels

    def sigmoid(self, z):
        return np.exp(z)/(1-np.exp(z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
