import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, regression=False):

        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)

        self.regression = regression

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
            bias2D = self.biases[layer][np.newaxis]
            z = np.matmul(input,self.weights[layer].transpose()) + bias2D
            input = self.sigmoid(z)
        if self.regression:
            return z.transpose()[0]
        else:
            return input.transpose()[0]

    def backpropagation(self, input, labels):
        """
        Function for calculationg the backwards propagating correction of the
        weights and biases, given a learning rate, using gradient descent
        """
        biases_gradient = [np.zeros(bias.shape) for bias in self.biases]
        weights_gradient = [np.zeros(weight.shape) for weight in self.weights]
        activation = input
        activations = [activation]
        zs = []

        for layer in range(self.n_layers-1):
            z = np.matmul(self.weights[layer], activation) + self.biases[layer]
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        if self.regression:
            delta = self.cost_derivative(zs[-1],labels)
        else:
            delta = self.cost_derivative(activations[-1],labels)*self.sigmoid_derivative(zs[-1])
        biases_gradient[-1] = delta
        #add new axis so that python handles matrix multiplication
        activation2D = activations[-2][np.newaxis]
        weights_gradient[-1] = np.matmul(delta, activation2D)

        for layer in range(2, self.n_layers):
            z = zs[-layer]
            delta = np.dot(self.weights[-layer+1].transpose(), delta)*self.sigmoid_derivative(z)
            biases_gradient[-layer] = delta
            #add new axis so that python handles matrix multiplication
            activation2D = activations[-layer-1][np.newaxis]
            delta2D = delta[np.newaxis].transpose()
            weights_gradient[-layer] = np.matmul(delta2D, activation2D)
        return biases_gradient, weights_gradient


    def train(self, training_input, training_labels ,n_epochs, batch_size, \
              learning_rate, test_input=None, test_labels=None, test=False):
        #code for stochastic gradient decent
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

            if test:
                if self.regression:
                    print('Epoch {} mse: {:.3f}'.format(epoch, self.evaluate(test_input, test_labels)))
                else:
                    print('Epoch {}: {:.3f} correct'.format(epoch, self.evaluate(test_input, test_labels)))
            else:
                print('Epoch {} complete'.format(epoch))

    def predict(self, input):
        """
        Function for applying the network on (new) input.
            input = array of inputs to the first layer
        Returns arrays with predictions
        """
        probabilities = self.feedforward(input)
        probabilities_array = np.empty(len(probabilities),dtype=np.uint)
        for i in range(len(probabilities)):
            if probabilities[i] > 0.5:
                probabilities_array[i] = 1
            if probabilities[i] <= 0.5:
                probabilities_array[i] = 0
        return probabilities_array

    def evaluate(self, input, labels):
        if self.regression:
            predictions = self.predict_regression(input)
            n = len(labels)
            error = np.sum((predictions - labels)**2)/n
            return error
        else:
            predictions = self.predict(input)
            count = 0
            for prediction, target in zip(predictions, labels):
                if prediction == target:
                    count += 1
            return count/len(labels)

    def predict_probabilitie(self, input):
        """
        Function for applying the network on (new) input.
            input = array of inputs to the first layer
        Returns the probability output
        """
        probabilities = self.feedforward(input)
        return probabilities

    def predict_regression(self, input):
        prediction = self.feedforward(input)
        return prediction

    def cost_derivative(self, output_activations, labels):
        return output_activations-labels

    def sigmoid(self, z):
        return np.exp(z)/(1+np.exp(z))

    def sigmoid_derivative(self, z):
        return np.exp(z)/(1 + np.exp(z))**2
