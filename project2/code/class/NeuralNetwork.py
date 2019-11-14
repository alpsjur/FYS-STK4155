import numpy as np
from sklearn import metrics

class NeuralNetwork:
    def __init__(self, layer_sizes, activation_function):

        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)

        self.activation_function = activation_function

        #initialize weights and biases with random numbers
        self.biases = [np.random.randn(size,1) for size in self.layer_sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
        return


    def feedforward(self, input):
        """
        Function for feeding the input through the Network
            input = array with inputs to the first layer of the network
            returns array with resulting output from the last layer
        """
        activation_function = self.activation_function
        input = input.transpose()
        for layer in range(self.n_layers-1):
            z = np.matmul(self.weights[layer],input) + self.biases[layer]
            input = activation_function(z)
        return input[0]

    def backpropagation(self, input, labels):
        """
        Function for calculationg the gradients of the weights and biases,
        by using the backpropagation algorithm for Neural Networks
        inputs and labels are arrays.
        Returns a list with arrays of bias gradients for each layers,
        and a list with nested arrays of weight gradients for each layer
        """
        biases_gradient = [np.zeros(bias.shape) for bias in self.biases]
        weights_gradient = [np.zeros(weight.shape) for weight in self.weights]
        activation = input.transpose()
        activations = [activation]
        zs = []
        activation_function = self.activation_function
        #calculating and storing activations for each layer
        for layer in range(self.n_layers-1):
            z = np.matmul(self.weights[layer],activation) + self.biases[layer]
            zs.append(z)
            activation = activation_function(z)
            activations.append(activation)
        #calculating gradient for the last layer
        delta = self.cost_derivative(activation[-1], labels)[np.newaxis]
        biases_gradient[-1] = np.sum(delta,axis=1)
        weights_gradient[-1] = np.matmul(delta,activations[-2].transpose())
        #itterating over rest of layers, from last to first
        for layer in range(2, self.n_layers):
            z = zs[-layer]
            delta = np.matmul(self.weights[-layer+1].transpose(), delta)*activation_function.derivative(z)
            biases_gradient[-layer] = np.sum(delta, axis=1)[np.newaxis].transpose()
            weights_gradient[-layer] = np.matmul(delta,activations[-layer-1].transpose())
        return biases_gradient, weights_gradient


    def train(self, training_input, training_labels, learning_rate_init=1, n_epochs=20, minibatch_size=100, \
              test_input=None, test_labels=None, test='accuracy', regularisation = 0.1):
        """
        Function for training the network, using stochastic gradient descent.
            training_input, training_labels are arrays
            test_input, test_labels are arrays, if testing, else None
            test can be either 'accuracy', for clasification, or 'mse', for regression
            learning_rate_init, n_epochs, minibatch_size and regularisation are scalars
        """
        n = len(training_labels)
        for epoch in range(n_epochs):
            idx = np.arange(n)
            np.random.shuffle(idx)
            training_input = training_input[idx]
            training_labels = training_labels[idx]
            labels_mini_batches = [training_labels[i:i+minibatch_size] for i in range(0, n, minibatch_size)]
            input_mini_batches = [training_input[i:i+minibatch_size] for i in range(0, n, minibatch_size)]
            for labels_mini_batch, input_mini_batch in zip(labels_mini_batches, input_mini_batches):
                length_mini_batch = len(labels_mini_batch)
                biases_gradient = [np.zeros(bias.shape) for bias in self.biases]
                weights_gradient = [np.zeros(weight.shape) for weight in self.weights]
                delta_bias_gradient, delta_weight_gradient= self.backpropagation(input_mini_batch, labels_mini_batch)
                biases_gradient = [bg + dbg for  bg, dbg in zip(biases_gradient, delta_bias_gradient)]
                weights_gradient = [wg + dwg for  wg, dwg in zip(weights_gradient, delta_weight_gradient)]
                self.weights = [(1-learning_rate_init*(regularisation/n))*w-(learning_rate_init/length_mini_batch)*wg for w, wg in zip(self.weights, weights_gradient)]
                self.biases = [b-(learning_rate_init/length_mini_batch)*bg for b, bg in zip(self.biases, biases_gradient)]
            if test=='mse':
                print('Epoch {} mse: {:.7f}'.format(epoch, self.mse(test_input, test_labels)))
            elif test=='accuracy':
                print('Epoch {}: {:.3f} accuracy'.format(epoch, self.accuracy(test_input, test_labels)))
            else:
                print('Epoch {} complete'.format(epoch))

    def predict(self, input):
        """
        Function for applying the network on (new) input.
            input = array of inputs to the first layer
        Returns arrays with predictions of binary clasificaion
        """
        probabilities = self.feedforward(input)
        probabilities_array = np.empty(len(probabilities),dtype=np.uint)
        for i in range(len(probabilities)):
            if probabilities[i] > 0.5:
                probabilities_array[i] = 1
            else:
                probabilities_array[i] = 0
        return probabilities_array

    def mse(self, input, labels):
        """
        function for calculating the mean squared error,
            input and labels are arrays
        return mse
        """
        probabilities = self.feedforward(input)
        error = np.mean((probabilities - labels)**2)
        return error

    def r2(self, input, labels, **kwargs):
        """
        Calculates the R2-value of the model.
        """
        probabilities = self.feedforward(input)
        error = 1 - np.sum((labels - probabilities)**2, **kwargs)\
                /np.sum((labels - np.mean(labels, **kwargs))**2, **kwargs)
        return error

    def accuracy(self, input, labels):
        """
        function for calculating the accuracy,
            input and labels are arrays
        return accuracy
        """
        predictions = self.predict(input)
        count = 0
        for prediction, target in zip(predictions, labels):
            if prediction == target:
                count += 1
        return count/len(labels)

    def auc(self,input,labels):
        """
        Function for calculating the AUC
            input and labels are arrays
        returns AUC
        """
        targets = self.predict_probabilities(input)
        score = metrics.roc_auc_score(labels,targets)
        return score

    def predict_probabilities(self, input):
        """
        Function for applying the network on (new) input.
            input = array of inputs to the first layer
        Returns the probability output, or in the regression case, the
        predicted function value
        """
        probabilities = self.feedforward(input)
        return probabilities


    def cost_derivative(self, output_activations, labels):
        return (output_activations-labels)
