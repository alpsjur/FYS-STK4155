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
        Function for calculationg the backwards propagating correction of the
        weights and biases, given a learning rate, using gradient descent
        """
        biases_gradient = [np.zeros(bias.shape) for bias in self.biases]
        weights_gradient = [np.zeros(weight.shape) for weight in self.weights]
        activation = input.transpose()
        activations = [activation]
        zs = []
        activation_function = self.activation_function
        for layer in range(self.n_layers-1):
            z = np.matmul(self.weights[layer],activation) + self.biases[layer]
            zs.append(z)
            activation = activation_function(z)
            activations.append(activation)

        delta = self.cost_derivative(activation[-1], labels)[np.newaxis]#*self.activation_function.derivative(zs[-1])
        biases_gradient[-1] = np.sum(delta,axis=1)
        #delta = self.cost_derivative(activation[-1], labels)#*self.activation_function.derivative(zs[-1])
        weights_gradient[-1] = np.matmul(delta,activations[-2].transpose())

        for layer in range(2, self.n_layers):
            z = zs[-layer]
            delta = np.matmul(self.weights[-layer+1].transpose(), delta)*activation_function.derivative(z)
            biases_gradient[-layer] = np.sum(delta, axis=1)[np.newaxis].transpose()
            weights_gradient[-layer] = np.matmul(delta,activations[-layer-1].transpose())
        return biases_gradient, weights_gradient


    def train(self, training_input, training_labels, learning_rate_init=1, n_epochs=20, minibatch_size=100, \
              test_input=None, test_labels=None, test='accuracy', regularisation = 0.1):
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
        Returns arrays with predictions
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
        n = len(labels)
        probabilities = self.feedforward(input)
        error = np.sum((probabilities - labels)**2)/n
        return error

    def accuracy(self, input, labels):
        predictions = self.predict(input)
        count = 0
        for prediction, target in zip(predictions, labels):
            if prediction == target:
                count += 1
        return count/len(labels)

    def predict_probabilities(self, input):
        """
        Function for applying the network on (new) input.
            input = array of inputs to the first layer
        Returns the probability output
        """
        probabilities = self.feedforward(input)
        return probabilities


    def cost_derivative(self, output_activations, labels):
        return (output_activations-labels)#/(output_activations*(1-output_activations))

    def learning_schedule(self, t, t0, t1):
        return t0/(t+t1)

    def auc(self,designMatrix,labels):
        targets = self.predict(designMatrix)
        score = metrics.roc_auc_score(targets,labels)
        return score
