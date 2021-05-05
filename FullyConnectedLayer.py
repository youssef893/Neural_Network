from Layer import Layer
import numpy as np


class FullyConnectedLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias  # calculate output
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)  # calculate input error =  derivative of error respect to weights * weights
        weights_error = np.dot(self.input.T, output_error)  # calculate input error =  derivative of error respect to weights * input
        # dBias = output_error

        # update parameters
        self.weights = self.weights - learning_rate * weights_error
        self.bias = self.bias - learning_rate * output_error
        return input_error
