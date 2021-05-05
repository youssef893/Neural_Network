import numpy as np


def sigmoid(inputs):
    return 1 / (1 + np.exp(inputs))


def sigmoid_derivative(inputs):
    return sigmoid(inputs) * (1 - sigmoid(inputs))


# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2));


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size;
