from generate_dataset import test_data, train_labels, test_labels, train_data
import numpy as np
from NeuralNetwork import NeuralNetwork
from FullyConnectedLayer import FullyConnectedLayer
from ActivationLayer import ActivationLayer
from activation_and_loss_function import sigmoid, sigmoid_derivative, mse, mse_prime


def getData(train_data, train_labels, test_data, test_labels):
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    return train_data, train_labels, test_data, test_labels


def calculate_accuracy(target, predictions):
    class1 = []
    for i in predictions:
        if i > 0.5:
            class1.append(1)
        else:
            class1.append(0)
    counter = 0
    for i in range(len(predictions)):
        if class1[i] == test_labels[i]:
            counter += 1
    print(counter / len(predictions))


def make_ANN(train_data, train_labels, test_labels):
    net = NeuralNetwork()
    net.add(FullyConnectedLayer(2, 3))
    net.add(ActivationLayer(sigmoid, sigmoid_derivative))
    net.add(FullyConnectedLayer(3, 3))
    net.add(ActivationLayer(sigmoid, sigmoid_derivative))
    net.add(FullyConnectedLayer(3, 1))
    net.add(ActivationLayer(sigmoid, sigmoid_derivative))

    net.loss_function(mse, mse_prime)
    net.fit(train_data, train_labels, epochs=100, learning_rate=0.1)

    out = net.predict(test_data)
    calculate_accuracy(test_labels, out)
    print("\n")
    print("predicted values : ", out)


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = getData(train_data, train_labels, test_data, test_labels)
    make_ANN(train_data, train_labels, test_labels)
