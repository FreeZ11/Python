import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


def relu(x):
    return x * (x > 0)


def relu_derivative(x):
    return 1. * (x > 0)


class NeuralNetwork:
    def __init__(self, x, y, func1, func2, func_d1, func_d2):
        self.input = x
        self.weights1 = np.random.rand(100, self.input.shape[1])
        self.weights2 = np.random.rand(1, 100)
        self.y        = y
        self.output   = np.zeros(self.y.shape)
        self.eta      = 0.0000008
        self.func1    = func1
        self.func2    = func2
        self.func_d1  = func_d1
        self.func_d2  = func_d2

    def feedforward(self):
        self.layer1 = self.func1(np.dot(self.input, self.weights1.T))
        self.output = self.func2(np.dot(self.layer1, self.weights2.T))

    def backprop(self):
        delta2 = (self.y - self.output) * self.func_d2(self.output)
        d_weights2 = self.eta * np.dot(delta2.T, self.layer1)
        delta1 = self.func_d1(self.layer1) * np.dot(delta2, self.weights2)
        d_weights1 = self.eta * np.dot(delta1.T, self.input)
        self.weights1 += d_weights1
        self.weights2 += d_weights2


if __name__ == '__main__':
    X = np.linspace(-50, 50, 101)
    x = np.array([[z, 1.0] for z in X])
    y = np.array([[z**2] for z in X])
    np.random.seed(19)
    nn = NeuralNetwork(x, y, sigmoid, relu, sigmoid_derivative, relu_derivative)
    c = 0
    for i in range(100000):
        nn.feedforward()
        nn.backprop()
        if c % 1000 == 0:
            plt.clf()
            plt.title("Step: {}".format(c))
            plt.scatter(X, nn.output)
            plt.draw()
            plt.pause(0.001)

        c += 1

    X = np.linspace(0, 2, 161)
    y = np.sin(3 * np.pi / 2) * X
    x = np.array([[z, 1.0] for z in X])
    y = np.array([[z] for z in X])

    np.random.seed(19)
    nn = NeuralNetwork(x, y, sigmoid, relu, sigmoid_derivative, relu_derivative)
    nn.weights1 = np.random.rand(50, nn.input.shape[1])
    nn.weights2 = np.random.rand(1, 50)
    c = 0
    for i in range(100000):
        nn.feedforward()
        nn.backprop()
        if c % 1000 == 0:
            plt.clf()
            plt.title("Step: {}".format(c))
            plt.scatter(X, nn.output)
            plt.draw()
            plt.pause(0.001)

        c += 1
