import numpy as np


def sigmoid(x):
    return 1.0/(1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


def relu(x):
    return x * (x > 0)


def relu_derivative(x):
    return 1. * (x > 0)


class NeuralNetwork:
    def __init__(self, x, y, func1, func2, func_d1, func_d2, flag):
        self.input = x
        self.weights1 = np.random.rand(4, self.input.shape[1])
        self.weights2 = np.random.rand(1, 4)
        self.y        = y
        self.output   = np.zeros(self.y.shape)
        self.eta      = 0.5
        self.func1    = func1
        self.func2    = func2
        self.func_d1  = func_d1
        self.func_d2  = func_d2
        self.flag = flag

    def feedforward(self):
        if self.flag == 'sigma':
            self.layer1 = self.func1(np.dot(self.input, self.weights1.T))
            self.output = self.func1(np.dot(self.layer1, self.weights2.T))
        elif self.flag == 'relu':
            self.layer1 = self.func2(np.dot(self.input, self.weights1.T))
            self.output = self.func2(np.dot(self.layer1, self.weights2.T))
        elif self.flag == 'sig-rel':
            self.layer1 = self.func1(np.dot(self.input, self.weights1.T))
            self.output = self.func2(np.dot(self.layer1, self.weights2.T))
        elif self.flag == 'rel-sig':
            self.layer1 = self.func2(np.dot(self.input, self.weights1.T))
            self.output = self.func1(np.dot(self.layer1, self.weights2.T))

    def backprop(self):
        if self.flag == 'sigma':

            delta2 = (self.y - self.output) * self.func_d1(self.output)
            d_weights2 = self.eta * np.dot(delta2.T, self.layer1)
            delta1 = self.func_d1(self.layer1) * np.dot(delta2, self.weights2)
            d_weights1 = self.eta * np.dot(delta1.T, self.input)
        elif self.flag == 'relu':
            delta2 = (self.y - self.output) * self.func_d2(self.output)
            d_weights2 = self.eta * np.dot(delta2.T, self.layer1)
            delta1 = self.func_d2(self.layer1) * np.dot(delta2, self.weights2)
            d_weights1 = self.eta * np.dot(delta1.T, self.input)
        elif self.flag == 'sig-rel':
            delta2 = (self.y - self.output) * self.func_d2(self.output)
            d_weights2 = self.eta * np.dot(delta2.T, self.layer1)
            delta1 = self.func_d1(self.layer1) * np.dot(delta2, self.weights2)
            d_weights1 = self.eta * np.dot(delta1.T, self.input)
        else:
            delta2 = (self.y - self.output) * self.func_d1(self.output)
            d_weights2 = self.eta * np.dot(delta2.T, self.layer1)
            delta1 = self.func_d2(self.layer1) * np.dot(delta2, self.weights2)
            d_weights1 = self.eta * np.dot(delta1.T, self.input)
        self.weights1 += d_weights1
        self.weights2 += d_weights2



if __name__ == '__main__':
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    y = np.array([[0], [1], [1], [0]])
    # sigma only
    nn = NeuralNetwork(X, y, sigmoid, relu, sigmoid_derivative, relu_derivative, 'sigma')

    for i in range(5000):
        nn.feedforward()
        nn.backprop()

    np.set_printoptions(precision=4, suppress=True)
    print("Sigma only results")
    print(nn.output)
    print(nn.weights1)
    print(nn.weights2)


    # relu only
    np.random.seed(17)
    nn = NeuralNetwork(X, y, sigmoid, relu, sigmoid_derivative, relu_derivative, 'relu')
    nn.eta = 0.01
    for i in range(5000):
        nn.feedforward()
        nn.backprop()

    np.set_printoptions(precision=4, suppress=True)
    print("\nrelu onlt results")
    print(nn.output)
    print(nn.weights1)
    print(nn.weights2)


    # sigma + relu
    np.random.seed(None)
    nn = NeuralNetwork(X, y, sigmoid, relu, sigmoid_derivative, relu_derivative, 'sig-rel')
    nn.eta = 0.2
    for i in range(5000):
        nn.feedforward()
        nn.backprop()

    np.set_printoptions(precision=4, suppress=True)
    print("\nsigma + relu")
    print(nn.output)
    print(nn.weights1)
    print(nn.weights2)


    # relu + sigma
    np.random.seed(None)
    nn = NeuralNetwork(X, y, sigmoid, relu, sigmoid_derivative, relu_derivative, 'rel-sig')
    nn.eta = 1.0
    for i in range(5000):
        nn.feedforward()
        nn.backprop()

    np.set_printoptions(precision=4, suppress=True)
    print("\nrelu + sigma")
    print(nn.output)
    print(nn.weights1)
    print(nn.weights2)

    # OR
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    y = np.array([[1], [1], [1], [0]])

    nn = NeuralNetwork(X, y, sigmoid, relu, sigmoid_derivative, relu_derivative, 'sigma')

    for i in range(5000):
        nn.feedforward()
        nn.backprop()

    np.set_printoptions(precision=4, suppress=True)
    print("\nOR problem")
    print(nn.output)
    print(nn.weights1)
    print(nn.weights2)

    #AND
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    y = np.array([[0], [0], [0], [1]])

    nn = NeuralNetwork(X, y, sigmoid, relu, sigmoid_derivative, relu_derivative, 'sigma')

    for i in range(5000):
        nn.feedforward()
        nn.backprop()

    np.set_printoptions(precision=4, suppress=True)
    print("\nAND problem")
    print(nn.output)
    print(nn.weights1)
    print(nn.weights2)

#ostatnia kolumna to tak zway bias który ma za zadanie odpowiednio poprawiać output zależnie od wag
