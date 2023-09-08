import numpy as np
import pandas as pd

class NeuralNetwork:

    def __init__(self, loss='cross-entropy', randomMultiplier=0.01):
        # Initializing an empty list to store layers
        self.layers = []

        self.randomMultiplier = randomMultiplier

        if loss == 'cross-entropy':
            self.lossFunction = self.crossEntropyLoss
            self.lossBackward = self.crossEntropyLossGrad
        elif loss == 'mean-square-error':
            self.lossFunction = self.meanSquareError
            self.lossBackward = self.meanSquareErrorGrad
        else:
            raise LossFunctionNotDefined
        self.loss = loss

    def addLayer(self, inputDimension=None, units=1, activation=''):
        if (inputDimension is None):
            if (len(self.layers) == 0):
                raise InputDimensionNotCorrect
            inputDimension = self.layers[-1].outputDimension()
        layer = DenseLayer(inputDimension, units, activation, randomMultiplier=self.randomMultiplier)
        self.layers.append(layer)

    def crossEntropyLoss(self, Y, A, epsilon=1e-15):
        m = Y.shape[1]
        loss = -1 * (Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon))
        cost = 1 / m * np.sum(loss)
        return np.squeeze(cost)

    def crossEntropyLossGrad(self, Y, A):
        dA = -(np.divide(Y, A) - np.divide(1 - Y, 1 - A))
        return dA

    def meanSquareError(self, Y, A):
        loss = np.square(Y - A)
        m = Y.shape[1]
        cost = 1 / m * np.sum(loss)
        return np.squeeze(cost)

    def meanSquareErrorGrad(self, Y, A):
        dA = -2 * (Y - A)
        return dA

    def cost(self, Y, A):
        return self.lossFunction(Y, A)

    def forward(self, X):
        x = np.copy(X)
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, A, Y):
        dA = self.lossBackward(Y, A)
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def update(self, learning_rate=0.01):
        for layer in self.layers:
            layer.update(learning_rate)

    def __repr__(self):
        layrepr = ['  ' + str(ix + 1) + ' -> ' + str(x) for ix, x in enumerate(self.layers)]
        return '[\n' + '\n'.join(layrepr) + '\n]'

    def numberOfParameters(self):
        n = 0
        for layer in self.layers:
            n += np.size(layer.weights) + len(layer.bias)
        print(f'There are {n} trainable parameters in the model.')


class DenseLayer:
    def __init__(self, inputDimension, units, activation='', randomMultiplier=0.01):
        self.weights, self.bias = self.initialize(inputDimension, units, randomMultiplier)

        if activation == 'sigmoid':
            self.activation = activation
            self.activationForward = self.sigmoid
            self.activationBackward = self.sigmoidGrad
        elif activation == 'relu':
            self.activation = activation
            self.activationForward = self.relu
            self.activationBackward = self.reluGrad
        elif activation == 'tanh':
            self.activation = activation
            self.activationForward = self.tanh
            self.activationBackward = self.tanhGrad
        else:
            # Default to linear (no activation) if none specified
            self.activation = 'none'
            self.activationForward = self.linear
            self.activationBackward = self.linear

    def initialize(self, nx, nh, randomMultiplier):
        weights = randomMultiplier * np.random.randn(nh, nx)
        bias = np.zeros([nh, 1])
        return weights, bias

    def sigmoid(self, Z):
        A = 1 / (1 + np.exp(-Z))
        return A

    def sigmoidGrad(self, dA):
        s = 1 / (1 + np.exp(-self.prevZ))
        dZ = dA * s * (1 - s)
        return dZ

    def relu(self, Z):
        A = np.maximum(0, Z)
        return A

    def reluGrad(self, dA):
        s = np.maximum(0, self.prevZ)
        dZ = (s > 0) * 1 * dA
        return dZ

    def tanh(self, Z):
        A = np.tanh(Z)
        return A

    def tanhGrad(self, dA):
        s = np.tanh(self.prevZ)
        dZ = (1 - s ** 2) * dA
        return dZ

    def linear(self, Z):
        return Z

    def forward(self, A):
        Z = np.dot(self.weights, A) + self.bias
        self.prevZ = Z
        self.prevA = A
        A = self.activationForward(Z)
        return A

    def backward(self, dA):
        dZ = self.activationBackward(dA)
        m = self.prevA.shape[1]
        self.dW = 1 / m * np.dot(dZ, self.prevA.T)
        self.db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        prevdA = np.dot(self.weights.T, dZ)
        return prevdA

    def update(self, learning_rate):
        self.weights = self.weights - learning_rate * self.dW
        self.bias = self.bias - learning_rate * self.db

    def outputDimension(self):
        return len(self.bias)

    def __repr__(self):
        # String representation of the layer, including the number of input and output units and the activation function
        act = 'none' if self.activation == '' else self.activation
        return f'Dense layer (nx={self.weights.shape[1]}, nh={self.weights.shape[0]}, activation={act})'


import numpy as np





data = pd.read_csv('CW_project4.txt', delim_whitespace=True, header=None)

from sklearn.model_selection import train_test_split



X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1,1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train = X_train.T
X_test = X_test.T
y_train = y_train.T.astype(np.float64)
y_test = y_test.T.astype(np.float64)



nn = NeuralNetwork(loss='mean-square-error')

nn.addLayer(inputDimension=X_train.shape[0], units=10, activation='relu')
nn.addLayer(units=1, activation='sigmoid')

learning_rate = 0.01
num_epochs = 1000

for epoch in range(num_epochs):
    A = nn.forward(X_train)
    nn.backward(A, y_train)
    nn.update(learning_rate)
    if epoch % 100 == 0:
        cost = nn.cost(y_train, A)
        print(f'Epoch {epoch}, Cost: {cost}')

sample_input = X_test[:,0].reshape(-1, 1)
sample_output = y_test[:,0].reshape(-1, 1)
predicted_output = nn.forward(sample_input)

print(f'Sample Input: {sample_input.T}')
print(f'Sample Output: {sample_output}')
print(f'Predicted Output: {predicted_output}')

A_test = nn.forward(X_test)
predictions = (A_test > 0.5).astype(int)
accuracy = np.mean(predictions == y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')