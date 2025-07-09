import numpy as np

class Layer:
    def __init__(self, weights, biases):
        self.weights = np.array(weights)
        self.biases = np.array(biases)

    def forward(self, inputs):
        return np.dot(inputs, self.weights.T) + self.biases