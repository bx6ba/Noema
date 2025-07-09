from Neuron import Neuron
from Layer import Layer

# Single neuron
neuron = Neuron([0.2, 0.8, -0.5, 1.0], 2.0)
print(neuron.forward([1.0, 2.0, 3.0, 2.5]))

# Layer of neurons
inputs = [
    [1.0, 2.0, 3.0, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]
weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]
biases = [2.0, 3.0, 0.5]

layer = Layer(weights, biases)
print(layer.forward(inputs))