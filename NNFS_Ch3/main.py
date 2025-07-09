import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from Layer_Dense import Layer_Dense

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
dense1.forward(X)

dense2 = Layer_Dense(3, 3)
dense2.forward(dense1.output)

print(dense2.output[:5])