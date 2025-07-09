import numpy as np
import nnfs
from nnfs.datasets import vertical_data

from Layer_Dense import Layer_Dense
from Activation_ReLU import Activation_ReLU
from Activation_Softmax import Activation_Softmax
from Loss import Loss_CategoricalCrossentropy

nnfs.init()

X, y = vertical_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss = loss_function.calculate(activation2.output, y)
print('Loss:', loss)

predictions = np.argmax(activation2.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)
print('Accuracy:', accuracy)