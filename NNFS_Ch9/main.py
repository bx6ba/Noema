import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from Layer_Dense_BP import Layer_Dense
from ReLU_BP import Activation_ReLU
from Softmax_Loss import Activation_Softmax_Loss_CategoricalCrossentropy

nnfs.init()
X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y)

for i, output in enumerate(loss_activation.output[:5]):
    print(f"Sample {i+1}: {output}")

print(f"\nLoss: {loss:.6f}")

predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.2f}")

loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

print("\nDense1 Weights Gradient:\n", dense1.dweights)
print("\nDense1 Biases Gradient:\n", dense1.dbiases)
print("\nDense2 Weights Gradient:\n", dense2.dweights)
print("\nDense2 Biases Gradient:\n", dense2.dbiases)
