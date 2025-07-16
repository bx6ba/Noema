import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from Layer_Dense import Layer_Dense
from Activation_ReLU import Activation_ReLU
from Activation_Softmax import Activation_Softmax
from Loss import Loss_CategoricalCrossentropy

#Initializing NNFS
nnfs.init()

#Load the dataset
X, y = spiral_data(samples=100, classes=3)

# Create model
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

#Loss function
loss_function = Loss_CategoricalCrossentropy()

#Track best model on each iteration
lowest_loss = 9999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

#Loop for optimizzation
for iteration in range(10000):
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)
    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense2.biases += 0.05 * np.random.randn(1, 3)

    #Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    #Calculating loss
    loss = loss_function.calculate(activation2.output, y)


    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)

    #Find best 
    if loss < lowest_loss:
        print(f'Iteration: {iteration}, loss: {loss:.7f}, acc: {accuracy:.4f}')
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    else:
        #Revert to previous best
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()