# Neural Networks from Scratch

## Chapter 2: Coding Our First Neurons
- `Neuron.py` & `Layer.py` implement:
  - A single neuron with and without NumPy
  - A layer of neurons
  - Matrix dot product & vector addition
  - Handling batches of data
- Concepts:
  - Tensors, arrays, vectors
  - Matrix products & transposition
  - Manually defined weights/biases
- Result: Forward pass through a layer using NumPy.

## Chapter 3: Adding Layers
- `Layer_Dense.py` introduces:
  - `Layer_Dense` class for reusability
  - Proper class encapsulation
  - Working with synthetic training data
- Result: Fully modular layer system with input/output handling.

## Chapter 4: Activation Functions
- Implemented in:
  - `Activation_ReLU.py`
  - `Activation_Sigmoid.py`
  - `Activation_Softmax.py`
  - `Activation_Step.py`
  - `Activation_Linear.py`
- Concepts:
  - What activation functions do and why we use them
  - ReLU and Softmax visualized on spiral data
- Result: Forward pass through layered dense → ReLU → dense → Softmax stack.

## Chapter 5: Calculating Network Error with Loss
- `Loss.py` contains:
  - `Loss` base class (averages batch loss)
  - `Loss_CategoricalCrossentropy` subclass:
    - Handles both sparse and one-hot labels
    - Uses `np.clip` to prevent log(0)
- Concepts:
  - Cross-entropy loss explained and implemented
  - Accuracy metrics with `argmax`
- Result: Full forward pass with loss and accuracy printed.