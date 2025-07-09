import numpy as np

class Activation_Step:
    def forward(self, inputs):
        self.output = np.array([1 if i > 0 else 0 for i in inputs])
