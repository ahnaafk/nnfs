import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

import matplotlib.pyplot as plt

#dense layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons) -> None:
        #initialize weights and biases
        self.weights = 0.01 * np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        #calculate output values from inputs * weights + biases
        self.output = np.dot(inputs, self.weights) + self.biases

X, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2,3)
dense1.forward(X)

print(dense1.output[:5])
print(dense1.output.shape)
