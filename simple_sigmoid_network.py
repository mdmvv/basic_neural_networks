"""
A simple neural network with a sigmoid activation function.
It takes input and weights and calculates and outputs an output value.
"""


import numpy as np


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


inputs = np.array([0, 1, 1])
weights = np.array([2, 3, 4])
outputs = sigmoid(np.dot(inputs, weights))


print(outputs)
