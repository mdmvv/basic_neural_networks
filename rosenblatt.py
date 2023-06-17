"""
Implementation of the Rosenblatt algorithm and backpropagation for training for binary classification.
The code defines activation functions (sigmoid), functions for calculating the error and their derivatives. After the
initialization of the input data, output data and weights, training and verification of the classification accuracy on
the test data takes place.
"""


import numpy as np


training_inputs = np.array(
	[
		[0.386, 0.726], [0.480, 0.719], [0.911, 0.739], [0.519, 0.329], [0.489, 0.501],
		[0.096, 0.485], [0.463, 0.479], [0.303, 0.403], [0.677, 0.643], [0.735, 0.805],
		[0.475, 0.462], [0.486, 0.907], [0.701, 0.903], [0.148, 0.429], [0.698, 0.436],
		[0.816, 0.590], [0.311, 0.030], [0.084, 0.109], [0.883, 0.153], [0.046, 0.203],
		[0.257, 0.773], [0.294, 0.593], [0.778, 0.361], [0.923, 0.496], [0.289, 0.885],
		[0.527, 0.632], [0.134, 0.812], [0.200, 0.219], [0.157, 0.113], [0.534, 0.975],
		[0.701, 0.255], [0.624, 0.121], [0.611, 0.208], [0.064, 0.374], [0.229, 0.441],
		[0.947, 0.953], [0.401, 0.921], [0.867, 0.576], [0.468, 0.118], [0.919, 0.464],
		[0.276, 0.001], [0.694, 0.334], [0.168, 0.038], [0.288, 0.405], [0.988, 0.198],
		[0.295, 0.396], [0.079, 0.950], [0.807, 0.526], [0.286, 0.262], [0.418, 0.513],
		[0.951, 0.604], [0.223, 0.577], [0.919, 0.943], [0.816, 0.435], [0.534, 0.750],
		[0.289, 0.307], [0.639, 0.992], [0.236, 0.911], [0.578, 0.138], [0.752, 0.467],
		[0.194, 0.158], [0.430, 0.692], [0.832, 0.519], [0.364, 0.806], [0.335, 0.191],
		[0.148, 0.528], [0.048, 0.642], [0.875, 0.076], [0.813, 0.841], [0.139, 0.168],
		[0.974, 0.257], [0.034, 0.025], [0.376, 0.858], [0.153, 0.363], [0.362, 0.114],
		[0.591, 0.403], [0.775, 0.993], [0.941, 0.863], [0.960, 0.963], [0.297, 0.962]
	]
)

test_inputs = np.array(
	[
		[0.760, 0.994], [0.252, 0.417], [0.519, 0.555], [0.901, 0.450], [0.081, 0.613],
		[0.837, 0.777], [0.790, 0.639], [0.154, 0.617], [0.488, 0.599], [0.445, 0.690],
		[0.521, 0.965], [0.805, 0.215], [0.731, 0.703], [0.827, 0.909], [0.463, 0.283],
		[0.605, 0.992], [0.560, 0.673], [0.732, 0.182], [0.366, 0.343], [0.303, 0.699]
	]
)

training_outputs = np.array(
	[
		0, 0, 1, 0, 0,
		0, 0, 0, 1, 1,
		0, 0, 1, 0, 1,
		1, 0, 0, 1, 0,
		0, 0, 1, 1, 0,
		0, 0, 0, 0, 1,
		1, 1, 0, 0, 0,
		1, 0, 1, 0, 1,
		0, 1, 0, 0, 1,
		0, 0, 1, 0, 0,
		1, 0, 1, 1, 0,
		0, 1, 0, 0, 1,
		0, 0, 1, 0, 0,
		0, 0, 1, 1, 0,
		1, 0, 0, 0, 0,
		1, 1, 1, 1, 0
	]
)

test_outputs = np.array(
	[
		1, 0, 0, 1, 0,
		1, 1, 0, 0, 0,
		1, 1, 1, 1, 0,
		1, 1, 1, 0, 0
	]
)


def sign(x):
	if x > 0:
		return 1
	else:
		return 0


def f(x):
	return 1 / (1 + np.exp(-x))


def df(x):
	return np.exp(-x) / ((1 + np.exp(-x))**2)


def rosenblatt(num, offset, inputs, outputs, weights, lscoeff):
	iterations = 0
	while True:
		err_sum = 0
		for i in range(num):
			output = sign(weights[0] * offset + weights[1] * inputs[i][0] + weights[2] * inputs[i][1])
			err = outputs[i] - output
			err_sum += abs(err)
			weights[0] = weights[0] + lscoeff * offset * err
			weights[1] = weights[1] + lscoeff * inputs[i][0] * err
			weights[2] = weights[2] + lscoeff * inputs[i][1] * err
		if err_sum == 0:
			print("iterations: %d" % iterations)
			return weights
		iterations += 1


def backpropagation(num, offset, inputs, outputs, weights, lscoeff):
	iterations = 0
	for k in range(2000):
		for i in range(num):
			z = weights[0] * offset + weights[1] * inputs[i][0] + weights[2] * inputs[i][1]
			err = f(z) - outputs[i]
			delta = err * df(z)
			weights[0] = weights[0] - lscoeff * delta * offset
			weights[1] = weights[1] - lscoeff * delta * inputs[i][0]
			weights[2] = weights[2] - lscoeff * delta * inputs[i][1]
		iterations += 1
	print("iterations: %d" % iterations)
	return weights


def predict(num, inputs, outputs, weights):
	correct_percent = 0
	for i in range(num):
		predicted_output = sign(weights[0] * offset + weights[1] * inputs[i][0] + weights[2] * inputs[i][1])
		if predicted_output == outputs[i]:
			correct_percent += 100 / num
	return correct_percent


print("Rosenblatt")
print("training:")
offset = 1
lscoeff = 0.1
training_num = 80
training_weights = 2 * np.random.random((3, 1)) - 1
weights_rosenblatt = rosenblatt(training_num, offset, training_inputs, training_outputs, training_weights, lscoeff)
print("received weights:")
print(weights_rosenblatt)
print("test:")
test_num = 20
print("correct percent: %d" % predict(test_num, test_inputs, test_outputs, weights_rosenblatt))
print("")


print("Backpropagation")
print("training:")
offset = 1
lscoeff = 1
training_num = 80
training_weights = 2 * np.random.random((3, 1)) - 1
weights_backpropagation = backpropagation(training_num, offset, training_inputs, training_outputs, training_weights, lscoeff)
print("received weights:")
print(weights_backpropagation)
print("test:")
test_num = 20
print("correct percent: %d" % predict(test_num, test_inputs, test_outputs, weights_backpropagation))
