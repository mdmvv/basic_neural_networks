"""
Radial basis function neural network (RBFN) for function interpolation.
The code defines the activation functions (Gaussian function), the RBFN class and its methods for training and
prediction. After RBFN is initialized and trained on the training data, the original function, training data, and
interpolated function are plotted, as well as the training error plot.
"""


import numpy as np
import matplotlib.pyplot as plt


def f(x):
	return np.sin(x) ** 2


a, b = 0, 10


def gauss(X, C, sigma):
	return np.exp(-(X - C) ** 2 / (2 * sigma ** 2))


class RBFN:
	def __init__(self, rbf, hidden_num, C, sigma):
		self.rbf = rbf
		self.hidden_num = hidden_num
		self.C = C
		self.sigma = sigma
		self.W = np.random.random(self.hidden_num)
		self.error = [0.]

	def forward(self, X):
		H = np.zeros(self.hidden_num)
		for i in range(self.hidden_num):
			H[i] = self.rbf(X, self.C[i], self.sigma)
		Y = self.W @ H
		return Y, H

	def predict(self, X):
		return self.forward(X)[0]

	def train(self, X, Y_out, lscoeff, epochs, eps):
		print("RBFN: training")
		for e in range(epochs):
			for i in range(self.hidden_num):
				Y, H = self.forward(X[i])
				for j in range(self.hidden_num):
					self.W[j] -= lscoeff * (Y - Y_out[i]) * H[j]
				self.error[e] += abs(Y - Y_out[i])
			if len(self.error) > 1 and abs(self.error[-1] - self.error[-2]) < eps:
				break
			self.error.append(0.)


hidden_num = 150
training_num = 150
X = np.sort(np.random.uniform(a, b, training_num))
Y = f(X)
C = X[np.random.randint(training_num, size=hidden_num)]
sigma = (np.max(X) - np.min(X)) / np.sqrt(2 * training_num)

rbfn = RBFN(gauss, hidden_num, C, sigma)
rbfn.train(X, Y, 0.03, 100, 0.00001)

plt.plot(np.linspace(a, b, 500), f(np.linspace(a, b, 500)), label="function")
plt.scatter(X, Y, label="training data")
plt.plot(np.linspace(a, b, 500), [rbfn.predict(np.array(x)) for x in np.linspace(a, b, 500)], label="interpolated")
plt.legend()
plt.show()

plt.plot(range(0, len(rbfn.error)), rbfn.error)
plt.grid()
plt.show()
