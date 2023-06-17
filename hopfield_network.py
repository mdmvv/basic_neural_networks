"""
Hopfield neural network for reconstructing patterns.
The accuracy of reconstructing is checked when the number of pixels changes, and the results of the analysis are
displayed in the form of a table with percentages of correct restoration.
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd


nb_patterns = 4
pattern_width = 5
pattern_height = 5
max_iterations = 10


I0 = np.array([[ 1,  1,  1,  1,  1,
				-1, -1, -1,  1, -1,
				-1, -1,  1, -1, -1,
				-1,  1, -1, -1, -1,
				 1,  1,  1,  1,  1], # Z
			   [ 1,  1,  1,  1,  1,
				 1, -1, -1, -1,  1,
				 1,  1,  1,  1,  1,
				 1, -1, -1, -1,  1,
				 1, -1, -1, -1,  1], # A
			   [ 1,  1,  1,  1,  1,
				 1, -1, -1, -1,  1,
				 1,  1,  1,  1,  1,
				 1, -1, -1, -1,  1,
				 1, -1, -1, -1,  1], # A
			   [ 1, -1, -1, -1,  1,
				 1,  1, -1, -1,  1,
				 1, -1,  1, -1,  1,
				 1, -1, -1,  1,  1,
				 1, -1, -1, -1,  1], # N
			   [ 1,  1,  1,  1, -1,
				 1, -1, -1, -1,  1,
				 1,  1,  1,  1, -1,
				 1, -1, -1, -1,  1,
				 1,  1,  1,  1, -1], # B
			   [ 1, -1, -1, -1,  1,
				-1,  1, -1,  1, -1,
				-1, -1,  1, -1, -1,
				-1, -1,  1, -1, -1,
				-1, -1,  1, -1, -1], # Y
			   [-1, -1, -1, -1, -1,
				-1, -1, -1, -1, -1,
				-1, -1, -1, -1, -1,
				-1, -1, -1, -1, -1,
				 1,  1,  1,  1,  1], # _
			   [ 1, -1, -1, -1,  1,
				 1,  1, -1, -1,  1,
				 1, -1,  1, -1,  1,
				 1, -1, -1,  1,  1,
				 1, -1, -1, -1,  1], # N
			   [ 1,  1,  1,  1,  1,
				 1, -1, -1, -1, -1,
				 1,  1,  1,  1,  1,
				 1, -1, -1, -1, -1,
				 1,  1,  1,  1,  1], # E
			   [ 1,  1,  1,  1,  1,
				-1, -1,  1, -1, -1,
				-1, -1,  1, -1, -1,
				-1, -1,  1, -1, -1,
				-1, -1,  1, -1, -1], # T
			   [ 1, -1,  1, -1,  1,
				 1, -1,  1, -1,  1,
				 1, -1,  1, -1,  1,
				 1, -1,  1, -1,  1,
				 1,  1,  1,  1,  1], # W
			   [ 1,  1,  1,  1,  1,
				 1, -1, -1, -1,  1,
				 1, -1, -1, -1,  1,
				 1, -1, -1, -1,  1,
				 1,  1,  1,  1,  1], # O
			   [ 1,  1,  1,  1, -1,
				 1, -1, -1, -1,  1,
				 1,  1,  1,  1, -1,
				 1, -1, -1, -1,  1,
				 1, -1, -1, -1,  1], # R
			   [ 1, -1, -1, -1,  1,
				 1, -1, -1,  1, -1,
				 1,  1,  1, -1, -1,
				 1, -1, -1,  1, -1,
				 1, -1, -1, -1,  1]  # K
				])

I2 = np.array([[ 1,  1,  1,  1,  1,
				-1, -1, -1,  1, -1,
				-1, -1, -1, -1, -1,
				-1,  1, -1, -1, -1,
				-1,  1,  1,  1,  1], # Z
			   [ 1,  1,  1, -1,  1,
				 1, -1, -1, -1,  1,
				 1,  1,  1,  1,  1,
				 1,  1, -1, -1,  1,
				 1, -1, -1, -1,  1], # A
			   [ 1,  1,  1,  1,  1,
				 1, -1,  1, -1,  1,
				 1,  1,  1,  1,  1,
				 1,  1, -1, -1,  1,
				 1, -1, -1, -1,  1], # A
			   [ 1, -1, -1, -1,  1,
				 1,  1, -1, -1,  1,
				 1, -1,  1, -1,  1,
				 1, -1, -1,  1, -1,
				 1, -1, -1, -1, -1], # N
			   [ 1,  1,  1, -1, -1,
				 1, -1, -1, -1,  1,
				 1,  1,  1,  1, -1,
				 1, -1,  1, -1,  1,
				 1,  1,  1,  1, -1], # B
			   [ 1, -1, -1, -1,  1,
				-1,  1, -1,  1, -1,
				-1, -1,  1, -1, -1,
				-1,  1,  1, -1, -1,
				-1,  1,  1, -1, -1], # Y
			   [-1, -1, -1, -1, -1,
				-1, -1, -1, -1, -1,
				-1, -1, -1,  1, -1,
				-1, -1, -1, -1, -1,
				 1, -1,  1,  1,  1], # _
			   [ 1, -1, -1, -1,  1,
				-1,  1, -1, -1,  1,
				 1, -1,  1,  1,  1,
				 1, -1, -1,  1,  1,
				 1, -1, -1, -1,  1], # N
			   [ 1, -1,  1,  1, -1,
				 1, -1, -1, -1, -1,
				 1,  1,  1,  1,  1,
				 1, -1, -1, -1, -1,
				 1,  1,  1,  1,  1], # E
			   [ 1,  1,  1,  1,  1,
				-1,  1,  1, -1, -1,
				 1, -1,  1, -1, -1,
				-1, -1,  1, -1, -1,
				-1, -1,  1, -1, -1], # T
			   [ 1, -1,  1,  1,  1,
				 1, -1,  1, -1, -1,
				 1, -1,  1, -1,  1,
				 1, -1,  1, -1,  1,
				 1,  1,  1,  1,  1], # W
			   [ 1,  1,  1,  1,  1,
				 1, -1, -1, -1, -1,
				 1, -1, -1, -1,  1,
				-1, -1, -1, -1,  1,
				 1,  1,  1,  1,  1], # O
			   [ 1, -1,  1,  1,  1,
				 1, -1, -1, -1,  1,
				 1,  1,  1,  1, -1,
				 1, -1, -1, -1,  1,
				 1, -1, -1, -1,  1], # R
			   [ 1, -1, -1, -1,  1,
				 1, -1, -1,  1, -1,
				 1,  1,  1, -1, -1,
				-1, -1,  1,  1, -1,
				 1, -1, -1, -1,  1]  # K
				])

I4 = np.array([[-1,  1,  1,  1,  1,
				-1, -1, -1,  1, -1,
				-1, -1, -1, -1, -1,
				-1,  1, -1, -1, -1,
				-1,  1,  1,  1, -1], # Z
			   [ 1,  1,  1, -1,  1,
				 1, -1, -1, -1, -1,
				 1,  1,  1,  1,  1,
				 1,  1, -1, -1, -1,
				 1, -1, -1, -1,  1], # A
			   [ 1,  1,  1,  1,  1,
				 1, -1,  1, -1, -1,
				-1,  1,  1,  1,  1,
				 1,  1, -1, -1,  1,
				 1, -1, -1, -1,  1], # A
			   [ 1, -1, -1, -1,  1,
				 1,  1, -1, -1,  1,
				-1, -1, -1, -1,  1,
				 1, -1, -1,  1, -1,
				 1, -1, -1, -1, -1], # N
			   [ 1,  1,  1, -1, -1,
				 1, -1, -1, -1,  1,
				 1, -1, -1,  1, -1,
				 1, -1,  1, -1,  1,
				 1,  1,  1,  1, -1], # B
			   [ 1, -1, -1, -1,  1,
				-1,  1, -1,  1, -1,
				-1, -1,  1, -1, -1,
				-1,  1,  1,  1, -1,
				-1,  1,  1,  1, -1], # Y
			   [-1, -1, -1, -1, -1,
				-1, -1, -1, -1, -1,
				 1,  1, -1,  1, -1,
				-1, -1, -1, -1, -1,
				 1, -1,  1,  1,  1], # _
			   [ 1, -1, -1, -1,  1,
				-1,  1, -1, -1,  1,
				 1,  1,  1,  1,  1,
				 1, -1, -1,  1,  1,
				 1, -1,  1, -1,  1], # N
			   [-1, -1,  1,  1, -1,
				 1, -1, -1, -1, -1,
				-1,  1,  1,  1,  1,
				 1, -1, -1, -1, -1,
				 1,  1,  1,  1,  1], # E
			   [ 1,  1,  1,  1,  1,
				-1,  1,  1, -1, -1,
				 1, -1,  1, -1, -1,
				-1,  1,  1,  1, -1,
				-1, -1,  1, -1, -1], # T
			   [ 1, -1,  1,  1,  1,
				 1, -1,  1, -1, -1,
				 1, -1,  1, -1,  1,
				 1,  1,  1, -1,  1,
				 1,  1, -1,  1,  1], # W
			   [ 1,  1,  1,  1,  1,
				 1, -1, -1, -1, -1,
				 1, -1, -1, -1,  1,
				-1, -1, -1,  1,  1,
				 1, -1,  1,  1,  1], # O
			   [ 1, -1,  1,  1,  1,
				-1, -1, -1, -1,  1,
				 1,  1,  1,  1, -1,
				 1, -1, -1, -1,  1,
				-1, -1, -1, -1,  1], # R
			   [ 1, -1, -1, -1,  1,
				 1, -1,  1,  1, -1,
				 1,  1,  1, -1, -1,
				-1, -1,  1,  1, -1,
				 1, -1,  1, -1,  1]  # K
				])

I6 = np.array([[-1,  1,  1,  1,  1,
				-1,  1, -1,  1, -1,
				-1, -1, -1,  1, -1,
				-1,  1, -1, -1, -1,
				-1,  1,  1,  1, -1], # Z
			   [ 1,  1,  1, -1,  1,
				 1, -1, -1, -1, -1,
				 1,  1,  1, -1,  1,
				 1,  1, -1, -1, -1,
				 1, -1, -1, -1, -1], # A
			   [ 1,  1,  1,  1,  1,
				 1, -1,  1, -1, -1,
				-1,  1,  1,  1, -1,
				-1,  1, -1, -1,  1,
				 1, -1, -1, -1,  1], # A
			   [ 1, -1, -1, -1,  1,
				 1,  1, -1, -1,  1,
				-1, -1, -1, -1,  1,
				 1, -1,  1,  1, -1,
				-1, -1, -1, -1, -1], # N
			   [ 1,  1,  1, -1, -1,
				 1, -1, -1,  1,  1,
				 1, -1, -1,  1, -1,
				 1,  1,  1, -1,  1,
				 1,  1,  1,  1, -1], # B
			   [ 1, -1, -1, -1,  1,
				-1,  1, -1,  1,  1,
				-1, -1,  1, -1,  1,
				-1,  1,  1,  1, -1,
				-1,  1,  1,  1, -1], # Y
			   [-1, -1, -1, -1, -1,
				-1, -1, -1, -1, -1,
				 1,  1, -1,  1, -1,
				-1,  1, -1,  1, -1,
				 1, -1,  1,  1,  1], # _
			   [ 1, -1, -1, -1,  1,
				-1, -1, -1, -1,  1,
				 1,  1,  1,  1,  1,
				 1, -1, -1,  1, -1,
				 1, -1,  1, -1,  1], # N
			   [-1, -1,  1,  1, -1,
				 1, -1, -1, -1, -1,
				-1,  1,  1,  1, -1,
				 1, -1, -1, -1, -1,
				 1,  1,  1, -1,  1], # E
			   [ 1,  1,  1,  1, -1,
				-1,  1,  1, -1, -1,
				 1, -1,  1, -1, -1,
				-1,  1,  1,  1,  1,
				-1, -1,  1, -1, -1], # T
			   [-1, -1,  1,  1,  1,
				 1, -1,  1, -1, -1,
				 1, -1,  1, -1,  1,
				-1,  1,  1, -1,  1,
				 1,  1, -1,  1,  1], # W
			   [ 1,  1,  1,  1,  1,
				 1, -1, -1,  1, -1,
				 1, -1,  1, -1,  1,
				-1, -1, -1,  1,  1,
				 1, -1,  1,  1,  1], # O
			   [ 1, -1,  1,  1,  1,
				-1, -1, -1, -1, -1,
				 1,  1,  1,  1, -1,
				 1, -1, -1, -1,  1,
				-1, -1, -1,  1,  1], # R
			   [ 1, -1, -1, -1,  1,
				 1, -1,  1, -1, -1,
				 1,  1,  1, -1, -1,
				-1, -1,  1,  1, -1,
				 1, -1,  1,  1,  1]  # K
				])


f0, ax = plt.subplots(1, len(I0), figsize=(14, 1))
for i in range(len(I0)):
	sb.heatmap(np.reshape(I0[i], (5, 5)), cmap=sb.light_palette("red"), ax=ax[i], cbar=False, yticklabels=False, xticklabels=False)

f2, ax = plt.subplots(1, len(I2), figsize=(14, 1))
for i in range(len(I2)):
	sb.heatmap(np.reshape(I2[i], (5, 5)), cmap=sb.light_palette("green"), ax=ax[i], cbar=False, yticklabels=False, xticklabels=False)

f4, ax = plt.subplots(1, len(I4), figsize=(14, 1))
for i in range(len(I4)):
	sb.heatmap(np.reshape(I4[i], (5, 5)), cmap=sb.light_palette("blue"), ax=ax[i], cbar=False, yticklabels=False, xticklabels=False)

f6, ax = plt.subplots(1, len(I6), figsize=(14, 1))
for i in range(len(I6)):
	sb.heatmap(np.reshape(I6[i], (5, 5)), cmap=sb.light_palette("purple"), ax=ax[i], cbar=False, yticklabels=False, xticklabels=False)

plt.show()


W = np.zeros((pattern_width * pattern_height, pattern_width * pattern_height))
for i in range(pattern_width * pattern_height):
	for j in range(pattern_width * pattern_height):
		if i == j or W[i, j] != 0:
			continue
		w = 0
		for n in range(nb_patterns):
			w += I0[n, i] * I0[n, j]
		W[i, j] = w / I0.shape[0]
		W[j, i] = W[i, j]

er2 = []
er4 = []
er6 = []


for n in range(len(I0)):
	A = I2[n]
	for _ in range(max_iterations):
		for i in range(pattern_width * pattern_height):
			A[i] = 1 if np.dot(W[i], A) > 0 else -1
	truth_list = (A == I0[n])
	if (False in truth_list):
		er2.append(1)
	else:
		er2.append(0)

for n in range(len(I0)):
	A = I4[n]
	for _ in range(max_iterations):
		for i in range(pattern_width * pattern_height):
			A[i] = 1 if np.dot(W[i], A) > 0 else -1
	truth_list = (A == I0[n])
	if (False in truth_list):
		er4.append(1)
	else:
		er4.append(0)

for n in range(len(I0)):
	A = I6[n]
	for _ in range(max_iterations):
		for i in range(pattern_width * pattern_height):
			A[i] = 1 if np.dot(W[i], A) > 0 else -1
	truth_list = (A == I0[n])
	if (False in truth_list):
		er6.append(1)
	else:
		er6.append(0)


word = ["Z", "A", "A", "N", "B", "Y", "_", "N", "E", "T", "W", "O", "R", "K"]
df = pd.DataFrame(
	data=list(zip(er2, er4, er6)),
	index=word,
	columns=["2 changed pixels", "4 changed pixels", "6 changed pixels"]
)

for col in df.columns:
	df[col] = df[col].map({1: True, 0: False})


print(df)
print()
print("2 changed pixels")
print("correct percent: %.2f" % (er2.count(1) / 14 * 100))
print()
print("4 changed pixels")
print("correct percent: %.2f" % (er4.count(1) / 14 * 100))
print()
print("6 changed pixels")
print("correct percent: %.2f" % (er6.count(1) / 14 * 100))
