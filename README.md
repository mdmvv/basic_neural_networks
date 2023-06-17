# Basic Neural Networks

Implementation of several basic neural networks using various algorithms and activation functions.

`simple_sigmoid_network.py`
A simple neural network with a sigmoid activation function.
It takes input and weights and calculates and outputs an output value.

`backpropagation.py`
A simple neural network with backpropagation for training on the example of binary classification.
It uses a sigmoid activation function, random initialization of weights, training iterations with weight correction
based on the calculated error.

`rosenblatt.py`
Implementation of the Rosenblatt algorithm and backpropagation for training for binary classification.
The code defines activation functions (sigmoid), functions for calculating the error and their derivatives. After the
initialization of the input data, output data and weights, training and verification of the classification accuracy on
the test data takes place.

`radial_basis_function_network.py`
Radial basis function neural network (RBFN) for function interpolation.
The code defines the activation functions (Gaussian function), the RBFN class and its methods for training and
prediction. After RBFN is initialized and trained on the training data, the original function, training data, and
interpolated function are plotted, as well as the training error plot.

`som_network.py`
Neural network of self-organizing Kohonen maps (SOM).
It is trained on a set of images and can recognize samples with different pixel changes, outputting the percentage of
correctly recognized samples for each case of pixel changes.

`hopfield_network.py`
Hopfield neural network for reconstructing patterns.
The accuracy of reconstructing is checked when the number of pixels changes, and the results of the analysis are
displayed in the form of a table with percentages of correct restoration.
