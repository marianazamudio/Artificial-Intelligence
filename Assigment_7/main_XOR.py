# --------------------------------------------------- #
# File: main_XOR.py
# Author: Mariana Zamudio Ayala
# Date: 01/06/2023
# Description: Program that trains a sigle layer
# perceptron to do XOR operation
# --------------------------------------------------- #
from interconnected_neuronal_network import InterconNeuralNet
import numpy as np

epochs = 50
alpha = 0.9
a = 1

# Initialize the single layer perceptronip
inputs = [0,0]
num_layers = 2
num_neurons = [2,1]
per_mult_layer = InterconNeuralNet(inputs, num_layers, num_neurons, 2, a)
print(per_mult_layer.weights)

# Initialize data set
data_set = [(0,0), (0,1), (1,0), (1,1)]
d =[1,0,0,1]

#for i in range(epochs):
d = 1
# Forward computation
y = per_mult_layer.compute_output()
# Obtain the output of output layer
o = y[:num_neurons[-1]]
# Compute error
e = d - o
# Backward computation
per_mult_layer.back_computation(d=1)