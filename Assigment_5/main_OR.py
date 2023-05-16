# --------------------------------------------------- #
# File: main_OR.py
# Author: Mariana Zamudio Ayala
# Date: 15/05/2023
# Description: Program that trains a sigle layer
# perceptron to do AND operation
# --------------------------------------------------- #
from interconnected_neuronal_network import InterconNeuralNet
import numpy as np
import matplotlib.pyplot as plt

# Initialize the single layer perceptron
inputs = [0,0]
num_layers = 1
num_neurons = [1]
perceptron = InterconNeuralNet(inputs, num_layers, num_neurons, 3)

# Select n
eta = 0.5

# Inicialize pairs input-output
r_false = np.array([-1])
r_true = np.array([1])
pairs_io = [[[0,0], r_false], [[0,1],r_true], [[1,0], r_true], [[1,1], r_true]]

# Training
print("***********************")
w = perceptron.train_perceptron(eta, pairs_io)[0]
print(w)
print("*******************")

# Test the perceptron and print table
print("----------------------------")
print("Entradas   Exp.Res   Obt.Res")
print("----------------------------")
for pair in pairs_io:
    inputs = pair[0]
    perceptron.set_inputs(pair[0])
    response_obtained = perceptron.compute_output()
    expected_response = pair[1]
    print(inputs, "\t  ",expected_response, "   ",response_obtained)
    

# Vizualization

# Plot points
points_x1 = [pair_io[0][0] for pair_io in pairs_io]
points_x2= [pair_io[0][1] for pair_io in pairs_io]
plt.scatter(points_x1, points_x2, color="red", label = "points")

# Plot class division
x1 = np.arange(-0.25,1.2,0.05)
x2 = -w[1]/w[2] * x1 - w[0]/w[2]
plt.plot(x1,x2, color = "blue", label = "div")

plt.title("Grafico de Clases Operación OR")
plt.legend()
plt.show()





