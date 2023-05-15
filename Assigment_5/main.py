# --------------------------------------------------- #
# File: main.py
# Author: Mariana Zamudio Ayala
# Date: 15/05/2023
# Description: Program that trains a sigle layer
# perceptron para que realice la operación AND
# --------------------------------------------------- #
from interconnected_neuronal_network import InterconNeuralNet
# Initialize the single layer perceptron
inputs = [0,0]
num_layers = 1
num_neurons = [1]
perceptron = InterconNeuralNet(inputs, num_layers, num_neurons, 3)

# Select n
n = 0.1

# Inicialize pair input-output
pairs_io = [[[0,0], 0], [[0,1], 0], [[1,0], 0], [[1,1], 1]]

# Training
#print(perceptron.train(n, pairs_io))

# Comprueba las salidas
for pair in pairs_io:
    print("Entradas:", pair[0])
    perceptron.set_inputs(pair[0])
    print("Salida Obtenida:", perceptron.compute_output())
    print("Salida Esperada:", pair[1])


# Vizualization