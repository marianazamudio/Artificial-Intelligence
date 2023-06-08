# --------------------------------------------------- #
# File: main_XOR.py
# Author: Mariana Zamudio Ayala
# Date: 01/06/2023
# Description: Program that trains a multi layer
# perceptron to do XOR operation
# --------------------------------------------------- #
from interconnected_neuronal_network import InterconNeuralNet
import numpy as np
import random
import matplotlib.pyplot as plt

# Tunning parameters
max_epochs = 1000
alpha = 0.03
a = 1.2
b=1
eta = 0.2

# Initialize the multi layer perceptron
inputs = [0,0]
num_layers = 2
num_neurons = [2,1]
per_mult_layer = InterconNeuralNet(inputs, num_layers, num_neurons, 4, a, b, wt="r")
print(per_mult_layer.weights)

# Weight initialization for convergence
#capa_1 = np.array([[0.0, 0.0, 0.0],[0.0, 0.0,0.0]])
#capa_2 = np.array([[0.0, 0.0, 1.0]])
#per_mult_layer.weights = [capa_1, capa_2]

#capa_1 = np.array([[-1.5, 1, 1],[-0.5, 1, 1]])
#capa_2 = np.array([[-0.5, -2, 1]])
#per_mult_layer.weights = [capa_1, capa_2]

# Initialize data set
data_set = [[0,0], [0,1], [1,0], [1,1]]
d =[-1,1,1,-1]

# List to plot MSE
MSE_list = []
# Iterate between epochs of training
for i in range(max_epochs):
    # Intialize list with idx for data set
    idx_list = list((range(4)))
    
    # Permutate list
    #random.shuffle(idx_list)

    # Initialize MSE result variable
    MSE = 0

    # Iterate in data set
    for idx in idx_list:
        # Configurar entradas
        per_mult_layer.set_inputs(data_set[idx])
        print(data_set[idx], "inputs ----")
        # Configurar valores deseados
        d_n = d[idx] 
        
        # Forward computation
        o_n = per_mult_layer.compute_output()[-1]
        
        # Backward computation
        per_mult_layer.back_computation(eta=eta,alpha=alpha,d=d_n)
        
        MSE += (d_n - o_n)**2
    
    # Compute MSE
    MSE = MSE/(len(d))
    print(MSE, "mse")
    MSE_list.append(MSE)
    
    #print(o, "o")
    #print(d,"d")
    #input()

    # Break condition 
    if MSE < 0.005:
        break
print(f"mse: {MSE}")
print(f"termino en {i} epocas")

# TEST last weights
print("XOR OPERATION ---- last data")
print("----------------------------")
print("Entradas   Exp.Res   Obt.Res")
print("----------------------------")
for (x_n,d_n) in zip(data_set, d):
    per_mult_layer.set_inputs(x_n)
    o_n = per_mult_layer.compute_output()[-1]
    print(x_n, "\t  ",d_n, "   ",o_n)
# Print weights
print(per_mult_layer.weights)

# Graficar MSE
x = np.arange(len(MSE_list))
plt.plot(x, MSE_list)
#print(MSE_list)
# Show plot
plt.show()