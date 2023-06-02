# --------------------------------------------------- #
# File: main_XOR.py
# Author: Mariana Zamudio Ayala
# Date: 01/06/2023
# Description: Program that trains a sigle layer
# perceptron to do XOR operation
# --------------------------------------------------- #
from interconnected_neuronal_network import InterconNeuralNet
import numpy as np
import random

max_epochs = 500
alpha = 0.01
a = 10
eta = 0.1

# Initialize the multi layer perceptron
inputs = [0,0]
num_layers = 2
num_neurons = [2,1]
per_mult_layer = InterconNeuralNet(inputs, num_layers, num_neurons, 2, a)
print(per_mult_layer.weights)

# Initialize data set
data_set = [[0,0], [0,1], [1,0], [1,1]]
d =[1,0,0,1]

for i in range(max_epochs):
    # Intialize list with idx for data set
    idx_list = list((range(4)))
    print(idx_list, "idx_list")
    # Permutate list
    random.shuffle(idx_list)
    print(idx_list, "idx_list")

    # Iterate in data set
    o = []
    for idx in idx_list:
        # Configurar entradas
        per_mult_layer.set_inputs(data_set[idx])
        # Configurar valores deseados
        d_n = d[idx] 

        # Forward computation
        o_n = per_mult_layer.compute_output()[-1]
        
        # Backward computation
        per_mult_layer.back_computation(eta=eta,alpha=alpha,d=d_n)

        o.append(o_n)
    
    # Compute MSE
    print(d)
    print(o)
    MSE = (np.array(d) - np.array(o))
    MSE = np.square(MSE)
    MSE = np.sum(MSE)/(len(d))
    print(MSE)
    #input()

    # break condition 
    if MSE == 0:
        break

print(f"termine en {i} epocas")

# TEST 
print("AND OPERATION")
print("----------------------------")
print("Entradas   Exp.Res   Obt.Res")
print("----------------------------")
for (x_n,d_n) in zip(data_set, d):
    per_mult_layer.set_inputs(x_n)
    o_n = per_mult_layer.compute_output()[-1]
    print(x_n, "\t  ",d_n, "   ",o_n)