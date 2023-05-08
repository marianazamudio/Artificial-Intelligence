# --------------------------------------------------- #
# File: main.py
# Author: Mariana Zamudio Ayala
# Date: 07/05/2023
# Description: Program that create a customizable 
# neural network and computes its output. 
# --------------------------------------------------- #
from interconnected_neuronal_network import InterconNeuralNet

# Print welcome message and a description of the program
print("\n\n-------------------------Welcome------------------------------")
print("This programs allows you to create a customizable neural network")
print("in terms of layers and number of neurons, weights are assigned")
print("randomly\n")

# Collect customizable data of the neural network
num_layers = int(input("Enter the number of layers in the neural network: "))
neurons_in_layers = [0 for num_layers in range(3)]
for i in range(num_layers):
    neurons_in_layers[i]= int(input(f"Enter the number of neurons in layer {i}: "))
    
#TODO
print("Select the activation function of the neural network:")
print("1. Senoide")
print("2. Sesgo")
print("3. Tanh")
act_funct = int(input())

# Initialize neural network
neural_net = InterconNeuralNet(num_layers, neurons_in_layers, act_funct)
print(neural_net.random_weights)
print(neural_net.output)

# Compute the output of the neural network
output = neural_net.compute_output()
print(f"The output of the neural network is: {output}")