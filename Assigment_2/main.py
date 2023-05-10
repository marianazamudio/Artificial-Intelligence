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
print("This program is sponsored by Coca-Cola")

# ---Collect customizable data of the neural network
num_layers = int(input("Enter the number of layers in the neural network: "))
neurons_in_layers = [0 for num_layers in range(num_layers)]
for i in range(num_layers):
    neurons_in_layers[i]= int(input(f"Enter the number of neurons in layer {i+1}: "))
    
# Menu to select the activation function
print("\nSelect the activation function of the neural network:")
print("1. Threshold  (Sesgo)")
print("2. Sigmoid    (Signoide)")
print("3. Signum     (Signo)")
print("4. Hyperbolic tangent ")
act_funct = int(input("\nChoice: "))
print("\n")
# ---

# Initialize neural network
neural_net = InterconNeuralNet(num_layers, neurons_in_layers, act_funct)

# Compute the output of the neural network
output = neural_net.compute_output()

# Print weight matrix for each layer
for i in range(num_layers):
    print(f"Weight matrix for inputs in layer {i+1}")
    print(neural_net.random_weights[i])
    print("\n")

print(f"The output in each neuron is: {neural_net.outputs}")
print(f"THE OUTPUT OF THE NEURAL NETWORK IS: {output}")