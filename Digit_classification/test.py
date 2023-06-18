# --------------------------------------------------- #
# File: main.py
# Author: Mariana Zamudio Ayala
# Date: 14/06/2023
# Description:
# Program that trains a multi layer
# perceptron to classify digits from 0-9. 
# --------------------------------------------------- #
from interconnected_neuronal_network import InterconNeuralNet
import dataset

# Tunning parameters
alpha = 0.01
a = 1
b = 1
eta = 0.25
num_epochs = 100
num_train_data = 6000
num_test_data = 1000

# Neural network's parameters
inputs = [0 for x in range(784)]
num_layers = 3
num_neurons = [400, 25, 10]

# Initialize neural network
perc_mult_layer = InterconNeuralNet(inputs, num_layers, num_neurons, 4, a, b, wt="r")
initial_weights = perc_mult_layer.weights
MSE_values = [1, 2, 3, 4, 5, 6]
final_weights = perc_mult_layer.weights
error_per_tr = 0.001
error_per_te = 0.001


def save_results(alpha, a, b, eta, num_epochs, num_train_data, num_test_data, num_layers, num_neurons,\
                 initial_weights, final_weights, MSE_values, error_per_tr, error_per_te):
    archivo = open("results.txt", "a")
    
    archivo.write(f"---------------------------------------------------------\n")
    archivo.write(f"alpha = {alpha}, eta = {eta}\n")
    archivo.write(f"a = {a}, b = {b}\n")
    archivo.write(f"#_layers = {num_layers} -> {num_neurons}\n")
    archivo.write(f"num_epochs = {num_epochs}\n")
    archivo.write(f"#_tr_data = {num_train_data}\n")
    archivo.write(f"#_te_data = {num_test_data}\n")
    archivo.write(f"initial_weights = {initial_weights}\n")
    archivo.write(f"final_weights = {final_weights}\n")
    archivo.write(f"MSE_values = {MSE_values}\n")
    archivo.write(f"ERROR TRAINNING  = {error_per_tr}\n")
    archivo.write(f"ERROR TEST  = {error_per_te}\n")
    archivo.write("\n")  # Agregar un salto de línea para separar los resultados de cada ejecución

    # Cerrar el archivo
    archivo.close()

save_results(alpha, a, b, eta, num_epochs, num_train_data, num_test_data, num_layers, num_neurons,\
                 initial_weights, final_weights, MSE_values, error_per_tr, error_per_te)