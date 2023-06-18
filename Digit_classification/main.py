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
first_weights = perc_mult_layer.weights

# Obtain data set
train_lb, train_im, test_lb, test_im = dataset.get_MNIST_dataset("archive")

# Give the data set a plain format (list) and sort it by digit. 
digits_images_tr, idx_digits_change_tr = dataset.obtain_images(train_lb, train_im, 3)
digits_images_te, idx_digits_change_te = dataset.obtain_images(test_lb, test_im, 3)

# Entrenar
MSE_values = perc_mult_layer.train_perceptron_mult(eta, alpha, num_epochs, \
                                    data_set=digits_images_tr, \
                                    class_indx= idx_digits_change_tr)
# Graficar MSE


# Pruebas para datos de entrenamiento
error_per_tr = perc_mult_layer.test(digits_images_tr, idx_digits_change_tr)
print(f"Error de entrenamiento: {error_per_tr} ")
# Pruebas para data set de test
error_per_te = perc_mult_layer.test(digits_images_te, idx_digits_change_te)
print(f"Error de prueba:{error_per_te} ")


print(perc_mult_layer.weights, "final weights")
print(first_weights, "initial weights")