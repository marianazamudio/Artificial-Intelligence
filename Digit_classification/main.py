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
import manage_results
import matplotlib.pyplot as plt

# Tunning parameters
alpha = 0.01
a = 1
b = 1
eta = 0.1
num_epochs = 100
num_train_data = 1000
num_test_data = 10000

# Neural network's parameters
inputs = [0 for x in range(784)]
num_layers = 2
num_neurons = [25, 10]


# Initialize neural network
perc_mult_layer = InterconNeuralNet(inputs, num_layers, num_neurons, 4, a, b, wt="r")
initial_weights = perc_mult_layer.weights
print(initial_weights)
input()
# Obtain data set
train_lb, train_im, test_lb, test_im = dataset.get_MNIST_dataset("archive")

# Give the data set a plain format (list) and sort it by digit. 
digits_images_tr, idx_digits_change_tr = dataset.obtain_images(train_lb, train_im, 2, num_train_data)
digits_images_te, idx_digits_change_te = dataset.obtain_images(test_lb, test_im, 2, num_test_data)

# Entrenar
MSE_values = perc_mult_layer.train_perceptron_mult(eta, alpha, num_epochs, \
                                    data_set=digits_images_tr, \
                                    class_indx= idx_digits_change_tr)
# --------------- Graficar MSE
# Crear el gráfico de línea
plt.plot(range(len(MSE_values)), MSE_values)

# Personalizar el gráfico
plt.xlabel('Épocas')
plt.ylabel('MSE')

# ----------------- Pruebas
# Pruebas para datos de entrenamiento
error_per_tr = perc_mult_layer.test(digits_images_tr, idx_digits_change_tr)
print(f"Error de entrenamiento: {error_per_tr} ")

# Pruebas para data set de test
error_per_te = perc_mult_layer.test(digits_images_te, idx_digits_change_te)
print(f"Error de prueba:{error_per_te} ")


# ----------Guardar resultados 
final_weights = perc_mult_layer.weights.copy()
print(final_weights, "final weights")
manage_results.save_results(alpha, a, b, eta, num_epochs, num_train_data, num_test_data, num_layers, num_neurons,\
                 initial_weights, final_weights, MSE_values, error_per_tr, error_per_te)


# Mostrar el gráfico
plt.show()