# --------------------------------------------------- #
# File: main.py
# Author: Mariana Zamudio Ayala
# Date: 15/05/2023
# Description:
# Program that trains a multi layer
# perceptron to classify points in a cartesian plane.
# Each class is a set of points inside a moon area
# defined by its radius, width and distance.
# The MCE is computed and ploted for each epoch.
# After the trainning with 1000 points of data, the 
# percepton is tested with 2000 points of data, and the
# error rate is computed. 
# --------------------------------------------------- #
from rnd_coord import generate_coords_in_moon
from interconnected_neuronal_network import InterconNeuralNet
import matplotlib.pyplot as plt
import numpy as np

# --- Define moon parameters 
# Center radious of the moon 
r = 10  
# Vertical distance between center of moon 
# in region A and moon in region B
d = -3
# Width of the moons 
w = 6
# num coordinates for trainning
n_train_coord = 1000
# num coordinates for testing 
n_test_coord = 2000

# ---- Define tunning parameters
# epochs
num_epochs = 50
# a and alpha
a = 1
b = 1
alpha = 0.01
# eta range in which it will lineatly vary during training
eta_range = (1e-1, 1e-5) 
# -----

# Define plots
fig, axs = plt.subplots(2, 2)

if n_train_coord%2 == 0:
    n_train_coord_A = int(n_train_coord/2)
    n_train_coord_B = n_train_coord_A
else:
    n_train_coord_A = int(n_train_coord/2)+1
    n_train_coord_B = n_train_coord_A - 1

if n_test_coord%2 == 0:
    n_test_coord_A = int(n_test_coord/2)
    n_test_coord_B = n_test_coord_A
else:
    n_test_coord_A = int(n_test_coord/2) + 1
    n_test_coord_B = n_test_coord_A - 1


# Obtain coordinates in region A to do the training of perceptron
coords_train = generate_coords_in_moon(r, w, d, n_train_coord_A)

# Obtain coordinates in region B and concatete them with the ones in region A
coords_train = np.hstack((coords_train, generate_coords_in_moon(r, w, d, n_train_coord_B, "B")))

# Initialize the multi layer perceptron
inputs = [0,0]
num_layers = 2
num_neurons = [20,1]

per_mult_layer = InterconNeuralNet(inputs, num_layers, num_neurons, 4, a, b, wt="r")

# ---- Train perceptron 
MSE_values = per_mult_layer.train_perceptron_mult(eta_range, alpha, num_epochs, \
                                    data_set=coords_train, \
                                    class_indx= n_train_coord_A)

print(per_mult_layer.weights)

# ------------------- [division] ------------------------ #
# Generar una malla de puntos en el espacio de entrada
x1 = np.linspace(-r-w, 2*r+w, 1000)
x2 = np.linspace(-r-d-w, r+w, 1000)
X1, X2 = np.meshgrid(x1, x2)

# Evaluar la red neuronal en cada punto de la malla
Z = np.zeros_like(X1)
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        # Calcular la salida de la red neuronal para el punto (X1[i, j], X2[i, j])
        # utilizando los pesos y sesgos correspondientes
        # y almacenar el resultado en Z[i, j]
        per_mult_layer.set_inputs([X1[i,j], X2[i,j]])
        Z[i,j] = per_mult_layer.compute_output()[-1]

# Trazar la curva de decisión
axs[0,0].contourf(X1, X2, Z, levels=1, colors=('#F9B7FF', '#B7FFF9'), alpha=0.5)
axs[1,1].contourf(X1, X2, Z, levels=1, colors=('#F9B7FF', '#B7FFF9'), alpha=0.5)
# -------------------------------------------------------------------


# ---Plot classification of dataset after training
error_training = 0
points_A_x = []
points_A_y = []
points_B_x = []
points_B_y = []
for idx in range(len(coords_train[0])):
    # Set inputs
    per_mult_layer.set_inputs([coords_train[0][idx],coords_train[1][idx]])
    # Compute output
    y = per_mult_layer.compute_output()[-1]

    if y < 0:
        color_p = "blue"
        if idx < n_train_coord_A:
            error_training += 1
            points_B_y.append(coords_train[1][idx])
            points_B_x.append(coords_train[0][idx])
                     
    else: 
        color_p = "orange"
        if idx >= n_train_coord_A:
            error_training += 1
            points_A_y.append(coords_train[1][idx])
            points_A_x.append(coords_train[0][idx])
            
    axs[0,0].scatter(coords_train[0][idx],coords_train[1][idx], color=color_p)


print("errors in training:", error_training)
error_training = (error_training/n_train_coord) * 100
print(f"error training: {error_training}%")
print(f"reliability training: {100-error_training}%")

# --- Plot eta
epochs = np.arange(1,len(MSE_values)+1,1)

# --- Plot MSE
axs[1,0].plot(epochs, MSE_values)
axs[1,0].set_title('Mean square error')
axs[1,0].set_xlabel('epochs')
axs[1,0].set_ylabel('MSE')


# ------ Test perceptron 
limit = int(n_test_coord/2)
# Generate inputs
# Region A
coords_test = generate_coords_in_moon(r, w, d, limit)
# Region B
coords_test = np.hstack((coords_test, generate_coords_in_moon(r, w, d, limit, "B")))

error_test = 0
for idx in range(n_test_coord):
    # Set inputs
    per_mult_layer.set_inputs([coords_test[0][idx],coords_test[1][idx]])
    # Compute output
    y = per_mult_layer.compute_output()[-1]

    if y < 0:
        color_p = "blue"
        if idx < n_test_coord_A:
            error_test += 1
            
    else: 
        color_p = "orange"
        if idx >= n_test_coord_A:
            error_test += 1
            
    axs[1,1].scatter(coords_test[0][idx],coords_test[1][idx], color=color_p)
print("-------------------------------------")
print("errors in test:", error_test)
error_test = (error_test/n_test_coord) * 100
print(f"error test: {error_test}%")
print(f"reliability test: {100-error_test}%")

# --- Show graphs
plt.tight_layout()
plt.show()

