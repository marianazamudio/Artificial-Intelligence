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
d = 0
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
a = 2
b = 1
alpha = 0.01
# eta range in which it will lineatly vary during training
eta_range = (1e-1, 1e-5) 
# -----

# Define plots
fig, axs = plt.subplots(2, 2)

# Obtain coordinates in region A to do the training of perceptron
coords_train = generate_coords_in_moon(r, w, d, int(n_train_coord/2))

# Obtain coordinates in region B and concatete them with the ones in region A
coords_train = np.hstack((coords_train, generate_coords_in_moon(r, w, d, int(n_train_coord/2), "B")))

# Initialize the multi layer perceptron
inputs = [0,0]
num_layers = 2
num_neurons = [2,1]
per_mult_layer = InterconNeuralNet(inputs, num_layers, num_neurons, 4, a, b)


# ---- Train perceptron 
pesos, eta_values, MSE_values = per_mult_layer.train_perceptron_mult(eta_range, alpha, num_epochs, \
                                                            data_set=coords_train, \
                                                            class_indx=int(n_train_coord/2))

# ---Plot training coordinates
axs[0,0].set_title('Tranning')
axs[0,0].set_xlabel('x1')
axs[0,0].set_ylabel('x2')
limit = int(n_train_coord/2)
axs[0,0].scatter(coords_train[0,:limit], coords_train[1,:limit])
axs[0,0].scatter(coords_train[0, limit:], coords_train[1, limit:], marker="x")
axs[0,0].grid(True)

# ---- Plot the hyperplane 
if pesos[2] != 0:
    x1 = np.arange(-r-w, 2*r+w, 0.05)
    x2 = -pesos[1]/pesos[2] * x1 - pesos[0]/pesos[2]
# Case for when the hiperplane has an inf slope 
else:
    x2 = np.arange(-r-w-d,r+w, 0.05)
    x1 = np.ones(x2.shape[0])
    x1 = -pesos[0]/pesos[1] * x1
    print(x1)
axs[0,0].plot(x1,x2, color = "blue")
axs[1,1].plot(x1,x2, color = "blue")

# --- Plot eta
epochs = np.arange(1,len(eta_values)+1,1)
axs[0,1].plot(epochs,eta_values)
axs[0,1].set_title('Eta')
axs[0,1].set_xlabel('epochs')
axs[0,1].set_ylabel('eta')

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

# --- Plot test
axs[1,1].set_title('Test')
axs[1,1].set_xlabel('x1')
axs[1,1].set_ylabel('x2')
axs[1,1].scatter(coords_test[0,:limit], coords_test[1,:limit])
axs[1,1].scatter(coords_test[0, limit:], coords_test[1, limit:], marker="x")
axs[1,1].grid(True)

# ---- Compute error rate and print errors
print("Entradas                           Exp.Res    Obt.Res ")
print("------------------------------------------------------")
# Initalize num of errors and expected response
errors = 0
expected_response = 1

for i in range(coords_test.shape[1]):
    if i == 1000:
        expected_response = -1
    inputs = coords_test[:,i]
    inputs = inputs.tolist()
    perceptron.set_inputs(inputs)
    actual_response = perceptron.compute_output()

    if actual_response != expected_response:
        errors +=1
        print(inputs, expected_response, actual_response)


# Print error rate
print(f"Found {errors} errors")
error_rate = (errors/n_test_coord) * 100
print(f"The error rate is: {error_rate}%")

# --- Show graphs
plt.tight_layout()
plt.show()
