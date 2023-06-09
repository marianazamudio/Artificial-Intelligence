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
d = -1
# Width of the moons 
w = 6
# num coordinates for trainning
n_train_coord = 1000
# num coordinates for testing 
n_test_coord = 2000

# ---- Define tunning parameters
# epochs
num_epochs = 100
# a and alpha
a = 1
b = 1
alpha = 0.09
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
num_neurons = [20,1]

per_mult_layer = InterconNeuralNet(inputs, num_layers, num_neurons, 4, a, b, wt="r")

# ---- Train perceptron 
MSE_values = per_mult_layer.train_perceptron_mult(eta_range, alpha, num_epochs, \
                                    data_set=coords_train, \
                                    class_indx=int(n_train_coord/2))
#print(best_w, "best_w")
#per_mult_layer.set_weights(best_w)
print("--------------")
print(per_mult_layer.weights)
# ---Plot classification of dataset after training
for idx in range(len(coords_train[0])):
    # Set inputs
    per_mult_layer.set_inputs([coords_train[0][idx],coords_train[1][idx]])
    # Compute output
    y = per_mult_layer.compute_output()[-1]

    if y < 0:
        color_p = "blue"
    else: 
        color_p = "orange"
    axs[0,0].scatter(coords_train[0][idx],coords_train[1][idx], color=color_p)




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

for idx in range(n_test_coord):
    # Set inputs
    per_mult_layer.set_inputs([coords_test[0][idx],coords_test[1][idx]])
    # Compute output
    y = per_mult_layer.compute_output()[-1]

    if y < 0:
        color_p = "blue"
    else: 
        color_p = "orange"
    axs[1,1].scatter(coords_test[0][idx],coords_test[1][idx], color=color_p)

# Print error rate
#print(f"Found {errors} errors in test")
#error_rate = (errors/n_test_coord) * 100
#print(f"The error rate is: {error_rate}%")

# --- Show graphs
plt.tight_layout()
plt.show()
