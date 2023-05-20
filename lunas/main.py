# --------------------------------------------------- #
# File: main.py
# Author: Mariana Zamudio Ayala
# Date: 15/05/2023
# Description:
# Program that trains a sigle layer
# perceptron during 50 epochs to classify points 
# in a cartesian plane. 
# Each class is a set of points inside a moon area
# defined by its radious, width and distance.
# The MCE is computed and ploted for each epoch.
# After the trainning with 1000 points of data, the 
# percepton is tested with 2000 points of data, and the
# error rate is computed. 
# --------------------------------------------------- #
from rnd_coord import generate_coords_in_moon
from interconnected_neuronal_network import InterconNeuralNet
import matplotlib.pyplot as plt
import numpy as np

# --- Define moon data 
# Center radious of the moon 
r = 3  
# Vertical distance between center of moon 
# in region A and moon in region B
d = -0.5
# Width of the moons 
w = 1  
# num coordinates for trainning
n_train_coord = 1000
# num coordinates for testing 
n_test_coord = 2000
# epochs
num_epochs = 50  
# eta range
eta_range = (1e-1, 1e-5) 
# -----

# Define plots
fig, axs = plt.subplots(2, 2)


# Obtain coordinates in region A
coords_xA, coords_yA = generate_coords_in_moon(r, w, d, int(n_train_coord/2))

# Obtain coordinates in region B
coords_xB, coords_yB = generate_coords_in_moon(r, w, d, int(n_train_coord/2), "B")

# Initialize the single layer perceptrons
inputs = [0,0] 
num_layers = 1
num_neurons = [1]
perceptron = InterconNeuralNet(inputs, num_layers, num_neurons, 3)

# Get pairs input- otput [[[x1, x2], output], ....[[x1, x2], output]]
pairs_io_A = [[[coords_xA[i], coords_yA[i]], 1] for i in range(len(coords_yA))]
pairs_io_B = [[[coords_xB[i], coords_yB[i]], -1] for i in range(len(coords_yB))]
pairs_io = pairs_io_A + pairs_io_B

# ---- Train perceptron 
pesos, eta_values, MSE_values = perceptron.train_perceptron(eta_range, num_epochs,pairs_io)
print(len(eta_values))
# Plot training coordinates
axs[0,0].set_title('Tranning')
axs[0,0].set_xlabel('x1')
axs[0,0].set_ylabel('x2')
axs[0,0].scatter(coords_xA, coords_yA)
axs[0,0].scatter(coords_xB, coords_yB, marker="x")
axs[0,0].grid(True)

# Plot the hyperplane TODO change how the line is plotted
x1 = np.arange(-r-w, 2*r+w, 0.05)
x2 = -pesos[1]/pesos[2] * x1 - pesos[0]/pesos[2]
axs[0,0].plot(x1,x2, color = "blue")
axs[1,1].plot(x1,x2, color = "blue")

# Plot eta
epochs = np.arange(1,len(eta_values)+1,1)
axs[0,1].plot(epochs,eta_values)
axs[0,1].set_title('Eta')
axs[0,1].set_xlabel('epochs')
axs[0,1].set_ylabel('eta')

# Plot MSE
axs[1,0].plot(epochs, MSE_values)
axs[1,0].set_title('Mean square error')
axs[1,0].set_xlabel('epochs')
axs[1,0].set_ylabel('MSE')



# ------ Test perceptron 
# Generate pairs
# Region A
coords_xA, coords_yA = generate_coords_in_moon(r, w, d, int(n_test_coord/2))
# Region B
coords_xB, coords_yB = generate_coords_in_moon(r, w, d, int(n_test_coord/2), "B")
pairs_io_A = [[[coords_xA[i], coords_yA[i]], 1] for i in range(len(coords_yA))]
pairs_io_B = [[[coords_xB[i], coords_yB[i]], -1] for i in range(len(coords_yB))]
pairs_io = pairs_io_A + pairs_io_B

# Print results
print("Entradas   Exp.Res   Obt.Res   Success")
print("--------------------------------------")
errors = 0
for pair in pairs_io:
    success = True
    inputs = pair[0]
    perceptron.set_inputs(inputs)
    response_obtained = perceptron.compute_output()
    expected_response = pair[1]
    if response_obtained != expected_response:
        errors += 1
        success = False
    print(inputs, "\t  ",expected_response, "   ",response_obtained[0], "    ", success)

# Plot test
axs[1,1].set_title('Test')
axs[1,1].set_xlabel('x1')
axs[1,1].set_ylabel('x2')
axs[1,1].scatter(coords_xA, coords_yA)
axs[1,1].scatter(coords_xB, coords_yB, marker="x")
axs[1,1].grid(True)

# Print error rate
print(f"Found {errors} errors")
error_rate = (n_test_coord/(n_test_coord-errors) * 100)-100
print(f"The error rate is: {error_rate}%")

# Show graphs
plt.tight_layout()
plt.show()