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
d = 2
# Width of the moons 
w = 1  
# num coordinates for trainning
n_train_coord = 1000
# num coordinates for testing 
n_test_coord = 2000
# epochs
num_epochs = 50  #TODO
# eta range
eta_range = (1e-1, 1e-5) #TODO
# -----

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
pesos, eta_values = perceptron.train_perceptron(eta_range, num_epochs,pairs_io)
print(pesos)
print(eta_values[-1])
print(len(eta_values))

# Plot training coordinates
plt.scatter(coords_xA, coords_yA)
plt.scatter(coords_xB, coords_yB, marker="x")
plt.grid(True)

# Plot the hyperplane TODO change how the line is plotted
x1 = np.arange(-r-w, 2*r+w, 0.05)
x2 = -pesos[1]/pesos[2] * x1 - pesos[0]/pesos[2]
plt.plot(x1,x2, color = "blue", label = "div")
plt.show()

# ------ Test perceptron 
"""
print("Entradas   Exp.Res   Obt.Res")
print("----------------------------")
for pair in pairs_io:
    inputs = pair[0]
    perceptron.set_inputs(pair[0])
    response_obtained = perceptron.compute_output()
    expected_response = pair[1]
    print(inputs, "\t  ",expected_response, "   ",response_obtained)
    """