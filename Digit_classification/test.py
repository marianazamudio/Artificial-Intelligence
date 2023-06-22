from interconnected_neuronal_network import InterconNeuralNet


# Neural network's parameters
inputs = [0 for x in range(2)]
num_layers = 2
num_neurons = [5, 4]
a = 1
b=1

# Initialize neural network
perc_mult_layer = InterconNeuralNet(inputs, num_layers, num_neurons, 4, a, b, wt="o")

a = perc_mult_layer.compute_output()

print(a)