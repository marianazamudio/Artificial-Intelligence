# --------------------------------------------------- #
# File: interconnected_neuronal_network.py
# Author: Mariana Zamudio Ayala
# Date: 07/05/2023
# Description: Program that defines the class 
# InterconNeuralNet 
# --------------------------------------------------- #
import numpy as np

class InterconNeuralNet:
    """
        Class that represents an interconnected neural network

        Attributes:
        ---------------
        num_layers: int
            Number of layers in the neural network

        neurons_in_layers: list
            List that contains the number of neurons in each layer
            ex. (N[1], N[2], ..., N[c]), c = num_layers. 
        TODO
        act_funct: int
            Type of activation function.
            1 --> signoide
            2 --> sesgo
            3 --> tanh
        
        random_weights: list
            List that contains weight matrices for each layer as
            numpy biderectional arrays
        
        output: numpy array
            Array that stores the output value of each neuron in the 
            last layer

        Methods:
        --------
        compute_output():
            Computes the output value of the neural network using a random 
            initialization of weights on each neuron and the inputs Xi = i 
            where i = 1,2,3, ..., n, and n is the number of neurons in the 
            network. 
    """

    def __init__(self, num_layers, neurons_in_layers, act_funct):
        """ 
            Inicializa una estancia de la clase InterconNeuralNet

            Parameters
            -----------
            num_layers: int
            Number of layers in the neural network

            neurons_in_layers: list
                List that contains the number of neurons in each layer
                ex. (N[1], N[2], ..., N[c]), c = num_layers. 
            TODO
            act_funct: int
                Type of activation function.
                1 --> signoide
                2 --> sesgo
                3 --> tanh
        """
        # Store customizable parameters
        self.num_layers =  num_layers
        self.neurons_in_layers =  neurons_in_layers
        self.act_funct = act_funct

        # Create void list for storing the outputs yj of the neurons
        output = np.zeros(neurons_in_layers[-1])

        # ---Initialize random weights to each connection of a neuron
        # Create void array, to store the weights of the connections in 
        # each layers as matrices.
        random_weights = []
        # Iterate between layers
        for i in range(self.num_layers-1):
            width = self.neurons_in_layers[i] 
            height = self.neurons_in_layers[i+1] 
            # Create matrix with random weights for the layer
            random_weights_layer = np.random.rand(width, height)
            # Append weights of the layer to the array with all the weights
            random_weights.append(random_weights_layer)
        # ---

        # Store other parameters
        self.output = output
        self.random_weights = random_weights
        

    def compute_output():
        """
            Computes the output value of the neural network using a random 
            initialization of weights on each neuron and the inputs Xi = i 
            where i = 1,2,3, ..., n, and n is the number of neurons in the 
            network. 
        """
        pass

