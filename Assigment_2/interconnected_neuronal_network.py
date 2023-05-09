# --------------------------------------------------- #
# File: interconnected_neuronal_network.py
# Author: Mariana Zamudio Ayala
# Date: 07/05/2023
# Description: Program that defines the class 
# InterconNeuralNet 
# --------------------------------------------------- #
import numpy as np
import math

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

        inputs: list
            List of inputs [X_1 , ..., X_N[1]], of lenght equal to
            the numbers of neurons in the first layer (N[1]).
            and X1 = 1; ... ; X_N[1] = N[1]
        
        random_weights: list
            List that contains weight matrices for each layer as
            numpy biderectional arrays
        
        outputs: numpy array
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

        # Select activation function
        if act_funct == 1:
            self.act_funct = InterconNeuralNet.evaluate_threshold_func
        elif act_funct == 2:
            self.act_funct = InterconNeuralNet.evaluate_sigmoid_func
        elif act_funct == 3:
            self.act_funct = InterconNeuralNet.evaluate_signum_func
        elif act_funct == 4:
            self.act_funct = InterconNeuralNet.evaluate_hiperb_tan_func
        else:
            raise ValueError("act_funct value is not valid, must be an int between 1 and 4")

        # Create void list for storing the outputs yj of the neurons
        outputs = np.zeros(neurons_in_layers)

        # ---Initialize random weights to each connection of a neuron
        # Create void array, to store the weights of the connections in 
        # each layers as matrices.
        random_weights = []
        # Iterate between layers
        for i in range(self.num_layers-1):
            width = self.neurons_in_layers[i] 
            height = self.neurons_in_layers[i+1] 
            # Create matrix with random weights for the layer
            random_weights_layer = np.random.uniform(low=-1, high=1,size=(width, height))
            # Append weights of the layer to the array with all the weights
            random_weights.append(random_weights_layer)
            
            # Create additional weight matrix for the inputs
            if i == 0:
                # Create matrix with random weights for the layer
                random_weights_layer = np.random.rand(width, height)
                # Append weights of the layer to the array with all the weights
                random_weights.append(random_weights_layer)
        # ---

        # Store other parameters
        self.outputs = outputs
        self.random_weights = random_weights
        self.inputs =  [i+1 for i in range(neurons_in_layers[0])]
        print(self.inputs)
        

    def compute_output(self):
        """
            Computes the output value of the neural network using a random 
            initialization of weights on each neuron and the inputs Xi = i 
            where i = 1,2,3, ..., n, and n is the number of neurons in the 
            network. 
        """
        # Get number of total neurons
        total_neurons = 0
        total_neurons = sum(self.neurons_in_layers)

        # Initialize counters
        neuron_counter_layer = 1
        current_layer = 0
        neurons_in_current_layer = self.neurons_in_layers[current_layer]
        
        # Iterate between neurons
        for k in range(total_neurons):
            # Get inputs of the neurons
            if current_layer == 0:
                inputs = self.inputs
            else: 
                start = sum(self.neurons_in_layers[:current_layer])
                inputs = self.outputs[start: start+neurons_in_current_layer]

            # Get weights of the neurons
            weights = self.random_weights[current_layer]

            v_k = 0
            # Iterate inputs
            for i in range(neurons_in_current_layer):
                # Compute v_k
                vk += inputs[i] * weights[k-start,i-start]

            # Compute output y_k
            yk = self.funct(v_k)
            # Store y_k in array with neuron's outputs.
            self.outputs[k] = y_k

            # Increase neuron counter for the current layer
            neuron_in_curr_layer += 1

            # Check if the current neuron corresponds to the next layer
            if neurons_in_current_layer < neuron_counter_layer:
                current_layer += 1
                neurons_in_current_layer = self.neurons_in_layers[current_layer]


        # Return the outputs of the neurons in the last layer of the net
        return self.outputs[-self.neurons_in_layers[-1]:]

    @staticmethod
    def evaluate_threshold_func(v):  # Sesgo
        if v >= 0:
            return 1
        elif v < 0:
            return 0
        else: 
            raise ValueError("v is not a valid value")

    @staticmethod
    def evaluate_sigmoid_func(v):  # Senoide
        a = 1
        return 1/(1+math.exp(-a*v))

    @staticmethod
    def evaluate_signum_func(v): # Signo
        if v < 0:
            return -1
        elif v = 0:
            return 0
        elif v > 0:
            return 1
        else:
            raise ValueError("v is not a valid value")

   
    @staticmethod
    def evaluate_hiperb_tan_func(v): #Tanh
        a = 1
        return (math.exp(a*v) - math.exp(-a*v))/(math.exp(a*v) + math.exp(-a*v))


