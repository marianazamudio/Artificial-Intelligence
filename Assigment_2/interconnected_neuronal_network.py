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
        
        act_funct: int
            Int that indicates the type of activation function used 
            in the neural network
            1 --> Threshold (Sesgo)
            2 --> Sigmoid (Sinoide)
            3 --> Sinum (Signo)
            4 --> Hiperbolic tangent

        inputs: numpy array
            array of inputs [X_1 , ..., X_N[1]], of lenght equal to
            the numbers of neurons in the first layer (N[1]).
            and X1 = 1; ... ; X_N[1] = N[1]
        
        random_weights: list of numpy bidirectional arrays
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
            
            act_funct: int
                Type of activation function.
                1 --> Threshold (Sesgo)
                2 --> Sigmoid (Sinoide)
                3 --> Sinum (Signo)
                4 --> Hiperbolic tangent
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
        self.outputs = np.zeros(sum(neurons_in_layers))


        # ---Initialize random weights to each connection of a neuron
        # Create void array, to store the weights of the connections in 
        # each layers as matrices.
        random_weights = []
        # Iterate between layers
        for i in range(-1,self.num_layers-1):
            # Create additional weight matrix for the inputs
            if i == -1:
                size = neurons_in_layers[0]
                # Create matrix with random weights for the layer
                random_weights_layer = np.random.rand(size, size)
                # Append weights of the layer to the array with all the weights
                random_weights.append(random_weights_layer)
            else:
                height = self.neurons_in_layers[i] 
                width = self.neurons_in_layers[i+1] 
                # Create matrix with random weights for the layer
                random_weights_layer = np.random.uniform(low=-1, high=1,size=(width, height))
                # Append weights of the layer to the array with all the weights
                random_weights.append(random_weights_layer)
        # ---

        # Store other parameters
        self.random_weights = random_weights
        self.inputs = np.array([i+1 for i in range(neurons_in_layers[0])])
        
        

    def compute_output(self):
        """
            Computes the output value of the neural network using a random 
            initialization of weights on each neuron and the inputs Xi = i 
            where i = 1,2,3, ..., n, and n is the number of neurons in the 
            network. 

            Returns: 
                output: an array that contains the outputs of the neurons in 
                        the last layer of the neural network
        """
        # Initialize variable for indexing the list of outputs of neurons
        # that are also the inputs in the current layer. 
        start = 0 
        current_neuron = 1
        # Iterate between layers
        for layer in range(self.num_layers):
            # Obtain input vector
            if layer == 0:
                inputs_of_layer = self.inputs
            else: 
                finish = start + self.neurons_in_layers[layer-1]
                inputs_of_layer = self.outputs[start:finish]
            
            # Iterate between neurons in current layer
            for i in range(self.neurons_in_layers[layer]):
                # Obtain the weight vector for inputs in neuron k
                weight_mat = self.random_weights[layer]
                # Tomar la fila i de la matriz que corresponde a los
                # pesos para las entradas de la neurona k
                weight_vect =  weight_mat[i, :]

                # Aplica producto punto de entradas y vector de pesos
                #print(inputs_of_layer, "inputs")
                #print(weight_vect, "weight_vect")
                #print(layer, "layer")
                #print(current_neuron, "current_neuron")
                v_k = np.dot(inputs_of_layer, weight_vect)

                # Aplica función de activación 
                y_k = self.act_funct(v_k)

                # Guarda resultado en el vector de salidas 
                self.outputs[current_neuron-1] = y_k

                current_neuron +=1
            
            if layer != 0:
                # Incrementar el indicador de inicio de entradas de la capa
                start += self.neurons_in_layers[layer]
        
        
        return self.outputs[-self.neurons_in_layers[-1]:]

    @staticmethod
    def evaluate_threshold_func(v):
        """
            Evaluates v in the threashold function  

            Returns: 
                output: integer 0 or 1
        """
        # Sesgo
        if v >= 0:
            return 1
        elif v < 0:
            return 0
        else: 
            raise ValueError("v is not a valid value")

    @staticmethod
    def evaluate_sigmoid_func(v):  # Senoide
        """
            Evaluates v in the sigmoid function  

            Returns: 
                output: float in range [0,1]
        """
        a = 1
        return 1/(1+math.exp(-a*v))

    @staticmethod
    def evaluate_signum_func(v): # Signo
        """
            Evaluates v in the signum function  

            Returns: 
                output: integer in {-1, 0, 1}
        """
        if v < 0:
            return -1
        elif v == 0:
            return 0
        elif v > 0:
            return 1
        else:
            raise ValueError("v is not a valid value")

   
    @staticmethod
    def evaluate_hiperb_tan_func(v): 
        """
            Evaluates v in the hiperbolic tangent function  

            Returns: 
                output: float in range [-1,1]
        """
        a = 1
        return (math.exp(a*v) - math.exp(-a*v))/(math.exp(a*v) + math.exp(-a*v))


