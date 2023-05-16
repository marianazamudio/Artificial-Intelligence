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
        inputs: list
            List with items float or int that are the input values for 
            the neural network. Do not consider bias, a 1 is added at
            the begginning automatically. 

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
        
        weights: list of numpy bidirectional arrays
            List that contains weight matrices for each layer as
            numpy biderectional arrays
            This is initialized with weights equal to 0
        
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
    

    def __init__(self, inputs, num_layers, neurons_in_layers, act_funct):
        """ 
            Inicializa una estancia de la clase InterconNeuralNet

            Parameters
            -----------
            inputs: list
                List with items float or int that are the input values for 
                the neural network. An item equal to +1 is added autom. at the 
                beginning for the bias input. 

            num_layers: int
                Number of layers in the neural network

            neurons_in_layers: list
                List that contains the number of neurons in each layer
                ex. (N[1], N[2], ..., N[c]), c = num_layers. 
            
            act_funct: int
                Type of activation function.
                1 --> Threshold (Sesgo)
                2 --> Sigmoid (Sinoide)
                3 --> Signum (Signo)
                4 --> Hiperbolic tangent
        """
        # Store customizable parameters
        self.inputs = [1] + inputs
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
        weights = []
        # Iterate between layers
        for i in range(-1,self.num_layers-1):
            # Create additional weight matrix for the inputs
            if i == -1:
                rows = self.neurons_in_layers[0]
                columns = len(self.inputs)
                # Create matrix with random weights for the layer
                weights_layer = np.zeros((rows, columns))
                # Append weights of the layer to the array with all the weights
                weights.append(weights_layer)
            else:
                rows = self.neurons_in_layers[i+1] 
                columns = self.neurons_in_layers[i] 
                # Create matrix with random weights for the layer
                weights_layer = np.zeros((rows,columns))
                # Append weights of the layer to the array with all the weights
                weights.append(weights_layer)
        # ---

        # Store weights as a network attribute
        self.weights = weights.copy()

    """
        Modifies inputs of the neural net, adds input X0 = +1 for the bias
        automatically, no need to enter it. 
        
        Parameters
        ----------
        inputs: list
            List with items float or int that are the input values for 
            the neural network. An item equal to +1 is added autom. at the 
            beginning for the bias input. 
    """
    def set_inputs(self, inputs):
        self.inputs = [1] + inputs
        
    
    """ 
        Trains the perceptron so it can adjust its weights' matrices,
        so it can classify input vectors into two possible
        linearly separable classes.

        Parameters
        ----------
        eta: float
            Float that indicates the learning-rate parameter, 
            a positive constant less than unity

        pairs_io: List
            List of Lists of size 2 with pairs input vector and desired response
            [x(i), d(i)]
    """
    def train_perceptron(self, eta, pairs_io):
        n = 0
        while(True):
            n += 1
            curr_weights = []
            for pair in pairs_io:
                counter = 0
                # Set inputs x(n)
                self.set_inputs(pair[0])
                # Obtain desired response d(n)
                d = pair[1]
                # Compute actual response y(n)
                y = self.compute_output()
                # Adaptation of weight vector 
                self.weights[0] = self.weights[0] + eta * (d - y) * self.inputs
                # Add weight matrix current input set
                curr_weights.append(self.weights[0])
                
                
                # ----- Uncomment to see data of each iteration ---- #
                #print(eta)
                #print(d)
                #print(y)
                #print(self.inputs)
                #print(self.weights[0])
                #input()
                # --------------------------------------------------- #
            
            # Check if weights did not change for the set of inputs used in training
            if np.all(curr_weights == curr_weights[0]):
                # Print number of iterations executed to train the perceptron
                print("iteraciones: ", n, "---------------")
                # Return the matrix of weights obtained after training
                return curr_weights[0]
            
            else:
                # Clear current weights matrix
                curr_weights = []


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
                # Obtain the weight
                #  vector for inputs in neuron k
                #print(self.weights, "weights")
                weight_mat = self.weights[layer]
                
                # Tomar la fila i de la matriz que corresponde a los
                # pesos para las entradas de la neurona k
                weight_vect =  weight_mat[i,:]

                # wT(i) x(i)
                #print(inputs_of_layer, "inputs")
                #print(weight_mat, "mat")
                #print(weight_vect, "vector")
                v_k = np.dot(inputs_of_layer, weight_vect)
                #print(v_k, "v_k")
                # Aplica función de activación 
                y_k = self.act_funct(v_k)

                # Guarda resultado en el vector de salidas 
                self.outputs[current_neuron-1] = y_k

                current_neuron +=1
            
            if layer != 0:
                # Incrementar el indicador de inicio de entradas de la capa
                start += self.neurons_in_layers[layer-1]
        
        
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


