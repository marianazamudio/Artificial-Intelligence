classdef InterconNeuralNet
    properties
        inputs           % List with items float or int that are the input values for the neural network. Do not consider bias, a 1 is added at the beginning automatically.
        num_layers       % Number of layers in the neural network
        neurons_in_layers % List that contains the number of neurons in each layer
        act_funct        % Int that indicates the type of activation function used in the neural network
                          % 1 --> Threshold (Sesgo)
                          % 2 --> Sigmoid (Sinoide)
                          % 3 --> Signum (Signo)
                          % 4 --> Hyperbolic tangent
        act_funct_num
        weights          % List of weight matrices for each layer
        a                % Coefficient used to determine the output on tanh and sigmoid functions
        outputs          % Array that stores the output value of each neuron in the last layer
        cambio_anterior
    end
    
    methods
        function obj = InterconNeuralNet(inputs, num_layers, neurons_in_layers, act_funct, a, wt)
            % Constructor method
            % Inputs:
            %   - inputs: List with items float or int that are the input values for the neural network. An item equal to +1 is added automatically at the beginning for the bias input.
            %   - num_layers: Number of layers in the neural network
            %   - neurons_in_layers: List that contains the number of neurons in each layer (e.g., [N(1), N(2), ..., N(c)], c = num_layers)
            %   - act_funct: Type of activation function (1 --> Threshold, 2 --> Sigmoid, 3 --> Signum, 4 --> Hyperbolic tangent)
            %   - a: Coefficient on exponential functions (e^-av) used to determine the output on tanh and sigmoid functions
            %   - b: Unused parameter (not translated)
            %   - wt: Type of initialization of weights ('r' --> random uniform [-1, 1], 'o' --> ones, 'z' --> zeros)
            
            % Store customizable parameters
            obj.inputs = inputs;
            obj.num_layers = num_layers;
            obj.neurons_in_layers = neurons_in_layers;
            obj.a = a;
            obj.act_funct_num = act_funct;
            obj.act_funct = act_funct;

            % Select activation function
            if act_funct == 1
                obj.act_funct = @InterconNeuralNet.evaluate_threshold_func;
            elseif act_funct == 2
                obj.act_funct = @InterconNeuralNet.evaluate_sigmoid_func;
            elseif act_funct == 3
                obj.act_funct = @InterconNeuralNet.evaluate_signum_func;
            elseif act_funct == 4
                obj.act_funct = @InterconNeuralNet.evaluate_hiperb_tan_func;
            else
                error('act_funct value is not valid, must be an int between 1 and 4');
            end

            % Create void list for storing the outputs yj of the neurons
            obj.outputs = zeros(1, sum(neurons_in_layers));
            
            % Initialize random weights to each connection of a neuron
            weights = cell(1, num_layers);
            
            elements_in_l = [length(inputs), neurons_in_layers];
            % Iterate between layers
            for i = 1:num_layers
                % num rows = num neurons in next layer 
                rows = elements_in_l(i+1);
                % num col = num neurons in current layer + 1 (bias) 
                columns = elements_in_l(i) + 1;
                
                % Create weight matrix that conects current layer with the
                % next one. There are 3 modes: ceros, ones, random. 
                if wt == "r"
                    % min and min limits for random elements of matrix
                    max_r= 2.4/784;
                    min_r= -2.4/784;
                    w = min_r + (max_r - min_r) * rand(rows, columns);
                elseif wt == "z"
                    w = zeros(rows, columns);
                elseif wt == "o"
                    w =  ones(rows, columns);
                end 
               % Add weights matrix of the layer, to the general weights
               % matrix
               weights{i} = w;
            end

            % Store weights as a network attribute
            obj.weights = weights;
            
            % Iterar entre numero de neuronas en cada capa
            for i = 1:numel(weights)
                weights{i} = weights{i} * 0;
            end
            obj.cambio_anterior = weights;
        end  
        
        function obj = set_inputs(obj, new_inputs)
            % Modifies inputs of the neural net, adds input X0 = +1 for the bias
            % automatically, no need to enter it.

            % Parameters:
            %   inputs: list
            %       List with items float or int that are the input values for
            %       the neural network. An item equal to +1 is added autom. at the
            %       beginning for the bias input.

            obj.inputs = new_inputs;
        end
        
        
        function output = compute_output(obj)
            % Computes the output value of the neural network using a random
            % initialization of weights on each neuron and the inputs Xi = i
            % where i = 1,2,3, ..., n, and n is the number of neurons in the
            % network.
            %
            % Returns:
            %     output: an array that contains the outputs of the neurons in
            %             the last layer of the neural network

            % Initialize variable for indexing the list of outputs of neurons
            % that are also the inputs in the current layer.
            
            % Iterate between layers
            obj.outputs = [];
            inputs_of_layer = [1, obj.inputs];
            inputs_of_layer = inputs_of_layer';
            
            for layer = 1:obj.num_layers
                w = obj.weights{layer};
                y = w*inputs_of_layer;
                y = obj.act_funct(y, obj.a);
                obj.outputs(end+1:end+numel(y))= y';
                inputs_of_layer = [1; y];
            end
            output = obj.outputs;

            
        end
        
        function [obj] = back_computation(obj, eta, alpha, d)
            % Updates the weights of the multilayer neural net,
            % from the last layer to the most hidden layer of neurons.
            %
            % Parameters:
            %     eta: float in range (0,1)
            %     alpha: float greater than 0
            %     d: int in {0, 1}
            %         desired output according to the input initialized
            %         in the perceptron

            % List of neurons in each layer, added the input layer
            % [1 2 1] ---> 1 input, 2 neurons in the second layer, 1 neuron in the output layer
            
            elements_in_layers = [length(obj.inputs), obj.neurons_in_layers];
            
            local_gradients = zeros(obj.num_layers, max(obj.neurons_in_layers));
            
            % List of outputs of neurons, added the input values
            y = [obj.inputs, obj.outputs];
            
            % obj.num_layers+1 to count inputs
            for layer = obj.num_layers+1:-1:2 %TODO
                % Define the indx where the outputs of current layer begins
                % in vector of outputs (y).
                    if layer == 1
                        start_curr_lay = 1;
                    else
                        start_curr_lay = sum(elements_in_layers(1:layer-1));
                    end
                % Iterate between neurons in the current layer
                for neuron = 1:elements_in_layers(layer)
                    % Obtain idx of y of this neuron
                    idx_neuron = start_curr_lay + neuron;
                    
                    % Obtain past layer outputs
                    start_prev_lay = start_curr_lay + 1 - elements_in_layers(layer-1);
                    y_prev_lay = y(start_prev_lay:start_curr_lay);
                    
                    % LAST LAYER
                    if layer == obj.num_layers+1
                        d_idx = neuron;
                        % SIGMOID
                        if obj.act_funct_num == 2
                            % Compute local gradient
                            local_gradients(layer, neuron) = obj.a * (d(d_idx) - y(idx_neuron)) * y(idx_neuron) * (1 - y(idx_neuron));
                        end
                        % TANH
                        if obj.act_funct_num == 4
                            local_gradients(layer, neuron) = (obj.a) * (d(d_idx) - y(idx_neuron)) * (1 - y(idx_neuron)^2);
                        end
                    % HIDDEN LAYER
                    elseif layer ~= obj.num_layers+1
                        local_gradients(layer-1, neuron) = 0;
                        %elements_in_layers %TODO
                        %layer+1
                        for k = 1:elements_in_layers(layer+1)
                            % local gradient of neuron k of next layer * weight k,neuron
                            %local_gradients(layer, neuron);
                            %local_gradients(layer+1, k);
                            %obj.weights{layer}(k, neuron+1);
                            local_gradients(layer, neuron) = local_gradients(layer, neuron) + local_gradients(layer+1, k) * obj.weights{layer}(k, neuron+1); % neuron+1 because weights include the bias in the first position
                            
                        end
                        % SIGMOID
                        if obj.act_funct_num == 2
                            local_gradients(layer, neuron) = local_gradients(layer, neuron) * (obj.a) * y(idx_neuron) * (1 - y(idx_neuron));
                        end
                        % TANH
                        if obj.act_funct_num == 4
                            local_gradients(layer, neuron) = local_gradients(layer, neuron) * (obj.a) * (1 - y(idx_neuron)^2);
                        end
                    end
                    
                    cambio_actual = eta * local_gradients(layer, neuron) * [1, y_prev_lay] + alpha * obj.cambio_anterior{layer-1}(neuron,:);

                    % Change weights
                    obj.weights{layer-1}(neuron,:) = obj.weights{layer-1}(neuron, :) + cambio_actual;

                    % Save actual change
                    obj.cambio_anterior{layer-1}(neuron,:) = cambio_actual;   
                    
                end
            end
            
        end

        function MSE_list = train_perc_mult(obj, eta, alpha, num_epochs, dataset, labels) 
            % Obtain num of outputs
            num_outputs = obj.neurons_in_layers(end);
            num_data = length(labels);
            eta_var = false;
            if length(eta) == 2
                % Compute step
                eta_step = (eta(0) - eta(1))/num_epochs;
                eta = eta(0);
                eta_var = true;
            end 
            MSE_list = zeros(1,num_epochs);

            for epoch=1:num_epochs
                display(epoch)
                % Generate rand indexes
                idxs = randperm(num_data); % Generar índices aleatorios
                % Initialize MSE 
                MSE = 0;

                for i = 1:num_data
                    % Configure input
                    idx = idxs(i);
                    obj.inputs = dataset(idx,:);
                    %obj.set_inputs(dataset(idx,:))
                     
                    % Get d_n
                    desired_res = labels(idx);
                    d_n = obj.get_d(desired_res, num_outputs);  %TODO

                    % Forward Computation
                    o_n = obj.compute_output();
                    obj.outputs = o_n;
                    o_n = o_n(end-num_outputs+1:end);

                    % Backward computation
                    obj.back_computation(eta, alpha, d_n)

                    %MSE
                    diferencias_cuadradas = (d_n - o_n).^2;
                    suma_dif_cuad = sum(diferencias_cuadradas);
                    MSE = MSE + suma_dif_cuad;

                end
                % Compute MSE of the epoch
                MSE = MSE/num_data;
                MSE
                MSE_list(epoch) = MSE;
                
                % Update eta
                if eta_var
                    eta = eta + eta_step;
                end
            end
        end

        function d = get_d(obj, desired_res, num_outputs)
            if obj.act_funct_num  == 3 || obj.act_funct_num == 4
                d = -1 * ones(1,num_outputs);
            else
                d = zeros(1,num_outputs);
            end
            d(desired_res+1) = 1;
            
        end
        
        function perc_error = test(obj, dataset, labels)
            errors = 0;
            num_outputs = obj.neurons_in_layers(end);
            for i = 1:length(dataset)
               % Configure inputs
               obj.set_inputs(dataset(i,:,:))
               
               % Compute output
               output = obj.compute_output();
               output = output(end-num_outputs-1:end);
               [val_max, output] = max(output);
               
               % Count errors
               d = labels(i);
               if output ~= d
                   errors = errors + 1;
               end
            end
            % Compute error percentage
            display(errors)
            perc_error = errors/length(labels);
        
        end
    end
    methods (Static)
        function output = evaluate_threshold_func(x)
            % Threshold (Sesgo) activation function
            output = double(x > 0);
        end
        
        function output = evaluate_sigmoid_func(x, a)
            % Sigmoid (Sinoide) activation function
            output = 1 ./ (1 + exp(-a .* x));
        end
        
        function output = evaluate_signum_func(x)
            % Signum (Signo) activation function
            output = sign(x);
        end
        
        function output = evaluate_hiperb_tan_func(x, a)
            % Hyperbolic tangent activation function
            output = tanh(a .* x);
        end
    end
end
