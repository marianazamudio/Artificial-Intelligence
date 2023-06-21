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
            obj.inputs = [1, inputs];
            obj.num_layers = num_layers;
            obj.neurons_in_layers = neurons_in_layers;
            obj.a = a;
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
            
            % Create vector with num of inputs and num of neurons in lay
            elements_in_l = [numel(obj.inputs), obj.neurons_in_layers];
            
            % Initialize random weights to each connection of a neuron
            weights = cell(1, numel(elements_in_l)-1);
            
            % Iterate between layers
            for i = 1:numel(elements_in_l)-1
                % num rows = num neurons in next layer 
                rows = elements_in_l(i+1);
                % num col = num neurons in current layer + 1 (bias) 
                columns = elements_in_l(i);
                if i ~= 1
                    columns = columns + 1;
                end
                % Create weight matrix that conects current layer with the
                % next one. There are 3 modes: ceros, ones, random. 
                if wt == "r"
                    % min and min limits for random elements of matrix
                    max_r= 1;
                    min_r= -1;
                    w = min_r + (max_r - min_r) * rand(columns, rows);
                elseif wt == "z"
                    w = zeros(columns, rows);
                elseif wt == "o"
                    w =  ones(columns, rows);
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
        
        function set_inputs(obj, inputs)
            % Modifies inputs of the neural net, adds input X0 = +1 for the bias
            % automatically, no need to enter it.

            % Parameters:
            %   inputs: list
            %       List with items float or int that are the input values for
            %       the neural network. An item equal to +1 is added autom. at the
            %       beginning for the bias input.

            obj.inputs = [1, inputs];
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
            start = 1;
            current_neuron = 1;

            % Iterate between layers
            for layer = 1:obj.num_layers
                % Obtain input vector
                if layer == 1
                    inputs_of_layer = obj.inputs;
                else
                    finish = start + obj.neurons_in_layers(layer-1) - 1;
                    inputs_of_layer = obj.outputs(start:finish);
                    % Add a '1' at the beginning for the bias
                    inputs_of_layer = [1, inputs_of_layer];
                end

                % Iterate between neurons in current layer
                for i = 1:obj.neurons_in_layers(layer)
                    % Obtain the weight vector for inputs in neuron k
                    weight_mat = obj.weights{layer};

                    % Take the row i of the matrix that corresponds to the
                    % weights for the inputs of neuron k
                    weight_vect = weight_mat(i, :);

                    % wT(i) * x(i)
                    v_k = dot(inputs_of_layer, weight_vect);

                    % Activation function
                    if obj.act_funct_num == 2
                        y_k = obj.act_funct(v_k, obj.a);
                    elseif obj.act_funct_num == 4
                        y_k = obj.act_funct(v_k, obj.a, obj.b);
                    else
                        y_k = obj.act_funct(v_k);
                    end

                    % Save output of the neuron
                    obj.outputs(current_neuron) = y_k;

                    current_neuron = current_neuron + 1;
                end

                if layer ~= 1
                    % Increment the start indicator for inputs of the layer
                    start = start + obj.neurons_in_layers(layer-1);
                end
            end

            output = obj.outputs;
        end
        
        function back_computation(obj, eta, alpha, d)
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
            neurons_in_layers = [length(obj.inputs)-1, obj.neurons_in_layers];

            % Initialize matrix of local gradients
            local_gradients = zeros(obj.num_layers, max(neurons_in_layers));

            % Iterate between layers, from the last one to the first one
            for layer = obj.num_layers:-1:1
                % Iterate between neurons in the current layer
                for neuron = 1:neurons_in_layers(layer)
                    % List of outputs of neurons, added the input values
                    y = [obj.inputs(2:end), obj.outputs];

                    % Search index for y of the current layer
                    start_curr_lay = sum(neurons_in_layers(1:layer));
                    idx_neuron = start_curr_lay + neuron;

                    % Obtain previous layer outputs
                    start_prev_lay = start_curr_lay - neurons_in_layers(layer-1);
                    y_prev_layer = y(start_prev_lay:start_curr_lay-1);

                    % LAST LAYER
                    if layer == obj.num_layers
                        d_idx = neuron;
                        % SIGMOID
                        if obj.act_funct_num == 2
                            % Compute local gradient
                            local_gradients(layer, neuron) = obj.a * (d(d_idx) - y(idx_neuron)) * y(idx_neuron) * (1 - y(idx_neuron));
                        end
                        % TANH
                        if obj.act_funct_num == 4
                            local_gradients(layer, neuron) = (obj.a / obj.b) * (d(d_idx) - y(idx_neuron)) * (obj.b - y(idx_neuron)^2);
                        end
                    % HIDDEN LAYER
                    elseif layer ~= obj.num_layers
                        local_gradients(layer, neuron) = 0;
                        for k = 1:neurons_in_layers(layer+1)
                            % local gradient of neuron k of next layer * weight k,neuron
                            local_gradients(layer, neuron) = local_gradients(layer, neuron) + local_gradients(layer+1, k) * obj.weights{layer+1}(k, neuron+1); % neuron+1 because weights include the bias in the first position
                        end
                        % SIGMOID
                        if obj.act_funct_num == 2
                            local_gradients(layer, neuron) = local_gradients(layer, neuron) * obj.a * y(idx_neuron) * (1 - y(idx_neuron));
                        end
                        % TANH
                        if obj.act_funct_num == 4
                            local_gradients(layer, neuron) = local_gradients(layer, neuron) * (obj.a / obj.b) * (obj.b - y(idx_neuron)^2);
                        end
                    end

                    % Compute weight change
                    idx = sum(obj.neurons_in_layers(1:layer-1)) + neuron;

                    % [1, y_prev_layer]
                    cambio_actual = eta * local_gradients(layer, neuron) * [1, y_prev_layer] + alpha * obj.cambio_anterior{idx};

                    % Change weights
                    obj.weights{layer}(neuron, :) = obj.weights{layer}(neuron, :) + cambio_actual;

                    % Save actual change
                    obj.cambio_anterior{idx} = cambio_actual;
                end
            end
        end

        function MSE_list = train_perc_mult(obj, eta, alpha, num_epochs, dataset, labels ) 
            % Obtain num of outputs
            num_outputs = obj.neurons_in_layers(end);
            num_data = length(labels);
                eta_var = false;
            if length(eta) == 1
                % Compute step
                eta_step = (eta(0) - eta(1))/num_epochs;
                eta = eta(0);
                eta_var = true;
            end 
            MSE_list = [];

            for i=0:num_epochs
                epoch

                % Generate rand indexes
                idxs = randperm(num_data); % Generar índices aleatorios
                % Initialize MSE 
                MSE = 0;

                for i = 1:num_data
                    % Configure input
                    idx = idxs(i);
                    obj.set_inputs(dataset(idx,:,:))

                    % Get d_n
                    desired_res = labels(idx);
                    d_n = obj.get_d(desired_res, num_outputs);  %TODO

                    % Forward Computation
                    o_n = obj.compute_output();
                    o_n = o_n(end-num_outputs-1:end);

                    % Backward computation
                    obj.back_computation(eta, alpha, d_n)

                    %MSE
                    MSE = MSE + (d_n - o_n)^2;

                end
            % Compute MSE of the epoch
            MSE = sum(MSE);
            MSE = MSE/num_data;
            MSE_list{end} = MSE;

            % Update eta
            if eta_var
                eta = eta + eta_step;
            end
            end
        end

        function d = get_d(obj, desired_res, num_outputs)
            if obj.act_funct_num  == 3 || obj.act_funct_num == 4
                d = -1 * ones(num_outputs, 1);
            else
                d = zeros(num_outputs,1);
            end
            d(desired_res) = 1;
            
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
