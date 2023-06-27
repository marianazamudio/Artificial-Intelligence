function [weights, cambio_actual] = initialize_weights(inputs, num_layers, neurons_in_layers, wt)
            % Constructor method
            % Inputs:
            %   - inputs: number of inputs of the neural network
            %   - num_layers: Number of layers in the neural network
            %   - neurons_in_layers: List that contains the number of neurons in each layer (e.g., [N(1), N(2), ..., N(c)], c = num_layers)
            %   - wt: Type of initialization of weights ('r' --> random uniform [-1, 1], 'o' --> ones, 'z' --> zeros)
            
            % Initialize random weights to each connection of a neuron
            weights = cell(1, num_layers);
            cambio_actual = cell(1, num_layers);
            
            elements_in_l = [inputs, neurons_in_layers];
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
            
            % Iterar entre numero de neuronas en cada capa
            for i = 1:numel(weights)
                cambio_actual{i} = weights{i} * 0;
            end
        end