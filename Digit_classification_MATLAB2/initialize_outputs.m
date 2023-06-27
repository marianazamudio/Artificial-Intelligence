function outputs = initialize_outputs(neurons_in_layers)
            % Inputs:
            %   - num_layers: Number of layers in the neural network
            % Create void list for storing the outputs yj of the neurons
            outputs = zeros(1, sum(neurons_in_layers));