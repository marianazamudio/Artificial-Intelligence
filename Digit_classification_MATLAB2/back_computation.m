function [weights, cambio_anterior] = back_computation(eta, alpha,a, d, inputs, neurons_in_layers, outputs, act_funct_num, weights, cambio_anterior)
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
            num_layers = length(neurons_in_layers);
            elements_in_layers = [length(inputs), neurons_in_layers];
            
            local_gradients = zeros(num_layers, max(neurons_in_layers));
            
            % List of outputs of neurons, added the input values
            y = [inputs, outputs];
            
            % num_layers+1 to count inputs
            for layer = num_layers+1:-1:2 %TODO
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
                    if layer == num_layers+1
                        d_idx = neuron;
                        % SIGMOID
                        if act_funct_num == 2
                            % Compute local gradient
                            local_gradients(layer, neuron) = a * (d(d_idx) - y(idx_neuron)) * y(idx_neuron) * (1 - y(idx_neuron));
                        end
                        % TANH
                        if act_funct_num == 4
                            local_gradients(layer, neuron) = (a) * (d(d_idx) - y(idx_neuron)) * (1 - y(idx_neuron)^2);
                        end
                    % HIDDEN LAYER
                    elseif layer ~= num_layers+1
                        local_gradients(layer-1, neuron) = 0;
                        %elements_in_layers %TODO
                        %layer+1
                        for k = 1:elements_in_layers(layer+1)
                            % local gradient of neuron k of next layer * weight k,neuron
                            %local_gradients(layer, neuron);
                            %local_gradients(layer+1, k);
                            local_gradients(layer, neuron) = local_gradients(layer, neuron) + local_gradients(layer+1, k) * weights{layer}(k, neuron+1); % neuron+1 because weights include the bias in the first position
                            
                        end
                        % SIGMOID
                        if act_funct_num == 2
                            local_gradients(layer, neuron) = local_gradients(layer, neuron) * (a) * y(idx_neuron) * (1 - y(idx_neuron));
                        end
                        % TANH
                        if act_funct_num == 4
                            local_gradients(layer, neuron) = local_gradients(layer, neuron) * (a) * (1 - y(idx_neuron)^2);
                        end
                    end
                    
                    cambio_actual = eta * local_gradients(layer, neuron) * [1, y_prev_lay] + alpha * cambio_anterior{layer-1}(neuron,:);

                    % Change weights
                    weights{layer-1}(neuron,:) = weights{layer-1}(neuron, :) + cambio_actual;

                    % Save actual change
                    cambio_anterior{layer-1}(neuron,:) = cambio_actual;   
                    
                end
            end
            
        end