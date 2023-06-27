function outputs = compute_output(act_func,a, inputs, num_layers, weights)
            outputs = [];
            inputs_of_layer = [1, inputs]';
            
            % Iterate between layers
            for layer = 1:num_layers
                w = weights{layer};
                y = w*inputs_of_layer;
                y = act_func(y, a);
                outputs(end+1:end+numel(y))= y';
                inputs_of_layer = [1; y];
            end            
end % TODO: TEST