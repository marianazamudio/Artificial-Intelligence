 function [perc_error, results] = test(dataset, labels, weights, neurons_in_layers, act_func, a)
            errors = 0;
            num_outputs = neurons_in_layers(end);
            num_layers = length(neurons_in_layers);
            results = [];
            for i = 1:length(dataset)
               % Configure inputs
               inputs = (dataset(i,:));
               
               % Compute output
               output = compute_output(act_func,a, inputs, num_layers, weights);
               o_n = output(end-num_outputs+1:end);
               [val_max, result] = max(o_n);
               result = result -1;
               
               % Count errors
               d = labels(i);
               if result ~= d
                   errors = errors + 1;
               end
               results(end+1,:) =  [d result o_n];
            end
            % Compute error percentage
            %display(errors)
            perc_error = errors/length(labels);
        
        end