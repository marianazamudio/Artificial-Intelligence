function [weights, MSE_list] = train_perc_mult(weights, cambio_anterior, eta, alpha, a, num_epochs, dataset, labels, neurons_in_layers, act_func_num) 
            % Obtain num of outputs
            num_outputs = neurons_in_layers(end);
            num_layers = length(neurons_in_layers);
            act_func = choose_activation_function(act_func_num);
            num_data = length(labels);
            eta_var = false;
            if length(eta) == 2
                % Compute step
                eta_step = (eta(0) - eta(1))/num_epochs;
                eta = eta(0);
                eta_var = true;
            end 
            %MSE_list = zeros(1,num_epochs);
            MSE_list = []
            for epoch=1:num_epochs
                disp(epoch)
                % Generate rand indexes
                idxs = randperm(num_data); % Generar índices aleatorios
                % Initialize MSE 
                MSE = 0;

                for i = 1:num_data
                    % Configure input
                    idx = idxs(i);
                    inputs = dataset(idx,:);
                    %obj.set_inputs(dataset(idx,:))
                     
                    % Get d_n
                    desired_res = labels(idx);
                    d_n = get_d(desired_res, num_outputs, act_func_num);

                    % Forward Computation
                    y = compute_output(act_func, a, inputs, num_layers, weights);
                    o_n = y(end-num_outputs+1:end);

                    % Backward computation
                    [weights, cambio_anterior] = back_computation(eta, alpha,a , d_n, inputs, neurons_in_layers, y, act_func_num, weights, cambio_anterior);

                    %MSE
                    diferencias_cuadradas = (d_n - o_n).^2;
                    suma_dif_cuad = sum(diferencias_cuadradas);
                    MSE = MSE + suma_dif_cuad;

                end
                % Compute MSE of the epoch
                MSE = MSE/num_data;
                MSE
                MSE_list= [MSE_list MSE];
                
                plot(MSE_list);
                drawnow;
                
                % Update eta
                if eta_var
                    eta = eta + eta_step;
                end
            end
        end
