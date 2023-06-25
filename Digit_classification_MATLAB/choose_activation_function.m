function act_func = choose_activation_function(choice)
    % Select activation function
    if choice == 1
        act_func = @evaluate_threshold_func;
    elseif choice == 2
        act_func = @evaluate_sigmoid_func;
    elseif choice == 3
        act_func = @evaluate_signum_func;
    elseif choice == 4
        act_func = @evaluate_hiperb_tan_func;
    else
        error('act_func value is not valid, must be an int between 1 and 4');
    end
end

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
