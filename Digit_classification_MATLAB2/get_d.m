function d = get_d(desired_res, num_outputs, act_func_num)
    if act_func_num  == 3 || act_func_num == 4
        d = -1 * ones(1,num_outputs);
    else
        d = zeros(1,num_outputs);
    end
    d(desired_res+1) = 1;

end