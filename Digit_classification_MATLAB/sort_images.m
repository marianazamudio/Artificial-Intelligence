function [digits_images, idx_digit_change] = sort_images(labels, images, mode, num_dig)
    % Mode 1 --> first num_dig 
    % Mode 2 --> all data set
    
    if mode == 1
       images_per_digit = floor(num_dig/10);
       module = mod(num_dig, 10);
       % List that indicates images per digit to collect. idx = digit.
       images_per_digit = repmat(images_per_digit, 1, 10);
       % Find random digits to acquire the module images.
       indices_aleatorios = randperm(10, module);
       for i = indices_aleatorios
           images_per_digit(i) = images_per_digit(i) + 1;
       end
       
        % Generate list where it is stored the index where new digit information start
        idx_digit_change = cumsum(images_per_digit);
        idx_digit_change(end+1) = num_dig;
       
    end
    
    if mode == 2
         % Generate empty lists
        images_per_digit = zeros(1, 10);
        idx_digit_change = [];
        
        
    end
    
    digits_images = {};
    % Iterate between digits and number of images to obtained per digit.
    for i = 0:9
        if mode == 1
            % Obtener las primeras n imagenes del digito i
            idx = find(labels == i, n, 'first');

        elseif mode == 2
            idx = find(labels == i);
            num_img = numel(idx);
            images_per_digit(i+1) = num_img;
            idx_digit_change = cumsum(images_per_digit);

        end
    
        i_images = images(idx, :);
        for j = 1:size(i_images, 1)
            image = i_images(j, :);
            image = image(:).';
            digits_images{end+1} = image;
        end  
    end 
    if mode == 1
        idx_digit_change(end+1) = num_dig;
    end
end
