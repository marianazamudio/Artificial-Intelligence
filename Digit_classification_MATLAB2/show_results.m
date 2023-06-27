function show_results(images, labels, weights, neurons_in_layers, act_func, a)
    num_layers = length(neurons_in_layers);
    num_outputs = neurons_in_layers(end);
    size_labels = size(labels);
    num_images = size_labels(1);
    
    for idx = 1:num_images
       % Choose  image
       image = images(idx,:);
       image_resh = reshape(image, 28, 28);
       image_resh = rot90(image_resh, -1);
       image_resh = fliplr(image_resh);
       subplot(1,2,1);
       imshow(image_resh)
       d = labels(idx);
       title(d);
       
       % Compute output
       output = compute_output(act_func,a, image, num_layers, weights);
       o_n = output(end-num_outputs+1:end);
       [val_max, result] = max(o_n);
       result = result -1;
       
       % Show result
       labels_plot = 0:9;
       subplot(1,2,2);
       bar(labels_plot, o_n);
       
       pause()
    end


end