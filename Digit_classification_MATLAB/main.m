%close all;
%clear all;

% Tunning parameters
alpha = 0.00005;
a = 0.8;
eta = 0.00001;
num_epochs = 500;
num_train_data = 5000;
num_test_data = 10000;

% Import data set, images as matrices, and labels as vectors. 
digits_images_tr = unpack_dataset('archive/train-images.idx3-ubyte');
labels_tr = unpack_labels('archive/train-labels.idx1-ubyte');
digits_images_te = unpack_dataset('archive/t10k-images.idx3-ubyte');
labels_te = unpack_labels('archive/t10k-labels.idx1-ubyte');

% Neural network parameters
num_layers = 3;
neurons_in_layers = [75, 25, 10];
act_func_num = 4; %Tanh
act_func = choose_activation_function(act_func_num);
size_dataset = size(digits_images_tr);
num_inputs = size_dataset(2);

% Initialize weights
[weights, delta_weights] = initialize_weights(num_inputs, num_layers, neurons_in_layers, "r");
% for i = 1:numel(weights)
%     cambio_actual{i} = weights{i} * 0;
% end
%inputs = digits_images_tr(1, :);
%outputs = compute_output(act_func, a, inputs, num_layers, weights);

% Train neural network 
[weights, MSE_list] = train_perc_mult(weights, delta_weights, eta, alpha, a, num_epochs, digits_images_tr(1:num_train_data,:), labels_tr(1:num_train_data,:), neurons_in_layers, act_func_num);

% MSE plot
%plot(MSE_list);

% % Trainning test
[error_tr, results] = test(digits_images_tr(1:num_train_data,:), labels_tr(1:num_train_data,:), weights, neurons_in_layers, act_func, a);

results

error_tr
% display("error_tr:");
% error_tr
% % Generalization test
% 
% 
% % Save final_weights
% 
% 
% 
% 
% 
% % Display image
% % image = digits_images_tr(200,:);
% % image = reshape(image, [28,28]);
% % image = rot90(image,3);
% % image = fliplr(image);
% % imshow(image)
% display(labels_tr(200))
% 
% 
% 
% 
% 
