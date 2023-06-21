close all;
clear all;

% Tunning parameters
alpha = 0.01;
a = 1;
eta = 0.1;
num_epochs = 100;
num_train_data = 1000;
num_test_data = 10000;

% Neural network parameters
num_layers = 2;
neurons_in_layers = [25, 10];

% Import data set, images as matrices, and labels as vectors. 
digits_images_tr = unpack_dataset('archive/train-images.idx3-ubyte');
labels_tr = unpack_labels('archive/train-labels.idx1-ubyte');
digits_images_te = unpack_dataset('archive/t10k-images.idx3-ubyte');
labels_te = unpack_labels('archive/t10k-labels.idx1-ubyte');

% Initialize neural network
inputs = zeros(1,10);
perc_mult_lay = InterconNeuralNet(inputs, num_layers, neurons_in_layers, 4, a, "o");


% Train neural network 
MSE_list = perc_mult_lay.train_perc_mult(eta, alpha, num_epochs, digits_images_tr(1:num_train_data,:), labels(1:num_train_data,:));

% MSE plot
plot(1:lenght(MSE_list), MSE_list);


% Trainning test
error_tr = perc_mult_lay.test(digits_images_tr, labels_tr);
display("error_tr:", error_tr);
% Generalization test


% Save final_weights





% Display image
% image = digits_images_tr(200,:);
% image = reshape(image, [28,28]);
% image = rot90(image,3);
% image = fliplr(image);
% imshow(image)
display(labels_tr(200))





