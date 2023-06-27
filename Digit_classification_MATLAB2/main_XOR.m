close all;
clear all;

% Tunning parameters
alpha = 0.000005;
a = 1;
eta = 0.2;
num_epochs = 50;
num_train_data = 1000;
num_test_data = 10000;

% Neural network parameters
num_layers = 2;
neurons_in_layers = [5, 1];

% Import data set, images as matrices, and labels as vectors. 
digits_images_tr = [0,0; 1,0; 0,1; 1,1];
labels_tr = [1,0,0,1];


% Initialize neural network
inputs = zeros(1,2);
perc_mult_lay = InterconNeuralNet(inputs, num_layers, neurons_in_layers, 4, a, "o");


% % Train neural network 
 MSE_list = perc_mult_lay.train_perc_mult(eta, alpha, num_epochs, digits_images_tr, labels_tr);

% MSE plot
plot(MSE_list);

% Trainning test
error_tr = perc_mult_lay.test(digits_images_tr, labels_tr);
display("error_tr:");
error_tr
% Generalization test


% Save final_weights





% Display image
% image = digits_images_tr(200,:);
% image = reshape(image, [28,28]);
% image = rot90(image,3);
% image = fliplr(image);
% imshow(image)
display(labels_tr(200))





