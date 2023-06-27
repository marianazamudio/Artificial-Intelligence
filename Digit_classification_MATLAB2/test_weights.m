load('best_weights.mat')

a = 1;

num_train_data = 15000;
num_test_data = 10000;

% Import data set, images as matrices, and labels as vectors. 
digits_images_tr = unpack_dataset('archive/train-images.idx3-ubyte');
labels_tr = unpack_labels('archive/train-labels.idx1-ubyte');
digits_images_te = unpack_dataset('archive/t10k-images.idx3-ubyte');
labels_te = unpack_labels('archive/t10k-labels.idx1-ubyte');
digits_images_tr = digits_images_tr./255;
digits_images_te = digits_images_te./255;


% Neural network parameters
num_layers = 2;
neurons_in_layers = [25 10];
act_func_num = 4; %Tanh
act_func = choose_activation_function(act_func_num);
size_dataset = size(digits_images_tr);
num_inputs = size_dataset(2);


%  Trainning test
[error_tr, results_tr] = test(digits_images_tr(1:num_train_data,:), labels_tr(1:num_train_data,:), weights, neurons_in_layers, act_func, a);
[error_te, results_te] = test(digits_images_te, labels_te, weights, neurons_in_layers, act_func, a);

error_te
error_tr
% Test random - Show result
show_results(digits_images_te, labels_te, weights, neurons_in_layers, act_func, a)

% Test random - Show result own database 
images_own_db = normalize_images();
labels_own_db = 0:9;
labels_own_db = [labels_own_db 0:9];
labels_own_db = labels_own_db';
show_results(images_own_db, labels_own_db, weights, neurons_in_layers, act_func, a)



%error_tr
%error_te
