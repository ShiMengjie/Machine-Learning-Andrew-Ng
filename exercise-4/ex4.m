%% Machine Learning Online Class - Exercise 4 Neural Network Learning
close all;clc

%% ======================== Part.1 Load data and visualizing ========================
fprintf('load data and visualizing ... \n');
load('ex4data1.mat');
randsel = randperm(size(X,1));
figure(1);
displayData(X(randsel(1:100),:));
hold off;

%% ======================= Part.2 Load parameters and compute cost (feedforward) ============
fprintf('load parameters ... \n');
load('ex4weights.mat');

fprintf('Compute cost (feedforward) ... \n');
input_layer_size = size(X,2);
hidden_layer_size =25;
label_num =10;

Theta = [Theta1(:);Theta2(:)];

J= nnCostFunction(input_layer_size,hidden_layer_size,label_num,Theta,X,y);
fprintf('Cost at parameters (loaded from ex4weights): %f \n (this value should be about 0.287629)\n',J);

%% ===================== Part.3 Compute cost with regularization ==========================
fprintf('Compute cost with regularization ... \n');
lambda =1;
J= nnCostFunction(input_layer_size,hidden_layer_size,label_num,Theta,X,y,lambda);

fprintf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n(this value should be about 0.383770)\n'], J);

%% ==================== Part.4 Initializing Pameters ================
% 随机初始化权重参数init_theta1和init_theta2
fprintf('\n Initializing Neural Network Parameters ...\n')

init_theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
init_theta2 = randInitializeWeights(hidden_layer_size, label_num);

initial_Theta = [init_theta1(:) ; init_theta2(:)];

%% =============== Part 7: Implement Backpropagation ===============
fprintf('\n Checking Backpropagation... \n');

%  Check gradients by running checkNNGradients
checkNNGradients;
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% =================== Part 8: Training NN ===================
fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 50);
%  You should also try different values of lambda
lambda = 1;
% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction( input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, p,X, y, lambda);
% Now, costFunction is a function that takes in only one argument (the neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================= Part 9: Visualize Weights =================
fprintf('\nVisualizing Neural Network... \n')
figure(2);
displayData(Theta1(:, 2:end));

hold off;

%% ================= Part 10: Implement Predict =================
pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
