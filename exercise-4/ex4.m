%% Machine Learning Online Class - Exercise 4 Neural Network Learning
close all;clc
%% Part.1 Load data and visualizing
fprintf('load data and visualizing ...\n');
load('ex4data1.mat');
randSel = randperm(size(X,1));
figure(1);
displayData(X(randSel(1:100),:));
hold off;

%% Part.2 Load parameters and compute cost
fprintf('load parameters ...\n');
load('ex4weights.mat');
nn_parameter = [Theta1(:);Theta2(:)];

input_layer_size = size(X,2);
hidden_layer_size = 25;
label_nums = 10;

[J,~] = nnCostFunction(X,y,nn_parameter ,input_layer_size,hidden_layer_size,label_nums);
fprintf('The cost at weights is (without lambda=0): %f3.2\n',J);
lambda =1;
[J,~] = nnCostFunction(X,y,nn_parameter ,input_layer_size,hidden_layer_size,label_nums,lambda);
fprintf('The cost at weights is (without lambda=1): %f3.2\n',J);

%% Part.3 Initializint Parameters
fprintf('Initializing parameters ....\n');
theta1 = randInitializeWeights(input_layer_size,hidden_layer_size);
theta2 = randInitializeWeights(hidden_layer_size,label_nums);
initial_theta = [theta1(:) ; theta2(:)];

fprintf('\n Checking Backpropagation... \n');

%  Check gradients by running checkNNGradients
checkNNGradients;
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% Part.4 Train NN
fprintf('\nTraining Neural Network... \n');
lambda =1;
costFunction = @(p) nnCostFunction(X,y,p,input_layer_size, ...
                                   hidden_layer_size, ...
                                   label_nums,lambda);
options = optimset('MaxIter', 50);
[nn_params, cost] = fmincg(costFunction, initial_theta , options);
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 label_nums, (hidden_layer_size + 1));             

%% Part.5 Visualize Weights and predict
fprintf('\nVisualizing Neural Network... \n')
figure(2);
displayData(Theta1(:, 2:end));
pred = predict(Theta1, Theta2, X);
% X一直没有添加第一维的偏置1
title(sprintf('\n The wights with training set accuracy: %f\n', mean(double(pred == y)) * 100));
hold off;
