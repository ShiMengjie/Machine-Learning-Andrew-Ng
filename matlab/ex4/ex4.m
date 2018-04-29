%% Machine Learning Online Class - Exercise 4：Neural Network Learning
close all;clc

%% 1.读取和可视化数据
fprintf('load data and visualizing ...\n');
load('ex4data1.mat');
randSel = randperm(size(X,1));
figure(1);
displayData(X(randSel(1:100),:));
hold off;

%% 2.读取参数并计算代价
fprintf('load parameters ...\n');
load('ex4weights.mat');
nn_parameter = [Theta1(:);Theta2(:)];
% 获得输入层、隐藏层、输出层的节点个数（不包括每一层的偏置维度）
input_layer_size = size(X,2);
hidden_layer_size = 25;
label_nums = 10;
% 计算代价：有正则项和没有正则项
[J,~] = nnCostFunction(X,y,nn_parameter ,input_layer_size,hidden_layer_size,label_nums);
fprintf('The cost at weights is (without lambda=0): %f3.2\n',J);
lambda =1;
[J,~] = nnCostFunction(X,y,nn_parameter ,input_layer_size,hidden_layer_size,label_nums,lambda);
fprintf('The cost at weights is (without lambda=1): %f3.2\n',J);

%% 3.随机初始化参数
fprintf('Initializing parameters ....\n');
theta1 = randInitializeWeights(input_layer_size,hidden_layer_size);
theta2 = randInitializeWeights(hidden_layer_size,label_nums);
initial_theta = [theta1(:) ; theta2(:)];

fprintf('\n Checking Backpropagation... \n');
% 作者提供的检测梯度函数是否正确的函数，意义不明
checkNNGradients;
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% 4.训练神经网络
fprintf('\nTraining Neural Network... \n');
lambda = 1;
costFunction = @(p) nnCostFunction(X,y,p,input_layer_size, ...
                                   hidden_layer_size, ...
                                   label_nums,lambda);
options = optimset('MaxIter', 50);
[nn_params, cost] = fmincg(costFunction, initial_theta , options);
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 label_nums, (hidden_layer_size + 1));             

%% 5.可视化中间层参数和预测
fprintf('\nVisualizing Neural Network... \n')
figure(2);
% 因为Theta1的列数与数据样本的列数相同，所以可以使用相同的显示函数绘制图像
displayData(Theta1(:, 2:end));
pred = predict_nn(X, Theta1, Theta2);
title(sprintf('\n The wights with training set accuracy: %f\n', mean(double(pred == y)) * 100));
hold off;
