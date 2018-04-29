%% Machine Learning Online Class - Exercise 3 : Neural Networks
% 使用简单的神经网络来实现“手写数字识别“，网络只有3层：输入层、中间隐藏层、输出层
% 中间隐藏层有25个unit，外加1个值为1的偏置unit
% 输出层有10个输出unit，对应数字的10个类别
close all;clc

%% 1.读取数据，显示随机样例
fprintf('Load data and visulizing .... \n');
load('ex3data1.mat');

index = randperm(size(X,1));
figure(1);
displayData(X(index(1:100),:));
title('Random Examples');
hold off;

%% 2.读取训练好的参数
% 载入训练好的参数：
% Thtea1：对应隐藏层每一个节点的输入参数
% Theta2：对应输出层每一个unit的参数
fprintf('Load parameters Theta1 and Thetha2 ....\n');
load('ex3weights.mat');

%% 3.进行预测
fprintf('Implement prediction ....\n');
% 这里的参数是训练好的，所以可以直接用来预测，造成了不需要进行训练的假象
P = predict_nn(X,Theta1,Theta2);

fprintf('The training set accuracy is %f\n',mean(double(P==y))*100);

% 随机取出X中的数据，参看对它的预测结果
figure(2);
for i =1 : size(X,1)
    fprintf('Displaying the random example ... \n');
    displayData(X(index(i),:));
    
    fprintf('The ture digit is %d ,the Neural Networks prediction : %d (digit %d)',mod(y(index(i)),10),mod(P(index(i)),10));
    
    fprintf('Program paused. Pleasr enter any key to continue.\n');
    pause;
end
