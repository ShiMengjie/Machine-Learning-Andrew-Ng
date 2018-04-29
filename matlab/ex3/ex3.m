%% Machine Laearning Online Class - Exercise 3: One-vs-All Logistic Regression
% 多个Logistic Regression同时运算，用来对手写数字进行识别
close all;clc
%% 1.读取数据和初始化，
fprintf('Load dataSet:ex3data1.mat\n');
load('ex3data1.mat');
% 从数据中随机选择100个样本，显示成10*10的数字表格
randIndex = randperm(size(X,1));
seldata = X(randIndex(1:100),:);
% 数据可视化
figure(1);
displayData(seldata);

%% 2.向量化Logistic Regression，并计算代价函数和梯度值
fprintf('Training data with one-Vs-All logistic regression\n');
% 设置正则化系数lambda，和输出的样本类别数labels
lambda = 0.05;
num_labels=10;
[all_theta] = oneVsAll(X,y,num_labels,lambda);

%% 3.使用OneVsAll来预测
P=predictOneVsAll(X,all_theta);
fprintf('Training set Accuracy is %f\n', mean(double(P==y)) * 100);
