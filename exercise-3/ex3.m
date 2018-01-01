%% Machine Laearning Online Class - Exercise 3: One-vs-All Logistic Regression
close all;clc
%% =================== Part.1 Inilitzation and Load/Visualizing Data =========================
fprintf('Load dataSet:ex3data1.mat\n');
load('ex3data1.mat');
%从数据中随机选择100行数据，显示成10*10的数字表格
%生成[1,m]的随机数据序列，只取100个即可
randIndex = randperm(size(X,1));
seldata = X(randIndex(1:100),:);
%数据可视化
figure(1);
displayData(seldata);
title('Random handwritten digits');
hold off;

%% =================== Part.2 Vectorizing Logistic Regression =============================
%  向量化Logisyic Regression，并计算代价函数和梯度值
fprintf('Training data with one-Vs-All logistic regression\n');

lambda = 0.1;
num_labels=10;
[all_theta] = oneVsAll(X,y,num_labels,lambda);

%% =================== Part.3 Predict for oneVsAll =============================
P=predictOneVsAll(all_theta,X);
fprintf('Training set Accuracy is %f\n',mean(double(P==y))*100);
%我不明白，为什么我的模型在训练集上的准确率是94.76%，达不到94.9%

