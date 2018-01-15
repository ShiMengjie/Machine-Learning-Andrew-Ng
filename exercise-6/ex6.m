%% Machine Learning Online Class 
% Exercise 6---Support Vector Machine
close all;clc
addpath(genpath('./kernel'));
%% 载入数据，绘制样本数据
fprintf('Loading and visualizing data ....\n');
load('ex6data1.mat');
figure(1);
plotData(X,y);
xlabel('Feature 1');
ylabel('Feature 2');

%% 训练SVM

svm