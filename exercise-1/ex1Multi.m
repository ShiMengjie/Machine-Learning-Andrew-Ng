%% Machine Learning - Linear Regression with Multi-Variables
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear regression exercise. 
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%
%% ================ Part 1: Load Data and Feature Normalization ================
close all;clc
fprintf('Loading data ......\n');
dataSet = load('ex1data2.txt');
%输入的特征变量
X = dataSet(:,1:2);
%输入的特征变量对应的输出
Y = dataSet(:,3);
%进行特征归一化
[X_norm,mu,sigma] = featureNormalize(X);
X = [ones(size(X_norm,1),1),X_norm];
%% ================ Part 2:  Gradient Descent ================
theta = zeros(size(X,2),1);
num_iters=150;
element=['r-','b-','g-','k-','m-','r.','b.','g.','k.','m.'];
alpha = [0.001,0.005,0.01,0.05,0.1,0.5];
figure;
for i = 1:6
    [theta_final,J] = gradientDescentMulti(X,Y,theta,alpha(i),num_iters);
    plot(J,element(i),'LineWidth',2);
    hold on
end
%alpha=0.5的时候，收敛很快，选择此时的theta作为最终的theta值
xlabel('Iterator Number');
ylabel('CostNum J');
title('Cost J at diferent \alpha');
legend('\alpha=0.001','\alpha=0.005','\alpha=0.01','\alpha=0.05','\alpha=0.1','\alpha=0.5');

%% ============= Part 3: Predicting  =============
price = [1,1650,3]*theta_final;

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);
