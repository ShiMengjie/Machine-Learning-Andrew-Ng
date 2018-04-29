%% Machine Learning Online Class - Exercise 2: Regularized Logistic Regression
close all;clc

%% 1.读取数据和绘制散点图
dataSet = load('ex2data2.txt');
X = dataSet(:,1:2);
Y = dataSet(:,3);
figure(1);
plotData(X,Y);
xlabel('Microchip test 1');
ylabel('Microchip test 2');
title('Visualizing data');
legend('Pass (y=1)','Not Pass (y=0)');

%% 2.映射特征到高阶维度，计算带有正则项的代价和梯度
% 从散点图中可以看出，两类数据不是线性可分的，那么把数据的特征向高维映射，形成特征[x1,x2]的多项式
%映射到特征的6次方
power = 6;
X = mapFeature(X(:,1),X(:,2),power);
% 此时，数据维度从2维增加到28维，会出现另一个问题：维度太高导致分类器容易过拟合，所以要在代价函数中添加正则化
init_Theta = zeros(size(X,2),1);
%正则项系数
lambda =1;
%计算正则项的代价函数和梯度值
[cost,grad] = costFunctionReg(init_Theta,X,Y,lambda);

fprintf('The cost value at init_Theta (zeros) is: %f\n',cost);
fprintf('The gradient value at init_Theta (zeros) is: %f\n',grad);

%% 3.优化求解并绘制分界线
%最小化代价值
maxIter = 400;
options = optimset('GradObj','on','MaxIter',maxIter);
costFun = @(the)costFunctionReg(the,X,Y,lambda);
for lambda =0 : 0.1 : 1
    [theta,cost] = fminunc(costFun , init_Theta ,options);
    %绘制分界线
    figure;
    plotDecisionBoundary(theta,X,Y);
    hold on;
    title(sprintf('lambda = %g',lambda));
    xlabel('Microchip Text 1');
    ylabel('Microchip Text 2');
    legend('y = 1','y = 0','Decision boundary');
    hold off;
end

% 计算在训练集上的准确率
P = predict(theta,X);
fprintf('The accuracy in training data is %f\n',mean(double(P==Y))*100);