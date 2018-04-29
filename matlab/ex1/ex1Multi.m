%% Machine Learning - Linear Regression with Multi-Variables
close all;clc
%% 1.读取数据并标准化特征值
fprintf('Loading data ......\n');
dataSet = load('ex1data2.txt');
X = dataSet(:,1:2);
Y = dataSet(:,3);
% 进行特征归一化
[X_norm,mu,sigma] = featureNormalize(X);
% fprintf('mu is %f\n',mu);
fprintf('sigma is %f\n',sigma);
% 添加偏置
X = [ones(size(X_norm,1),1) , X_norm];

%% 2.使用梯度下降法优化求解
theta = zeros(size(X,2),1);
num_iters=150;
element=['r-','b-','g-','k-','m-','r.','b.','g.','k.','m.'];
alpha = [0.001,0.005,0.01,0.05,0.1,0.5];
figure;
for i = 1:length(alpha)
    [theta_final,J] = gradientDescentMulti(X,Y,theta,alpha(i),num_iters);
    plot(J,element(i),'LineWidth',2);
    hold on
end
% alpha=0.5的时候，收敛很快，选择此时的theta作为最终的theta值
xlabel('Iterator Number');
ylabel('CostNum J');
title('Cost J at diferent \alpha');
legend('\alpha=0.001','\alpha=0.005','\alpha=0.01','\alpha=0.05','\alpha=0.1','\alpha=0.5');

%% 3.预测
Xtest = [1,1650,3];
price = Xtest * theta_final;
fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);
