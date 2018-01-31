%% Machine Learning Online Class
%  Exercise 8 | Anomaly Detection and Collaborative Filtering
close all;clc
%% 1.导入数据
fprintf('Visualizing example dataset for outlier detection.\n\n');
load('ex8data1.mat');
figure(1);
plot(X(:,1),X(:,2),'bx');
xlabel('Latency(ms)');
ylabel('Throughput(mb/s)');
axis([0 30 0 30]);
hold on;
%% 2.估计数据集的分布模型参数
fprintf('Visualizing Gaussian fit.\n\n');
[mu,sigma2] = estimateGaussian(X);
visualizeFit(X,  mu, sigma2);
hold on;

%% 3.选择门限epsilon
pval = multivariateGaussian(Xval,mu,sigma2);
[epsilon,F1] = selectThreshold(yval,pval);
fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
fprintf('   (you should see a value epsilon of about 8.99e-05)\n\n');

%% 4.找到异常点
p = multivariateGaussian(X, mu, sigma2);
% 找到概率值小于门限的样本
outliers = find(p < epsilon);
plot(X(outliers,1),X(outliers,2),'ro','LineWidth',2,'MarkerSize',10);
hold off;
fprintf('Program paused. Press enter to continue.\n');
pause;

%% 5.高维数据
clear all;clc
load('ex8data2.mat');
% 估计模型参数
[mu,sigma2] = estimateGaussian(X);
% 计算训练集和验证集的数据概率
p = multivariateGaussian(X, mu, sigma2);
pval = multivariateGaussian(Xval, mu, sigma2);
% 寻找门限
[epsilon,F1] = selectThreshold(yval, pval);

fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
fprintf('# Outliers found: %d\n', sum(p < epsilon));