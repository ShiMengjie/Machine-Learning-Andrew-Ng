%% Machine Learning Online Class - 
% Exercise 5 | Regularized Linear Regression and Bias-Variance
close all;clc
%% Part.1 Load data and visualizing 
fprintf('load data ...\n');
load('ex5data1.mat');
fprintf('visualize data ...\n');
figure(1);
plot(X,y,'rx','MarkerSize',10,'LineWIdth',2);
xlabel('Change in water level (x)','FontWeight','bold');
ylabel('Water flowing out of the dam (y)','FontWeight','bold');
hold on;

%% Part.2 Calculate cost and gradient
X=[ones(size(X,1),1),X];
init_theta = ones(size(X,2),1);
lambda= 0;
[J,grad] = linearRegCostFunction(X,y,init_theta,lambda);
fprintf('The cost at init_theta ([1;1]) is :%f3.2 \n',J);
fprintf('The gradient at init_theta ([1;1]) is :%f3.2\n',grad);

%% Part.3 Train parameters and regression line
lambda = 0;
[theta] = trainLinearReg(X,y,lambda);
plot(X(:,2),X*theta,'--','MarkerSize',10,'LineWidth',2);
hold off

%% Part.4 Learning curve
% 随着训练样本数目的增加，训练误差是增大的
% 因为模型太过简单，对训练数据和验证数据都是欠拟合，所以随着样本数目增加bias增大，验证误差也很大
Xval = [ones(size(Xval,1),1),Xval];
[error_train,error_val] = learningCurve(X,y,Xval,yval,lambda);
figure(2);
plot(1:size(X,1),error_train,1:size(X,1),error_val,'LineWidth',2);
xlabel('Number of training examples','FontWeight','bold');
ylabel('Error','FontWeight','bold');
legend({'Train', 'Cross Validation'},'FontSize',12,'TextColor','black');
axis([0 13 0 150]);
hold off

%% Part.5 Polynomial Regression 、Normalize and Learning Curve (lambda =0)
% 线性模型简单，所以增加特征的阶数，把特征映射到高维高阶
p=8;
[X_poly]= polyFeatures(X(:,2),p);
% 映射到高维高阶后，数据跨度太大，要进行标准化
[X_poly,mu,sigma] = featureNormalize(X_poly);
X_poly = [ones(size(X_poly,1),1),X_poly];

% 对标准化后的多项式特征进行训练
lambda = 0;
[theta] = trainLinearReg(X_poly,y,lambda);
% 绘制拟合曲线
[x_fit,y_fit] = plotFit(min(X(:,2)),max(X(:,2)),mu,sigma,theta,p);
figure(3);
plot(X(:,2),y,'rx','MarkerSize', 10,'LineWidth',2);
hold on
plot(x_fit,y_fit,'--','LineWidth',2);
xlabel('Change in water level (x)','FontWeight','bold');
ylabel('Water flowing out of the dam (y)','FontWeight','bold');
title (sprintf('Polynomial Regression Fit (lambda = %3.2f)', lambda));
legend({'Data','Polynomial Line'},'FontSize',12,'TextColor','black');
hold off;

X_val_poly = polyFeatures(Xval(:,2),p);
X_val_poly = bsxfun(@minus,X_val_poly,mu);
X_val_poly = bsxfun(@rdivide,X_val_poly,sigma);
X_val_poly = [ones(size(X_val_poly,1),1),X_val_poly];
[error_train,error_val]=learningCurve(X_poly,y,X_val_poly,yval,lambda);
figure(4);
% 可以看出，随着样本数目增加，训练误差基本没有变化，因为多项式能很好的拟合已有的数据
% 但是验证误差依然很大，因为多项式是过拟合的
plot(1:size(X,1), error_train, 1:size(X,1), error_val,'LineWidth',2);
title(sprintf('Learning curve for polynomial regression (lambda=%3.2f)',lambda));
legend({'Train', 'Cross Validation'},'FontSize',12,'TextColor','black');
xlabel('Number of training examples','FontWeight','bold');
ylabel('Error','FontWeight','bold');
axis([0 13 0 100]);
hold off

%% Part.5 Polynomial Regression 、Normalize and Learning Curve (lambda =1)
% 令lambda=1，增加正则项
lambda =1;
[theta] = trainLinearReg(X_poly,y,lambda);
% 绘制拟合曲线
[x_fit,y_fit] = plotFit(min(X(:,2)),max(X(:,2)),mu,sigma,theta,p);
figure(5);
plot(X(:,2),y,'rx','MarkerSize', 10,'LineWidth',2);
hold on
plot(x_fit,y_fit,'--','LineWidth',2);
xlabel('Change in water level (x)','FontWeight','bold');
ylabel('Water flowing out of the dam (y)','FontWeight','bold');
title (sprintf('Polynomial Regression Fit (lambda = %3.2f)', lambda));
legend({'Data','Polynomial Line'},'FontSize',12,'TextColor','black');
hold off;
% 刚开始训练样本少时，训练误差较大，说明正则项限制了过拟合、
% 随着样本数目的增加，训练误差减小，说明在不断求解最优参数
% 验证误差随着样本数目的增加不断减小，说明正则项发挥作用，限制了过拟合验证集的误差没有增大
[error_train,error_val] = learningCurve(X_poly,y,X_val_poly,yval,lambda);
figure(6);
plot(1:size(X,1), error_train, 1:size(X,1), error_val,'LineWidth',2);
title(sprintf('Learning curve for polynomial regression (lambda=%3.2f)',lambda));
legend({'Train', 'Cross Validation'},'FontSize',12,'TextColor','black');
xlabel('Number of training examples','FontWeight','bold');
ylabel('Error','FontWeight','bold');
axis([0 13 0 100]);
hold off

%% Part.6 Validation for Selecting Lambda
% 绘制不同lambda值下的学习曲线
[lambda_vec,error_train,error_val] = validationCurve(X_poly,y,X_val_poly,yval);
figure(7);
plot(lambda_vec,error_train,lambda_vec,error_val,'LineWidth',2);
xlabel('lambda');
ylabel('Error');
hold on 
%% Part.7 Test error
X_poly_test = polyFeatures(Xtest,p);
X_poly_test = bsxfun(@minus,X_poly_test,mu);
X_poly_test = bsxfun(@rdivide,X_poly_test,sigma);
X_poly_test = [ones(size(X_poly_test,1),1),X_poly_test];
[lambda_vec,~,error_val] = validationCurve(X_poly,y,X_poly_test,ytest);
plot(lambda_vec,error_val,'LineWidth',2);
title('Error for Train 、Validation and Test set');
legend({'Train','Validation','Test'},'FontSize',12,'TextColor','black');
