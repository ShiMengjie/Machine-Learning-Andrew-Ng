%% Machine Learning Online Class-Exerceise 5 
% 正则化线性回归和学习曲线：偏差-方差曲线
close all;clc

%% 1.读取和显示数据
fprintf('load data ...\n');
load('ex5data1.mat');
fprintf('visualize data ...\n');
figure(1);
plot(X,y,'rx','MarkerSize',10,'LineWIdth',2);
xlabel('Change in water level (x)','FontWeight','bold');
ylabel('Water flowing out of the dam (y)','FontWeight','bold');
hold on;

%% 2.计算代价和梯度
X=[ones(size(X,1),1),X];
init_theta = ones(size(X,2),1);
lambda= 0;
[J,grad] = linearRegCostFunction(X,y,init_theta,lambda);
fprintf('The cost at init_theta ([1;1]) is :%f3.2 \n',J);
fprintf('The gradient at init_theta ([1;1]) is :[%f3.2 , %f3.2]\n',grad);

%% 3.训练参数和绘制回归曲线
lambda = 0;
[theta] = trainLinearReg(X,y,lambda);
plot(X(:,2),X*theta,'b--','MarkerSize',10,'LineWidth',2);
hold off

%% Part.4 Learning curve
Xval = [ones(size(Xval,1),1),Xval];
[error_train,error_val] = learningCurve(X,y,Xval,yval,lambda);
figure(2);
plot(1:size(X,1),error_train,1:size(X,1),error_val,'LineWidth',2);
xlabel('Number of training examples','FontWeight','bold');
ylabel('Error','FontWeight','bold');
legend({'Train', 'Cross Validation'},'FontSize',12,'TextColor','black');
axis([0 13 0 150]);
hold off
% 从曲线图中可以看出，随着训练样本的增加，训练集误差增大，而验证集的误差减小
% 因为数据分布是非线性的，而模型是线性的，是欠拟合的，偏差Bias必然会随着训练样本的增加而增加
% 同理，在验证集上趋势是下降，这是因为模型虽然在欠拟合，但是程度在减小，在验证集上欠拟合的程度在“减小”，
% 但是误差依然很大

%% 5.投影特征为多项式，再绘制学习曲线，lmda=0
p = 8;
[X_poly]= polyFeatures(X(:,2),p);
% 因为在高次项，特征的值变化尺度大，需要先标准化
[X_poly,mu,sigma] = featureNormalize(X_poly);
X_poly = [ones(size(X_poly,1),1),X_poly];

lambda = 0;
[theta] = trainLinearReg(X_poly,y,lambda);

[x_fit,y_fit] = plotFit(min(X(:,2)),max(X(:,2)),mu,sigma,theta,p);

figure(3);
plot(X(:,2),y,'rx','MarkerSize', 10,'LineWidth',2);
hold on
plot(x_fit,y_fit,'b--','LineWidth',2);
xlabel('Change in water level (x)','FontWeight','bold');
ylabel('Water flowing out of the dam (y)','FontWeight','bold');
title (sprintf('Polynomial Regression Fit (lambda = %3.2f)', lambda));
legend({'Data','Polynomial Line'},'FontSize',12,'TextColor','black');
hold off;
% 对验证集进行同样的投影，并使用训练集标准化的参数来标准化验证集数据
X_val_poly = polyFeatures(Xval(:,2),p);
X_val_poly = bsxfun(@minus,X_val_poly,mu);
X_val_poly = bsxfun(@rdivide,X_val_poly,sigma);
X_val_poly = [ones(size(X_val_poly,1),1),X_val_poly];
[error_train,error_val] = learningCurve(X_poly,y,X_val_poly,yval,lambda);
figure(4);
plot(1:size(X,1), error_train, 1:size(X,1), error_val,'LineWidth',2);
title(sprintf('Learning curve for polynomial regression (lambda=%3.2f)',lambda));
legend({'Train', 'Cross Validation'},'FontSize',12,'TextColor','black');
xlabel('Number of training examples','FontWeight','bold');
ylabel('Error','FontWeight','bold');
axis([0 13 0 100]);
hold off
% 从拟合曲线可以看出，高次曲线完全拟合训练集，所以在学习曲线中训练集误差一直很小
% 但是验证集误差还是很高，因为是过拟合，所有有一个很大的方差Variance

%% 6.投影特征为多项式，再绘制学习曲线 (lambda =1)
lambda =1;
[theta] = trainLinearReg(X_poly,y,lambda);

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

[error_train,error_val] = learningCurve(X_poly,y,X_val_poly,yval,lambda);
figure(6);
plot(1:size(X,1), error_train, 1:size(X,1), error_val,'LineWidth',2);
title(sprintf('Learning curve for polynomial regression (lambda=%3.2f)',lambda));
legend({'Train', 'Cross Validation'},'FontSize',12,'TextColor','black');
xlabel('Number of training examples','FontWeight','bold');
ylabel('Error','FontWeight','bold');
axis([0 13 0 100]);
hold off
% 从拟合曲线上可以看出，与前面正则项系数lambda=0相比，此时的拟合程度较低
% 在学习曲线上，训练集的误差曲线增加，验证集曲线却下降到比训练集还低，因为在训练集上没有过拟合，泛化性能更好

%% 7.通过验证集选择lambda，计算测试集误差
[lambda_vec,error_train,error_val] = validationCurve(X_poly,y,X_val_poly,yval);
figure(7);
plot(lambda_vec,error_train,lambda_vec,error_val,'LineWidth',2);
xlabel('lambda');
ylabel('Error');
hold on 
% 投影和标准化测试集数据
X_poly_test = polyFeatures(Xtest,p);
X_poly_test = bsxfun(@minus,X_poly_test,mu);
X_poly_test = bsxfun(@rdivide,X_poly_test,sigma);
X_poly_test = [ones(size(X_poly_test,1),1),X_poly_test];
[lambda_vec,~,error_test] = validationCurve(X_poly,y,X_poly_test,ytest);
plot(lambda_vec,error_test,'LineWidth',2);
title('Error for Train+Validation and Test set');
legend({'Train','Validation','Test'},'FontSize',12,'TextColor','black');
