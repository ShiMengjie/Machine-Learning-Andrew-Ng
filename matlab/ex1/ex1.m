%% Machine Learning Online Class - Exercise 1: Linear Regression with UniVariable
close all;clc

%% 1.读入数据并绘制图像
fprintf('Loading and plot data......\n');
%读入数据
dataSet = load('ex1data1.txt');
%把数据分成输入变量和输出变量
inputVariable = dataSet(:,1);
outputVariable = dataSet(:,2);
plotData(inputVariable,outputVariable);
xlabel('Population of City in 10,000s');
ylabel('Profit of City in $10,000');

%% 2.计算代价和梯度
fprintf('Compute Cost and Gradient Descent......\n');
%对输入数据进行添加一个偏置维度，值全是1，并把数据X写作n*1的形式，n是inputVariable的维度
X=[ones(length(inputVariable),1),inputVariable];
X=X.';
Y=outputVariable.';
% 定义参数 -- theta、alpha、num_itera
theta = zeros(size(X,1),1);
alpha = 0.01;
num_itera = 1500;
% 梯度下降法
[theta_Final , thetaAll] = gradientDescent(X,Y,theta,alpha,num_itera);

fprintf('Theta found last is :\n');
fprintf('%f\n%f\n',theta_Final(1),theta_Final(2));
hold on;
plot(X(2,:),theta_Final.' * X,'b-','LineWidth',2);
legend('Train data','Linear regression');
hold off;

%% 3.预测和可视化代价J
fprintf('Predicting the profits of population = 35,000 and 70,000 ...\n')
% 预测人口数为 35,000 和 70,000 的利润
predict1 = [1, 3.5] * theta_Final ;
fprintf('For population = 35,000, we predict a profit of %f\n',...
    predict1*10000);
predict2 = [1, 7] * theta_Final ;
fprintf('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);

%绘制代价函数J(theta_0, theta_1)的三维图像
fprintf('Visualizing J(theta_0, theta_1) ...\n')
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);
J_vals  = visualCostJ(X,Y,theta0_vals,theta1_vals);
figure;
surf(theta0_vals,theta1_vals,J_vals);

% 绘制等高线---椭圆线
figure;
% logspace(a, n, n)：把[a,b]取n等份，然后每一份对应的值为10^linspace(a,b,n)
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20),'ShowText','on');
xlabel('\theta_0'); 
ylabel('\theta_1');
hold on;
plot(theta_Final(1), theta_Final(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
