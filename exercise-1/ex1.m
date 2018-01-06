%% Machine Learning Online Class - Exercise 1: Linear Regression with UniVariable
%  Instructions
%  -------------------------------
%  Attention:
%  This file contains code that helps you get started on the linear exercise. 
%  You will need to complete the following functions in this exericse:
%     plotData.m : plot the data
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
% inputVariable : the population size in 10,000s
% outputVariable : refers to the profit in $10,000s
%
%% Start
clear all;close all;clc
%% ==================== Part 1: Load and Plot Data ====================
fprintf('Loading and plot data......\n');
%读入数据
dataSet = load('ex1data1.txt');
%把数据分成输入变量和输出变量
inputVariable = dataSet(:,1);
outputVariable = dataSet(:,2);
plotData(inputVariable,outputVariable);
xlabel('Population of City in 10,000s');
ylabel('Profit of City in $10,000');
%% ==================== Part 2: Compute Cost and Gradient Descent ====================
fprintf('Compute Cost and Gradient Descent');
%对输入数据进行添加一个维度，值全是1，并把数据X写作n*1的形式，n是inputVariable的维度
X=[ones(length(inputVariable),1),inputVariable];
X=X.';
Y=outputVariable.';
%定义parameters -- theta，n*1的形式
[d,~]=size(X);
theta = zeros(d,1);
alpha = 0.01;
num_itera = 1500;
%Gradient descent
[theta_Final , thetaAll] = gradientDescent(X,Y,theta,alpha,num_itera);

fprintf('Theta found last is :\n');
fprintf('%f\n%f\n',theta_Final(1),theta_Final(2));

hold on;
plot(X(2,:),theta_Final.'*X,'b-','LineWidth',2);
legend('Train data','Linear regression');
hold off;

%% ============= Part 3: Predicting and Visualizing J(theta_0, theta_1) =============
fprintf('Predicting the profits of population = 35,000 and 70,000 ...\n')
% 预测人口数为 35,000 和 70,000 的利润
predict1 = [1, 3.5] *theta_Final ;
fprintf('For population = 35,000, we predict a profit of %f\n',...
    predict1*10000);
predict2 = [1, 7] * theta_Final ;
fprintf('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);
%绘制代价函数J(theta_0, theta_1)
fprintf('Visualizing J(theta_0, theta_1) ...\n')

theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

J_vals = zeros(length(theta0_vals), length(theta1_vals));
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  J_vals(i,j) = computeCost(X, Y, [theta0_vals(i); theta1_vals(j)]);
    end
end
J_vals = J_vals.';
figure;
surf(theta0_vals,theta1_vals,J_vals);
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 1000
%logspace(a, n, n)
%对[a,b]取n等份，然后每一份对应的值为10^linspace(a,b,n)
figure;
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20),'ShowText','on');
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta_Final(1), theta_Final(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
