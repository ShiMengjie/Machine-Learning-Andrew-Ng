%% Machine Learning Online Class - Exercise 2: Logistic Regression
% 使用logistic进行分类
close all;clc

%% 1.读取数据
fprintf('Loading and plot data......\n');
%读入数据
dataSet = load('ex2data1.txt');
X = dataSet(:,1:2);
Y = dataSet(:,3);
figure;
plotData(X,Y);
xlabel('Exam1 score');
ylabel('Exam2 score');
legend('Admiited','Not admiited');
hold off;

%% 2.计算代价和梯度
X = [ones(size(X,1),1),X];
initTheta = zeros(size(X,2),1);

[cost,grad] = costFunction(initTheta,X,Y);
fprintf('The cost value at initThtea (zeros) is : %f\n ',cost);
fprintf('The gradient value at initThtea (zeros) is : %f\n ',grad);

%% 3.优化求解
% fminunc函数功能：
% 找到一个多变量函数的最小值，从一个估计的初试值开始，通常用来优化无约束非线性问题
% x =fminunc(fun,x0,options)    根据结构体options中的设置来找到最小值，可用optimset来设置options
% [x,fval]= fminunc(...)    返回目标函数fun在解x处的函数值fval

%使用自定义的梯度函数，迭代次数为400次
maxIter = 400;
J = zeros(maxIter,1);
theta = initTheta;
options = optimset('GradObj', 'on', 'MaxIter', 1);
costFun = @(the)costFunction(the,X,Y);
for i =1:maxIter
    [theta,cost] = fminunc(costFun,theta,options);
    J(i) = cost;
end

figure(2);
plot((1:maxIter),J,'b-','LineWidth',2);
xlabel('Iterator number');
ylabel('Cost value');

fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

%绘制分界线
figure(3);
plotDecisionBoundary( theta,X,Y );

xlabel('Exam1 score');
ylabel('Exam2 score');
legend('Admitted', 'Not admitted', 'Decision Boundary')
hold off;

%% 4.预测
prob = sigmoid([1 45 85] * theta);
fprintf(['For a student with scores 45 and 85, we predict an admission ' ...
         'probability of %f\n\n'], prob);
P = predict (theta,X);
fprintf('The model train accuracy is: %f\n', mean(double(P == Y))*100);
