function [theta_Final , thetaAll] = gradientDescent ( X,Y,theta_init,alpha,num_itera )
%% 函数功能：梯度下降法求解theta值
%  X：输入
%  Y：对应输出
%  alpha：学习率/步长
%  theta_Init：theta初始值
%  num_itera：迭代计算次数
%%
[n,m]=size(X);
theta = theta_init;

thetaAll = zeros(n,num_itera);
thetaAll(:,1) = theta_init;

J = zeros(num_itera,1);
% 保存每一次迭代的代价值，并更新theta
for i = 1:num_itera
    J(i) = computeCost(X,Y,theta);
    theta = theta - alpha * (1/m) * (X * (theta.' * X - Y).');
    thetaAll(:,i) = theta;
end
theta_Final = theta;
end

