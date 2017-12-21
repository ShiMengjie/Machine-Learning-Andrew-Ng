function [theta_Final , thetaAll] = gradientDescent ( X,Y,theta_Init,alpha,num_itera )
%%
%  X：输入
%  Y：对应输出
%  alpha：学习率/步长
%  theta_Init：theta初始值
%  num_itera：迭代计算次数
%%
[n,m]=size(X);
theta = theta_Init;
J = zeros(num_itera,1);
J(1) = computeCost(X,Y,theta);

thetaAll = zeros(n,num_itera);
thetaAll(:,1) = theta_Init;

fprintf('The J:%f\n',J(1));
for i = 2:1:num_itera
    theta = theta - alpha*(1/m) * (X*(theta.'*X - Y).');
    thetaAll(:,i) = theta;
    %代价函数
    J(i,1) = computeCost(X,Y,theta);
    fprintf('The J:%f\n',J(i));
end
theta_Final = theta;
end

