function [theta_final,J_his] = gradientDescentMulti(X, Y, theta, alpha, num_iters)
%% 函数功能：多维数据的梯度下降法
J_his = zeros(num_iters,1);
for i =1:num_iters
    delta = (X*theta -Y).' * X;
    theta = theta - (delta.') * alpha/size(X,1);
    J_his(i) = computeCostMulti(X,Y,theta);
end
theta_final = theta;
end