function [J, grad] = costFunctionReg(theta, X, Y, lambda)
%% 计算具有正则项的代价函数和梯度值，其中：
%  lambda是正则项的系数
[m,n]=size(X);
[cost1,grad1] = costFunction(theta,X,Y);
J= cost1  + lambda/(2*m) * (theta.'*theta);

grad = zeros(n,1);
grad(1) = grad1(1);
grad(2:n) = grad1(2:n).' + lambda/m.*theta(2:n);

end


