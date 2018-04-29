function [J,grad]=linearRegCostFunction(X,Y,theta,lambda)
%% 函数功能：计算有正则项的代价函数和梯度
m=size(X,1);
hyp = X * theta - Y;

J = (hyp.' * hyp + lambda * (theta.' * theta)) / (2*m);

grad = (X.' * hyp + lambda * [0;theta(2:end)])/m;

end
