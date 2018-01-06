function [Jcost,grad] = lrCostFunction(theta,X,Y,lambda)
%% 计算代价函数和梯度值
% X:m*n    Y:m*1   theta:n*1 
[m,~]=size(X);
hypthesis = sigmoid(X*theta);

%有正则项的代价函数表达式
Jcost = ( -Y.' * log(hypthesis) - (1-Y.') * log(1-hypthesis) ) / (m) + lambda  / (2*m) * (theta.' * theta);

grad =( (X.') * (hypthesis - Y) +lambda * [0;theta(2:end)] ) / m;

end
