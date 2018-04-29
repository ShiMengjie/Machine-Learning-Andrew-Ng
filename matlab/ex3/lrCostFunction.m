function [Jcost,grad] = lrCostFunction(X,Y,theta,lambda)
%% 函数功能：计算代价函数和梯度值
[m,~]=size(X);
hypthesis = sigmoid(X * theta);

%有正则项的代价函数表达式
Jcost = ( -Y.' * log(hypthesis) - (1-Y.') * log(1-hypthesis) ) / (m) + lambda * (theta.' * theta)  / (2*m) ;

grad =( (X.') * (hypthesis - Y) +lambda * [0;theta(2:end)] ) / m;

end
