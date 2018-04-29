function J = computeCostMulti( X,Y,theta )
%% 函数功能：计算多维数据的代价
[m,~] = size(X);
hypthesis = X * theta;
err = hypthesis - Y;
J = (err.'*err) / (2*m);
end