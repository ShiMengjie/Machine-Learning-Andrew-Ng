function J = computeCost( X,Y,theta )
%% 函数功能：计算误差平方和作为代价值
%  X：增加了一个维度后的输入特征
%  J：X对应的输出值
%  theta：带入的参数
[~,m]=size(X);
hypthesis = theta.' * X;
err = hypthesis - Y;
J = (err * err.') / (2*m);
end
