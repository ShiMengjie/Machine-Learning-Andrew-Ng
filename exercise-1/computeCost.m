function J = computeCost( X,Y,theta )
%%
%  X：增加了一个维度后的输入特征
%  T：X对应的输出值
%  theta：每次带入的参数
[~,m]=size(X);
hypthesis = theta.'*X;
err = hypthesis -Y;
J = (err*err.') /(2*m);
end

