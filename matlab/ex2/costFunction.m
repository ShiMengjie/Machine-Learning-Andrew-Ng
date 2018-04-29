function [ cost,grad ] = costFunction( theta,X,Y )
%% 函数功能：根据输入的参数theta，数据的特征X，数据的输出Y，计算出此时的代价和梯度
%需要注意的是，logistic的代价函数不是最小均方差，而是似然函数
[m,~]=size(X);
z = sigmoid(X * theta);

cost = ( -Y.' * log(z) - (1 - Y.') * log(1 - z)) / m;

grad = - (Y.' - (z).') * X / m;

end

