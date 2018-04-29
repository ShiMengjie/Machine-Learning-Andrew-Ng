function [all_theta] = oneVsAll(X,Y,num_labels,lambda)
%% 函数功能：对多个类别的数据同时进行训练，迭代计算出每个类别的训练模型参数theta，返回所有theta的集合all_theta
[m,n] = size(X);

all_theta = zeros(num_labels , n+1);
X=[ones(m,1) , X];
options = optimset('GradObj', 'on', 'MaxIter', 50);

% 和前面所写的计算梯度的最大区别在于，这里是对多个theta向量进行计算，前面是对单个theta向量进行求解，所以需要加一个for循环
% 在计算labe-K的梯度代价和梯度时，是把其他类别都设置成0(Y==k)，转换成两分类来计算
for K =1:num_labels
    init_theta = zeros(n+1,1);
    costFun = @(t) lrCostFunction(X, (Y == K),t, lambda);
    % fmincg的用法和fminunc类似，但在处理大量参数时，更高效
    [theta] = fmincg(costFun , init_theta , options);  
    all_theta(K,:) = theta.';
end

end
