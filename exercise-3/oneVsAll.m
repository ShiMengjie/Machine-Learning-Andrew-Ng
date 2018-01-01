function [all_theta] = oneVsAll(X,Y,num_labels,lambda)
%% 对多个类别的数据同时进行训练，迭代计算出每个类别的训练模型参数theta，以及它们的集合all_theta
% num_labels：数据所属的类别的个数
[m,n] = size(X);

all_theta = zeros(num_labels,n+1);
X=[ones(m,1) , X];
options = optimset('GradObj', 'on', 'MaxIter', 50);
% 和前面缩写的求解梯度的最大区别在于，这里是对多个theta向量进行求解，前面是对单个theta向量进行求解，所以需要加一个for循环
for c =1:num_labels
    init_theta = zeros(n+1,1);
    %fmincg的用法和fminunc类似，但在处理大量参数时，更高效
    [theta] = fmincg(@(t) (lrCostFunction(t,X, (Y == c), lambda)) , init_theta , options);  
    all_theta(c,:) = theta.';
end

        
    
    
   

