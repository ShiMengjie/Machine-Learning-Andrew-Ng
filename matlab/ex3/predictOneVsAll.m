function P = predictOneVsAll(X,all_theta)
%% 函数说明：对数据进行预测

X=[ones(size(X,1),1),X];
z=X * all_theta.';
h = sigmoid(z);
% h是一个m*K的矩阵，h(m,k)的值表示第m个数据预测属于第k类的概率
% 返回h中，每一列的最大值的下标，就是预测的最可能的类别
[~,P] = max(h,[],2);
end
