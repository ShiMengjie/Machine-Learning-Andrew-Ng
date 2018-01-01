function P = predictOneVsAll(all_theta,X)
%% 函数说明：对数据进行预测
X=[ones(size(X,1),1),X];
z=X * all_theta.';
h=sigmoid(z);

% h是一个m*K的矩阵，h(m,k)的值表示第m个数据预测属于第k类的概率
[~,P]=max(h,[],2);
end
