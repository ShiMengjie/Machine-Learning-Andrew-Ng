function [X_poly] = polyFeatures(X,p)
%% 函数功能：把输入的维度投影成高阶多项式
X_poly=zeros(size(X,1),p);
for i = 1:p
    X_poly(:,i) = X .^ i;
end
end
