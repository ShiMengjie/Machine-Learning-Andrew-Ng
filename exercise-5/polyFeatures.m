function [X_poly]= polyFeatures(X,p)
%% 函数说明：把X的特征映射成多项式的形式
X_poly=zeros(size(X,1),p);
for i = 1:p
    X_poly(:,i) = X.^i;
end
