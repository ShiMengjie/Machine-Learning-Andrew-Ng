function [ X_Norm,mu,sigma ] = featureNormalize(X)
%% 函数功能：标准化输入变量X的特征
mu = mean(X,1);
sigma=sqrt(sum((X-mu) .^ 2) / size(X,1));
X_Norm = (X-mu) ./ sigma;
end
