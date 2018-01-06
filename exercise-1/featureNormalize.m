function [ X_Norm,mu,sigma ] = featureNormalize(X)
%%函数功能：Feature Normalization
%  归一化输入变量X的特征
%  X_Norm：归一化后的输入变量
%  mu：输入变量的特征均值
%  sigma：输入变量的特征标准差
mu = mean(X,1);
%sigma = std(X,1);
sigma=sqrt(sum((X-mu).^2)/size(X,1));
X_Norm = (X-mu)./sigma;

end

