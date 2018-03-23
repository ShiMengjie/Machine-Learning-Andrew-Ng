function [X_norm,mu,sigma] = featureNormalize(X)
%% 函数功能：把特征标准化，下面是标准化的标准流程
mu = mean(X);

X_norm = bsxfun(@minus, X, mu);

sigma = std(X_norm);

X_norm = bsxfun(@rdivide, X_norm, sigma);
end
