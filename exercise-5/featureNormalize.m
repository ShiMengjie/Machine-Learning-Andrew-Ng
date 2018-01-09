function [X_norm,mu,sigma] = featureNormalize(X)
%% 函数说明：标准化特征数据数据
mu = mean(X);

X_norm = bsxfun(@minus, X, mu);

sigma = std(X_norm);

X_norm = bsxfun(@rdivide, X_norm, sigma);
end
