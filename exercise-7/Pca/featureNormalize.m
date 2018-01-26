function [X_norm, mu, sigma] = featureNormalize(X)
%% 对数据进行标准化，并返回
% 这种写法，是Mattlab标准写法
mu = mean(X);
X_norm = bsxfun(@minus, X, mu);
sigma = std(X_norm); % 得到标准差
X_norm = bsxfun(@rdivide, X_norm, sigma);

end
