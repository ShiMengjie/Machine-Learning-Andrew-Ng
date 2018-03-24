function [mu,sigma2] = estimateGaussian(X)
%% 函数功能：假设数据是服从高斯分布的，来估计高斯分布的参数
[m, ~] = size(X);

mu = sum(X) / m;
sigma2 = sum( bsxfun(@minus,X,mu) .^ 2) / (m-1);

end
