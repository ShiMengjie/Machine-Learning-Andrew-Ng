function p = multivariateGaussian(X, mu, sigma2)
%% 函数功能：计算每个样本在多维高斯分布中的概率值
% 生成方差矩阵
if (size(sigma2, 2) == 1) || (size(sigma2, 1) == 1)
    sigma2 = diag(sigma2);
end

k= length(mu);
X = bsxfun(@minus,X,mu);

p = (2*pi) ^ (-k/2) * det(sigma2) ^ (-0.5) * ...
    exp(-0.5 * sum( bsxfun(@times,X * pinv(sigma2),X),2));

end
