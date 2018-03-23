function [U,S] = pca(X)
%% 函数功能：使用SVD进行PCA分解
Sigma = X.' * X / (size(X,1));
[U, S, ~] = svd(Sigma);

end
