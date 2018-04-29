function [y] = sample_gp_prior(kernel, X)
%% 根据核函数矩阵，生成随机值
  K = compute_kernel_matrix(kernel, X, X);
  
  % 为什么要进行SVD分解呢？？
  [U,S,V] = svd(K);
  A = U*sqrt(S)*V';
  y = A * randn(size(X,1),1);

end
