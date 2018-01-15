function [K] = compute_kernel_matrix(fun,X,Z)
%% 计算核函数矩阵K，K(i,j)=k(X(i,:),Z(j,:)),K的作用相当于方差矩阵

m = size(X,1);
n = size(Z,1);
K = zeros(m,n);
for i = 1:m
    for j = 1:n
        K(i,j) = fun(X(i,:).', Z(j,:).');
    end
end

end