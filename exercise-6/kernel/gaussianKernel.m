function K = gaussianKernel(x1, x2,sigma)
%% 函数功能：高斯核函数
X1 = sum(x1.^2,2);
X2 = sum(x2.^2,2);
% 下面这一句必须这么写 %
sim = bsxfun(@plus,X1,bsxfun(@plus,X2.', - 2 * (x1 * x2.')));
K = exp(-1/(2*sigma*sigma)) .^ sim;
 
end
