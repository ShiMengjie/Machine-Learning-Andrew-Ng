function sim = gaussianKernel(x1, x2,sigma)
%% 高斯核函数
x1 = x1(:); x2 = x2(:);

p = (x1-x2).' * (x1-x2);
q = 2*(sigma.' * sigma);

sim = exp(-p/q);

end
