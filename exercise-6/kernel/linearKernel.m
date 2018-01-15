function sim = linearKernel(x1,x2)
%% 返回x2和x2的线性核--内积

% 保证x1和x2都是列向量
x1 = x1(:); 
x2 = x2(:);

sim = x1.' * x2;

end
