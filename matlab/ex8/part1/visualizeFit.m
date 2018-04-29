function visualizeFit(mu, sigma2)
%% 函数功能：绘制数据分布的概率等高线
[X1,X2] = meshgrid(0:.5:35); 
% 带入前面估计出的参数，计算每个网格点处的多元高斯分布
Z = multivariateGaussian([X1(:),X2(:)],mu,sigma2);
Z = reshape(Z,size(X1));

% 如果存在无穷大或无穷小的值，就不绘制图形
if ( sum(isinf(Z)) == 0 )
    contour(X1,X2,Z,10.^(-20:2:0));
end

end
