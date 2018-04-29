function [x,y_poly]=plotFit(min_x,max_x,mu,sigma,theta,p)
%% 函数功能：在多项式特征值的区间上，绘制这个区间上坐标与theta的线性组合曲线
% 确定坐标范围
x = (min_x -15 : 0.05 : max_x+25).';
% 投影到与特征数据相同的项次
x_poly = polyFeatures(x,p);
x_poly = bsxfun(@minus,x_poly,mu);
x_poly = bsxfun(@rdivide,x_poly,sigma);
% 计算曲线的值
x_poly = [ones(size(x_poly,1),1),x_poly];
y_poly = x_poly * theta;

end
