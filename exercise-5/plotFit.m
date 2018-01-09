function [x,y_poly]=plotFit(min_x,max_x,mu,sigma,theta,p)
%% 函数说明：使用训练出来的theta来绘制拟合曲线
x = (min_x -15:0.05:max_x+25).';

x_poly = polyFeatures(x,p);
x_poly = bsxfun(@minus,x_poly,mu);
x_poly = bsxfun(@rdivide,x_poly,sigma);

x_poly = [ones(size(x_poly,1),1),x_poly];
y_poly = x_poly * theta;

end
