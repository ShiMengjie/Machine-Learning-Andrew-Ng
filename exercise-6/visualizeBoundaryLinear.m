function visualizeBoundaryLinear(X, model)
%% 函数功能：绘制SVM训练出来的线性分界线
w = model.w;
b = model.b;
xp = linspace(min(X(:,1)), max(X(:,1)), 100);
yp = - (w(1)*xp + b)/w(2);
plot(xp,yp,'LineWidth',2);

end
