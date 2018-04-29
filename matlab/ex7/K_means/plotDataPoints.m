function plotDataPoints(X, idx, K)
%% 函数功能：绘制数据点
% 生成K+1种颜色的色彩
palette = hsv(K);
% 根据idx取出每个样本对应的颜色
colors = palette(idx,:);
% 绘制数据散点图
scatter(X(:,1),X(:,2),15,colors,'LineWidth',2);

end
