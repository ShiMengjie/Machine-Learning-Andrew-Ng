function plotProgresskMeans( centroids, previous)
%% 函数功能：用短直线绘制K-means寻找新的中心点每一步过程
% 用黑色x表示中心点
plot(centroids(:,1),centroids(:,2),'x','MarkerEdgeColor','k','MarkerSize',10,'LineWidth',3);
% 用短直线连接上一步的中心点与当前中心点
for j=1:size(centroids,1)
    drawLine(centroids(j, :), previous(j, :));
end

end
