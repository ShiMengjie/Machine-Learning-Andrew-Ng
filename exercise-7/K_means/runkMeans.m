function [centroids, idx] = runkMeans(X, init_centroids,max_iter, isplot)
%% 绘制K-means计算过程
%% 判断是否要绘制图像
if ~exist('isplot','var') || isempty(isplot)
    isplot = false;
end

%% 初始化参数
K = size(init_centroids,1);
centroids = init_centroids;
previous_centroids = centroids;
idx = zeros(size(X,1),1);

if isplot
    figure;
    hold on;
end
%% 进行K-means计算过程
% 在每一次迭代过程中，计算每个数据所属的簇，再计算新的中心坐标
for i=1:max_iter
    fprintf('K-Means iteration %d/%d...\n', i, max_iter);
    % K-means一共有两步：
    % step1：
    idx = findClosestCentroids(X, centroids);
    % 绘制每个一步的过程
    if isplot
        % 绘制数据点
        plotDataPoints(X, idx, K);
        hold on
        plotProgresskMeans(centroids, previous_centroids);
        fprintf('Press enter to continue.\n');
        pause;
    end
    previous_centroids = centroids;
    % step2：
    centroids = computeCentroids(X, idx, K);
end

if isplot
    hold off;
end

end
