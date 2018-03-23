function [centroids, idx] = runkMeans(X, init_centroids,max_iter, isplot)
%% 函数功能：进行K-means运算，并绘制K-means计算过程

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
% K-means可以分为3步：
% 1.判断簇的个数K，随机初始化每个簇的中心坐标，并把每个样本分到所属的簇
% 2.用每个簇中所有样本的均值来更新簇的坐标
% 3.用新的簇中心坐标来更新每个样本所属的簇
% 重复进行上述2.3过程

for i=1:max_iter
    fprintf('K-Means iteration %d/%d...\n', i, max_iter);
    % 把每个样本分到所属的簇
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
    % 用每个簇中所有样本的均值来更新簇的坐标
    centroids = computeCentroids(X, idx, K);
end

if isplot
    hold off;
end

end
