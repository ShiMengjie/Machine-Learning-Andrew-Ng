function [idx] = findClosestCentroids(X,init_centroids)
%% 函数功能：找到X每个样本所属的簇
% init_cetroids 的列数和X的列数相等
[m,~] = size(X);
K = size(init_centroids,1);
% 用来保存每个样本距离每个簇中心的距离
% distance(m,k)：第m个样本距离第k个簇中心的距离
distance = zeros(m,K);

for i =1:K
    distance(:,i) = sum(bsxfun(@minus,X,init_centroids(i,:)) .^ 2,2) ;
end
% 找到每一列中的最小值，并返回行数
[~,I] = min(distance.');
idx = I.';

end
